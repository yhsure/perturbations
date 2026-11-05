"""Small helpers for the 2024-02-12 mouse CELLxGENE Census scVI model.

The decoder returns normalised expression (px_scale), with the original batch
held fixed for each cell. No model is trained. Only load a trusted checkpoint:
the published scVI file contains Python objects, not just tensor weights.
"""

import h5py
import numpy as np
import pandas as pd
import torch
from scvi.nn import DecoderSCVI


def load_census(atlas, checkpoint, n_cells=10000, seed=42, device="cpu"):
    """Load one decoder and sample primary islet cells with normal/T2D labels.

    Returns ``decoder, cells, genes``. Cell rows are in source-atlas order;
    genes has ``gene_id`` and ``gene_name`` columns in decoder-output order.
    Batch category names, rather than atlas category codes, are matched to the
    saved model. Cells with missing or unrecognised batch names are excluded.
    """
    if not isinstance(n_cells, (int, np.integer)) or n_cells < 1:
        raise ValueError("n_cells must be a positive integer")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    registry = state["attr_dict"]["registry_"]
    model_batches = np.asarray(registry["field_registries"]["batch"]
                               ["state_registry"]["categorical_mapping"]).astype(str)
    with h5py.File(atlas) as f:
        ids = f["var"]["feature_id"][:].astype(str)
        np.testing.assert_array_equal(ids, np.asarray(state["var_names"]).astype(str))
        genes = pd.DataFrame({"gene_id": ids, "gene_name": f["var"]["feature_name"][:].astype(str)})

        def categorical(key):
            q = f["obs"][key]
            return q["categories"][:].astype(str), q["codes"][:]

        tissue, tc = categorical("tissue")
        disease, dc = categorical("disease")
        cell_type, cc = categorical("cell_type")
        batch_names, bc = categorical("batch")
        lookup = {name: i for i, name in enumerate(model_batches)}
        mapping = np.array([lookup.get(name, -1) for name in batch_names])
        mapped = np.full(len(bc), -1, dtype=np.int64)
        valid = (bc >= 0) & (bc < len(mapping))
        mapped[valid] = mapping[bc[valid]]  # Never interpret missing code -1 as a batch.
        eligible = f["obs"]["is_primary_data"][:].astype(bool) & (mapped >= 0)
        eligible &= np.isin(tc, np.flatnonzero(tissue == "islet of Langerhans"))
        eligible &= np.isin(dc, np.flatnonzero(np.isin(
            disease, ["normal", "type 2 diabetes mellitus"])))
        pool = np.flatnonzero(eligible)
        if not len(pool):
            raise ValueError("No eligible cells; check the atlas and model release")
        rows = np.sort(np.random.RandomState(seed).choice(pool, min(n_cells, len(pool)), replace=False))
        labels = np.full(len(rows), "unannotated", dtype=object)
        known = (cc[rows] >= 0) & (cc[rows] < len(cell_type))
        labels[known] = cell_type[cc[rows][known]]
        cells = dict(latents=f["obsm"]["_scvi_latent_qzm"][rows].astype(np.float32),
                     batches=mapped[rows], disease=disease[dc[rows]], cell_type=labels.astype(str),
                     sample_ids=f["obs"]["_index"][rows].astype(str), source_row_indices=rows)
        np.testing.assert_array_equal(model_batches[cells["batches"]], batch_names[bc[rows]])
    if cells["latents"].shape[1] != 50 or len(genes) != 8000:
        raise ValueError("Expected the mouse model's 50 latent dimensions and 8,000 genes")
    if not np.isfinite(cells["latents"]).all():
        raise ValueError("Non-finite saved latent coordinates")
    decoder = DecoderSCVI(50, 8000, n_cat_list=[len(model_batches)], n_layers=1,
                         n_hidden=128, use_batch_norm=True, use_layer_norm=False)
    decoder.load_state_dict({k[len("decoder."):]: v for k, v in state["model_state_dict"].items()
                             if k.startswith("decoder.")}, strict=True)
    decoder.to(device).eval().requires_grad_(False)
    return decoder, cells, genes


def _inputs(decoder, latents, batches, batch_size):
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    z, b = np.asarray(latents), np.asarray(batches)
    if z.ndim != 2 or z.shape[1] != 50 or not len(z) or b.shape != (len(z),):
        raise ValueError("Expected nonempty (cells, 50) latents and one batch index per cell")
    if not np.isfinite(z).all() or not np.issubdtype(b.dtype, np.integer) or (b < 0).any():
        raise ValueError("Latents must be finite and batch indices must be nonnegative integers")
    if decoder.training:
        raise ValueError("Use decoder.eval(): batch normalisation must be fixed")
    parameter = next(decoder.parameters())
    return (torch.as_tensor(z, dtype=parameter.dtype, device=parameter.device),
            torch.as_tensor(b, dtype=torch.long, device=parameter.device)[:, None])


def _expression(decoder, z, batches):
    # Normalised expression is independent of the library-size argument.
    return decoder("gene", z, z.new_zeros((len(z), 1)), batches)[0]


def gene_gradient(decoder, latents, batches, gene_index, batch_size=128):
    """Return decoded expression and its exact latent gradient for one gene."""
    if not isinstance(gene_index, (int, np.integer)) or not 0 <= gene_index < 8000:
        raise ValueError("gene_index must index one of the 8,000 decoder outputs")
    z, b = _inputs(decoder, latents, batches, batch_size)
    values, gradients = [], []
    for start in range(0, len(z), batch_size):
        zi = z[start:start + batch_size].detach().requires_grad_(True)
        value = _expression(decoder, zi, b[start:start + batch_size])[:, gene_index]
        gradient = torch.autograd.grad(value.sum(), zi)[0]
        values.append(value.detach().cpu().numpy())
        gradients.append(gradient.detach().cpu().numpy())
    values, gradients = np.concatenate(values), np.concatenate(gradients)
    if not np.isfinite(values).all() or not np.isfinite(gradients).all():
        raise ValueError("Non-finite decoded expression or gradients")
    return values, gradients


def mean_alignment(decoder, latents, batches, direction, batch_size=16):
    """Mean cosine between each gene's gradient and a supplied latent direction.

    Returns 8,000 scores in decoder-output order. Forward-mode differentiation
    over every latent coordinate gives exact gradient norms, without a random
    norm estimator or a stored full Jacobian. Zero gradients contribute zero.
    For normal-to-T2D scores, supply normal cells and the mean T2D-minus-normal
    displacement; reverse both the cells and direction for T2D-to-normal scores.
    This exact calculation differs from the original approximate norm estimator.
    """
    z, b = _inputs(decoder, latents, batches, batch_size)
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (50,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("direction must be a finite, nonzero 50-dimensional vector")
    unit = torch.as_tensor(direction / np.linalg.norm(direction), dtype=z.dtype, device=z.device)
    total = torch.zeros(8000, dtype=torch.float64, device=z.device)
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            zi, bi = z[start:start + batch_size], b[start:start + batch_size]
            def decode(x):
                return _expression(decoder, x, bi)
            numerator = torch.func.jvp(decode, (zi,), (unit.expand_as(zi),))[1]
            squared_norm = torch.zeros_like(numerator)
            for coordinate in range(z.shape[1]):
                basis = torch.zeros_like(zi)
                basis[:, coordinate] = 1
                derivative = torch.func.jvp(decode, (zi,), (basis,))[1]
                squared_norm += derivative.square()
            norm = squared_norm.sqrt()
            cosine = numerator / norm.clamp_min(torch.finfo(norm.dtype).tiny)
            total += cosine.clamp(-1, 1).sum(0, dtype=torch.float64)
    result = (total / len(z)).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Non-finite gene-alignment scores")
    return result
