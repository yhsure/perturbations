"""Display latent-space gradients on an existing embedding.

This adapts the neighbour-based RNA-velocity projection of La Manno et al.
(2018, doi:10.1038/s41586-018-0414-6) and Bergen et al.
(2020, doi:10.1038/s41587-020-0591-3). It projects decoder directions;
it does not estimate RNA velocity or differentiate the embedding itself.
"""

import numpy as np
from scipy.spatial import cKDTree


def project_gradients(latents, gradients, embedding, n_neighbors=30, sigma=1.0):
    """Return one projected direction per cell, in the supplied row order.

    ``latents`` and ``gradients`` have shape (cells, latent dimensions).
    ``embedding`` has shape (cells, embedding dimensions), usually two.
    Use the same cells in the same order in all three arrays.

    Among up to ``n_neighbors`` nearest non-self cells in latent space,
    softmax(cosine / sigma**2) weights favour displacements aligned with
    the gradient. We subtract uniform weights, then combine unit directions
    to those neighbours in the embedding. Coincident latent positions are
    skipped; zero gradients or neighbourhoods without a direction give zero.
    These are visual directions, not calibrated displacements or speeds.
    """
    z, g, xy = (np.asarray(a, dtype=np.float64)
                for a in (latents, gradients, embedding))
    if any(a.ndim != 2 for a in (z, g, xy)):
        raise ValueError("All inputs must be two-dimensional arrays")
    if g.shape != z.shape or len(xy) != len(z):
        raise ValueError("Latents, gradients and embedding must have matching cells")
    if not z.shape[1] or not xy.shape[1] or len(z) < 2:
        raise ValueError("At least two cells and one coordinate per space are required")
    if not all(np.isfinite(a).all() for a in (z, g, xy)):
        raise ValueError("All input coordinates and gradients must be finite")
    if (isinstance(n_neighbors, (bool, np.bool_))
            or not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1):
        raise ValueError("n_neighbors must be a positive integer")
    sigma = float(sigma)
    sigma_squared = sigma * sigma
    if sigma <= 0 or not np.isfinite(sigma_squared) or sigma_squared == 0:
        raise ValueError("sigma must be positive with a finite, nonzero square")

    k = min(n_neighbors, len(z) - 1)
    neighbours = cKDTree(z).query(z, k=k + 1)[1]
    projected = np.zeros_like(xy)
    for i, candidates in enumerate(neighbours):
        gradient_norm = np.linalg.norm(g[i])
        if gradient_norm == 0:
            continue
        # Explicit self-removal also works when duplicate points tie for first.
        indices = candidates[candidates != i][:k]
        dz = z[indices] - z[i]
        lengths = np.linalg.norm(dz, axis=1)
        keep = lengths > 0
        indices, dz, lengths = indices[keep], dz[keep], lengths[keep]
        if len(indices) == 0:
            continue
        cosine = (dz / lengths[:, None]) @ (g[i] / gradient_norm)
        # Subtract before scaling to avoid positive exponential overflow.
        with np.errstate(over="ignore"):
            weights = np.exp((cosine - cosine.max()) / sigma_squared)
        weights /= weights.sum()
        weights -= 1.0 / len(indices)
        delta = xy[indices] - xy[i]
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        unit = np.divide(delta, distance, out=np.zeros_like(delta), where=distance > 0)
        projected[i] = weights @ unit
    return projected
