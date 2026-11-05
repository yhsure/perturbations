# What do single-cell models already know about perturbations?

![Perturbations logo](data/perturbs_small.png)

<p align="center">
  <a href="https://www.mdpi.com/2073-4425/16/12/1439" target="_blank">
    <img src="https://img.shields.io/badge/Genes-Paper-cc4778?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAMAAAAolt3jAAABCFBMVEVjRoBkR4FjR4BgQn1hRH9iRH9iRX9fQX1kR4BgQ35yV4xrT4ZfQn1zWYxpToVnS4NoTIR0W45nS4RzWY2Tf6ZqToV1W46PfKNlSIGqmrhpTYWfjbBvVImTgKd7ZJRwVYpmSoNyWIybia17Y5KGb5xhQ35jRYB2XY9tU4heQHxtUodoS4NhRH5pTYRmSYJiRYCah6x0Wo1eQXyfjrBoTYSPeqN5YJGMd6FoTINfQXyIcp5lSYJuVImllbR7Y5N2W46TfqZ9ZZWVgqhkRoBqToZgQ31xV4t5YZJsUIeNeKJrUIeOeaKSfaWXg6mXhKl3XpCSfqawor58Y5N8ZJVrUIZsUYduU4lgQn4GkBRDAAAAwUlEQVQI1y3O6VaCUAAE4IG7YaSQeDMLUyokiCzNCpcszVxyza33f5OIOmfO/Js5HwAlCf5bJVQhTGXgggNa6kA/TGeowcyjLIewcvI4r5wU5OmZyUDs4nkpV3YuLsWVqwIV79oPbkJ5S+zqXfx3X6s/NKxH9+k5YlCarXbnpfv6Fvb6noDmSuO9OvgYmhhJCnc80T+nZjhDMM8w+N5iuVyx4VdXzGIG6Nohm2002PF9QvW/VZ9yGu/+5NovPhZowA8dMRRR0T1dRAAAAABJRU5ErkJggg==&logoColor=white" alt="Paper">
  </a>
  <a href="https://yhsure.github.io/blog/perturbations" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Blogpost-6a0dad?style=for-the-badge&logo=github&logoColor=white&logoHeight=20" alt="GitHub Blog">
  </a>
</p>
<br>

> **What do single-cell models already know about perturbations?**<br>
> Andreas Bjerregaard, Iñigo Prada-Luengo, Vivek Das and Anders Krogh<br>
> *Genes 16 (12) 2025; also ICLR MLGenX 2025*<br>
> <a href="https://www.mdpi.com/2073-4425/16/12/1439" target="_blank">*https://www.mdpi.com/2073-4425/16/12/1439*</a> <br>
> 
> **Abstract:**
> **Background:** Virtual cells are embedded in widely used single-cell generative models. Nonetheless, the models' implicit knowledge of perturbations remains unclear. **Methods:** We train variational autoencoders on three gene expression datasets spanning genetic, chemical, and temporal perturbations, and infer perturbations by differentiating decoder outputs with respect to latent variables. This yields vector fields of infinitesimal change in gene expression. Furthermore, we probe a publicly released scVI decoder trained on the CELL × GENE Discover Census ($\sim\!5.7$M mouse cells) and score genes by the alignment between local gradients and an empirical healthy-to-disease axis, followed by a novel large language model-based evaluation of pathways. **Results:** Gradient flows recover known transitions in *Irf8* knockout microglia, cardiotoxin-treated muscle, and worm embryogenesis. In the pretrained Census model, gradients help identify pathways with stronger statistical support and higher type 2 diabetes relevance than an average expression baseline. **Conclusions:** Trained single-cell decoders already contain rich perturbation-relevant information that can be accessed by automatic differentiation, enabling in-silico perturbation simulations and principled ranking of genes along observed disease or treatment axes without bespoke architectures or perturbation labels. 


**Update:** Decoder gradients can now be displayed on an existing UMAP embedding using a neighbour-based projection. The [pancreatic islet notebook](pancreatic_islets.ipynb) includes examples with the pretrained scVI model.

### Running the project
To reproduce the results, follow these steps.
1. Download the following datasets:
- C. elegans embryogenesis [1], [GEO: GSE126954][celegans]. Files to be placed in `data/celegans/` with: cell annotations as `cells.csv`, transcript names as `features.csv` and counts as `counts.mtx`.
- Irf8-cKO mouse brains [2], [GEO: GSE128855][irf8] ("Full aggregate"). Files to be placed in `data/mousebrain/` with: cell barcodes as `cells.tsv`, transcript names as `features.tsv`, counts as `counts.mtx` and metadata as `annotations.csv`.
- Cardiotoxin-induced injury in mice [3], [ArrayExpress: E-MTAB-9715][ctx]. Files to be placed in `data/cardiotoxin/` with: cell barcodes as `cells.mtx_cols`, transcript names as `features.mtx_rows` , counts as `counts.mtx` and design file as `annotations.tsv`.

2. For each of these three datasets, run the relevant preprocessing script in the `preprocessing` folder.

3. Finally, the `.ipynb` notebooks can be run to generate the results.

### Pancreatic islets: pretrained scVI

The paper also studies pancreatic islet cells from the **2024-02-12 mouse CELLxGENE Census**, using its pretrained scVI model (about 5.7 million training cells, 50 latent coordinates and 8,000 genes).

Download the matching [minified atlas (3.0 GB)][census-atlas] to `data/census/adata_minified.h5ad` and [checkpoint (59 MB)][census-model] to `checkpoints/census/model.pt`. The atlas contains the saved latent representations and cell metadata. No training or preprocessing is needed for this example.

Run [pancreatic_islets.ipynb](pancreatic_islets.ipynb) from the repository root. It shows the islet representation, Ins1 and Gcg gradients in PCA, gene rankings along the normal–diabetic difference, and Ins1 gradients on UMAP. The supporting functions in `_scvi.py` check gene order and match batch names to the checkpoint before differentiating normalized decoder expression.

The notebook was tested with Python 3.11 and the dependencies below (the checkpoint itself was saved with scVI 1.0.4):

```sh
python -m pip install 'numpy<2' 'scvi-tools==1.2.2.post2' 'torch==2.5.0' 'scikit-learn==1.3.2' 'umap-learn==0.5.5' h5py matplotlib scipy ipykernel
```

The notebook uses a fresh sample of 10,000 primary islet cells (seed 42). Gene rankings use exact gradient norms on up to 256 cells per group; the original analysis estimated these norms from random directions. This illustrates the method rather than regenerating the paper's saved rankings or pathway results. Only load the linked checkpoint or another trusted file: the original model format requires PyTorch's pickle loader.

Streamline plots pair projected decoder directions with a softly coloured background of decoded expression. `_streamlines.py` interpolates both from nearby cells and leaves unsupported gaps empty. The beta-cell UMAP uses 80 neighbours, `min_dist=0.65` and a fixed random seed (`random_state=0`).

### Displaying gradients on an existing UMAP

`gradient_projection.py` implements a neighbour-based display adapted from [La Manno et al. (2018)][rna-velocity] and [Bergen et al. (2020)][scvelo]. The notebook's UMAP section explains the method. The helper also works with other embeddings:

```python
from gradient_projection import project_gradients
arrows = project_gradients(latents, gradients, umap_coordinates)
```

All three arrays must contain the same cells in the same order. The arrows show directions of increasing decoder output, not RNA velocities or derivatives of UMAP coordinates. The projection keeps the supplied embedding fixed.

### Citation
To cite the work or codebase: 

```bibtex
@article{bjerregaard2025single,
  title={What do single-cell models already know about perturbations?},
  author={Bjerregaard, Andreas and Prada-Luengo, I{\~n}igo and Das, Vivek and Krogh, Anders},
  journal={Genes},
  volume={16},
  number={12},
  pages={1439},
  year={2025},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

### References
1. Packer, Jonathan S., et al. "A lineage-resolved molecular atlas of C. elegans embryogenesis at single-cell resolution." *Science* 365.6459 (2019): eaax1971.
2. Van Hove, Hannah, et al. "A single-cell atlas of mouse brain macrophages reveals unique transcriptional identities shaped by ontogeny and tissue environment." *Nature neuroscience* 22.6 (2019): 1021-1035.
3. Takada, Naoki, et al. "Galectin-3 promotes the adipogenic differentiation of PDGFRα+ cells and ectopic fat formation in regenerating muscle." *Development* 149.3 (2022): dev199443.

[celegans]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954
[irf8]: https://www.brainimmuneatlas.org/download.php
[ctx]: https://www.ebi.ac.uk/gxa/sc/experiments/E-MTAB-9715/downloads
[census-atlas]: https://cellxgene-contrib-public.s3.amazonaws.com/models/scvi/2024-02-12/mus_musculus/adata_minified.h5ad
[census-model]: https://cellxgene-contrib-public.s3.amazonaws.com/models/scvi/2024-02-12/mus_musculus/model.pt
[rna-velocity]: https://doi.org/10.1038/s41586-018-0414-6
[scvelo]: https://doi.org/10.1038/s41587-020-0591-3
