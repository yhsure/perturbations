![Perturbations logo](data/perturbs_small.png)
# Explainable perturbations for models in single-cell sequencing

### Abstract
Generative models learn underlying dynamics of data and can do more than just reconstruction. By leveraging output gradients with respect to the latent dimensions, we explore a simple approach to infer arbitrary perturbation effects on interpretive flow maps within high-dimensional biological datasets. By applying this method to several cases in single-cell RNA-sequencing, we demonstrate its use in inferring effects from knockdown, overexpression, toxin response and embryonic development. This approach can further add global structure to dimensionality reductions which only preserve local patterns. Needing only a decoder, our method simplifies analyses, is adaptable with already trained models, and offers clearer insights into cellular dynamics without complex setups. In turn, this gives a more straightforward interpretation of results, making it easier to discern underlying biological pathways with easily understandable visual representations.

### Running the project
To reproduce the results, follow these steps.
1. Download the following datasets:
- C. elegans embryogenesis [1], [GEO: GSE126954][celegans]. Files to be placed in `data/celegans/` with: cell annotations as `cells.csv`, transcript names as `features.csv` and counts as `counts.mtx`.
- Irf8-cKO mouse brains [2], [GEO: GSE128855][irf8] ("Full aggregate"). Files to be placed in `data/mousebrain/` with: cell barcodes as `cells.tsv`, transcript names as `features.tsv`, counts as `counts.mtx` and metadata as `annotations.csv`.
- Cardiotoxin-induced injury in mice [3], [ArrayExpress: E-MTAB-9715][ctx]. Files to be placed in `data/cardiotoxin/` with: cell barcodes as `cells.mtx_cols`, transcript names as `features.mtx_rows` , counts as `counts.mtx` and design file as `annotations.tsv`.

2. For each dataset, run the relevant preprocessing script in the `preprocessing` folder. 

3. Finally, the `.ipynb` notebooks can be run to generate the results. 

**Note**: This readme will be extended upon approval.

### Bibliography
1. Packer, Jonathan S., et al. "A lineage-resolved molecular atlas of C. elegans embryogenesis at single-cell resolution." Science 365.6459 (2019): eaax1971.
2. Van Hove, Hannah, et al. "A single-cell atlas of mouse brain macrophages reveals unique transcriptional identities shaped by ontogeny and tissue environment." Nature neuroscience 22.6 (2019): 1021-1035.
3. Takada, Naoki, et al. "Galectin-3 promotes the adipogenic differentiation of PDGFRα+ cells and ectopic fat formation in regenerating muscle." Development 149.3 (2022): dev199443.

[celegans]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954
[irf8]: https://www.brainimmuneatlas.org/download.php
[ctx]: https://www.ebi.ac.uk/gxa/sc/experiments/E-MTAB-9715/downloads
