# %%
import numpy as np
import pandas as pd
from fast_matrix_market import mmread
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("../")

# %%
# read indices and columns
path = "data/mousebrain/cells.tsv"
df = pd.read_csv(path, sep="\t", header=None, names=["cells"], dtype="str")
cells = df.values[:, 0]

path = "data/mousebrain/features.tsv"
df = pd.read_csv(path, sep="\t", header=None, names=["cells"], dtype="str")
genes = df.values[:, 0]

# locate Irf8
gene_str = "Irf8"
gene_id = np.where(genes==gene_str)[0][0]
print(gene_str + " gene_id:", gene_id, "\n")

print(len(cells), "cells:", cells[:4], "\n")
print(len(genes), "genes:", genes[:5])

# %%
# read counts
data = mmread("data/mousebrain/counts.mtx").transpose()
data = csr_matrix(data, dtype=np.float32)
data = ad.AnnData(X=data)
data.obs_names = cells
data.var_names = genes
print("Original data shape:", data.shape)


# %%
# Columns of interest
type = "cluster"
cluster_id = "ident"
cols = [type, cluster_id]

design_file_path = "data/mousebrain/annotations.csv"
df = pd.read_csv(design_file_path, sep=',', index_col="cell", dtype="str")[cols].astype("str")
df.rename(columns={type: "cell_type"}, inplace=True)
print(df["cell_type"].value_counts())

# "cluster" can be either "WT microglia", "Irf8 KO microglia", "WT CP-BAM" or "Irf8 KO CP-BAM"
df["genotype"] = df["cell_type"].apply(lambda x: "WT" if x.startswith("WT") else "KO")
print("\n", df["genotype"].value_counts(), "\n")

data = data[df.index] # index data object with the valid cells which have annotations
data.obs = df 
data.var_names_make_unique()

# %%
sc.pp.filter_cells(data, min_genes=200)
sc.pp.filter_cells(data, min_counts=500)
sc.pp.filter_cells(data, max_genes=5000) 

# Filter out genes that are not expressed in a minimum number of cells
sc.pp.filter_genes(data, min_cells=5)

# save data 
data.write("data/mousebrain/anndata.h5ad")
print("Saved data:", data.shape)

# %%
# optionally take only highly variable genes ----

# scale and log
data.layers["counts"] = data.X.copy()
sc.pp.normalize_total(data)
sc.pp.log1p(data)
sc.pp.highly_variable_genes(data, n_top_genes=3451)
print("Irf8 in HVG set:", data.var["highly_variable"].loc["Irf8"])

# filter for hvg and undo normalization
data = data[:, data.var['highly_variable']] 
data.X = data.layers["counts"]
del data.layers["counts"]

# save hvg data
data.write("data/mousebrain/anndata_hvg.h5ad")
print("Saved hvg data:", data.shape)


