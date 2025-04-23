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
path = "data/cardiotoxin/cells.mtx_cols"
df = pd.read_csv(path, sep="\t", header=None, names=["cells"], dtype="str")
cells = df.values[:, 0]

path = "data/cardiotoxin/features.mtx_rows"
df = pd.read_csv(path, sep="\t", header=None, names=["genes"], dtype="str")
genes = df.values[:, 0]

print(len(cells), "cells:", cells[:4], "\n")
print(len(genes), "genes:", genes[:5])

# %%
data = mmread("data/cardiotoxin/counts.mtx").transpose()
data = csr_matrix(data, dtype=np.float32)
data = ad.AnnData(X=data)
data.obs_names = cells
data.var_names = genes
print("Original data shape:", data.shape)


# %%
# Columns of interest
sample_pool = "Sample Characteristic[individual]"
diet = "Factor Value[diet]"
coumpound = "Factor Value[compound]"
time = "Factor Value[sampling time point]"
cols = [sample_pool, diet, coumpound, time]

design_file_path = "data/cardiotoxin/annotations.tsv"
df = pd.read_csv(design_file_path, sep='\t', index_col="Assay", dtype="str")[cols].astype("str")
df.rename(columns={sample_pool:"sample_pool", diet:"diet", coumpound:"compound", time:"time"}, inplace=True)

df["compound"] = df["compound"].apply(lambda x: "none" if x=="none" else "cardiotoxin")
df["compound_int"] = df["compound"].apply(lambda x: 0 if x=="none" else 1)
print("\n", df["compound"].value_counts(1), "\n")
print("\n", df["diet"].value_counts(1), "\n")
print("\n", df["time"].value_counts(1), "\n")

data.obs = df 
data.var_names_make_unique()


# %%
sc.pp.filter_cells(data, min_genes=200)
sc.pp.filter_cells(data, min_counts=500)
sc.pp.filter_cells(data, max_genes=5000) 

# Filter out genes that are not expressed in a minimum number of cells
sc.pp.filter_genes(data, min_cells=5)

# save data 
data.write("data/cardiotoxin/anndata.h5ad")
print("Saved data:", data.shape)

# %%
# optionally take only highly variable genes

# scale and log
data.layers["counts"] = data.X.copy()
sc.pp.normalize_total(data)
sc.pp.log1p(data)
sc.pp.highly_variable_genes(data)

# filter for hvg and undo normalization
data = data[:, data.var['highly_variable']] 
data.X = data.layers["counts"]
del data.layers["counts"]

# save hvg data
data.write("data/cardiotoxin/anndata_hvg.h5ad")
print("Saved hvg data:", data.shape)
