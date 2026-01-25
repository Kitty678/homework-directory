from __future__ import annotations
import anndata as ad
import scanpy as sc
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime


sc.settings.verbosity = 3 
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.settings.figdir = 'figures'  


data_dir = 'GSE193337_RAW'
samples = ['GSM5793824', 'GSM5793825', 'GSM5793826', 'GSM5793827', 
           'GSM5793828', 'GSM5793829', 'GSM5793831', 'GSM5793832']
adatas = []

for sample in samples:
    sample_dir = os.path.join(data_dir, sample)
    
    matrix_files = glob.glob(os.path.join(sample_dir, '*_matrix.mtx.gz'))
    if not matrix_files:
        print(f"Warning: No matrix file found in {sample_dir}")
        continue
    
    matrix_file = os.path.basename(matrix_files[0])
    prefix = matrix_file.replace('matrix.mtx.gz', '')
    
    try:
        adata_sample = sc.read_10x_mtx(
            sample_dir,
            var_names='gene_symbols',
            cache=True,
            prefix=prefix
        )
        
        adata_sample.obs['sample'] = sample
        
        sample_type_map = {
            'GSM5793824': 'benign',  # P1n
            'GSM5793825': 'benign',  # P2n
            'GSM5793826': 'benign',  # P3n
            'GSM5793827': 'benign',  # P4n
            'GSM5793828': 'tumor',   # P1t
            'GSM5793829': 'tumor',   # P2t
            'GSM5793831': 'tumor',   # P3t
            'GSM5793832': 'tumor',   # P4t
        }
        adata_sample.obs['type'] = sample_type_map.get(sample, 'unknown')
        
        print(f"Loaded {sample}: {adata_sample.shape[0]} cells, {adata_sample.shape[1]} genes, type={sample_type_map.get(sample)}")
        adatas.append(adata_sample)
        
    except Exception as e:
        print(f"Error loading {sample}: {e}")
        continue

if len(adatas) > 0:
    adata = ad.concat(adatas, label='batch', index_unique='-')
    
    # Dataset overview
    print("\n" + "="*60)
    print("Dataset Overview")
    print("="*60)
    print(f"Total cells: {adata.shape[0]:,}")
    print(f"Total genes: {adata.shape[1]:,}")
    print(f"\nSample distribution:")
    print(adata.obs["sample"].value_counts())
    print(f"\nType distribution:")
    print(adata.obs["type"].value_counts())
    print("="*60 + "\n")
else:
    print("Error: No data loaded!")
    adata = None

# Quality Control
# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    save='_qc_violin.png'
)
sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", save='_qc_scatter.png')
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)

#Doublet Detection
sc.external.pp.scrublet(adata, batch_key="sample")
adata.obs['predicted_doublet'] = adata.obs['predicted_doublet'].astype('category')
#Normalization
# Saving count data
adata.layers["counts"] = adata.X.copy()
# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)

#Feature Selection  
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")
sc.pl.highly_variable_genes(adata, save='_hvg.png')

#Dimensionality Reduction
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, save='_variance_ratio.png')
sc.pl.pca(
    adata,
    color=["sample", "sample", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=2,
    save='_pca.png'
)

#Nearest neighbor graph construction and visualization
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color="sample",
    size=2,
    save='_umap_sample.png'
)

#Clustering
sc.tl.leiden(adata, n_iterations=2)
sc.pl.umap(adata, color=["leiden"], save='_umap_leiden.png')

#Re-assess quality control and cell filtering

sc.pl.umap(
    adata,
    color=["leiden", "predicted_doublet", "doublet_score"],
    wspace=0.5,
    size=3,
    save='_umap_doublet.png'
)
sc.pl.umap(
    adata,
    color=["leiden", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
    wspace=0.5,
    ncols=2,
    save='_umap_qc.png'
)

#Manual cell-type annotation
for res in [0.02, 0.5, 2.0]:
    sc.tl.leiden(adata, key_added=f"leiden_res_{res:4.2f}", resolution=res)

# Print cluster numbers for easy reference
print("\n" + "="*60)
print("NUMBER OF CLUSTERS")
print("="*60)
print(f"Default leiden clustering: {len(adata.obs['leiden'].unique())} clusters")
print(f"Leiden resolution 0.02: {len(adata.obs['leiden_res_0.02'].unique())} clusters")
print(f"Leiden resolution 0.50: {len(adata.obs['leiden_res_0.50'].unique())} clusters")
print(f"Leiden resolution 2.00: {len(adata.obs['leiden_res_2.00'].unique())} clusters")
print("\nCells per cluster (default leiden):")
for cluster, count in sorted(adata.obs['leiden'].value_counts().items()):
    pct = count / len(adata) * 100
    print(f"  Cluster {cluster}: {count:,} cells ({pct:.1f}%)")
print("="*60 + "\n")

sc.pl.umap(
    adata,
    color=["leiden_res_0.02", "leiden_res_0.50", "leiden_res_2.00"],
    legend_loc="on data",
    save='_umap_leiden_res.png'
)

# Marker gene set
marker_genes = {
    
    "Prostate Epithelial / Luminal": ["KRT8", "KRT18", "NKX3-1", "KLK3", "KLK2", "TMPRSS2", "AR"],
    "Basal Epithelial": ["KRT5", "KRT14", "TP63", "KRT15"],
    "Cancer / Malignant Epithelial": ["AMACR", "ERG", "PTEN", "MYC", "AR"],  # PTEN常丢失但RNA可能仍低表达

    
    "CD4+ T": ["CD4", "IL7R", "CCR7", "TCF7"],
    "CD8+ T": ["CD8A", "CD8B", "GZMA", "GZMB", "CCL5"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4"],
    "NK": ["NKG7", "GNLY", "KLRD1", "FCGR3A"],
    "B cells": ["MS4A1", "CD19", "CD79A", "CD79B"],
    "Plasma cells": ["MZB1", "JCHAIN", "IGKC", "SDC1"],
    "Macrophage / Mono": ["CD68", "CD163", "MRC1", "CD14", "FCGR3A"],
    "cDC": ["HLA-DRA", "CD1C", "FCER1A", "CLEC10A"],
    "pDC": ["LILRA4", "IL3RA", "CLEC4C"],

    
    "Fibroblast": ["DCN", "LUM", "COL1A1", "COL1A2", "FAP", "PDGFRA"],
    "Myofibroblast / CAF": ["ACTA2", "MYH11", "RGS5", "PDGFRB"],
    "Endothelial": ["PECAM1", "CDH5", "VWF", "CLDN5", "KDR"],
    "Smooth Muscle": ["ACTA2", "MYH11", "CNN1", "TAGLN"],

    
    "Erythroid": ["HBA1", "HBB"],
    "Proliferating": ["MKI67", "TOP2A", "PCNA"],
}

sc.pl.dotplot(adata, marker_genes, groupby="leiden_res_0.02", standard_scale="var")