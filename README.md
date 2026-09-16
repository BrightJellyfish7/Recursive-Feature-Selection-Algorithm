# Recursieve

Recursieve is an iterative gene selection algorithm that identifies discriminative genes between two biological groups in single-cell datasets. It uses an ensemble learning approach with a blend of random forest classifiers and gaussian mixture models to progressively select genes that best separate the groups, collapsing selected genes into a lower dimensional representational axis at each iteration.

## Installation

Install directly from the GitHub repository:

```bash
pip install git+https://github.com/BrightJellyfish7/Recursive-Feature-Selection-Algorithm.git
```

## Usage

```python
import anndata as ad
from recursieve import recursieve

adata = ad.read_h5ad("data/alz.h5ad")
model = recursieve(
    adata=adata,
    group1="control",
    group2="disease",
    field_name="condition",
    max_iterations=100,
)

selected_genes = model.unique_gene_panel
scores = model.accuracy_scores
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| adata | AnnData | required | Annotated data matrix |
| group1 | str | required | First group label |
| group2 | str | required | Second group label |
| field_name | str | "sample" | Column in obs containing group labels |
| plots | bool | False | Generate visualization plots |
| print_to_console | bool | False | Print selected genes during iteration |
| max_iterations | int | 100 | Maximum iterations to run |
| additive | bool | True | Add new genes to panel or replace |
| flip_rate_percentage | float | 0.01 | Threshold for label oscillation detection |
| pval_cutoff | float | 0.05 | P-value threshold for significance |
| seed | int | 42 | Random seed for reproducibility |
| summary_method | str | "mean" | Method to collapse genes: "mean" or "pca" |
| n_estimators | int | 300 | Number of random forest trees |
| n_jobs | int | -1 | Number of parallel jobs (-1 uses all cores) |

## Citation

RecurSieve: Ensemble Machine Learning Algorithm for Single-Cell Feature Selection Identifies Unique Gene Signatures
[bioRxiv] link coming soon