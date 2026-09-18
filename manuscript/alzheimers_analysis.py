"""Alzheimer's disease analysis using recursieve on reduced dataset.

This script:
1. Loads the reduced Alzheimer's dataset (alz_reduced.h5ad)
2. Runs recursieve with specified parameters to identify discriminative genes
   between Disease.Group "low" and "high"
3. Saves results including:
   - Selected genes and their performance metrics
   - Unique gene panel (genes prioritized over DE results)
   - Differential expression results

Outputs are written to the current directory (expected: manuscript/):
- alzheimers_genes.csv: All selected genes with accuracy and AUC scores
- alzheimers_unique_genes.csv: Unique gene panel (non-DE genes)
- alzheimers_de_results.csv: Differential expression analysis
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

import recursieve


def main() -> None:
	"""Load Alzheimer's dataset and run recursieve analysis."""
	# Set up paths
	data_dir = Path(__file__).resolve().parent.parent / "data"
	out_dir = Path(__file__).resolve().parent
	adata_path = data_dir / "alz.h5ad"

	# Check data exists
	if not adata_path.exists():
		raise FileNotFoundError(f"Data file not found: {adata_path}")

	print(f"Loading data from: {adata_path}")
	adata_alz = ad.read_h5ad(adata_path)
	
	#subsample the data to 1000 cells using scanpy's subsample function
	adata_alz = sc.pp.subsample(adata_alz, n_obs=5000, copy=True)

	# Replace ensemble IDs with gene names and make unique
	adata_alz.var_names = adata_alz.var['feature_name']
	adata_alz.var_names_make_unique()
	
	print(f"Data shape: {adata_alz.n_obs} cells x {adata_alz.n_vars} genes")

	# Run recursieve
	print("\nRunning recursieve analysis...")
	model = recursieve.recursieve(
		adata=adata_alz,
		group1="low",
		group2="high",
		field_name="Disease.Group",
		plots=False,
		print_to_console=False,
		max_iterations=25,
	)

	print(f"Analysis complete. Selected {len(model.genes)} genes over {len(model.accuracy_scores)} iterations")

	# Save selected genes with performance metrics
	genes_df = pd.DataFrame({
		"gene": model.genes,
		"accuracy": model.accuracy_scores[:len(model.genes)],
		"auc": model.auc_scores[:len(model.genes)],
	})
	genes_path = out_dir / "alzheimers_genes.csv"
	genes_df.to_csv(genes_path, index=False)
	print(f"Saved gene list: {genes_path}")

	# Save unique gene panel
	unique_genes_df = pd.DataFrame({
		"gene": model.unique_gene_panel,
	})
	unique_path = out_dir / "alzheimers_unique_genes.csv"
	unique_genes_df.to_csv(unique_path, index=False)
	print(f"Saved unique genes: {unique_path}")

	# Get DE results
	hits, de_results = model.intersect_genes_and_de(top_n_genes=len(model.genes))
	de_path = out_dir / "alzheimers_de_results.csv"
	de_results.to_csv(de_path, index=False)
	print(f"Saved DE results: {de_path}")

	print("\nAnalysis complete.")
	print(f"Total genes selected: {len(model.genes)}")
	print(f"Unique gene panel size: {len(model.unique_gene_panel)}")
	print(f"Final accuracy: {model.accuracy_scores[-1]:.4f}")
	print(f"Final AUC: {model.auc_scores[-1]:.4f}")


if __name__ == "__main__":
	main()
