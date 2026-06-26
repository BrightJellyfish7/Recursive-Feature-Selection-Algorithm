"""
Benchmark: which methods recover staged co-expressed genes?

Methods compared:
  1. Wilcoxon DE (scanpy default) — univariate baseline
  2. Logistic regression DE (scanpy) — multivariate-ish baseline
  3. Permutation Wilcoxon — empirical null
  4. Hotspot — co-expression / informative gene detector
  5. scVI + DE on latent — neural latent space + DE
  6. Your algo — recursive RF + GMM relabeling

Required installs:
  pip install hotspotsc scvi-tools tqdm

Note: scVI requires PyTorch. If you don't already have it, on CPU:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import hypergeom, mannwhitneyu
from scsim import scsim

import algo  # your algo module


# =========================================================================
# 1. Build staged dataset
# =========================================================================
def build_adata(sim_seed=32, stage_seed=0):
	simulator = scsim(
		ngenes=5000, ncells=10000, ngroups=2,
		libloc=7.64, libscale=0.78,
		mean_rate=7.68, mean_shape=0.34,
		expoutprob=0.00286, expoutloc=6.15, expoutscale=0.49,
		diffexpprob=0.025, diffexpdownprob=0., diffexploc=1.0, diffexpscale=1.0,
		bcv_dispersion=0.448, bcv_dof=22.087, ndoublets=0,
		nproggenes=400, progdownprob=0., progdeloc=1.0, progdescale=1.0,
		progcellfrac=0.35, proggoups=None,
		minprogusage=0.1, maxprogusage=0.7, seed=sim_seed,
	)
	simulator.simulate()

	adata = ad.AnnData(
		X=simulator.counts, obs=simulator.cellparams, var=simulator.geneparams
	)
	adata.var_names = adata.var_names.astype(str)
	adata.obs["group"] = adata.obs["group"].apply(
		lambda x: "group_1" if x == 1 else "group_2"
	)

	# Stage 10 co-expressed genes in group_2 (matched marginals via shared rank order)
	rng = np.random.default_rng(stage_seed)
	m1 = (adata.obs["group"] == "group_1").values
	n1, n2 = int(m1.sum()), int((~m1).sum())

	g1_block = rng.negative_binomial(5, 0.5, size=(n1, 10))
	g2_block = rng.negative_binomial(5, 0.5, size=(n2, 10))
	shared_order = np.argsort(rng.standard_normal(n2))
	for j in range(10):
		g2_block[:, j] = np.sort(g2_block[:, j])[shared_order]

	X = np.asarray(adata.X).copy().astype(np.int64)
	X[m1, :10] = g1_block
	X[~m1, :10] = g2_block
	adata.X = X
	adata.obs["group"] = adata.obs["group"].astype("category")

	staged_genes = list(adata.var_names[:10])
	return adata, staged_genes


# =========================================================================
# 2. Method runners — each returns a ranked list of genes
# =========================================================================

def run_wilcoxon(adata, n_top=50):
	a = adata.copy()
	sc.pp.normalize_total(a, target_sum=1e4)
	sc.pp.log1p(a)
	sc.tl.rank_genes_groups(
		a, groupby="group", groups=["group_2"], reference="group_1",
		method="wilcoxon", use_raw=False,
	)
	return list(a.uns["rank_genes_groups"]["names"]["group_2"][:n_top])


def run_logreg(adata, n_top=50):
	a = adata.copy()
	sc.pp.normalize_total(a, target_sum=1e4)
	sc.pp.log1p(a)
	sc.tl.rank_genes_groups(
		a, groupby="group", groups=["group_2"], reference="group_1",
		method="logreg", use_raw=False, max_iter=500,
	)
	return list(a.uns["rank_genes_groups"]["names"]["group_2"][:n_top])


def run_permutation_wilcoxon(adata, n_perm=100, n_top=50, seed=0):
	"""Empirical null Wilcoxon — same test, just with permuted labels for null."""
	a = adata.copy()
	sc.pp.normalize_total(a, target_sum=1e4)
	sc.pp.log1p(a)

	X = np.asarray(a.X)
	y = (a.obs["group"] == "group_2").values
	rng = np.random.default_rng(seed)

	# observed U statistics
	obs_stats = np.zeros(X.shape[1])
	for j in range(X.shape[1]):
		try:
			u, _ = mannwhitneyu(X[y, j], X[~y, j], alternative="two-sided")
			obs_stats[j] = u
		except ValueError:
			obs_stats[j] = np.nan

	# permutation null — count how often null exceeds observed
	exceed = np.zeros(X.shape[1])
	for _ in range(n_perm):
		y_perm = rng.permutation(y)
		for j in range(X.shape[1]):
			try:
				u, _ = mannwhitneyu(X[y_perm, j], X[~y_perm, j], alternative="two-sided")
				if u >= obs_stats[j]:
					exceed[j] += 1
			except ValueError:
				pass

	emp_p = (exceed + 1) / (n_perm + 1)
	order = np.argsort(emp_p)
	return list(a.var_names[order][:n_top])


def run_hotspot(adata, n_top=50):
	"""Hotspot — finds genes with informative local autocorrelation."""
	import hotspot

	a = adata.copy()
	# Filter zero-variance genes (Hotspot requirement)
	X_counts = np.asarray(a.X)
	gene_var = X_counts.var(axis=0)
	gene_sum = X_counts.sum(axis=0)
	keep = (gene_var > 0) & (gene_sum > 0)
	n_dropped = int((~keep).sum())
	if n_dropped > 0:
		print(f"  Dropping {n_dropped} zero-variance genes for Hotspot")
	a = a[:, keep].copy()

	# Hotspot needs raw counts in layers and a latent rep (use PCA on log-norm)
	a.layers["counts"] = a.X.copy()
	sc.pp.normalize_total(a, target_sum=1e4)
	sc.pp.log1p(a)
	sc.pp.scale(a, max_value=10)
	sc.tl.pca(a, n_comps=30)

	hs = hotspot.Hotspot(
		a, layer_key="counts", model="danb",
		latent_obsm_key="X_pca",
	)
	hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
	hs_results = hs.compute_autocorrelations()
	hs_results = hs_results.sort_values("Z", ascending=False)
	return list(hs_results.index[:n_top])


def run_scvi_de(adata, n_top=50, max_epochs=50):
	"""scVI latent + DE on group label."""
	import scvi

	a = adata.copy()
	a.layers["counts"] = a.X.copy().astype(np.int64)
	scvi.model.SCVI.setup_anndata(a, layer="counts", batch_key=None)
	model = scvi.model.SCVI(a, n_latent=10, n_layers=2)
	model.train(max_epochs=max_epochs, early_stopping=True, accelerator="cpu")

	de_df = model.differential_expression(
		groupby="group", group1="group_2", group2="group_1",
	)
	de_df = de_df.sort_values("bayes_factor", ascending=False)
	return list(de_df.index[:n_top])


def run_your_algo(adata, n_top=None, seed=42, max_iterations=25):
	a = adata.copy()
	model = algo.algo(
		a, group1="group_1", group2="group_2",
		field_name="group", max_iterations=max_iterations,
		flip_rate_percentage=0.01, seed=seed,
		print_to_console=False, plots=False,
	)
	# Use full gene list, ordered by RF rank
	genes = list(model.genes)
	if n_top is not None:
		genes = genes[:n_top]
	return genes, list(model.unique_gene_panel)


# =========================================================================
# 3. Evaluation
# =========================================================================

def evaluate(method_name, hits_list, staged, total_genes, panel_size):
	hits_in_panel = [g for g in hits_list[:panel_size] if g in staged]
	n_hits = len(hits_in_panel)
	# Hypergeometric: P(>= n_hits) when drawing panel_size from total_genes,
	# 10 are staged
	pval = hypergeom.sf(n_hits - 1, total_genes, len(staged), panel_size)
	return {
		"method": method_name,
		"panel_size": panel_size,
		"hits": n_hits,
		"hit_genes": hits_in_panel,
		"hypergeom_p": pval,
	}


# =========================================================================
# 4. Run benchmark
# =========================================================================

if __name__ == "__main__":
	print("Building staged dataset...")
	adata, staged_genes = build_adata(sim_seed=32, stage_seed=0)
	print(f"Staged genes: {staged_genes}")
	print(f"Total genes: {adata.n_vars}, cells: {adata.n_obs}")

	results = []
	all_top_genes = {}

	# How many top genes each method returns. For your algo, use the actual
	# number it picks; for others, match it (or use 50 as default).
	N_TOP = 50

	print("\n--- Wilcoxon DE ---")
	try:
		g = run_wilcoxon(adata, n_top=N_TOP)
		all_top_genes["Wilcoxon"] = g
		results.append(evaluate("Wilcoxon", g, staged_genes, adata.n_vars, N_TOP))
	except Exception as e:
		print(f"  Failed: {e}")

	print("\n--- Logistic regression DE ---")
	try:
		g = run_logreg(adata, n_top=N_TOP)
		all_top_genes["LogReg"] = g
		results.append(evaluate("LogReg", g, staged_genes, adata.n_vars, N_TOP))
	except Exception as e:
		print(f"  Failed: {e}")

	print("\n--- Permutation Wilcoxon (slow, n_perm=50) ---")
	try:
		g = run_permutation_wilcoxon(adata, n_perm=50, n_top=N_TOP, seed=0)
		all_top_genes["PermWilcoxon"] = g
		results.append(evaluate("PermWilcoxon", g, staged_genes, adata.n_vars, N_TOP))
	except Exception as e:
		print(f"  Failed: {e}")

	print("\n--- Hotspot ---")
	try:
		g = run_hotspot(adata, n_top=N_TOP)
		all_top_genes["Hotspot"] = g
		results.append(evaluate("Hotspot", g, staged_genes, adata.n_vars, N_TOP))
	except Exception as e:
		print(f"  Failed: {e}")

	print("\n--- scVI + DE ---")
	try:
		g = run_scvi_de(adata, n_top=N_TOP, max_epochs=50)
		all_top_genes["scVI"] = g
		results.append(evaluate("scVI", g, staged_genes, adata.n_vars, N_TOP))
	except Exception as e:
		print(f"  Failed: {e}")

	print("\n--- Your algo ---")
	try:
		full, unique = run_your_algo(adata, seed=42)
		all_top_genes["YourAlgo_full"] = full
		all_top_genes["YourAlgo_unique"] = unique
		# Evaluate at the algo's natural panel size
		results.append(evaluate("YourAlgo_full", full, staged_genes,
								adata.n_vars, len(full)))
		results.append(evaluate("YourAlgo_unique", unique, staged_genes,
								adata.n_vars, len(unique)))
	except Exception as e:
		print(f"  Failed: {e}")

	# =========================================================================
	# 5. Report
	# =========================================================================
	print("\n" + "=" * 70)
	print("RESULTS")
	print("=" * 70)
	df = pd.DataFrame(results)
	print(df[["method", "panel_size", "hits", "hypergeom_p"]].to_string(index=False))

	print("\nStaged genes found by each method:")
	for r in results:
		print(f"  {r['method']:<20s} ({r['hits']}/10): {r['hit_genes']}")

	print("\nTop 20 genes per method:")
	for name, genes in all_top_genes.items():
		print(f"\n{name}:")
		print("  " + ", ".join(map(str, genes[:20])))

	# Save to CSV for later
	df.to_csv("benchmark_results.csv", index=False)
	pd.DataFrame({k: pd.Series(v) for k, v in all_top_genes.items()}).to_csv(
		"benchmark_top_genes.csv", index=False
	)
	print("\nSaved: benchmark_results.csv, benchmark_top_genes.csv")