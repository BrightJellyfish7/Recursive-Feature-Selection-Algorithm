"""
Benchmark: which methods recover staged genes?

build_adata stages a block of genes (n_staged) into group_2. Two designs:

  stage_mode="bimodal"  (default) — each staged gene is OFF in half of group_2's
    cells and ON in the other half, while group_1 sits steadily in the middle.
    Both groups end up at the same expression level, so the difference is in the
    SHAPE of the distribution, not its location. Wilcoxon tests stochastic
    dominance and the t-test tests means, so both are blind; a tree can split on
    a threshold, so RF-based selection recovers them.

  stage_mode="coexpression" — matched marginals, correlated only within group_2.
    Kept for comparison, but no method here recovers it: the signal lives between
    genes, while every method scores each gene against the group label.

Methods compared:
  1. Wilcoxon DE (scanpy) — univariate non-parametric baseline
  2. t-test DE (scanpy) — univariate parametric baseline
  3. Permutation Wilcoxon — empirical null (defined, commented out in main: slow)
  4. scVI + DE on latent — neural latent space + DE (commented out in main)
  5. Your algo — recursive RF + GMM relabeling

Outputs: benchmark_results.csv, benchmark_top_genes.csv

Required installs:
  pip install scvi-tools tqdm

"""
import warnings
warnings.filterwarnings("ignore")
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import hypergeom, mannwhitneyu, poisson
from scsim import scsim
import recursieve


# =========================================================================
# 1. Build staged dataset
# =========================================================================
def _match_high_mode(level, low, scale, max_k=400):
	"""Find the ON level whose OFF/ON mixture matches `level` after normalization.

	The tests operate on log1p(scale * counts), not on raw counts, and log1p is
	not scale-invariant — so simply matching geometric means of the raw modes
	leaves a location difference once per-cell normalization rescales them, and
	the t-test then picks the genes up for the wrong reason. Here we solve
	numerically for `high` such that

	    0.5 * E[log1p(s*Pois(low))] + 0.5 * E[log1p(s*Pois(high))]
	        == E[log1p(s*Pois(level))]

	using exact Poisson expectations, so the staged genes differ from group_1 in
	SHAPE only.
	"""
	k = np.arange(max_k)

	def elog(mu):
		w = poisson.pmf(k, mu)
		return float(np.sum(w * np.log1p(scale * k)) / w.sum())

	target = 2.0 * elog(level) - elog(low)
	lo_b, hi_b = level, max(level * 50.0, level + 10.0)
	if elog(hi_b) < target:            # cannot reach it; fall back to the bound
		return hi_b
	for _ in range(80):                # bisection: elog is increasing in mu
		mid = 0.5 * (lo_b + hi_b)
		if elog(mid) < target:
			lo_b = mid
		else:
			hi_b = mid
	return 0.5 * (lo_b + hi_b)


def _stage_bimodal(rng, n1, n2, n_staged, level=8.0, low=2.0, scale=1.0):
	"""On/off staging — the design DE is blind to.

	Each staged gene is OFF in half of group_2's cells and ON in the other half,
	while group_1 sits steadily in the middle. After normalization both groups
	sit at the same expression level, so the difference is in the SHAPE of the
	distribution, not its location: Wilcoxon tests stochastic dominance and the
	t-test tests means, so both are blind, while a tree can still split on a
	threshold.

	`scale` is the per-cell normalization factor the pipeline will apply
	(target_sum / median library size); it is what makes the location matching
	hold on the data the tests actually see.
	"""
	high = _match_high_mode(level, low, scale)
	g1_block = rng.poisson(level, size=(n1, n_staged))
	on = rng.random((n2, n_staged)) < 0.5
	g2_block = rng.poisson(np.where(on, high, low))
	return g1_block, g2_block


def _stage_coexpressed(rng, n1, n2, n_staged):
	"""Co-expression staging — matched marginals, correlated only in group_2.

	Kept for comparison. Note that no method in this benchmark recovers it: the
	signal lives between genes, while every method here scores each gene against
	the group label.
	"""
	g1_block = rng.negative_binomial(5, 0.5, size=(n1, n_staged))
	g2_block = rng.negative_binomial(5, 0.5, size=(n2, n_staged))
	shared_order = np.argsort(rng.standard_normal(n2))
	for j in range(n_staged):
		g2_block[:, j] = np.sort(g2_block[:, j])[shared_order]
	return g1_block, g2_block


def build_adata(sim_seed=32, stage_seed=0, n_staged=10, stage_mode="bimodal",
				stage_level=8.0, stage_low=2.0):
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

	# Overwrite the first n_staged genes with the staged signal.
	rng = np.random.default_rng(stage_seed)
	m1 = (adata.obs["group"] == "group_1").values
	n1, n2 = int(m1.sum()), int((~m1).sum())

	if stage_mode == "bimodal":
		# The scale the DE runners will see: normalize_total(target_sum=1e4).
		lib = np.asarray(adata.X).sum(axis=1)
		scale = 1e4 / float(np.median(lib))
		g1_block, g2_block = _stage_bimodal(rng, n1, n2, n_staged,
											level=stage_level, low=stage_low,
											scale=scale)
	elif stage_mode == "coexpression":
		g1_block, g2_block = _stage_coexpressed(rng, n1, n2, n_staged)
	else:
		raise ValueError(f"stage_mode must be 'bimodal' or 'coexpression', "
						 f"got {stage_mode!r}")

	X = np.asarray(adata.X).copy().astype(np.int64)
	X[m1, :n_staged] = g1_block
	X[~m1, :n_staged] = g2_block
	adata.X = X
	adata.obs["group"] = adata.obs["group"].astype("category")

	staged_genes = list(adata.var_names[:n_staged])
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


def run_ttest(adata, n_top=50):
	a = adata.copy()
	sc.pp.normalize_total(a, target_sum=1e4)
	sc.pp.log1p(a)
	sc.tl.rank_genes_groups(
		a, groupby="group", groups=["group_2"], reference="group_1",
		method="t-test", use_raw=False,
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

def run_recursieve(adata, n_top=None, seed=42, max_iterations=25):
	a = adata.copy()
	model = recursieve.recursieve(
		a, group1="group_1", group2="group_2",
		field_name="group", max_iterations=max_iterations,
		flip_rate_percentage=0.01, seed=seed,
		print_to_console=False, plots=False,
	)
	# Use full gene list, ordered by RF rank
	genes = list(model.genes)
	if n_top is not None:
		genes = genes[:n_top]
	# unique_gene_panel is algo-defined uniqueness: not DE-significant by Scanpy DE.
	return genes, list(model.unique_gene_panel), model

# =========================================================================
# 3. Evaluation
# =========================================================================

def evaluate(method_name, hits_list, staged, total_genes, panel_size):
	hits_in_panel = [g for g in hits_list[:panel_size] if g in staged]
	n_hits = len(hits_in_panel)
	# Hypergeometric: P(>= n_hits) when drawing panel_size from total_genes,
	# len(staged) of which are staged
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

	# How many top genes each method returns. For recursieve algo, use the actual
	# number it picks; for others, match it (or use 50 as default).
	N_TOP = 50
	LOGFC_CUTOFF = 0.25

	print("\n--- Wilcoxon DE ---")
	try:
		g = run_wilcoxon(adata, n_top=N_TOP)
		all_top_genes["Wilcoxon"] = g
		results.append(evaluate("Wilcoxon", g, staged_genes, adata.n_vars, N_TOP))
	except Exception as e:
		print(f"  Failed: {e}")

	print("\n--- t-test DE ---")
	try:
		g = run_ttest(adata, n_top=N_TOP)
		all_top_genes["TTest"] = g
		results.append(evaluate("TTest", g, staged_genes, adata.n_vars, N_TOP))
	except Exception as e:
		print(f"  Failed: {e}")

	# print("\n--- Permutation Wilcoxon (slow, n_perm=50) ---")
	# try:
	# 	g = run_permutation_wilcoxon(adata, n_perm=50, n_top=N_TOP, seed=0)
	# 	all_top_genes["PermWilcoxon"] = g
	# 	results.append(evaluate("PermWilcoxon", g, staged_genes, adata.n_vars, N_TOP))
	# except Exception as e:
	# 	print(f"  Failed: {e}")


	print("\n--- recursieve algo ---")
	try:
		full, unique_de, model = run_recursieve(adata, seed=42)

		# Baseline-defined uniqueness: in recursieve_full but absent from every
		# non-recursive method list currently present in all_top_genes.
		baseline_keys = [k for k in all_top_genes.keys() if not k.startswith("recursieve")]
		baseline_gene_set = set()
		for k in baseline_keys:
			baseline_gene_set.update(g for g in all_top_genes[k] if pd.notna(g))
		unique_vs_baselines = [g for g in full if g not in baseline_gene_set]

		# DE uniqueness with an extra effect-size filter.
		de_gene_set_logfc = set()
		for g, stats in getattr(model, "de_dict", {}).items():
			logfc = stats.get("logfc", np.nan)
			if np.isfinite(logfc) and abs(logfc) >= LOGFC_CUTOFF:
				de_gene_set_logfc.add(g)
		unique_de_logfc = [g for g in full if g not in de_gene_set_logfc]

		all_top_genes["recursieve_full"] = full
		# Keep recursieve_unique aligned with benchmark column semantics:
		# genes unique to recursieve vs other benchmark columns.
		all_top_genes["recursieve_unique"] = unique_vs_baselines
		# Also report the original algo-internal uniqueness definition.
		all_top_genes["recursieve_unique_de"] = unique_de
		all_top_genes[f"recursieve_unique_de_logfc{LOGFC_CUTOFF:g}"] = unique_de_logfc
		# Evaluate at the algo's natural panel size
		results.append(evaluate("recursieve_full", full, staged_genes,
								adata.n_vars, len(full)))
		results.append(evaluate("recursieve_unique", unique_vs_baselines,
								staged_genes, adata.n_vars, len(unique_vs_baselines)))
		results.append(evaluate("recursieve_unique_de", unique_de,
								staged_genes, adata.n_vars, len(unique_de)))
		results.append(evaluate(f"recursieve_unique_de_logfc{LOGFC_CUTOFF:g}",
								unique_de_logfc, staged_genes,
								adata.n_vars, len(unique_de_logfc)))
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
	df_out = df[["method", "panel_size", "hits", "hit_genes"]].copy()

	print("\nStaged genes found by each method:")
	for r in results:
		print(f"  {r['method']:<20s} ({r['hits']}/{len(staged_genes)}): {r['hit_genes']}")

	print("\nTop 25 genes per method:")
	for name, genes in all_top_genes.items():
		print(f"\n{name}:")
		print("  " + ", ".join(map(str, genes[:25])))

	# Save to CSV for later in the directory the script was launched from.
	output_dir = Path.cwd()
	df_out.to_csv(output_dir / "benchmark_results.csv", index=False)
	pd.DataFrame({k: pd.Series(v) for k, v in all_top_genes.items()}).to_csv(
		output_dir / "benchmark_top_genes.csv", index=False
	)
