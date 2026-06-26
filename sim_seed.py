import algo
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scsim import scsim


def build_adata(sim_seed=32, stage_seed=0):
	"""Simulate scsim data + stage 10 co-expressed genes in group_2."""
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
	adata.obs["group"] = adata.obs["group"].apply(
		lambda x: "group_1" if x == 1 else "group_2"
	)

	# stage co-expression structure in genes 0-9
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
	return adata


# ----- seed variance experiment -----
N_SEEDS = 10
results = []
staged_set = set(str(i) for i in range(10))  # scsim names genes by integer

for s in range(N_SEEDS):
	print(f"\n===== Run {s} =====")
	
	# Vary algo seed only; keep data fixed so we measure algo stability.
	# To measure data+algo stability instead, change to sim_seed=s.
	adata_run = build_adata(sim_seed=32, stage_seed=0)

	#confirm staged gene names actually exist in var_names (scsim quirk)
	if s == 0:
		first_ten = list(adata_run.var_names[:10])
		print("First 10 var_names:", first_ten)
		staged_set = set(str(g) for g in first_ten)

	model = algo.algo(
		adata_run, group1="group_1", group2="group_2",
		field_name="group", max_iterations=50,
		flip_rate_percentage=0.01, seed=s,
		print_to_console=False, plots=False,
	)

	panel = set(str(g) for g in model.unique_gene_panel)
	full_panel = set(str(g) for g in model.genes)
	hits_unique = len(panel & staged_set)
	hits_full = len(full_panel & staged_set)

	results.append({
		"seed": s,
		"panel_size_unique": len(panel),
		"panel_size_full": len(full_panel),
		"hits_in_unique": hits_unique,
		"hits_in_full": hits_full,
	})
	print(f"seed={s}  unique_panel={len(panel)}  hits_unique={hits_unique}  hits_full={hits_full}")

df = pd.DataFrame(results)
print("\n===== Summary =====")
print(df.to_string(index=False))
print(f"\nUnique-panel hits: mean={df.hits_in_unique.mean():.2f}  std={df.hits_in_unique.std():.2f}")
print(f"Full-genes hits:   mean={df.hits_in_full.mean():.2f}  std={df.hits_in_full.std():.2f}")
print(f"Mean unique-panel size: {df.panel_size_unique.mean():.1f}")

# hypergeometric p-value for the average run
from scipy.stats import hypergeom
mean_panel = int(df.panel_size_unique.mean())
mean_hits = int(round(df.hits_in_unique.mean()))
total_genes = 5000  # before HVG filter; effectively the relevant population
pval = hypergeom.sf(mean_hits - 1, total_genes, 10, mean_panel)
print(f"Hypergeometric p (avg run, ≥{mean_hits} of 10 in {mean_panel} draws): {pval:.2e}")