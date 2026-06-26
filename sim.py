import algo_og
import argparse
import os, sys
from scsim import scsim
import numpy as np
import time
import pandas as pd
import anndata as ad
import scanpy as sc


#params for the simulation
simulator = scsim(ngenes=5000, ncells=10000, ngroups=2, libloc=7.64, libscale=0.78,
				mean_rate=7.68, mean_shape=0.34, expoutprob=0.00286,
				expoutloc=6.15, expoutscale=0.49,
				diffexpprob=0.025, diffexpdownprob=0., diffexploc=1, diffexpscale=1,
				bcv_dispersion=0.448, bcv_dof=22.087, ndoublets=0,
				nproggenes=400, progdownprob=0., progdeloc=1,
				progdescale=1, progcellfrac=0.35, proggoups=None,
				minprogusage=.1, maxprogusage=.7, seed=32)

#run the simulation
simulator.simulate()

#create anndata object
adata = ad.AnnData(X=simulator.counts, obs=simulator.cellparams, var=simulator.geneparams)

#rename groups
adata.obs["group"] = adata.obs["group"].apply(lambda x: "group_1" if x == 1 else "group_2")

#Stage 10 genes with matched marginals but different co-expression structure ----
rng = np.random.default_rng(0)
m1 = (adata.obs["group"] == "group_1").values
n1 = int(m1.sum())
n2 = int((~m1).sum())

#Group 1: 10 independent NB genes
g1_block = rng.negative_binomial(5, 0.5, size=(n1, 10))

#Group 2: same marginal NB, but tied together by a shared rank ordering
g2_block = rng.negative_binomial(5, 0.5, size=(n2, 10))
shared_order = np.argsort(rng.standard_normal(n2))
for j in range(10):
    g2_block[:, j] = np.sort(g2_block[:, j])[shared_order]

#Write back to the full matrix (avoids view-assignment bug)
X = np.asarray(adata.X).copy().astype(np.int64)
X[m1, :10] = g1_block
X[~m1, :10] = g2_block
adata.X = X

#sanity checks — marginals should match, correlations should differ
print("Group 1 means:", X[m1, :10].mean(axis=0).round(2))
print("Group 2 means:", X[~m1, :10].mean(axis=0).round(2))
print("Group 1 vars: ", X[m1, :10].var(axis=0).round(2))
print("Group 2 vars: ", X[~m1, :10].var(axis=0).round(2))
print("G1 mean off-diag corr:",
      np.corrcoef(g1_block.T)[np.triu_indices(10, 1)].mean().round(3))
print("G2 mean off-diag corr:",
      np.corrcoef(g2_block.T)[np.triu_indices(10, 1)].mean().round(3))

#preprocessing
adata.obs["group"] = adata.obs["group"].astype("category")
adata.layers["raw"] = adata.X.copy()

sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

#DE (should NOT pick up genes 0-9)
sc.tl.rank_genes_groups(
    adata,
    groupby="group",
    groups=["group_2"],
    reference="group_1",
    method="wilcoxon",
    use_raw=False,
)

de_genes = adata.uns["rank_genes_groups"]["names"]["group_2"]
de_pvals = adata.uns["rank_genes_groups"]["pvals_adj"]["group_2"]
print("Top 25 DE genes:")
print(de_genes[:50])

#Where do the staged genes (0-9) rank in DE results?
staged = [str(i) for i in range(10)]
ranks = [list(de_genes).index(g) if g in de_genes else -1 for g in staged]
print("DE ranks of staged genes 0-9:", ranks)


#run the algo
model = algo_og.algo(adata, group1="group_1", group2="group_2",
                  field_name="group",
                  print_to_console=True,
                  max_iterations=50,
                  flip_rate_percentage=0.01,
				  seed=42)

print(model.genes)
print(model.unique_gene_panel)
