import gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, HDBSCAN
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

pd.set_option('display.max_columns', None)  # show all columns


class algo:
	def __init__(self, adata, group1, group2, field_name="sample", plots=False, max_iterations=100, additive=True, flip_rate_percentage=0.01):
		self.adata = adata
		self.group1 = group1
		self.group2 = group2
		self.field_name = field_name
		self.plots = plots
		self.max_iterations = max_iterations
		self.additive = additive
		self.flip_rate_percentage = flip_rate_percentage
		self.gene_expression_log = {}
		self.genes = []
		self.accuracy_scores = []
		self.auc_scores = []
		self.rows = []
		self.replace_var_names()
		self.preprocessing()
		self.filter_groups()
		self.scanpy_de_original_groups()
		self.ensemble_learner()
		self.ensemble_recursion()
		self.unique_gene_panel = self.unique_gene_panel()

	def replace_var_names(self):
		if "feature_name" in self.adata.var.columns:
			self.adata.var["ensembl_id"] = self.adata.var_names.astype(str)
			self.adata.var_names = self.adata.var["feature_name"].astype(str)
			self.adata.var_names_make_unique()
		else:
			raise KeyError("feature_name not found in adata.var. Cannot replace var_names.")

	def preprocessing(self):
		self.adata.layers["counts"] = self.adata.X.copy()
		self.adata.var["mt"] = self.adata.var_names.str.upper().str.startswith("MT-")
		self.adata.var["ribo"] = self.adata.var_names.str.upper().str.match(r"^RPS|^RPL")
		self.adata.var["hb"] = self.adata.var_names.str.upper().str.startswith("HB")
		# removes mt,ribo, and hb entirely for interpretability
		self.adata = self.adata[:, ~(self.adata.var["mt"] |
									 self.adata.var["ribo"] |
									 self.adata.var["hb"])].copy()
		mito_genes = self.adata.var_names.str.startswith("MT-")
		adata_filtered = self.adata[:, ~mito_genes]
		sc.pp.calculate_qc_metrics(self.adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
		sc.pp.filter_cells(self.adata, min_genes=100)
		sc.pp.filter_genes(self.adata, min_cells=10)
		sc.pp.normalize_total(self.adata)
		sc.pp.log1p(self.adata)
		sc.pp.highly_variable_genes(self.adata, n_top_genes=2000, subset=True)

	def filter_groups(self):
		self.adata_group1 = self.adata[self.adata.obs[self.field_name] == self.group1]
		self.adata_group2 = self.adata[self.adata.obs[self.field_name] == self.group2]
		self.adata = ad.concat([self.adata_group1, self.adata_group2, ])
		gc.collect()

	def ensemble_learner(self):

		X = self.adata.X
		y = self.adata.obs[self.field_name].values

		# remove any genes in the self.genes list from X
		if len(self.genes) > 0:
			genes_to_remove = self.genes
			genes_mask = ~self.adata.var_names.isin(genes_to_remove)
			X = X[:, genes_mask]
			self.adata = self.adata[:, genes_mask].copy()
			gc.collect()  # cleanup memory

		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.3, random_state=42, stratify=y
		)

		self.rf_params = {
			"n_estimators": 300,
			"max_depth": None,
			"max_features": "sqrt",
			"min_samples_split": 2,
			"min_samples_leaf": 1,
			"class_weight": "balanced_subsample",
			"n_jobs": -1,
		}

		self.rf = RandomForestClassifier(**self.rf_params, random_state=42)
		self.rf.fit(X_train, y_train)
		predictions = self.rf.predict(X_test)

		self.df = pd.DataFrame({
			"genes": self.adata.var_names,
			"importance": self.rf.feature_importances_
		}).sort_values("importance", ascending=False).reset_index(drop=True)

		self.genes.append(self.df["genes"][0])

		if self.plots:
			self.ensemble_learner_plots()

		#auc score and accuracy
		if len(set(y)) == 2:  # binary
			y_test_binary = [1 if label == self.group2 else 0 for label in y_test]
			predictions_binary = [1 if label == self.group2 else 0 for label in predictions]
			auc = roc_auc_score(y_test_binary, predictions_binary)
		self.accuracy_scores.append(accuracy_score(y_test, predictions))
		self.auc_scores.append(auc)

		#add the top gene and its expression to gene_expression_log
		self.gene_expression_log[self.df["genes"][0]] = self.adata[:, self.df["genes"][0]].X.copy()

	def ensemble_learner_plots(self):
		#scatterplot of the top gene vs log1p_total_counts colored by group
		sns.scatterplot(x=self.adata.obs.log1p_total_counts.values.flatten(),
						y=self.adata[:, self.df["genes"][0]].to_df().values.flatten(),
						hue=self.adata.obs["Disease.Group"])
		plt.xlabel("log1p_total_counts")
		plt.ylabel(self.df["genes"][0])
		plt.show()

		#distribution of the top gene by group
		sns.violinplot(x=self.adata.obs[self.field_name],
					   y=self.adata[:, self.df["genes"][0]].to_df().values.flatten())
		plt.xlabel(self.field_name)
		plt.ylabel(self.df["genes"][0])
		plt.show()
		plt.clf()

		#distiribution of the top gene as a histogram by group
		sns.histplot(data=self.adata.to_df(), x=self.df["genes"][0], hue=self.adata.obs[self.field_name],
					 element="step", stat="density", common_norm=False)
		plt.xlabel(self.df["genes"][0])
		plt.ylabel("Density")
		plt.show()
		plt.clf()

	def cluster_infectivity_gmm(self, gene=None):
		# get the last gene if none provided
		gene = self.genes[-1]

		if self.additive:
			# convert logged gene expressions to a dataframe
			df = pd.DataFrame({
				g: (
					x.toarray().ravel()
					if hasattr(x, "toarray")
					else np.asarray(x).ravel()
				)
				for g, x in self.gene_expression_log.items()
				if g != "summed"
			})
			#df["summed"] = df.sum(axis=1)
			df["summed"] = df.mean(axis=1)  # keeps scale stable

			expr = df["summed"].to_numpy().reshape(-1, 1)
			self.gene_expression_log["summed"] = expr
			self.adata.obs["rf_summed"] = expr.flatten()
			counts = self.adata.obs["log1p_total_counts"].to_numpy().reshape(-1, 1)
			X = np.hstack([counts, expr])
			X = StandardScaler().fit_transform(X)
		else:
			counts = self.adata.obs["log1p_total_counts"].to_numpy().reshape(-1, 1)
			expr = self.adata[:, gene].X
			if hasattr(expr, "toarray"):
				expr = expr.toarray()
			expr = expr.reshape(-1, 1)
			X = np.hstack([counts, expr])
			X = StandardScaler().fit_transform(X)

		gmm = GaussianMixture(n_components=2, random_state=42)
		labels = gmm.fit_predict(X)

		# mean expression of the gene in each GMM cluster, finds which cluster has higher expression, then labels
		means = []
		for k in [0, 1]:
			means.append(expr[labels == k].mean())

		hi = int(np.argmax(means))  # cluster with higher gene expression
		labels = (labels == hi).astype(int)
		key = f"gmm_{gene}"
		self.adata.obs[key] = labels.astype(str)
		self.gmm = gmm

		if self.plots:
			# plot GMM clusters
			sns.scatterplot(x=counts.flatten(),
							y=expr.flatten(),
							hue=self.adata.obs[key], s=3)
			plt.xlabel("log1p_total_counts")
			plt.ylabel(gene)
			plt.title(f"GMM Clusters for {gene}")
			plt.show()
			plt.clf()

		return gmm, labels

	def _flip_rate(self, a, b):
		#checks how many labels changed
		a = np.asarray(a).astype(int).ravel()
		b = np.asarray(b).astype(int).ravel()
		d1 = np.mean(a != b)
		d2 = np.mean((1 - a) != b)
		return float(min(d1, d2))

	def ensemble_recursion(self, iteration=1, additive=False):
		"""
		1) Run GMM clustering on (log1p_total_counts, gene expr)
		2) Train RF to predict GMM cluster labels
		3) Return the top gene feature importance dataframe
		"""
		self.prev_labels = None
		while (iteration <= self.max_iterations):
			print(f"Iteration {iteration}")

			# Step 1: cluster + store labels in self.adata.obs
			gene = self.genes[-1]

			_, labels = self.cluster_infectivity_gmm()
			y = self.adata.obs[f"gmm_{gene}"].values

			#stop if <x% of labels change vs previous iteration
			if hasattr(self, "prev_labels") and self.prev_labels is not None:
				
				fr = self._flip_rate(labels, self.prev_labels)
				
				print(f"flip_rate={fr:.4f}")
				if fr <self.flip_rate_percentage:
					print("stopping: <x% label change vs previous iteration")
					break

			self.prev_labels = labels.copy()

			# Step 2: RF to predict cluster labels
			X = self.adata.X

			# remove already-used genes
			genes_mask = ~self.adata.var_names.isin(self.genes)

			X = X[:, genes_mask]
			adata_rf = self.adata[:, genes_mask].copy()

			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=0.3, random_state=42, stratify=y
			)

			rf = RandomForestClassifier(**self.rf_params, random_state=42)
			rf.fit(X_train, y_train)
			preds = rf.predict(X_test)

			self.df = pd.DataFrame({
				"genes": adata_rf.var_names,
				"importance": rf.feature_importances_
			}).sort_values("importance", ascending=False).reset_index(drop=True)

			# Step 3: add metrics and gene to lists
			auc = float("nan")
			if len(set(y)) == 2:  # binary
				y_test_binary = [1 if label == '1' else 0 for label in y_test]
				proba = rf.predict_proba(X_test)[:, 1]  # P(class=='1')
				auc = roc_auc_score(y_test_binary, proba)
			
			self.auc_scores.append(auc)
			self.accuracy_scores.append(accuracy_score(y_test, preds))
			self.genes.append(self.df["genes"][0])
			self.gene_expression_log[self.df["genes"][0]] = self.adata[:, self.df["genes"][0]].X.copy()

			iteration += 1


	def scanpy_de_original_groups(self, n_top=1000, method="wilcoxon"):
		"""
		Run Scanpy differential expression for group2 vs group1 using self.field_name.
		Stores results in self.de_scanpy (OrderedDict)
		"""

		# Rank genes: group2 vs reference group1
		sc.tl.rank_genes_groups(
			self.adata,
			groupby=self.field_name,
			groups=[self.group2],
			reference=self.group1,
			method=method,
			n_genes=n_top,
			use_raw=False,
		)

		# Convert to a tidy dataframe
		df = sc.get.rank_genes_groups_df(self.adata, group=self.group2)

		# Column names can vary slightly by scanpy version; normalize them
		col_logfc = "logfoldchanges" if "logfoldchanges" in df.columns else ("logfc" if "logfc" in df.columns else None)
		col_pval = "pvals" if "pvals" in df.columns else ("pval" if "pval" in df.columns else None)
		col_padj = "pvals_adj" if "pvals_adj" in df.columns else ("pval_adj" if "pval_adj" in df.columns else None)
		col_score = "scores" if "scores" in df.columns else ("score" if "score" in df.columns else None)

		if col_logfc is None or col_pval is None or col_padj is None:
			raise KeyError(f"Expected DE columns not found. Got columns: {list(df.columns)}")

		# Build ordered dict: top -> bottom
		self.de_dict = OrderedDict()
		for _, row in df.head(n_top).iterrows():
			gene = row["names"]
			self.de_dict[gene] = {
				"logfc": float(row[col_logfc]),
				"pval": float(row[col_pval]),
				"pval_adj": float(row[col_padj]),
				"score": float(row[col_score]) if col_score is not None else None,
			}

	def intersect_genes_and_de(self, top_n_genes=None, keep_order="genes", return_df=True):
		genes_list = list(self.genes)
		if top_n_genes is not None:
			genes_list = genes_list[:int(top_n_genes)]

		de_genes = list(self.de_dict.keys()) if hasattr(self, "de_dict") else []
		de_set = set(de_genes)

		if keep_order == "de":
			intersection = [g for g in de_genes if g in set(genes_list)]
		else:
			intersection = [g for g in genes_list if g in de_set]

		if not return_df:
			return intersection

		# rf rank/order (0 = first picked)
		rf_rank = {g: i for i, g in enumerate(genes_list)}

		#self.rows = []
		for g in intersection:
			stats = self.de_dict.get(g, {})
			self.rows.append({
				"gene": g,
				"rf_rank": rf_rank.get(g, np.nan),
				"logfc": stats.get("logfc", np.nan),
				"pval": stats.get("pval", np.nan),
				"pval_adj": stats.get("pval_adj", np.nan),
				"score": stats.get("score", np.nan),
			})

		merged_df = pd.DataFrame(self.rows).sort_values("rf_rank", ascending=True).reset_index(drop=True)
		return intersection, merged_df
	
	def unique_gene_panel(self):
		#intersection between algorithm genes and DE
		hits, hits_df = self.intersect_genes_and_de()

		#return what's in genes that isn't in hits
		return set(self.genes) - set(hits)

	def plot_common_expression(adata, genes_list):
		
		genes = [g for g in genes_list if g in adata.var_names]
		X = adata[:, genes].X
		if sp.issparse(X):
			X = X.toarray()

		X = np.log1p(X)
		expr = pd.DataFrame(X, index=adata.obs_names, columns=genes)
		corr = expr.corr(method="spearman")
		Z = linkage(corr.values, method="average")
		order = leaves_list(Z)
		corr2 = corr.iloc[order, order]
		genes2 = corr2.columns.tolist()
		plt.figure(figsize=(0.35*len(genes2)+4, 0.35*len(genes2)+4))
		plt.imshow(corr2.values, aspect="auto")
		plt.xticks(range(len(genes2)), genes2, rotation=90)
		plt.yticks(range(len(genes2)), genes2)
		plt.colorbar(label="spearman r")
		plt.tight_layout()
		plt.show()

		#returns a list of genes ordered by clustering
		return genes2, corr2

##################################################################################################################
# Run the algorithm on the reduced Alzheimer's dataset
# Dataset: 	https://cellxgene.cziscience.com/collections/0d35c0fd-ef0b-4b70-bce6-645a4660e5fa
##################################################################################################################

# open the full dataset
adata_alz = sc.read_h5ad("data/alz_reduced.h5ad")

#run the recursive algorithm
model = algo(adata_alz, group1="low", group2="high",
			 field_name="Disease.Group",
			 plots=True,
			 max_iterations=50)


#model properties
print(model.genes)

#shows unique gene panel to the algorithm
print(model.unique_gene_panel)

#plot expression correlation of unique gene panel
model.plot_common_expression(model.adata, model.unique_gene_panel)

#get top DE genes for enrichr
de_df = pd.DataFrame.from_dict(model.de_dict, orient="index")

#filter significant DE genes only
de_sig = (
    de_df
    .query("pval_adj < 0.05")
    .sort_values("pval_adj")
)

de_sig_genes = de_sig.index.tolist()
print("\n".join(de_sig_genes[:100])) 

