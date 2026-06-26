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
from sklearn.decomposition import PCA
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, leaves_list


class algo:
	def __init__(self, adata, group1, group2, field_name="sample", plots=False,
				 print_to_console=False, max_iterations=100, additive=True,
				 flip_rate_percentage=0.01, pval_cutoff=0.05, seed=42,
				 summary_method="mean", n_estimators=300, n_jobs=-1):
		"""
		Parameters
		----------
		summary_method : "mean" | "pca"
			How to collapse the running set of selected genes into a single axis
			for the GMM step. "pca" uses the first principal component, which
			handles redundant (co-expressed) genes better than mean.
		seed : int
			Controls RF, train/test split, and GMM. Set differently across runs
			to assess stability.
		"""
		self.adata = adata
		self.group1 = group1
		self.group2 = group2
		self.field_name = field_name
		self.plots = plots
		self.print_to_console = print_to_console
		self.max_iterations = max_iterations
		self.additive = additive
		self.flip_rate_percentage = flip_rate_percentage
		self.pval_cutoff = pval_cutoff
		self.seed = seed
		self.summary_method = summary_method
		self.gene_expression_log = OrderedDict()  # preserves selection order
		self.genes = []
		self.accuracy_scores = []
		self.auc_scores = []
		self.rows = []
		self.label_history = []  # for oscillation detection

		self.rf_params = {
			"n_estimators": n_estimators,
			"max_depth": None,
			"max_features": "sqrt",
			"min_samples_split": 2,
			"min_samples_leaf": 1,
			"class_weight": "balanced_subsample",
			"n_jobs": n_jobs,
		}

		self.preprocessing()
		self.filter_groups()
		self.scanpy_de_original_groups()
		self.ensemble_learner()
		self.ensemble_recursion()
		self.unique_gene_panel = self._unique_gene_panel()

	def replace_var_names(self):
		if "feature_name" in self.adata.var.columns:
			self.adata.var["ensembl_id"] = self.adata.var_names.astype(str)
			self.adata.var_names = self.adata.var["feature_name"].astype(str)
			self.adata.var_names_make_unique()
		else:
			raise KeyError("feature_name not found in adata.var.")

	def preprocessing(self):
		self.adata.layers["counts"] = self.adata.X.copy()
		self.adata.var["mt"] = self.adata.var_names.str.upper().str.startswith("MT-")
		self.adata.var["ribo"] = self.adata.var_names.str.upper().str.match(r"^RPS|^RPL")
		self.adata.var["hb"] = self.adata.var_names.str.upper().str.startswith("HB")
		self.adata = self.adata[:, ~(self.adata.var["mt"] |
									 self.adata.var["ribo"] |
									 self.adata.var["hb"])].copy()
		sc.pp.calculate_qc_metrics(self.adata, qc_vars=["mt", "ribo", "hb"],
								   inplace=True, log1p=True)
		sc.pp.filter_cells(self.adata, min_genes=100)
		sc.pp.filter_genes(self.adata, min_cells=10)
		sc.pp.normalize_total(self.adata)
		sc.pp.log1p(self.adata)
		sc.pp.highly_variable_genes(self.adata, n_top_genes=2000, subset=True)

		# SPEEDUP: densify once if it fits, RF doesn't benefit from sparse
		# and we re-slice X many times
		if sp.issparse(self.adata.X):
			# only densify if reasonable size (< ~4GB float32)
			nbytes_dense = self.adata.n_obs * self.adata.n_vars * 4
			if nbytes_dense < 4e9:
				self.adata.X = self.adata.X.toarray()

	def filter_groups(self):
		mask1 = self.adata.obs[self.field_name] == self.group1
		mask2 = self.adata.obs[self.field_name] == self.group2
		n1, n2 = int(mask1.sum()), int(mask2.sum())
		if n1 == 0 or n2 == 0:
			available = self.adata.obs[self.field_name].unique().tolist()
			raise ValueError(
				f"group1='{self.group1}' has {n1} cells, group2='{self.group2}' "
				f"has {n2} cells in field '{self.field_name}'. "
				f"Available: {available}"
			)
		self.adata = self.adata[mask1 | mask2].copy()
		gc.collect()

	def _get_X_dense(self, adata=None):
		"""Return dense X view; densify on the fly if sparse."""
		a = adata if adata is not None else self.adata
		X = a.X
		if sp.issparse(X):
			X = X.toarray()
		return X

	def ensemble_learner(self):
		X = self._get_X_dense()
		y = self.adata.obs[self.field_name].values

		# remove already-used genes (none on first call, kept for parity)
		if len(self.genes) > 0:
			genes_mask = np.asarray(~self.adata.var_names.isin(self.genes))
			X = X[:, genes_mask]
			self.adata = self.adata[:, genes_mask].copy()
			gc.collect()

		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.3, random_state=self.seed, stratify=y
		)

		self.rf = RandomForestClassifier(**self.rf_params, random_state=self.seed)
		self.rf.fit(X_train, y_train)
		predictions = self.rf.predict(X_test)

		self.df = pd.DataFrame({
			"genes": self.adata.var_names,
			"importance": self.rf.feature_importances_
		}).sort_values("importance", ascending=False).reset_index(drop=True)

		top_gene = self.df["genes"].iloc[0]
		self.genes.append(top_gene)

		if self.plots:
			self.ensemble_learner_plots()
		if self.print_to_console:
			print(top_gene)

		auc = float("nan")
		if len(set(y)) == 2:
			y_test_bin = (y_test == self.group2).astype(int)
			proba = self.rf.predict_proba(X_test)[:, list(self.rf.classes_).index(self.group2)]
			auc = roc_auc_score(y_test_bin, proba)
		self.accuracy_scores.append(accuracy_score(y_test, predictions))
		self.auc_scores.append(auc)

		# store expression as 1D float32 array (memory + downstream simplicity)
		expr = self.adata[:, top_gene].X
		if sp.issparse(expr):
			expr = expr.toarray()
		self.gene_expression_log[top_gene] = np.asarray(expr).ravel().astype(np.float32)

	def ensemble_learner_plots(self):
		top = self.df["genes"].iloc[0]
		expr = self.adata[:, top].X
		if sp.issparse(expr):
			expr = expr.toarray()
		expr = np.asarray(expr).ravel()

		sns.scatterplot(x=self.adata.obs.log1p_total_counts.values.flatten(),
						y=expr, hue=self.adata.obs[self.field_name], s=4)
		plt.xlabel("log1p_total_counts"); plt.ylabel(top); plt.show(); plt.clf()

		sns.violinplot(x=self.adata.obs[self.field_name], y=expr)
		plt.xlabel(self.field_name); plt.ylabel(top); plt.show(); plt.clf()

		sns.histplot(x=expr, hue=self.adata.obs[self.field_name].values,
					 element="step", stat="density", common_norm=False)
		plt.xlabel(top); plt.ylabel("Density"); plt.show(); plt.clf()

	def _summarize_selected(self):
		"""Collapse all selected gene expressions to a single 1D axis."""
		mat = np.column_stack([
			v for k, v in self.gene_expression_log.items() if k != "summed"
		])  # cells x n_selected

		if self.summary_method == "pca" and mat.shape[1] >= 2:
			# first PC handles redundancy: co-expressed genes contribute once
			pc = PCA(n_components=1, random_state=self.seed).fit_transform(
				StandardScaler().fit_transform(mat)
			).ravel()
			return pc.astype(np.float32)
		# default: mean (stable scale across iterations)
		return mat.mean(axis=1).astype(np.float32)

	def cluster_infectivity_gmm(self, gene=None):
		gene = self.genes[-1]

		if self.additive:
			expr = self._summarize_selected().reshape(-1, 1)
			self.gene_expression_log["summed"] = expr.ravel()
			self.adata.obs["rf_summed"] = expr.ravel()
		else:
			e = self.adata[:, gene].X
			if sp.issparse(e):
				e = e.toarray()
			expr = np.asarray(e).reshape(-1, 1)

		counts = self.adata.obs["log1p_total_counts"].to_numpy().reshape(-1, 1)
		X = StandardScaler().fit_transform(np.hstack([counts, expr]))

		gmm = GaussianMixture(n_components=2, random_state=self.seed)
		labels = gmm.fit_predict(X)

		# label cluster with higher expr as "1"
		means = [expr[labels == k].mean() for k in [0, 1]]
		hi = int(np.argmax(means))
		labels = (labels == hi).astype(int)

		self.adata.obs[f"gmm_{gene}"] = labels.astype(str)
		self.gmm = gmm

		if self.plots:
			sns.scatterplot(x=counts.ravel(), y=expr.ravel(),
							hue=self.adata.obs[f"gmm_{gene}"], s=3)
			plt.xlabel("log1p_total_counts"); plt.ylabel(gene)
			plt.title(f"GMM Clusters for {gene}"); plt.show(); plt.clf()

		if self.print_to_console:
			print(gene)
		return gmm, labels

	@staticmethod
	def _flip_rate(a, b):
		a = np.asarray(a).astype(int).ravel()
		b = np.asarray(b).astype(int).ravel()
		return float(min(np.mean(a != b), np.mean((1 - a) != b)))

	def _check_oscillation(self, labels, window=3):
		"""
		Detect 2-cycle oscillation by comparing current labels to the
		label set 2 iterations ago. If flip_rate to t-2 is near zero
		but flip_rate to t-1 is large, we're oscillating.
		"""
		self.label_history.append(labels.copy())
		if len(self.label_history) < window:
			return False
		fr_prev = self._flip_rate(labels, self.label_history[-2])
		fr_prev2 = self._flip_rate(labels, self.label_history[-3])
		# trim history
		if len(self.label_history) > window:
			self.label_history.pop(0)
		return (fr_prev2 < self.flip_rate_percentage) and (fr_prev > self.flip_rate_percentage)

	def ensemble_recursion(self, iteration=1):
		self.prev_labels = None
		while iteration <= self.max_iterations:
			print(f"Iteration {iteration}")
			gene = self.genes[-1]
			_, labels = self.cluster_infectivity_gmm()
			y = self.adata.obs[f"gmm_{gene}"].values

			if self.prev_labels is not None:
				fr = self._flip_rate(labels, self.prev_labels)
				print(f"flip_rate={fr:.4f}")
				if fr < self.flip_rate_percentage:
					print("stopping: <threshold label change vs previous iteration")
					break
				if self._check_oscillation(labels):
					print("stopping: 2-cycle oscillation detected")
					break

			self.prev_labels = labels.copy()

			# RF on remaining genes
			genes_mask = ~self.adata.var_names.isin(self.genes)
			genes_mask = np.asarray(genes_mask)
			X = self._get_X_dense()[:, genes_mask]
			remaining_names = self.adata.var_names[genes_mask]

			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=0.3, random_state=self.seed, stratify=y
			)
			rf = RandomForestClassifier(**self.rf_params, random_state=self.seed)
			rf.fit(X_train, y_train)
			preds = rf.predict(X_test)

			self.df = pd.DataFrame({
				"genes": remaining_names,
				"importance": rf.feature_importances_
			}).sort_values("importance", ascending=False).reset_index(drop=True)

			auc = float("nan")
			if len(set(y)) == 2:
				y_test_bin = (y_test == "1").astype(int)
				pos_idx = list(rf.classes_).index("1")
				proba = rf.predict_proba(X_test)[:, pos_idx]
				auc = roc_auc_score(y_test_bin, proba)
			self.auc_scores.append(auc)
			self.accuracy_scores.append(accuracy_score(y_test, preds))

			top_gene = self.df["genes"].iloc[0]
			self.genes.append(top_gene)

			expr = self.adata[:, top_gene].X
			if sp.issparse(expr):
				expr = expr.toarray()
			self.gene_expression_log[top_gene] = np.asarray(expr).ravel().astype(np.float32)

			if self.print_to_console:
				print(top_gene)
			iteration += 1

	def scanpy_de_original_groups(self, n_top=1000, method="wilcoxon"):
		sc.tl.rank_genes_groups(
			self.adata, groupby=self.field_name,
			groups=[self.group2], reference=self.group1,
			method=method, n_genes=n_top, use_raw=False,
		)
		df = sc.get.rank_genes_groups_df(self.adata, group=self.group2)

		col_logfc = next((c for c in ["logfoldchanges", "logfc"] if c in df.columns), None)
		col_pval = next((c for c in ["pvals", "pval"] if c in df.columns), None)
		col_padj = next((c for c in ["pvals_adj", "pval_adj"] if c in df.columns), None)
		col_score = next((c for c in ["scores", "score"] if c in df.columns), None)
		if not all([col_logfc, col_pval, col_padj]):
			raise KeyError(f"DE columns missing. Got: {list(df.columns)}")

		df = df[df[col_pval] < self.pval_cutoff]
		self.de_dict = OrderedDict()
		for _, row in df.head(n_top).iterrows():
			self.de_dict[row["names"]] = {
				"logfc": float(row[col_logfc]),
				"pval": float(row[col_pval]),
				"pval_adj": float(row[col_padj]),
				"score": float(row[col_score]) if col_score else None,
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

		rf_rank = {g: i for i, g in enumerate(genes_list)}
		self.rows = []
		for g in intersection:
			s = self.de_dict.get(g, {})
			self.rows.append({
				"gene": g, "rf_rank": rf_rank.get(g, np.nan),
				"logfc": s.get("logfc", np.nan), "pval": s.get("pval", np.nan),
				"pval_adj": s.get("pval_adj", np.nan), "score": s.get("score", np.nan),
			})
		merged_df = pd.DataFrame(self.rows).sort_values("rf_rank").reset_index(drop=True)
		return intersection, merged_df

	def _unique_gene_panel(self):
		"""Genes selected by algo but NOT in DE results, preserving RF rank order."""
		hits, _ = self.intersect_genes_and_de()
		hits_set = set(hits)
		# preserve order of self.genes (RF rank) instead of returning a set
		return [g for g in self.genes if g not in hits_set]

	@staticmethod
	def plot_common_expression(adata, genes_list):
		genes = [g for g in genes_list if g in adata.var_names]
		if len(genes) < 2:
			print("Need at least 2 genes to plot correlation.")
			return genes, None
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
		plt.figure(figsize=(0.35 * len(genes2) + 4, 0.35 * len(genes2) + 4))
		plt.imshow(corr2.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
		plt.xticks(range(len(genes2)), genes2, rotation=90)
		plt.yticks(range(len(genes2)), genes2)
		plt.colorbar(label="spearman r")
		plt.tight_layout(); plt.show()
		return genes2, corr2