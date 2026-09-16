import unittest

import numpy as np
import pandas as pd
import anndata as ad

import recursieve


class TestRecursieve(unittest.TestCase):
    def _make_small_dataset(self):
        rng = np.random.default_rng(0)
        n_group_1 = 80
        n_group_2 = 80
        n_genes = 400

        X = rng.poisson(lam=2.0, size=(n_group_1 + n_group_2, n_genes))
        X[:n_group_1, 0] += 10
        X[:n_group_1, 1] += 8
        X[n_group_1:, 0] += 1
        X[n_group_1:, 1] += 1
        X[n_group_1:, 2] += 12
        X[:n_group_1, 2] += 1

        obs = pd.DataFrame({
            "group": ["group_1"] * n_group_1 + ["group_2"] * n_group_2,
            "sample": ["s1"] * (n_group_1 + n_group_2),
        })
        var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        return adata

    def test_package_import_and_class_exists(self):
        self.assertTrue(hasattr(recursieve, "recursieve"))
        self.assertTrue(callable(recursieve.recursieve))

    def test_model_runs_on_small_dataset(self):
        adata = self._make_small_dataset()

        model = recursieve.recursieve(
            adata=adata,
            group1="group_1",
            group2="group_2",
            field_name="group",
            max_iterations=3,
            plots=False,
            print_to_console=False,
            seed=42,
            n_estimators=50,
            n_jobs=1,
        )

        self.assertIsNotNone(model)
        self.assertGreater(len(model.genes), 0)
        self.assertGreaterEqual(len(model.accuracy_scores), 1)
        self.assertGreaterEqual(len(model.auc_scores), 1)
        self.assertIsInstance(model.unique_gene_panel, list)
        self.assertTrue(all(g in adata.var_names for g in model.genes))

    def test_missing_group_raises_value_error(self):
        adata = self._make_small_dataset().copy()
        adata = adata[adata.obs["group"] == "group_1"].copy()

        with self.assertRaises(ValueError):
            recursieve.recursieve(
                adata=adata,
                group1="group_1",
                group2="group_2",
                field_name="group",
                max_iterations=2,
                plots=False,
                print_to_console=False,
                seed=0,
                n_estimators=20,
                n_jobs=1,
            )

    def test_intersect_genes_and_de_returns_expected_shapes(self):
        adata = self._make_small_dataset()
        model = recursieve.recursieve(
            adata=adata,
            group1="group_1",
            group2="group_2",
            field_name="group",
            max_iterations=2,
            plots=False,
            print_to_console=False,
            seed=7,
            n_estimators=25,
            n_jobs=1,
        )

        hits, df = model.intersect_genes_and_de(top_n_genes=min(5, len(model.genes)))
        self.assertIsInstance(hits, list)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(set(df.columns).issuperset({"gene", "rf_rank", "logfc"}))


if __name__ == "__main__":
    unittest.main()
