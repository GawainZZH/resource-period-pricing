import unittest
import numpy as np
from resource_pricing.equilibrium_selection import (
    NonmonotoneSelectionInstance,
    enumerate_equilibria,
    run_selection_robust_study,
    solve_class_values,
)


class SelectionRobustPricingTests(unittest.TestCase):

    def setUp(self):
        self.instance = NonmonotoneSelectionInstance(
            name="test", roots=(0.2, 0.5, 0.8), interaction_scale=1.0, price_bound=0.004
        )

    def test_equilibrium_enumeration_recovers_all_unpriced_branches(self):
        equilibria = enumerate_equilibria(self.instance, price_difference=0.0)
        self.assertEqual(len(equilibria), 3)
        np.testing.assert_allclose(equilibria, self.instance.roots, atol=1e-10)

    def test_selection_robust_value_exceeds_optimistic_value(self):
        result = solve_class_values(self.instance, grid_size=1201)
        self.assertLessEqual(result.optimistic_value, result.robust_value + 1e-12)
        self.assertLess(result.optimistic_value, 1e-10)
        self.assertGreater(result.robust_value, 0.01)
        self.assertLess(result.maximum_vi_gap, 1e-09)

    def test_study_exports_values_summary_and_figure(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            values, summary = run_selection_robust_study(out_dir=out, instances=4, grid_size=801)
            self.assertEqual(len(values), 4)
            self.assertIn("Selection-robust class value", values.columns)
            self.assertIn("Selection exposure", values.columns)
            self.assertGreater(summary["positive_selection_exposure_frequency"], 0.0)
            self.assertTrue((out / "selection_robust_values.csv").exists())
            self.assertTrue((out / "selection_robust_summary.csv").exists())
            self.assertTrue((out / "figures" / "selection_robust_class_values.pdf").exists())
            self.assertTrue(
                (out / "figures" / "selection_robust_equilibrium_branches.pdf").exists()
            )
            self.assertTrue(
                (out / "figures" / "selection_robust_class_value_comparison.pdf").exists()
            )


if __name__ == "__main__":
    unittest.main()
