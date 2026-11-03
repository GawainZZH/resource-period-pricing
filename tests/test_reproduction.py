from pathlib import Path
import unittest

import numpy as np

import reproduce

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def test_constructed_inputs_match_reported_sample(self):
        cases = reproduce.load_inputs("constructed")
        self.assertEqual(len(cases), 200)
        self.assertEqual(len({c["seed"] for c in cases}), 200)
        self.assertEqual(cases[0]["seed"], 2026062500000)

    def test_random_inputs_include_all_generated_networks(self):
        cases = reproduce.load_inputs("random")
        self.assertEqual(len(cases), 300)
        self.assertEqual(cases[0]["seed"], 2026062608000)

    def test_summary_recovers_constructed_results(self):
        summary = reproduce.saved_summary()
        row = summary[summary["sample"] == "constructed"].iloc[0]
        self.assertEqual(row["instances"], 200)
        self.assertAlmostEqual(row["mean_poa"], 1.054, places=3)
        self.assertAlmostEqual(row["corridor_gcr"], 0.571, places=3)
        self.assertAlmostEqual(row["node_gcr"], 0.403, places=3)
        self.assertAlmostEqual(row["all_resource_gcr"], 0.977, places=3)
        self.assertAlmostEqual(row["contemporaneous_gcr"], 0.235, places=3)

    def test_summary_retains_random_acceptance_rule(self):
        summary = reproduce.saved_summary()
        row = summary[summary["sample"] == "random"].iloc[0]
        self.assertEqual(row["instances"], 295)

    def test_exact_example_responses(self):
        result = reproduce.exact_examples()
        self.assertEqual(result["omitted_node_loss"], "1/2")
        self.assertEqual(result["complete_node_loss"], "0")

    def test_reservation_grid_inputs(self):
        cases = reproduce.load_inputs("reservation_grids")
        self.assertEqual(cases["resource_minutes"], [1, 1.2, 1.5, 2, 3, 6])
        self.assertEqual(cases["capacity_multipliers"], [1, 2, 3, 5, 7, 10])

    def test_protected_target_is_portable(self):
        from resource_pricing.exact_prices import load_target

        model, flow, cost = load_target()
        self.assertEqual(len(flow), 180)
        self.assertEqual(len(cost), 180)
        self.assertLessEqual(max(model.feasibility(np.asarray(flow, dtype=float)).values()), 1e-8)

    def test_regenerated_networks_match_saved_matrices(self):
        from resource_pricing.convex_programs import from_instance

        for instance in reproduce.generated_instances(200):
            problem = from_instance(instance)
            path = ROOT / "data/class_gaps/constructed" / f"flows_{instance.config.seed}.npz"
            with np.load(path, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(problem.occupation, arrays["occupation"])
                # Capacity construction includes floating-point matrix products.
                np.testing.assert_allclose(
                    problem.capacity,
                    arrays["capacity"],
                    rtol=16 * np.finfo(float).eps,
                    atol=0.0,
                )
                np.testing.assert_array_equal(problem.demand, arrays["demand"])


if __name__ == "__main__":
    unittest.main()
