import tempfile
import unittest
from pathlib import Path
import numpy as np
from resource_pricing.controlled_networks import (
    StudyConfig,
    generate_instance,
    run_batch,
    solve_instance,
)


class ControlledSyntheticStudyTests(unittest.TestCase):

    def test_generate_instance_has_departure_and_resource_horizons(self):
        cfg = StudyConfig(
            n_od=3,
            paths_per_od=3,
            n_dep=6,
            bottleneck_type="delayed-node",
            pressure=0.8,
            reliability="medium",
            window_width=4,
            seed=11,
        )
        inst = generate_instance(cfg)
        self.assertEqual(inst.n_od, 3)
        self.assertEqual(inst.n_dep, 6)
        self.assertGreater(inst.n_res, inst.n_dep)
        self.assertEqual(inst.delta.shape, (inst.n_resources, inst.n_res, inst.n_alt))
        self.assertEqual(inst.capacity.shape, (inst.n_resources, inst.n_res))
        self.assertTrue(np.all(inst.capacity > 0))
        self.assertTrue(np.any(inst.delta[:, inst.n_dep :, :] > 0))

    def test_solve_instance_returns_consistent_efficiency_metrics(self):
        cfg = StudyConfig(
            n_od=2,
            paths_per_od=2,
            n_dep=6,
            bottleneck_type="mixed",
            pressure=0.75,
            reliability="medium",
            window_width=4,
            seed=3,
        )
        inst = generate_instance(cfg)
        result = solve_instance(inst, maxiter=180)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["DUE cost"], result["SO cost"] - 1e-06)
        self.assertGreaterEqual(result["PoA"], 1.0 - 1e-08)
        self.assertIn("Resource-externality recovery", result)
        self.assertIn("Delay-plus-reliability recovery", result)
        self.assertIn("Delayed toll share", result)
        self.assertLessEqual(result["Maximum capacity violation"], 1e-05)

    def test_run_batch_writes_instance_and_summary_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results, summary = run_batch(out, target_instances=4, seed=5, maxiter=120)
            self.assertGreaterEqual(len(results), 4)
            self.assertTrue((out / "controlled_synthetic_instances.csv").exists())
            self.assertTrue((out / "controlled_synthetic_summary.csv").exists())
            self.assertIn("Resource-externality recovery", results.columns)
            self.assertIn("Delay-plus-reliability recovery", results.columns)
            self.assertIn("Bottleneck type", summary.columns)


if __name__ == "__main__":
    unittest.main()
