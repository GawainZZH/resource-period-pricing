import unittest
import pandas as pd
from resource_pricing.controlled_networks import (
    StudyConfig,
    _reliability_threshold,
    generate_instance,
)
from resource_pricing.reliability_sensitivity import completed_network_ids, evaluate_instance


class ReliabilityThresholdSensitivityTests(unittest.TestCase):

    def _config(self, threshold=None):
        return StudyConfig(
            n_od=3,
            paths_per_od=3,
            n_dep=12,
            bottleneck_type="mixed",
            pressure=0.85,
            reliability="medium",
            window_width=4,
            seed=2026072201,
            reliability_threshold=threshold,
        )

    def test_explicit_numeric_threshold_overrides_legacy_mapping(self):
        inst = generate_instance(self._config(threshold=0.67))
        self.assertAlmostEqual(_reliability_threshold(inst), 0.67)

    def test_legacy_threshold_remains_available(self):
        inst = generate_instance(self._config())
        self.assertAlmostEqual(_reliability_threshold(inst), 0.74)

    def test_focused_evaluation_returns_required_metrics(self):
        inst = generate_instance(self._config(threshold=0.75))
        result = evaluate_instance(inst, maxiter=180)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["Reliability threshold"], 0.75)
        self.assertIn("Resource SO support gap", result)
        self.assertIn("Delay-plus-reliability SO support gap", result)
        self.assertIn("Full MC SO support gap", result)
        self.assertLessEqual(result["Maximum capacity violation"], 1e-05)
        self.assertLessEqual(result["Maximum MR-MC price difference"], 1e-09)

    def test_resume_requires_all_thresholds_for_a_network(self):
        rows = [
            {"Network ID": 1, "Reliability threshold": eta}
            for eta in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        ]
        rows.extend(({"Network ID": 2, "Reliability threshold": eta} for eta in [0.6, 0.65, 0.7]))
        completed = completed_network_ids(pd.DataFrame(rows))
        self.assertEqual(completed, {1})


if __name__ == "__main__":
    unittest.main()
