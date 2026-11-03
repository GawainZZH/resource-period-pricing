import unittest
import numpy as np
from resource_pricing.controlled_networks import (
    StudyConfig,
    _resource_tolls,
    _vi_gap,
    generate_instance,
    private_cost,
    solve_core_equilibria,
)
from resource_pricing.price_class_gap import (
    build_gap_price_parameterization,
    solve_minimum_price_class_primal_gap,
)


class MinimumPriceClassPrimalGapTests(unittest.TestCase):

    def setUp(self):
        config = StudyConfig(
            n_od=2,
            paths_per_od=2,
            n_dep=6,
            bottleneck_type="delayed-node",
            pressure=0.85,
            reliability="medium",
            window_width=4,
            seed=41,
        )
        self.instance = generate_instance(config)
        _, self.system_optimum = solve_core_equilibria(self.instance, maxiter=260)

    def test_lp_value_matches_direct_primal_gap(self):
        result = solve_minimum_price_class_primal_gap(
            self.instance, self.system_optimum, "corridor"
        )
        direct_gap = _vi_gap(
            self.instance,
            self.system_optimum,
            private_cost(self.instance, self.system_optimum) + result.alternative_price,
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.minimum_gap, direct_gap, delta=1e-07)
        self.assertGreaterEqual(float(np.min(result.resource_prices)), -1e-10)

    def test_minimum_gap_dominates_the_component_price(self):
        result = solve_minimum_price_class_primal_gap(self.instance, self.system_optimum, "node")
        component_gap = _vi_gap(
            self.instance,
            self.system_optimum,
            private_cost(self.instance, self.system_optimum)
            + _resource_tolls(self.instance, self.system_optimum)["node"],
        )
        self.assertLessEqual(result.minimum_gap, component_gap + 1e-07)

    def test_complete_resource_period_price_has_zero_minimum_gap(self):
        parameterization = build_gap_price_parameterization(self.instance, "all_resource")
        result = solve_minimum_price_class_primal_gap(
            self.instance, self.system_optimum, "all_resource"
        )
        self.assertEqual(parameterization.design.shape[0], self.instance.n_alt)
        self.assertTrue(result.success)
        self.assertLessEqual(result.minimum_gap, 2e-06)


if __name__ == "__main__":
    unittest.main()
