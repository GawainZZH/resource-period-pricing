import math
from resource_pricing.controlled_networks import (
    StudyConfig,
    _projection_residuals,
    generate_instance,
    solve_core_equilibria,
    total_cost_gradient,
    private_cost,
)


def test_projection_residuals_are_finite_and_all_resource_is_exact():
    cfg = StudyConfig(
        n_od=3,
        paths_per_od=3,
        n_dep=12,
        bottleneck_type="delayed-node",
        pressure=0.85,
        reliability="tight",
        window_width=4,
        seed=2026062603,
    )
    inst = generate_instance(cfg)
    due_h, so_h = solve_core_equilibria(inst, maxiter=150)
    full_toll = total_cost_gradient(inst, so_h) - private_cost(inst, so_h)
    residuals = _projection_residuals(inst, so_h, full_toll)
    required = ["corridor", "node", "contemporaneous", "all_resource"]
    for key in required:
        assert key in residuals
        assert math.isfinite(residuals[key])
        assert residuals[key] >= 0.0
    assert residuals["all_resource"] <= 1e-07
    assert max(residuals["corridor"], residuals["node"], residuals["contemporaneous"]) > 1e-05
