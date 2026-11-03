import math
from resource_pricing.controlled_networks import StudyConfig, generate_instance, solve_instance


def test_solve_instance_reports_estimated_toll_metrics():
    cfg = StudyConfig(
        n_od=3,
        paths_per_od=3,
        n_dep=12,
        bottleneck_type="delayed-node",
        pressure=0.85,
        reliability="tight",
        window_width=4,
        seed=2026062601,
    )
    row = solve_instance(generate_instance(cfg), maxiter=160)
    required = [
        "DUE-estimated resource recovery",
        "DUE-estimated delay+reliability recovery",
        "Iterative resource recovery",
        "Iterative delay+reliability recovery",
        "Iterative resource iterations",
        "Iterative delay+reliability iterations",
        "DUE-estimated resource VI residual",
        "Iterative resource VI residual",
    ]
    for key in required:
        assert key in row
        assert math.isfinite(float(row[key]))
    assert row["Iterative resource iterations"] >= 1
    assert row["Iterative delay+reliability iterations"] >= 1
