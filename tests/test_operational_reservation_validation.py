import importlib
import numpy as np
import pytest


@pytest.fixture
def operational():
    return importlib.import_module("resource_pricing.nominal_reservations")


def test_knot_conversion(operational):
    assert operational.knots_to_km_min(120) == pytest.approx(3.704)


def test_outward_rounding_and_half_open_intervals(operational):
    assert operational.reservation_slots(1.5, 2, 1.5) == (1, 3)
    assert operational.reservation_slots(0, 6, 3) == (0, 2)
    assert operational.reservation_slots(6, 6, 3) == (2, 4)


@pytest.mark.parametrize(
    "arguments", [(0, 0, 3), (-1, 2, 3), (0, 2, 0), (0, np.nan, 3), (2, 2, 1.5)]
)
def test_invalid_intervals_rejected(operational, arguments):
    with pytest.raises(ValueError):
        operational.reservation_slots(*arguments)


def test_physical_topology_and_fixed_choice_set(operational):
    case = operational.build_operational_case()
    assert len(case.problem.base) == 180
    assert case.problem.conservation.shape == (10, 180)
    assert case.problem.demand.sum() == 40
    assert case.metadata["corridor_edges"] == 57
    assert case.metadata["resource_count"] == 130
    assert np.all(case.problem.occupation.sum(axis=0) > 0)
    for grid in (1.5, 6):
        other = operational.build_operational_case(grid_minutes=grid)
        np.testing.assert_array_equal(other.choices, case.choices)
        np.testing.assert_allclose(other.problem.base, case.problem.base)


def test_capacity_not_derived_from_demand(operational):
    a = operational.build_operational_case(demand_per_od=4)
    b = operational.build_operational_case(demand_per_od=8)
    np.testing.assert_array_equal(a.problem.capacity, b.problem.capacity)
    np.testing.assert_array_equal(a.problem.occupation, b.problem.occupation)


def test_terminal_capacity_is_explicit_not_load_adjustment(operational):
    case = operational.build_operational_case()
    terminal_rows = (case.row_resource >= 114) & (case.row_resource < 126)
    assert np.all(case.problem.capacity[terminal_rows] == 2)
    single = operational.build_operational_case(terminal_channels=1)
    assert np.all(single.problem.capacity[terminal_rows] == 1)


def test_unit_weighted_price_mapping(operational):
    case = operational.build_operational_case()
    D = case.problem.occupation
    for name in ("corridor", "node", "contemporaneous"):
        P = case.price_map(name)
        assert P.shape[0] == 180
        assert np.all(P >= 0)
    np.testing.assert_allclose(case.price_map("all"), D.T * 3)
    assert np.max(case.price_map("contemporaneous").sum(axis=1)) == 6


def test_scale_preserves_per_flight_cost(operational):
    case = operational.build_operational_case()
    scaled = operational.scale_problem(case.problem, 5)
    h = np.ones(180) * (4 / 18)
    assert scaled.cost(5 * h) == pytest.approx(5 * case.problem.cost(h))
    np.testing.assert_allclose(scaled.private(5 * h), case.problem.private(h))


def test_integer_projection_and_failed_fractional_demand(operational):
    from resource_pricing.convex_programs import ResourceProblem

    p = ResourceProblem(
        np.array([1.0, 2.0]),
        np.eye(2),
        np.ones(2),
        np.ones(2) * 0.1,
        0,
        0.7,
        np.ones((1, 2)),
        np.array([1.0]),
    )
    result = operational.integer_projection(p, np.array([0.4, 0.6]))
    assert result["success"]
    assert result["flow"] == pytest.approx([0, 1])
    assert max(p.feasibility(result["flow"]).values()) == 0
    p.demand[:] = 0.5
    with pytest.raises(ValueError):
        operational.integer_projection(p, np.array([0.25, 0.25]))


def test_differentiable_equilibrium_potential(operational):
    case = operational.build_operational_case()
    h = np.ones(180) * 0.05
    d = np.linspace(-1, 1, 180) * 0.01
    eps = 1e-05
    numerical = (
        operational.potential(case.problem, h + eps * d)
        - operational.potential(case.problem, h - eps * d)
    ) / (2 * eps)
    assert numerical == pytest.approx(case.problem.private(h) @ d, rel=1e-07, abs=1e-07)


def test_original_departure_centers_preserved(operational):
    from resource_pricing.controlled_networks import build_metropolitan_corridor_instance

    legacy, _ = build_metropolitan_corridor_instance()
    case = operational.build_operational_case()
    actual = np.array(case.metadata["preferred_departure_minutes"])
    expected = [
        6
        * legacy.dep_of_alt[
            np.flatnonzero(legacy.od_of_alt == w)[
                np.argmin(legacy.schedule_penalty[legacy.od_of_alt == w])
            ]
        ]
        for w in range(10)
    ]
    np.testing.assert_array_equal(actual, expected)


def test_solver_exception_returns_failure_not_flow(operational, monkeypatch):
    import cvxpy as cp

    def failed_solve(*args, **kwargs):
        raise cp.error.SolverError("controlled failure")

    monkeypatch.setattr(cp.Problem, "solve", failed_solve)
    result = operational.solve_reservation(operational.build_operational_case().problem)
    assert result["success"] is False
    assert "flow" not in result
    assert "controlled failure" in result["status"]
