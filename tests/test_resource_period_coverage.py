import importlib
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import linprog
from resource_pricing.convex_programs import ResourceProblem


def test_coverage_module_exists():
    assert Path("resource_pricing/coverage.py").exists()


@pytest.fixture
def coverage():
    return importlib.import_module("resource_pricing.coverage")


def problem(occupation=None, capacity=None, classes=None, demand=None):
    D = np.eye(2) if occupation is None else np.asarray(occupation, float)
    H = np.ones((1, D.shape[1])) if classes is None else np.asarray(classes, float)
    return ResourceProblem(
        base=np.zeros(D.shape[1]),
        occupation=D,
        capacity=np.full(D.shape[0], 10.0) if capacity is None else np.asarray(capacity, float),
        alpha=np.zeros(D.shape[0]),
        rho=0,
        threshold=0.7,
        conservation=H,
        demand=np.ones(H.shape[0]) if demand is None else np.asarray(demand, float),
    )


def independent_price_lp(p, h, P, cost, domain, cap=None):
    n, k = P.shape
    m, q = (len(p.capacity), len(p.demand))
    theta_bounds = [(None, None)] * k if domain == "signed" else [(0, None)] * k
    if domain == "capped":
        theta_bounds = [(0, float(v)) for v in cap]
    result = linprog(
        np.r_[P.T @ h, p.capacity, p.demand],
        A_ub=-np.column_stack([P, p.occupation.T, p.conservation.T]),
        b_ub=cost,
        bounds=theta_bounds + [(0, None)] * m + [(None, None)] * q,
        method="highs-ds",
    )
    assert result.success
    return float(cost @ h + result.fun)


def test_equal_visible_occupation_is_boundary_obstruction(coverage):
    p = problem()
    h = np.array([0.0, 1.0])
    P = np.ones((2, 1))
    for domain in ("nonnegative", "signed"):
        result = coverage.resource_coverage_gap(p, h, P, cost=np.array([1.0, 1.5]), domain=domain)
        assert result["gap"] == pytest.approx(0.5)
        assert result["witness"] == pytest.approx([1.0, 0.0])
        assert result["visible_change"] == pytest.approx([0.0])


def test_decreasing_visible_load_distinguishes_signed_from_nonnegative(coverage):
    p = problem()
    h = np.array([0.0, 1.0])
    P = np.array([[0.0], [1.0]])
    cost = np.array([1.0, 1.5])
    positive = coverage.resource_coverage_gap(p, h, P, cost=cost)
    signed = coverage.resource_coverage_gap(p, h, P, cost=cost, domain="signed")
    assert positive["gap"] == pytest.approx(0.5)
    assert positive["visible_change"] == pytest.approx([-1.0])
    assert signed["gap"] == pytest.approx(0.0)


@pytest.mark.parametrize("cap,expected", [(0.0, 0.5), (0.2, 0.3), (0.5, 0.0), (2.0, 0.0)])
def test_price_cap_insufficient_despite_full_coverage(coverage, cap, expected):
    p = problem()
    result = coverage.resource_coverage_gap(
        p,
        np.array([0.0, 1.0]),
        np.array([[1.0], [0.0]]),
        cost=np.array([1.0, 1.5]),
        domain="capped",
        price_cap=np.array([cap]),
    )
    assert result["gap"] == pytest.approx(expected)


def test_hard_capacity_can_remove_the_same_obstruction(coverage):
    p = problem(capacity=[0.5, 1.0])
    h = np.array([0.5, 0.5])
    result = coverage.resource_coverage_gap(p, h, np.ones((2, 1)), cost=np.array([1.0, 1.5]))
    assert result["gap"] == pytest.approx(0)
    assert result["witness"] == pytest.approx(h)


@pytest.mark.parametrize("domain", ["nonnegative", "signed", "capped"])
def test_random_multiclass_flow_and_price_lps_agree(coverage, domain):
    rng = np.random.default_rng(814)
    H = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    for _ in range(12):
        D = rng.uniform(0.1, 1.2, (5, 6))
        h = np.r_[rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))]
        p = problem(D, D @ h + rng.uniform(0.0, 0.2, 5), H)
        P = rng.uniform(0.0, 1.0, (6, 3))
        cost = rng.uniform(0.0, 3.0, 6)
        cap = rng.uniform(0.0, 0.8, 3) if domain == "capped" else None
        result = coverage.resource_coverage_gap(p, h, P, cost=cost, domain=domain, price_cap=cap)
        expected = independent_price_lp(p, h, P, cost, domain, cap)
        assert result["gap"] == pytest.approx(expected, abs=1e-09)
        assert result["feasibility_violation"] < 1e-09
        assert result["objective_discrepancy"] < 1e-09


def test_zero_parameter_domain_is_unpriced_gap(coverage):
    p = problem()
    result = coverage.resource_coverage_gap(
        p, np.array([0.0, 1.0]), np.zeros((2, 0)), cost=np.array([1.0, 1.5])
    )
    assert result["gap"] == pytest.approx(0.5)
    assert result["visible_change"].shape == (0,)


def test_tiny_cost_difference_is_not_hidden_by_class_constant(coverage):
    p = problem()
    result = coverage.resource_coverage_gap(
        p, np.array([0.0, 1.0]), np.zeros((2, 0)), cost=np.array([1.0, 1.0 + 1e-11])
    )
    assert result["gap"] == pytest.approx(1e-11, rel=0.0001, abs=1e-15)


@pytest.mark.parametrize("domain", ["signed", "nonnegative"])
def test_tiny_mapping_coefficients_do_not_hide_visible_increases(coverage, domain):
    result = coverage.resource_coverage_gap(
        problem(),
        np.array([0.0, 1.0]),
        np.array([[1e-12], [0.0]]),
        cost=np.array([1.0, 1.5]),
        domain=domain,
    )
    assert result["gap"] == pytest.approx(0, abs=1e-12)


def test_large_cap_on_zero_mapping_does_not_suppress_cost(coverage):
    result = coverage.resource_coverage_gap(
        problem(),
        np.array([0.0, 1.0]),
        np.zeros((2, 1)),
        cost=np.array([1.0, 1.5]),
        domain="capped",
        price_cap=np.array([1000000000000.0]),
    )
    assert result["gap"] == pytest.approx(0.5)


@pytest.mark.parametrize("domain", ["signed", "nonnegative"])
def test_tiny_contrast_in_class_constant_mapping_is_preserved(coverage, domain):
    result = coverage.resource_coverage_gap(
        problem(),
        np.array([1.0, 0.0]),
        np.array([[1.0], [1.0 + 1e-12]]),
        cost=np.array([1.5, 1.0]),
        domain=domain,
    )
    assert result["gap"] == pytest.approx(0, abs=1e-12)


def test_solver_nan_objective_is_rejected(coverage, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        coverage,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, x=np.array([0.0, 1.0]), fun=float("nan"), message="invalid objective"
        ),
    )
    with pytest.raises(RuntimeError):
        coverage.resource_coverage_gap(problem(), np.array([0.0, 1.0]), np.zeros((2, 0)))


@pytest.mark.parametrize(
    "change", ["infeasible", "nonfinite_cost", "wrong_mapping", "invalid_domain", "negative_cap"]
)
def test_invalid_input_does_not_become_zero_gap(coverage, change):
    p = problem()
    h, P, cost, domain, cap = (np.array([0.0, 1.0]), np.eye(2), np.ones(2), "nonnegative", None)
    if change == "infeasible":
        h[1] = 0.8
    elif change == "nonfinite_cost":
        cost[0] = np.nan
    elif change == "wrong_mapping":
        P = np.ones((3, 1))
    elif change == "invalid_domain":
        domain = "unknown"
    else:
        domain, cap = ("capped", np.array([-1.0, 1.0]))
    with pytest.raises(ValueError):
        coverage.resource_coverage_gap(p, h, P, cost=cost, domain=domain, price_cap=cap)
