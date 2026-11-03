import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

MODULE = "resource_pricing.convex_programs"


def test_audit_module_exists():
    assert Path("resource_pricing/convex_programs.py").exists()


@pytest.fixture
def audit():
    return importlib.import_module(MODULE)


@pytest.fixture
def problem(audit):
    return audit.ResourceProblem(
        base=np.array([0.0, 0.0]),
        occupation=np.eye(2),
        capacity=np.array([2.0, 2.0]),
        alpha=np.ones(2),
        rho=0.0,
        threshold=0.7,
        conservation=np.ones((1, 2)),
        demand=np.ones(1),
    )


def test_independent_objective_derivatives(problem):
    h = np.array([0.4, 0.6])
    assert problem.cost(h) == pytest.approx((0.4**3 + 0.6**3) / 4)
    assert problem.gradient(h) == pytest.approx(3 * h**2 / 4)
    assert problem.hessian(h) == pytest.approx(np.diag(1.5 * h))


def test_reliability_derivatives_above_and_at_threshold(audit):
    p = audit.ResourceProblem(
        base=np.zeros(1),
        occupation=np.ones((1, 1)),
        capacity=np.array([2.0]),
        alpha=np.zeros(1),
        rho=3.0,
        threshold=0.5,
        conservation=np.ones((1, 1)),
        demand=np.ones(1),
    )
    assert p.cost(np.array([1.0])) == 0
    assert p.gradient(np.array([1.0]))[0] == 0
    assert p.hessian(np.array([1.0]))[0, 0] == 0
    x = np.array([1.4])
    eps = 1e-06
    assert p.cost(x) == pytest.approx(1.4 * 3 * 0.2**2)
    assert p.gradient(x)[0] == pytest.approx(
        (p.cost(x + eps) - p.cost(x - eps)) / (2 * eps), rel=1e-07
    )
    assert p.hessian(x)[0, 0] == pytest.approx(
        (p.gradient(x + eps)[0] - p.gradient(x - eps)[0]) / (2 * eps), rel=1e-07
    )


def test_raw_gap_does_not_hide_infeasible_negative_value(problem, audit):
    h = np.array([0.4, 0.5])
    result = audit.linear_gap(problem, h, np.ones(2))
    assert result["raw_gap"] == pytest.approx(-0.1)
    assert result["demand_violation"] == pytest.approx(0.1)


def test_class_gap_direct_and_dual_agree_at_known_solution(problem, audit):
    h = np.array([0.5, 0.5])
    result = audit.minimum_class_gap(problem, h, np.eye(2))
    assert result["success"]
    assert abs(result["raw_gap"]) < 1e-10
    assert result["duality_difference"] < 1e-10


def test_independent_conic_solution_known_optimum(problem, audit):
    result = audit.solve_conic(problem, tolerance=1e-10)
    assert result["status"] in {"optimal", "optimal_inaccurate"}
    assert result["flow"] == pytest.approx([0.5, 0.5], abs=2e-06)
    assert result["model_objective"] == pytest.approx(0.0625, abs=1e-08)
    assert result["objective_difference"] < 1e-08
    assert "raw_gap" in result


def test_source_hash_and_frozen_seeds(audit):
    source = audit.load_frozen_source()
    assert len(source) == 200
    assert len(audit.FROZEN_SEEDS) == 13
    assert set(audit.FROZEN_SEEDS).issubset(set(source["Seed"]))


def test_feasibility_reports_all_constraint_families(problem):
    checks = problem.feasibility(np.array([-0.2, 2.3]))
    assert checks["nonnegativity_violation"] == pytest.approx(0.2)
    assert checks["demand_violation"] == pytest.approx(1.1)
    assert checks["capacity_violation"] == pytest.approx(0.3)


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_invalid_coefficients_rejected(audit, bad):
    with pytest.raises(ValueError):
        audit.ResourceProblem(
            base=np.zeros(1),
            occupation=np.ones((1, 1)),
            capacity=np.array([bad]),
            alpha=np.ones(1),
            rho=0,
            threshold=0.7,
            conservation=np.ones((1, 1)),
            demand=np.ones(1),
        )


@pytest.mark.parametrize("start", [[0.4, 0.6], [0, 1]])
def test_active_face_newton_refines_and_releases_zero_flow(problem, audit, start):
    result = audit.refine_active_face(problem, np.array(start, dtype=float))
    assert result["flow"] == pytest.approx([0.5, 0.5], abs=1e-09)
    assert abs(result["raw_gap"]) < 1e-10
    assert result["demand_violation"] < 1e-10


def test_active_face_newton_respects_binding_capacity(audit):
    p = audit.ResourceProblem(
        base=np.array([-10.0, 0]),
        occupation=np.eye(2),
        capacity=np.array([0.3, 2]),
        alpha=np.ones(2),
        rho=0,
        threshold=0.7,
        conservation=np.ones((1, 2)),
        demand=np.ones(1),
    )
    result = audit.refine_active_face(p, np.array([0.1, 0.9]))
    assert result["flow"] == pytest.approx([0.3, 0.7], abs=1e-09)
    assert abs(result["raw_gap"]) < 1e-09
    assert result["capacity_violation"] < 1e-10


def test_full_audit_preserves_source_and_returns_all_classes(audit, tmp_path):
    solutions, classes = audit.run_full_audit(tmp_path, seeds=(audit.FROZEN_SEEDS[0],))
    assert len(solutions) == 3
    assert len(classes) == 12
    assert set(classes.price_class) == {"corridor", "node", "contemporaneous", "all_resource"}
    assert classes[classes.method == "active_face_newton"]["SO_precision_1e-08"].all()
    assert (tmp_path / "flows_2026062500000.npz").exists()
    assert len(audit.load_frozen_source()) == 200
    plotted = audit.export_refined_plot_data(tmp_path)
    assert len(plotted) == 1
    assert plotted["AllResource projection residual"].max() < 1e-10
    assert plotted["success"].all()


def test_gap_upper_bound_does_not_accept_tiny_cost_difference(audit):
    p = audit.ResourceProblem(
        base=np.array([0, 5e-11]),
        occupation=np.eye(2),
        capacity=np.array([2000.0, 2000.0]),
        alpha=np.zeros(2),
        rho=0,
        threshold=0.7,
        conservation=np.ones((1, 2)),
        demand=np.array([1000.0]),
    )
    gap = audit.linear_gap(p, np.array([0.0, 1000.0]), p.base)
    assert gap["gap_upper"] >= 4.99e-08
    refined = audit.refine_active_face(p, np.array([0.0, 1000.0]))
    assert refined["gap_upper"] < 1e-10
    assert refined["flow"][0] == pytest.approx(1000)


def test_small_positive_flow_initialization_keeps_feasibility(problem, audit):
    refined = audit.refine_active_face(problem, np.array([1 - 5e-08, 5e-08]))
    assert refined["demand_violation"] < 1e-10
    assert refined["flow"] == pytest.approx([0.5, 0.5], abs=1e-08)


def test_minimum_gap_handles_small_cost_difference(audit):
    p = audit.ResourceProblem(
        base=np.array([1.0, 1.0 + 5e-11]),
        occupation=np.ones((1, 2)),
        capacity=np.array([2000.0]),
        alpha=np.ones(1),
        rho=0,
        threshold=0.7,
        conservation=np.ones((1, 2)),
        demand=np.array([1000.0]),
    )
    gap = audit.minimum_class_gap(p, np.array([0.0, 1000.0]), np.ones((2, 1)))
    assert gap["success"]
    assert gap["gap_upper"] >= 4.99e-08
    assert gap["raw_gap"] == pytest.approx(5e-08, rel=0.0001)


def test_failed_classification_is_unresolved_not_positive(audit):
    assert (
        audit.classify_gap({"success": False, "raw_gap": float("nan")}, False, 1e-07)
        == "unresolved"
    )


def test_illustrative_audit_checks_complete_so_precision(audit, tmp_path):
    solutions, classes = audit.run_illustrative_audit(tmp_path)
    assert len(solutions) == 3
    assert len(classes) == 4
    assert classes["classification_1e-08"].tolist() == ["zero", "positive", "zero", "zero"]
    assert solutions.iloc[-1].gap_upper <= 1e-10


def test_infeasible_oracle_below_dual_bound_is_not_hidden(audit, monkeypatch):
    p = audit.ResourceProblem(
        base=np.array([1000.0, 0]),
        occupation=np.eye(2),
        capacity=np.array([2.0, 2.0]),
        alpha=np.zeros(2),
        rho=0,
        threshold=0.7,
        conservation=np.ones((1, 2)),
        demand=np.ones(1),
    )
    fake = SimpleNamespace(
        success=True,
        x=np.array([-5e-11, 1 + 5e-11]),
        ineqlin=SimpleNamespace(marginals=np.zeros(2)),
        eqlin=SimpleNamespace(marginals=np.zeros(1)),
    )
    monkeypatch.setattr(audit, "linprog", lambda *args, **kwargs: fake)
    result = audit.linear_gap(p, np.array([0.0, 1]), p.base)
    assert result["LP_uncertainty"] >= 4.99e-08


def test_inconsistent_bounds_are_unresolved(audit):
    result = {"success": True, "raw_gap": 5e-08, "gap_upper": 0, "uncertainty": 0}
    assert audit.classify_gap(result, True, 1e-08) == "unresolved"


def test_plot_export_refuses_unresolved_rows(audit, tmp_path):
    import pandas as pd

    audit.run_full_audit(tmp_path, seeds=(audit.FROZEN_SEEDS[0],))
    classes = pd.read_csv(tmp_path / "class_gaps.csv")
    classes.loc[classes.method == "active_face_newton", "classification_1e-08"] = "unresolved"
    classes.to_csv(tmp_path / "class_gaps.csv", index=False)
    with pytest.raises(ValueError, match="Unresolved"):
        audit.export_refined_plot_data(tmp_path)
