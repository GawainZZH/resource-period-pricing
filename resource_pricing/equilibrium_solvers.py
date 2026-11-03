from __future__ import annotations
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

try:
    from .controlled_networks import (
        OUT,
        StudyConfig,
        _capacity_violation,
        _constraints,
        _initial_flow,
        _minimize_flow,
        due_potential,
        due_potential_gradient,
        generate_instance,
        total_cost,
        total_cost_gradient,
    )
except ImportError:
    from resource_pricing.controlled_networks import (
        OUT,
        StudyConfig,
        _capacity_violation,
        _constraints,
        _initial_flow,
        _minimize_flow,
        due_potential,
        due_potential_gradient,
        generate_instance,
        total_cost,
        total_cost_gradient,
    )
SOLVER_VALIDATION_OUT = OUT / "solver_validation_results"


def _random_od_simplex_flow(inst, rng: np.random.Generator) -> np.ndarray:
    h = np.zeros(inst.n_alt)
    for w in range(inst.n_od):
        idx = np.where(inst.od_of_alt == w)[0]
        weights = rng.dirichlet(np.ones(len(idx)))
        h[idx] = inst.demand[w] * weights
    return h


def _random_feasible_start(inst, rng: np.random.Generator) -> np.ndarray:
    base = _initial_flow(inst)
    candidate = _random_od_simplex_flow(inst, rng)
    if _capacity_violation(inst, candidate) <= 1e-09:
        return candidate
    for weight in np.linspace(0.85, 0.05, 17):
        blended = weight * candidate + (1.0 - weight) * base
        if _capacity_violation(inst, blended) <= 1e-09:
            return blended
    return base


def _solve_multistart(
    inst, objective, gradient, starts: list[np.ndarray], maxiter: int
) -> tuple[np.ndarray, list[float], bool]:
    solutions = []
    values = []
    ok_all = True
    for x0 in starts:
        h, ok, _ = _minimize_flow(inst, objective, gradient, x0, maxiter)
        ok_all = ok_all and ok
        solutions.append(h)
        values.append(float(objective(h)))
    best_idx = int(np.argmin(values))
    return (solutions[best_idx], values, ok_all)


def _relative_spread(values: list[float]) -> float:
    if not values:
        return float("nan")
    return (max(values) - min(values)) / max(1.0, abs(min(values)))


def _kkt_residual(
    inst, h: np.ndarray, gradient: np.ndarray, active_tol: float = 1e-06, flow_tol: float = 1e-07
) -> tuple[float, float]:
    eq, _, _, a_ub, b_ub = _constraints(inst)
    a_eq = np.asarray(eq.A, dtype=float)
    slack = b_ub - a_ub @ h
    active_cap = slack <= active_tol * np.maximum(1.0, np.abs(b_ub))
    zero_flow = h <= flow_tol
    blocks = [a_eq.T]
    lower = [np.full(a_eq.shape[0], -np.inf)]
    upper = [np.full(a_eq.shape[0], np.inf)]
    n_eq = a_eq.shape[0]
    n_cap = int(active_cap.sum())
    n_zero = int(zero_flow.sum())
    if n_cap:
        blocks.append(a_ub[active_cap].T)
        lower.append(np.zeros(n_cap))
        upper.append(np.full(n_cap, np.inf))
    if n_zero:
        blocks.append(-np.eye(inst.n_alt)[:, zero_flow])
        lower.append(np.zeros(n_zero))
        upper.append(np.full(n_zero, np.inf))
    matrix = np.hstack(blocks)
    lb = np.concatenate(lower)
    ub = np.concatenate(upper)
    result = lsq_linear(matrix, -gradient, bounds=(lb, ub), lsmr_tol="auto", max_iter=500)
    stationarity = float(
        np.linalg.norm(matrix @ result.x + gradient) / max(1.0, np.linalg.norm(gradient))
    )
    multipliers = result.x[n_eq:]
    dual_feasibility = float(max(0.0, -np.min(multipliers))) if multipliers.size else 0.0
    return (stationarity, dual_feasibility)


def _max_jensen_violation(inst, rng: np.random.Generator, samples: int = 24) -> float:
    max_violation = 0.0
    objectives = (lambda h: due_potential(inst, h), lambda h: total_cost(inst, h))
    for _ in range(samples):
        h_a = _random_feasible_start(inst, rng)
        h_b = _random_feasible_start(inst, rng)
        theta = float(rng.uniform(0.1, 0.9))
        h_mid = theta * h_a + (1.0 - theta) * h_b
        for objective in objectives:
            lhs = float(objective(h_mid))
            rhs = theta * float(objective(h_a)) + (1.0 - theta) * float(objective(h_b))
            max_violation = max(max_violation, lhs - rhs)
    return max(0.0, max_violation)


def _solver_configs(instances: int, seed: int) -> list[StudyConfig]:
    bottlenecks = ("corridor", "node", "mixed", "delayed-node")
    pressures = (0.6, 0.75, 0.85, 0.95)
    reliability = ("loose", "medium", "tight")
    configs = []
    for idx in range(instances):
        configs.append(
            StudyConfig(
                n_od=3 + idx % 3,
                paths_per_od=3,
                n_dep=12,
                bottleneck_type=bottlenecks[idx % len(bottlenecks)],
                pressure=pressures[idx % len(pressures)],
                reliability=reliability[idx % len(reliability)],
                window_width=4,
                seed=seed + 97 * idx,
            )
        )
    return configs


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield ((first,) + rest)


def _small_grid_check(grid_slices: int, maxiter: int) -> dict[str, float]:
    inst = generate_instance(
        StudyConfig(
            n_od=1,
            paths_per_od=2,
            n_dep=5,
            bottleneck_type="mixed",
            pressure=0.6,
            reliability="loose",
            window_width=2,
            seed=2026062606,
        )
    )
    x0 = _initial_flow(inst)
    due_h, due_ok, _ = _minimize_flow(
        inst,
        lambda h: due_potential(inst, h),
        lambda h: due_potential_gradient(inst, h),
        x0,
        maxiter,
    )
    so_h, so_ok, _ = _minimize_flow(
        inst, lambda h: total_cost(inst, h), lambda h: total_cost_gradient(inst, h), due_h, maxiter
    )
    due_best = float("inf")
    so_best = float("inf")
    feasible_points = 0
    demand = float(inst.demand[0])
    for comp in _compositions(grid_slices, inst.n_alt):
        h = demand * np.asarray(comp, dtype=float) / float(grid_slices)
        if _capacity_violation(inst, h) > 1e-09:
            continue
        feasible_points += 1
        due_best = min(due_best, float(due_potential(inst, h)))
        so_best = min(so_best, float(total_cost(inst, h)))
    due_value = float(due_potential(inst, due_h))
    so_value = float(total_cost(inst, so_h))
    return {
        "grid_feasible_points": float(feasible_points),
        "grid_due_upper_bound_gap": float((due_best - due_value) / max(1.0, abs(due_value))),
        "grid_so_upper_bound_gap": float((so_best - so_value) / max(1.0, abs(so_value))),
        "grid_due_success": float(due_ok),
        "grid_so_success": float(so_ok),
    }


def _solve_validation_instance(
    cfg: StudyConfig, instance_id: int, multistarts: int, seed: int, maxiter: int
) -> dict[str, float | int | str | bool]:
    inst = generate_instance(cfg)
    rng = np.random.default_rng(seed + 1009 * instance_id)
    starts = [_initial_flow(inst)] + [
        _random_feasible_start(inst, rng) for _ in range(multistarts - 1)
    ]
    due_h, due_values, due_ok = _solve_multistart(
        inst,
        lambda h: due_potential(inst, h),
        lambda h: due_potential_gradient(inst, h),
        starts,
        maxiter,
    )
    so_h, so_values, so_ok = _solve_multistart(
        inst, lambda h: total_cost(inst, h), lambda h: total_cost_gradient(inst, h), starts, maxiter
    )
    due_kkt, due_dual = _kkt_residual(inst, due_h, due_potential_gradient(inst, due_h))
    so_kkt, so_dual = _kkt_residual(inst, so_h, total_cost_gradient(inst, so_h))
    jensen = _max_jensen_violation(inst, rng)
    return {
        "Instance": int(instance_id),
        "Alternatives": int(inst.n_alt),
        "Bottleneck type": cfg.bottleneck_type,
        "Pressure": float(cfg.pressure),
        "Reliability": cfg.reliability,
        "DUE multistart objective spread": float(_relative_spread(due_values)),
        "SO multistart objective spread": float(_relative_spread(so_values)),
        "DUE KKT stationarity residual": float(due_kkt),
        "SO KKT stationarity residual": float(so_kkt),
        "DUE dual feasibility residual": float(due_dual),
        "SO dual feasibility residual": float(so_dual),
        "Maximum Jensen violation": float(jensen),
        "DUE capacity violation": float(_capacity_violation(inst, due_h)),
        "SO capacity violation": float(_capacity_violation(inst, so_h)),
        "success": bool(due_ok and so_ok),
    }


def _summarize(results: pd.DataFrame, grid: dict[str, float]) -> dict[str, float]:
    ok = results[results["success"] == True].copy()
    if ok.empty:
        ok = results.copy()
    summary = {
        "instances": float(len(results)),
        "successful_instances": float((results["success"] == True).sum()),
        "mean_alternatives": float(ok["Alternatives"].mean()),
        "mean_due_multistart_spread": float(ok["DUE multistart objective spread"].mean()),
        "max_due_multistart_spread": float(ok["DUE multistart objective spread"].max()),
        "mean_so_multistart_spread": float(ok["SO multistart objective spread"].mean()),
        "max_so_multistart_spread": float(ok["SO multistart objective spread"].max()),
        "mean_due_kkt_residual": float(ok["DUE KKT stationarity residual"].mean()),
        "max_due_kkt_residual": float(ok["DUE KKT stationarity residual"].max()),
        "mean_so_kkt_residual": float(ok["SO KKT stationarity residual"].mean()),
        "max_so_kkt_residual": float(ok["SO KKT stationarity residual"].max()),
        "max_due_dual_feasibility": float(ok["DUE dual feasibility residual"].max()),
        "max_so_dual_feasibility": float(ok["SO dual feasibility residual"].max()),
        "max_jensen_violation": float(ok["Maximum Jensen violation"].max()),
        "max_capacity_violation": float(
            max(ok["DUE capacity violation"].max(), ok["SO capacity violation"].max())
        ),
    }
    summary.update(grid)
    return summary


def run_solver_validation_panel(
    out_dir: Path = SOLVER_VALIDATION_OUT,
    instances: int = 24,
    multistarts: int = 5,
    seed: int = 2026062605,
    maxiter: int = 220,
    grid_slices: int = 36,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _solve_validation_instance(cfg, idx, multistarts, seed, maxiter)
        for idx, cfg in enumerate(_solver_configs(instances, seed))
    ]
    results = pd.DataFrame(rows)
    grid = _small_grid_check(grid_slices=grid_slices, maxiter=maxiter)
    summary = _summarize(results, grid)
    results.to_csv(out_dir / "solver_validation_panel.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "solver_validation_summary.csv", index=False)
    return (results, summary)
