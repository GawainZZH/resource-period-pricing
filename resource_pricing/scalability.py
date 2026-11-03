from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize_scalar

try:
    from .controlled_networks import (
        OUT,
        StudyConfig,
        _binding_frequency,
        _capacity_violation,
        _constraints,
        _flat_load,
        _initial_flow,
        _minimize_flow,
        _resource_tolls,
        _vi_gap,
        due_potential,
        due_potential_gradient,
        generate_instance,
        private_cost,
        total_cost,
        total_cost_gradient,
    )
except ImportError:
    from resource_pricing.controlled_networks import (
        OUT,
        StudyConfig,
        _binding_frequency,
        _capacity_violation,
        _constraints,
        _flat_load,
        _initial_flow,
        _minimize_flow,
        _resource_tolls,
        _vi_gap,
        due_potential,
        due_potential_gradient,
        generate_instance,
        private_cost,
        total_cost,
        total_cost_gradient,
    )
SCALABILITY_OUT = OUT / "scalability_results"


def _scale_config(target_alternatives: int, instance_id: int, seed: int) -> StudyConfig:
    settings = {
        120: (4, 3, 10),
        500: (10, 5, 10),
        1000: (20, 5, 10),
        2000: (40, 5, 10),
        5000: (50, 10, 10),
    }
    n_od, paths_per_od, window_width = settings.get(
        int(target_alternatives), (max(4, int(round(target_alternatives / 50))), 5, 10)
    )
    bottlenecks = ("mixed", "node", "delayed-node", "corridor")
    pressures = (0.75, 0.85, 0.95)
    reliability_levels = ("medium", "tight", "loose")
    return StudyConfig(
        n_od=n_od,
        paths_per_od=paths_per_od,
        n_dep=24,
        bottleneck_type=bottlenecks[instance_id % len(bottlenecks)],
        pressure=pressures[instance_id % len(pressures)],
        reliability=reliability_levels[instance_id % len(reliability_levels)],
        window_width=window_width,
        seed=seed + 1009 * int(target_alternatives) + instance_id,
    )


def _recovery(due_cost: float, priced_cost: float, so_cost: float) -> float:
    denominator = due_cost - so_cost
    if abs(denominator) < 1e-09:
        return 0.0
    return (due_cost - priced_cost) / denominator


def _project_simplex(values: np.ndarray, total: float) -> np.ndarray:
    if total <= 0.0:
        return np.zeros_like(values)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - total
    positive = ordered - cumulative / np.arange(1, len(values) + 1) > 0.0
    if not np.any(positive):
        return np.full_like(values, total / max(len(values), 1))
    rho = int(np.where(positive)[0][-1])
    theta = cumulative[rho] / float(rho + 1)
    return np.maximum(values - theta, 0.0)


def _project_od_simplexes(inst, target: np.ndarray) -> np.ndarray:
    projected = np.zeros_like(target)
    for w in range(inst.n_od):
        idx = np.where(inst.od_of_alt == w)[0]
        projected[idx] = _project_simplex(target[idx], float(inst.demand[w]))
    return projected


def _simplex_vi_residual(inst, h: np.ndarray, cost: np.ndarray) -> float:
    gap = 0.0
    scale = 1.0
    for w in range(inst.n_od):
        idx = np.where(inst.od_of_alt == w)[0]
        local_h = h[idx]
        local_cost = cost[idx]
        gap += float(np.dot(local_h, local_cost) - inst.demand[w] * np.min(local_cost))
        scale += float(np.dot(local_h, np.abs(local_cost)))
    return max(gap, 0.0) / max(scale, 1e-09)


def _projected_gradient_simplex(
    inst, cost_map, x0: np.ndarray, maxiter: int, initial_step: float = 0.18, tol: float = 1e-07
) -> tuple[np.ndarray, float, int]:
    h = _project_od_simplexes(inst, x0)
    for iteration in range(1, maxiter + 1):
        step = initial_step / np.sqrt(float(iteration))
        h_next = _project_od_simplexes(inst, h - step * cost_map(h))
        move = float(np.linalg.norm(h_next - h) / max(1.0, np.linalg.norm(h)))
        h = h_next
        if move < tol:
            break
    return (h, _simplex_vi_residual(inst, h, cost_map(h)), iteration)


def _inflate_capacity_to_nonbinding(inst, margin: float = 20.0) -> None:
    load = _flat_load(inst, _initial_flow(inst))
    inst.capacity = np.maximum(inst.capacity, margin * np.maximum(load, 1e-09))
    inst.peak_pressure = float(np.max(load / inst.capacity))


def _normalize_vi_gap(inst, h: np.ndarray, cost: np.ndarray) -> float:
    gap = _vi_gap(inst, h, cost)
    if not np.isfinite(gap):
        return np.nan
    scale = 1.0 + float(np.dot(h, np.abs(cost)))
    return float(gap / max(scale, 1e-09))


def _occupied_binding_frequency(inst, h: np.ndarray, tol: float = 1e-05) -> float:
    load = _flat_load(inst, h)
    occupied = load > 1e-08
    if not np.any(occupied):
        return 0.0
    return float(np.mean(load[occupied] >= (1.0 - tol) * inst.capacity[occupied]))


def _feasible_hard_capacity_flow(inst) -> np.ndarray:
    eq, _, _, a_ub, b_ub = _constraints(inst)
    res = linprog(
        c=np.zeros(inst.n_alt),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=eq.A,
        b_eq=inst.demand,
        bounds=[(0, None)] * inst.n_alt,
        method="highs-ds",
    )
    if not res.success:
        raise RuntimeError(f"Hard-capacity feasible flow not found: {res.message}")
    return np.maximum(res.x, 0.0)


def _frank_wolfe_hard_capacity(
    inst, objective, gradient, x0: np.ndarray, maxiter: int, tol: float = 5e-06
) -> tuple[np.ndarray, float, int, bool, str]:
    eq, _, _, a_ub, b_ub = _constraints(inst)
    h = np.maximum(x0, 0.0)
    if _capacity_violation(inst, h) > 1e-08:
        h = _feasible_hard_capacity_flow(inst)
    residual = np.inf
    message = "iteration limit reached"
    for iteration in range(1, maxiter + 1):
        grad = gradient(h)
        linear = linprog(
            c=grad,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=eq.A,
            b_eq=inst.demand,
            bounds=[(0, None)] * inst.n_alt,
            method="highs-ds",
        )
        if not linear.success:
            return (h, np.nan, iteration, False, str(linear.message))
        search = np.maximum(linear.x, 0.0)
        direction = search - h
        raw_gap = max(0.0, float(np.dot(grad, h - search)))
        residual = raw_gap / max(1.0 + float(np.dot(h, np.abs(grad))), 1e-09)
        if residual <= tol:
            return (h, residual, iteration, True, "Frank-Wolfe residual tolerance reached")
        line = minimize_scalar(
            lambda step: objective(h + step * direction),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 0.0001},
        )
        step = float(line.x) if line.success else 2.0 / float(iteration + 2)
        h = np.maximum(h + step * direction, 0.0)
        message = "Frank-Wolfe iteration limit reached"
    residual = _normalize_vi_gap(inst, h, gradient(h))
    return (h, float(residual), maxiter, residual <= 0.0003, message)


def _augment_resources_for_active_panel(inst, target_resources: int, seed: int) -> None:
    extra_count = max(0, int(target_resources) - int(inst.n_resources))
    if extra_count <= 0:
        return
    rng = np.random.default_rng(seed)
    extra_delta = np.zeros((extra_count, inst.n_res, inst.n_alt), dtype=float)
    extra_types = np.empty(extra_count, dtype=inst.resource_types.dtype)
    extra_alpha = np.zeros(extra_count, dtype=float)
    for k in range(extra_count):
        src = int(rng.integers(0, inst.n_resources))
        template = inst.delta[src].copy()
        touched = np.where(template.sum(axis=0) > 0.0)[0]
        if touched.size:
            keep = rng.random(inst.n_alt) < rng.uniform(0.5, 0.88)
            if np.any(keep[touched]):
                template[:, ~keep] = 0.0
        scale = float(rng.uniform(0.7, 1.15))
        extra_delta[k] = scale * template
        extra_types[k] = inst.resource_types[src]
        extra_alpha[k] = float(inst.alpha[src] * rng.uniform(0.04, 0.1))
    inst.delta = np.concatenate([inst.delta, extra_delta], axis=0)
    inst.resource_types = np.concatenate([inst.resource_types, extra_types])
    inst.alpha = np.concatenate([inst.alpha, extra_alpha])
    inst.n_resources = int(inst.delta.shape[0])
    h0 = _initial_flow(inst)
    extra_load = np.tensordot(extra_delta, h0, axes=([2], [0]))
    type_floor = np.where(extra_types == "corridor", 0.35, 0.25)[:, None]
    cap_factor = rng.uniform(1.0, 1.02, size=extra_load.shape)
    extra_capacity = np.where(
        extra_load > 1e-08,
        np.maximum(type_floor, extra_load * cap_factor),
        np.maximum(
            type_floor,
            (
                0.75 * inst.capacity[:extra_count]
                if extra_count <= inst.capacity.shape[0]
                else type_floor
            ),
        ),
    )
    inst.capacity = np.concatenate([inst.capacity, extra_capacity], axis=0)
    inst.base_load = np.concatenate([inst.base_load, extra_load], axis=0)
    inst.peak_pressure = float(np.max(inst.base_load / np.maximum(inst.capacity, 1e-09)))
    inst.delayed_intensity = float(
        inst.delta[:, inst.n_dep :, :].sum() / max(float(inst.delta.sum()), 1e-12)
    )


def solve_scalability_instance(
    cfg: StudyConfig, target_alternatives: int, instance_id: int, maxiter: int
) -> dict[str, float | int | str | bool]:
    started = time.perf_counter()
    inst = generate_instance(cfg)
    _inflate_capacity_to_nonbinding(inst)
    x0 = _initial_flow(inst)
    due_h, due_residual, due_iters = _projected_gradient_simplex(
        inst, lambda h: private_cost(inst, h), x0, maxiter
    )
    so_h, _, so_iters = _projected_gradient_simplex(
        inst, lambda h: total_cost_gradient(inst, h), due_h, maxiter
    )
    resource_delay_toll = _resource_tolls(inst, so_h)["resource"]
    priced_h, priced_residual, priced_iters = _projected_gradient_simplex(
        inst, lambda h, tau=resource_delay_toll: private_cost(inst, h) + tau, due_h, maxiter
    )
    elapsed = time.perf_counter() - started
    due_cost = total_cost(inst, due_h)
    so_cost = total_cost(inst, so_h)
    priced_cost = total_cost(inst, priced_h)
    cap_violation = float(
        max(
            _capacity_violation(inst, due_h),
            _capacity_violation(inst, so_h),
            _capacity_violation(inst, priced_h),
        )
    )
    return {
        "Alternative target": int(target_alternatives),
        "Instance": int(instance_id),
        "Alternatives": int(inst.n_alt),
        "Resources": int(inst.n_resources),
        "Resource periods": int(inst.n_res),
        "Bottleneck type": cfg.bottleneck_type,
        "Pressure": float(cfg.pressure),
        "Reliability": cfg.reliability,
        "Solver": "OD-simplex projected gradient",
        "Solve time seconds": float(elapsed),
        "DUE VI residual": float(due_residual),
        "Priced-DUE VI residual": float(priced_residual),
        "Capacity violation": cap_violation,
        "PoA": float(due_cost / max(so_cost, 1e-09)),
        "All-resource delay recovery": float(_recovery(due_cost, priced_cost, so_cost)),
        "DUE iterations": int(due_iters),
        "SO iterations": int(so_iters),
        "Priced-DUE iterations": int(priced_iters),
        "success": True,
        "message": "completed",
    }


def solve_active_capacity_scalability_instance(
    cfg: StudyConfig,
    target_alternatives: int,
    target_resources: int,
    instance_id: int,
    maxiter: int,
) -> dict[str, float | int | str | bool]:
    started = time.perf_counter()
    inst = generate_instance(cfg)
    _augment_resources_for_active_panel(
        inst, target_resources=target_resources, seed=cfg.seed + 37 * (instance_id + 1)
    )
    x0 = _initial_flow(inst)
    due_h, due_residual, due_iters, due_ok, due_msg = _frank_wolfe_hard_capacity(
        inst,
        lambda h: due_potential(inst, h),
        lambda h: due_potential_gradient(inst, h),
        x0,
        maxiter,
    )
    so_h, so_residual, so_iters, so_ok, so_msg = _frank_wolfe_hard_capacity(
        inst, lambda h: total_cost(inst, h), lambda h: total_cost_gradient(inst, h), due_h, maxiter
    )
    resource_delay_toll = _resource_tolls(inst, so_h)["resource"]
    priced_h, priced_residual, priced_iters, priced_ok, priced_msg = _frank_wolfe_hard_capacity(
        inst,
        lambda h, tau=resource_delay_toll: due_potential(inst, h, tau),
        lambda h, tau=resource_delay_toll: due_potential_gradient(inst, h, tau),
        due_h,
        maxiter,
    )
    elapsed = time.perf_counter() - started
    due_cost = total_cost(inst, due_h)
    so_cost = total_cost(inst, so_h)
    priced_cost = total_cost(inst, priced_h)
    cap_violation = float(
        max(
            _capacity_violation(inst, due_h),
            _capacity_violation(inst, so_h),
            _capacity_violation(inst, priced_h),
        )
    )
    completed = bool(
        np.isfinite(due_residual)
        and np.isfinite(so_residual)
        and np.isfinite(priced_residual)
        and (cap_violation <= 1e-06)
    )
    return {
        "Panel type": "active hard-capacity",
        "Alternative target": int(target_alternatives),
        "Resource target": int(target_resources),
        "Instance": int(instance_id),
        "Alternatives": int(inst.n_alt),
        "Resources": int(inst.n_resources),
        "Resource periods": int(inst.n_res),
        "Bottleneck type": cfg.bottleneck_type,
        "Pressure": float(cfg.pressure),
        "Reliability": cfg.reliability,
        "Solver": "constrained Frank-Wolfe",
        "Solve time seconds": float(elapsed),
        "DUE VI residual": float(due_residual),
        "SO VI residual": float(so_residual),
        "Priced-DUE VI residual": float(priced_residual),
        "DUE binding frequency": _occupied_binding_frequency(inst, due_h),
        "SO binding frequency": _occupied_binding_frequency(inst, so_h),
        "Priced binding frequency": _occupied_binding_frequency(inst, priced_h),
        "DUE all-cell binding frequency": _binding_frequency(inst, due_h),
        "SO all-cell binding frequency": _binding_frequency(inst, so_h),
        "Priced all-cell binding frequency": _binding_frequency(inst, priced_h),
        "Capacity violation": cap_violation,
        "PoA": float(due_cost / max(so_cost, 1e-09)),
        "All-resource delay recovery": float(_recovery(due_cost, priced_cost, so_cost)),
        "DUE success": bool(due_ok),
        "SO success": bool(so_ok),
        "Priced-DUE success": bool(priced_ok),
        "DUE iterations": int(due_iters),
        "SO iterations": int(so_iters),
        "Priced-DUE iterations": int(priced_iters),
        "success": completed,
        "message": "; ".join([due_msg, so_msg, priced_msg]),
    }


def _solve_priced_due_hard_capacity(
    inst, toll: np.ndarray, x0: np.ndarray, maxiter: int
) -> tuple[np.ndarray, float, int, bool, str]:
    return _frank_wolfe_hard_capacity(
        inst,
        lambda h, tau=toll: due_potential(inst, h, tau),
        lambda h, tau=toll: due_potential_gradient(inst, h, tau),
        x0,
        maxiter,
    )


def _damped_active_capacity_price_update(
    inst, initial_h: np.ndarray, maxiter: int, adaptive_steps: int, damping: float
) -> tuple[np.ndarray, np.ndarray, float, int, bool, str]:
    toll = np.zeros(inst.n_alt)
    h = initial_h.copy()
    residual = np.nan
    iterations = 0
    ok_all = True
    messages: list[str] = []
    for step in range(1, adaptive_steps + 1):
        estimated = _resource_tolls(inst, h)["resource"]
        step_size = min(float(damping / np.sqrt(float(step))), 1.0)
        toll = (1.0 - step_size) * toll + step_size * estimated
        h, residual, iters, ok, message = _solve_priced_due_hard_capacity(inst, toll, h, maxiter)
        iterations += int(iters)
        ok_all = ok_all and bool(ok)
        messages.append(message)
    return (h, toll, float(residual), iterations, ok_all, "; ".join(messages))


def solve_active_capacity_adaptive_pricing_instance(
    cfg: StudyConfig,
    target_alternatives: int,
    target_resources: int,
    instance_id: int,
    maxiter: int,
    adaptive_steps: int,
    damping: float,
) -> dict[str, float | int | str | bool]:
    started = time.perf_counter()
    inst = generate_instance(cfg)
    _augment_resources_for_active_panel(
        inst, target_resources=target_resources, seed=cfg.seed + 53 * (instance_id + 1)
    )
    x0 = _initial_flow(inst)
    due_h, due_residual, due_iters, due_ok, due_msg = _frank_wolfe_hard_capacity(
        inst,
        lambda h: due_potential(inst, h),
        lambda h: due_potential_gradient(inst, h),
        x0,
        maxiter,
    )
    so_h, so_residual, so_iters, so_ok, so_msg = _frank_wolfe_hard_capacity(
        inst, lambda h: total_cost(inst, h), lambda h: total_cost_gradient(inst, h), due_h, maxiter
    )
    one_shot_toll = _resource_tolls(inst, due_h)["resource"]
    one_h, one_residual, one_iters, one_ok, one_msg = _solve_priced_due_hard_capacity(
        inst, one_shot_toll, due_h, maxiter
    )
    adaptive_h, adaptive_toll, adaptive_residual, adaptive_iters, adaptive_ok, adaptive_msg = (
        _damped_active_capacity_price_update(
            inst, initial_h=due_h, maxiter=maxiter, adaptive_steps=adaptive_steps, damping=damping
        )
    )
    elapsed = time.perf_counter() - started
    due_cost = total_cost(inst, due_h)
    so_cost = total_cost(inst, so_h)
    one_cost = total_cost(inst, one_h)
    adaptive_cost = total_cost(inst, adaptive_h)
    cap_violation = float(
        max(
            _capacity_violation(inst, due_h),
            _capacity_violation(inst, so_h),
            _capacity_violation(inst, one_h),
            _capacity_violation(inst, adaptive_h),
        )
    )
    completed = bool(
        np.isfinite(due_residual)
        and np.isfinite(so_residual)
        and np.isfinite(one_residual)
        and np.isfinite(adaptive_residual)
        and (cap_violation <= 1e-06)
    )
    one_support_gap = _normalize_vi_gap(inst, so_h, private_cost(inst, so_h) + one_shot_toll)
    adaptive_support_gap = _normalize_vi_gap(inst, so_h, private_cost(inst, so_h) + adaptive_toll)
    return {
        "Panel type": "active hard-capacity adaptive pricing",
        "Alternative target": int(target_alternatives),
        "Resource target": int(target_resources),
        "Instance": int(instance_id),
        "Alternatives": int(inst.n_alt),
        "Resources": int(inst.n_resources),
        "Resource periods": int(inst.n_res),
        "Bottleneck type": cfg.bottleneck_type,
        "Pressure": float(cfg.pressure),
        "Reliability": cfg.reliability,
        "Solver": "constrained Frank-Wolfe",
        "Adaptive steps": int(adaptive_steps),
        "Damping": float(damping),
        "Solve time seconds": float(elapsed),
        "DUE VI residual": float(due_residual),
        "SO VI residual": float(so_residual),
        "One-shot priced-DUE VI residual": float(one_residual),
        "Damped adaptive priced-DUE VI residual": float(adaptive_residual),
        "One-shot SO support gap": float(one_support_gap),
        "Damped adaptive SO support gap": float(adaptive_support_gap),
        "Capacity violation": cap_violation,
        "DUE binding frequency": _occupied_binding_frequency(inst, due_h),
        "SO binding frequency": _occupied_binding_frequency(inst, so_h),
        "One-shot binding frequency": _occupied_binding_frequency(inst, one_h),
        "Damped adaptive binding frequency": _occupied_binding_frequency(inst, adaptive_h),
        "PoA": float(due_cost / max(so_cost, 1e-09)),
        "One-shot recovery": float(_recovery(due_cost, one_cost, so_cost)),
        "Damped adaptive recovery": float(_recovery(due_cost, adaptive_cost, so_cost)),
        "DUE iterations": int(due_iters),
        "SO iterations": int(so_iters),
        "One-shot iterations": int(one_iters),
        "Damped adaptive iterations": int(adaptive_iters),
        "DUE success": bool(due_ok),
        "SO success": bool(so_ok),
        "One-shot success": bool(one_ok),
        "Damped adaptive success": bool(adaptive_ok),
        "success": completed,
        "message": "; ".join([due_msg, so_msg, one_msg, adaptive_msg]),
    }


def summarize_scalability(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    ok = results[results["success"] == True].copy()
    if ok.empty:
        ok = results.copy()
    grouped = ok.groupby("Alternative target", as_index=False).agg(
        Instances=("Instance", "count"),
        **{
            "Mean alternatives": ("Alternatives", "mean"),
            "Mean resources": ("Resources", "mean"),
            "Resource periods": ("Resource periods", "mean"),
            "Mean solve time seconds": ("Solve time seconds", "mean"),
            "Mean DUE VI residual": ("DUE VI residual", "mean"),
            "Mean priced-DUE VI residual": ("Priced-DUE VI residual", "mean"),
            "Mean capacity violation": ("Capacity violation", "mean"),
            "Mean PoA": ("PoA", "mean"),
            "Mean all-resource delay recovery": ("All-resource delay recovery", "mean"),
        },
    )
    return grouped


def summarize_active_capacity_scalability(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    ok = results[results["success"] == True].copy()
    if ok.empty:
        ok = results.copy()
    return ok.groupby("Panel type", as_index=False).agg(
        Instances=("Instance", "count"),
        **{
            "Mean alternatives": ("Alternatives", "mean"),
            "Mean resources": ("Resources", "mean"),
            "Resource periods": ("Resource periods", "mean"),
            "Mean solve time seconds": ("Solve time seconds", "mean"),
            "Mean DUE VI residual": ("DUE VI residual", "mean"),
            "Mean priced-DUE VI residual": ("Priced-DUE VI residual", "mean"),
            "Mean capacity violation": ("Capacity violation", "mean"),
            "Mean DUE binding frequency": ("DUE binding frequency", "mean"),
            "Mean SO binding frequency": ("SO binding frequency", "mean"),
            "Mean priced binding frequency": ("Priced binding frequency", "mean"),
            "Mean PoA": ("PoA", "mean"),
            "Mean all-resource delay recovery": ("All-resource delay recovery", "mean"),
        },
    )


def summarize_active_capacity_multiscale(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    ok = results[results["success"] == True].copy()
    if ok.empty:
        ok = results.copy()
    return ok.groupby("Alternative target", as_index=False).agg(
        Instances=("Instance", "count"),
        **{
            "Mean alternatives": ("Alternatives", "mean"),
            "Mean resources": ("Resources", "mean"),
            "Resource periods": ("Resource periods", "mean"),
            "Mean solve time seconds": ("Solve time seconds", "mean"),
            "Mean DUE VI residual": ("DUE VI residual", "mean"),
            "Mean SO VI residual": ("SO VI residual", "mean"),
            "Mean priced-DUE VI residual": ("Priced-DUE VI residual", "mean"),
            "Mean capacity violation": ("Capacity violation", "mean"),
            "Mean DUE binding frequency": ("DUE binding frequency", "mean"),
            "Mean SO binding frequency": ("SO binding frequency", "mean"),
            "Mean priced binding frequency": ("Priced binding frequency", "mean"),
            "Mean PoA": ("PoA", "mean"),
            "Mean all-resource delay recovery": ("All-resource delay recovery", "mean"),
        },
    )


def summarize_active_capacity_adaptive_pricing(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    ok = results[results["success"] == True].copy()
    if ok.empty:
        ok = results.copy()
    return ok.groupby("Panel type", as_index=False).agg(
        Instances=("Instance", "count"),
        **{
            "Mean alternatives": ("Alternatives", "mean"),
            "Mean resources": ("Resources", "mean"),
            "Resource periods": ("Resource periods", "mean"),
            "Mean solve time seconds": ("Solve time seconds", "mean"),
            "Mean DUE VI residual": ("DUE VI residual", "mean"),
            "Mean SO VI residual": ("SO VI residual", "mean"),
            "Mean one-shot priced-DUE VI residual": ("One-shot priced-DUE VI residual", "mean"),
            "Mean damped adaptive priced-DUE VI residual": (
                "Damped adaptive priced-DUE VI residual",
                "mean",
            ),
            "Mean one-shot SO support gap": ("One-shot SO support gap", "mean"),
            "Mean damped adaptive SO support gap": ("Damped adaptive SO support gap", "mean"),
            "Mean capacity violation": ("Capacity violation", "mean"),
            "Mean DUE binding frequency": ("DUE binding frequency", "mean"),
            "Mean SO binding frequency": ("SO binding frequency", "mean"),
            "Mean one-shot binding frequency": ("One-shot binding frequency", "mean"),
            "Mean damped adaptive binding frequency": ("Damped adaptive binding frequency", "mean"),
            "Mean PoA": ("PoA", "mean"),
            "Mean one-shot recovery": ("One-shot recovery", "mean"),
            "Mean damped adaptive recovery": ("Damped adaptive recovery", "mean"),
        },
    )


def run_scalability_panel(
    out_dir: Path = SCALABILITY_OUT,
    scales: tuple[int, ...] = (500, 1000, 2000),
    instances_per_scale: int = 10,
    seed: int = 2026062604,
    maxiter: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    for target in scales:
        for instance_id in range(instances_per_scale):
            cfg = _scale_config(target, instance_id, seed)
            try:
                rows.append(
                    solve_scalability_instance(
                        cfg=cfg,
                        target_alternatives=target,
                        instance_id=instance_id,
                        maxiter=maxiter,
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "Alternative target": int(target),
                        "Instance": int(instance_id),
                        "Alternatives": 0,
                        "Resources": 0,
                        "Resource periods": 0,
                        "Bottleneck type": cfg.bottleneck_type,
                        "Pressure": float(cfg.pressure),
                        "Reliability": cfg.reliability,
                        "Solve time seconds": np.nan,
                        "DUE VI residual": np.nan,
                        "Priced-DUE VI residual": np.nan,
                        "Capacity violation": np.nan,
                        "PoA": np.nan,
                        "All-resource delay recovery": np.nan,
                        "success": False,
                        "message": str(exc),
                    }
                )
    results = pd.DataFrame(rows)
    summary = summarize_scalability(results)
    results.to_csv(out_dir / "scalability_panel.csv", index=False)
    summary.to_csv(out_dir / "scalability_summary.csv", index=False)
    return (results, summary)


def run_active_capacity_adaptive_pricing_panel(
    out_dir: Path = SCALABILITY_OUT,
    target_alternatives: int = 500,
    target_resources: int = 80,
    instances: int = 5,
    seed: int = 2026062609,
    maxiter: int = 100,
    adaptive_steps: int = 5,
    damping: float = 0.65,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    for instance_id in range(instances):
        cfg = _scale_config(target_alternatives, instance_id, seed)
        try:
            rows.append(
                solve_active_capacity_adaptive_pricing_instance(
                    cfg=cfg,
                    target_alternatives=target_alternatives,
                    target_resources=target_resources,
                    instance_id=instance_id,
                    maxiter=maxiter,
                    adaptive_steps=adaptive_steps,
                    damping=damping,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "Panel type": "active hard-capacity adaptive pricing",
                    "Alternative target": int(target_alternatives),
                    "Resource target": int(target_resources),
                    "Instance": int(instance_id),
                    "Alternatives": 0,
                    "Resources": 0,
                    "Resource periods": 0,
                    "Bottleneck type": cfg.bottleneck_type,
                    "Pressure": float(cfg.pressure),
                    "Reliability": cfg.reliability,
                    "Solve time seconds": np.nan,
                    "DUE VI residual": np.nan,
                    "SO VI residual": np.nan,
                    "One-shot priced-DUE VI residual": np.nan,
                    "Damped adaptive priced-DUE VI residual": np.nan,
                    "One-shot SO support gap": np.nan,
                    "Damped adaptive SO support gap": np.nan,
                    "Capacity violation": np.nan,
                    "PoA": np.nan,
                    "One-shot recovery": np.nan,
                    "Damped adaptive recovery": np.nan,
                    "success": False,
                    "message": str(exc),
                }
            )
    results = pd.DataFrame(rows)
    summary = summarize_active_capacity_adaptive_pricing(results)
    results.to_csv(out_dir / "active_capacity_adaptive_pricing_panel.csv", index=False)
    summary.to_csv(out_dir / "active_capacity_adaptive_pricing_summary.csv", index=False)
    return (results, summary)


def run_active_capacity_scalability_panel(
    out_dir: Path = SCALABILITY_OUT,
    target_alternatives: int = 500,
    target_resources: int = 120,
    instances: int = 10,
    seed: int = 2026062607,
    maxiter: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    for instance_id in range(instances):
        cfg = _scale_config(target_alternatives, instance_id, seed)
        try:
            rows.append(
                solve_active_capacity_scalability_instance(
                    cfg=cfg,
                    target_alternatives=target_alternatives,
                    target_resources=target_resources,
                    instance_id=instance_id,
                    maxiter=maxiter,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "Panel type": "active hard-capacity",
                    "Alternative target": int(target_alternatives),
                    "Resource target": int(target_resources),
                    "Instance": int(instance_id),
                    "Alternatives": 0,
                    "Resources": 0,
                    "Resource periods": 0,
                    "Bottleneck type": cfg.bottleneck_type,
                    "Pressure": float(cfg.pressure),
                    "Reliability": cfg.reliability,
                    "Solve time seconds": np.nan,
                    "DUE VI residual": np.nan,
                    "Priced-DUE VI residual": np.nan,
                    "DUE binding frequency": np.nan,
                    "SO binding frequency": np.nan,
                    "Priced binding frequency": np.nan,
                    "Capacity violation": np.nan,
                    "PoA": np.nan,
                    "All-resource delay recovery": np.nan,
                    "success": False,
                    "message": str(exc),
                }
            )
    results = pd.DataFrame(rows)
    summary = summarize_active_capacity_scalability(results)
    results.to_csv(out_dir / "active_capacity_scalability_panel.csv", index=False)
    summary.to_csv(out_dir / "active_capacity_scalability_summary.csv", index=False)
    return (results, summary)


def run_active_capacity_scalability_multiscale_panel(
    out_dir: Path = SCALABILITY_OUT,
    scales: tuple[int, ...] = (500, 1000, 2000),
    resource_targets: dict[int, int] | None = None,
    instances_per_scale: int = 5,
    seed: int = 2026062608,
    maxiter: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if resource_targets is None:
        resource_targets = {500: 80, 1000: 120, 2000: 150}
    rows: list[dict[str, float | int | str | bool]] = []
    for target in scales:
        target_resources = int(resource_targets.get(int(target), 120))
        for instance_id in range(instances_per_scale):
            cfg = _scale_config(target, instance_id, seed)
            try:
                rows.append(
                    solve_active_capacity_scalability_instance(
                        cfg=cfg,
                        target_alternatives=target,
                        target_resources=target_resources,
                        instance_id=instance_id,
                        maxiter=maxiter,
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "Panel type": "active hard-capacity",
                        "Alternative target": int(target),
                        "Resource target": int(target_resources),
                        "Instance": int(instance_id),
                        "Alternatives": 0,
                        "Resources": 0,
                        "Resource periods": 0,
                        "Bottleneck type": cfg.bottleneck_type,
                        "Pressure": float(cfg.pressure),
                        "Reliability": cfg.reliability,
                        "Solve time seconds": np.nan,
                        "DUE VI residual": np.nan,
                        "SO VI residual": np.nan,
                        "Priced-DUE VI residual": np.nan,
                        "DUE binding frequency": np.nan,
                        "SO binding frequency": np.nan,
                        "Priced binding frequency": np.nan,
                        "Capacity violation": np.nan,
                        "PoA": np.nan,
                        "All-resource delay recovery": np.nan,
                        "success": False,
                        "message": str(exc),
                    }
                )
    results = pd.DataFrame(rows)
    summary = summarize_active_capacity_multiscale(results)
    results.to_csv(out_dir / "active_capacity_scalability_multiscale_panel.csv", index=False)
    summary.to_csv(out_dir / "active_capacity_scalability_multiscale_summary.csv", index=False)
    return (results, summary)
