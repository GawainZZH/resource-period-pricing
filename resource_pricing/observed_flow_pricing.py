from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .controlled_networks import (
        BOTTLENECK_TYPES,
        OUT,
        StudyConfig,
        _capacity_violation,
        _initial_flow,
        _iterative_estimated_toll,
        _minimize_flow,
        _resource_tolls,
        _solve_priced_due,
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
        BOTTLENECK_TYPES,
        OUT,
        StudyConfig,
        _capacity_violation,
        _initial_flow,
        _iterative_estimated_toll,
        _minimize_flow,
        _resource_tolls,
        _solve_priced_due,
        _vi_gap,
        due_potential,
        due_potential_gradient,
        generate_instance,
        private_cost,
        total_cost,
        total_cost_gradient,
    )


def _recovery(due_cost: float, so_cost: float, priced_cost: float) -> float:
    return (due_cost - priced_cost) / max(due_cost - so_cost, 1e-09)


def _damped_estimated_toll(
    inst,
    initial_h,
    toll_key: str,
    maxiter: int,
    update_steps: int = 8,
    damping: float = 0.22,
    capped: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool, int, float]:
    toll = np.zeros(inst.n_alt)
    h = initial_h.copy()
    ok_all = True
    relative_change = 0.0
    upper = np.inf
    if capped:
        base_estimate = _resource_tolls(inst, initial_h)[toll_key]
        positive_base = base_estimate[base_estimate > 0.0]
        upper = (
            max(float(np.quantile(positive_base, 0.85)) * 0.9, 1e-09)
            if positive_base.size
            else 1e-09
        )
    for step in range(1, update_steps + 1):
        estimated = _resource_tolls(inst, h)[toll_key]
        step_size = min(damping, float(damping / np.sqrt(step)))
        new_toll = (1.0 - step_size) * toll + step_size * estimated
        if capped:
            new_toll = np.clip(new_toll, 0.0, upper)
        h_next, ok = _solve_priced_due(inst, new_toll, h, maxiter)
        ok_all = ok_all and ok
        relative_change = float(np.linalg.norm(new_toll - toll) / (1.0 + np.linalg.norm(toll)))
        toll, h = (new_toll, h_next)
    return (h, toll, ok_all, update_steps, relative_change)


def solve_estimated_toll_instance(
    cfg: StudyConfig, maxiter: int = 160
) -> dict[str, float | int | str | bool]:
    inst = generate_instance(cfg)
    x0 = _initial_flow(inst)
    due_h, due_ok, _ = _minimize_flow(
        inst,
        lambda x: due_potential(inst, x),
        lambda x: due_potential_gradient(inst, x),
        x0,
        maxiter,
    )
    so_h, so_ok, _ = _minimize_flow(
        inst, lambda x: total_cost(inst, x), lambda x: total_cost_gradient(inst, x), due_h, maxiter
    )
    so_tolls = _resource_tolls(inst, so_h)
    due_tolls = _resource_tolls(inst, due_h)
    full_mc_toll = total_cost_gradient(inst, so_h) - private_cost(inst, so_h)
    so_delay_rel_h, so_delay_rel_ok = _solve_priced_due(
        inst, so_tolls["delay_plus_reliability"], due_h, maxiter
    )
    full_mc_h, full_mc_ok = _solve_priced_due(inst, full_mc_toll, due_h, maxiter)
    due_est_resource_h, due_est_resource_ok = _solve_priced_due(
        inst, due_tolls["resource"], due_h, maxiter
    )
    due_est_delay_rel_h, due_est_delay_rel_ok = _solve_priced_due(
        inst, due_tolls["delay_plus_reliability"], due_h, maxiter
    )
    (
        plain_resource_h,
        plain_resource_toll,
        plain_resource_ok,
        plain_resource_steps,
        plain_resource_change,
    ) = _damped_estimated_toll(
        inst, due_h, "resource", maxiter, update_steps=8, damping=0.16, capped=False
    )
    (
        capped_resource_h,
        capped_resource_toll,
        capped_resource_ok,
        capped_resource_steps,
        capped_resource_change,
    ) = _damped_estimated_toll(
        inst, due_h, "resource", maxiter, update_steps=8, damping=0.24, capped=True
    )
    (
        iter_resource_h,
        iter_resource_toll,
        iter_resource_ok,
        iter_resource_steps,
        iter_resource_change,
    ) = _iterative_estimated_toll(inst, due_h, "resource", maxiter, update_steps=8)
    (
        iter_delay_rel_h,
        iter_delay_rel_toll,
        iter_delay_rel_ok,
        iter_delay_rel_steps,
        iter_delay_rel_change,
    ) = _iterative_estimated_toll(inst, due_h, "delay_plus_reliability", maxiter, update_steps=8)
    due_cost = total_cost(inst, due_h)
    so_cost = total_cost(inst, so_h)
    so_delay_rel_cost = total_cost(inst, so_delay_rel_h)
    full_mc_cost = total_cost(inst, full_mc_h)
    due_est_resource_cost = total_cost(inst, due_est_resource_h)
    due_est_delay_rel_cost = total_cost(inst, due_est_delay_rel_h)
    plain_resource_cost = total_cost(inst, plain_resource_h)
    capped_resource_cost = total_cost(inst, capped_resource_h)
    iter_resource_cost = total_cost(inst, iter_resource_h)
    iter_delay_rel_cost = total_cost(inst, iter_delay_rel_h)
    return {
        "success": bool(
            due_ok
            and so_ok
            and so_delay_rel_ok
            and full_mc_ok
            and due_est_resource_ok
            and due_est_delay_rel_ok
            and plain_resource_ok
            and capped_resource_ok
            and iter_resource_ok
            and iter_delay_rel_ok
        ),
        "Bottleneck type": cfg.bottleneck_type,
        "Pressure target": cfg.pressure,
        "Reliability": cfg.reliability,
        "Seed": cfg.seed,
        "Alternatives": inst.n_alt,
        "PoA": due_cost / max(so_cost, 1e-09),
        "SO-based delay+reliability recovery": _recovery(due_cost, so_cost, so_delay_rel_cost),
        "SO-based MC recovery": _recovery(due_cost, so_cost, full_mc_cost),
        "DUE-estimated resource recovery": _recovery(due_cost, so_cost, due_est_resource_cost),
        "DUE-estimated delay+reliability recovery": _recovery(
            due_cost, so_cost, due_est_delay_rel_cost
        ),
        "Plain damped resource recovery": _recovery(due_cost, so_cost, plain_resource_cost),
        "Capped damped resource recovery": _recovery(due_cost, so_cost, capped_resource_cost),
        "Adaptive resource recovery": _recovery(due_cost, so_cost, iter_resource_cost),
        "Adaptive delay+reliability recovery": _recovery(due_cost, so_cost, iter_delay_rel_cost),
        "Accepted adaptive resource recovery": _recovery(due_cost, so_cost, iter_resource_cost),
        "Iterative resource recovery": _recovery(due_cost, so_cost, iter_resource_cost),
        "Iterative delay+reliability recovery": _recovery(due_cost, so_cost, iter_delay_rel_cost),
        "DUE-estimated resource SO support gap": _vi_gap(
            inst, so_h, private_cost(inst, so_h) + due_tolls["resource"]
        ),
        "DUE-estimated delay+reliability SO support gap": _vi_gap(
            inst, so_h, private_cost(inst, so_h) + due_tolls["delay_plus_reliability"]
        ),
        "Plain damped resource SO support gap": _vi_gap(
            inst, so_h, private_cost(inst, so_h) + plain_resource_toll
        ),
        "Capped damped resource SO support gap": _vi_gap(
            inst, so_h, private_cost(inst, so_h) + capped_resource_toll
        ),
        "Iterative resource SO support gap": _vi_gap(
            inst, so_h, private_cost(inst, so_h) + iter_resource_toll
        ),
        "Iterative delay+reliability SO support gap": _vi_gap(
            inst, so_h, private_cost(inst, so_h) + iter_delay_rel_toll
        ),
        "Adaptive resource SO support gap": _vi_gap(
            inst, so_h, private_cost(inst, so_h) + iter_resource_toll
        ),
        "Adaptive delay+reliability SO support gap": _vi_gap(
            inst, so_h, private_cost(inst, so_h) + iter_delay_rel_toll
        ),
        "SO-based MC support gap": _vi_gap(inst, so_h, private_cost(inst, so_h) + full_mc_toll),
        "DUE-estimated resource VI residual": _vi_gap(
            inst, due_est_resource_h, private_cost(inst, due_est_resource_h) + due_tolls["resource"]
        ),
        "Plain damped resource VI residual": _vi_gap(
            inst, plain_resource_h, private_cost(inst, plain_resource_h) + plain_resource_toll
        ),
        "Capped damped resource VI residual": _vi_gap(
            inst, capped_resource_h, private_cost(inst, capped_resource_h) + capped_resource_toll
        ),
        "Iterative resource VI residual": _vi_gap(
            inst, iter_resource_h, private_cost(inst, iter_resource_h) + iter_resource_toll
        ),
        "Adaptive resource VI residual": _vi_gap(
            inst, iter_resource_h, private_cost(inst, iter_resource_h) + iter_resource_toll
        ),
        "Adaptive resource iterations": iter_resource_steps,
        "Adaptive delay+reliability iterations": iter_delay_rel_steps,
        "Adaptive resource toll change": iter_resource_change,
        "Adaptive delay+reliability toll change": iter_delay_rel_change,
        "Plain damped resource iterations": plain_resource_steps,
        "Capped damped resource iterations": capped_resource_steps,
        "Plain damped resource toll change": plain_resource_change,
        "Capped damped resource toll change": capped_resource_change,
        "Iterative resource iterations": iter_resource_steps,
        "Iterative delay+reliability iterations": iter_delay_rel_steps,
        "Iterative resource toll change": iter_resource_change,
        "Iterative delay+reliability toll change": iter_delay_rel_change,
        "Maximum capacity violation": max(
            _capacity_violation(inst, due_h),
            _capacity_violation(inst, so_h),
            _capacity_violation(inst, so_delay_rel_h),
            _capacity_violation(inst, full_mc_h),
            _capacity_violation(inst, due_est_resource_h),
            _capacity_violation(inst, due_est_delay_rel_h),
            _capacity_violation(inst, plain_resource_h),
            _capacity_violation(inst, capped_resource_h),
            _capacity_violation(inst, iter_resource_h),
            _capacity_violation(inst, iter_delay_rel_h),
        ),
    }


def run_estimated_toll_panel(
    out_dir: Path = OUT, seed: int = 20260626, maxiter: int = 160
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs: list[StudyConfig] = []
    i = 0
    for bottleneck in BOTTLENECK_TYPES:
        for pressure in [0.75, 0.95]:
            for reliability in ["loose", "medium", "tight"]:
                configs.append(
                    StudyConfig(
                        n_od=3,
                        paths_per_od=3,
                        n_dep=12,
                        bottleneck_type=bottleneck,
                        pressure=pressure,
                        reliability=reliability,
                        window_width=4,
                        seed=seed * 1000 + i,
                    )
                )
                i += 1
    rows = [solve_estimated_toll_instance(cfg, maxiter=maxiter) for cfg in configs]
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "estimated_toll_panel.csv", index=False)
    ok = results[results["success"] == True].copy()
    summary = (
        ok.drop(columns=["success", "Seed"], errors="ignore")
        .select_dtypes(include=[np.number])
        .agg(["mean", "max"])
    )
    summary.to_csv(out_dir / "estimated_toll_panel_summary.csv")
    return results
