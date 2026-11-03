from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .controlled_networks import (
        BOTTLENECK_TYPES,
        StudyConfig,
        _capacity_violation,
        _initial_flow,
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
        StudyConfig,
        _capacity_violation,
        _initial_flow,
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
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/adaptive_pricing"


def _recovery(due_cost: float, so_cost: float, priced_cost: float) -> float:
    return (due_cost - priced_cost) / max(due_cost - so_cost, 1e-09)


def _record_state(
    rows: list[dict[str, float | int | str | bool]],
    *,
    instance_id: int,
    cfg: StudyConfig,
    method: str,
    toll_component: str,
    iteration: int,
    inst,
    h: np.ndarray,
    toll: np.ndarray,
    previous_toll: np.ndarray,
    due_cost: float,
    so_cost: float,
    so_h: np.ndarray,
    ok: bool,
    accepted: bool = True,
) -> None:
    priced_cost = total_cost(inst, h)
    delta_toll = toll - previous_toll
    rows.append(
        {
            "Instance": instance_id,
            "Method": method,
            "Toll component": toll_component,
            "Iteration": iteration,
            "Accepted": bool(accepted),
            "Solver success": bool(ok),
            "Bottleneck type": cfg.bottleneck_type,
            "Pressure target": cfg.pressure,
            "Reliability": cfg.reliability,
            "Seed": cfg.seed,
            "Alternatives": inst.n_alt,
            "Resources": inst.n_resources,
            "DUE cost": due_cost,
            "SO cost": so_cost,
            "Priced cost": priced_cost,
            "Recovery": _recovery(due_cost, so_cost, priced_cost),
            "SO support gap": _vi_gap(inst, so_h, private_cost(inst, so_h) + toll),
            "Priced-DUE VI residual": _vi_gap(inst, h, private_cost(inst, h) + toll),
            "Relative toll-change norm": float(
                np.linalg.norm(delta_toll) / (1.0 + np.linalg.norm(previous_toll))
            ),
            "Absolute toll-change norm": float(np.linalg.norm(delta_toll)),
            "Toll norm": float(np.linalg.norm(toll)),
            "Maximum capacity violation": _capacity_violation(inst, h),
        }
    )


def _solve_core(inst, maxiter: int) -> tuple[np.ndarray, np.ndarray, bool, bool]:
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
    return (due_h, so_h, due_ok, so_ok)


def _plain_damped_trace(
    rows: list[dict[str, float | int | str | bool]],
    *,
    instance_id: int,
    cfg: StudyConfig,
    inst,
    initial_h: np.ndarray,
    so_h: np.ndarray,
    due_cost: float,
    so_cost: float,
    maxiter: int,
    update_steps: int,
    damping: float,
    capped: bool,
    method: str,
    toll_component: str,
) -> None:
    toll = np.zeros(inst.n_alt)
    h = initial_h.copy()
    upper = np.inf
    if capped:
        base_estimate = _resource_tolls(inst, initial_h)[toll_component]
        positive_base = base_estimate[base_estimate > 0.0]
        upper = (
            max(float(np.quantile(positive_base, 0.85)) * 0.9, 1e-09)
            if positive_base.size
            else 1e-09
        )
    _record_state(
        rows,
        instance_id=instance_id,
        cfg=cfg,
        method=method,
        toll_component=toll_component,
        iteration=0,
        inst=inst,
        h=h,
        toll=toll,
        previous_toll=toll,
        due_cost=due_cost,
        so_cost=so_cost,
        so_h=so_h,
        ok=True,
    )
    for step in range(1, update_steps + 1):
        previous_toll = toll.copy()
        estimated = _resource_tolls(inst, h)[toll_component]
        step_size = min(damping, float(damping / np.sqrt(step)))
        toll = (1.0 - step_size) * toll + step_size * estimated
        if capped:
            toll = np.clip(toll, 0.0, upper)
        h, ok = _solve_priced_due(inst, toll, h, maxiter)
        _record_state(
            rows,
            instance_id=instance_id,
            cfg=cfg,
            method=method,
            toll_component=toll_component,
            iteration=step,
            inst=inst,
            h=h,
            toll=toll,
            previous_toll=previous_toll,
            due_cost=due_cost,
            so_cost=so_cost,
            so_h=so_h,
            ok=ok,
        )


def _accepted_adaptive_trace(
    rows: list[dict[str, float | int | str | bool]],
    *,
    instance_id: int,
    cfg: StudyConfig,
    inst,
    initial_h: np.ndarray,
    so_h: np.ndarray,
    due_cost: float,
    so_cost: float,
    maxiter: int,
    update_steps: int,
    damping: float,
    toll_component: str,
) -> None:
    best_toll = np.zeros(inst.n_alt)
    best_h = initial_h.copy()
    best_cost = total_cost(inst, best_h)
    base_estimate = _resource_tolls(inst, initial_h)[toll_component]
    positive_base = base_estimate[base_estimate > 0.0]
    upper = (
        max(float(np.quantile(positive_base, 0.9)) * 1.2, 1e-09) if positive_base.size else 1e-09
    )
    base_steps = np.array([0.9, 0.65, 0.45, 0.3, 0.2, 0.12, 0.07, 0.035])
    _record_state(
        rows,
        instance_id=instance_id,
        cfg=cfg,
        method="accepted adaptive",
        toll_component=toll_component,
        iteration=0,
        inst=inst,
        h=best_h,
        toll=best_toll,
        previous_toll=best_toll,
        due_cost=due_cost,
        so_cost=so_cost,
        so_h=so_h,
        ok=True,
    )
    for step in range(1, update_steps + 1):
        previous_toll = best_toll.copy()
        estimated = _resource_tolls(inst, best_h)[toll_component]
        accepted: tuple[np.ndarray, np.ndarray, float, bool] | None = None
        trial_best = best_cost
        for base_step in base_steps:
            step_size = min(damping, float(base_step / np.sqrt(step)))
            candidate_toll = (1.0 - step_size) * best_toll + step_size * estimated
            candidate_toll = np.clip(candidate_toll, 0.0, upper)
            candidate_h, ok = _solve_priced_due(inst, candidate_toll, best_h, maxiter)
            candidate_cost = total_cost(inst, candidate_h)
            if candidate_cost < trial_best - 1e-07:
                trial_best = candidate_cost
                accepted = (candidate_h, candidate_toll, candidate_cost, ok)
        if accepted is None:
            _record_state(
                rows,
                instance_id=instance_id,
                cfg=cfg,
                method="accepted adaptive",
                toll_component=toll_component,
                iteration=step,
                inst=inst,
                h=best_h,
                toll=best_toll,
                previous_toll=previous_toll,
                due_cost=due_cost,
                so_cost=so_cost,
                so_h=so_h,
                ok=True,
                accepted=False,
            )
            break
        best_h, best_toll, best_cost, ok = accepted
        _record_state(
            rows,
            instance_id=instance_id,
            cfg=cfg,
            method="accepted adaptive",
            toll_component=toll_component,
            iteration=step,
            inst=inst,
            h=best_h,
            toll=best_toll,
            previous_toll=previous_toll,
            due_cost=due_cost,
            so_cost=so_cost,
            so_h=so_h,
            ok=ok,
            accepted=True,
        )


def _one_shot_trace(
    rows: list[dict[str, float | int | str | bool]],
    *,
    instance_id: int,
    cfg: StudyConfig,
    inst,
    initial_h: np.ndarray,
    so_h: np.ndarray,
    due_cost: float,
    so_cost: float,
    maxiter: int,
    toll_component: str,
) -> None:
    zero = np.zeros(inst.n_alt)
    toll = _resource_tolls(inst, initial_h)[toll_component]
    h, ok = _solve_priced_due(inst, toll, initial_h, maxiter)
    _record_state(
        rows,
        instance_id=instance_id,
        cfg=cfg,
        method="one-shot DUE-estimated",
        toll_component=toll_component,
        iteration=1,
        inst=inst,
        h=h,
        toll=toll,
        previous_toll=zero,
        due_cost=due_cost,
        so_cost=so_cost,
        so_h=so_h,
        ok=ok,
    )


def _configs(seed: int) -> list[StudyConfig]:
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
    return configs


def run_adaptive_pricing_trajectories(
    out_dir: Path = OUT_DIR, seed: int = 20260626, maxiter: int = 160, update_steps: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    for instance_id, cfg in enumerate(_configs(seed), start=1):
        inst = generate_instance(cfg)
        due_h, so_h, due_ok, so_ok = _solve_core(inst, maxiter)
        due_cost = total_cost(inst, due_h)
        so_cost = total_cost(inst, so_h)
        if not (due_ok and so_ok):
            print(
                {
                    "instance": instance_id,
                    "due_ok": due_ok,
                    "so_ok": so_ok,
                    "message": "core solve not fully successful",
                }
            )
        for toll_component in ["resource", "delay_plus_reliability"]:
            _one_shot_trace(
                rows,
                instance_id=instance_id,
                cfg=cfg,
                inst=inst,
                initial_h=due_h,
                so_h=so_h,
                due_cost=due_cost,
                so_cost=so_cost,
                maxiter=maxiter,
                toll_component=toll_component,
            )
            _accepted_adaptive_trace(
                rows,
                instance_id=instance_id,
                cfg=cfg,
                inst=inst,
                initial_h=due_h,
                so_h=so_h,
                due_cost=due_cost,
                so_cost=so_cost,
                maxiter=maxiter,
                update_steps=update_steps,
                damping=0.65,
                toll_component=toll_component,
            )
        _plain_damped_trace(
            rows,
            instance_id=instance_id,
            cfg=cfg,
            inst=inst,
            initial_h=due_h,
            so_h=so_h,
            due_cost=due_cost,
            so_cost=so_cost,
            maxiter=maxiter,
            update_steps=update_steps,
            damping=0.16,
            capped=False,
            method="plain damped",
            toll_component="resource",
        )
        _plain_damped_trace(
            rows,
            instance_id=instance_id,
            cfg=cfg,
            inst=inst,
            initial_h=due_h,
            so_h=so_h,
            due_cost=due_cost,
            so_cost=so_cost,
            maxiter=maxiter,
            update_steps=update_steps,
            damping=0.24,
            capped=True,
            method="capped damped",
            toll_component="resource",
        )
        print({"instance": instance_id, "rows": len(rows)})
    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(out_dir / "adaptive_pricing_trajectories.csv", index=False)
    summary = (
        trajectory.groupby(["Method", "Toll component", "Iteration"], dropna=False)
        .agg(
            instances=("Instance", "nunique"),
            accepted_rate=("Accepted", "mean"),
            solver_success_rate=("Solver success", "mean"),
            recovery_mean=("Recovery", "mean"),
            recovery_median=("Recovery", "median"),
            support_gap_mean=("SO support gap", "mean"),
            support_gap_median=("SO support gap", "median"),
            vi_residual_mean=("Priced-DUE VI residual", "mean"),
            toll_change_mean=("Relative toll-change norm", "mean"),
            toll_change_median=("Relative toll-change norm", "median"),
            capacity_violation_max=("Maximum capacity violation", "max"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "adaptive_pricing_trajectory_summary.csv", index=False)
    plot_ready = _make_plot_ready_trajectory(trajectory, update_steps)
    plot_ready.to_csv(out_dir / "adaptive_pricing_trajectory_plot_ready.csv", index=False)
    plot_summary = (
        plot_ready.groupby(["Method", "Toll component", "Iteration"], dropna=False)
        .agg(
            instances=("Instance", "nunique"),
            carried_forward_rate=("Carried forward", "mean"),
            solver_success_rate=("Solver success", "mean"),
            recovery_mean=("Recovery", "mean"),
            recovery_median=("Recovery", "median"),
            support_gap_mean=("SO support gap", "mean"),
            support_gap_median=("SO support gap", "median"),
            vi_residual_mean=("Priced-DUE VI residual", "mean"),
            toll_change_mean=("Relative toll-change norm", "mean"),
            toll_change_median=("Relative toll-change norm", "median"),
            capacity_violation_max=("Maximum capacity violation", "max"),
        )
        .reset_index()
    )
    plot_summary.to_csv(out_dir / "adaptive_pricing_trajectory_plot_summary.csv", index=False)
    return (trajectory, summary)


def _make_plot_ready_trajectory(trajectory: pd.DataFrame, update_steps: int) -> pd.DataFrame:
    filled: list[pd.DataFrame] = []
    key_cols = ["Instance", "Method", "Toll component"]
    for _, part in trajectory.groupby(key_cols, sort=False):
        part = part.sort_values("Iteration").copy()
        method = str(part["Method"].iloc[0])
        if method == "one-shot DUE-estimated":
            iter_range = range(1, update_steps + 1)
        else:
            iter_range = range(0, update_steps + 1)
        last = part.iloc[0].copy()
        rows: list[pd.Series] = []
        by_iter = {int(r["Iteration"]): r for _, r in part.iterrows()}
        for iteration in iter_range:
            if iteration in by_iter:
                last = by_iter[iteration].copy()
                last["Carried forward"] = False
            else:
                last = last.copy()
                last["Iteration"] = iteration
                last["Accepted"] = False
                last["Relative toll-change norm"] = 0.0
                last["Absolute toll-change norm"] = 0.0
                last["Carried forward"] = True
            rows.append(last.copy())
        filled.append(pd.DataFrame(rows))
    out = pd.concat(filled, ignore_index=True)
    if "Carried forward" not in out.columns:
        out["Carried forward"] = False
    out["Carried forward"] = out["Carried forward"].fillna(False).astype(bool)
    return out
