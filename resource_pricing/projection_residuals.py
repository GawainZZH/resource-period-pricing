from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .controlled_networks import (
        BOTTLENECK_TYPES,
        OUT,
        StudyConfig,
        _projection_residuals,
        _resource_tolls,
        _solve_priced_due,
        _vi_gap,
        generate_instance,
        private_cost,
        solve_core_equilibria,
        total_cost,
        total_cost_gradient,
    )
except ImportError:
    from resource_pricing.controlled_networks import (
        BOTTLENECK_TYPES,
        OUT,
        StudyConfig,
        _projection_residuals,
        _resource_tolls,
        _solve_priced_due,
        _vi_gap,
        generate_instance,
        private_cost,
        solve_core_equilibria,
        total_cost,
        total_cost_gradient,
    )


def _recovery(due_cost: float, so_cost: float, priced_cost: float) -> float:
    return (due_cost - priced_cost) / max(due_cost - so_cost, 1e-09)


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return 0.0
    if pair.iloc[:, 0].std() <= 1e-12 or pair.iloc[:, 1].std() <= 1e-12:
        return 0.0
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def solve_projection_instance(
    cfg: StudyConfig, maxiter: int = 150
) -> dict[str, float | int | str | bool]:
    inst = generate_instance(cfg)
    due_h, so_h = solve_core_equilibria(inst, maxiter=maxiter)
    tolls = _resource_tolls(inst, so_h)
    marginal_toll = total_cost_gradient(inst, so_h) - private_cost(inst, so_h)
    residuals = _projection_residuals(inst, so_h, marginal_toll)
    corridor_h, corridor_ok = _solve_priced_due(inst, tolls["corridor"], due_h, maxiter)
    node_h, node_ok = _solve_priced_due(inst, tolls["node"], due_h, maxiter)
    contemp_h, contemp_ok = _solve_priced_due(inst, tolls["contemporaneous"], due_h, maxiter)
    resource_h, resource_ok = _solve_priced_due(
        inst, tolls["delay_plus_reliability"], due_h, maxiter
    )
    due_cost = total_cost(inst, due_h)
    so_cost = total_cost(inst, so_h)
    corridor_cost = total_cost(inst, corridor_h)
    node_cost = total_cost(inst, node_h)
    contemp_cost = total_cost(inst, contemp_h)
    resource_cost = total_cost(inst, resource_h)
    corridor_gap = _vi_gap(inst, so_h, private_cost(inst, so_h) + tolls["corridor"])
    node_gap = _vi_gap(inst, so_h, private_cost(inst, so_h) + tolls["node"])
    contemp_gap = _vi_gap(inst, so_h, private_cost(inst, so_h) + tolls["contemporaneous"])
    resource_gap = _vi_gap(inst, so_h, private_cost(inst, so_h) + tolls["delay_plus_reliability"])
    return {
        "success": bool(corridor_ok and node_ok and contemp_ok and resource_ok),
        "Bottleneck type": cfg.bottleneck_type,
        "Pressure target": cfg.pressure,
        "Reliability": cfg.reliability,
        "Seed": cfg.seed,
        "Alternatives": inst.n_alt,
        "PoA": due_cost / max(so_cost, 1e-09),
        "Corridor projection residual": residuals["corridor"],
        "Node projection residual": residuals["node"],
        "Contemporaneous projection residual": residuals["contemporaneous"],
        "All-resource projection residual": residuals["all_resource"],
        "Corridor-only recovery": _recovery(due_cost, so_cost, corridor_cost),
        "Node-only recovery": _recovery(due_cost, so_cost, node_cost),
        "Contemporaneous-only recovery": _recovery(due_cost, so_cost, contemp_cost),
        "All-resource recovery": _recovery(due_cost, so_cost, resource_cost),
        "Corridor-only recovery loss": 1.0 - _recovery(due_cost, so_cost, corridor_cost),
        "Node-only recovery loss": 1.0 - _recovery(due_cost, so_cost, node_cost),
        "Contemporaneous-only recovery loss": 1.0 - _recovery(due_cost, so_cost, contemp_cost),
        "Corridor SO support gap": corridor_gap,
        "Node SO support gap": node_gap,
        "Contemporaneous SO support gap": contemp_gap,
        "All-resource SO support gap": resource_gap,
    }


def _summarize_projection_panel(results: pd.DataFrame) -> dict[str, float]:
    ok = results[results["success"] == True].copy()
    if ok.empty:
        return {"instances": float(len(results)), "successful_instances": 0.0}
    residual_loss = pd.concat(
        [
            pd.DataFrame(
                {
                    "residual": ok["Corridor projection residual"],
                    "loss": ok["Corridor-only recovery loss"],
                    "gap": ok["Corridor SO support gap"],
                }
            ),
            pd.DataFrame(
                {
                    "residual": ok["Node projection residual"],
                    "loss": ok["Node-only recovery loss"],
                    "gap": ok["Node SO support gap"],
                }
            ),
            pd.DataFrame(
                {
                    "residual": ok["Contemporaneous projection residual"],
                    "loss": ok["Contemporaneous-only recovery loss"],
                    "gap": ok["Contemporaneous SO support gap"],
                }
            ),
        ],
        ignore_index=True,
    )
    return {
        "instances": float(len(results)),
        "successful_instances": float(len(ok)),
        "mean_corridor_projection_residual": float(ok["Corridor projection residual"].mean()),
        "mean_node_projection_residual": float(ok["Node projection residual"].mean()),
        "mean_contemporaneous_projection_residual": float(
            ok["Contemporaneous projection residual"].mean()
        ),
        "mean_all_resource_projection_residual": float(
            ok["All-resource projection residual"].mean()
        ),
        "max_all_resource_projection_residual": float(ok["All-resource projection residual"].max()),
        "residual_recovery_loss_corr": _safe_corr(residual_loss["residual"], residual_loss["loss"]),
        "residual_so_support_gap_corr": _safe_corr(residual_loss["residual"], residual_loss["gap"]),
        "corridor_residual_loss_corr": _safe_corr(
            ok["Corridor projection residual"], ok["Corridor-only recovery loss"]
        ),
        "node_residual_loss_corr": _safe_corr(
            ok["Node projection residual"], ok["Node-only recovery loss"]
        ),
        "contemporaneous_residual_loss_corr": _safe_corr(
            ok["Contemporaneous projection residual"], ok["Contemporaneous-only recovery loss"]
        ),
    }


def run_projection_residual_panel(
    out_dir: Path = OUT, instances: int = 80, seed: int = 20260626, maxiter: int = 150
) -> tuple[pd.DataFrame, dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    rng = np.random.default_rng(seed)
    for i in range(instances):
        cfg = StudyConfig(
            n_od=int(rng.choice([3, 5])),
            paths_per_od=int(rng.choice([3, 5])),
            n_dep=12,
            bottleneck_type=str(rng.choice(BOTTLENECK_TYPES)),
            pressure=float(rng.choice([0.75, 0.85, 0.95])),
            reliability=str(rng.choice(["loose", "medium", "tight"])),
            window_width=int(rng.choice([4, 6])),
            seed=seed * 1000 + i,
        )
        try:
            rows.append(solve_projection_instance(cfg, maxiter=maxiter))
        except Exception as exc:
            rows.append(
                {
                    "success": False,
                    "Seed": cfg.seed,
                    "Bottleneck type": cfg.bottleneck_type,
                    "error": str(exc),
                }
            )
    results = pd.DataFrame(rows)
    summary = _summarize_projection_panel(results)
    results.to_csv(out_dir / "projection_residual_panel.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "projection_residual_summary.csv", index=False)
    return (results, summary)
