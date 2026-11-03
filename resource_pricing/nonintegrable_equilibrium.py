from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from .controlled_networks import (
        OUT,
        StudyConfig,
        _constraints,
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
        _constraints,
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


def _project_to_feasible(inst, target: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
    eq, cap, bounds, _, _ = _constraints(inst)
    return _project_to_feasible_with_constraints(inst, target, x0, eq, cap, bounds)


def _project_to_feasible_with_constraints(
    inst, target: np.ndarray, x0: np.ndarray | None, eq, cap, bounds
) -> np.ndarray:
    start = _initial_flow(inst) if x0 is None else x0
    res = minimize(
        lambda x: 0.5 * float(np.dot(x - target, x - target)),
        start,
        method="SLSQP",
        jac=lambda x: x - target,
        bounds=bounds,
        constraints=[eq, cap],
        options={"ftol": 1e-10, "maxiter": 180, "disp": False},
    )
    if not res.success:
        raise RuntimeError(f"Projection failed: {res.message}")
    return np.maximum(res.x, 0.0)


def _extragradient_vi(
    inst, cost_map, x0: np.ndarray, step: float = 0.055, max_iter: int = 420, tol: float = 1e-05
) -> tuple[np.ndarray, float, int]:
    eq, cap, bounds, _, _ = _constraints(inst)
    project = lambda target, start: _project_to_feasible_with_constraints(
        inst, target, start, eq, cap, bounds
    )
    x = project(x0, x0)
    for it in range(1, max_iter + 1):
        y = project(x - step * cost_map(x), x)
        x_next = project(x - step * cost_map(y), y)
        move = float(np.linalg.norm(x_next - x) / max(1.0, np.linalg.norm(x)))
        x = x_next
        if move < tol:
            break
    return (x, _vi_gap(inst, x, cost_map(x)), it)


def _skew_operator(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=(n, n))
    skew = raw - raw.T
    return skew / max(float(np.sqrt(n)), 1.0)


def _normalized_gap(inst, h: np.ndarray, cost: np.ndarray) -> float:
    return float(_vi_gap(inst, h, cost) / max(1.0, abs(float(np.dot(cost, h)))))


def _nonintegrable_configs(
    instances: int, seed: int, min_alternatives: int, max_alternatives: int
) -> list[StudyConfig]:
    rng = np.random.default_rng(seed)
    candidates: list[tuple[int, int, int]] = []
    for n_od in (3, 4, 5, 6, 8, 10):
        for paths in (3, 4, 5, 6):
            for window in (4, 5, 6, 8):
                alt_count = n_od * paths * window
                if min_alternatives <= alt_count <= max_alternatives:
                    candidates.append((n_od, paths, window))
    if not candidates:
        raise ValueError("No configuration satisfies the requested alternative range.")
    bottlenecks = ["corridor", "node", "mixed", "delayed-node"]
    pressures = [0.75, 0.85, 0.95]
    reliability = ["medium", "tight"]
    configs: list[StudyConfig] = []
    for idx in range(instances):
        n_od, paths, window = candidates[int(rng.integers(0, len(candidates)))]
        configs.append(
            StudyConfig(
                n_od=n_od,
                paths_per_od=paths,
                n_dep=12,
                bottleneck_type=bottlenecks[idx % len(bottlenecks)],
                pressure=pressures[int(rng.integers(0, len(pressures)))],
                reliability=reliability[int(rng.integers(0, len(reliability)))],
                window_width=window,
                seed=seed + 1009 + idx * 37,
            )
        )
    return configs


def _solve_nonintegrable_instance(
    cfg: StudyConfig, max_iter: int, gamma: float = 0.025, vi_tol: float = 1e-05
) -> dict[str, float | int | str | bool]:
    inst = generate_instance(cfg)
    x0 = _initial_flow(inst)
    potential_due, _, _ = _minimize_flow(
        inst, lambda h: due_potential(inst, h), lambda h: due_potential_gradient(inst, h), x0, 220
    )
    due_vi, due_gap, due_iter = _extragradient_vi(
        inst, lambda h: private_cost(inst, h), x0, max_iter=max_iter, tol=vi_tol
    )
    so_h, _, _ = _minimize_flow(
        inst, lambda h: total_cost(inst, h), lambda h: total_cost_gradient(inst, h), due_vi, 220
    )
    tolls = _resource_tolls(inst, so_h)
    skew = _skew_operator(inst.n_alt, cfg.seed)

    def asymmetric_private(h: np.ndarray) -> np.ndarray:
        return private_cost(inst, h) + gamma * (skew @ h)

    asym_due, asym_due_gap, asym_due_iter = _extragradient_vi(
        inst, asymmetric_private, due_vi, max_iter=max_iter, tol=vi_tol
    )
    resource_vi, resource_gap, resource_iter = _extragradient_vi(
        inst,
        lambda h, tau=tolls["resource"]: asymmetric_private(h) + tau,
        asym_due,
        max_iter=max_iter,
        tol=vi_tol,
    )
    contemp_vi, contemp_gap, contemp_iter = _extragradient_vi(
        inst,
        lambda h, tau=tolls["contemporaneous"]: asymmetric_private(h) + tau,
        asym_due,
        max_iter=max_iter,
        tol=vi_tol,
    )
    asym_full_toll = total_cost_gradient(inst, so_h) - asymmetric_private(so_h)
    potential_due_cost = asymmetric_private(potential_due)
    separable_due_cost = private_cost(inst, due_vi)
    asym_due_cost = asymmetric_private(asym_due)
    resource_cost = asymmetric_private(resource_vi) + tolls["resource"]
    contemp_cost = asymmetric_private(contemp_vi) + tolls["contemporaneous"]
    return {
        "success": True,
        "Bottleneck type": cfg.bottleneck_type,
        "Pressure target": cfg.pressure,
        "Reliability": cfg.reliability,
        "Alternatives": inst.n_alt,
        "Resources": inst.n_resources,
        "Potential DUE residual under nonintegrable cost": _normalized_gap(
            inst, potential_due, potential_due_cost
        ),
        "Separable DUE VI residual": _normalized_gap(inst, due_vi, separable_due_cost),
        "Nonintegrable DUE VI residual": _normalized_gap(inst, asym_due, asym_due_cost),
        "Nonintegrable DUE raw VI gap": asym_due_gap,
        "Nonintegrable DUE iterations": asym_due_iter,
        "Nonintegrable all-resource delay SO support gap": _vi_gap(
            inst, so_h, asymmetric_private(so_h) + tolls["resource"]
        ),
        "Nonintegrable contemporaneous SO support gap": _vi_gap(
            inst, so_h, asymmetric_private(so_h) + tolls["contemporaneous"]
        ),
        "Nonintegrable MC benchmark SO support gap": _vi_gap(
            inst, so_h, asymmetric_private(so_h) + asym_full_toll
        ),
        "Nonintegrable resource-period priced-DUE VI residual": _normalized_gap(
            inst, resource_vi, resource_cost
        ),
        "Nonintegrable resource-period priced-DUE raw VI gap": resource_gap,
        "Nonintegrable contemporaneous priced-DUE VI residual": _normalized_gap(
            inst, contemp_vi, contemp_cost
        ),
        "Nonintegrable contemporaneous priced-DUE raw VI gap": contemp_gap,
        "Nonintegrable resource-period priced-DUE iterations": resource_iter,
        "Nonintegrable contemporaneous priced-DUE iterations": contemp_iter,
        "Nonintegrable all-resource delay priced cost": total_cost(inst, resource_vi),
        "Nonintegrable contemporaneous priced cost": total_cost(inst, contemp_vi),
    }


def _summarize_nonintegrable_panel(result: pd.DataFrame) -> dict[str, float]:
    ok = result[result["success"] == True].copy()
    if ok.empty:
        return {"instances": float(len(result)), "successful_instances": 0.0}
    return {
        "instances": float(len(result)),
        "successful_instances": float(len(ok)),
        "mean_alternatives": float(ok["Alternatives"].mean()),
        "max_alternatives": float(ok["Alternatives"].max()),
        "mean_nonintegrable_due_vi_residual": float(ok["Nonintegrable DUE VI residual"].mean()),
        "max_nonintegrable_due_vi_residual": float(ok["Nonintegrable DUE VI residual"].max()),
        "mean_nonintegrable_resource_priced_vi_residual": float(
            ok["Nonintegrable resource-period priced-DUE VI residual"].mean()
        ),
        "max_nonintegrable_resource_priced_vi_residual": float(
            ok["Nonintegrable resource-period priced-DUE VI residual"].max()
        ),
        "mean_potential_due_residual_under_nonintegrable_cost": float(
            ok["Potential DUE residual under nonintegrable cost"].mean()
        ),
        "mean_nonintegrable_resource_support_gap": float(
            ok["Nonintegrable all-resource delay SO support gap"].mean()
        ),
        "max_nonintegrable_resource_support_gap": float(
            ok["Nonintegrable all-resource delay SO support gap"].max()
        ),
        "mean_nonintegrable_contemporaneous_support_gap": float(
            ok["Nonintegrable contemporaneous SO support gap"].mean()
        ),
        "max_nonintegrable_contemporaneous_support_gap": float(
            ok["Nonintegrable contemporaneous SO support gap"].max()
        ),
        "mean_nonintegrable_mc_support_gap": float(
            ok["Nonintegrable MC benchmark SO support gap"].mean()
        ),
        "max_nonintegrable_mc_support_gap": float(
            ok["Nonintegrable MC benchmark SO support gap"].max()
        ),
    }


def _summarize_tight_audit(result: pd.DataFrame) -> dict[str, float]:
    ok = result[result["success"] == True].copy()
    if ok.empty:
        return {
            "tight_audit_instances": float(len(result)),
            "tight_audit_successful_instances": 0.0,
        }
    return {
        "tight_audit_instances": float(len(result)),
        "tight_audit_successful_instances": float(len(ok)),
        "mean_base_resource_priced_vi_residual": float(
            ok["Base resource priced VI residual"].mean()
        ),
        "mean_tight_resource_priced_vi_residual": float(
            ok["Tight resource priced VI residual"].mean()
        ),
        "max_tight_resource_priced_vi_residual": float(
            ok["Tight resource priced VI residual"].max()
        ),
        "mean_resource_residual_ratio": float(ok["Resource residual ratio"].mean()),
        "mean_base_due_vi_residual": float(ok["Base nonintegrable DUE VI residual"].mean()),
        "mean_tight_due_vi_residual": float(ok["Tight nonintegrable DUE VI residual"].mean()),
        "max_tight_due_vi_residual": float(ok["Tight nonintegrable DUE VI residual"].max()),
        "mean_due_residual_ratio": float(ok["DUE residual ratio"].mean()),
        "mean_tight_resource_iterations": float(ok["Tight resource priced VI iterations"].mean()),
        "mean_tight_due_iterations": float(ok["Tight nonintegrable DUE iterations"].mean()),
    }


def run_tight_nonintegrable_vi_audit(
    out_dir: Path = OUT,
    instances: int = 10,
    seed: int = 20260627,
    base_iter: int = 60,
    tight_iter: int = 160,
    min_alternatives: int = 36,
    max_alternatives: int = 120,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    for cfg in _nonintegrable_configs(instances, seed, min_alternatives, max_alternatives):
        try:
            base = _solve_nonintegrable_instance(cfg, max_iter=base_iter, vi_tol=1e-05)
            tight = _solve_nonintegrable_instance(cfg, max_iter=tight_iter, vi_tol=1e-08)
            base_resource = float(base["Nonintegrable resource-period priced-DUE VI residual"])
            tight_resource = float(tight["Nonintegrable resource-period priced-DUE VI residual"])
            base_due = float(base["Nonintegrable DUE VI residual"])
            tight_due = float(tight["Nonintegrable DUE VI residual"])
            rows.append(
                {
                    "success": True,
                    "Bottleneck type": cfg.bottleneck_type,
                    "Pressure target": cfg.pressure,
                    "Reliability": cfg.reliability,
                    "Alternatives": tight["Alternatives"],
                    "Resources": tight["Resources"],
                    "Base nonintegrable DUE VI residual": base_due,
                    "Tight nonintegrable DUE VI residual": tight_due,
                    "DUE residual ratio": tight_due / max(base_due, 1e-12),
                    "Base resource priced VI residual": base_resource,
                    "Tight resource priced VI residual": tight_resource,
                    "Resource residual ratio": tight_resource / max(base_resource, 1e-12),
                    "Base nonintegrable DUE iterations": base["Nonintegrable DUE iterations"],
                    "Tight nonintegrable DUE iterations": tight["Nonintegrable DUE iterations"],
                    "Base resource priced VI iterations": base[
                        "Nonintegrable resource-period priced-DUE iterations"
                    ],
                    "Tight resource priced VI iterations": tight[
                        "Nonintegrable resource-period priced-DUE iterations"
                    ],
                    "Tight resource SO support gap": tight[
                        "Nonintegrable all-resource delay SO support gap"
                    ],
                    "Tight contemporaneous SO support gap": tight[
                        "Nonintegrable contemporaneous SO support gap"
                    ],
                    "Tight MC benchmark SO support gap": tight[
                        "Nonintegrable MC benchmark SO support gap"
                    ],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "success": False,
                    "Bottleneck type": cfg.bottleneck_type,
                    "Pressure target": cfg.pressure,
                    "Reliability": cfg.reliability,
                    "Alternatives": cfg.n_od * cfg.paths_per_od * cfg.window_width,
                    "error": str(exc),
                }
            )
    result = pd.DataFrame(rows)
    summary = _summarize_tight_audit(result)
    result.to_csv(out_dir / "nonintegrable_vi_tight_audit.csv", index=False)
    pd.DataFrame([summary]).to_csv(
        out_dir / "nonintegrable_vi_tight_audit_summary.csv", index=False
    )
    return (result, summary)


def run_nonintegrable_vi_panel(
    out_dir: Path = OUT,
    instances: int = 40,
    seed: int = 20260626,
    max_iter: int = 90,
    min_alternatives: int = 100,
    max_alternatives: int = 300,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    for cfg in _nonintegrable_configs(instances, seed, min_alternatives, max_alternatives):
        try:
            rows.append(_solve_nonintegrable_instance(cfg, max_iter=max_iter))
        except Exception as exc:
            rows.append(
                {
                    "success": False,
                    "Bottleneck type": cfg.bottleneck_type,
                    "Pressure target": cfg.pressure,
                    "Reliability": cfg.reliability,
                    "Alternatives": cfg.n_od * cfg.paths_per_od * cfg.window_width,
                    "error": str(exc),
                }
            )
    result = pd.DataFrame(rows)
    summary = _summarize_nonintegrable_panel(result)
    result.to_csv(out_dir / "nonintegrable_vi_panel.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "nonintegrable_vi_summary.csv", index=False)
    return (result, summary)


def run_vi_validation(out_dir: Path = OUT, seed: int = 20260625) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        StudyConfig(3, 3, 12, "corridor", 0.85, "tight", 4, seed + 11),
        StudyConfig(3, 3, 12, "node", 0.85, "tight", 4, seed + 23),
        StudyConfig(3, 3, 12, "mixed", 0.9, "medium", 4, seed + 37),
        StudyConfig(3, 3, 12, "delayed-node", 0.9, "tight", 4, seed + 51),
    ]
    rows: list[dict[str, float | int | str]] = []
    for cfg in configs:
        inst = generate_instance(cfg)
        x0 = _initial_flow(inst)
        potential_due, _, _ = _minimize_flow(
            inst,
            lambda h: due_potential(inst, h),
            lambda h: due_potential_gradient(inst, h),
            x0,
            260,
        )
        due_vi, due_gap, due_iter = _extragradient_vi(inst, lambda h: private_cost(inst, h), x0)
        so_h, _, _ = _minimize_flow(
            inst, lambda h: total_cost(inst, h), lambda h: total_cost_gradient(inst, h), due_vi, 260
        )
        tolls = _resource_tolls(inst, so_h)
        resource_vi, resource_gap, resource_iter = _extragradient_vi(
            inst, lambda h, tau=tolls["resource"]: private_cost(inst, h) + tau, due_vi
        )
        contemp_vi, contemp_gap, contemp_iter = _extragradient_vi(
            inst, lambda h, tau=tolls["contemporaneous"]: private_cost(inst, h) + tau, due_vi
        )
        skew = _skew_operator(inst.n_alt, cfg.seed)
        gamma = 0.025

        def asymmetric_private(h: np.ndarray) -> np.ndarray:
            return private_cost(inst, h) + gamma * (skew @ h)

        asym_due, asym_due_gap, asym_due_iter = _extragradient_vi(inst, asymmetric_private, due_vi)
        asym_full_toll = total_cost_gradient(inst, so_h) - asymmetric_private(so_h)
        rows.append(
            {
                "Bottleneck type": cfg.bottleneck_type,
                "Alternatives": inst.n_alt,
                "Potential DUE VI gap": _vi_gap(
                    inst, potential_due, private_cost(inst, potential_due)
                ),
                "Extragradient DUE VI gap": due_gap,
                "DUE solution distance": float(
                    np.linalg.norm(due_vi - potential_due) / max(1.0, np.linalg.norm(potential_due))
                ),
                "Resource priced VI gap": resource_gap,
                "Contemporaneous priced VI gap": contemp_gap,
                "Asymmetric DUE VI gap": asym_due_gap,
                "Asymmetric DUE iterations": asym_due_iter,
                "Asymmetric resource SO support gap": _vi_gap(
                    inst, so_h, asymmetric_private(so_h) + tolls["resource"]
                ),
                "Asymmetric contemporaneous SO support gap": _vi_gap(
                    inst, so_h, asymmetric_private(so_h) + tolls["contemporaneous"]
                ),
                "Asymmetric MC benchmark SO support gap": _vi_gap(
                    inst, so_h, asymmetric_private(so_h) + asym_full_toll
                ),
                "Resource VI iterations": resource_iter,
                "Contemporaneous VI iterations": contemp_iter,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "vi_validation_results.csv", index=False)
    summary = (
        result.select_dtypes(include=[np.number])
        .mean()
        .to_frame("mean")
        .join(result.select_dtypes(include=[np.number]).max().to_frame("max"))
    )
    summary.to_csv(out_dir / "vi_validation_summary.csv")
    return result
