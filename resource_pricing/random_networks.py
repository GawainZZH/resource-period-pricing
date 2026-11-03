from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .controlled_networks import (
        OUT,
        StudyConfig,
        SyntheticInstance,
        _softmax_profile,
        solve_instance,
    )
except ImportError:
    from resource_pricing.controlled_networks import (
        OUT,
        StudyConfig,
        SyntheticInstance,
        _softmax_profile,
        solve_instance,
    )


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return 0.0
    if pair.iloc[:, 0].std() <= 1e-12 or pair.iloc[:, 1].std() <= 1e-12:
        return 0.0
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def build_random_geometric_instance(seed: int, pressure: float | None = None) -> SyntheticInstance:
    rng = np.random.default_rng(seed)
    n_zones = int(rng.integers(8, 13))
    n_merge = int(rng.integers(3, 6))
    n_od = 3
    paths_per_od = 3
    n_dep = 12
    window_width = 4
    max_lag = 4
    n_res = n_dep + max_lag
    if pressure is None:
        pressure = float(rng.uniform(0.7, 0.96))
    zone_coords = rng.uniform(-5.0, 5.0, size=(n_zones, 2))
    merge_coords = rng.normal(0.0, 2.0, size=(n_merge, 2))
    od_pairs: list[tuple[int, int]] = []
    while len(od_pairs) < n_od:
        origin, dest = rng.choice(n_zones, size=2, replace=False)
        if origin != dest:
            od_pairs.append((int(origin), int(dest)))
    edge_to_idx: dict[tuple[str, int, str, int], int] = {}

    def corridor_idx(edge: tuple[str, int, str, int]) -> int:
        if edge not in edge_to_idx:
            edge_to_idx[edge] = len(edge_to_idx)
        return edge_to_idx[edge]

    merge_options: list[list[int]] = []
    for origin, dest in od_pairs:
        midpoint = 0.5 * (zone_coords[origin] + zone_coords[dest])
        dist = np.linalg.norm(merge_coords - midpoint, axis=1)
        nearest = list(np.argsort(dist)[: min(paths_per_od, n_merge)])
        while len(nearest) < paths_per_od:
            nearest.append(int(rng.integers(0, n_merge)))
        rng.shuffle(nearest)
        merge_options.append([int(x) for x in nearest[:paths_per_od]])
        for merge in merge_options[-1]:
            corridor_idx(("z", origin, "m", merge))
            corridor_idx(("m", merge, "z", dest))
    n_corridors = len(edge_to_idx)
    n_nodes = n_zones + n_merge
    n_resources = n_corridors + n_nodes
    resource_types = np.array(["corridor"] * n_corridors + ["node"] * n_nodes)
    corridor_mask = resource_types == "corridor"
    node_mask = resource_types == "node"
    demand = rng.uniform(4.0, 7.2, size=n_od)
    demand *= 1.0 + 0.12 * (pressure - 0.8)
    pref_center = rng.integers(3, 9, size=n_od)
    alts: list[tuple[int, int, int]] = []
    for w in range(n_od):
        start = int(np.clip(pref_center[w] - window_width // 2, 0, n_dep - window_width))
        for p in range(paths_per_od):
            for t in range(start, start + window_width):
                alts.append((w, p, t))
    n_alt = len(alts)
    od_of_alt = np.array([a[0] for a in alts], dtype=int)
    path_of_alt = np.array([a[1] for a in alts], dtype=int)
    dep_of_alt = np.array([a[2] for a in alts], dtype=int)
    delta = np.zeros((n_resources, n_res, n_alt), dtype=float)
    planned_time = np.zeros(n_alt, dtype=float)
    schedule_penalty = np.zeros(n_alt, dtype=float)
    deadline = np.zeros(n_alt, dtype=float)
    edge_index = {edge: idx for edge, idx in edge_to_idx.items()}
    for j, (w, p, t) in enumerate(alts):
        origin, dest = od_pairs[w]
        merge = merge_options[w][p]
        c1 = edge_index["z", origin, "m", merge]
        c2 = edge_index["m", merge, "z", dest]
        merge_node = n_corridors + n_zones + merge
        dest_node = n_corridors + dest
        distance = np.linalg.norm(zone_coords[origin] - merge_coords[merge]) + np.linalg.norm(
            merge_coords[merge] - zone_coords[dest]
        )
        node_lag = int(rng.choice([1, 2, 3, 4], p=[0.18, 0.34, 0.32, 0.16]))
        service_lag = min(node_lag + int(rng.choice([0, 1], p=[0.45, 0.55])), max_lag)
        delta[c1, t, j] = 1.0
        if t + 1 < n_res:
            delta[c2, t + 1, j] = 1.0
        if t + node_lag < n_res:
            delta[merge_node, t + node_lag, j] = rng.uniform(0.7, 1.25)
        if t + service_lag < n_res:
            delta[dest_node, t + service_lag, j] += rng.uniform(0.2, 0.85)
        planned_time[j] = 1.6 + 0.22 * distance + rng.uniform(0.0, 0.2)
        schedule_penalty[j] = 0.32 * abs(t - pref_center[w])
        deadline[j] = pref_center[w] + planned_time[j] + float(rng.uniform(2.5, 5.5))
    preferred = np.zeros(n_alt)
    for w in range(n_od):
        mask = od_of_alt == w
        profile = _softmax_profile(dep_of_alt[mask], pref_center[w], 1.2)
        path_base = np.exp(-0.25 * path_of_alt[mask] + rng.normal(0.0, 0.08, size=mask.sum()))
        weights = profile * path_base
        weights = weights / weights.sum()
        preferred[mask] = demand[w] * weights
    base_load = np.tensordot(delta, preferred, axes=([2], [0]))
    capacity = np.maximum(1.2, base_load / max(pressure, 1e-06))
    random_capacity_factor = rng.lognormal(mean=0.0, sigma=0.28, size=capacity.shape)
    corridor_global = float(rng.lognormal(mean=0.0, sigma=0.45))
    node_global = float(rng.lognormal(mean=0.0, sigma=0.45))
    type_factor = np.where(
        corridor_mask[:, None], corridor_global * rng.uniform(0.72, 1.3, size=(n_resources, 1)), 1.0
    )
    node_factor = np.where(
        node_mask[:, None], node_global * rng.uniform(0.68, 1.36, size=(n_resources, 1)), 1.0
    )
    period_factor = rng.uniform(0.86, 1.16, size=(1, n_res))
    capacity = np.maximum(
        1.0, capacity * random_capacity_factor * type_factor * node_factor * period_factor
    )
    corridor_alpha = float(rng.lognormal(mean=np.log(0.68), sigma=0.35))
    node_alpha = float(rng.lognormal(mean=np.log(0.86), sigma=0.35))
    alpha = np.where(
        corridor_mask,
        corridor_alpha * rng.uniform(0.7, 1.35, size=n_resources),
        node_alpha * rng.uniform(0.65, 1.45, size=n_resources),
    )
    delayed_occ = delta[:, n_dep:, :].sum()
    total_occ = delta.sum()
    delayed_intensity = float(delayed_occ / max(total_occ, 1e-12))
    peak_pressure = float(np.max(base_load / capacity))
    reliability = str(rng.choice(["loose", "medium", "tight"], p=[0.3, 0.4, 0.3]))
    cfg = StudyConfig(
        n_od=n_od,
        paths_per_od=paths_per_od,
        n_dep=n_dep,
        bottleneck_type="mixed",
        pressure=float(pressure),
        reliability=reliability,
        window_width=window_width,
        seed=seed,
    )
    return SyntheticInstance(
        config=cfg,
        n_od=n_od,
        paths_per_od=paths_per_od,
        n_dep=n_dep,
        n_res=n_res,
        n_resources=n_resources,
        n_alt=n_alt,
        resource_types=resource_types,
        delta=delta,
        capacity=capacity,
        alpha=alpha,
        demand=demand,
        od_of_alt=od_of_alt,
        path_of_alt=path_of_alt,
        dep_of_alt=dep_of_alt,
        planned_time=planned_time,
        schedule_penalty=schedule_penalty,
        deadline=deadline,
        rho_late=1.8,
        base_load=base_load,
        peak_pressure=peak_pressure,
        delayed_intensity=delayed_intensity,
    )


def _posthoc_class(row: pd.Series) -> str:
    delayed = float(row.get("Delayed toll share", 0.0))
    corridor = float(row.get("Corridor externality share", 0.0))
    node = float(row.get("Node externality share", 0.0))
    if delayed >= 0.84:
        return "delayed-resource"
    if corridor >= 0.58:
        return "corridor-heavy"
    if node >= 0.58:
        return "node-heavy"
    return "mixed"


def _posthoc_subclass(row: pd.Series) -> str:
    corridor_res = float(row.get("Corridor projection residual", 0.0))
    node_res = float(row.get("Node projection residual", 0.0))
    contemp_res = float(row.get("Contemporaneous projection residual", 0.0))
    delayed = float(row.get("Delayed toll share", 0.0))
    values = {
        "corridor-hidden": corridor_res,
        "node-hidden": node_res,
        "temporal-hidden": contemp_res,
    }
    dominant = max(values, key=values.get)
    if max(values.values()) <= 0.05:
        return "low-residual"
    if dominant == "temporal-hidden" and delayed >= 0.65:
        return "delayed-temporal-hidden"
    return dominant


def _add_residual_groups(results: pd.DataFrame) -> pd.DataFrame:
    if "Maximum restricted projection residual" not in results.columns:
        return results
    ok = results["success"] == True
    results["Residual quantile group"] = "failed"
    if int(ok.sum()) == 0:
        return results
    values = results.loc[ok, "Maximum restricted projection residual"].astype(float)
    if values.nunique(dropna=True) < 3:
        median = values.median()
        results.loc[ok, "Residual quantile group"] = np.where(values <= median, "low", "high")
        return results
    grouped = pd.qcut(values, q=3, labels=["low", "middle", "high"], duplicates="drop")
    results.loc[ok, "Residual quantile group"] = grouped.astype(str).to_numpy()
    return results


def _standardized_column(values: pd.Series) -> np.ndarray:
    arr = values.astype(float).to_numpy()
    std = float(np.nanstd(arr))
    if std <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - float(np.nanmean(arr))) / std


def _hc1_ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> pd.DataFrame:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = x.T @ (resid[:, None] ** 2 * x)
    scale = n / max(n - k, 1)
    cov = scale * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_stat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 1e-12)
    return pd.DataFrame({"term": names, "coefficient": beta, "robust_se": se, "robust_t": t_stat})


def _mechanism_regression_panel(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["success"] == True].copy()
    rows: list[dict[str, float | str]] = []
    for _, row in ok.iterrows():
        due_so_gap = float(row["PoA"]) - 1.0
        pressure = float(row.get("Realized peak pressure", row.get("Pressure target", 0.0)))
        delayed = float(row.get("Delayed occupation intensity", row.get("Delayed toll share", 0.0)))
        concentration = max(
            float(row.get("Corridor externality share", 0.0)),
            float(row.get("Node externality share", 0.0)),
        )
        variants = [
            ("corridor", "Corridor projection residual", "Corridor-only recovery loss"),
            ("node", "Node projection residual", "Node-only recovery loss"),
            (
                "contemporaneous",
                "Contemporaneous projection residual",
                "Contemporaneous-only recovery loss",
            ),
        ]
        for price_class, residual_col, loss_col in variants:
            if residual_col not in row.index or loss_col not in row.index:
                continue
            rows.append(
                {
                    "price_class": price_class,
                    "recovery_loss": float(row[loss_col]),
                    "projection_residual": float(row[residual_col]),
                    "due_so_gap": due_so_gap,
                    "delayed_intensity": delayed,
                    "pressure": pressure,
                    "resource_concentration": concentration,
                }
            )
    return pd.DataFrame(rows)


def _summarize_mechanism_regression(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    panel = _mechanism_regression_panel(results).replace([np.inf, -np.inf], np.nan).dropna()
    if len(panel) < 3:
        empty = pd.DataFrame(columns=["term", "coefficient", "robust_se", "robust_t"])
        return (
            empty,
            {
                "mechanism_regression_observations": float(len(panel)),
                "mechanism_regression_projection_residual_coef": 0.0,
                "mechanism_regression_projection_residual_t": 0.0,
                "mechanism_regression_r2": 0.0,
            },
        )
    x_cols = [
        ("projection_residual", "Projection residual"),
        ("due_so_gap", "DUE-SO gap"),
        ("delayed_intensity", "Delayed intensity"),
        ("pressure", "Pressure"),
        ("resource_concentration", "Resource concentration"),
    ]
    x_parts = [np.ones((len(panel), 1))]
    names = ["Intercept"]
    for col, name in x_cols:
        x_parts.append(_standardized_column(panel[col])[:, None])
        names.append(name)
    if panel["price_class"].nunique() >= 3 and len(panel) >= 12:
        x_parts.append((panel["price_class"].to_numpy() == "node").astype(float)[:, None])
        names.append("Node price class")
        x_parts.append(
            (panel["price_class"].to_numpy() == "contemporaneous").astype(float)[:, None]
        )
        names.append("Contemporaneous price class")
    x = np.hstack(x_parts)
    y = panel["recovery_loss"].astype(float).to_numpy()
    regression = _hc1_ols(y, x, names)
    fitted = x @ regression["coefficient"].to_numpy()
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum((y - fitted) ** 2))
    r2 = 0.0 if sst <= 1e-12 else 1.0 - sse / sst
    projection = regression[regression["term"] == "Projection residual"].iloc[0]
    summary = {
        "mechanism_regression_observations": float(len(panel)),
        "mechanism_regression_projection_residual_coef": float(projection["coefficient"]),
        "mechanism_regression_projection_residual_t": float(projection["robust_t"]),
        "mechanism_regression_r2": float(r2),
    }
    for term in ["DUE-SO gap", "Delayed intensity", "Pressure", "Resource concentration"]:
        match = regression[regression["term"] == term]
        if not match.empty:
            key = term.lower().replace("-", "_").replace(" ", "_")
            summary[f"mechanism_regression_{key}_coef"] = float(match.iloc[0]["coefficient"])
            summary[f"mechanism_regression_{key}_t"] = float(match.iloc[0]["robust_t"])
    regression["r2"] = r2
    regression["observations"] = float(len(panel))
    return (regression, summary)


def _summarize_random_geometric(results: pd.DataFrame) -> dict[str, float]:
    ok = results[results["success"] == True].copy()
    if ok.empty:
        return {
            "instances": float(len(results)),
            "successful_instances": 0.0,
            "mean_poa": 0.0,
            "corridor_omitted_loss_corr": 0.0,
            "node_omitted_loss_corr": 0.0,
            "delayed_share_contemp_gap_corr": 0.0,
            "negative_recovery_frequency": 0.0,
        }
    negative = (
        (ok["Corridor-only recovery"] < 0)
        | (ok["Node-only recovery"] < 0)
        | (ok["Contemporaneous-only recovery"] < 0)
    )
    ok["Contemporaneous-only recovery loss"] = 1.0 - ok["Contemporaneous-only recovery"]
    ok["Maximum restricted recovery loss"] = ok[
        [
            "Corridor-only recovery loss",
            "Node-only recovery loss",
            "Contemporaneous-only recovery loss",
        ]
    ].max(axis=1)
    low = ok[ok["Residual quantile group"] == "low"]
    high = ok[ok["Residual quantile group"] == "high"]
    summary = {
        "instances": float(len(results)),
        "successful_instances": float(len(ok)),
        "mean_poa": float(ok["PoA"].mean()),
        "corridor_omitted_loss_corr": _safe_corr(
            ok["Corridor-only omitted share"], ok["Corridor-only recovery loss"]
        ),
        "node_omitted_loss_corr": _safe_corr(
            ok["Node-only omitted share"], ok["Node-only recovery loss"]
        ),
        "delayed_share_contemp_gap_corr": _safe_corr(
            ok["Delayed toll share"], ok["Contemporaneous SO support gap"]
        ),
        "corridor_residual_loss_corr": _safe_corr(
            ok["Corridor projection residual"], ok["Corridor-only recovery loss"]
        ),
        "node_residual_loss_corr": _safe_corr(
            ok["Node projection residual"], ok["Node-only recovery loss"]
        ),
        "contemporaneous_residual_loss_corr": _safe_corr(
            ok["Contemporaneous projection residual"], ok["Contemporaneous-only recovery loss"]
        ),
        "max_residual_loss_corr": _safe_corr(
            ok["Maximum restricted projection residual"], ok["Maximum restricted recovery loss"]
        ),
        "mean_corridor_projection_residual": float(ok["Corridor projection residual"].mean()),
        "mean_node_projection_residual": float(ok["Node projection residual"].mean()),
        "mean_contemporaneous_projection_residual": float(
            ok["Contemporaneous projection residual"].mean()
        ),
        "mean_all_resource_projection_residual": float(
            ok["All-resource projection residual"].mean()
        ),
        "mean_high_residual_recovery_loss": float(
            high["Maximum restricted recovery loss"].mean() if not high.empty else 0.0
        ),
        "mean_low_residual_recovery_loss": float(
            low["Maximum restricted recovery loss"].mean() if not low.empty else 0.0
        ),
        "negative_recovery_frequency": float(negative.mean()),
        "mean_corridor_recovery": float(ok["Corridor-only recovery"].mean()),
        "mean_node_recovery": float(ok["Node-only recovery"].mean()),
        "mean_contemporaneous_recovery": float(ok["Contemporaneous-only recovery"].mean()),
        "mean_resource_recovery": float(ok["Resource-externality recovery"].mean()),
        "mean_delayed_toll_share": float(ok["Delayed toll share"].mean()),
        "mean_contemporaneous_support_gap": float(ok["Contemporaneous SO support gap"].mean()),
    }
    class_share = ok["Post-hoc bottleneck class"].value_counts(normalize=True)
    for cls in ["corridor-heavy", "node-heavy", "delayed-resource", "mixed"]:
        summary[f"posthoc_{cls}_share"] = float(class_share.get(cls, 0.0))
    subclass_share = ok["Post-hoc bottleneck subclass"].value_counts(normalize=True)
    for cls in [
        "corridor-hidden",
        "node-hidden",
        "temporal-hidden",
        "delayed-temporal-hidden",
        "low-residual",
    ]:
        summary[f"posthoc_{cls}_share"] = float(subclass_share.get(cls, 0.0))
    _, regression_summary = _summarize_mechanism_regression(results)
    summary.update(regression_summary)
    return summary


def _posthoc_bottleneck_summary(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["success"] == True].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "Post-hoc bottleneck class",
                "Instances",
                "Mean PoA",
                "Corridor-only recovery",
                "Node-only recovery",
                "Contemporaneous-only recovery",
                "All-resource delay recovery",
                "Maximum restricted projection residual",
                "Negative restricted-recovery frequency",
            ]
        )
    ok["Negative restricted recovery"] = (
        (ok["Corridor-only recovery"] < 0)
        | (ok["Node-only recovery"] < 0)
        | (ok["Contemporaneous-only recovery"] < 0)
    )
    grouped = ok.groupby("Post-hoc bottleneck class", as_index=False).agg(
        Instances=("success", "size"),
        **{
            "Mean PoA": ("PoA", "mean"),
            "Corridor-only recovery": ("Corridor-only recovery", "mean"),
            "Node-only recovery": ("Node-only recovery", "mean"),
            "Contemporaneous-only recovery": ("Contemporaneous-only recovery", "mean"),
            "All-resource delay recovery": ("Resource-externality recovery", "mean"),
            "Maximum restricted projection residual": (
                "Maximum restricted projection residual",
                "mean",
            ),
            "Negative restricted-recovery frequency": ("Negative restricted recovery", "mean"),
        },
    )
    order = {"corridor-heavy": 0, "node-heavy": 1, "delayed-resource": 2, "mixed": 3}
    grouped["_order"] = grouped["Post-hoc bottleneck class"].map(order).fillna(99)
    return grouped.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def run_random_geometric_panel(
    out_dir: Path = OUT, instances: int = 150, seed: int = 20260626, maxiter: int = 150
) -> tuple[pd.DataFrame, dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    rng = np.random.default_rng(seed)
    for i in range(instances):
        pressure = float(rng.uniform(0.68, 0.97))
        inst = build_random_geometric_instance(seed=seed * 1000 + i, pressure=pressure)
        try:
            row = solve_instance(inst, maxiter=maxiter)
        except Exception as exc:
            row = {"success": False, "Seed": seed * 1000 + i, "error": str(exc)}
        row["Generator"] = "random-geometric"
        if row.get("success") is True:
            series = pd.Series(row)
            row["Post-hoc bottleneck class"] = _posthoc_class(series)
            row["Post-hoc bottleneck subclass"] = _posthoc_subclass(series)
            row["Contemporaneous-only recovery loss"] = 1.0 - float(
                row["Contemporaneous-only recovery"]
            )
            row["Maximum restricted recovery loss"] = max(
                float(row["Corridor-only recovery loss"]),
                float(row["Node-only recovery loss"]),
                float(row["Contemporaneous-only recovery loss"]),
            )
        else:
            row["Post-hoc bottleneck class"] = "failed"
            row["Post-hoc bottleneck subclass"] = "failed"
        rows.append(row)
    results = pd.DataFrame(rows)
    results = _add_residual_groups(results)
    summary = _summarize_random_geometric(results)
    regression, _ = _summarize_mechanism_regression(results)
    posthoc = _posthoc_bottleneck_summary(results)
    results.to_csv(out_dir / "random_geometric_validation_instances.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "random_geometric_validation_summary.csv", index=False)
    regression.to_csv(out_dir / "random_geometric_mechanism_regression.csv", index=False)
    posthoc.to_csv(out_dir / "random_geometric_posthoc_bottleneck_summary.csv", index=False)
    return (results, summary)
