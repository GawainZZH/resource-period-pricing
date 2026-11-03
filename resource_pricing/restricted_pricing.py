from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize
from resource_pricing.controlled_networks import (
    BOTTLENECK_TYPES,
    SyntheticInstance,
    _config_stream,
    _constraints,
    _flat_load,
    _initial_flow,
    _reliability_externality_signal,
    _reliability_threshold,
    _resource_tolls,
    _solve_priced_due,
    _vi_gap,
    build_metropolitan_corridor_instance,
    generate_instance,
    private_cost,
    solve_core_equilibria,
    total_cost,
    total_cost_gradient,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "best_in_class_pricing_v1"
PRICE_CLASSES = ("corridor", "node", "contemporaneous")


@dataclass(frozen=True)
class PriceParameterization:
    price_class: str
    design: np.ndarray
    resource_indices: np.ndarray
    periods: np.ndarray
    resource_types: np.ndarray


@dataclass
class BestInClassResult:
    price_class: str
    flow: np.ndarray
    resource_prices: np.ndarray
    alternative_price: np.ndarray
    system_cost: float
    gap_closure_ratio: float
    vi_residual: float
    normalized_vi_residual: float
    outer_iterations: int
    objective_evaluations: int
    upper_bound_fraction: float
    success: bool
    message: str


def build_price_parameterization(
    inst: SyntheticInstance, price_class: str
) -> PriceParameterization:
    """Map nonnegative resource-period price parameters to alternative prices."""
    if price_class not in PRICE_CLASSES:
        raise ValueError(f"Unknown price class: {price_class}")
    flat_delta = inst.delta.reshape(inst.n_resources * inst.n_res, inst.n_alt).T
    resource_indices = np.repeat(np.arange(inst.n_resources), inst.n_res)
    periods = np.tile(np.arange(inst.n_res), inst.n_resources)
    if price_class == "corridor":
        allowed = inst.resource_types[resource_indices] == "corridor"
        design = flat_delta[:, allowed]
        resource_indices = resource_indices[allowed]
        periods = periods[allowed]
    elif price_class == "node":
        allowed = inst.resource_types[resource_indices] == "node"
        design = flat_delta[:, allowed]
        resource_indices = resource_indices[allowed]
        periods = periods[allowed]
    else:
        design = flat_delta.copy()
        design[inst.dep_of_alt[:, None] != periods[None, :]] = 0.0
    used = np.any(np.abs(design) > 1e-12, axis=0)
    design = design[:, used]
    resource_indices = resource_indices[used]
    periods = periods[used]
    resource_types = inst.resource_types[resource_indices]
    return PriceParameterization(
        price_class=price_class,
        design=design,
        resource_indices=resource_indices,
        periods=periods,
        resource_types=resource_types,
    )


def _resource_marginal_signal(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    y = _flat_load(inst, h)
    delay = 2.0 * inst.alpha[:, None] * y**2 / inst.capacity**2
    reliability = _reliability_externality_signal(inst, h)
    return delay + reliability


def _selected_signal(parameterization: PriceParameterization, signal: np.ndarray) -> np.ndarray:
    return signal[parameterization.resource_indices, parameterization.periods].astype(
        float, copy=True
    )


def _component_price_start(
    inst: SyntheticInstance, parameterization: PriceParameterization, h_so: np.ndarray
) -> np.ndarray:
    y = _flat_load(inst, h_so)
    delay_signal = 2.0 * inst.alpha[:, None] * y**2 / inst.capacity**2
    return _selected_signal(parameterization, delay_signal)


def _potential_hessian(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    y = _flat_load(inst, h)
    excess = np.maximum(y / inst.capacity - _reliability_threshold(inst), 0.0)
    slope = (
        2.0 * inst.alpha[:, None] * y / inst.capacity**2
        + 2.0 * inst.rho_late * excess / inst.capacity
    )
    occupation = inst.delta.reshape(inst.n_resources * inst.n_res, inst.n_alt)
    return occupation.T @ (slope.reshape(-1, 1) * occupation)


def _best_in_class_gradient(
    inst: SyntheticInstance, h: np.ndarray, design: np.ndarray, ridge: float = 1e-08
) -> np.ndarray:
    """Differentiate the selected priced equilibrium on its current active set."""
    _, _, _, a_ub, b_ub = _constraints(inst)
    free = h > 1e-07
    if not np.any(free):
        return np.zeros(design.shape[1])
    slack = b_ub - a_ub @ h
    active_capacity = slack <= 2e-06 * np.maximum(1.0, np.abs(b_ub))
    demand = np.zeros((inst.n_od, inst.n_alt))
    for j, w in enumerate(inst.od_of_alt):
        demand[w, j] = 1.0
    rows = [demand[:, free]]
    if np.any(active_capacity):
        rows.append(a_ub[active_capacity][:, free])
    linearized_constraints = np.vstack(rows)
    hessian = _potential_hessian(inst, h)[np.ix_(free, free)]
    scale = max(float(np.max(np.diag(hessian))) if hessian.size else 0.0, 1.0)
    hessian = hessian + ridge * scale * np.eye(hessian.shape[0])
    n_free = int(np.sum(free))
    n_constraints = linearized_constraints.shape[0]
    kkt = np.block(
        [
            [hessian, linearized_constraints.T],
            [linearized_constraints, np.zeros((n_constraints, n_constraints))],
        ]
    )
    rhs = np.concatenate([total_cost_gradient(inst, h)[free], np.zeros(n_constraints)])
    adjoint, *_ = np.linalg.lstsq(kkt.T, rhs, rcond=1e-10)
    return -(design[free].T @ adjoint[:n_free])


def _deduplicate_starts(starts: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for start in starts:
        if not any((np.allclose(start, prior, atol=1e-09, rtol=1e-08) for prior in unique)):
            unique.append(start)
    return unique


def solve_best_in_class_price(
    inst: SyntheticInstance,
    price_class: str,
    due_h: np.ndarray | None = None,
    so_h: np.ndarray | None = None,
    max_outer_iterations: int = 30,
    max_inner_iterations: int = 260,
    multistart: int = 3,
    price_bound_factor: float = 20.0,
) -> BestInClassResult:
    """Compute a price and equilibrium flow by multistart optimization."""
    if due_h is None or so_h is None:
        due_h, so_h = solve_core_equilibria(inst, maxiter=max_inner_iterations)
    parameterization = build_price_parameterization(inst, price_class)
    design = parameterization.design
    if design.shape[1] == 0:
        raise ValueError(f"The {price_class} price class has no occupied resource-period cells.")
    full_signal = _resource_marginal_signal(inst, so_h)
    full_start = _selected_signal(parameterization, full_signal)
    component_start = _component_price_start(inst, parameterization, so_h)
    full_marginal_price = total_cost_gradient(inst, so_h) - private_cost(inst, so_h)
    projected_start = lsq_linear(
        design, full_marginal_price, bounds=(0.0, np.inf), lsmr_tol="auto", max_iter=200
    ).x
    positive_scale = max(
        float(np.max(full_start)) if full_start.size else 0.0,
        float(np.max(projected_start)) if projected_start.size else 0.0,
        1.0,
    )
    upper = price_bound_factor * positive_scale
    starts = _deduplicate_starts(
        [
            np.zeros(design.shape[1]),
            np.clip(component_start, 0.0, upper),
            np.clip(full_start, 0.0, upper),
            np.clip(projected_start, 0.0, upper),
        ]
    )
    due_cost = total_cost(inst, due_h)
    so_cost = total_cost(inst, so_h)
    denominator = max(due_cost - so_cost, 1e-09)
    candidates: list[tuple[float, np.ndarray, np.ndarray, bool, str, int, int]] = []

    def evaluate(theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, bool]:
        alternative_price = design @ theta
        h, inner_ok = _solve_priced_due(inst, alternative_price, due_h, max_inner_iterations)
        objective = total_cost(inst, h)
        gradient = _best_in_class_gradient(inst, h, design)
        return (objective, gradient, h, inner_ok)

    evaluated_starts: list[tuple[float, np.ndarray, np.ndarray, bool]] = []
    for start in starts:
        objective, gradient, h, inner_ok = evaluate(start)
        evaluated_starts.append((objective, start, h, inner_ok))
        candidates.append((objective, start, h, inner_ok, "initial candidate", 0, 1))
    optimized_starts = sorted(evaluated_starts, key=lambda item: item[0])[: max(1, multistart)]
    objective_scale = max(abs(due_cost), 1.0)
    for _, start, _, _ in optimized_starts:
        cache: dict[str, object] = {}

        def objective_with_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
            prior = cache.get("theta")
            if prior is None or not np.array_equal(theta, prior):
                value, gradient, h, inner_ok = evaluate(theta)
                cache["theta"] = theta.copy()
                cache["value"] = value
                cache["gradient"] = gradient
                cache["flow"] = h
                cache["inner_ok"] = inner_ok
            return (
                float(cache["value"]) / objective_scale,
                np.asarray(cache["gradient"]) / objective_scale,
            )

        optimized = minimize(
            objective_with_gradient,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, upper)] * design.shape[1],
            options={"maxiter": max_outer_iterations, "ftol": 1e-10, "gtol": 2e-05, "maxls": 12},
        )
        value, _, h, inner_ok = evaluate(optimized.x)
        candidates.append(
            (
                value,
                optimized.x.copy(),
                h,
                bool(inner_ok),
                str(optimized.message),
                int(optimized.nit),
                int(optimized.nfev),
            )
        )
    best = min(candidates, key=lambda item: item[0])
    value, theta, h, inner_ok, message, iterations, evaluations = best
    alternative_price = design @ theta
    vi_residual = _vi_gap(inst, h, private_cost(inst, h) + alternative_price)
    priced_cost = private_cost(inst, h) + alternative_price
    normalized_vi_residual = vi_residual / max(1.0 + float(h @ np.abs(priced_cost)), 1.0)
    upper_fraction = float(np.mean(theta >= (1.0 - 1e-06) * upper))
    success = bool(
        inner_ok
        and np.isfinite(value)
        and np.isfinite(vi_residual)
        and (normalized_vi_residual <= 1e-05)
        and (value <= due_cost + 1e-05)
    )
    return BestInClassResult(
        price_class=price_class,
        flow=h,
        resource_prices=theta,
        alternative_price=alternative_price,
        system_cost=float(value),
        gap_closure_ratio=float((due_cost - value) / denominator),
        vi_residual=float(vi_residual),
        normalized_vi_residual=float(normalized_vi_residual),
        outer_iterations=iterations,
        objective_evaluations=evaluations,
        upper_bound_fraction=upper_fraction,
        success=success,
        message=message,
    )


def solve_best_in_class_instance(
    inst: SyntheticInstance,
    max_outer_iterations: int = 30,
    max_inner_iterations: int = 260,
    multistart: int = 3,
) -> tuple[
    dict[str, float | int | str | bool],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    due_h, so_h = solve_core_equilibria(inst, maxiter=max_inner_iterations)
    due_cost = total_cost(inst, due_h)
    so_cost = total_cost(inst, so_h)
    component_prices = _resource_tolls(inst, so_h)
    component_keys = {"corridor": "corridor", "node": "node", "contemporaneous": "contemporaneous"}
    row: dict[str, float | int | str | bool] = {
        "Seed": inst.config.seed,
        "Bottleneck type": inst.config.bottleneck_type,
        "Pressure target": inst.config.pressure,
        "Reliability": inst.config.reliability,
        "Alternatives": inst.n_alt,
        "Resources": inst.n_resources,
        "DUE cost": due_cost,
        "SO cost": so_cost,
        "PoA": due_cost / max(so_cost, 1e-12),
    }
    resource_price_records: list[dict[str, float | int | str]] = []
    alternative_records: list[dict[str, float | int | str]] = []
    all_success = True
    for price_class in PRICE_CLASSES:
        component_h, component_ok = _solve_priced_due(
            inst, component_prices[component_keys[price_class]], due_h, max_inner_iterations
        )
        component_cost = total_cost(inst, component_h)
        result = solve_best_in_class_price(
            inst,
            price_class,
            due_h=due_h,
            so_h=so_h,
            max_outer_iterations=max_outer_iterations,
            max_inner_iterations=max_inner_iterations,
            multistart=multistart,
        )
        parameterization = build_price_parameterization(inst, price_class)
        for parameter, resource, period, resource_type in zip(
            result.resource_prices,
            parameterization.resource_indices,
            parameterization.periods,
            parameterization.resource_types,
        ):
            resource_price_records.append(
                {
                    "Seed": inst.config.seed,
                    "Price class": price_class,
                    "Resource": int(resource),
                    "Resource type": str(resource_type),
                    "Period": int(period),
                    "Price": float(parameter),
                }
            )
        for alternative, (w, p, t, price, flow) in enumerate(
            zip(
                inst.od_of_alt,
                inst.path_of_alt,
                inst.dep_of_alt,
                result.alternative_price,
                result.flow,
            )
        ):
            alternative_records.append(
                {
                    "Seed": inst.config.seed,
                    "Price class": price_class,
                    "Alternative": int(alternative),
                    "OD": int(w),
                    "Path": int(p),
                    "Departure period": int(t),
                    "Alternative price": float(price),
                    "Priced DUE flow": float(flow),
                }
            )
        prefix = price_class.capitalize()
        row[f"{prefix} component GCR"] = (due_cost - component_cost) / max(
            due_cost - so_cost, 1e-09
        )
        row[f"{prefix} stationary price GCR"] = result.gap_closure_ratio
        row[f"{prefix} computed remaining gap"] = 1.0 - result.gap_closure_ratio
        row[f"{prefix} improvement over component price"] = (
            result.gap_closure_ratio - row[f"{prefix} component GCR"]
        )
        row[f"{prefix} VI residual"] = result.vi_residual
        row[f"{prefix} normalized VI residual"] = result.normalized_vi_residual
        row[f"{prefix} iterations"] = result.outer_iterations
        row[f"{prefix} evaluations"] = result.objective_evaluations
        row[f"{prefix} upper-bound fraction"] = result.upper_bound_fraction
        row[f"{prefix} success"] = bool(result.success and component_ok)
        all_success = all_success and bool(result.success and component_ok)
    row["success"] = all_success
    return (row, resource_price_records, alternative_records)


def run_illustrative_case(
    out_dir: Path = OUT,
    max_outer_iterations: int = 40,
    max_inner_iterations: int = 300,
    multistart: int = 4,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    inst, _ = build_metropolitan_corridor_instance(seed=20260625)
    row, resource_prices, alternatives = solve_best_in_class_instance(
        inst,
        max_outer_iterations=max_outer_iterations,
        max_inner_iterations=max_inner_iterations,
        multistart=multistart,
    )
    frame = pd.DataFrame([row])
    frame.to_csv(out_dir / "best_in_class_illustrative_case.csv", index=False)
    pd.DataFrame(resource_prices).to_csv(
        out_dir / "best_in_class_illustrative_prices.csv", index=False
    )
    pd.DataFrame(alternatives).to_csv(
        out_dir / "best_in_class_illustrative_alternatives.csv", index=False
    )
    return frame


def run_constructed_batch(
    out_dir: Path = OUT,
    instances: int = 200,
    seed: int = 20260625,
    max_outer_iterations: int = 24,
    max_inner_iterations: int = 240,
    multistart: int = 2,
    progress_every: int = 10,
    workers: int = 1,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    resource_price_rows: list[dict[str, float | int | str]] = []
    alternative_rows: list[dict[str, float | int | str]] = []
    configs = _config_stream(instances, seed)
    arguments = [(cfg, max_outer_iterations, max_inner_iterations, multistart) for cfg in configs]
    if workers <= 1:
        iterator = map(_solve_config, arguments)
        for index, solved in enumerate(iterator, start=1):
            row, prices, alternatives = solved
            rows.append(row)
            resource_price_rows.extend(prices)
            alternative_rows.extend(alternatives)
            if progress_every and (index % progress_every == 0 or index == instances):
                successful = sum((bool(item.get("success")) for item in rows))
                print(
                    {"processed": index, "successful": successful, "target": instances}, flush=True
                )
    else:
        ordered: dict[
            int,
            tuple[
                dict[str, float | int | str | bool],
                list[dict[str, float | int | str]],
                list[dict[str, float | int | str]],
            ],
        ] = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_solve_config, argument): index
                for index, argument in enumerate(arguments)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                ordered[futures[future]] = future.result()
                if progress_every and (completed % progress_every == 0 or completed == instances):
                    successful = sum((bool(item[0].get("success")) for item in ordered.values()))
                    print(
                        {"processed": completed, "successful": successful, "target": instances},
                        flush=True,
                    )
        for index in range(instances):
            row, prices, alternatives = ordered[index]
            rows.append(row)
            resource_price_rows.extend(prices)
            alternative_rows.extend(alternatives)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "best_in_class_constructed_instances.csv", index=False)
    pd.DataFrame(resource_price_rows).to_csv(
        out_dir / "best_in_class_resource_prices.csv", index=False
    )
    pd.DataFrame(alternative_rows).to_csv(
        out_dir / "best_in_class_alternative_allocations.csv", index=False
    )
    return frame


def _solve_config(
    arguments: tuple[object, int, int, int],
) -> tuple[
    dict[str, float | int | str | bool],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    cfg, max_outer_iterations, max_inner_iterations, multistart = arguments
    try:
        inst = generate_instance(cfg)
        return solve_best_in_class_instance(
            inst,
            max_outer_iterations=max_outer_iterations,
            max_inner_iterations=max_inner_iterations,
            multistart=multistart,
        )
    except Exception as exc:
        return (
            {
                "Seed": cfg.seed,
                "Bottleneck type": cfg.bottleneck_type,
                "Pressure target": cfg.pressure,
                "Reliability": cfg.reliability,
                "success": False,
                "error": str(exc),
            },
            [],
            [],
        )


def _set_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_best_in_class_figures(
    illustrative: pd.DataFrame,
    constructed: pd.DataFrame,
    out_dir: Path = OUT,
    illustrative_minimum_gap: pd.DataFrame | None = None,
) -> None:
    _set_figure_style()
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    classes = ["Corridor", "Node", "Contemporaneous"]
    labels = ["Corridor", "Node", "Contemporaneous"]
    colors = ["#2F5D8C", "#8A4B2A", "#4F6B4F"]
    row = illustrative.iloc[0]
    component = np.array([row[f"{name} component GCR"] for name in classes])
    stationary = np.array([row[f"{name} stationary price GCR"] for name in classes])
    x = np.arange(len(classes))
    width = 0.34
    fig, ax = plt.subplots(figsize=(4.65, 3.45), constrained_layout=True)
    ax.bar(x - width / 2, component, width, color="#A7B8C9", label="Component price")
    ax.bar(x + width / 2, stationary, width, color="#2F5D8C", label="Computed price")
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle=":")
    ax.set_xticks(x, labels)
    ax.set_ylabel("DUE--SO gap closure ratio", fontsize=10.8)
    ax.set_ylim(0.0, 1.06)
    ax.tick_params(axis="x", labelsize=10.8)
    ax.tick_params(axis="y", labelsize=9.6)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        fontsize=10.6,
        handlelength=1.45,
        columnspacing=1.0,
    )
    fig.savefig(figure_dir / "best_in_class_illustrative_comparison.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "best_in_class_illustrative_comparison.png", bbox_inches="tight")
    plt.close(fig)
    ok = constructed[constructed["success"] == True].copy()
    long_rows: list[pd.DataFrame] = []
    for name, label in zip(classes, labels):
        part = ok[
            [
                "Bottleneck type",
                f"{name} component GCR",
                f"{name} stationary price GCR",
                f"{name} computed remaining gap",
                f"{name} improvement over component price",
            ]
        ].copy()
        part.columns = [
            "Capacity configuration",
            "Component GCR",
            "Stationary price GCR",
            "Computed remaining gap",
            "Improvement over component price",
        ]
        part["Price class"] = label
        long_rows.append(part)
    long = pd.concat(long_rows, ignore_index=True)
    long.to_csv(out_dir / "best_in_class_plot_data.csv", index=False)
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(4.65, 3.45), constrained_layout=True)
    rng = np.random.default_rng(20260825)
    for idx, (label, color) in enumerate(zip(labels, colors)):
        values = long.loc[long["Price class"] == label, "Stationary price GCR"].to_numpy()
        jitter = positions[idx] + rng.uniform(-0.105, 0.105, size=values.size)
        ax.scatter(jitter, values, s=8, alpha=0.14, color=color, edgecolors="none", zorder=1)
        violin = ax.violinplot(
            values,
            positions=[positions[idx]],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.24)
            body.set_linewidth(1.1)
            body.set_zorder(2)
        ax.boxplot(
            values,
            positions=[positions[idx]],
            widths=0.24,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.05},
            medianprops={"color": color, "linewidth": 1.35},
            whiskerprops={"color": "#333333", "linewidth": 0.9},
            capprops={"color": "#333333", "linewidth": 0.9},
            zorder=3,
        )
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle=":")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Gap closure ratio", fontsize=10.8)
    ax.tick_params(axis="x", labelsize=10.8)
    ax.tick_params(axis="y", labelsize=9.6)
    observed_minimum = long["Stationary price GCR"].min()
    lower_limit = max(0.0, np.floor((observed_minimum - 0.02) * 20.0) / 20.0)
    ax.set_ylim(lower_limit, 1.02)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    fig.savefig(figure_dir / "best_in_class_gcr_distribution.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "best_in_class_gcr_distribution.png", bbox_inches="tight")
    plt.close(fig)
    grouped = long.groupby(["Capacity configuration", "Price class"], as_index=False)[
        ["Computed remaining gap", "Improvement over component price"]
    ].mean()
    order = BOTTLENECK_TYPES
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharey=True)
    for ax, label in zip(axes, labels):
        part = (
            grouped[grouped["Price class"] == label]
            .set_index("Capacity configuration")
            .reindex(order)
            .fillna(0.0)
        )
        remaining_gap = np.array([part.loc[item, "Computed remaining gap"] for item in order])
        achieved_improvement = np.array(
            [part.loc[item, "Improvement over component price"] for item in order]
        )
        x = np.arange(len(order))
        ax.bar(x, remaining_gap, color="#2F5D8C", alpha=0.88, label="Computed remaining gap")
        ax.bar(
            x,
            achieved_improvement,
            bottom=remaining_gap,
            color="#C7C7C7",
            label="Improvement over component price",
        )
        ax.set_title(label)
        ax.set_xticks(x, ["Corridor", "Node", "Mixed", "Delayed node"], rotation=42, ha="right")
        ax.axhline(0.0, color="#555555", linewidth=0.7)
    axes[0].set_ylabel("Average remaining gap ratio")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.subplots_adjust(top=0.78, bottom=0.3, left=0.09, right=0.99, wspace=0.08)
    fig.savefig(figure_dir / "best_in_class_loss_decomposition.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "best_in_class_loss_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    illustrative: pd.DataFrame, constructed: pd.DataFrame, out_dir: Path = OUT
) -> pd.DataFrame:
    ok = constructed[constructed["success"] == True].copy()
    records = []
    for name in ("Corridor", "Node", "Contemporaneous"):
        records.append(
            {
                "Price class": name,
                "Successful instances": len(ok),
                "Mean component GCR": ok[f"{name} component GCR"].mean(),
                "Mean stationary price GCR": ok[f"{name} stationary price GCR"].mean(),
                "Median stationary price GCR": ok[f"{name} stationary price GCR"].median(),
                "Mean computed remaining gap": ok[f"{name} computed remaining gap"].mean(),
                "Mean improvement over component price": ok[
                    f"{name} improvement over component price"
                ].mean(),
                "Maximum VI residual": ok[f"{name} VI residual"].max(),
                "Maximum normalized VI residual": ok[f"{name} normalized VI residual"].max(),
                "Maximum upper-bound fraction": ok[f"{name} upper-bound fraction"].max(),
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(out_dir / "best_in_class_summary.csv", index=False)
    return summary


def reassess_saved_results(out_dir: Path = OUT, seed: int = 20260625) -> pd.DataFrame:
    """Recompute scale-adjusted VI residuals from saved prices and equilibrium flows."""
    instance_path = out_dir / "best_in_class_constructed_instances.csv"
    allocation_path = out_dir / "best_in_class_alternative_allocations.csv"
    results = pd.read_csv(instance_path)
    allocations = pd.read_csv(allocation_path)
    configurations = {cfg.seed: cfg for cfg in _config_stream(len(results), seed)}
    for row_index, row in results.iterrows():
        instance_seed = int(row["Seed"])
        inst = generate_instance(configurations[instance_seed])
        instance_allocations = allocations[allocations["Seed"] == instance_seed]
        class_success = []
        for price_class in PRICE_CLASSES:
            selected = instance_allocations[
                instance_allocations["Price class"] == price_class
            ].sort_values("Alternative")
            h = selected["Priced DUE flow"].to_numpy(dtype=float)
            price = selected["Alternative price"].to_numpy(dtype=float)
            cost = private_cost(inst, h) + price
            residual = _vi_gap(inst, h, cost)
            normalized = residual / max(1.0 + float(h @ np.abs(cost)), 1.0)
            prefix = price_class.capitalize()
            results.at[row_index, f"{prefix} VI residual"] = residual
            results.at[row_index, f"{prefix} normalized VI residual"] = normalized
            successful = bool(np.isfinite(normalized) and normalized <= 1e-05)
            results.at[row_index, f"{prefix} success"] = successful
            class_success.append(successful)
        results.at[row_index, "success"] = all(class_success)
    results.to_csv(instance_path, index=False)
    illustrative = pd.read_csv(out_dir / "best_in_class_illustrative_case.csv")
    illustrative_allocations = pd.read_csv(out_dir / "best_in_class_illustrative_alternatives.csv")
    illustrative_inst, _ = build_metropolitan_corridor_instance(seed=seed)
    illustrative_success = []
    for price_class in PRICE_CLASSES:
        selected = illustrative_allocations[
            illustrative_allocations["Price class"] == price_class
        ].sort_values("Alternative")
        h = selected["Priced DUE flow"].to_numpy(dtype=float)
        price = selected["Alternative price"].to_numpy(dtype=float)
        cost = private_cost(illustrative_inst, h) + price
        residual = _vi_gap(illustrative_inst, h, cost)
        normalized = residual / max(1.0 + float(h @ np.abs(cost)), 1.0)
        prefix = price_class.capitalize()
        illustrative.at[0, f"{prefix} VI residual"] = residual
        illustrative.at[0, f"{prefix} normalized VI residual"] = normalized
        successful = bool(np.isfinite(normalized) and normalized <= 1e-05)
        illustrative.at[0, f"{prefix} success"] = successful
        illustrative_success.append(successful)
    illustrative.at[0, "success"] = all(illustrative_success)
    illustrative.to_csv(out_dir / "best_in_class_illustrative_case.csv", index=False)
    write_summary(illustrative, results, out_dir)
    make_best_in_class_figures(illustrative, results, out_dir)
    return results
