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
from scipy.optimize import linprog
from resource_pricing.restricted_pricing import (
    PRICE_CLASSES,
    PriceParameterization,
    build_price_parameterization,
)
from resource_pricing.controlled_networks import (
    SyntheticInstance,
    _config_stream,
    _constraints,
    _projection_residual,
    _resource_tolls,
    _vi_gap,
    build_metropolitan_corridor_instance,
    generate_instance,
    private_cost,
    solve_core_equilibria,
    total_cost,
    total_cost_gradient,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/class_gaps/initial"
GAP_PRICE_CLASSES = (*PRICE_CLASSES, "all_resource")


@dataclass
class MinimumPrimalGapResult:
    price_class: str
    minimum_gap: float
    normalized_gap: float
    resource_prices: np.ndarray
    alternative_price: np.ndarray
    direct_gap: float
    duality_difference: float
    iterations: int
    success: bool
    message: str


def build_gap_price_parameterization(
    inst: SyntheticInstance, price_class: str
) -> PriceParameterization:
    """Map nonnegative resource-period parameters to alternative prices."""
    if price_class in PRICE_CLASSES:
        return build_price_parameterization(inst, price_class)
    if price_class != "all_resource":
        raise ValueError(f"Unknown price class: {price_class}")
    flat_delta = inst.delta.reshape(inst.n_resources * inst.n_res, inst.n_alt).T
    resource_indices = np.repeat(np.arange(inst.n_resources), inst.n_res)
    periods = np.tile(np.arange(inst.n_res), inst.n_resources)
    used = np.any(np.abs(flat_delta) > 1e-12, axis=0)
    return PriceParameterization(
        price_class=price_class,
        design=flat_delta[:, used],
        resource_indices=resource_indices[used],
        periods=periods[used],
        resource_types=inst.resource_types[resource_indices[used]],
    )


def solve_minimum_price_class_primal_gap(
    inst: SyntheticInstance, h_so: np.ndarray, price_class: str
) -> MinimumPrimalGapResult:
    """Solve the finite-dimensional LP for the minimum class primal gap."""
    parameterization = build_gap_price_parameterization(inst, price_class)
    design = parameterization.design
    if design.shape[1] == 0:
        raise ValueError(f"The {price_class} price class has no occupied cells.")
    eq, _, _, capacity_matrix, capacity = _constraints(inst)
    conservation = np.asarray(eq.A, dtype=float)
    demand = np.asarray(inst.demand, dtype=float)
    private = private_cost(inst, h_so)
    n_theta = design.shape[1]
    n_capacity = capacity_matrix.shape[0]
    n_conservation = conservation.shape[0]
    objective = np.concatenate([design.T @ h_so, capacity, demand])
    inequality = np.hstack([-design, -capacity_matrix.T, -conservation.T])
    bounds = [(0.0, None)] * n_theta + [(0.0, None)] * n_capacity + [(None, None)] * n_conservation
    solved = linprog(c=objective, A_ub=inequality, b_ub=private, bounds=bounds, method="highs-ds")
    if not solved.success:
        return MinimumPrimalGapResult(
            price_class=price_class,
            minimum_gap=np.nan,
            normalized_gap=np.nan,
            resource_prices=np.full(n_theta, np.nan),
            alternative_price=np.full(inst.n_alt, np.nan),
            direct_gap=np.nan,
            duality_difference=np.nan,
            iterations=int(getattr(solved, "nit", 0)),
            success=False,
            message=str(solved.message),
        )
    theta = np.maximum(np.asarray(solved.x[:n_theta], dtype=float), 0.0)
    alternative_price = design @ theta
    minimum_gap = float(private @ h_so + solved.fun)
    if minimum_gap < 0.0 and abs(minimum_gap) <= 1e-07:
        minimum_gap = 0.0
    direct_gap = _vi_gap(inst, h_so, private + alternative_price)
    scale = max(float(total_cost(inst, h_so)), 1.0)
    return MinimumPrimalGapResult(
        price_class=price_class,
        minimum_gap=max(minimum_gap, 0.0),
        normalized_gap=max(minimum_gap, 0.0) / scale,
        resource_prices=theta,
        alternative_price=alternative_price,
        direct_gap=direct_gap,
        duality_difference=abs(minimum_gap - direct_gap),
        iterations=int(getattr(solved, "nit", 0)),
        success=bool(
            solved.success
            and np.isfinite(direct_gap)
            and (abs(minimum_gap - direct_gap) <= 1e-06 * max(1.0, abs(direct_gap)))
        ),
        message=str(solved.message),
    )


def solve_instance_minimum_gaps(
    inst: SyntheticInstance, max_iterations: int = 300
) -> tuple[dict[str, float | int | str | bool], list[dict[str, float | int | str]]]:
    due_h, h_so = solve_core_equilibria(inst, maxiter=max_iterations)
    due_cost = total_cost(inst, due_h)
    so_cost = total_cost(inst, h_so)
    marginal_price = total_cost_gradient(inst, h_so) - private_cost(inst, h_so)
    component_prices = _resource_tolls(inst, h_so)
    row: dict[str, float | int | str | bool] = {
        "Seed": int(inst.config.seed),
        "Capacity configuration": str(inst.config.bottleneck_type),
        "Pressure": float(inst.config.pressure),
        "Reliability": str(inst.config.reliability),
        "Alternatives": int(inst.n_alt),
        "Resources": int(inst.n_resources),
        "DUE cost": float(due_cost),
        "SO cost": float(so_cost),
    }
    price_records: list[dict[str, float | int | str]] = []
    all_success = True
    for price_class in GAP_PRICE_CLASSES:
        result = solve_minimum_price_class_primal_gap(inst, h_so, price_class)
        parameterization = build_gap_price_parameterization(inst, price_class)
        prefix = price_class.replace("_", " ").title().replace(" ", "")
        row[f"{prefix} minimum primal gap"] = result.minimum_gap
        row[f"{prefix} normalized minimum primal gap"] = result.normalized_gap
        row[f"{prefix} direct primal gap"] = result.direct_gap
        row[f"{prefix} LP duality difference"] = result.duality_difference
        row[f"{prefix} projection residual"] = _projection_residual(
            inst, h_so, marginal_price, price_class, normalized=True
        )
        row[f"{prefix} success"] = result.success
        all_success = all_success and result.success
        for value, resource, period, resource_type in zip(
            result.resource_prices,
            parameterization.resource_indices,
            parameterization.periods,
            parameterization.resource_types,
        ):
            price_records.append(
                {
                    "Seed": int(inst.config.seed),
                    "Price class": price_class,
                    "Resource": int(resource),
                    "Resource type": str(resource_type),
                    "Period": int(period),
                    "Price": float(value),
                }
            )
    selected_keys = {
        "corridor": "corridor",
        "node": "node",
        "contemporaneous": "contemporaneous",
        "all_resource": "delay_plus_reliability",
    }
    for price_class, component_key in selected_keys.items():
        prefix = price_class.replace("_", " ").title().replace(" ", "")
        row[f"{prefix} component primal gap"] = _vi_gap(
            inst, h_so, private_cost(inst, h_so) + component_prices[component_key]
        )
    row["success"] = all_success
    return (row, price_records)


def _solve_config(arguments: tuple[object, int]):
    config, max_iterations = arguments
    try:
        return solve_instance_minimum_gaps(generate_instance(config), max_iterations=max_iterations)
    except Exception as exc:
        return (
            {
                "Seed": int(config.seed),
                "Capacity configuration": str(config.bottleneck_type),
                "success": False,
                "error": str(exc),
            },
            [],
        )


def run_constructed_batch(
    out_dir: Path = OUT,
    instances: int = 200,
    seed: int = 20260625,
    max_iterations: int = 300,
    workers: int = 1,
    progress_every: int = 10,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    arguments = [(config, max_iterations) for config in _config_stream(instances, seed)]
    rows: list[dict[str, float | int | str | bool]] = []
    price_rows: list[dict[str, float | int | str]] = []
    if workers <= 1:
        iterator = enumerate(map(_solve_config, arguments), start=1)
        for index, (row, prices) in iterator:
            rows.append(row)
            price_rows.extend(prices)
            if progress_every and (index % progress_every == 0 or index == instances):
                print(
                    {
                        "processed": index,
                        "successful": sum((bool(item.get("success")) for item in rows)),
                        "target": instances,
                    },
                    flush=True,
                )
    else:
        ordered = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_solve_config, argument): index
                for index, argument in enumerate(arguments)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                ordered[futures[future]] = future.result()
                if progress_every and (completed % progress_every == 0 or completed == instances):
                    print(
                        {
                            "processed": completed,
                            "successful": sum(
                                (bool(item[0].get("success")) for item in ordered.values())
                            ),
                            "target": instances,
                        },
                        flush=True,
                    )
        for index in range(instances):
            row, prices = ordered[index]
            rows.append(row)
            price_rows.extend(prices)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "minimum_primal_gap_constructed_instances.csv", index=False)
    pd.DataFrame(price_rows).to_csv(out_dir / "minimum_primal_gap_resource_prices.csv", index=False)
    return frame


def run_illustrative_case(out_dir: Path = OUT, max_iterations: int = 350) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    inst, _ = build_metropolitan_corridor_instance(seed=20260625)
    row, prices = solve_instance_minimum_gaps(inst, max_iterations=max_iterations)
    frame = pd.DataFrame([row])
    frame.to_csv(out_dir / "minimum_primal_gap_illustrative_case.csv", index=False)
    pd.DataFrame(prices).to_csv(out_dir / "minimum_primal_gap_illustrative_prices.csv", index=False)
    return frame


def _set_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figures(
    results: pd.DataFrame, out_dir: Path = OUT, best_in_class_path: Path | None = None
) -> None:
    _set_figure_style()
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    valid = results[results["success"] == True].copy()
    classes = ["Corridor", "Node", "Contemporaneous", "AllResource"]
    labels = ["Corridor", "Node", "Contemporaneous", "All resource-period"]
    colors = ["#2F5D8C", "#9A5A3A", "#637A52", "#6A5D82"]
    long = []
    for name, label in zip(classes, labels):
        part = pd.DataFrame(
            {
                "Price class": label,
                "Minimum primal gap": valid[f"{name} minimum primal gap"],
                "Normalized minimum primal gap": valid[f"{name} normalized minimum primal gap"],
                "Projection residual": valid[f"{name} projection residual"],
            }
        )
        long.append(part)
    long_frame = pd.concat(long, ignore_index=True)
    long_frame.to_csv(out_dir / "minimum_primal_gap_plot_data.csv", index=False)
    tolerance = 1e-07
    nondecentralization_frequency = np.array(
        [
            np.mean(valid[f"{name} minimum primal gap"].to_numpy(dtype=float) > tolerance)
            for name in classes
        ]
    )
    positions = np.arange(len(labels))
    panel_size = (5.25, 3.75)
    panel_margins = {"left": 0.17, "right": 0.98, "bottom": 0.19, "top": 0.98}
    fig, ax = plt.subplots(figsize=panel_size)
    fig.subplots_adjust(**panel_margins)
    bars = ax.bar(
        positions, nondecentralization_frequency, width=0.68, color=colors, edgecolor="none"
    )
    for bar, value in zip(bars, nondecentralization_frequency):
        inside = value >= 0.08
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value - 0.055 if inside else 0.025,
            f"{100.0 * value:.1f}\\%" if plt.rcParams["text.usetex"] else f"{100.0 * value:.1f}%",
            ha="center",
            va="center" if inside else "bottom",
            color="white" if inside else "#222222",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(0.0, 1.04)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Non-decentralization frequency ($G_{\\mathcal{U}}^{*}>10^{-7}$)", fontsize=10.0)
    ax.tick_params(axis="x", labelsize=9.4)
    ax.tick_params(axis="y", labelsize=9.4)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    fig.savefig(figure_dir / "figure6a_minimum_price_class_primal_gap.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=panel_size)
    fig.subplots_adjust(**panel_margins)
    for name, label, color in zip(classes[:3], labels[:3], colors[:3]):
        projection = valid[f"{name} projection residual"].to_numpy(dtype=float)
        gap = valid[f"{name} minimum primal gap"].to_numpy(dtype=float)
        positive_complete_gap = gap > tolerance
        local_zero_global_positive = (projection <= tolerance) & (gap > tolerance)
        ax.scatter(
            projection[positive_complete_gap],
            gap[positive_complete_gap],
            s=17,
            alpha=0.58,
            color=color,
            edgecolors="none",
            label=label,
        )
        if np.any(local_zero_global_positive):
            ax.scatter(
                projection[local_zero_global_positive],
                gap[local_zero_global_positive],
                s=46,
                facecolors="none",
                edgecolors="#222222",
                linewidths=1.0,
            )
    ax.set_xscale("symlog", linthresh=0.001, linscale=0.6)
    ax.set_yscale("log")
    ax.set_xlim(-0.0001, 1.0)
    ax.set_ylim(0.001, 10.0)
    ax.set_xticks([0.0, 0.001, 0.01, 0.1, 1.0])
    ax.set_xlabel(
        "Normalized projection residual ($\\bar{\\mathcal{R}}_{\\mathrm{SO}}^{\\mathcal{U}}$)",
        fontsize=11.0,
    )
    ax.set_ylabel("Minimum price-class primal gap ($G_{\\mathcal{U}}^{*}$)", fontsize=10.0)
    ax.tick_params(axis="x", labelsize=10.2)
    ax.tick_params(axis="y", labelsize=9.4)
    ax.grid(color="#D7D7D7", linewidth=0.55, alpha=0.8)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=10.6,
        handletextpad=0.5,
        columnspacing=1.0,
        markerscale=1.15,
    )
    fig.savefig(figure_dir / "figure6b_projection_vs_minimum_primal_gap.pdf")
    plt.close(fig)
    if best_in_class_path is None or not best_in_class_path.exists():
        return
    best = pd.read_csv(best_in_class_path)
    merged = valid.merge(best, on="Seed", suffixes=("", " best"))
    fig, ax = plt.subplots(figsize=(6.8, 3.55), constrained_layout=True)
    for name, label, color in zip(classes[:3], labels[:3], colors[:3]):
        x = np.maximum(merged[f"{name} normalized minimum primal gap"].to_numpy(dtype=float), 1e-12)
        y = merged[f"{name} price-class loss"].to_numpy(dtype=float)
        ax.scatter(x, y, s=18, alpha=0.62, color=color, edgecolors="none", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Normalized minimum primal gap ($G_{\\mathcal{U}}^{*}/\\Phi^{\\mathrm{SO}}$)")
    ax.set_ylabel("Price-class loss")
    ax.grid(color="#D7D7D7", linewidth=0.55, alpha=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(figure_dir / "minimum_primal_gap_price_class_loss.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(results: pd.DataFrame, out_dir: Path = OUT) -> pd.DataFrame:
    valid = results[results["success"] == True].copy()
    records = []
    for name, label in zip(
        ["Corridor", "Node", "Contemporaneous", "AllResource"],
        ["Corridor", "Node", "Contemporaneous", "All resource-period"],
    ):
        values = valid[f"{name} minimum primal gap"]
        normalized = valid[f"{name} normalized minimum primal gap"]
        records.append(
            {
                "Price class": label,
                "Instances": len(valid),
                "Mean minimum primal gap": values.mean(),
                "Median minimum primal gap": values.median(),
                "Maximum minimum primal gap": values.max(),
                "Mean normalized minimum primal gap": normalized.mean(),
                "Positive-gap frequency": float(np.mean(values > 1e-07)),
                "Maximum LP duality difference": valid[f"{name} LP duality difference"].max(),
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(out_dir / "minimum_primal_gap_summary.csv", index=False)
    return summary
