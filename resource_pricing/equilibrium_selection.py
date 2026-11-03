from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "selection_robust_pricing_v1"


@dataclass(frozen=True)
class NonmonotoneSelectionInstance:
    name: str
    roots: tuple[float, float, float]
    interaction_scale: float
    price_bound: float

    def __post_init__(self) -> None:
        r1, r2, r3 = self.roots
        if not 0.0 < r1 < r2 < r3 < 1.0:
            raise ValueError("Equilibrium roots must be strictly ordered in (0, 1).")
        if self.interaction_scale <= 0.0 or self.price_bound <= 0.0:
            raise ValueError("Interaction scale and price bound must be positive.")


@dataclass(frozen=True)
class SelectionRobustResult:
    name: str
    optimistic_value: float
    robust_value: float
    selection_exposure: float
    optimistic_price_difference: float
    robust_price_difference: float
    equilibria_at_robust_price: tuple[float, ...]
    maximum_vi_gap: float


def private_cost_difference(
    instance: NonmonotoneSelectionInstance, flow: float, price_difference: float
) -> float:
    r1, r2, r3 = instance.roots
    return float(
        instance.interaction_scale * (flow - r1) * (flow - r2) * (flow - r3) + price_difference
    )


def system_objective(instance: NonmonotoneSelectionInstance, flow: float) -> float:
    target = instance.roots[1]
    scale = max(target * target, (1.0 - target) * (1.0 - target))
    return float((flow - target) ** 2 / scale)


def scalar_vi_gap(
    instance: NonmonotoneSelectionInstance, flow: float, price_difference: float
) -> float:
    difference = private_cost_difference(instance, flow, price_difference)
    minimum_linear_value = min(0.0, difference)
    return float(difference * flow - minimum_linear_value)


def enumerate_equilibria(
    instance: NonmonotoneSelectionInstance, price_difference: float, tolerance: float = 1e-10
) -> tuple[float, ...]:
    coefficients = instance.interaction_scale * np.poly(instance.roots)
    coefficients[-1] += price_difference
    polynomial_roots = np.roots(coefficients)
    equilibria: list[float] = []
    for root in polynomial_roots:
        if abs(float(np.imag(root))) <= tolerance:
            value = float(np.real(root))
            if -tolerance <= value <= 1.0 + tolerance:
                equilibria.append(float(np.clip(value, 0.0, 1.0)))
    if private_cost_difference(instance, 0.0, price_difference) >= -tolerance:
        equilibria.append(0.0)
    if private_cost_difference(instance, 1.0, price_difference) <= tolerance:
        equilibria.append(1.0)
    equilibria.sort()
    unique: list[float] = []
    for value in equilibria:
        if not unique or abs(value - unique[-1]) > 1e-08:
            unique.append(value)
    return tuple(unique)


def _value_at_price(
    instance: NonmonotoneSelectionInstance, price_difference: float, robust: bool
) -> float:
    equilibria = enumerate_equilibria(instance, price_difference)
    if not equilibria:
        raise RuntimeError("The scalar VI has no enumerated equilibrium.")
    values = [system_objective(instance, flow) for flow in equilibria]
    return float(max(values) if robust else min(values))


def _fold_prices(instance: NonmonotoneSelectionInstance) -> list[float]:
    derivative_roots = np.roots(np.polyder(instance.interaction_scale * np.poly(instance.roots)))
    prices: list[float] = []
    for root in derivative_roots:
        if abs(float(np.imag(root))) <= 1e-12:
            flow = float(np.real(root))
            if 0.0 < flow < 1.0:
                prices.append(-private_cost_difference(instance, flow, 0.0))
    return prices


def _minimize_class_value(
    instance: NonmonotoneSelectionInstance, robust: bool, grid_size: int
) -> tuple[float, float]:
    bound = instance.price_bound
    grid = np.linspace(-bound, bound, max(int(grid_size), 101))
    grid = np.unique(np.concatenate([grid, np.array([0.0]), np.array(_fold_prices(instance))]))
    grid = grid[(grid >= -bound) & (grid <= bound)]
    grid.sort()
    values = np.array([_value_at_price(instance, price, robust) for price in grid])
    candidates: list[tuple[float, float]] = [
        (float(values[0]), float(grid[0])),
        (float(values[-1]), float(grid[-1])),
    ]
    for index in range(1, len(grid) - 1):
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
            left = float(grid[index - 1])
            right = float(grid[index + 1])
            solved = minimize_scalar(
                lambda price: _value_at_price(instance, float(price), robust),
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-12, "maxiter": 200},
            )
            candidates.append((float(solved.fun), float(solved.x)))
            candidates.append((float(values[index]), float(grid[index])))
    return min(candidates, key=lambda item: (item[0], abs(item[1])))


def solve_class_values(
    instance: NonmonotoneSelectionInstance, grid_size: int = 4001
) -> SelectionRobustResult:
    optimistic_value, optimistic_price = _minimize_class_value(
        instance, robust=False, grid_size=grid_size
    )
    robust_value, robust_price = _minimize_class_value(instance, robust=True, grid_size=grid_size)
    robust_equilibria = enumerate_equilibria(instance, robust_price)
    maximum_vi_gap = max(
        (scalar_vi_gap(instance, flow, robust_price) for flow in robust_equilibria)
    )
    return SelectionRobustResult(
        name=instance.name,
        optimistic_value=optimistic_value,
        robust_value=robust_value,
        selection_exposure=max(robust_value - optimistic_value, 0.0),
        optimistic_price_difference=optimistic_price,
        robust_price_difference=robust_price,
        equilibria_at_robust_price=robust_equilibria,
        maximum_vi_gap=maximum_vi_gap,
    )


def selection_instances() -> list[NonmonotoneSelectionInstance]:
    specifications = [
        ("S1", (0.2, 0.5, 0.8), 1.0, 1.0),
        ("S2", (0.15, 0.45, 0.78), 1.1, 1.0),
        ("S3", (0.22, 0.48, 0.82), 0.9, 1.0),
        ("S4", (0.18, 0.52, 0.75), 1.2, 1.0),
        ("S5", (0.12, 0.42, 0.72), 0.8, 1.0),
        ("S6", (0.25, 0.55, 0.85), 1.3, 1.0),
        ("S7", (0.17, 0.5, 0.83), 1.0, 1.0),
        ("S8", (0.2, 0.43, 0.7), 1.4, 1.0),
        ("S9", (0.28, 0.53, 0.79), 1.0, 1.0),
        ("S10", (0.13, 0.47, 0.76), 1.2, 1.0),
        ("S11", (0.21, 0.57, 0.87), 0.9, 1.0),
        ("S12", (0.16, 0.39, 0.68), 1.1, 1.0),
    ]
    return [NonmonotoneSelectionInstance(*specification) for specification in specifications]


def _set_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def make_selection_robust_figure(
    instances: list[NonmonotoneSelectionInstance], values: pd.DataFrame, out_dir: Path
) -> None:
    _set_figure_style()
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    representative = instances[0]
    display_bound = min(representative.price_bound, 0.004)
    price_grid = np.linspace(-display_bound, display_bound, 401)
    branch_prices: list[float] = []
    branch_flows: list[float] = []
    for price in price_grid:
        for flow in enumerate_equilibria(representative, float(price)):
            branch_prices.append(float(price))
            branch_flows.append(flow)
    branch_fig, branch_ax = plt.subplots(figsize=(3.45, 3.05), constrained_layout=True)
    branch_ax.scatter(branch_prices, branch_flows, s=7, color="#2F5D8C", alpha=0.78)
    robust_price = float(values.iloc[0]["Selection-robust price difference"])
    branch_ax.axvline(robust_price, color="#A84A2A", linewidth=1.2)
    branch_ax.axhline(representative.roots[1], color="#555555", linewidth=0.8, linestyle=":")
    branch_ax.set_xlabel("Resource-period price difference ($\\delta$)")
    branch_ax.set_ylabel("Equilibrium flow on alternative 1 ($x$)")
    branch_fig.savefig(
        figure_dir / "selection_robust_equilibrium_branches.pdf", bbox_inches="tight"
    )
    branch_fig.savefig(
        figure_dir / "selection_robust_equilibrium_branches.png", bbox_inches="tight"
    )
    plt.close(branch_fig)
    positions = np.arange(len(values))
    optimistic = values["Optimistic class value"].to_numpy(dtype=float)
    robust = values["Selection-robust class value"].to_numpy(dtype=float)
    value_fig, value_ax = plt.subplots(figsize=(3.45, 3.05), constrained_layout=True)
    for position, low, high in zip(positions, optimistic, robust):
        value_ax.plot([position, position], [low, high], color="#9A9A9A", linewidth=1.2)
    value_ax.scatter(
        positions, optimistic, s=25, color="#2F5D8C", label="Optimistic class value", zorder=3
    )
    value_ax.scatter(
        positions, robust, s=25, color="#A84A2A", label="Selection-robust class value", zorder=3
    )
    value_ax.set_xticks(positions, values["Instance"].tolist(), rotation=45, ha="right")
    value_ax.set_ylabel("Normalized system objective")
    value_ax.legend(
        frameon=False,
        fontsize=7.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    value_fig.savefig(
        figure_dir / "selection_robust_class_value_comparison.pdf", bbox_inches="tight"
    )
    value_fig.savefig(
        figure_dir / "selection_robust_class_value_comparison.png", bbox_inches="tight"
    )
    plt.close(value_fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    axes[0].scatter(branch_prices, branch_flows, s=7, color="#2F5D8C", alpha=0.78)
    axes[0].axvline(robust_price, color="#A84A2A", linewidth=1.2)
    axes[0].axhline(representative.roots[1], color="#555555", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("Resource-period price difference ($\\delta$)")
    axes[0].set_ylabel("Equilibrium flow on alternative 1 ($x$)")
    axes[0].set_title("Equilibrium branches")
    for position, low, high in zip(positions, optimistic, robust):
        axes[1].plot([position, position], [low, high], color="#9A9A9A", linewidth=1.2)
    axes[1].scatter(
        positions, optimistic, s=25, color="#2F5D8C", label="Optimistic class value", zorder=3
    )
    axes[1].scatter(
        positions, robust, s=25, color="#A84A2A", label="Selection-robust class value", zorder=3
    )
    axes[1].set_xticks(positions, values["Instance"].tolist(), rotation=45, ha="right")
    axes[1].set_ylabel("Normalized system objective")
    axes[1].set_title("Class values")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(figure_dir / "selection_robust_class_values.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "selection_robust_class_values.png", bbox_inches="tight")
    plt.close(fig)


def run_selection_robust_study(
    out_dir: Path = OUT, instances: int = 12, grid_size: int = 4001
) -> tuple[pd.DataFrame, dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = selection_instances()[:instances]
    records: list[dict[str, float | int | str]] = []
    for instance in selected:
        result = solve_class_values(instance, grid_size=grid_size)
        records.append(
            {
                "Instance": instance.name,
                "Root 1": instance.roots[0],
                "SO root": instance.roots[1],
                "Root 3": instance.roots[2],
                "Interaction scale": instance.interaction_scale,
                "Price bound": instance.price_bound,
                "Optimistic class value": result.optimistic_value,
                "Selection-robust class value": result.robust_value,
                "Selection exposure": result.selection_exposure,
                "Optimistic price difference": result.optimistic_price_difference,
                "Selection-robust price difference": result.robust_price_difference,
                "Equilibria at robust price": len(result.equilibria_at_robust_price),
                "Maximum VI gap": result.maximum_vi_gap,
            }
        )
    values = pd.DataFrame(records)
    summary = {
        "instances": float(len(values)),
        "mean_optimistic_class_value": float(values["Optimistic class value"].mean()),
        "mean_selection_robust_class_value": float(values["Selection-robust class value"].mean()),
        "minimum_selection_robust_class_value": float(values["Selection-robust class value"].min()),
        "maximum_selection_robust_class_value": float(values["Selection-robust class value"].max()),
        "mean_selection_exposure": float(values["Selection exposure"].mean()),
        "positive_selection_exposure_frequency": float(
            (values["Selection exposure"] > 1e-08).mean()
        ),
        "maximum_vi_gap": float(values["Maximum VI gap"].max()),
    }
    values.to_csv(out_dir / "selection_robust_values.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "selection_robust_summary.csv", index=False)
    make_selection_robust_figure(selected, values, out_dir)
    return (values, summary)
