from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from resource_pricing.controlled_networks import (
    SyntheticInstance,
    _config_stream,
    _solve_priced_due,
    _vi_gap,
    build_metropolitan_corridor_instance,
    generate_instance,
    private_cost,
    solve_core_equilibria,
    total_cost,
)
from resource_pricing.price_class_gap import (
    MinimumPrimalGapResult,
    solve_minimum_price_class_primal_gap,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/initialization"
PRICE_CLASSES = ("corridor", "node", "contemporaneous")


@dataclass
class EquilibriumSelectionAudit:
    price_class: str
    minimum_primal_gap: float
    due_initialized_gcr: float
    so_initialized_gcr: float
    initialization_gcr_difference: float
    due_initialized_vi_residual: float
    so_initialized_vi_residual: float
    equilibrium_flow_distance: float
    due_initialized_success: bool
    so_initialized_success: bool


def audit_decentralizing_price(
    inst: SyntheticInstance,
    h_due: np.ndarray,
    h_so: np.ndarray,
    minimum_gap: MinimumPrimalGapResult,
    max_iterations: int = 300,
) -> EquilibriumSelectionAudit:
    """Compare priced equilibria obtained from DUE and SO initializations."""
    price = minimum_gap.alternative_price
    from_due, due_ok = _solve_priced_due(inst, price, h_due, max_iterations)
    from_so, so_ok = _solve_priced_due(inst, price, h_so, max_iterations)
    due_cost = total_cost(inst, h_due)
    so_cost = total_cost(inst, h_so)
    denominator = max(due_cost - so_cost, 1e-09)
    due_initialized_gcr = (due_cost - total_cost(inst, from_due)) / denominator
    so_initialized_gcr = (due_cost - total_cost(inst, from_so)) / denominator
    due_residual = _vi_gap(inst, from_due, private_cost(inst, from_due) + price)
    so_residual = _vi_gap(inst, from_so, private_cost(inst, from_so) + price)
    total_flow = max(float(np.sum(h_so)), 1e-12)
    flow_distance = 0.5 * float(np.sum(np.abs(from_due - from_so))) / total_flow
    return EquilibriumSelectionAudit(
        price_class=minimum_gap.price_class,
        minimum_primal_gap=minimum_gap.minimum_gap,
        due_initialized_gcr=due_initialized_gcr,
        so_initialized_gcr=so_initialized_gcr,
        initialization_gcr_difference=so_initialized_gcr - due_initialized_gcr,
        due_initialized_vi_residual=due_residual,
        so_initialized_vi_residual=so_residual,
        equilibrium_flow_distance=flow_distance,
        due_initialized_success=bool(due_ok and np.isfinite(due_residual)),
        so_initialized_success=bool(so_ok and np.isfinite(so_residual)),
    )


def audit_instance(
    inst: SyntheticInstance, gap_tolerance: float = 1e-07, max_iterations: int = 300
) -> list[dict[str, float | int | str | bool]]:
    h_due, h_so = solve_core_equilibria(inst, maxiter=max_iterations)
    records: list[dict[str, float | int | str | bool]] = []
    for price_class in PRICE_CLASSES:
        minimum_gap = solve_minimum_price_class_primal_gap(inst, h_so, price_class)
        if minimum_gap.minimum_gap > gap_tolerance:
            continue
        audit = audit_decentralizing_price(
            inst, h_due, h_so, minimum_gap, max_iterations=max_iterations
        )
        record = asdict(audit)
        record.update(
            {
                "seed": int(inst.config.seed),
                "capacity_configuration": str(inst.config.bottleneck_type),
                "pressure": float(inst.config.pressure),
                "reliability": str(inst.config.reliability),
                "alternatives": int(inst.n_alt),
                "resources": int(inst.n_resources),
            }
        )
        records.append(record)
    return records


def _solve_config(arguments: tuple[object, float, int]):
    config, gap_tolerance, max_iterations = arguments
    return audit_instance(
        generate_instance(config), gap_tolerance=gap_tolerance, max_iterations=max_iterations
    )


def run_constructed_audit(
    out_dir: Path = OUT,
    instances: int = 200,
    seed: int = 20260625,
    gap_tolerance: float = 1e-07,
    max_iterations: int = 300,
    workers: int = 1,
    progress_every: int = 10,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    arguments = [
        (config, gap_tolerance, max_iterations) for config in _config_stream(instances, seed)
    ]
    records: list[dict[str, float | int | str | bool]] = []
    if workers <= 1:
        for completed, rows in enumerate(map(_solve_config, arguments), start=1):
            records.extend(rows)
            if progress_every and (completed % progress_every == 0 or completed == instances):
                print(
                    {
                        "processed": completed,
                        "target": instances,
                        "zero_gap_comparisons": len(records),
                    },
                    flush=True,
                )
    else:
        ordered: dict[int, list[dict[str, float | int | str | bool]]] = {}
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
                            "target": instances,
                            "zero_gap_comparisons": sum((len(rows) for rows in ordered.values())),
                        },
                        flush=True,
                    )
        for index in range(instances):
            records.extend(ordered[index])
    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "equilibrium_selection_constructed.csv", index=False)
    return frame


def run_illustrative_audit(
    out_dir: Path = OUT, gap_tolerance: float = 1e-07, max_iterations: int = 350
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    inst, _ = build_metropolitan_corridor_instance(seed=20260625)
    frame = pd.DataFrame(
        audit_instance(inst, gap_tolerance=gap_tolerance, max_iterations=max_iterations)
    )
    frame.to_csv(out_dir / "equilibrium_selection_illustrative.csv", index=False)
    return frame


def write_summary(frame: pd.DataFrame, out_dir: Path = OUT) -> pd.DataFrame:
    summary = frame.groupby("price_class", as_index=False).agg(
        comparisons=("seed", "size"),
        mean_due_initialized_gcr=("due_initialized_gcr", "mean"),
        mean_so_initialized_gcr=("so_initialized_gcr", "mean"),
        mean_initialization_gcr_difference=("initialization_gcr_difference", "mean"),
        positive_initialization_difference_frequency=(
            "initialization_gcr_difference",
            lambda values: float(np.mean(np.asarray(values) > 1e-05)),
        ),
        mean_flow_distance=("equilibrium_flow_distance", "mean"),
        maximum_due_initialized_vi_residual=("due_initialized_vi_residual", "max"),
        maximum_so_initialized_vi_residual=("so_initialized_vi_residual", "max"),
    )
    summary.to_csv(out_dir / "equilibrium_selection_summary.csv", index=False)
    return summary
