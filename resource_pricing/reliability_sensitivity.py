from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import numpy as np
import pandas as pd
from resource_pricing.controlled_networks import (
    StudyConfig,
    SyntheticInstance,
    _capacity_violation,
    _initial_flow,
    _minimize_flow,
    _reliability_threshold,
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
DEFAULT_SOURCE = ROOT / "data/component_prices" / "controlled_synthetic_instances.csv"
DEFAULT_OUT = ROOT / "results/reliability"
THRESHOLDS = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])


def evaluate_instance(inst: SyntheticInstance, maxiter: int = 220) -> dict[str, float | str | bool]:
    x0 = _initial_flow(inst)
    due_h, due_ok, due_message = _minimize_flow(
        inst,
        lambda x: due_potential(inst, x),
        lambda x: due_potential_gradient(inst, x),
        x0,
        maxiter,
    )
    so_h, so_ok, so_message = _minimize_flow(
        inst, lambda x: total_cost(inst, x), lambda x: total_cost_gradient(inst, x), due_h, maxiter
    )
    prices = _resource_tolls(inst, so_h)
    full_mc_price = total_cost_gradient(inst, so_h) - private_cost(inst, so_h)
    resource_price = prices["resource"]
    delay_reliability_price = prices["delay_plus_reliability"]
    resource_h, resource_ok = _solve_priced_due(inst, resource_price, due_h, maxiter)
    delay_reliability_h, delay_reliability_ok = _solve_priced_due(
        inst, delay_reliability_price, due_h, maxiter
    )
    full_mc_h, full_mc_ok = _solve_priced_due(inst, full_mc_price, due_h, maxiter)
    due_cost = total_cost(inst, due_h)
    so_cost = total_cost(inst, so_h)
    resource_cost = total_cost(inst, resource_h)
    delay_reliability_cost = total_cost(inst, delay_reliability_h)
    full_mc_cost = total_cost(inst, full_mc_h)
    denominator = max(due_cost - so_cost, 1e-09)
    resource_support_gap = _vi_gap(inst, so_h, private_cost(inst, so_h) + resource_price)
    delay_reliability_support_gap = _vi_gap(
        inst, so_h, private_cost(inst, so_h) + delay_reliability_price
    )
    full_mc_support_gap = _vi_gap(inst, so_h, private_cost(inst, so_h) + full_mc_price)
    max_capacity_violation = max(
        _capacity_violation(inst, due_h),
        _capacity_violation(inst, so_h),
        _capacity_violation(inst, resource_h),
        _capacity_violation(inst, delay_reliability_h),
        _capacity_violation(inst, full_mc_h),
    )
    max_price_difference = float(np.max(np.abs(delay_reliability_price - full_mc_price)))
    finite = np.all(
        np.isfinite(
            [
                due_cost,
                so_cost,
                resource_cost,
                delay_reliability_cost,
                full_mc_cost,
                resource_support_gap,
                delay_reliability_support_gap,
                full_mc_support_gap,
            ]
        )
    )
    success = bool(
        due_ok
        and so_ok
        and resource_ok
        and delay_reliability_ok
        and full_mc_ok
        and finite
        and (max_capacity_violation <= 1e-05)
    )
    return {
        "success": success,
        "DUE message": due_message,
        "SO message": so_message,
        "Reliability threshold": _reliability_threshold(inst),
        "DUE cost": due_cost,
        "SO cost": so_cost,
        "GCR(tauM)": (due_cost - resource_cost) / denominator,
        "GCR(tauMR)": (due_cost - delay_reliability_cost) / denominator,
        "GCR(tauMC)": (due_cost - full_mc_cost) / denominator,
        "Resource SO support gap": resource_support_gap,
        "Delay-plus-reliability SO support gap": delay_reliability_support_gap,
        "Full MC SO support gap": full_mc_support_gap,
        "Maximum capacity violation": max_capacity_violation,
        "Maximum MR-MC price difference": max_price_difference,
    }


def _config_from_row(row: pd.Series) -> StudyConfig:
    return StudyConfig(
        n_od=int(row["OD count"]),
        paths_per_od=int(row["Paths per OD"]),
        n_dep=int(row["Departure periods"]),
        bottleneck_type=str(row["Bottleneck type"]),
        pressure=float(row["Pressure target"]),
        reliability=str(row["Reliability"]),
        window_width=int(row["Window width"]),
        seed=int(row["Seed"]),
    )


def completed_network_ids(results: pd.DataFrame, thresholds: np.ndarray = THRESHOLDS) -> set[int]:
    if results.empty:
        return set()
    expected = {round(float(value), 10) for value in thresholds}
    completed: set[int] = set()
    for network_id, group in results.groupby("Network ID"):
        observed = {round(float(value), 10) for value in group["Reliability threshold"].unique()}
        if observed == expected:
            completed.add(int(network_id))
    return completed


def evaluate_network(
    network_id: int, row_data: dict[str, object], thresholds: list[float], maxiter: int
) -> list[dict[str, float | int | str | bool]]:
    row = pd.Series(row_data)
    inst = generate_instance(_config_from_row(row))
    records: list[dict[str, float | int | str | bool]] = []
    for threshold in thresholds:
        inst.reliability_threshold = float(threshold)
        result = evaluate_instance(inst, maxiter=maxiter)
        result.update(
            {
                "Network ID": network_id,
                "OD count": inst.n_od,
                "Paths per OD": inst.paths_per_od,
                "Departure periods": inst.n_dep,
                "Window width": int(row["Window width"]),
                "Bottleneck type": str(row["Bottleneck type"]),
                "Pressure target": float(row["Pressure target"]),
                "Seed": int(row["Seed"]),
                "Alternatives": inst.n_alt,
                "Resources": inst.n_resources,
            }
        )
        records.append(result)
    return records


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    successful = results.loc[results["success"]].copy()
    metrics = [
        "GCR(tauM)",
        "GCR(tauMR)",
        "GCR(tauMC)",
        "Resource SO support gap",
        "Delay-plus-reliability SO support gap",
        "Full MC SO support gap",
        "Maximum capacity violation",
        "Maximum MR-MC price difference",
    ]
    summary = successful.groupby("Reliability threshold", as_index=False)[metrics].mean()
    counts = successful.groupby("Reliability threshold").size().rename("Successful instances")
    summary = summary.merge(counts, on="Reliability threshold", how="left")
    return summary


def run_study(
    source: Path = DEFAULT_SOURCE,
    out_dir: Path = DEFAULT_OUT,
    thresholds: np.ndarray = THRESHOLDS,
    maxiter: int = 220,
    progress_every: int = 10,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = pd.read_csv(source)
    if len(source_rows) != 200:
        raise ValueError(f"Expected 200 fixed networks, found {len(source_rows)} in {source}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "reliability_threshold_instance_results.csv"
    existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    completed_ids = completed_network_ids(existing, thresholds)
    records: list[dict[str, float | int | str | bool]] = (
        existing.to_dict("records") if not existing.empty else []
    )
    total = len(source_rows) * len(thresholds)
    pending = [
        (network_id, row.to_dict())
        for network_id, (_, row) in enumerate(source_rows.iterrows(), start=1)
        if network_id not in completed_ids
    ]
    finished_networks = len(completed_ids)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                evaluate_network,
                network_id,
                row_data,
                [float(value) for value in thresholds],
                maxiter,
            ): network_id
            for network_id, row_data in pending
        }
        for future in as_completed(futures):
            records.extend(future.result())
            finished_networks += 1
            if progress_every > 0 and (
                finished_networks % progress_every == 0 or finished_networks == len(source_rows)
            ):
                interim = pd.DataFrame(records)
                interim = interim.drop_duplicates(
                    subset=["Network ID", "Reliability threshold"], keep="last"
                ).sort_values(["Network ID", "Reliability threshold"])
                interim.to_csv(results_path, index=False)
                print(
                    {
                        "networks": finished_networks,
                        "evaluations": len(interim),
                        "successful": int(interim["success"].sum()),
                        "target": total,
                    },
                    flush=True,
                )
    results = pd.DataFrame(records)
    results = results.drop_duplicates(
        subset=["Network ID", "Reliability threshold"], keep="last"
    ).sort_values(["Network ID", "Reliability threshold"])
    summary = summarize(results)
    results.to_csv(results_path, index=False)
    summary.to_csv(out_dir / "reliability_threshold_summary.csv", index=False)
    return (results, summary)
