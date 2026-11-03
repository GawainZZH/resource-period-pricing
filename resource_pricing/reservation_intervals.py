from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from resource_pricing.controlled_networks import build_metropolitan_corridor_instance
from resource_pricing.nominal_reservations import (
    build_operational_case,
    knots_to_km_min,
    reservation_slots,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/nominal_reservations"


@dataclass(frozen=True)
class Stage:
    name: str
    resource: int
    start: float
    end: float
    nominal: float
    advances: bool


def reconstruct(grid):
    case = build_operational_case(grid_minutes=grid)
    _, meta = build_metropolitan_corridor_instance()
    nc = case.metadata["corridor_edges"]
    edge_times = np.zeros(nc)
    for edge, idx in meta["edge_to_idx"].items():
        a = meta["zone_coords"][edge[1]] if edge[0] == "z" else meta["merge_coords"][edge[1]]
        b = meta["zone_coords"][edge[3]] if edge[2] == "z" else meta["merge_coords"][edge[3]]
        edge_times[idx] = 5 * np.linalg.norm(a - b) / knots_to_km_min(120)
    routes = []
    for j, (_, _, t) in enumerate(case.choices):
        origin, merge, dest, ci, co, _ = meta["path_edges"][j]
        now = float(6 * t)
        stages = []
        specifications = [
            ("origin", 2 * nc + origin, 6, True),
            ("inbound_entry", ci, 2, False),
            ("inbound", nc + ci, edge_times[ci], True),
            ("merge", 2 * nc + 12 + merge, 2, True),
            ("outbound_entry", co, 2, False),
            ("outbound", nc + co, edge_times[co], True),
            ("destination", 2 * nc + dest, 6, True),
        ]
        for name, resource, nominal, advance in specifications:
            first, last = reservation_slots(now, nominal, grid)
            stages.append(Stage(name, resource, first * grid, last * grid, float(nominal), advance))
            if advance:
                now = last * grid
        routes.append(stages)
    return (case, routes)


def containment_deficit(stage, delay):
    return max(0.0, stage.nominal + delay - (stage.end - stage.start))


def propagate(stages, delays):
    """Hypothetical serial execution; not an endogenous network loading model."""
    starts, ends = ([], [])
    previous = stages[0].start
    for stage, delay in zip(stages, delays):
        start = max(previous, stage.start)
        end = start + stage.nominal + delay
        starts.append(start)
        ends.append(end)
        previous = end
    return (np.array(starts), np.array(ends))


def delay_rows(problem, flow):
    return problem.alpha * (problem.occupation @ flow / problem.capacity) ** 2


def stage_delays(case, routes, flow):
    rows = delay_rows(case.problem, flow)
    values = np.zeros((len(routes), 7))
    for j, route in enumerate(routes):
        for k, stage in enumerate(route):
            selected = (
                (case.row_resource == stage.resource)
                & (case.row_start >= stage.start)
                & (case.row_start < stage.end)
            )
            values[j, k] = rows[selected].sum()
    return values


def diagnose(case, routes, flow, label, tolerance=1e-08):
    flow = np.asarray(flow)
    if flow.shape != case.problem.base.shape or not np.all(np.isfinite(flow)):
        raise ValueError("A finite saved flow with the original dimensions is required")
    violations = case.problem.feasibility(flow)
    if max(violations.values()) > tolerance:
        raise ValueError(f"Saved flow fails reservation feasibility: {violations}")
    mass = flow.sum()
    active = flow > tolerance
    delays = stage_delays(case, routes, flow)
    np.testing.assert_allclose(
        delays.sum(axis=1), case.problem.occupation.T @ delay_rows(case.problem, flow), atol=1e-11
    )
    details = []
    padding = np.zeros(len(flow))
    deficits = np.zeros_like(delays)
    propagation_lateness = np.zeros(len(flow))
    for j, route in enumerate(routes):
        for k, stage in enumerate(route):
            slack = stage.end - stage.start - stage.nominal
            deficits[j, k] = containment_deficit(stage, delays[j, k])
            if stage.advances:
                padding[j] += slack
            details.append(
                {
                    "allocation": label,
                    "choice": j,
                    "stage": stage.name,
                    "resource": stage.resource,
                    "flow": flow[j],
                    "positive_flow": bool(active[j]),
                    "start_minutes": stage.start,
                    "end_minutes": stage.end,
                    "nominal_minutes": stage.nominal,
                    "rounding_slack_minutes": slack,
                    "delay_minutes": delays[j, k],
                    "local_deficit_minutes": deficits[j, k],
                }
            )
        serial = [route[k] for k in (0, 2, 3, 5, 6)]
        serial_delay = delays[j, [0, 2, 3, 5, 6]].copy()
        serial_delay[1] += delays[j, 1]
        serial_delay[3] += delays[j, 4]
        _, ends = propagate(serial, serial_delay)
        propagation_lateness[j] = max(0, ends[-1] - route[-1].end)
    endpoint_excess = np.maximum(delays.sum(axis=1) - padding, 0)
    weighted = lambda values: float(flow @ values / mass)
    affected = lambda values: float(flow[active & (values > tolerance)].sum() / mass)
    summary = {
        "allocation": label,
        "grid_minutes": case.grid,
        "flights_mass": float(mass),
        "positive_choices": int(active.sum()),
        "recomputed_cost": case.problem.cost(flow),
        **violations,
        "weighted_rounding_padding_minutes": weighted(padding),
        "weighted_delay_minutes": weighted(delays.sum(axis=1)),
        "origin_deficit_weighted_minutes": weighted(deficits[:, 0]),
        "destination_deficit_weighted_minutes": weighted(deficits[:, 6]),
        "origin_deficit_flow_share": affected(deficits[:, 0]),
        "destination_deficit_flow_share": affected(deficits[:, 6]),
        "any_local_deficit_flow_share": affected(deficits.max(axis=1)),
        "nominal_arrival_excess_flow_share": affected(endpoint_excess),
        "nominal_arrival_excess_max_minutes": float(endpoint_excess[active].max()),
        "serial_finish_excess_weighted_minutes": weighted(propagation_lateness),
        "serial_finish_excess_max_minutes": float(propagation_lateness[active].max()),
    }
    return (summary, details)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def growth_loss_bound(gap, growth, gap_lipschitz, power):
    if (
        not np.all(np.isfinite([gap, growth, gap_lipschitz, power]))
        or gap < 0
        or min(growth, gap_lipschitz) <= 0
        or (power < 1)
    ):
        raise ValueError("Invalid growth bound arguments")
    return float(growth * (gap / gap_lipschitz) ** power)


def conservative_constants(case, kind, price_cap):
    p = case.problem
    m = len(p.capacity)
    residual_threshold = max(1 - p.threshold, 0)
    fmax = p.alpha + p.rho * residual_threshold**2
    gradmax = 3 * p.alpha + p.rho * (residual_threshold**2 + 2 * residual_threshold)
    derivative = (2 * p.alpha + 2 * p.rho * residual_threshold) / p.capacity
    selected = case.resource_kind[case.row_resource] == kind
    pmax = case.grid * price_cap * np.sqrt(selected.sum())
    B = (
        np.linalg.norm(gradmax)
        + np.linalg.norm(fmax)
        + pmax
        + np.linalg.norm(p.capacity) * max(derivative)
    )
    kappa = np.min(p.alpha / p.capacity**2) / np.sqrt(m)
    return (float(kappa), float(B))


def sharper_constants(case, so, kind, price_cap):
    p = case.problem
    y = p.occupation @ so
    u = y / p.capacity
    a = p.alpha / p.capacity**2
    if np.any(a <= 0):
        raise ValueError("Positive cubic coefficients are required")
    marginal = 2 * p.alpha * u**2 + 2 * p.rho * u * np.maximum(u - p.threshold, 0)
    selected = case.resource_kind[case.row_resource] == kind
    upper = case.grid * price_cap * selected
    error = np.maximum(np.abs(marginal), np.abs(marginal - upper))
    slope = max((2 * p.alpha + 2 * p.rho * max(1 - p.threshold, 0)) / p.capacity)
    diameter = np.linalg.norm(p.capacity)
    positive = y > 0
    weighted = np.linalg.norm(error[positive] / np.sqrt(2 * a[positive] * y[positive]))
    return {
        "sharp_kappa3": float(1 / np.linalg.norm(1 / a)),
        "sharp_L": float(np.linalg.norm(error) + diameter * slope),
        "quadratic_denominator": float((weighted + diameter * np.sqrt(slope)) ** 2),
        "minimum_positive_load_raw": float(y[positive].min()),
    }


def run(output):
    saved_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    required = {
        ROOT / "experiments/operational_reservation_validation.py": saved_manifest["script_sha256"]
    }
    required.update(
        {
            ROOT / "experiments" / name: digest
            for name, digest in saved_manifest["dependency_sha256"].items()
        }
    )
    for path, expected in required.items():
        if sha(path).lower() != expected.lower():
            raise ValueError(f"Source differs from saved execution: {path}")
    reference = pd.read_csv(SOURCE / "operational_results.csv")
    summaries, all_details, files, bound_rows, analytical = ([], [], {}, [], [])
    gaps = pd.read_csv(SOURCE / "coverage_gaps.csv")
    for name, grid in (("baseline", 3), ("grid_1.5", 1.5)):
        case, routes = reconstruct(grid)
        saved_meta = saved_manifest["metadata"][name]
        if case.metadata != saved_meta:
            raise ValueError(f"Reconstructed metadata differs: {name}")
        reconstructed = np.zeros_like(case.problem.occupation)
        for j, route in enumerate(routes):
            for stage in route:
                selected = (
                    (case.row_resource == stage.resource)
                    & (case.row_start >= stage.start)
                    & (case.row_start < stage.end)
                )
                reconstructed[selected, j] = 1
        np.testing.assert_array_equal(reconstructed, case.problem.occupation)
        for allocation in ("due", "so", "corridor_priced", "node_priced", "contemporaneous_priced"):
            file = SOURCE / f"{name}_{allocation}.npy"
            files[str(file.relative_to(ROOT))] = sha(file)
            h = np.load(file, allow_pickle=False)
            summary, details = diagnose(case, routes, h, f"{name}:{allocation}")
            subset = reference[reference["case"] == name]
            if allocation in ("due", "so"):
                expected = subset.iloc[0]["DUE_cost" if allocation == "due" else "SO_cost"]
            else:
                expected = subset[subset["class"] == allocation.replace("_priced", "")].iloc[0][
                    "priced_cost"
                ]
            if abs(summary["recomputed_cost"] - expected) > 1e-08:
                raise ValueError(f"Cost mismatch for {name}:{allocation}")
            summaries.append(summary)
            all_details.extend(details)
        for kind in ("corridor", "node"):
            cap = 2
            kappa, B = conservative_constants(case, kind, cap)
            row = gaps[
                (gaps["case"] == name) & (gaps["class"] == kind) & (gaps["price_cap"] == str(cap))
            ].iloc[0]
            gap = max(0, float(row["gap"]))
            epsilon = max(0, float(reference[reference["case"] == name].iloc[0]["SO_gap_upper"]))
            correction = epsilon + B * (epsilon / kappa) ** (1 / 3)
            corrected_gap = max(0, gap - correction)
            sharp = sharper_constants(
                case, np.load(SOURCE / f"{name}_so.npy", allow_pickle=False), kind, cap
            )
            bound_rows.append(
                {
                    "case": name,
                    "class": kind,
                    "price_cap": cap,
                    "approximate_target_gap": gap,
                    "growth_kappa": kappa,
                    "B_upper": B,
                    "nominal_exact_SO_formula": growth_loss_bound(gap, kappa, B, 3),
                    "SO_first_order_upper": epsilon,
                    "gap_position_correction": correction,
                    "corrected_gap": corrected_gap,
                    "position_adjusted_formula": growth_loss_bound(corrected_gap, kappa, B, 3),
                    **sharp,
                    "nominal_sharp_cubic": growth_loss_bound(
                        gap, sharp["sharp_kappa3"], sharp["sharp_L"], 3
                    ),
                    "nominal_quadratic": gap**2 / sharp["quadratic_denominator"],
                }
            )
    for cap in (0, 0.125, 0.25, 0.375, 0.5, 0.75):
        gap = max(0.5 - cap, 0)
        for kind in ("existing_linear_resource", "quadratic_resource_with_background"):
            linear = kind.startswith("existing")
            x = gap if linear else max(np.sqrt(max(1.5 - cap, 0)) - 1, 0)
            loss = 0.5 * x + x * x if linear else (1 + x) ** 3 + 1.5 * (1 - x) - 2.5
            bound = growth_loss_bound(gap, 1, (6 if linear else 17) + cap, 2 if linear else 3)
            if bound > loss + 1e-14:
                raise AssertionError("Analytical loss bound exceeds exact value")
            analytical.append(
                {
                    "example": kind,
                    "price_cap": cap,
                    "gap": gap,
                    "equilibrium_x": x,
                    "exact_loss": loss,
                    "general_bound": bound,
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "buffer_summary.csv", index=False)
    pd.DataFrame(all_details).to_csv(output / "stage_details.csv", index=False)
    pd.DataFrame(bound_rows).to_csv(output / "loss_bound_constants.csv", index=False)
    pd.DataFrame(analytical).to_csv(output / "analytical_loss_checks.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "diagnostic_sha256": sha(Path(__file__)),
                "saved_source_verified": True,
                "saved_allocations_sha256": files,
                "new_equilibrium_solves": 0,
                "delay_interpretation": "Congestion delay only; reliability cost excluded. Serial propagation is a conditional diagnostic, not the source model.",
                "new_parameters": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pd.DataFrame(summaries)
