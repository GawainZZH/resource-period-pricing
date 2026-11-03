from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from resource_pricing.controlled_networks import build_metropolitan_corridor_instance
from resource_pricing.convex_programs import ResourceProblem, linear_gap, refine_active_face
from resource_pricing.coverage import resource_coverage_gap


def knots_to_km_min(speed):
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Speed must be positive and finite")
    return float(speed) * 1.852 / 60


def reservation_slots(start, duration, grid):
    if not np.all(np.isfinite([start, duration, grid])) or start < 0 or min(duration, grid) <= 0:
        raise ValueError("Invalid reservation interval")
    if abs(start / grid - round(start / grid)) > 1e-10:
        raise ValueError("Reservation start must be on the resource grid")
    first = round(start / grid)
    length = math.ceil(duration / grid - 1e-12)
    return (first, first + length)


@dataclass
class OperationalCase:
    problem: ResourceProblem
    choices: np.ndarray
    row_resource: np.ndarray
    row_start: np.ndarray
    resource_kind: np.ndarray
    grid: float
    metadata: dict

    def price_map(self, name):
        D = self.problem.occupation
        if name == "all":
            return self.grid * D.T
        if name in ("corridor", "node"):
            return self.grid * D[self.resource_kind[self.row_resource] == name].T
        if name == "contemporaneous":
            start = 6 * self.choices[:, 2]
            visible = (self.row_start[:, None] >= start) & (self.row_start[:, None] < start + 6)
            return (self.grid * D * visible).T
        raise ValueError("Unknown price coverage")


def build_operational_case(
    grid_minutes=3, headway_minutes=2, service_minutes=6, demand_per_od=4, terminal_channels=2
):
    if (
        not np.all(np.isfinite([grid_minutes, headway_minutes, service_minutes, demand_per_od]))
        or min(grid_minutes, headway_minutes, service_minutes, demand_per_od) <= 0
        or (not np.isfinite(terminal_channels))
        or (terminal_channels < 1)
        or (terminal_channels != round(terminal_channels))
        or (abs(6 / grid_minutes - round(6 / grid_minutes)) > 1e-12)
    ):
        raise ValueError(
            "Positive parameters and a resource grid dividing six minutes are required"
        )
    legacy, meta = build_metropolitan_corridor_instance()
    choices = np.column_stack((legacy.od_of_alt, legacy.path_of_alt, legacy.dep_of_alt))
    edges = meta["edge_to_idx"]
    nc = len(edges)
    nr = 2 * nc + 16
    speed = knots_to_km_min(120)
    edge_time = np.zeros(nc)
    for edge, idx in edges.items():
        a = meta["zone_coords"][edge[1]] if edge[0] == "z" else meta["merge_coords"][edge[1]]
        b = meta["zone_coords"][edge[3]] if edge[2] == "z" else meta["merge_coords"][edge[3]]
        edge_time[idx] = 5 * np.linalg.norm(a - b) / speed
    kinds = np.array(["corridor"] * (2 * nc) + ["node"] * 16)
    capacities = np.ones(nr)
    capacities[nc : 2 * nc] = np.maximum(1, np.floor(edge_time / headway_minutes))
    capacities[2 * nc : 2 * nc + 12] = terminal_channels
    rate = np.r_[np.full(nc, 0.01), np.full(nc, 0.05), np.full(16, 0.2)]
    H = np.eye(10)[choices[:, 0]].T
    base = np.zeros(len(choices))
    reservations = []
    centers = {}
    for w in range(10):
        selected = np.flatnonzero(legacy.od_of_alt == w)
        preferred = selected[np.argmin(legacy.schedule_penalty[selected])]
        centers[w] = 6 * legacy.dep_of_alt[preferred]
    for j, (w, p, t) in enumerate(choices):
        origin, merge, dest, ci, co, _ = meta["path_edges"][j]
        time_slot = round(6 * t / grid_minutes)
        intervals = []

        def reserve(resource, duration, advance=True):
            nonlocal time_slot
            first, last = reservation_slots(time_slot * grid_minutes, duration, grid_minutes)
            intervals.append((resource, first, last))
            if advance:
                time_slot = last

        reserve(2 * nc + origin, service_minutes)
        reserve(ci, headway_minutes, advance=False)
        reserve(nc + ci, edge_time[ci])
        reserve(2 * nc + 12 + merge, headway_minutes)
        reserve(co, headway_minutes, advance=False)
        reserve(nc + co, edge_time[co])
        reserve(2 * nc + dest, service_minutes)
        reservations.append(intervals)
        base[j] = edge_time[ci] + edge_time[co] + 2 * service_minutes + headway_minutes
        base[j] += 0.2 * abs(6 * t - centers[w])
    nslots = max((last for intervals in reservations for _, _, last in intervals))
    D = np.zeros((nr * nslots, len(choices)))
    for j, intervals in enumerate(reservations):
        for resource, first, last in intervals:
            D[resource * nslots + first : resource * nslots + last, j] = 1
    row_resource = np.repeat(np.arange(nr), nslots)
    row_start = np.tile(np.arange(nslots) * grid_minutes, nr)
    used = np.any(D != 0, axis=1)
    row_resource, row_start, D = (row_resource[used], row_start[used], D[used])
    problem = ResourceProblem(
        base,
        D,
        capacities[row_resource],
        rate[row_resource] * grid_minutes,
        0.1 * grid_minutes,
        0.7,
        H,
        np.full(10, float(demand_per_od)),
    )
    return OperationalCase(
        problem,
        choices,
        row_resource,
        row_start,
        kinds,
        float(grid_minutes),
        {
            "corridor_edges": nc,
            "resource_count": nr,
            "used_resource_periods": len(D),
            "speed_knots": 120,
            "speed_km_min": speed,
            "coordinate_scale_km": 5,
            "corridor_length_min_km": float(edge_time.min() * speed),
            "corridor_length_max_km": float(edge_time.max() * speed),
            "headway_minutes": headway_minutes,
            "service_minutes": service_minutes,
            "terminal_channels": terminal_channels,
            "preferred_departure_minutes": [int(centers[w]) for w in range(10)],
            "grid_minutes": grid_minutes,
            "demand_per_od": demand_per_od,
            "reservation_horizon_minutes": nslots * grid_minutes,
            "route_cruise_min_minutes": float(
                min((edge_time[ci] + edge_time[co] for _, _, _, ci, co, _ in meta["path_edges"]))
            ),
            "route_cruise_max_minutes": float(
                max((edge_time[ci] + edge_time[co] for _, _, _, ci, co, _ in meta["path_edges"]))
            ),
        },
    )


def potential(problem, h, price=None):
    y = problem.occupation @ h
    v = np.maximum(y / problem.capacity - problem.threshold, 0)
    price = np.zeros_like(h) if price is None else np.asarray(price)
    return float(
        (problem.base + price) @ h
        + np.sum(problem.alpha * y**3 / (3 * problem.capacity**2))
        + problem.rho / 3 * np.sum(problem.capacity * v**3)
    )


class PotentialProblem:

    def __init__(self, problem, price):
        self.problem = problem
        self.price = price
        for key in ("base", "occupation", "capacity", "conservation", "demand"):
            setattr(self, key, getattr(problem, key))

    def cost(self, h):
        return potential(self.problem, h, self.price)

    def gradient(self, h):
        return self.problem.private(h) + self.price

    def hessian(self, h):
        p = self.problem
        u = p.occupation @ h / p.capacity
        weights = (2 * p.alpha * u + 2 * p.rho * np.maximum(u - p.threshold, 0)) / p.capacity
        return p.occupation.T @ (weights[:, None] * p.occupation)

    def feasibility(self, h):
        return self.problem.feasibility(h)


def solve_reservation(problem, system=False, price=None):
    n = len(problem.base)
    price = np.zeros(n) if price is None else np.asarray(price, float)
    if price.shape != (n,) or not np.all(np.isfinite(price)):
        raise ValueError("Invalid price vector")
    start = time.perf_counter()
    h = cp.Variable(n, nonneg=True)
    y = csr_matrix(problem.occupation) @ h
    v = cp.Variable(len(problem.capacity), nonneg=True)
    if system:
        objective = problem.base @ h + cp.sum(
            cp.multiply(problem.alpha / problem.capacity**2, cp.power(y, 3))
        )
        objective += problem.rho * cp.sum(
            cp.multiply(problem.capacity, cp.power(v, 3) + problem.threshold * cp.square(v))
        )
        gradient_problem = problem
    else:
        objective = (problem.base + price) @ h
        objective += cp.sum(cp.multiply(problem.alpha / (3 * problem.capacity**2), cp.power(y, 3)))
        objective += problem.rho / 3 * cp.sum(cp.multiply(problem.capacity, cp.power(v, 3)))
        gradient_problem = PotentialProblem(problem, price)
    program = cp.Problem(
        cp.Minimize(objective),
        [
            problem.conservation @ h == problem.demand,
            y <= problem.capacity,
            v >= y / problem.capacity - problem.threshold,
        ],
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            program.solve(
                solver="CLARABEL",
                tol_gap_abs=1e-10,
                tol_gap_rel=1e-10,
                tol_feas=1e-10,
                max_iter=500,
            )
    except cp.error.SolverError as error:
        return {"success": False, "status": str(error), "seconds": time.perf_counter() - start}
    if h.value is None:
        return {"success": False, "status": program.status, "seconds": time.perf_counter() - start}
    initial = np.asarray(h.value).reshape(-1)
    if max(problem.feasibility(initial).values()) > 1e-08:
        return {
            "success": False,
            "status": "conic flow fails feasibility",
            "seconds": time.perf_counter() - start,
        }
    corrected = refine_active_face(gradient_problem, initial, max_iterations=120)
    flow = corrected["flow"]
    checks = linear_gap(problem, flow, gradient_problem.gradient(flow))
    success = (
        checks["LP_success"]
        and max(abs(checks["raw_gap"]), checks["gap_upper"]) <= 1e-07
        and (max(problem.feasibility(flow).values()) <= 1e-09)
    )
    return {
        "success": bool(success),
        "status": corrected["status"],
        "conic_status": program.status,
        "flow": flow,
        "objective": problem.cost(flow),
        "seconds": time.perf_counter() - start,
        **checks,
    }


def scale_problem(problem, multiplier):
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Invalid scale")
    return ResourceProblem(
        problem.base.copy(),
        problem.occupation.copy(),
        multiplier * problem.capacity,
        problem.alpha.copy(),
        problem.rho,
        problem.threshold,
        problem.conservation.copy(),
        multiplier * problem.demand,
    )


def integer_projection(problem, flow, seconds=30):
    flow = np.asarray(flow, float)
    if (
        flow.shape != problem.base.shape
        or not np.all(np.isfinite(flow))
        or np.max(abs(problem.demand - np.rint(problem.demand)), initial=0) > 1e-10
    ):
        raise ValueError("Integer demands and a finite flow vector are required")
    n = len(flow)
    I = np.eye(n)
    result = milp(
        np.r_[np.zeros(n), np.ones(n)],
        integrality=np.r_[np.ones(n), np.zeros(n)],
        bounds=Bounds(np.zeros(2 * n), np.full(2 * n, np.inf)),
        constraints=[
            LinearConstraint(
                csr_matrix(np.c_[problem.conservation, np.zeros_like(problem.conservation)]),
                problem.demand,
                problem.demand,
            ),
            LinearConstraint(
                csr_matrix(np.c_[problem.occupation, np.zeros_like(problem.occupation)]),
                -np.inf,
                problem.capacity,
            ),
            LinearConstraint(np.c_[I, -I], -np.inf, flow),
            LinearConstraint(np.c_[-I, -I], -np.inf, -flow),
        ],
        options={"time_limit": seconds, "mip_rel_gap": 1e-09},
    )
    if result.x is None or not np.all(np.isfinite(result.x)):
        return {"success": False, "status": str(result.message)}
    integer = np.rint(result.x[:n])
    if (
        np.max(abs(integer - result.x[:n]), initial=0) > 1e-06
        or max(problem.feasibility(integer).values()) > 1e-08
    ):
        return {"success": False, "status": "MILP incumbent fails integer feasibility"}
    return {
        "success": True,
        "status": str(result.message),
        "distance_optimal": bool(result.success),
        "flow": integer,
        "l1_distance": float(np.abs(integer - flow).sum()),
        "objective_upper": problem.cost(integer),
    }


CONFIGURATIONS = (
    ("baseline", 3, 2, 6, 2),
    ("grid_1.5", 1.5, 2, 6, 2),
    ("grid_6", 6, 2, 6, 2),
    ("headway_1", 3, 1, 6, 2),
    ("headway_3", 3, 3, 6, 2),
    ("service_3", 3, 2, 3, 2),
    ("service_9", 3, 2, 9, 2),
    ("single_terminal", 3, 2, 6, 1),
)


def run_batch(output, baseline_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configurations = CONFIGURATIONS[:1] if baseline_only else CONFIGURATIONS
    rows, gaps, integers = ([], [], [])
    metadata = {}
    for name, grid, headway, service, channels in configurations:
        case = build_operational_case(grid, headway, service, terminal_channels=channels)
        p = case.problem
        metadata[name] = case.metadata
        due = solve_reservation(p)
        so = solve_reservation(p, system=True)
        print(
            name,
            "DUE",
            due["success"],
            due["status"],
            "SO",
            so["success"],
            so["status"],
            flush=True,
        )
        for tag, result in (("due", due), ("so", so)):
            if "flow" in result:
                np.save(output / f"{name}_{tag}.npy", result["flow"])
        if not due["success"] or not so["success"]:
            rows.append(
                {
                    "case": name,
                    "success": False,
                    "DUE_status": due["status"],
                    "SO_status": so["status"],
                }
            )
            pd.DataFrame(rows).to_csv(output / "operational_results.csv", index=False)
            continue
        h = so["flow"]
        u = p.occupation @ h / p.capacity
        marginal = 2 * p.alpha * u**2
        indices = case.resource_kind[case.row_resource]
        visible = (case.row_start[:, None] >= 6 * case.choices[:, 2]) & (
            case.row_start[:, None] < 6 * case.choices[:, 2] + 6
        )
        denominator = due["objective"] - so["objective"]
        for cls in ("corridor", "node", "contemporaneous"):
            if cls == "contemporaneous":
                price = (p.occupation * visible).T @ marginal
            else:
                price = p.occupation[indices == cls].T @ marginal[indices == cls]
            priced = solve_reservation(p, price=price)
            row = {
                "case": name,
                "class": cls,
                "success": priced["success"],
                "grid_minutes": grid,
                "headway_minutes": headway,
                "service_minutes": service,
                "PoA": due["objective"] / so["objective"],
                "DUE_cost": due["objective"],
                "SO_cost": so["objective"],
                "DUE_gap_upper": due["gap_upper"],
                "SO_gap_upper": so["gap_upper"],
                "SO_capacity_violation": so["capacity_violation"],
                "SO_capacity_fraction": float(np.mean(abs(p.capacity - p.occupation @ h) < 1e-07)),
            }
            if priced["success"]:
                row.update(
                    GCR=(due["objective"] - priced["objective"]) / denominator,
                    priced_cost=priced["objective"],
                    priced_gap_upper=priced["gap_upper"],
                )
                np.save(output / f"{name}_{cls}_priced.npy", priced["flow"])
            rows.append(row)
            P = case.price_map(cls)
            for cap in (None, 2, 5, 10):
                result = resource_coverage_gap(
                    p,
                    h,
                    P,
                    domain="nonnegative" if cap is None else "capped",
                    price_cap=None if cap is None else np.full(P.shape[1], cap),
                )
                gaps.append(
                    {
                        "case": name,
                        "class": cls,
                        "price_cap": "unbounded" if cap is None else cap,
                        **{k: v for k, v in result.items() if np.isscalar(v)},
                    }
                )
        if name == "baseline":
            for scale in (1, 2, 5, 10):
                scaled = scale_problem(p, scale)
                projected = integer_projection(scaled, scale * h)
                if projected["success"]:
                    np.save(output / f"integer_scale_{scale}.npy", projected["flow"])
                    lower = scale * (so["objective"] - max(0, so["gap_upper"]))
                    integers.append(
                        {
                            "scale": scale,
                            "success": True,
                            "continuous_cost": scale * so["objective"],
                            "continuous_lower": lower,
                            "integer_cost_upper": projected["objective_upper"],
                            "relative_cost_bound": (projected["objective_upper"] - lower) / lower,
                            "l1_distance": projected["l1_distance"],
                            "capacity_violation": scaled.feasibility(projected["flow"])[
                                "capacity_violation"
                            ],
                            "distance_optimal": projected["distance_optimal"],
                        }
                    )
                else:
                    integers.append({"scale": scale, **projected})
        pd.DataFrame(rows).to_csv(output / "operational_results.csv", index=False)
        pd.DataFrame(gaps).to_csv(output / "coverage_gaps.csv", index=False)
        pd.DataFrame(integers).to_csv(output / "integer_approximation.csv", index=False)
    script = Path(__file__).read_bytes()
    (output / "execution_source.py").write_bytes(script)
    dependencies = (
        "controlled_synthetic_study.py",
        "high_precision_so_audit.py",
        "resource_period_coverage.py",
    )
    dependency_sha = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in dependencies
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "script_sha256": hashlib.sha256(script).hexdigest(),
                "dependency_sha256": dependency_sha,
                "metadata": metadata,
                "configurations": configurations,
                "baseline_only": baseline_only,
                "source": "https://ntrs.nasa.gov/citations/20250001367",
                "source_status": "Simulation assumptions; not regulatory separation or observed service times",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
