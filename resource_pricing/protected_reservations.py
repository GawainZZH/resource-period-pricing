from __future__ import annotations
import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack
from resource_pricing.reservation_intervals import Stage, reconstruct

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/nominal_reservations"


@dataclass
class ProtectedCase:
    occupation: np.ndarray
    row_capacity: np.ndarray
    row_alpha: np.ndarray
    row_resource: np.ndarray
    row_start: np.ndarray
    conservation: np.ndarray
    demand: np.ndarray
    choices: np.ndarray
    routes: list
    stage_rows: list
    departure_penalty: np.ndarray
    grid: float
    metadata: dict


def protected_length(nominal, rate, grid, extra=0):
    if (
        not np.all(np.isfinite([nominal, rate, grid, extra]))
        or nominal <= 0
        or grid <= 0
        or (not 0 <= rate < 1)
        or (extra < 0)
    ):
        raise ValueError("Invalid protected interval parameters")
    slots = math.ceil((nominal + extra) / (1 - rate) / grid - 1e-12)
    length = slots * grid
    if length + 1e-12 < nominal + rate * length + extra:
        length += grid
    return float(length)


def build_protected_case():
    original, old_routes = reconstruct(3)
    old = original.problem
    grid = 3.0
    nr = original.metadata["resource_count"]
    capacities = np.full(nr, np.nan)
    rates = np.full(nr, np.nan)
    for resource in range(nr):
        rows = np.flatnonzero(original.row_resource == resource)
        if not len(rows):
            continue
        capacities[resource] = old.capacity[rows[0]]
        rates[resource] = old.alpha[rows[0]] / grid
    routes, penalties = ([], [])
    for j, old_route in enumerate(old_routes):
        entry1 = protected_length(old_route[1].nominal, 0, grid)
        entry2 = protected_length(old_route[4].nominal, 0, grid)
        upstream_extra = {
            0: rates[old_route[1].resource] * entry1,
            3: rates[old_route[4].resource] * entry2,
        }
        now = old_route[0].start
        route = []
        for k, prior in enumerate(old_route):
            rate = 0 if not prior.advances else rates[prior.resource]
            length = protected_length(prior.nominal, rate, grid, upstream_extra.get(k, 0))
            route.append(
                Stage(prior.name, prior.resource, now, now + length, prior.nominal, prior.advances)
            )
            if prior.advances:
                now += length
        routes.append(route)
        nominal = sum((s.nominal for s in old_route if s.advances))
        penalties.append(old.base[j] - nominal)
    nslots = round(max((r[-1].end for r in routes)) / grid)
    D = np.zeros((nr * nslots, len(routes)))
    for j, route in enumerate(routes):
        for stage in route:
            first, last = (round(stage.start / grid), round(stage.end / grid))
            D[stage.resource * nslots + first : stage.resource * nslots + last, j] = 1
    row_resource = np.repeat(np.arange(nr), nslots)
    row_start = np.tile(np.arange(nslots) * grid, nr)
    selected = D.any(axis=1)
    D, row_resource, row_start = (D[selected], row_resource[selected], row_start[selected])
    if not np.all(np.isfinite(capacities[row_resource] + rates[row_resource])):
        raise ValueError("A protected path uses an unspecified original resource")
    stage_rows = [
        [
            np.flatnonzero(
                (row_resource == s.resource) & (row_start >= s.start) & (row_start < s.end)
            )
            for s in route
        ]
        for route in routes
    ]
    metadata = dict(original.metadata)
    metadata.update(
        {
            "used_resource_periods": len(D),
            "reservation_horizon_minutes": nslots * grid,
            "rho_per_period": old.rho,
            "reliability_threshold": old.threshold,
            "origin_reservation_minutes": routes[0][0].end - routes[0][0].start,
            "destination_reservation_minutes": routes[0][-1].end - routes[0][-1].start,
            "cost_rule": "Protected-stage handoffs; only destination service delay extends final completion",
        }
    )
    return ProtectedCase(
        D,
        capacities[row_resource],
        rates[row_resource] * grid,
        row_resource,
        row_start,
        old.conservation.copy(),
        old.demand.copy(),
        original.choices.copy(),
        routes,
        stage_rows,
        np.asarray(penalties),
        grid,
        metadata,
    )


def execution_metrics(case, utilization):
    u = np.asarray(utilization, float)
    if (
        u.shape != case.row_capacity.shape
        or not np.all(np.isfinite(u))
        or np.any(u < 0)
        or np.any(u > 1)
    ):
        raise ValueError("Utilization must be in the protected interval [0, 1]")
    row_delay = case.row_alpha * u**2
    delay = np.array([[row_delay[rows].sum() for rows in route] for route in case.stage_rows])
    n = len(case.choices)
    nominal, waiting, completed, arrival, deadline = (np.zeros(n) for _ in range(5))
    for j, route in enumerate(case.routes):
        serial = (0, 2, 3, 5, 6)
        work = delay[j, list(serial)].copy()
        work[0] += delay[j, 1]
        work[2] += delay[j, 4]
        nominal[j] = sum((route[k].nominal for k in serial))
        for k, idx in enumerate(serial[:-1]):
            waiting[j] += route[idx].end - route[idx].start - route[idx].nominal - work[k]
        arrival[j] = route[-1].start
        completed[j] = arrival[j] + route[-1].nominal + delay[j, -1]
        deadline[j] = route[-1].end
    elapsed = completed - 6 * case.choices[:, 2]
    reliability = case.occupation.T @ (
        case.metadata["rho_per_period"]
        * np.maximum(u - case.metadata["reliability_threshold"], 0) ** 2
    )
    return {
        "arrival_time": arrival,
        "completion_time": completed,
        "protected_end": deadline,
        "elapsed_minutes": elapsed,
        "nominal_minutes": nominal,
        "physical_delay_minutes": delay.sum(axis=1),
        "remaining_wait_minutes": waiting,
        "destination_delay_minutes": delay[:, -1],
        "stage_delay_minutes": delay,
        "departure_penalty": case.departure_penalty.copy(),
        "reliability_penalty": reliability,
        "generalized_cost": elapsed + case.departure_penalty + reliability,
    }


def audit_routes(case):
    upper = execution_metrics(case, np.ones(len(case.row_capacity)))
    details, handoffs, overlaps = ([], [], [])
    for j, route in enumerate(case.routes):
        delays = upper["stage_delay_minutes"][j]
        for k, stage in enumerate(route):
            assigned = delays[k] if stage.advances else 0
            if k == 0:
                assigned += delays[1]
            elif k == 3:
                assigned += delays[4]
            margin = stage.end - stage.start - stage.nominal - assigned
            details.append(
                {
                    "choice": j,
                    "stage": stage.name,
                    "resource": stage.resource,
                    "start_minutes": stage.start,
                    "end_minutes": stage.end,
                    "nominal_minutes": stage.nominal,
                    "own_delay_upper": delays[k],
                    "physical_extension_upper": assigned,
                    "budget_margin_minutes": margin,
                    "entry_delay_allocated_upstream": not stage.advances,
                }
            )
        serial = [route[k] for k in (0, 2, 3, 5, 6)]
        handoffs.extend((abs(a.end - b.start) <= 1e-12 for a, b in zip(serial[:-1], serial[1:])))
        overlaps.extend(
            [
                route[1].start == route[2].start,
                route[4].start == route[5].start,
                route[1].end <= route[2].end,
                route[4].end <= route[5].end,
            ]
        )
    margins = [row["budget_margin_minutes"] for row in details]
    return {
        "all_budgets_pass": bool(min(margins) >= -1e-12),
        "all_handoffs_pass": bool(all(handoffs)),
        "entry_overlaps_preserved": bool(all(overlaps)),
        "minimum_budget_margin": float(min(margins)),
        "choices_checked": len(case.routes),
        "stage_rows": len(details),
        "details": details,
        "time_identity_error": float(
            np.max(
                abs(
                    upper["nominal_minutes"]
                    + upper["physical_delay_minutes"]
                    + upper["remaining_wait_minutes"]
                    - upper["elapsed_minutes"]
                )
            )
        ),
        "worst_completion_margin": float(np.min(upper["protected_end"] - upper["completion_time"])),
    }


def capacity_feasibility(case):
    D, H, K, q = (case.occupation, case.conservation, case.row_capacity, case.demand)
    if (
        not np.all((D == 0) | (D == 1))
        or not np.all((H == 0) | (H == 1))
        or (not np.all(H.sum(axis=0) == 1))
        or (not np.all(K == np.rint(K)))
        or (not np.all(q == np.rint(q)))
        or np.any(K < 0)
        or np.any(q < 0)
    ):
        raise ValueError("Exact LP evidence requires binary incidence and integer right hand sides")
    n = D.shape[1]
    result = linprog(
        -np.ones(n),
        A_ub=vstack([csr_matrix(D), csr_matrix(H)]),
        b_ub=np.r_[K, q],
        bounds=(0, None),
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-10,
            "primal_feasibility_tolerance": 1e-10,
            "time_limit": 60,
        },
    )
    if not result.success:
        return {"lp_success": False, "status": str(result.message), "full_demand_feasible": False}
    flow = result.x
    owners = H.argmax(axis=0)
    multipliers = [Fraction(str(max(0.0, -v))) for v in result.ineqlin.marginals]
    lam, pi = (multipliers[: len(K)], multipliers[len(K) :])
    coverage = [sum((lam[i] for i in np.flatnonzero(D[:, j])), Fraction(0)) for j in range(n)]
    for j in range(n):
        w = owners[j]
        pi[w] += max(Fraction(0), 1 - coverage[j] - pi[w])
    dual_feasible = all((coverage[j] + pi[owners[j]] >= 1 for j in range(n)))
    dual_upper = sum((int(k) * v for k, v in zip(K, lam)), Fraction(0))
    dual_upper += sum((int(amount) * v for amount, v in zip(q, pi)), Fraction(0))
    rational = [Fraction(float(v)).limit_denominator(10**8) for v in flow]
    load = [sum((rational[j] for j in np.flatnonzero(row)), Fraction(0)) for row in D]
    assigned = [sum((rational[j] for j in np.flatnonzero(row)), Fraction(0)) for row in H]
    primal_feasible = (
        all((v >= 0 for v in rational))
        and all((v <= int(k) for v, k in zip(load, K)))
        and all((v <= int(amount) for v, amount in zip(assigned, q)))
    )
    total = sum(rational, Fraction(0))
    required = Fraction(int(q.sum()))
    full = bool(primal_feasible and all((v == int(amount) for v, amount in zip(assigned, q))))
    infeasible_proved = bool(dual_feasible and dual_upper < required)
    return {
        "lp_success": True,
        "status": str(result.message),
        "full_demand_feasible": full,
        "full_demand_infeasible_proved": infeasible_proved,
        "maximum_assigned": float(flow.sum()),
        "required_total": int(q.sum()),
        "exact_primal_feasible": primal_feasible,
        "exact_primal_total": str(total),
        "exact_dual_feasible": dual_feasible,
        "exact_dual_upper": str(dual_upper),
        "exact_optimum_proved": bool(primal_feasible and dual_feasible and (total == dual_upper)),
        "unmet_by_od": [float(amount - v) for amount, v in zip(q, assigned)],
        "flow": flow,
        "rational_flow": [str(v) for v in rational],
        "dual_rows": [str(v) for v in lam],
        "dual_od": [str(v) for v in pi],
    }


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    if output.exists():
        raise ValueError("Output already exists; do not replace a completed pilot")
    saved = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    checks = {ROOT / "experiments/operational_reservation_validation.py": saved["script_sha256"]}
    checks.update(
        {ROOT / "experiments" / name: value for name, value in saved["dependency_sha256"].items()}
    )
    for path, value in checks.items():
        if file_hash(path).lower() != value.lower():
            raise ValueError(f"Original source changed: {path}")
    case = build_protected_case()
    audit = audit_routes(case)
    if (
        not all(
            (
                audit[key]
                for key in ("all_budgets_pass", "all_handoffs_pass", "entry_overlaps_preserved")
            )
        )
        or audit["time_identity_error"] > 1e-10
        or audit["worst_completion_margin"] < -1e-10
    ):
        raise ValueError("Protected construction failed before the capacity LP")
    capacity = capacity_feasibility(case)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(audit.pop("details")).to_csv(output / "protected_stage_intervals.csv", index=False)
    np.savez_compressed(
        output / "protected_case.npz",
        occupation=case.occupation,
        capacity=case.row_capacity,
        row_resource=case.row_resource,
        row_start=case.row_start,
        alpha=case.row_alpha,
        conservation=case.conservation,
        demand=case.demand,
        choices=case.choices,
        departure_penalty=case.departure_penalty,
    )
    if capacity["lp_success"]:
        flow = capacity.pop("flow")
        np.save(output / "maximum_assignment_flow.npy", flow)
        pd.DataFrame(
            {
                "choice": np.arange(len(flow)),
                "od": case.choices[:, 0],
                "flow": flow,
                "exact_flow": capacity.pop("rational_flow"),
            }
        ).to_csv(output / "maximum_assignment.csv", index=False)
        pd.DataFrame(
            {
                "resource": case.row_resource,
                "start_minutes": case.row_start,
                "capacity": case.row_capacity,
                "exact_dual_multiplier": capacity.pop("dual_rows"),
            }
        ).to_csv(output / "capacity_dual.csv", index=False)
        pd.DataFrame(
            {
                "od": np.arange(len(case.demand)),
                "required": case.demand,
                "unmet": capacity["unmet_by_od"],
                "exact_dual_multiplier": capacity.pop("dual_od"),
            }
        ).to_csv(output / "demand_dual.csv", index=False)
    result = {
        "metadata": case.metadata,
        "construction": audit,
        "capacity": capacity,
        "configuration_count": 1,
        "capacity_lp_count": 1,
        "new_equilibrium_solves": 0,
        "new_outer_price_solves": 0,
        "new_integer_solves": 0,
        "followup_allowed": bool(capacity.get("full_demand_feasible", False)),
        "source_hashes": {str(path.relative_to(ROOT)): file_hash(path) for path in checks},
        "script_sha256": file_hash(Path(__file__)),
        "execution_assumptions": [
            "Entrance waiting fits in the upstream protected node",
            "Early completion waits within the same protected resource until the next planned handoff",
            "Resource occupancy protection is not a general geometric separation model",
        ],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
