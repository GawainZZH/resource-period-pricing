import csv
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import resource_pricing.integer_allocations as core

STAGES = core.ROOT / "data/protected_reservations/protected_stage_intervals.csv"


def audit(model, raw):
    h = [F(v) for v in raw]
    if (
        len(h) != len(model.base)
        or any((v.denominator != 1 for v in h))
        or (not core.bounds().is_exact_feasible(model, h))
    ):
        raise ValueError("Reservation implementation requires exactly feasible integer flow")
    with STAGES.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    grouped = [[] for _ in h]
    for row in records:
        grouped[int(row["choice"])].append(row)
    loads = core.bounds().exact_loads(model.occupation, h)
    delay = [
        F(float(alpha)) * (y / F(float(K))) ** 2
        for alpha, y, K in zip(model.alpha, loads, model.capacity)
    ]
    capacities = {}
    for resource, capacity in zip(model.row_resource, model.capacity):
        resource, capacity = (int(resource), int(capacity))
        if resource in capacities and capacities[resource] != capacity:
            raise ValueError("The frozen instance must have constant positions per resource")
        capacities[resource] = capacity
    bookings, completions, buffers = ([], [], [])
    operation = 0
    for j, amount in enumerate(h):
        route = grouped[j]
        if len(route) != 7:
            raise ValueError("Seven protected stages required")
        start = [F(row["start_minutes"]) for row in route]
        end = [F(row["end_minutes"]) for row in route]
        assert end[0] == start[1] == start[2]
        assert end[2] == start[3] and end[3] == start[4] == start[5] and (end[5] == start[6])
        own = []
        for row, a, b in zip(route, start, end):
            mask = (
                (model.row_resource == int(row["resource"]))
                & (model.row_start >= float(a))
                & (model.row_start < float(b))
            )
            own.append(sum((delay[r] for r in np.flatnonzero(mask)), F(0)))
        extensions = [own[0] + own[1], F(0), own[2], own[3] + own[4], F(0), own[5], own[6]]
        for _ in range(int(amount)):
            for row, a, b, extension in zip(route, start, end, extensions):
                nominal = F(row["nominal_minutes"])
                finish = a + nominal + extension
                if finish > b:
                    raise ValueError("Actual processing exceeds a protected reservation")
                buffers.append(b - finish)
                bookings.append(
                    {
                        "operation": operation,
                        "choice": j,
                        "stage": row["stage"],
                        "resource": int(row["resource"]),
                        "start_exact": str(a),
                        "end_exact": str(b),
                        "processing_finish_exact": str(finish),
                        "extension_exact": str(extension),
                    }
                )
            completions.append(
                {
                    "operation": operation,
                    "choice": j,
                    "completion_exact": str(start[6] + F(route[6]["nominal_minutes"]) + own[6]),
                    "reservation_end_exact": str(end[6]),
                }
            )
            operation += 1
    positions = {}
    for row in sorted(
        bookings, key=lambda x: (F(x["start_exact"]), F(x["end_exact"]), x["operation"], x["stage"])
    ):
        resource = row["resource"]
        available = positions.setdefault(resource, [F(-1)] * capacities[resource])
        free = next(
            (i for i, finish in enumerate(available) if finish <= F(row["start_exact"])), None
        )
        if free is None:
            raise ValueError("No parallel reservation position is available")
        row["position"] = free
        available[free] = F(row["end_exact"])
    if len(bookings) != 7 * operation or operation != int(sum(model.demand)):
        raise ValueError("Incomplete request assignment")
    return {
        "passed": True,
        "requests": operation,
        "stage_reservations": len(bookings),
        "all_original_capacities_exact": True,
        "all_stage_handoffs_exact": True,
        "all_position_intervals_nonoverlapping": True,
        "all_completion_times_within_reservations": True,
        "minimum_stage_buffer_exact": str(min(buffers)),
        "minimum_stage_buffer_minutes": core.bounds().downward(min(buffers)),
        "bookings": bookings,
        "completions": completions,
        "scope": "Integer reservations and stage budgets; no microscopic trajectory optimization",
    }
