from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize


@dataclass(frozen=True)
class ClearingResult:
    flow: np.ndarray
    capacity_violation: float
    conservation_violation: float
    first_order_gap: float


def clear_reservations(
    request: np.ndarray,
    conservation: np.ndarray,
    demand: np.ndarray,
    delta: np.ndarray,
    capacity: np.ndarray,
) -> ClearingResult:
    if any((np.iscomplexobj(value) for value in (request, conservation, demand, delta, capacity))):
        raise ValueError("The clearing problem requires real coefficients.")
    request, conservation, demand, delta, capacity = [
        np.asarray(value, dtype=float) for value in (request, conservation, demand, delta, capacity)
    ]
    if (
        request.ndim != 1
        or demand.ndim != 1
        or capacity.ndim != 1
        or (conservation.ndim != 2)
        or (delta.ndim != 2)
        or (conservation.shape != (len(demand), len(request)))
        or (delta.shape != (len(capacity), len(request)))
    ):
        raise ValueError("Inconsistent clearing problem dimensions.")
    if not all(
        (np.all(np.isfinite(value)) for value in (request, conservation, demand, delta, capacity))
    ):
        raise ValueError("The clearing problem must contain finite coefficients.")
    response_scale = max(
        1.0, float(np.abs(request).max(initial=0.0)), float(np.abs(demand).max(initial=0.0))
    )
    if np.spacing(response_scale) > 1e-10:
        raise RuntimeError(
            "Clearing acceptance failed: request or demand scale exceeds audit precision."
        )
    bounds = [(0.0, None)] * len(request)
    feasible = linprog(
        np.zeros(len(request)),
        A_eq=conservation,
        b_eq=demand,
        A_ub=delta,
        b_ub=capacity,
        bounds=bounds,
        method="highs-ds",
    )
    if feasible.status == 2:
        raise ValueError("The shared capacity feasible set is empty.")
    if not feasible.success:
        raise RuntimeError(feasible.message)
    constraints = []
    if conservation.shape[0]:
        constraints.append(LinearConstraint(conservation, demand, demand))
    if delta.shape[0]:
        constraints.append(LinearConstraint(delta, -np.inf, capacity))
    result = minimize(
        lambda flow: 0.5 * float((flow - request) @ (flow - request)),
        feasible.x,
        jac=lambda flow: flow - request,
        constraints=constraints,
        bounds=bounds,
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(result.message)
    gradient = result.x - request
    gap_minimum = linprog(
        gradient,
        A_eq=conservation,
        b_eq=demand,
        A_ub=delta,
        b_ub=capacity,
        bounds=bounds,
        method="highs-ds",
    )
    if not gap_minimum.success:
        raise RuntimeError(gap_minimum.message)
    capacity_violation = float(np.maximum(delta @ result.x - capacity, 0.0).max(initial=0.0))
    conservation_violation = float(np.abs(conservation @ result.x - demand).max(initial=0.0))
    nonnegativity_violation = float(np.maximum(-result.x, 0.0).max(initial=0.0))
    first_order_gap = max(0.0, float(gradient @ result.x - gap_minimum.fun))
    if (
        max(capacity_violation, conservation_violation, nonnegativity_violation) > 1e-09
        or first_order_gap > 1e-07
    ):
        raise RuntimeError(
            f"Clearing acceptance failed: capacity={capacity_violation:g}, conservation={conservation_violation:g}, nonnegativity={nonnegativity_violation:g}, first-order gap={first_order_gap:g}."
        )
    return ClearingResult(
        flow=result.x.copy(),
        capacity_violation=capacity_violation,
        conservation_violation=conservation_violation,
        first_order_gap=first_order_gap,
    )
