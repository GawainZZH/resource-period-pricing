from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linprog
from resource_pricing.convex_programs import LP_OPTIONS


def resource_coverage_gap(
    problem, h, price_map, *, domain="nonnegative", price_cap=None, cost=None
):
    h = np.asarray(h, dtype=float)
    P = np.asarray(price_map, dtype=float)
    c = problem.private(h) if cost is None else np.asarray(cost, dtype=float)
    n = len(problem.base)
    if (
        h.shape != (n,)
        or c.shape != (n,)
        or P.ndim != 2
        or (P.shape[0] != n)
        or (not np.all(np.isfinite(h)))
        or (not np.all(np.isfinite(c)))
        or (not np.all(np.isfinite(P)))
    ):
        raise ValueError("Invalid flow, cost or price mapping")
    if max(problem.feasibility(h).values()) > 1e-09:
        raise ValueError("Reference allocation must be feasible")
    if domain not in {"signed", "nonnegative", "capped"}:
        raise ValueError("Unknown price domain")
    k = P.shape[1]
    original_P = P
    cap = None
    if domain == "capped":
        cap = np.asarray(price_cap, dtype=float)
        if cap.shape != (k,) or not np.all(np.isfinite(cap)) or np.any(cap < 0):
            raise ValueError("Invalid price cap")
    elif price_cap is not None:
        raise ValueError("Price caps require the capped domain")
    P = P.copy()
    for row in problem.conservation:
        members = row.astype(bool)
        if np.any(members):
            P[members] -= np.min(P[members], axis=0)
    column_scale = np.max(np.abs(P), axis=0)
    occupied = column_scale > 0
    P = P[:, occupied] / column_scale[occupied]
    if cap is not None:
        cap = cap[occupied] * column_scale[occupied]
        if not np.all(np.isfinite(cap)):
            raise ValueError("Price caps exceed a safe floating-point scale")
    k = P.shape[1]
    centered = c.copy()
    for row in problem.conservation:
        members = row.astype(bool)
        if np.any(members):
            centered[members] -= np.min(c[members])
    scale = max(float(np.max(np.abs(centered))), 1e-20)
    D, H = (problem.occupation, problem.conservation)
    if domain == "capped":
        objective = np.r_[centered, cap] / scale
        A_ub = np.vstack(
            [np.column_stack([D, np.zeros((len(D), k))]), np.column_stack([P.T, -np.eye(k)])]
        )
        b_ub = np.r_[problem.capacity, P.T @ h]
        A_eq = np.column_stack([H, np.zeros((len(H), k))])
        b_eq = problem.demand
    else:
        objective = centered / scale
        A_ub, b_ub = (D, problem.capacity)
        A_eq, b_eq = (H, problem.demand)
        if k and domain == "nonnegative":
            A_ub, b_ub = (np.vstack([D, P.T]), np.r_[b_ub, P.T @ h])
        elif k:
            A_eq, b_eq = (np.vstack([H, P.T]), np.r_[b_eq, P.T @ h])
    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs-ds",
        options=LP_OPTIONS,
    )
    if (
        not result.success
        or result.x is None
        or (not np.all(np.isfinite(result.x)))
        or (not np.isfinite(result.fun))
    ):
        raise RuntimeError(f"Flow-space LP failed: {result.message}")
    z = result.x[:n]
    visible = P.T @ (z - h)
    violation = max(problem.feasibility(z).values())
    if k and domain == "nonnegative":
        violation = max(violation, float(np.max(visible)))
    elif k and domain == "signed":
        violation = max(violation, float(np.max(np.abs(visible))))
    gain = math.fsum((float(v) * float(d) for v, d in zip(centered, h - z)))
    penalty = (
        0
        if cap is None
        else math.fsum((float(v) * float(a) for v, a in zip(cap, np.maximum(visible, 0))))
    )
    gap = gain - penalty
    lp_gap = (
        math.fsum((float(v) * float(x) for v, x in zip(centered, h))) - float(result.fun) * scale
    )
    discrepancy = abs(gap - lp_gap)
    lam = np.minimum(np.asarray(result.ineqlin.marginals), 0)
    nu = np.asarray(result.eqlin.marginals).copy()
    if cap is not None and k:
        lam[-k:] = np.maximum(lam[-k:], -objective[n:])
    reduced = objective - A_ub.T @ lam - A_eq.T @ nu
    for w, row in enumerate(H):
        members = row.astype(bool)
        if np.any(members) and np.min(reduced[:n][members]) < 0:
            nu[w] += np.nextafter(float(np.min(reduced[:n][members])), -np.inf)
    reduced = objective - A_ub.T @ lam - A_eq.T @ nu
    dual_value = math.fsum((float(v) * float(x) for v, x in zip(lam, b_ub)))
    dual_value += math.fsum((float(v) * float(x) for v, x in zip(nu, b_eq)))
    gap_upper = math.fsum((float(v) * float(x) for v, x in zip(centered, h))) - dual_value * scale
    uncertainty = abs(gap_upper - gap)
    if (
        not np.all(np.isfinite(reduced))
        or not np.isfinite(gap_upper)
        or np.min(reduced) < -1e-10
        or (gap_upper < gap - 1e-10)
        or (violation > 1e-08)
        or (max(discrepancy, uncertainty) > 1e-09 * max(1.0, abs(gap)))
    ):
        raise RuntimeError("Flow-space LP violates feasibility or objective consistency")
    return {
        "gap": gap,
        "gap_upper": gap_upper,
        "LP_uncertainty": uncertainty,
        "witness": z,
        "visible_change": original_P.T @ (z - h),
        "private_gain": gain,
        "cap_penalty": penalty,
        "feasibility_violation": violation,
        "objective_discrepancy": discrepancy,
        "domain": domain,
    }
