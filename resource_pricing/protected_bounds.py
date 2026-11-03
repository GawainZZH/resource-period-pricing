import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
from resource_pricing.protected_costs import load_three_channel

DATA = HERE / "data"
WITNESS = ROOT / "data/reservation_inputs/three_channel_maximum_assignment.npy"
VI_DIRECTORY = ROOT / "deliverables/protected_unpriced_vi_gate_20261021"
CANDIDATE_SECONDS, GAP_SECONDS, CONIC_SECONDS = (120, 30, 60)


def old_helpers():
    name = "_protected_vi_readonly_helpers"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("protected_equilibrium.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def write_json(path, value):
    old_helpers().write_json(path, value)


def save_arrays(path, **values):
    old_helpers().save_arrays(path, **values)


def downward(value):
    number = float(value)
    return float(np.nextafter(number, -math.inf)) if F(number) > value else number


def upward(value):
    number = float(value)
    return float(np.nextafter(number, math.inf)) if F(number) < value else number


def rational(v):
    return F(float(v))


def cell_envelope(a, d):
    if not (np.isfinite(a) and np.isfinite(d) and (0 <= a <= d <= 1)):
        raise ValueError("Expected 0 <= destination load <= total load <= 1")
    t = 1 - d + a
    return 0.0 if t == 0 else a**3 / t**2


def cell_derivatives(a, d):
    cell_envelope(a, d)
    t = 1 - d + a
    if t == 0:
        raise ValueError("No unique gradient at the degenerate corner")
    return (3 * a * a / t**2 - 2 * a**3 / t**3, 2 * a**3 / t**3)


def exact_loads(matrix, h):
    return [sum((rational(row[j]) * h[j] for j in np.flatnonzero(row)), F(0)) for row in matrix]


def is_exact_feasible(model, h):
    return (
        all((v >= 0 for v in h))
        and all(
            (v <= rational(k) for v, k in zip(exact_loads(model.occupation, h), model.capacity))
        )
        and all(
            (v == rational(q) for v, q in zip(exact_loads(model.conservation, h), model.demand))
        )
    )


def reconstruct_face(model, raw):
    raw = np.asarray(raw, float)
    support = np.flatnonzero(raw > 1e-08)
    active = model.capacity - model.occupation @ raw <= 1e-07
    A = np.vstack([model.conservation[:, support], model.occupation[active][:, support]])
    b = np.r_[model.demand, model.capacity[active]]
    rows = list(
        dict.fromkeys((tuple((rational(v) for v in np.r_[row, rhs])) for row, rhs in zip(A, b)))
    )
    pivots = []
    rank = 0
    for col in range(len(support)):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = (rows[pivot], rows[rank])
        divisor = rows[rank][col]
        rows[rank] = tuple((v / divisor for v in rows[rank]))
        for i in range(len(rows)):
            if i != rank and rows[i][col]:
                multiple = rows[i][col]
                rows[i] = tuple((v - multiple * p for v, p in zip(rows[i], rows[rank])))
        pivots.append(col)
        rank += 1
        if rank == len(rows):
            break
    if any((not any(row[:-1]) and row[-1] for row in rows)):
        return {"exact_feasible": False, "reason": "Chosen active face is inconsistent"}
    free = [col for col in range(len(support)) if col not in pivots]
    values = [F(0) for _ in support]
    for col in free:
        values[col] = rational(raw[support[col]]).limit_denominator(10**9)
    for i, col in enumerate(pivots):
        values[col] = rows[i][-1] - sum((rows[i][j] * values[j] for j in free), F(0))
    h = [F(0) for _ in raw]
    for j, v in zip(support, values):
        h[j] = v
    feasible = is_exact_feasible(model, h)
    return {
        "exact_feasible": feasible,
        "rational_flow": [str(v) for v in h],
        "max_flow_change": float(max((abs(rational(v) - r) for v, r in zip(raw, h)))),
        "support_size": len(support),
        "active_face_rank": rank,
    }


def exact_original_cost(model, h):
    y, z = (exact_loads(model.occupation, h), exact_loads(model.destination, h))
    value = sum((rational(b) * v for b, v in zip(model.base, h)), F(0))
    for yr, zr, cap, alpha in zip(y, z, model.capacity, model.alpha):
        u = yr / rational(cap)
        v = max(F(0), u - rational(model.threshold))
        value += rational(alpha) * zr * u * u + rational(model.rho) * yr * v * v
    return value


def exact_original_gradient(model, h):
    y, z = (exact_loads(model.occupation, h), exact_loads(model.destination, h))
    gradient = [rational(v) for v in model.base]
    for r, (yr, zr, capacity, alpha) in enumerate(zip(y, z, model.capacity, model.alpha)):
        K, alpha = (rational(capacity), rational(alpha))
        u, rho = (yr / K, rational(model.rho))
        v = max(F(0), u - rational(model.threshold))
        total_coefficient = rho * v * v + 2 * alpha * zr * yr / (K * K) + 2 * rho * yr * v / K
        destination_coefficient = alpha * u * u
        for j in np.flatnonzero(model.occupation[r]):
            gradient[j] += rational(model.occupation[r, j]) * total_coefficient
        for j in np.flatnonzero(model.destination[r]):
            gradient[j] += rational(model.destination[r, j]) * destination_coefficient
    return gradient


def exact_gradient_gap(model, h, cost, baseline, scale, result):
    if not result.success or result.x is None:
        return {"LP_success": False, "status": str(result.message)}
    lambdas = [min(F(0), rational(v)) * rational(scale) for v in result.ineqlin.marginals]
    nus = [
        rational(v) * rational(scale) + rational(b)
        for v, b in zip(result.eqlin.marginals, baseline)
    ]
    coverage = [
        sum(
            (
                rational(model.occupation[r, j]) * lambdas[r]
                for r in np.flatnonzero(model.occupation[:, j])
            ),
            F(0),
        )
        for j in range(len(h))
    ]
    for w, row in enumerate(model.conservation):
        nus[w] += min(F(0), min((cost[j] - coverage[j] - nus[w] for j in np.flatnonzero(row))))
    owner = model.conservation.argmax(axis=0)
    feasible = all((coverage[j] + nus[owner[j]] <= cost[j] for j in range(len(h))))
    lower = sum((rational(v) * l for v, l in zip(model.capacity, lambdas)), F(0))
    lower += sum((rational(v) * l for v, l in zip(model.demand, nus)), F(0))
    target = sum((c * v for c, v in zip(cost, h)), F(0))
    witness = [rational(v).limit_denominator(10**9) for v in result.x]
    witness_ok = is_exact_feasible(model, witness)
    gap_upper = target - lower
    gap_lower = (
        max(F(0), target - sum((c * v for c, v in zip(cost, witness)), F(0)))
        if witness_ok
        else F(0)
    )
    return {
        "LP_success": True,
        "exact_dual_feasible": feasible,
        "exact_primal_witness_feasible": witness_ok,
        "gap_lower_exact": str(gap_lower),
        "gap_upper_exact": str(gap_upper),
        "gap_lower": downward(gap_lower),
        "gap_upper": upward(gap_upper),
        "bound_width": upward(gap_upper - gap_lower),
        "gradient_exact": [str(v) for v in cost],
        "source": "exact original objective gradient at the exact feasible rational upper witness",
        "scope": "first-order gap only; nonconvex global validity comes from the independent L/U bounds",
    }


def exact_support_lower(model, raw, capacity_dual):
    h0 = [max(F(0), rational(v)) for v in raw]
    y = exact_loads(model.occupation, h0)
    maximum = max([F(1)] + [v / rational(k) for v, k in zip(y, model.capacity)])
    factor = (1 - F(1, 2**30)) / maximum
    h0 = [factor * v for v in h0]
    y, z = (exact_loads(model.occupation, h0), exact_loads(model.destination, h0))
    gradient = [rational(v) for v in model.base]
    value = sum((rational(b) * v for b, v in zip(model.base, h0)), F(0))
    for r, (yr, zr, capacity, alpha) in enumerate(zip(y, z, model.capacity, model.alpha)):
        K, alpha = (rational(capacity), rational(alpha))
        a, d = (zr / K, yr / K)
        assert 0 <= a <= d < 1
        t = 1 - d + a
        ga = alpha * (3 * a * a / t**2 - 2 * a**3 / t**3)
        gd = 2 * alpha * a**3 / t**3
        v = max(F(0), d - rational(model.threshold))
        risk_prime = rational(model.rho) * (v * v + 2 * d * v)
        value += alpha * K * a**3 / t**2 + rational(model.rho) * yr * v * v
        for j in np.flatnonzero(model.occupation[r]):
            gradient[j] += rational(model.occupation[r, j]) * (gd + risk_prime)
        for j in np.flatnonzero(model.destination[r]):
            gradient[j] += rational(model.destination[r, j]) * ga
    dual = [max(F(0), rational(v)) for v in capacity_dual]
    reduced = gradient.copy()
    for r, multiplier in enumerate(dual):
        for j in np.flatnonzero(model.occupation[r]):
            reduced[j] += rational(model.occupation[r, j]) * multiplier
    lower = value - sum((v * x for v, x in zip(gradient, h0)), F(0))
    lower -= sum((v * rational(k) for v, k in zip(dual, model.capacity)), F(0))
    for row, amount in zip(model.conservation, model.demand):
        lower += rational(amount) * min((reduced[j] for j in np.flatnonzero(row)))
    return {
        "lower_exact": str(lower),
        "lower_float": downward(lower),
        "anchor_domain_valid": True,
        "point_scale_exact": str(factor),
        "new_solver_calls": 0,
    }


def exact_envelope_value(model, h):
    y, z = (exact_loads(model.occupation, h), exact_loads(model.destination, h))
    value = sum((rational(b) * v for b, v in zip(model.base, h)), F(0))
    for yr, zr, cap, alpha in zip(y, z, model.capacity, model.alpha):
        a, d = (zr / rational(cap), yr / rational(cap))
        if not 0 <= a <= d <= 1:
            raise ValueError("Point outside cell envelope domain")
        t = 1 - d + a
        value += rational(alpha) * rational(cap) * (a**3 / t**2 if t else F(0))
        value += rational(model.rho) * yr * max(F(0), d - rational(model.threshold)) ** 2
    return value


def convex_model(model):
    import cvxpy as cp
    from scipy.sparse import csr_matrix

    D, E, H = (csr_matrix(v) for v in (model.occupation, model.destination, model.conservation))
    has_destination = np.any(model.destination != 0, axis=1)
    pure = np.flatnonzero(has_destination & np.all(model.destination == model.occupation, axis=1))
    mixed = np.flatnonzero(has_destination & ~np.all(model.destination == model.occupation, axis=1))
    h = cp.Variable(len(model.base), nonneg=True, name="flow")
    y, z = (D @ h, E @ h)
    d, a = (cp.multiply(1 / model.capacity, y), cp.multiply(1 / model.capacity, z))
    risk = cp.Variable(len(model.capacity), nonneg=True, name="risk")
    u, v = (cp.Variable(len(mixed), nonneg=True), cp.Variable(len(mixed), nonneg=True))
    cap_constraint = y <= model.capacity
    constraints = [H @ h == model.demand, cap_constraint, risk >= d - model.threshold]
    t = 1 - d[mixed] + a[mixed]
    constraints += [
        cp.SOC(u + t, cp.vstack([2 * a[mixed], u - t]), axis=0),
        cp.SOC(v + a[mixed], cp.vstack([2 * u, v - a[mixed]]), axis=0),
    ]
    delay = cp.sum(cp.multiply(model.alpha[pure] * model.capacity[pure], cp.power(d[pure], 3)))
    delay += cp.sum(cp.multiply(model.alpha[mixed] * model.capacity[mixed], v))
    risk_cost = cp.sum(
        cp.multiply(
            model.rho * model.capacity, cp.power(risk, 3) + model.threshold * cp.square(risk)
        )
    )
    problem = cp.Problem(cp.Minimize(model.base @ h + delay + risk_cost), constraints)
    return (
        problem,
        {"flow": h, "risk": risk, "u": u, "v": v, "mixed_rows": mixed, "pure_rows": pure},
        cap_constraint,
    )


def candidate_worker(directory):
    from scipy.optimize import minimize, Bounds, LinearConstraint

    model = load_three_channel()
    initial = np.load(WITNESS, allow_pickle=False)
    owner = model.conservation.argmax(axis=0)
    start = time.monotonic()
    last = initial.copy()
    count = 0

    def objective(h):
        nonlocal last, count
        last, count = (h.copy(), count + 1)
        value = model.cost(h)
        save_arrays(directory / "last_evaluated.npz", flow=h, objective=value)
        if time.monotonic() - start > CANDIDATE_SECONDS - 8:
            raise TimeoutError("Candidate internal deadline exceeded")
        return value

    try:
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=model.gradient,
            bounds=Bounds(np.zeros(len(initial)), model.demand[owner]),
            constraints=[
                LinearConstraint(model.conservation, model.demand, model.demand),
                LinearConstraint(model.occupation, -np.inf, model.capacity),
            ],
            options={"ftol": 1e-12, "maxiter": 500, "disp": False},
        )
        h = result.x.copy()
        info = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "source": "raw SLSQP return",
        }
    except Exception as error:
        h = last
        info = {
            "success": False,
            "message": f"{type(error).__name__}: {error}",
            "source": "last evaluated state; no restart",
        }
    save_arrays(
        directory / "candidate_raw.npz", flow=h, gradient=model.gradient(h), objective=model.cost(h)
    )
    info.update(
        seconds=time.monotonic() - start, evaluations=count, feasibility=model.feasibility(h)
    )
    write_json(directory / "candidate_raw.json", info)
    reconstructed = reconstruct_face(model, h)
    if reconstructed["exact_feasible"]:
        upper_h = [F(v) for v in reconstructed["rational_flow"]]
        reconstructed["source"] = "single exact active-face reconstruction of raw candidate"
    else:
        upper_h = [rational(v) for v in initial]
        reconstructed["source"] = (
            "original integer capacity witness; candidate reconstruction failed"
        )
    if not is_exact_feasible(model, upper_h):
        raise ValueError("No exact feasible upper witness")
    upper = exact_original_cost(model, upper_h)
    reconstructed.update(
        upper_flow=[str(v) for v in upper_h], upper_exact=str(upper), upper_float=upward(upper)
    )
    save_arrays(directory / "upper_witness.npz", flow=np.array([float(v) for v in upper_h]))
    write_json(directory / "upper_certificate.json", reconstructed)


def gap_worker(directory):
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    model = load_three_channel()
    upper = json.loads((directory / "upper_certificate.json").read_text(encoding="utf-8"))
    exact_h = [F(v) for v in upper["upper_flow"]]
    if not is_exact_feasible(model, exact_h):
        raise ValueError("Exact upper witness is infeasible")
    h = np.array([float(v) for v in exact_h])
    gradient = model.gradient(h)
    baseline = np.array([np.min(gradient[row == 1]) for row in model.conservation])
    centered = gradient - model.conservation.T @ baseline
    scale = max(float(np.max(abs(centered))), 1e-20)
    start = time.monotonic()
    result = linprog(
        centered / scale,
        A_ub=csr_matrix(model.occupation),
        b_ub=model.capacity,
        A_eq=csr_matrix(model.conservation),
        b_eq=model.demand,
        bounds=(0, None),
        method="highs-ds",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
            "time_limit": GAP_SECONDS - 5,
        },
    )
    save_arrays(
        directory / "gap_raw.npz",
        flow=h,
        gradient=gradient,
        baseline=baseline,
        scale=scale,
        lp_flow=np.asarray(result.x if result.x is not None else []),
        inequality_dual=np.asarray(
            result.ineqlin.marginals if result.ineqlin.marginals is not None else []
        ),
        equality_dual=np.asarray(
            result.eqlin.marginals if result.eqlin.marginals is not None else []
        ),
    )
    write_json(
        directory / "gap_raw.json",
        {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "seconds": time.monotonic() - start,
        },
    )
    out = old_helpers().exact_gap_bounds(model, h, gradient, baseline, scale, result)
    if out.get("LP_success"):
        correction = sum(
            (rational(g) * (v - rational(x)) for g, v, x in zip(gradient, exact_h, h)), F(0)
        )
        out["rational_target_correction_exact"] = str(correction)
        out["corrected_gap_upper"] = upward(F(out["gap_upper_exact"]) + correction)
        if out.get("gap_lower_exact") is not None:
            out["corrected_gap_lower"] = downward(F(out["gap_lower_exact"]) + correction)
    out["scope"] = (
        "first-order gap for saved evaluated gradient; not a nonconvex global certificate"
    )
    write_json(directory / "gap_bounds.json", out)
    exact_out = exact_gradient_gap(
        model, exact_h, exact_original_gradient(model, exact_h), baseline, scale, result
    )
    write_json(directory / "exact_gradient_gap.json", exact_out)


def conic_worker(directory):
    model = load_three_channel()
    problem, variables, capacity_constraint = convex_model(model)
    if not problem.is_dcp():
        raise ValueError("Envelope model is not convex")
    start = time.monotonic()
    error = None
    try:
        problem.solve(
            solver="CLARABEL",
            max_iter=300,
            time_limit=float(CONIC_SECONDS - 10),
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-11,
            verbose=False,
        )
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    available = {
        key: value.value is not None if hasattr(value, "value") else True
        for key, value in variables.items()
    }
    arrays = {
        key: (
            np.asarray(value.value if hasattr(value, "value") else value)
            if available[key]
            else np.empty(0)
        )
        for key, value in variables.items()
    }
    arrays["capacity_dual"] = np.asarray(
        capacity_constraint.dual_value if capacity_constraint.dual_value is not None else []
    )
    for i, constraint in enumerate(problem.constraints):
        value = constraint.dual_value
        if isinstance(value, list):
            for j, part in enumerate(value):
                arrays[f"dual_{i}_{j}"] = np.asarray(part if part is not None else [])
        else:
            arrays[f"dual_{i}"] = np.asarray(value if value is not None else [])
    save_arrays(directory / "conic_raw.npz", **arrays)
    objective = problem.value
    finite_objective = objective is not None and np.isfinite(objective)
    write_json(
        directory / "conic_raw.json",
        {
            "status": problem.status,
            "objective": float(objective) if finite_objective else None,
            "objective_raw_text": str(objective),
            "seconds": time.monotonic() - start,
            "error": error,
            "iterations": getattr(problem.solver_stats, "num_iters", None),
            "available": available,
        },
    )
    h, dual = (variables["flow"].value, capacity_constraint.dual_value)
    if h is None or dual is None or (not np.all(np.isfinite(h))) or (not np.all(np.isfinite(dual))):
        raise ValueError("No finite conic support point and capacity dual; no new solve allowed")
    write_json(directory / "lower_certificate.json", exact_support_lower(model, h, dual))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(directory):
    out = {"new_vi_calls": 0, "new_price_calls": 0, "global_interval_closed": False}
    upper = json.loads((directory / "upper_certificate.json").read_text(encoding="utf-8"))
    lower = json.loads((directory / "lower_certificate.json").read_text(encoding="utf-8"))
    U, L = (F(upper["upper_exact"]), F(lower["lower_exact"]))
    if L > U:
        raise ValueError("Lower bound exceeds exact feasible upper; evidence fails")
    width = U - L
    tolerance = F(1, 10**7) * max(F(1), abs(U))
    with np.load(VI_DIRECTORY / "data/vi_raw.npz", allow_pickle=False) as saved:
        vi_h = saved["flow"].copy()
    out.update(
        upper=upward(U),
        lower=downward(L),
        width=upward(width),
        relative_width=upward(width / max(F(1), abs(U))),
        tolerance=downward(tolerance),
        global_interval_closed=bool(width <= tolerance),
        upper_source=upper["source"],
        unpriced_numeric_cost=load_three_channel().cost(vi_h),
        unpriced_cost_is_exact_feasible_upper=False,
    )
    gap_path = directory / "exact_gradient_gap.json"
    if gap_path.exists():
        out["first_order_gap"] = json.loads(gap_path.read_text(encoding="utf-8"))
        out["first_order_gap"].pop("gradient_exact", None)
    return out
