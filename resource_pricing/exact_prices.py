from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
BOUND_DIR = ROOT / "data/protected_system_cost"
CLASSES = ("corridor", "node", "contemporaneous", "all_resource")


def value_interval(lower, upper):
    lower, upper = (F(lower), F(upper))
    if not 0 < lower <= upper:
        raise ValueError("A positive global lower and a valid equilibrium upper are required")
    return {
        "class_value_lower": bounds().downward(lower),
        "class_value_upper": bounds().upward(upper),
        "class_value_width": bounds().upward(upper - lower),
        "relative_cost_excess_percent_upper": bounds().upward(100 * (upper / lower - 1)),
        "relative_cost_excess_exact": str(upper / lower - 1),
    }


def read_module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def bounds():
    return read_module("_exact_price_old_bounds", Path(__file__).with_name("protected_bounds.py"))


def load_target():
    api = bounds()
    m = api.load_three_channel()
    stored = json.loads((BOUND_DIR / "upper_certificate.json").read_text(encoding="utf-8"))
    h = [F(v) for v in stored["upper_flow"]]
    if not api.is_exact_feasible(m, h):
        raise ValueError("Frozen target must be exactly feasible")
    loads = api.exact_loads(m.occupation, h)
    cost = [F(float(v)) for v in m.base]
    for r, (load, capacity, alpha) in enumerate(zip(loads, m.capacity, m.alpha)):
        utilization = load / F(float(capacity))
        delay = F(float(alpha)) * utilization * utilization
        risk = F(float(m.rho)) * max(F(0), utilization - F(float(m.threshold))) ** 2
        for j in np.flatnonzero(m.destination[r]):
            cost[j] += F(float(m.destination[r, j])) * delay
        for j in np.flatnonzero(m.occupation[r]):
            cost[j] += F(float(m.occupation[r, j])) * risk
    return (m, h, cost)


def price_map(model, name):
    R = model.resource_grid * model.occupation
    if name == "corridor":
        indices = np.flatnonzero(model.row_resource < 114)
    elif name == "node":
        indices = np.flatnonzero(model.row_resource >= 114)
    elif name == "contemporaneous":
        start = model.departure_period * model.choices[:, 2]
        R = R * (
            (model.row_start[:, None] >= start)
            & (model.row_start[:, None] < start + model.departure_period)
        )
        indices = np.arange(len(R))
    elif name == "all_resource":
        indices = np.arange(len(R))
    else:
        raise ValueError("Unknown fixed price class")
    indices = indices[np.any(R[indices] != 0, axis=1)]
    return (R[indices], indices)


def dot(row, values):
    return sum((F(float(row[j])) * values[j] for j in np.flatnonzero(row)), F(0))


def solve_affine(A, b, raw):
    import sympy as sp

    n = len(raw)
    rows = list(
        dict.fromkeys((tuple([F(float(v)) for v in row] + [F(rhs)]) for row, rhs in zip(A, b)))
    )
    if not rows:
        return ([F(float(v)).limit_denominator(10**9) for v in raw], 0)
    matrix = sp.Matrix([[sp.Rational(v.numerator, v.denominator) for v in row] for row in rows])
    reduced, pivots = matrix.rref(normalize_last=True)
    if n in pivots:
        return (None, len(pivots) - 1)
    free = [j for j in range(n) if j not in pivots]
    out = [F(0) for _ in raw]
    for j in free:
        out[j] = F(float(raw[j])).limit_denominator(10**9)
    for i, j in enumerate(pivots):
        rhs = reduced[i, n] - sum(
            (reduced[i, k] * sp.Rational(out[k].numerator, out[k].denominator) for k in free),
            sp.S.Zero,
        )
        out[j] = F(int(rhs.p), int(rhs.q))
    return (out, len(pivots))


def rational_face(Aeq, beq, Aub, bub, raw):
    raw = np.asarray(raw, float)
    Aeq = np.asarray(Aeq, float).reshape((-1, len(raw)))
    Aub = np.asarray(Aub, float).reshape((-1, len(raw)))
    support = np.flatnonzero(raw > 1e-08)
    active = np.flatnonzero(np.array([float(v) for v in bub]) - Aub @ raw <= 1e-07)
    matrix = np.vstack([Aeq[:, support], Aub[active][:, support]])
    rhs = list(beq) + [bub[i] for i in active]
    values, rank = solve_affine(matrix, rhs, raw[support])
    if values is None:
        return {"exact_feasible": False, "reason": "Inconsistent selected face"}
    h = [F(0) for _ in raw]
    for j, value in zip(support, values):
        h[j] = value
    valid = (
        all((v >= 0 for v in h))
        and all((dot(row, h) == F(rhs) for row, rhs in zip(Aeq, beq)))
        and all((dot(row, h) <= F(rhs) for row, rhs in zip(Aub, bub)))
    )
    return {
        "exact_feasible": bool(valid),
        "vector": h,
        "rank": rank,
        "support": len(support),
        "active_rows": len(active),
    }


def check_flow_witness(model, target, cost, R, z):
    feasible = bounds().is_exact_feasible(model, z)
    observed = all((dot(row, z) <= dot(row, target) for row in R))
    gap = sum((v * (h - x) for v, h, x in zip(cost, target, z)), F(0))
    return {
        "exact_feasible": bool(feasible and observed),
        "physical_feasible": bool(feasible),
        "occupation_dominance": bool(observed),
        "gap_lower_exact": str(max(F(0), gap) if feasible and observed else F(0)),
        "strict_target_exclusion": bool(feasible and observed and (gap > 0)),
    }


def check_price(model, target, cost, R, theta, mu, nu):
    y = bounds().exact_loads(model.occupation, target)
    reduced = [
        cost[j]
        + sum((F(float(R[i, j])) * theta[i] for i in np.flatnonzero(R[:, j])), F(0))
        + sum(
            (
                F(float(model.occupation[r, j])) * mu[r]
                for r in np.flatnonzero(model.occupation[:, j])
            ),
            F(0),
        )
        - sum(
            (
                F(float(model.conservation[w, j])) * nu[w]
                for w in np.flatnonzero(model.conservation[:, j])
            ),
            F(0),
        )
        for j in range(len(target))
    ]
    nonnegative = all((v >= 0 for v in theta + mu))
    reduced_feasible = all((v >= 0 for v in reduced))
    flow_complementarity = all((v * h == 0 for v, h in zip(reduced, target)))
    capacity_complementarity = all(
        (v * (F(float(k)) - load) == 0 for v, k, load in zip(mu, model.capacity, y))
    )
    passed = (
        bounds().is_exact_feasible(model, target)
        and nonnegative
        and reduced_feasible
        and flow_complementarity
        and capacity_complementarity
    )
    return {
        "exact_equilibrium": bool(passed),
        "nonnegative_parameters": bool(nonnegative),
        "all_reduced_costs_nonnegative": bool(reduced_feasible),
        "all_flow_complementarity": bool(flow_complementarity),
        "all_capacity_complementarity": bool(capacity_complementarity),
        "max_price_per_minute": float(max(theta, default=F(0))),
    }


def reconstruct_price(
    model, target, cost, R, capacity_dual, price_dual, equality_dual, baseline, scale
):
    m, k, w = (len(model.capacity), len(R), len(model.demand))
    theta0 = -np.asarray(price_dual) * scale
    mu0 = -np.asarray(capacity_dual) * scale
    nu0 = np.asarray(equality_dual) * scale + baseline
    loads = bounds().exact_loads(model.occupation, target)
    price_support = np.flatnonzero(theta0 > 1e-08)
    capacity_support = np.array(
        [r for r in range(m) if mu0[r] > 1e-08 and loads[r] == F(float(model.capacity[r]))], int
    )
    raw = np.r_[theta0[price_support], mu0[capacity_support], nu0]
    columns = np.column_stack(
        [R[price_support].T, model.occupation[capacity_support].T, -model.conservation.T]
    )
    reduced0 = (
        np.array([float(v) for v in cost])
        + R.T @ theta0
        + model.occupation.T @ mu0
        - model.conservation.T @ nu0
    )
    active = np.array([j for j, h in enumerate(target) if h > 0 or abs(reduced0[j]) <= 1e-07], int)
    value, rank = solve_affine(columns[active], [-cost[j] for j in active], raw)
    if value is None:
        return {"exact_equilibrium": False, "reason": "Selected price equations inconsistent"}
    theta, mu = ([F(0) for _ in range(k)], [F(0) for _ in range(m)])
    p = len(price_support)
    a = len(capacity_support)
    for i, v in zip(price_support, value[:p]):
        theta[i] = v
    for i, v in zip(capacity_support, value[p : p + a]):
        mu[i] = v
    nu = value[p + a :]
    report = check_price(model, target, cost, R, theta, mu, nu)
    report.update(
        theta_exact=[str(v) for v in theta],
        capacity_multipliers_exact=[str(v) for v in mu],
        demand_multipliers_exact=[str(v) for v in nu],
        rank=rank,
        positive_price_parameters=len(price_support),
    )
    return report


def dual_gap_bound(
    model, target, cost, R, capacity_dual, price_dual, equality_dual, baseline, scale
):
    lambdas = [min(F(0), F(float(v))) * F(float(scale)) for v in capacity_dual]
    etas = [min(F(0), F(float(v))) * F(float(scale)) for v in price_dual]
    nu = [F(float(v)) * F(float(scale)) + F(float(b)) for v, b in zip(equality_dual, baseline)]
    coverage = [
        sum(
            (
                F(float(model.occupation[r, j])) * lambdas[r]
                for r in np.flatnonzero(model.occupation[:, j])
            ),
            F(0),
        )
        + sum((F(float(R[r, j])) * etas[r] for r in np.flatnonzero(R[:, j])), F(0))
        for j in range(len(target))
    ]
    for w, row in enumerate(model.conservation):
        nu[w] += min(F(0), min((cost[j] - coverage[j] - nu[w] for j in np.flatnonzero(row))))
    owner = model.conservation.argmax(axis=0)
    feasible = all((coverage[j] + nu[owner[j]] <= cost[j] for j in range(len(target))))
    lower = sum((F(float(k)) * v for k, v in zip(model.capacity, lambdas)), F(0))
    lower += sum((dot(row, target) * v for row, v in zip(R, etas)), F(0))
    lower += sum((F(float(q)) * v for q, v in zip(model.demand, nu)), F(0))
    gap = sum((v * h for v, h in zip(cost, target)), F(0)) - lower
    if not feasible or gap < 0:
        raise ValueError("Exact dual validation failed")
    return {
        "exact_dual_feasible": True,
        "gap_upper_exact": str(gap),
        "gap_upper": bounds().upward(gap),
    }
