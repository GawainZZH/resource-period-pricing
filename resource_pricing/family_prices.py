from fractions import Fraction as R
from pathlib import Path
import hashlib
import importlib.util
import json
import sys


def oracle_module():
    name = "_bounded_a_exact_oracle"
    if name not in sys.modules:
        path = Path(__file__).with_name("analytical_family.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, R)):
        raise TypeError("exact rational strings, integers or Fractions required")
    return R(value)


def json_safe(value):
    if isinstance(value, R):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def make_family(data):
    values = {}
    for key, value in data.items():
        values[key] = (
            value
            if key == "n"
            else (
                tuple(map(rational, value)) if isinstance(value, (tuple, list)) else rational(value)
            )
        )
    return oracle_module().Family(**values)


def at(value, w):
    return value[w] if isinstance(value, tuple) else value


def feasible(f, x):
    return (
        len(x) == f.n
        and all((isinstance(z, R) and 0 <= z <= q for z, q in zip(x, f.Q)))
        and (sum(x) <= f.K)
    )


def difference(f, w, z, u):
    slow = at(f.S, w) + f.delta[w] * (f.Q[w] - z)
    return at(f.F, w) + f.beta[w] * z + f.gamma * u - slow - f.theta[w] * max(slow - f.L[w], R(0))


def cost(f, x):
    u = sum(x, R(0))
    result = R(0)
    for w, z in enumerate(x):
        slow = at(f.S, w) + f.delta[w] * (f.Q[w] - z)
        result += z * (at(f.F, w) + f.beta[w] * z + f.gamma * u)
        result += (f.Q[w] - z) * (slow + f.theta[w] * max(slow - f.L[w], R(0)))
    return result


def so_certificate(f, x):
    if not feasible(f, x):
        return dict(verified=False, feasible=False)
    u = sum(x, R(0))
    lower, upper = (R(0), R(0) if u < f.K else None)
    derivatives = []
    for w, z in enumerate(x):
        q = f.Q[w] - z
        arrival = at(f.S, w) + f.delta[w] * q
        plain = at(f.F, w) + 2 * f.beta[w] * z - at(f.S, w) - 2 * f.delta[w] * q + 2 * f.gamma * u
        left = right = plain
        if arrival > f.L[w]:
            left = right = plain - f.theta[w] * (arrival - f.L[w] + f.delta[w] * q)
        elif arrival == f.L[w]:
            left = plain - f.theta[w] * f.delta[w] * q
        derivatives.append((left, right))
        if z < f.Q[w]:
            lower = max(lower, -right)
        if z > 0:
            upper = -left if upper is None else min(upper, -left)
    return dict(
        verified=upper is None or lower <= upper,
        feasible=True,
        multiplier_interval=(lower, upper),
        subgradients=derivatives,
    )


def vi_gap(f, x, p):
    if not feasible(f, x):
        raise ValueError("infeasible VI witness")
    vector = [difference(f, w, z, sum(x)) + p for w, z in enumerate(x)]
    remaining, minimum = (f.K, R(0))
    for w in sorted(range(f.n), key=lambda w: vector[w]):
        if vector[w] >= 0:
            break
        amount = min(remaining, f.Q[w])
        minimum += amount * vector[w]
        remaining -= amount
    return sum((v * z for v, z in zip(vector, x)), R(0)) - minimum


def geometry(f):
    """Independent private-cost breakpoints; no oracle branch internals.

    Invert each local continuous affine segment at its breakpoints. This is
    certificate reconstruction, not a call to an SO or price optimizer.
    """
    rows, knots = ([], set())
    for w in range(f.n):
        edges = [R(0), f.Q[w]]
        kink = (at(f.S, w) + f.delta[w] * f.Q[w] - f.L[w]) / f.delta[w]
        if f.theta[w] and 0 < kink < f.Q[w]:
            edges.append(kink)
        edges.sort()
        row = [(z, -difference(f, w, z, R(0))) for z in edges]
        rows.append(row)
        knots.update((mu for _, mu in row))

    def flow(mu):
        result = []
        for row in rows:
            if mu >= row[0][1]:
                z = row[0][0]
            elif mu <= row[-1][1]:
                z = row[-1][0]
            else:
                for (a, ma), (b, mb) in zip(row, row[1:]):
                    if mb <= mu <= ma:
                        z = a + (b - a) * (ma - mu) / (ma - mb)
                        break
            result.append(z)
        return tuple(result)

    knots = sorted(knots)
    hits = []
    for a, b in zip(knots, knots[1:]):
        ua, ub = (sum(flow(a)), sum(flow(b)))
        if ua == ub == f.K:
            hits.append(b)
        elif ub <= f.K <= ua and ua != ub:
            hits.append(a + (b - a) * (ua - f.K) / (ua - ub))
    mu = max(hits)
    xk = flow(mu)
    pk = mu - f.gamma * f.K
    points = [(pk, xk)]
    points += [(m - f.gamma * sum(flow(m)), flow(m)) for m in knots if m > mu]
    return points


def expected_candidates(f, interval):
    """Certificate obligations from independent knots and quadratic costs."""
    lo, hi = interval
    points = geometry(f)
    pk, xk = points[0]
    expected = []

    def add(kind, p, x, span):
        expected.append(dict(kind=kind, p=p, x=x, cost=cost(f, x), interval=span))

    if lo <= pk:
        add("capacity", lo, xk, (lo, min(hi, pk)))
    for (a, xa), (b, xb) in zip(points, points[1:]):
        left, right = (max(lo, a), min(hi, b))
        if left > right or right <= pk:
            continue

        def interp(p):
            return tuple((x + (y - x) * (p - a) / (b - a) for x, y in zip(xa, xb)))

        span = (left, right)
        xl, xr = (interp(left), interp(right))
        if xa == xb:
            add("constant", left, xl, span)
            continue
        for p in sorted({left, right}):
            if p > pk:
                add("endpoint", p, interp(p), span)
        if left < right:
            c0, c1 = (cost(f, xl), cost(f, xr))
            cm = cost(f, tuple(((x + y) / 2 for x, y in zip(xl, xr))))
            aa = 2 * (c1 + c0 - 2 * cm)
            bb = c1 - c0 - aa
            t = -bb / (2 * aa)
            if 0 < t < 1:
                p = left + t * (right - left)
                add("stationary", p, interp(p), span)
    pz, xz = points[-1]
    if hi >= max(lo, pz) and hi > pk:
        add("constant", max(lo, pz), xz, (max(lo, pz), hi))
    return expected


def audit_global(f, result, interval):
    expected = expected_candidates(f, interval)

    def signature(row):
        return (row["kind"], row["p"], tuple(row["x"]), row["cost"], tuple(row["interval"]))

    actual = result.get("candidates", [])
    coverage = {signature(c) for c in expected} == {signature(c) for c in actual}
    checks = dict(
        candidate_branch_coverage=coverage,
        candidate_costs=all((feasible(f, c["x"]) and cost(f, c["x"]) == c["cost"] for c in actual)),
        candidate_vi=all((feasible(f, c["x"]) and vi_gap(f, c["x"], c["p"]) == 0 for c in actual)),
    )
    best = min(expected, key=lambda c: (c["cost"], c["p"], c["x"]))
    checks["selected_minimum"] = all((result[k] == best[k] for k in ("cost", "p", "x")))
    return dict(
        verified=all(checks.values()),
        checks=checks,
        expected_candidates=len(expected),
        actual_candidates=len(actual),
    )


def decode_global(row):
    result = {k: rational(row[k]) for k in ("cost", "p")}
    result["x"] = tuple(map(rational, row["x"]))
    result["candidates"] = [
        dict(
            kind=c["kind"],
            p=rational(c["p"]),
            cost=rational(c["cost"]),
            x=tuple(map(rational, c["x"])),
            interval=tuple(map(rational, c["interval"])),
        )
        for c in row["candidates"]
    ]
    return result
