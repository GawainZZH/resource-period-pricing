from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT.parent / "v3.21_replacement_network_20261024"
spec = importlib.util.spec_from_file_location(
    "sealed_static", Path(__file__).with_name("two_class_network.py")
)
static = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static)


def dot(a, b):
    return sum((F(x) * F(y) for x, y in zip(a, b)), F(0))


def compact(rows):
    """Normalize only by positive factors; retain the tightest parallel row."""
    out = {}
    for a, b in rows:
        a, b = (tuple(map(F, a)), F(b))
        scale = next((abs(v) for v in a if v), None)
        if scale is None:
            if b < 0:
                return [(tuple((F(0) for _ in a)), F(-1))]
            continue
        key = tuple((v / scale for v in a))
        rhs = b / scale
        out[key] = min(rhs, out.get(key, rhs))
    return sorted(out.items())


def project_last(rows):
    pos, neg, zero = ([], [], [])
    for a, b in compact(rows):
        if a[-1] > 0:
            pos.append((a, b))
        elif a[-1] < 0:
            neg.append((a, b))
        else:
            zero.append((a[:-1], b))
    if len(pos) * len(neg) + len(zero) > 20000:
        raise RuntimeError("Predeclared elimination row ceiling exceeded")
    for a, b in pos:
        for c, d in neg:
            zero.append(
                (
                    tuple((a[i] / a[-1] - c[i] / c[-1] for i in range(len(a) - 1))),
                    b / a[-1] - d / c[-1],
                )
            )
    return compact(zero)


def intersection(a, b, c, d):
    det = a[0] * c[1] - a[1] * c[0]
    if not det:
        return None
    return ((b * c[1] - a[1] * d) / det, (a[0] * d - b * c[0]) / det)


def feasible(x, rows):
    return all((dot(a, x) <= b for a, b in rows))


def polygon_vertices(rows):
    rows = compact(rows)
    ans = set()
    for (a, b), (c, d) in combinations(rows, 2):
        x = intersection(a, b, c, d)
        if x is not None and feasible(x, rows):
            ans.add(x)
    return sorted(ans)


def has_recession(rows):
    directions = [(F(1), F(0)), (F(-1), F(0)), (F(0), F(1)), (F(0), F(-1))]
    for a, _ in rows:
        if any(a):
            directions.extend([(a[1], -a[0]), (-a[1], a[0])])
    return any((all((dot(a, d) <= 0 for a, _ in rows)) for d in directions))


def value(x, H, g, c):
    return (
        F(c) + dot(g, x) + sum((x[i] * F(H[i][j]) * x[j] for i in range(2) for j in range(2))) / 2
    )


def quadratic_minimum(rows, H, g, c):
    rows = compact(rows)
    H = tuple((tuple(map(F, row)) for row in H))
    g, c = (tuple(map(F, g)), F(c))
    det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
    if H[0][1] != H[1][0] or H[0][0] < 0 or H[1][1] < 0 or (det < 0):
        raise ValueError("Only convex symmetric quadratic branches are supported")
    if any((not any(a) and b < 0 for a, b in rows)):
        return None
    if has_recession(rows):
        raise ValueError("Only bounded flow projections are supported")
    verts = polygon_vertices(rows)
    if not verts:
        return None
    candidates = set(verts)
    if det:
        x = intersection(H[0], -g[0], H[1], -g[1])
        if feasible(x, rows):
            candidates.add(x)
    for p, q in combinations(verts, 2):
        d = tuple((q[i] - p[i] for i in range(2)))
        denom = sum((d[i] * H[i][j] * d[j] for i in range(2) for j in range(2)))
        if denom:
            grad = tuple((g[i] + dot(H[i], p) for i in range(2)))
            t = -dot(grad, d) / denom
            if 0 <= t <= 1:
                candidates.add(tuple((p[i] + t * d[i] for i in range(2))))
    x = min(candidates, key=lambda z: (value(z, H, g, c), z))
    grad = tuple((g[i] + dot(H[i], x) for i in range(2)))
    margins = [dot(grad, tuple((v[i] - x[i] for i in range(2)))) for v in verts]
    if min(margins) < 0:
        raise ArithmeticError("Exact convex first-order verification failed")
    return dict(
        x=x,
        value=value(x, H, g, c),
        vertices=verts,
        first_order_min=min(margins),
        candidate_count=len(candidates),
    )


def flow_rows(k):
    return [
        ((F(-1), F(0)), F(0)),
        ((F(1), F(0)), F(1)),
        ((F(0), F(-1)), F(0)),
        ((F(0), F(1)), F(1)),
        ((F(1), F(1)), F(k)),
    ]


def affine_arrivals(data, state):
    samples = [
        static.matrix_arrivals(data, state, tuple(map(F, x))) for x in [(0, 0), (1, 0), (0, 1)]
    ]
    return [
        (samples[0][j], samples[1][j] - samples[0][j], samples[2][j] - samples[0][j])
        for j in range(4)
    ]


def cost_branch(data, state, bits):
    arrivals = affine_arrivals(data, state)
    for j in (0, 2):
        if max(
            (
                arrivals[j][0] + dot(arrivals[j][1:], v)
                for v in polygon_vertices(flow_rows(F(state["capacity"])))
            )
        ) >= F(data["deadlines"][j]):
            raise ValueError("Fast deadline envelope is outside the sealed design")
    flags = (0, bits[0], 0, bits[1])
    C, signs = ([], [])
    for j, a in enumerate(arrivals):
        deadline, rate = (F(data["deadlines"][j]), F(data["lateness_rates"][j]))
        C.append(
            (
                (1 + rate * flags[j]) * a[0] - rate * flags[j] * deadline,
                (1 + rate * flags[j]) * a[1],
                (1 + rate * flags[j]) * a[2],
            )
        )
        if j in (1, 3):
            sign = -1 if flags[j] else 1
            signs.append((tuple((sign * z for z in a[1:])), sign * (deadline - a[0])))
    f = [tuple((C[a][i] - C[b][i] for i in range(3))) for a, b in [(0, 1), (2, 3)]]
    h0 = (0, 1, 0, 1)
    slopes = ((1, 0), (-1, 0), (0, 1), (0, -1))
    c = sum((F(h0[j]) * C[j][0] for j in range(4)))
    g = tuple(
        (sum((h0[j] * C[j][i + 1] + slopes[j][i] * C[j][0] for j in range(4))) for i in range(2))
    )
    H = tuple(
        (
            tuple(
                (
                    sum((slopes[j][i] * C[j][l + 1] + slopes[j][l] * C[j][i + 1] for j in range(4)))
                    for l in range(2)
                )
            )
            for i in range(2)
        )
    )
    return dict(f=f, H=H, g=g, c=c, signs=signs, costs=C)


def price_rows(kind, cap):
    k = F(cap)
    if kind == "shared_zero":
        if k != 0:
            raise ValueError("Shared zero domain requires cap=0")
        kind = "contemporaneous"
    if kind == "contemporaneous":
        return [((1, 0), k / 2), ((-1, 0), k / 2), ((0, 1), k / 2), ((0, -1), k / 2)]
    if kind == "node":
        return [((1, -1), 0), ((-1, 1), 0), ((1, 0), 2 * k), ((-1, 0), k / 2)]
    if kind == "corridor":
        L, U, R = (-F(13, 2) * k, 5 * k, k / 2)
        return [
            ((1, 0), U + R),
            ((-1, 0), -L),
            ((0, 1), U),
            ((0, -1), -L + R),
            ((1, 1), 2 * U),
            ((-1, -1), -2 * L),
        ]
    raise ValueError(kind)


def expanded_rows(X, signs, f, I, D):
    n = len(I)
    rows = [(tuple(a) + (F(0),) * n, F(b)) for a, b in X + signs]
    for q in I:
        a, b = X[q]
        rows.append((tuple((-v for v in a)) + (F(0),) * n, -b))
    for j in range(n):
        a = [F(0)] * (2 + n)
        a[2 + j] = -1
        rows.append((tuple(a), F(0)))
    for a, b in D:
        coefficients = [-sum((F(a[j]) * f[j][i + 1] for j in range(2))) for i in range(2)]
        coefficients += [-dot(a, X[q][0]) for q in I]
        rows.append((tuple(coefficients), F(b) + sum((F(a[j]) * f[j][0] for j in range(2)))))
    return compact(rows)


def multiplier_witness(rows, x, n):
    remaining = compact([(a[2:], b - dot(a[:2], x)) for a, b in rows])
    if n == 0:
        if not feasible((), remaining):
            raise ArithmeticError("No zero-dimensional multiplier witness")
        return ()
    if n == 1:
        lo, hi = (F(0), None)
        for a, b in remaining:
            if a[0] < 0:
                lo = max(lo, b / a[0])
            elif a[0] > 0:
                hi = b / a[0] if hi is None else min(hi, b / a[0])
            elif b < 0:
                raise ArithmeticError("Infeasible multiplier equality")
        result = (lo,)
        if hi is not None and lo > hi:
            raise ArithmeticError("Empty multiplier interval")
    else:
        verts = polygon_vertices(remaining)
        if not verts:
            raise ArithmeticError("No rational multiplier vertex")
        result = min(verts, key=lambda v: (sum(v), v))
    if not feasible(result, remaining):
        raise ArithmeticError("Invalid multiplier witness")
    return result


def solve_task(data, task, seconds=300):
    start = time.monotonic()
    state = next((s for s in data["states"] if s["id"] == task["state"]))
    X = flow_rows(F(state["capacity"]))
    is_so = task["type"] == "SO"
    subsets = [()] if is_so else [I for size in range(3) for I in combinations(range(5), size)]
    branches, best = ([], None)
    for bits in product((0, 1), repeat=2):
        model = cost_branch(data, state, bits)
        for I in subsets:
            if time.monotonic() - start > seconds:
                raise TimeoutError("Task time budget exceeded; no global bound released")
            if is_so:
                original = compact(X + model["signs"])
            else:
                original = expanded_rows(
                    X, model["signs"], model["f"], I, price_rows(task["class_name"], F(task["cap"]))
                )
            projected = original
            for _ in I:
                projected = project_last(projected)
            ans = quadratic_minimum(projected, model["H"], model["g"], model["c"])
            entry = dict(
                bits=bits,
                normal_generators=I,
                original_rows=original,
                projected_rows=projected,
                H=model["H"],
                g=model["g"],
                c=model["c"],
                f=model["f"],
                status="empty" if ans is None else "optimal",
            )
            if ans is not None:
                entry.update(ans)
                if not is_so:
                    lam = multiplier_witness(original, ans["x"], len(I))
                    fvalue = tuple((row[0] + dot(row[1:], ans["x"]) for row in model["f"]))
                    d = tuple(
                        (
                            -fvalue[j] - sum((lam[l] * X[q][0][j] for l, q in enumerate(I)))
                            for j in range(2)
                        )
                    )
                    entry.update(multipliers=lam, price_differences=d)
                if best is None or (ans["value"], ans["x"]) < (best["value"], best["x"]):
                    best = dict(entry, branch_index=len(branches))
            branches.append(entry)
    if best is None:
        raise ArithmeticError("All branches empty; expected a nonempty frozen problem")
    theta = None
    if not is_so:
        kind = "contemporaneous" if task["class_name"] == "shared_zero" else task["class_name"]
        theta = static.reverse_prices(data, kind, F(task["cap"]), best["price_differences"])
    return dict(
        task=task,
        status="complete_exact",
        cost_lower=best["value"],
        cost_upper=best["value"],
        x=best["x"],
        theta=theta,
        price_differences=best.get("price_differences"),
        best_branch=best["branch_index"],
        branches=branches,
        branches_expected=4 * len(subsets),
        branches_completed=len(branches),
        solve_seconds=time.monotonic() - start,
    )


def jsonable(obj):
    if isinstance(obj, F):
        return str(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(jsonable(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
