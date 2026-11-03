import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from fractions import Fraction as F
import numpy as np
import sympy as sp
from scipy.optimize import linprog

OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def q(x):
    return x if isinstance(x, F) else F(float(x))


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


def rows(a):
    return [[q(v) for v in row] for row in a]


def mv(a, x):
    return [dot(row, x) for row in a]


def transpose(a):
    return list(map(list, zip(*a)))


class RationalModel:

    def __init__(self, A, K, E, Q, base, alpha, rho, eta):
        self.A, self.E = (rows(A), rows(E))
        self.K, self.Q, self.base, self.alpha = ([q(v) for v in a] for a in (K, Q, base, alpha))
        self.rho, self.eta = (q(rho), q(eta))
        self.n = len(base)
        assert all((sum((row[j] for row in self.E)) == 1 for j in range(self.n)))

    def feasible(self, h, extra=None, bound=None):
        return (
            min(h) >= 0
            and mv(self.E, h) == self.Q
            and all((v <= k for v, k in zip(mv(self.A, h), self.K)))
            and (extra is None or all((v <= k for v, k in zip(mv(extra, h), bound))))
        )

    def costs(self, h):
        y = mv(self.A, h)
        u = [v / k for v, k in zip(y, self.K)]
        v = [max(F(0), a - self.eta) for a in u]
        d = [a * z * z + self.rho * w * w for a, z, w in zip(self.alpha, u, v)]
        dg = [3 * a * z * z + self.rho * (w * w + 2 * z * w) for a, z, w in zip(self.alpha, u, v)]
        c = [b + x for b, x in zip(self.base, mv(transpose(self.A), d))]
        g = [b + x for b, x in zip(self.base, mv(transpose(self.A), dg))]
        return (dot(c, h), c, g)


def repair_point(p, point, extra=None, bound=None, max_change=1e-08):
    """Reconstruct nearby exact active constraints; reject any infeasible result."""
    raw = [q(v) for v in point]
    free = [i for i, v in enumerate(raw) if v > F("1e-9")]
    if not free:
        raise ValueError("No positive variables")
    A = p.A + (extra or [])
    K = p.K + (bound or [])
    active = [i for i, (row, k) in enumerate(zip(A, K)) if abs(float(dot(row, raw) - k)) < 1e-08]
    C = p.E + [A[i] for i in active]
    b = p.Q + [K[i] for i in active]
    m = sp.Matrix([[sp.Rational(row[i].numerator, row[i].denominator) for i in free] for row in C])
    independent = list(m.T.rref()[1])
    mr = m[independent, :]
    pivots = list(mr.rref()[1])
    other = [i for i in range(len(free)) if i not in pivots]
    rhs = sp.Matrix([sp.Rational(b[i].numerator, b[i].denominator) for i in independent])
    if other:
        rhs -= mr[:, other] * sp.Matrix(
            [sp.Rational(raw[free[i]].numerator, raw[free[i]].denominator) for i in other]
        )
    solved = mr[:, pivots].inv() * rhs
    result = [F(0)] * p.n
    for i in other:
        result[free[i]] = raw[free[i]]
    for i, v in zip(pivots, solved):
        result[free[i]] = F(int(v.p), int(v.q))
    if max((abs(float(x - y)) for x, y in zip(result, raw))) > max_change:
        raise ValueError("Repair exceeds frozen change bound")
    if not p.feasible(result, extra, bound):
        raise ValueError("Exact reconstructed point is infeasible")
    return result


def linear_program(p, h, c, extra=None, bound=None):
    A = p.A + (extra or [])
    K = p.K + (bound or [])
    baseline = [min((c[j] for j, v in enumerate(row) if v)) for row in p.E]
    centered = [v - b for v, b in zip(c, mv(transpose(p.E), baseline))]
    scale = max([abs(v) for v in centered] + [F(1)])
    lp = linprog(
        np.array([float(v / scale) for v in centered]),
        A_ub=np.array(A, dtype=float),
        b_ub=np.array(K, dtype=float),
        A_eq=np.array(p.E, dtype=float),
        b_eq=np.array(p.Q, dtype=float),
        bounds=(0, None),
        method="highs-ds",
        options=OPTIONS,
    )
    if not lp.success:
        raise RuntimeError(lp.message)
    y = [min(F(0), q(v) * scale) for v in lp.ineqlin.marginals]
    z = [q(v) * scale + b for v, b in zip(lp.eqlin.marginals, baseline)]
    reduced = [v - a - b for v, a, b in zip(c, mv(transpose(A), y), mv(transpose(p.E), z))]
    for i, row in enumerate(p.E):
        z[i] += min(F(0), min((reduced[j] for j, v in enumerate(row) if v)))
    assert all((a + b <= v for a, b, v in zip(mv(transpose(A), y), mv(transpose(p.E), z), c)))
    lower = dot(K, y) + dot(p.Q, z)
    x = repair_point(p, lp.x, extra, bound)
    upper = dot(c, x)
    assert lower <= upper
    return dict(lower=lower, upper=upper, flow=x, inequality_dual=y, equality_dual=z)


def oracle_interval(p, h, c):
    assert p.feasible(h)
    lp = linear_program(p, h, c)
    base = dot(c, h)
    return dict(lower=base - lp["upper"], upper=base - lp["lower"], witness=lp)


def class_interval(p, h, design):
    assert p.feasible(h)
    D = rows(design)
    DT = transpose(D)
    _, c, _ = p.costs(h)
    baseline = [min((c[j] for j, v in enumerate(row) if v)) for row in p.E]
    centered = [v - b for v, b in zip(c, mv(transpose(p.E), baseline))]
    scale = max([abs(v) for v in centered] + [F(1)])
    k = len(DT)
    m = len(p.A)
    r = len(p.Q)
    obj = mv(DT, h) + p.K + p.Q
    matrix = np.hstack(
        (-np.array(D, dtype=float), -np.array(p.A, dtype=float).T, -np.array(p.E, dtype=float).T)
    )
    lp = linprog(
        np.array(obj, dtype=float),
        A_ub=matrix,
        b_ub=np.array([float(v / scale) for v in centered]),
        bounds=[(0, None)] * (k + m) + [(None, None)] * r,
        method="highs-ds",
        options=OPTIONS,
    )
    if not lp.success:
        raise RuntimeError(lp.message)
    theta = [max(F(0), q(v) * scale) for v in lp.x[:k]]
    priced = [v + a for v, a in zip(c, mv(D, theta))]
    direct = oracle_interval(p, h, priced)
    dual = linear_program(p, h, c, DT, mv(DT, h))
    lower = max(F(0), dot(c, h) - dual["upper"])
    upper = direct["upper"]
    assert lower <= upper
    return dict(
        lower=lower,
        upper=upper,
        theta=theta,
        direct=direct,
        flow_witness=dual["flow"],
        independent_lp=dual,
    )
