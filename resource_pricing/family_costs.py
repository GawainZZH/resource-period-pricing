from __future__ import annotations
import time
from fractions import Fraction
import numpy as np


class BudgetExceeded(RuntimeError):
    pass


class Budget:

    def __init__(self, limits, seconds):
        self.limits = dict(limits)
        self.calls = {k: 0 for k in limits}
        self.started = time.monotonic()
        self.deadline = self.started + seconds

    def use(self, kind):
        self.check()
        if self.calls[kind] >= self.limits[kind]:
            raise BudgetExceeded(kind)
        self.calls[kind] += 1

    def check(self, *args):
        if time.monotonic() >= self.deadline:
            raise BudgetExceeded("wall_time")

    def remaining(self):
        self.check()
        return self.deadline - time.monotonic()


class CostAdapter:

    def __init__(self, family):
        self.family = family
        self.n = family.n
        self.Q, self.beta, self.delta, self.theta, self.L = [
            np.asarray(getattr(family, k), dtype=float)
            for k in ("Q", "beta", "delta", "theta", "L")
        ]
        self.F, self.S, self.gamma, self.K = [
            float(getattr(family, k)) for k in ("F", "S", "gamma", "K")
        ]

    def private(self, x):
        x = np.asarray(x)
        af = self.F + self.beta * x + self.gamma * x.sum()
        arrival = self.S + self.delta * (self.Q - x)
        return (af, arrival + self.theta * np.maximum(arrival - self.L, 0))

    def difference(self, x):
        a, b = self.private(x)
        return a - b

    def cost(self, x):
        a, b = self.private(x)
        return float(x @ a + (self.Q - x) @ b)

    def gradient(self, x):
        a, b = self.private(x)
        s = self.Q - x
        eta = self.delta * (1 + self.theta * (self.S + self.delta * s > self.L))
        return a + self.beta * x + self.gamma * x.sum() - b - s * eta

    def system_subgradient(self, x):
        g = self.gradient(x)
        at = self.S + self.delta * (self.Q - x) == self.L
        lower = g - self.theta * self.delta * (self.Q - x) * at
        return (lower, g)

    def hessian(self, x):
        eta = self.delta * (1 + self.theta * (self.S + self.delta * (self.Q - x) > self.L))
        return np.diag(self.beta + eta) + self.gamma * np.ones((self.n, self.n))

    def potential(self, x, p=0.0):
        a = self.S + self.delta * self.Q - self.L
        value = (self.F - self.S - self.delta * self.Q) * x + 0.5 * (self.beta + self.delta) * x * x
        value += (
            self.theta
            / (2 * self.delta)
            * (np.maximum(a - self.delta * x, 0) ** 2 - np.maximum(a, 0) ** 2)
        )
        return float(value.sum() + 0.5 * self.gamma * x.sum() ** 2 + p * x.sum())

    def adjoint(self, x, rate_design, ridge=1e-08, full_flow=None):
        h = np.r_[x, self.Q - x] if full_flow is None else np.asarray(full_flow)
        x, slow = (h[: self.n], h[self.n :])
        free = h > 1e-07
        eta = self.delta * (1 + self.theta * (self.S + self.delta * slow > self.L))
        H = np.zeros((2 * self.n, 2 * self.n))
        H[: self.n, : self.n] = np.diag(self.beta) + self.gamma
        H[self.n :, self.n :] = np.diag(eta)
        H = H[np.ix_(free, free)]
        H += ridge * max(1.0, np.diag(H).max()) * np.eye(len(H))
        A = np.hstack((np.eye(self.n), np.eye(self.n)))[:, free]
        if self.K - x.sum() <= 2e-06 * max(1.0, abs(self.K)):
            A = np.vstack((A, np.r_[np.ones(self.n), np.zeros(self.n)][free]))
        cf = self.F + self.beta * x + self.gamma * x.sum()
        arrival = self.S + self.delta * slow
        cs = arrival + self.theta * np.maximum(arrival - self.L, 0)
        g = np.r_[cf + self.beta * x + self.gamma * x.sum(), cs + slow * eta][free]
        matrix = np.block([[H, A.T], [A, np.zeros((len(A), len(A)))]])
        adj = np.linalg.lstsq(matrix.T, np.r_[g, np.zeros(len(A))], rcond=1e-10)[0][: len(H)]
        design = np.zeros((2 * self.n, 2))
        design[: self.n, 0] = rate_design[0]
        design[self.n :, 1] = -rate_design[1]
        return -design[free].T @ adj


def protection_check(f, p):
    step, tf, wf, ts, ws = [p[k] for k in ("step", "TF", "WF", "TS", "WS")]
    checks = {
        "positive_grid": step > 0,
        "aligned": all(((z / step).denominator == 1 for z in (tf, wf, ts, ws))),
        "fast_start": step <= tf < f.F,
        "fast_finish": max((f.F + b * q + f.gamma * f.K for b, q in zip(f.beta, f.Q))) < wf,
        "handoff": wf < ts < f.S,
        "slow_finish": ws > max((f.S + d * q for d, q in zip(f.delta, f.Q))),
        "service": 0 < p["service"] < min(f.F - tf, f.S - ts),
        "capacities": f.K < p["Kplus"] < p["M"] and p["M"] > sum(f.Q),
    }
    sums = (
        wf / step * (step / wf) == 1
        and (ws - ts) / step * (step / (ws - ts)) == 1
        and ((wf - tf) / step * (step / (wf - tf)) == 1)
    )
    checks["exact_resource_sums"] = sums
    checks["passed"] = all(checks.values())
    return checks


def reservation_matrix(f, p):
    """Explicit independent tiny-fixture expansion; official inputs stay compressed."""
    dt = p["step"]
    tf, wf, ts, ws = [int(p[k] / dt) for k in ("TF", "WF", "TS", "WS")]
    rows = []
    caps = []
    slopes = []
    for w in range(f.n):
        for start, end, col, cap, slope in (
            (0, wf, w, f.Q[w], f.beta[w] * dt / p["WF"]),
            (ts, ws, f.n + w, f.Q[w], f.delta[w] * dt / (p["WS"] - p["TS"])),
        ):
            for _ in range(start, end):
                row = np.zeros(2 * f.n)
                row[col] = 1
                rows.append(row)
                caps.append(float(cap))
                slopes.append(float(slope))
    for start, end, block, cap, slope in (
        (tf, wf, slice(0, f.n), f.K, f.gamma * dt / (p["WF"] - p["TF"])),
        (ts, ws, slice(f.n, 2 * f.n), p["M"], Fraction(0)),
    ):
        for _ in range(start, end):
            row = np.zeros(2 * f.n)
            row[block] = 1
            rows.append(row)
            caps.append(float(cap))
            slopes.append(float(slope))
    return (np.asarray(rows), np.asarray(caps), np.asarray(slopes))
