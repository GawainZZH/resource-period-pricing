import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.sparse import bmat, csr_matrix, diags

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
from resource_pricing.protected_costs import load_three_channel

DATA = HERE / "data"
WITNESS = ROOT / "data/reservation_inputs/three_channel_maximum_assignment.npy"
NONLINEAR_SECONDS = 120
GAP_SECONDS = 30
MAX_NFEV = 600


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_arrays(path, **values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def fb(a, b):
    return np.hypot(a, b) - a - b


def fb_partials(a, b):
    radius = np.hypot(a, b)
    da = np.divide(a, radius, out=np.zeros_like(radius), where=radius > 0) - 1
    db = np.divide(b, radius, out=np.zeros_like(radius), where=radius > 0) - 1
    return (da, db)


class ComplementaritySystem:

    def __init__(self, model):
        self.model = model
        self.n = len(model.base)
        self.m = len(model.capacity)
        self.w = len(model.demand)
        self.D, self.H = (csr_matrix(model.occupation), csr_matrix(model.conservation))
        self.owner = model.conservation.argmax(axis=0)
        self.choice_demand = model.demand[self.owner]
        if np.any(model.demand <= 0):
            raise ValueError("This gate requires positive class demand")
        self.centers = np.array([np.min(model.base[row == 1]) for row in model.conservation])
        self.scale = max(1.0, float(np.max(abs(model.base - model.conservation.T @ self.centers))))

    def _state(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n + self.m + self.w,) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid complementarity state")
        h, lam, nu = (x[: self.n], x[self.n : self.n + self.m], x[self.n + self.m :])
        F = (
            (self.model.private(h) - self.model.conservation.T @ self.centers) / self.scale
            + self.D.T @ lam
            + self.H.T @ nu
        )
        slack = (self.model.capacity - self.D @ h) / self.model.capacity
        return (h, lam, nu, F, slack)

    def initial(self, h):
        h = self.model._vector(h)
        centered = (self.model.private(h) - self.model.conservation.T @ self.centers) / self.scale
        nu = np.array([-np.mean(centered[(row == 1) & (h > 0)]) for row in self.model.conservation])
        return np.r_[h, np.zeros(self.m), nu]

    def residual(self, x):
        h, lam, _, F, slack = self._state(x)
        return np.r_[
            fb(h / self.choice_demand, F),
            fb(lam, slack),
            (self.H @ h - self.model.demand) / self.model.demand,
        ]

    def jacobian(self, x):
        h, lam, _, F, slack = self._state(x)
        ah, bh = fb_partials(h / self.choice_demand, F)
        al, bl = fb_partials(lam, slack)
        hh = diags(ah / self.choice_demand) + diags(bh) @ csr_matrix(
            self.model.jacobian(h) / self.scale
        )
        return bmat(
            [
                [hh, diags(bh) @ self.D.T, diags(bh) @ self.H.T],
                [-diags(bl / self.model.capacity) @ self.D, diags(al), None],
                [diags(1 / self.model.demand) @ self.H, None, None],
            ],
            format="csr",
        )


def feasibility_passed(values):
    limits = {
        "nonnegativity_violation": 1e-10,
        "capacity_violation": 1e-09,
        "demand_violation": 1e-09,
    }
    return all((np.isfinite(values[k]) and 0 <= values[k] <= v for k, v in limits.items()))


def accept_gate(feasibility, gap):
    if not feasibility_passed(feasibility):
        return False
    try:
        numbers = np.array(
            [gap.get(k) for k in ("raw_gap", "gap_upper", "LP_uncertainty")], dtype=float
        )
    except (TypeError, ValueError):
        return False
    return bool(
        gap.get("LP_success")
        and gap.get("exact_dual_feasible")
        and gap.get("exact_primal_witness_feasible")
        and np.all(np.isfinite(numbers))
        and (abs(gap["raw_gap"]) <= 1e-07)
        and (gap["gap_upper"] <= 1e-07)
        and (gap["gap_upper"] >= gap["raw_gap"] - 1e-12)
        and (0 <= gap["LP_uncertainty"] <= 1e-09)
    )


def frac(value):
    return Fraction(float(value))


def upward(value):
    number = float(value)
    return float(np.nextafter(number, math.inf)) if frac(number) < value else number


def exact_gap_bounds(model, h, c, baseline, scale, result):
    if not result.success or result.x is None:
        return {"LP_success": False, "status": str(result.message)}
    D, H, K, q = (model.occupation, model.conservation, model.capacity, model.demand)
    if not np.all((D == 0) | (D == 1)) or not np.all(K == np.rint(K)):
        raise ValueError("Exact gap bound requires frozen binary occupancy and integer capacities")
    owner = H.argmax(axis=0)
    cost = [frac(v) for v in c]
    lambdas = [frac(min(0.0, v)) * frac(scale) for v in result.ineqlin.marginals]
    nus = [frac(v) * frac(scale) + frac(baseline[w]) for w, v in enumerate(result.eqlin.marginals)]
    coverage = [
        sum((lambdas[r] for r in np.flatnonzero(D[:, j])), Fraction(0)) for j in range(len(h))
    ]
    repairs = []
    for w, row in enumerate(H):
        deficit = min((cost[j] - coverage[j] - nus[w] for j in np.flatnonzero(row)))
        repair = min(Fraction(0), deficit)
        nus[w] += repair
        repairs.append(-repair)
    exact_dual = all((coverage[j] + nus[owner[j]] <= cost[j] for j in range(len(h))))
    dual_lower = sum((frac(k) * v for k, v in zip(K, lambdas)), Fraction(0))
    dual_lower += sum((frac(amount) * v for amount, v in zip(q, nus)), Fraction(0))
    target_cost = sum((coef * frac(flow) for coef, flow in zip(cost, h)), Fraction(0))
    witness = [frac(v).limit_denominator(10**9) for v in result.x]
    primal_ok = (
        all((v >= 0 for v in witness))
        and all(
            (
                sum((witness[j] for j in np.flatnonzero(row)), Fraction(0)) <= frac(k)
                for row, k in zip(D, K)
            )
        )
        and all(
            (
                sum((witness[j] for j in np.flatnonzero(row)), Fraction(0)) == frac(amount)
                for row, amount in zip(H, q)
            )
        )
    )
    raw_primal = sum((coef * frac(flow) for coef, flow in zip(cost, result.x)), Fraction(0))
    raw_gap = target_cost - raw_primal
    gap_upper = target_cost - dual_lower
    record = {
        "LP_success": True,
        "raw_gap": float(raw_gap),
        "gap_upper": upward(gap_upper),
        "exact_dual_feasible": bool(exact_dual),
        "exact_primal_witness_feasible": bool(primal_ok),
        "dual_repair_max": float(max(repairs)),
        "exact_dual_lower": str(dual_lower),
        "exact_target_cost": str(target_cost),
        "raw_gap_exact": str(raw_gap),
        "gap_upper_exact": str(gap_upper),
        "raw_lp_min_flow": float(np.min(result.x)),
    }
    if primal_ok:
        primal_upper = sum((coef * flow for coef, flow in zip(cost, witness)), Fraction(0))
        lower = target_cost - primal_upper
        record.update(
            gap_lower=float(lower),
            gap_lower_exact=str(lower),
            LP_uncertainty=upward(gap_upper - lower),
            rational_witness=[str(v) for v in witness],
        )
    else:
        record.update(LP_uncertainty=None)
    return record


def vi_worker(directory):
    from scipy.optimize import least_squares

    model = load_three_channel()
    system = ComplementaritySystem(model)
    h0 = np.load(WITNESS, allow_pickle=False)
    x0 = system.initial(h0)
    save_arrays(
        directory / "initial.npz", state=x0, flow=h0, centers=system.centers, scale=system.scale
    )
    start = time.monotonic()
    evaluations = 0
    best_norm = math.inf
    best = x0.copy()

    def residual(x):
        nonlocal evaluations, best_norm, best
        value = system.residual(x)
        evaluations += 1
        norm = float(np.max(abs(value)))
        save_arrays(directory / "last_evaluated.npz", state=x, flow=x[:180], residual=value)
        if norm < best_norm:
            best_norm, best = (norm, x.copy())
            save_arrays(directory / "best_residual.npz", state=x, flow=x[:180], residual=value)
        if time.monotonic() - start > NONLINEAR_SECONDS - 8:
            raise TimeoutError("Nonlinear internal wall-clock budget exhausted")
        return value

    try:
        result = least_squares(
            residual,
            x0,
            jac=system.jacobian,
            method="trf",
            tr_solver="lsmr",
            x_scale="jac",
            ftol=None,
            xtol=1e-14,
            gtol=1e-13,
            max_nfev=MAX_NFEV,
            tr_options={"atol": 1e-13, "btol": 1e-13, "maxiter": 5000, "regularize": True},
        )
        x = result.x
        info = {
            "solver_success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": int(result.njev),
            "optimality": float(result.optimality),
            "source": "raw solver return",
        }
    except Exception as error:
        x = best
        info = {
            "solver_success": False,
            "message": f"{type(error).__name__}: {error}",
            "source": "saved best residual evaluation, no restart",
        }
    value = system.residual(x)
    save_arrays(
        directory / "vi_raw.npz",
        state=x,
        flow=x[:180],
        residual=value,
        private_cost=model.private(x[:180]),
        centers=system.centers,
        scale=system.scale,
    )
    info.update(
        seconds=time.monotonic() - start,
        residual_evaluations=evaluations,
        complementarity_residual_max=float(np.max(abs(value))),
        feasibility=model.feasibility(x[:180]),
        numeric_flow_postprocessing=False,
    )
    write_json(directory / "vi_raw.json", info)


def gap_worker(directory):
    from scipy.optimize import linprog

    model = load_three_channel()
    with np.load(directory / "vi_raw.npz", allow_pickle=False) as archive:
        h = archive["flow"].copy()
    if not feasibility_passed(model.feasibility(h)):
        raise ValueError("Candidate is infeasible; gap LP not authorized")
    c = model.private(h)
    baseline = np.array([np.min(c[row == 1]) for row in model.conservation])
    centered = c - model.conservation.T @ baseline
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
        private_cost=c,
        centered=centered,
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
    bound = exact_gap_bounds(model, h, c, baseline, scale, result)
    write_json(directory / "gap_bounds.json", bound)
