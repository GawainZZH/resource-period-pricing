from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import warnings
import sys
import platform
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy
import clarabel
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from resource_pricing.controlled_networks import (
    _config_stream,
    _constraints,
    _local_exchange_basis,
    _reliability_threshold,
    generate_instance,
    solve_core_equilibria,
    total_cost,
    total_cost_gradient,
    build_metropolitan_corridor_instance,
)
from resource_pricing.price_class_gap import GAP_PRICE_CLASSES, build_gap_price_parameterization

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/class_gaps/initial/minimum_primal_gap_constructed_instances.csv"
SOURCE_SHA256 = "FDEC64A7B4FEC3946A692A378EECFA53209BABA96131AAF7D377E25383CE1E70"
FROZEN_SEEDS = tuple(
    (2026062500000 + i for i in (0, 1, 2, 4, 16, 24, 56, 88, 107, 158, 166, 179, 195))
)
LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


@dataclass
class ResourceProblem:
    base: np.ndarray
    occupation: np.ndarray
    capacity: np.ndarray
    alpha: np.ndarray
    rho: float
    threshold: float
    conservation: np.ndarray
    demand: np.ndarray

    def __post_init__(self):
        for key in ("base", "occupation", "capacity", "alpha", "conservation", "demand"):
            value = np.asarray(getattr(self, key))
            if np.iscomplexobj(value) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid {key}")
            setattr(self, key, value.astype(float))
        m, n = self.occupation.shape
        if (
            self.base.shape != (n,)
            or self.capacity.shape != (m,)
            or self.alpha.shape != (m,)
            or (self.conservation.shape != (len(self.demand), n))
        ):
            raise ValueError("Inconsistent resource problem dimensions")
        if (
            np.any(self.capacity <= 0)
            or np.any(self.alpha < 0)
            or np.any(self.occupation < 0)
            or np.any(self.demand < 0)
            or (not np.isfinite(self.rho))
            or (self.rho < 0)
            or (not np.isfinite(self.threshold))
            or (self.threshold < 0)
        ):
            raise ValueError("Invalid convex resource coefficients")
        if not np.all((self.conservation == 0) | (self.conservation == 1)) or not np.all(
            self.conservation.sum(axis=0) == 1
        ):
            raise ValueError("Conservation must partition choices by operation class")

    def cost(self, h):
        y = self.occupation @ h
        v = np.maximum(y / self.capacity - self.threshold, 0)
        return float(
            self.base @ h
            + np.sum(self.alpha * y**3 / self.capacity**2)
            + self.rho * np.sum(y * v**2)
        )

    def private(self, h):
        u = self.occupation @ h / self.capacity
        v = np.maximum(u - self.threshold, 0)
        return self.base + self.occupation.T @ (self.alpha * u**2 + self.rho * v**2)

    def gradient(self, h):
        u = self.occupation @ h / self.capacity
        v = np.maximum(u - self.threshold, 0)
        return self.base + self.occupation.T @ (
            3 * self.alpha * u**2 + self.rho * (v**2 + 2 * u * v)
        )

    def hessian(self, h):
        u = self.occupation @ h / self.capacity
        weights = 6 * self.alpha * u / self.capacity
        weights += (
            self.rho * np.where(u > self.threshold, 6 * u - 4 * self.threshold, 0) / self.capacity
        )
        return self.occupation.T @ (weights[:, None] * self.occupation)

    def feasibility(self, h):
        return {
            "demand_violation": float(
                np.max(np.abs(self.conservation @ h - self.demand), initial=0)
            ),
            "capacity_violation": float(
                max(0, np.max(self.occupation @ h - self.capacity, initial=0))
            ),
            "nonnegativity_violation": float(max(0, -np.min(h, initial=0))),
        }


def from_instance(inst):
    eq, _, _, d, k = _constraints(inst)
    return ResourceProblem(
        inst.planned_time + inst.schedule_penalty,
        d,
        k,
        np.repeat(inst.alpha, inst.n_res),
        inst.rho_late,
        _reliability_threshold(inst),
        np.asarray(eq.A),
        inst.demand,
    )


def _dot(a, b):
    return math.fsum((float(x) * float(y) for x, y in zip(a, b)))


def _center_cost(problem, cost):
    baseline = np.array([np.min(cost[row == 1]) for row in problem.conservation])
    centered = cost - problem.conservation.T @ baseline
    return (baseline, centered, max(float(np.max(np.abs(centered), initial=0)), 1e-20))


def _bounded_oracle(problem, h, cost, extra_matrix=None, extra_bound=None):
    baseline, centered, scale = _center_cost(problem, cost)
    a = (
        problem.occupation
        if extra_matrix is None
        else np.vstack((problem.occupation, extra_matrix))
    )
    b = problem.capacity if extra_bound is None else np.concatenate((problem.capacity, extra_bound))
    lp = linprog(
        centered / scale,
        A_ub=a,
        b_ub=b,
        A_eq=problem.conservation,
        b_eq=problem.demand,
        bounds=(0, None),
        method="highs-ds",
        options=LP_OPTIONS,
    )
    if not lp.success:
        return {
            "raw_gap": float("nan"),
            "gap_upper": float("nan"),
            "LP_success": False,
            "LP_uncertainty": float("nan"),
            "LP_primal_violation": float("nan"),
        }
    inequality_dual = np.minimum(lp.ineqlin.marginals, 0)
    equality_dual = lp.eqlin.marginals.copy()
    reduced = centered / scale - a.T @ inequality_dual - problem.conservation.T @ equality_dual
    for i, row in enumerate(problem.conservation):
        equality_dual[i] += min(0.0, float(np.min(reduced[row == 1])) - 1e-15)
    residual = centered / scale - a.T @ inequality_dual - problem.conservation.T @ equality_dual
    deficit = max(0.0, -float(np.min(residual, initial=0)))
    dual_lower = scale * (
        _dot(b, inequality_dual)
        + _dot(problem.demand, equality_dual)
        - deficit * float(problem.demand.sum())
    )
    correction = _dot(baseline, problem.conservation @ h - problem.demand)
    raw = _dot(centered, h - lp.x) + correction
    upper = _dot(centered, h) - dual_lower + correction
    violation = max(
        max(problem.feasibility(lp.x).values()), max(0.0, float(np.max(a @ lp.x - b, initial=0)))
    )
    return {
        "raw_gap": raw,
        "gap_upper": upper,
        "LP_success": True,
        "LP_uncertainty": abs(upper - raw),
        "LP_primal_violation": violation,
    }


def linear_gap(problem, h, cost):
    return {**_bounded_oracle(problem, h, np.asarray(cost, dtype=float)), **problem.feasibility(h)}


def minimum_class_gap(problem, h, design):
    p = np.asarray(design, dtype=float)
    m, n = problem.occupation.shape
    r, k = (problem.conservation.shape[0], p.shape[1])
    c = problem.private(h)
    baseline, centered, scale = _center_cost(problem, c)
    objective = np.concatenate((p.T @ h, problem.capacity, problem.demand))
    lp = linprog(
        objective,
        A_ub=np.hstack((-p, -problem.occupation.T, -problem.conservation.T)),
        b_ub=centered / scale,
        bounds=[(0, None)] * (k + m) + [(None, None)] * r,
        method="highs-ds",
        options=LP_OPTIONS,
    )
    if not lp.success:
        return {
            "success": False,
            "status": str(lp.message),
            "raw_gap": float("nan"),
            "duality_difference": float("nan"),
            "direct_gap": float("nan"),
            "gap_upper": float("nan"),
            "uncertainty": float("nan"),
            "theta": np.full(k, np.nan),
        }
    theta = scale * lp.x[:k]
    raw = scale * (_dot(centered / scale, h) + _dot(objective, lp.x)) + _dot(
        baseline, problem.conservation @ h - problem.demand
    )
    direct = linear_gap(problem, h, c + p @ theta)
    independent = _bounded_oracle(problem, h, c, p.T, p.T @ h)
    difference = abs(raw - direct["raw_gap"])
    uncertainty = max(
        difference,
        abs(raw - independent["raw_gap"]),
        direct["LP_uncertainty"],
        independent["LP_uncertainty"],
    )
    success = bool(
        direct["LP_success"]
        and independent["LP_success"]
        and np.all(np.isfinite([raw, uncertainty, independent["gap_upper"]]))
        and (difference <= 1e-09)
        and (independent["LP_primal_violation"] <= 1e-09)
    )
    return {
        "success": success,
        "status": str(lp.message),
        "raw_gap": raw,
        "direct_gap": direct["raw_gap"],
        "duality_difference": abs(raw - direct["raw_gap"]),
        "gap_upper": min(direct["gap_upper"], independent["gap_upper"]),
        "uncertainty": uncertainty,
        "theta": theta,
    }


def classify_gap(result, so_precision, threshold):
    if not so_precision or not result["success"]:
        return "unresolved"
    values = [result["raw_gap"], result["gap_upper"], result["uncertainty"]]
    if not np.all(np.isfinite(values)) or result["uncertainty"] > min(1e-09, threshold / 10):
        return "unresolved"
    if result["gap_upper"] < result["raw_gap"] - 1e-12:
        return "unresolved"
    if max(abs(result["raw_gap"]), result["gap_upper"]) <= threshold:
        return "zero"
    if result["raw_gap"] - result["uncertainty"] > threshold:
        return "positive"
    return "unresolved"


def so_precision(result, threshold):
    return bool(
        result["LP_success"]
        and max(abs(result["raw_gap"]), result["gap_upper"]) <= threshold
        and (result["LP_uncertainty"] <= min(1e-09, threshold / 10))
        and (
            max(
                (
                    result[k]
                    for k in ("demand_violation", "capacity_violation", "nonnegativity_violation")
                )
            )
            <= 1e-12
        )
    )


def runtime_identity():
    return {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "cvxpy": cp.__version__,
        "clarabel": clarabel.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "accumulation": "math.fsum of float64 products",
    }


def solve_conic(problem, tolerance):
    h = cp.Variable(len(problem.base), nonneg=True)
    y = problem.occupation @ h
    v = cp.Variable(len(problem.capacity), nonneg=True)
    expression = problem.base @ h + cp.sum(
        cp.multiply(problem.alpha / problem.capacity**2, cp.power(y, 3))
    )
    expression += problem.rho * cp.sum(
        cp.multiply(problem.capacity, cp.power(v, 3) + problem.threshold * cp.square(v))
    )
    program = cp.Problem(
        cp.Minimize(expression),
        [
            problem.conservation @ h == problem.demand,
            y <= problem.capacity,
            v >= y / problem.capacity - problem.threshold,
        ],
    )
    if not program.is_dcp():
        raise ValueError("Independent resource objective is not convex")
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        program.solve(
            solver="CLARABEL",
            tol_gap_abs=tolerance,
            tol_gap_rel=tolerance,
            tol_feas=tolerance,
            max_iter=500,
        )
    if h.value is None:
        raise RuntimeError(f"No conic solution: {program.status}")
    flow = np.asarray(h.value).reshape(-1)
    return {
        "flow": flow,
        "status": program.status,
        "requested_tolerance": tolerance,
        "model_objective": float(program.value),
        "objective_difference": abs(float(program.value) - problem.cost(flow)),
        "iterations": program.solver_stats.num_iters,
        "seconds": time.perf_counter() - start,
        "warnings": " | ".join((str(w.message) for w in caught)),
        **linear_gap(problem, flow, problem.gradient(flow)),
    }


def polish_solution(problem, flow):
    start = time.perf_counter()
    solved = minimize(
        problem.cost,
        flow,
        jac=problem.gradient,
        method="SLSQP",
        bounds=Bounds(0, np.inf),
        constraints=[
            LinearConstraint(problem.conservation, problem.demand, problem.demand),
            LinearConstraint(problem.occupation, -np.inf, problem.capacity),
        ],
        options={"ftol": 1e-13, "maxiter": 2000},
    )
    return {
        "flow": solved.x,
        "status": str(solved.message),
        "solver_success": bool(solved.success),
        "iterations": solved.nit,
        "seconds": time.perf_counter() - start,
        **linear_gap(problem, solved.x, problem.gradient(solved.x)),
    }


def refine_active_face(problem, flow, max_iterations=200):
    """Newton correction with active-face changes, accepted only by direct gap."""
    h = np.asarray(flow, dtype=float).copy()
    if h.shape != problem.base.shape or not np.all(np.isfinite(h)):
        raise ValueError("Invalid initial flow")
    if max(problem.feasibility(h).values()) > 1e-08:
        raise ValueError("Newton correction needs a near-feasible initial flow")
    start = time.perf_counter()
    free = h > 1e-07
    h[~free] = 0
    if max(problem.feasibility(h).values()) > 1e-12:
        n = len(h)
        identity = np.eye(n)
        repaired = linprog(
            np.concatenate((np.zeros(n), np.ones(n))),
            A_ub=np.vstack(
                (
                    np.hstack((problem.occupation, np.zeros((len(problem.capacity), n)))),
                    np.hstack((identity, -identity)),
                    np.hstack((-identity, -identity)),
                )
            ),
            b_ub=np.concatenate((problem.capacity, h, -h)),
            A_eq=np.hstack((problem.conservation, np.zeros((len(problem.demand), n)))),
            b_eq=problem.demand,
            bounds=[(0, None) if f else (0, 0) for f in free] + [(0, None)] * n,
            method="highs-ds",
            options=LP_OPTIONS,
        )
        if not repaired.success:
            h = np.asarray(flow, dtype=float).copy()
            free = h > 0
        else:
            h = repaired.x[:n]
    active = problem.capacity - problem.occupation @ h < 1e-08
    history = []
    status = "iteration limit"
    for iteration in range(max_iterations):
        gradient = problem.gradient(h)
        checks = linear_gap(problem, h, gradient)
        history.append(checks["raw_gap"])
        if (
            checks["LP_success"]
            and max(abs(checks["raw_gap"]), checks["gap_upper"]) <= 1e-10
            and (max(problem.feasibility(h).values()) <= 1e-12)
        ):
            status = "direct gap tolerance reached"
            break
        indices = np.flatnonzero(free)
        constraints = np.vstack((problem.conservation, problem.occupation[active]))
        target = np.concatenate((problem.demand, problem.capacity[active]))
        a = constraints[:, indices]
        hessian = problem.hessian(h)[np.ix_(indices, indices)]
        hessian += np.eye(len(indices)) * 1e-14
        matrix = np.block([[hessian, a.T], [a, np.zeros((len(target), len(target)))]])
        rhs = np.concatenate((-gradient[indices], target - constraints @ h))
        answer = np.linalg.lstsq(matrix, rhs, rcond=1e-15)[0]
        direction = np.zeros_like(h)
        direction[indices] = answer[: len(indices)]
        multiplier = answer[len(indices) :]
        if np.max(np.abs(matrix @ answer - rhs), initial=0) > 1e-08:
            status = "inconsistent active-face Newton system"
            break
        reduced = gradient + constraints.T @ multiplier
        if np.max(np.abs(direction), initial=0) < 1e-08:
            if np.any(~free) and np.min(reduced[~free]) < -1e-14:
                release = np.flatnonzero(~free)[np.argmin(reduced[~free])]
                free[release] = True
                continue
            cap_multiplier = multiplier[len(problem.demand) :]
            if len(cap_multiplier) and np.min(cap_multiplier) < -1e-14:
                active[np.flatnonzero(active)[np.argmin(cap_multiplier)]] = False
                continue
        step = 1.0
        negative = direction < -1e-14
        hit_bound = None
        if np.any(negative):
            candidates = np.flatnonzero(negative)
            ratios = -h[candidates] / direction[candidates]
            location = np.argmin(ratios)
            if ratios[location] < step:
                step = max(0.0, ratios[location])
                hit_bound = candidates[location]
        load_direction = problem.occupation @ direction
        growing = ~active & (load_direction > 1e-14)
        hit_capacity = None
        if np.any(growing):
            candidates = np.flatnonzero(growing)
            ratios = (
                problem.capacity[candidates] - (problem.occupation @ h)[candidates]
            ) / load_direction[candidates]
            location = np.argmin(ratios)
            if ratios[location] < step:
                step = max(0.0, ratios[location])
                hit_capacity, hit_bound = (candidates[location], None)
        boundary_step = step
        value = problem.cost(h)
        for _ in range(40):
            candidate = h + step * direction
            if problem.cost(candidate) <= value + 1e-12 + 0.0001 * step * min(
                0, gradient @ direction
            ):
                break
            step *= 0.5
        else:
            status = "line search failed"
            break
        h += step * direction
        if step == boundary_step:
            if hit_bound is not None:
                h[hit_bound], free[hit_bound] = (0, False)
            if hit_capacity is not None:
                active[hit_capacity] = True
        if step == 0 and hit_bound is None and (hit_capacity is None):
            status = "zero step without a constraint change"
            break
    return {
        "flow": h,
        "status": status,
        "iterations": iteration + 1,
        "seconds": time.perf_counter() - start,
        "gap_history": history,
        **linear_gap(problem, h, problem.gradient(h)),
    }


def load_frozen_source():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest().upper() != SOURCE_SHA256:
        raise ValueError("Frozen source data changed")
    return pd.read_csv(SOURCE)


def run_audit(out_dir, seeds=FROZEN_SEEDS):
    source = load_frozen_source().set_index("Seed")
    out_dir.mkdir(parents=True, exist_ok=True)
    solutions, classes = ([], [])
    configs = {c.seed: c for c in _config_stream(200, 20260625)}
    manifest = {
        "source_sha256": SOURCE_SHA256,
        "seeds": list(seeds),
        "tolerances": [1e-08, 1e-10, 1e-12],
        "cvxpy": cp.__version__,
        "LP_options": LP_OPTIONS,
        "classification_thresholds": [1e-06, 1e-07, 1e-08],
        "scope": "Fixed-point numerical audit, not a uniform SO perturbation bound",
        **runtime_identity(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for seed in seeds:
        inst = generate_instance(configs[seed])
        problem = from_instance(inst)
        start = time.perf_counter()
        _, original = solve_core_equilibria(inst, maxiter=300)
        candidates = [
            (
                "original_slsqp",
                1e-08,
                {
                    "flow": original,
                    "status": "legacy solver",
                    "seconds": time.perf_counter() - start,
                    **linear_gap(problem, original, problem.gradient(original)),
                },
            )
        ]
        for tolerance in manifest["tolerances"]:
            conic = solve_conic(problem, tolerance)
            candidates.extend(
                (
                    ("independent_conic", tolerance, conic),
                    ("warm_start_slsqp", tolerance, polish_solution(problem, conic["flow"])),
                )
            )
        refined = refine_active_face(problem, candidates[-1][2]["flow"])
        candidates.append(("active_face_newton", 1e-10, refined))
        arrays = {
            "occupation": problem.occupation,
            "conservation": problem.conservation,
            "capacity": problem.capacity,
            "demand": problem.demand,
        }
        for index, (method, tolerance, result) in enumerate(candidates):
            h = result["flow"]
            arrays[f"flow_{index}"] = h
            row = {
                "Seed": seed,
                "method": method,
                "tolerance": tolerance,
                "alternatives": inst.n_alt,
                **{key: value for key, value in result.items() if key != "flow"},
                "objective": problem.cost(h),
                "legacy_cost_difference": abs(problem.cost(h) - total_cost(inst, h)),
                "legacy_gradient_difference": float(
                    np.max(np.abs(problem.gradient(h) - total_cost_gradient(inst, h)))
                ),
                "original_source_cost_difference": problem.cost(original)
                - float(source.loc[seed, "SO cost"]),
            }
            solutions.append(row)
            for price_class in GAP_PRICE_CLASSES:
                p = build_gap_price_parameterization(inst, price_class).design
                gap = minimum_class_gap(problem, h, p)
                arrays[f"theta_{index}_{price_class}"] = gap.pop("theta")
                prefix = price_class.replace("_", " ").title().replace(" ", "")
                old = float(source.loc[seed, f"{prefix} minimum primal gap"])
                class_row = {
                    "Seed": seed,
                    "method": method,
                    "tolerance": tolerance,
                    "price_class": price_class,
                    "original_gap": old,
                    **gap,
                }
                for threshold in manifest["classification_thresholds"]:
                    precise = so_precision(result, threshold)
                    classification = classify_gap(gap, precise, threshold)
                    numeric_zero = (
                        classification == "zero" if classification != "unresolved" else None
                    )
                    class_row[f"classification_{threshold:g}"] = classification
                    class_row[f"zero_{threshold:g}"] = numeric_zero
                    class_row[f"SO_precision_{threshold:g}"] = precise
                    class_row[f"changed_{threshold:g}"] = (
                        numeric_zero != (old <= threshold) if numeric_zero is not None else None
                    )
                classes.append(class_row)
        np.savez_compressed(out_dir / f"flows_{seed}.npz", **arrays)
        pd.DataFrame(solutions).to_csv(out_dir / "SO_solutions.csv", index=False)
        pd.DataFrame(classes).to_csv(out_dir / "class_gaps.csv", index=False)
        print(
            f"Completed {seed}: {len(solutions)} SO solutions, {len(classes)} class LPs", flush=True
        )
    return (pd.DataFrame(solutions), pd.DataFrame(classes))


def run_full_audit(out_dir, seeds=None):
    source = load_frozen_source().set_index("Seed")
    seeds = tuple((int(s) for s in (source.index if seeds is None else seeds)))
    configs = {c.seed: c for c in _config_stream(200, 20260625)}
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_sha256": SOURCE_SHA256,
        "seeds": list(seeds),
        "cvxpy": cp.__version__,
        "conic_requested_tolerance": 1e-10,
        "newton_actual_gap_target": 1e-10,
        "LP_options": LP_OPTIONS,
        "classification_thresholds": [1e-06, 1e-07, 1e-08],
        "reason": "Frozen audit expansion: full-resource classifications changed at 1e-8",
        "scope": "Numerical stability, not a rigorous uniform SO perturbation bound",
        **runtime_identity(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    solutions, classes = ([], [])
    for seed in seeds:
        inst = generate_instance(configs[seed])
        if inst.n_alt != int(source.loc[seed, "Alternatives"]) or inst.n_resources != int(
            source.loc[seed, "Resources"]
        ):
            raise ValueError("Frozen instance metadata changed")
        problem = from_instance(inst)
        conic = solve_conic(problem, 1e-10)
        warm = polish_solution(problem, conic["flow"])
        refined = refine_active_face(problem, warm["flow"])
        arrays = {
            "occupation": problem.occupation,
            "conservation": problem.conservation,
            "capacity": problem.capacity,
            "demand": problem.demand,
            "base": problem.base,
            "alpha": problem.alpha,
        }
        for index, (method, result) in enumerate(
            zip(
                ("independent_conic", "warm_start_slsqp", "active_face_newton"),
                (conic, warm, refined),
            )
        ):
            h = result["flow"]
            arrays[f"flow_{index}"] = h
            solutions.append(
                {
                    "Seed": seed,
                    "method": method,
                    "alternatives": inst.n_alt,
                    **{key: value for key, value in result.items() if key != "flow"},
                    "objective": problem.cost(h),
                    "original_objective": float(source.loc[seed, "SO cost"]),
                    "legacy_cost_difference": abs(problem.cost(h) - total_cost(inst, h)),
                    "legacy_gradient_difference": float(
                        np.max(np.abs(problem.gradient(h) - total_cost_gradient(inst, h)))
                    ),
                }
            )
            for price_class in GAP_PRICE_CLASSES:
                gap = minimum_class_gap(
                    problem, h, build_gap_price_parameterization(inst, price_class).design
                )
                arrays[f"theta_{index}_{price_class}"] = gap.pop("theta")
                prefix = price_class.replace("_", " ").title().replace(" ", "")
                old = float(source.loc[seed, f"{prefix} minimum primal gap"])
                row = {
                    "Seed": seed,
                    "method": method,
                    "price_class": price_class,
                    "original_gap": old,
                    **gap,
                }
                for threshold in manifest["classification_thresholds"]:
                    precise = so_precision(result, threshold)
                    row[f"SO_precision_{threshold:g}"] = precise
                    classification = classify_gap(gap, precise, threshold)
                    row[f"classification_{threshold:g}"] = classification
                    zero = classification == "zero" if classification != "unresolved" else None
                    row[f"zero_{threshold:g}"] = zero
                    row[f"changed_{threshold:g}"] = (
                        zero != (old <= threshold) if zero is not None else None
                    )
                classes.append(row)
        np.savez_compressed(out_dir / f"flows_{seed}.npz", **arrays)
        pd.DataFrame(solutions).to_csv(out_dir / "SO_solutions.csv", index=False)
        pd.DataFrame(classes).to_csv(out_dir / "class_gaps.csv", index=False)
        print(
            f"Full audit {seed}: SO gap {refined['raw_gap']:.3e}, {refined['status']}", flush=True
        )
    return (pd.DataFrame(solutions), pd.DataFrame(classes))


def export_refined_plot_data(out_dir):
    solutions = pd.read_csv(out_dir / "SO_solutions.csv")
    classes = pd.read_csv(out_dir / "class_gaps.csv")
    refined = solutions[solutions.method == "active_face_newton"].set_index("Seed")
    gaps = classes[classes.method == "active_face_newton"].set_index(["Seed", "price_class"])
    for threshold in (1e-06, 1e-07, 1e-08):
        if (gaps[f"classification_{threshold:g}"] == "unresolved").any():
            raise ValueError("Unresolved classifications must not be omitted from plots")
    frame = load_frozen_source().set_index("Seed").loc[refined.index].copy()
    configs = {c.seed: c for c in _config_stream(200, 20260625)}
    for seed in frame.index:
        inst = generate_instance(configs[seed])
        problem = from_instance(inst)
        with np.load(out_dir / f"flows_{seed}.npz") as arrays:
            h = arrays["flow_2"]
        basis = _local_exchange_basis(inst, h)
        target = basis.T @ (problem.gradient(h) - problem.private(h))
        frame.loc[seed, "SO cost"] = problem.cost(h)
        all_success = True
        for price_class in GAP_PRICE_CLASSES:
            value = gaps.loc[seed, price_class]
            if isinstance(value, pd.DataFrame):
                raise ValueError("Duplicate refined price-class rows")
            p = basis.T @ build_gap_price_parameterization(inst, price_class).design
            if basis.shape[1] == 0:
                residual = 0.0
            else:
                coefficient = np.linalg.lstsq(p, target, rcond=None)[0]
                residual = float(
                    np.linalg.norm(target - p @ coefficient) / max(np.linalg.norm(target), 1e-12)
                )
            prefix = price_class.replace("_", " ").title().replace(" ", "")
            frame.loc[seed, f"{prefix} minimum primal gap"] = value.raw_gap
            frame.loc[seed, f"{prefix} normalized minimum primal gap"] = value.raw_gap / max(
                problem.cost(h), 1
            )
            frame.loc[seed, f"{prefix} direct primal gap"] = value.direct_gap
            frame.loc[seed, f"{prefix} LP duality difference"] = value.duality_difference
            frame.loc[seed, f"{prefix} projection residual"] = residual
            frame.loc[seed, f"{prefix} classification"] = value["classification_1e-07"]
            frame.loc[seed, f"{prefix} gap upper"] = value.gap_upper
            frame.loc[seed, f"{prefix} uncertainty"] = value.uncertainty
            success = bool(value.success and value["SO_precision_1e-08"])
            frame.loc[seed, f"{prefix} success"] = success
            all_success = all_success and success
        frame.loc[seed, "success"] = all_success
    frame = frame.reset_index()
    frame.to_csv(out_dir / "minimum_primal_gap_constructed_instances.csv", index=False)
    return frame


def run_illustrative_audit(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    inst, _ = build_metropolitan_corridor_instance()
    problem = from_instance(inst)
    conic = solve_conic(problem, 1e-10)
    warm = polish_solution(problem, conic["flow"])
    refined = refine_active_face(problem, warm["flow"])
    solutions = pd.DataFrame(
        [
            {
                "method": method,
                "objective": problem.cost(value["flow"]),
                **{k: v for k, v in value.items() if k != "flow"},
            }
            for method, value in zip(
                ("independent_conic", "warm_start_slsqp", "active_face_newton"),
                (conic, warm, refined),
            )
        ]
    )
    classes = []
    arrays = {
        "flow_conic": conic["flow"],
        "flow_warm": warm["flow"],
        "flow_refined": refined["flow"],
        "occupation": problem.occupation,
        "capacity": problem.capacity,
        "conservation": problem.conservation,
        "demand": problem.demand,
    }
    for price_class in GAP_PRICE_CLASSES:
        gap = minimum_class_gap(
            problem, refined["flow"], build_gap_price_parameterization(inst, price_class).design
        )
        arrays[f"theta_{price_class}"] = gap.pop("theta")
        row = {"price_class": price_class, **gap}
        for threshold in (1e-06, 1e-07, 1e-08):
            row[f"classification_{threshold:g}"] = classify_gap(
                gap, so_precision(refined, threshold), threshold
            )
        classes.append(row)
    source = SOURCE.with_name("minimum_primal_gap_illustrative_case.csv")
    manifest = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config": vars(inst.config),
        "arrays_sha256": hashlib.sha256(
            b"".join(
                (
                    a.tobytes()
                    for a in (
                        problem.occupation,
                        problem.capacity,
                        problem.conservation,
                        problem.demand,
                    )
                )
            )
        ).hexdigest(),
        **runtime_identity(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    solutions.to_csv(out_dir / "SO_solutions.csv", index=False)
    classes = pd.DataFrame(classes)
    classes.to_csv(out_dir / "class_gaps.csv", index=False)
    np.savez_compressed(out_dir / "flows.npz", **arrays)
    return (solutions, classes)
