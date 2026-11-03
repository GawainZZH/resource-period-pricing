import os

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

ROOT = Path(__file__).resolve().parents[1]
from resource_pricing import convex_programs as audit
from resource_pricing.controlled_networks import SyntheticInstance, StudyConfig
from resource_pricing.restricted_pricing import (
    build_price_parameterization,
    _component_price_start,
    _best_in_class_gradient,
)

ResourceProblem = audit.ResourceProblem


class BudgetExceeded(RuntimeError):
    pass


class Budget:

    def __init__(self, seconds):
        self.end = time.monotonic() + seconds
        self.calls = {}

    def check(self):
        if time.monotonic() >= self.end:
            raise BudgetExceeded("Frozen wall-time budget exhausted")

    def consume(self, key, limit):
        self.check()
        if self.calls.get(key, 0) >= limit:
            raise BudgetExceeded(f"Frozen evaluation budget exhausted: {key}")
        self.calls[key] = self.calls.get(key, 0) + 1


def load_input(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = data["instance"]
    fields["config"] = StudyConfig(**fields["config"])
    for key, value in fields.items():
        if isinstance(value, list):
            fields[key] = np.array(value)
    inst = SyntheticInstance(**fields)
    designs = {k: np.array(v, dtype=float) for k, v in data["price_designs"].items()}
    return (inst, audit.from_instance(inst), designs)


def check_designs(inst, designs):
    columns = {k: [] for k in designs}
    for r in range(inst.n_resources):
        for s in range(inst.n_res):
            full = inst.delta[r, s].copy()
            current = full * (inst.dep_of_alt == s)
            for key, column in (
                ("all_resource", full),
                (str(inst.resource_types[r]), full),
                ("contemporaneous", current),
            ):
                if np.any(np.abs(column) > 1e-12):
                    columns[key].append(column)
    return all((np.array_equal(designs[k], np.column_stack(columns[k])) for k in designs))


class PricedProblem:

    def __init__(self, problem, tau):
        self.problem = problem
        self.tau = np.asarray(tau, dtype=float)
        if self.tau.shape != problem.base.shape or not np.all(np.isfinite(self.tau)):
            raise ValueError("Invalid price vector")

    def __getattr__(self, key):
        return getattr(self.problem, key)

    def cost(self, h):
        p = self.problem
        y = p.occupation @ h
        excess = np.maximum(y / p.capacity - p.threshold, 0)
        return float(
            (p.base + self.tau) @ h
            + np.sum(p.alpha * y**3 / (3 * p.capacity**2))
            + p.rho * np.sum(p.capacity * excess**3) / 3
        )

    def gradient(self, h):
        return self.problem.private(h) + self.tau

    def hessian(self, h):
        p = self.problem
        u = p.occupation @ h / p.capacity
        weights = (2 * p.alpha * u + 2 * p.rho * np.maximum(u - p.threshold, 0)) / p.capacity
        return p.occupation.T @ (weights[:, None] * p.occupation)


def system_expression_value(p, h):
    y = p.occupation @ h
    v = np.maximum(y / p.capacity - p.threshold, 0)
    return float(
        p.base @ h
        + np.sum(p.alpha / p.capacity**2 * y**3)
        + p.rho * np.sum(p.capacity * (v**3 + p.threshold * v**2))
    )


def direct_gap(p, h, cost):
    h, cost = (np.asarray(h), np.asarray(cost))
    if (
        h.shape != p.base.shape
        or cost.shape != p.base.shape
        or (not np.all(np.isfinite([h, cost])))
    ):
        raise ValueError("Invalid direct gap arguments")
    return audit.linear_gap(p, h, cost)


def flow_pass(r, scale, tolerance):
    keys = (
        "raw_gap",
        "gap_upper",
        "LP_uncertainty",
        "LP_primal_violation",
        "demand_violation",
        "capacity_violation",
        "nonnegativity_violation",
    )
    return bool(
        r.get("LP_success", False)
        and all((np.isfinite(r.get(k, np.nan)) for k in keys))
        and (r["raw_gap"] >= -r["LP_uncertainty"])
        and (max(abs(r["raw_gap"]), r["gap_upper"]) / scale <= tolerance)
        and (r["LP_uncertainty"] / scale <= 1e-10)
        and (max((r[k] for k in keys[3:])) <= 1e-10)
    )


def class_gap(p, h, design):
    design = np.asarray(design)
    if (
        design.ndim != 2
        or design.shape[0] != len(h)
        or (not np.all(np.isfinite(design)))
        or np.any(design < 0)
    ):
        raise ValueError("Invalid nonnegative price map")
    m, n = p.occupation.shape
    k, r = (design.shape[1], len(p.demand))
    c = p.private(h)
    baseline, centered, cscale = audit._center_cost(p, c)
    objective = np.concatenate((design.T @ h, p.capacity, p.demand))
    matrix = np.hstack((-design, -p.occupation.T, -p.conservation.T))
    rhs = centered / cscale
    lp = audit.linprog(
        objective,
        A_ub=matrix,
        b_ub=rhs,
        bounds=[(0, None)] * (k + m) + [(None, None)] * r,
        method="highs-ds",
        options=audit.LP_OPTIONS,
    )
    if not lp.success:
        return dict(
            success=False,
            status=str(lp.message),
            theta=np.full(k, np.nan),
            raw_gap=np.nan,
            gap_upper=np.nan,
            uncertainty=np.nan,
        )
    theta = cscale * lp.x[:k]
    raw = cscale * (audit._dot(centered / cscale, h) + audit._dot(objective, lp.x))
    raw += audit._dot(baseline, p.conservation @ h - p.demand)
    direct = direct_gap(p, h, c + design @ theta)
    independent = audit._bounded_oracle(p, h, c, design.T, design.T @ h)
    primal_violation = max(0.0, float(np.max(matrix @ lp.x - rhs)), -float(np.min(lp.x[: k + m])))
    values = [
        raw,
        direct["raw_gap"],
        direct["gap_upper"],
        direct["LP_uncertainty"],
        independent["raw_gap"],
        independent["gap_upper"],
        independent["LP_uncertainty"],
    ]
    uncertainty = (
        max(
            abs(raw - direct["raw_gap"]),
            abs(raw - independent["raw_gap"]),
            direct["LP_uncertainty"],
            independent["LP_uncertainty"],
        )
        if np.all(np.isfinite(values))
        else np.nan
    )
    scale = max(1.0, abs(p.cost(h)))
    success = bool(
        direct["LP_success"]
        and independent["LP_success"]
        and np.isfinite(uncertainty)
        and (uncertainty / scale <= 1e-10)
        and (raw >= -uncertainty)
        and (primal_violation <= 1e-10)
        and (direct["LP_primal_violation"] <= 1e-10)
        and (independent["LP_primal_violation"] <= 1e-10)
        and (max(p.feasibility(h).values()) <= 1e-10)
    )
    return dict(
        success=success,
        status=str(lp.message),
        theta=theta,
        raw_gap=raw,
        direct_gap=direct["raw_gap"],
        gap_upper=direct["gap_upper"],
        uncertainty=uncertainty,
        duality_difference=abs(raw - direct["raw_gap"]),
        primal_violation=primal_violation,
        price_direct_check=direct,
        independent_check=independent,
    )


def classify_pair(results, so_pass, scale):
    if not all(so_pass) or len(results) != 2:
        return "UNRESOLVED"
    for r in results:
        if not r.get("success") or not all(
            (np.isfinite(r.get(k, np.nan)) for k in ("raw_gap", "gap_upper", "uncertainty"))
        ):
            return "UNRESOLVED"
        if r["uncertainty"] / scale > 1e-10 or r["raw_gap"] < -r["uncertainty"]:
            return "UNRESOLVED"
    if all((max(abs(r["raw_gap"]), r["gap_upper"]) / scale <= 1e-08 for r in results)):
        return "STABLE_ZERO"
    if all(((r["raw_gap"] - r["uncertainty"]) / scale > 1e-06 for r in results)):
        return "STABLE_POSITIVE"
    return "BORDERLINE"


def classify_threshold(r, accepted_so, scale, threshold):
    if not accepted_so or not r.get("success"):
        return "UNRESOLVED"
    if not all((np.isfinite(r.get(k, np.nan)) for k in ("raw_gap", "gap_upper", "uncertainty"))):
        return "UNRESOLVED"
    if r["uncertainty"] / scale > 1e-10 or r["raw_gap"] < -r["uncertainty"]:
        return "UNRESOLVED"
    if max(abs(r["raw_gap"]), r["gap_upper"]) / scale <= threshold:
        return "ZERO"
    if (r["raw_gap"] - r["uncertainty"]) / scale > threshold:
        return "POSITIVE"
    return "BORDERLINE"


def price_bounds(n):
    return [(0.0, None)] * n


def gcr_value(due_cost, so_cost, priced_cost, errors):
    if not np.all(np.isfinite([due_cost, so_cost, priced_cost, *errors])):
        return None
    denominator = due_cost - so_cost
    if denominator <= 100 * max(errors) or denominator <= 0:
        return None
    return (due_cost - priced_cost) / denominator


def constraints(p):
    return [
        LinearConstraint(p.conservation, p.demand, p.demand),
        LinearConstraint(p.occupation, -np.inf, p.capacity),
    ]


def solve_priced(p, tau, h0, scale, budget, final=False):
    budget.check()
    priced = PricedProblem(p, tau)
    sol = minimize(
        priced.cost,
        h0,
        jac=priced.gradient,
        method="SLSQP",
        bounds=Bounds(0, np.inf),
        constraints=constraints(p),
        callback=lambda x: budget.check(),
        options={"ftol": 1e-13, "maxiter": 800},
    )
    h = sol.x.copy()
    result = dict(
        flow=h, solver_success=bool(sol.success), status=str(sol.message), iterations=int(sol.nit)
    )
    if final and np.all(np.isfinite(h)) and (max(p.feasibility(h).values()) <= 1e-08):
        budget.check()
        correction = audit.refine_active_face(priced, h, max_iterations=200)
        result["before_refinement"] = {**result, **direct_gap(p, h, priced.gradient(h))}
        result["refinement"] = correction
        h = correction["flow"]
        result["flow"] = h
    result.update(direct_gap(p, h, priced.gradient(h)))
    result["passed"] = flow_pass(result, scale, 1e-09)
    result["system_cost"] = p.cost(h)
    return result


def local_search(inst, p, design, start, due, scale, budget, key, checkpoint=None):
    history, cache, candidates = ([], {}, [])

    def save(status):
        if checkpoint:
            checkpoint(
                dict(
                    status=status, evaluations=len(history), history=history, candidates=candidates
                )
            )

    def objective(theta):
        token = theta.tobytes()
        if token in cache:
            return cache[token]
        budget.consume(key, 100)
        save("EVALUATING_CURRENT_PRICE")
        r = solve_priced(p, design @ theta, due, scale, budget)
        gradient = _best_in_class_gradient(inst, r["flow"], design)
        if not np.isfinite(r["system_cost"]) or not np.all(np.isfinite(gradient)):
            raise ValueError("Nonfinite local evaluation")
        history.append(
            dict(
                theta=theta.copy(),
                cost=r["system_cost"],
                raw_gap=r["raw_gap"],
                gap_upper=r["gap_upper"],
                solver_success=r["solver_success"],
                passed=r["passed"],
            )
        )
        save("SEARCHING")
        cache[token] = (r["system_cost"], gradient)
        return cache[token]

    try:
        solved = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=price_bounds(len(start)),
            options={"maxiter": 24, "maxfun": 100, "ftol": 1e-12, "gtol": 1e-08},
        )
        endpoint = solved.x
        status = str(solved.message)
        iterations = int(solved.nit)
    except BudgetExceeded:
        budget.check()
        endpoint = history[-1]["theta"] if history else start
        status, iterations = ("evaluation budget reached", None)
    for name, theta in (("start", start), ("endpoint", endpoint)):
        if any((np.array_equal(theta, item["theta"]) for item in candidates)):
            continue
        save("VERIFYING_" + name.upper())
        verified = solve_priced(p, design @ theta, due, scale, budget, final=True)
        candidates.append(dict(name=name, theta=theta, result=verified))
        save("VERIFYING_FINAL_CANDIDATES")
    return dict(
        status=status,
        iterations=iterations,
        evaluations=len(history),
        history=history,
        candidates=candidates,
    )


def run_network(path, checkpoint=None):
    budget = Budget(1800)
    inst, p, designs = load_input(path)
    if not check_designs(inst, designs):
        raise ValueError("Frozen price-map mismatch")
    out = dict(
        seed=inst.config.seed, status="RUNNING", so=[], classes={}, local={}, calls=budget.calls
    )

    def save():
        if checkpoint:
            checkpoint(out)

    for tol in (1e-10, 1e-12):
        budget.check()
        conic = audit.solve_conic(p, tol)
        budget.check()
        polished = audit.polish_solution(p, conic["flow"])
        budget.check()
        refined = audit.refine_active_face(p, polished["flow"], 200)
        scale = max(1.0, abs(p.cost(refined["flow"])))
        refined["passed"] = flow_pass(refined, scale, 1e-10)
        out["so"].append(
            dict(
                conic=conic,
                polished=polished,
                refined=refined,
                cost=p.cost(refined["flow"]),
                scale=scale,
                first_order_cost_interval=[
                    p.cost(refined["flow"]) - refined["gap_upper"],
                    p.cost(refined["flow"]),
                ],
            )
        )
        save()
    scale = max((x["scale"] for x in out["so"]))
    passes = [x["refined"]["passed"] for x in out["so"]]
    for cls, design in designs.items():
        results = []
        for state in out["so"]:
            budget.consume("class_targets", 8)
            results.append(class_gap(p, state["refined"]["flow"], design))
        out["classes"][cls] = dict(
            results=results, classification=classify_pair(results, passes, scale)
        )
        out["classes"][cls]["threshold_checks"] = {
            str(threshold): [
                classify_threshold(r, accepted, scale, threshold)
                for r, accepted in zip(results, passes)
            ]
            for threshold in (1e-08, 1e-07, 1e-06)
        }
        save()
    if not all(passes):
        out["status"] = "UNRESOLVED_SO_PRECISION"
        save()
        return out
    hso = out["so"][-1]["refined"]["flow"]
    uniform = np.array([inst.demand[w] / np.sum(inst.od_of_alt == w) for w in inst.od_of_alt])
    due = solve_priced(p, np.zeros(inst.n_alt), uniform, scale, budget, final=True)
    out["due"] = due
    save()
    if not due["passed"]:
        out["status"] = "UNRESOLVED_DUE_PRECISION"
        save()
        return out
    for cls in ("corridor", "node", "contemporaneous"):
        design = designs[cls]
        parameterization = build_price_parameterization(inst, cls)
        gap_result = out["classes"][cls]["results"][-1]
        starts = [
            ("zero", np.zeros(design.shape[1])),
            ("delay_component", _component_price_start(inst, parameterization, hso)),
            ("minimum_class_gap_price", gap_result["theta"]),
        ]
        out["local"][cls] = {}
        for name, theta in starts:
            if not np.all(np.isfinite(theta)) or (
                name == "minimum_class_gap_price" and (not gap_result["success"])
            ):
                out["local"][cls][name] = dict(status="UNRESOLVED_START")
            else:

                def progress(value):
                    out["local"][cls][name] = value
                    save()

                out["local"][cls][name] = local_search(
                    inst,
                    p,
                    design,
                    theta,
                    due["flow"],
                    scale,
                    budget,
                    cls + "/" + name,
                    checkpoint=progress,
                )
            save()
        candidates = [
            dict(start=name, **c)
            for name, search in out["local"][cls].items()
            for c in search.get("candidates", [])
            if c["result"]["passed"]
        ]
        if candidates:
            best = min(candidates, key=lambda c: c["result"]["system_cost"])
            cso = p.cost(hso)
            errors = [
                due["gap_upper"],
                out["so"][-1]["refined"]["gap_upper"],
                best["result"]["gap_upper"],
            ]
            out["local"][cls]["accepted_summary"] = dict(
                accepted_candidates=len(candidates),
                chosen_start=best["start"],
                chosen_point=best["name"],
                system_cost=best["result"]["system_cost"],
                remaining_cost=best["result"]["system_cost"] - cso,
                GCR=gcr_value(due["system_cost"], cso, best["result"]["system_cost"], errors),
                classification="FEASIBLE_COMPUTED_PRICE_NOT_GLOBAL_OPTIMUM",
            )
        else:
            out["local"][cls]["accepted_summary"] = dict(
                accepted_candidates=0, classification="UNRESOLVED"
            )
    out["status"] = "FIXED_BATCH_FINISHED"
    save()
    return out
