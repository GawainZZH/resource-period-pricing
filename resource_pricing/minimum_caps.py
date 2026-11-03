from dataclasses import dataclass
from fractions import Fraction as F
import time
import numpy as np
from scipy import sparse
from scipy.linalg import qr
import sympy as sp


@dataclass
class CapLP:
    A: sparse.csr_matrix
    b: list
    equality: np.ndarray
    free: np.ndarray
    k: int
    names: list
    meta: dict

    @property
    def n(self):
        return self.A.shape[1]


def make_lp(core, rhs, equality, k, free_extra, names, meta):
    core = sparse.csr_matrix(core, dtype=np.int64)
    core = sparse.hstack([core, sparse.csr_matrix((core.shape[0], 1))], format="csr")
    cap = sparse.hstack(
        [
            sparse.eye(k, format="csr", dtype=np.int64),
            sparse.csr_matrix((k, core.shape[1] - k - 1)),
            -np.ones((k, 1), dtype=np.int64),
        ],
        format="csr",
    )
    A = sparse.vstack([core, cap], format="csr").astype(np.int64)
    return CapLP(
        A,
        list(map(F, rhs)) + [F(0)] * k,
        np.r_[equality, np.zeros(k, bool)].astype(bool),
        np.r_[np.zeros(k, bool), free_extra, False].astype(bool),
        k,
        names + ["cap"],
        meta,
    )


def continuous_lp(D, H, h, K, c, R):
    D, H, R = (np.asarray(v, dtype=np.int64) for v in (D, H, R))
    h, K, c = (list(map(F, v)) for v in (h, K, c))
    loads = [sum((h[j] * int(row[j]) for j in np.flatnonzero(row)), F(0)) for row in D]
    if min(h) < 0 or any((y > k for y, k in zip(loads, K))):
        raise ValueError("Infeasible fixed target")
    active = np.array([r for r, (y, k) in enumerate(zip(loads, K)) if y == k], dtype=int)
    A = np.column_stack([R.T, D[active].T, H.T])
    positive = np.array([v > 0 for v in h])
    sign = np.where(positive, 1, -1)
    A = sign[:, None] * A
    rhs = [-c[j] * int(sign[j]) for j in range(len(c))]
    k, a, w = (len(R), len(active), len(H))
    return make_lp(
        A,
        rhs,
        positive,
        k,
        np.r_[np.zeros(a, bool), np.ones(w, bool)],
        [f"theta_{i}" for i in range(k)]
        + [f"lambda_{i}" for i in active]
        + [f"phi_{i}" for i in range(w)],
        {
            "kind": "continuous",
            "active_capacity_rows": active.tolist(),
            "positive_target_coordinates": np.flatnonzero(positive).tolist(),
            "core_rows": len(c),
        },
    )


def integer_lp(R, deviations):
    R = np.asarray(R, dtype=np.int64)
    A = np.array([R[:, i] - R[:, j] for i, j, _ in deviations], dtype=np.int64)
    rhs = [-F(gain) for _, _, gain in deviations]
    return make_lp(
        A,
        rhs,
        np.zeros(len(rhs), bool),
        len(R),
        np.zeros(0, bool),
        [f"theta_{i}" for i in range(len(R))],
        {
            "kind": "integer",
            "core_rows": len(rhs),
            "deviations": [[i, j, str(g)] for i, j, g in deviations],
        },
    )


def solve_once(lp, time_limit=50):
    import highspy

    h = highspy.Highs()
    options = {
        "solver": "simplex",
        "presolve": "off",
        "threads": 1,
        "parallel": "off",
        "random_seed": 0,
        "output_flag": False,
        "time_limit": float(time_limit),
        "primal_feasibility_tolerance": 1e-09,
        "dual_feasibility_tolerance": 1e-09,
    }
    for key, value in options.items():
        status = h.setOptionValue(key, value)
        if status != highspy.HighsStatus.kOk:
            raise ValueError(f"Invalid solver option: {key}")
    a = lp.A.tocsr()
    model = highspy.HighsLp()
    model.num_col_, model.num_row_ = (lp.n, a.shape[0])
    model.col_cost_ = np.r_[np.zeros(lp.n - 1), 1.0]
    model.col_lower_ = np.where(lp.free, -highspy.kHighsInf, 0.0)
    model.col_upper_ = np.full(lp.n, highspy.kHighsInf)
    b = np.array([float(v) for v in lp.b])
    model.row_lower_ = np.where(lp.equality, b, -highspy.kHighsInf)
    model.row_upper_ = b
    model.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.a_matrix_.start_, model.a_matrix_.index_, model.a_matrix_.value_ = (
        a.indptr,
        a.indices,
        a.data.astype(float),
    )
    if h.passModel(model) != highspy.HighsStatus.kOk:
        raise ValueError("LP model rejected")
    start = time.perf_counter()
    run_status = h.run()
    elapsed = time.perf_counter() - start
    status = h.modelStatusToString(h.getModelStatus())
    solution, basis, info = (h.getSolution(), h.getBasis(), h.getInfo())
    result = {
        "status": status,
        "run_status": str(run_status),
        "solver_seconds": elapsed,
        "options": options,
        "solver_version": h.version(),
        "solver_calls": 1,
        "objective": float(info.objective_function_value),
        "simplex_iterations": int(info.simplex_iteration_count),
        "x": list(solution.col_value),
        "row_dual": list(solution.row_dual),
        "col_dual": list(solution.col_dual),
        "col_basis": [str(v) for v in basis.col_status],
        "row_basis": [str(v) for v in basis.row_status],
        "primal_status": int(info.primal_solution_status),
        "dual_status": int(info.dual_solution_status),
        "max_primal_infeasibility": float(info.max_primal_infeasibility),
        "max_dual_infeasibility": float(info.max_dual_infeasibility),
    }
    if status == "Infeasible":
        _, exists = h.getDualRayExist()
        if exists:
            _, valid, ray = h.getDualRay()
            if valid:
                result["dual_ray"] = ray.tolist()
    return result


def exact_matvec(A, x):
    A = A.tocsr()
    return [
        sum(
            (int(A.data[p]) * x[int(A.indices[p])] for p in range(A.indptr[r], A.indptr[r + 1])),
            F(0),
        )
        for r in range(A.shape[0])
    ]


def exact_check(lp, x):
    x = list(map(F, x))
    ax = exact_matvec(lp.A, x)
    eq = all((v == b for v, b, same in zip(ax, lp.b, lp.equality) if same))
    margins = [b - v for v, b, same in zip(ax, lp.b, lp.equality) if not same]
    bounds = all((v >= 0 for v, free in zip(x, lp.free) if not free))
    return {
        "feasible": bool(eq and bounds and all((v >= 0 for v in margins))),
        "equalities_exact": eq,
        "nonnegative_variables": bounds,
        "minimum_inequality_margin_exact": str(min(margins, default=F(0))),
    }


def exact_dual_check(lp, y, objective=True):
    y = list(map(F, y))
    aty = exact_matvec(lp.A.T, y)
    cost = [F(0)] * (lp.n - 1) + [F(1) if objective else F(0)]
    rc = [c - v for c, v in zip(cost, aty)]
    valid = all((v <= 0 for v, same in zip(y, lp.equality) if not same)) and all(
        (v == 0 if free else v >= 0 for v, free in zip(rc, lp.free))
    )
    value = sum((b * v for b, v in zip(lp.b, y)), F(0))
    return (valid, value, rc)


def affine(A, b, raw):
    """Select an independent exact square system, then verify every original row."""
    A = np.asarray(A, dtype=np.int64)
    raw = list(map(float, raw))
    if A.shape[1] == 0:
        return [] if all((F(v) == 0 for v in b)) else None
    if not len(A):
        return [F(v).limit_denominator(10**9) for v in raw]
    _, r, col = qr(A.astype(float), mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(r)) > max(1.0, np.max(np.abs(r))) * 1e-10))
    pivots = list(map(int, col[:rank]))
    free = [j for j in range(A.shape[1]) if j not in pivots]
    _, _, row = qr(A[:, pivots].T.astype(float), mode="economic", pivoting=True)
    rows = list(map(int, row[:rank]))
    x = [F(0)] * A.shape[1]
    for j in free:
        x[j] = F(raw[j]).limit_denominator(10**9)
    if rank:
        matrix = sp.Matrix(A[np.ix_(rows, pivots)].tolist())
        inverse = matrix.inv(method="DM")
        rhs = [F(b[i]) - sum((int(A[i, j]) * x[j] for j in free), F(0)) for i in rows]
        for k, j in enumerate(pivots):
            x[j] = sum((F(inverse[k, l]) * rhs[l] for l in range(rank)), F(0))
    if all((sum((int(a) * v for a, v in zip(row, x)), F(0)) == F(rhs) for row, rhs in zip(A, b))):
        return x
    return None


def reconstruct(lp, raw):
    """No optimizer calls: exact primal/dual witnesses or an explicit incomplete check."""
    if raw["status"] == "Infeasible":
        ray = raw.get("dual_ray")
        if ray is None:
            return {"exact_infeasibility": False, "reason": "No saved dual ray"}
        for sign in (1, -1):
            y = [sign * F(float(v)).limit_denominator(10**9) for v in ray]
            valid, value, _ = exact_dual_check(lp, y, objective=False)
            if valid and value > 0:
                return {
                    "exact_infeasibility": True,
                    "dual_ray_exact": list(map(str, y)),
                    "contradiction_exact": str(value),
                }
        return {"exact_infeasibility": False, "reason": "Saved ray not rationally verified"}
    if raw["status"] != "Optimal":
        return {
            "exact_primal_feasible": False,
            "exact_dual_feasible": False,
            "reason": "Solver did not return an optimal basis",
        }
    x0 = np.asarray(raw["x"])
    ax0 = lp.A @ x0
    b0 = np.array([float(v) for v in lp.b])
    active = lp.equality | (np.abs(ax0 - b0) < 1e-08)
    cap_positive = x0[-1] > 1e-08
    cap_indices = [j for j in range(lp.k) if cap_positive and abs(x0[j] - x0[-1]) < 1e-08]
    keep = [j for j in range(lp.n - 1) if (lp.free[j] or x0[j] > 1e-08) and j not in cap_indices]
    if cap_positive:
        keep.append(lp.n - 1)
    B = np.zeros((lp.n, len(keep)), dtype=np.int64)
    for k, j in enumerate(keep):
        B[j, k] = 1
    if cap_positive:
        for j in cap_indices:
            B[j, -1] = 1
    reduced = np.asarray(lp.A @ B)
    chosen = np.flatnonzero(active)
    z = affine(reduced[chosen], [lp.b[i] for i in chosen], x0[keep])
    x = None if z is None else exact_matvec(sparse.csr_matrix(B), z)
    primal = {"feasible": False} if x is None else exact_check(lp, x)
    y = [F(float(v)).limit_denominator(10**9) for v in raw["row_dual"]]
    dual_valid, lower, rc = exact_dual_check(lp, y)
    if not dual_valid:
        support = [i for i, v in enumerate(raw["row_dual"]) if abs(v) > 1e-10]
        zeros = [j for j in range(lp.n) if lp.free[j] or x0[j] > 1e-08]
        matrix = lp.A[support].T[zeros].toarray()
        rhs = [F(int(j == lp.n - 1)) for j in zeros]
        small = affine(matrix, rhs, np.asarray(raw["row_dual"])[support])
        if small is not None:
            y = [F(0)] * lp.A.shape[0]
            for i, v in zip(support, small):
                y[i] = v
            dual_valid, lower, rc = exact_dual_check(lp, y)
    report = {
        "exact_primal_feasible": bool(primal["feasible"]),
        "exact_dual_feasible": bool(dual_valid),
        "primal_checks": primal,
        "equal_bounds": bool(primal["feasible"] and dual_valid and (x[-1] == lower)),
        "reconstruction_variables": len(keep),
        "cap_active_coordinates": len(cap_indices),
    }
    if x is not None:
        report["x_exact"] = list(map(str, x))
        report["upper_exact"] = str(x[-1])
    if dual_valid:
        report["lower_exact"] = str(lower)
        report["row_dual_exact"] = list(map(str, y))
        report["reduced_cost_exact"] = list(map(str, rc))
    return report
