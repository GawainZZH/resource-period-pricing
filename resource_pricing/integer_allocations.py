from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]


def bounds():
    name = "_protected_integer_old_bounds"
    if name not in sys.modules:
        path = Path(__file__).with_name("protected_bounds.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def build(model):
    D, E, H = (model.occupation, model.destination, model.conservation)
    if (
        not np.all((D == 0) | (D == 1))
        or not np.all((E == 0) | (E == 1))
        or np.any(E > D)
        or np.any(model.capacity != np.floor(model.capacity))
        or np.any(model.demand != np.floor(model.demand))
    ):
        raise ValueError("Integer capacities, demand and binary occupation are required")
    n = D.shape[1]
    groups, mapping = ([], {})
    for r, (row, dest, cap, alpha) in enumerate(zip(D, E, model.capacity, model.alpha)):
        if not np.any(row):
            continue
        key = (tuple((int(v) for v in row)), tuple((int(v) for v in dest)), int(cap))
        if key not in mapping:
            mapping[key] = len(groups)
            groups.append(
                {
                    "D": key[0],
                    "E": key[1],
                    "K": int(cap),
                    "alpha": F(0),
                    "rho": F(0),
                    "original_rows": [],
                }
            )
        group = groups[mapping[key]]
        group["alpha"] += F(float(alpha))
        group["rho"] += F(float(model.rho))
        group["original_rows"].append(r)
    center = [min((F(float(model.base[j])) for j in np.flatnonzero(row))) for row in H]
    owner = H.argmax(axis=0)
    cost = [F(float(value)) - center[owner[j]] for j, value in enumerate(model.base)]
    constant = sum((F(float(q)) * v for q, v in zip(model.demand, center)), F(0))
    rows, columns, values = ([], [], [])
    rhs = [int(v) for v in model.demand]
    for w, row in enumerate(H):
        for j in np.flatnonzero(row):
            rows.append(w)
            columns.append(int(j))
            values.append(1)
    eta = F(float(model.threshold))
    for group in groups:
        Dg, Eg, K = (np.array(group["D"]), np.array(group["E"]), group["K"])
        mode = "zero" if not np.any(Eg) else "same" if np.array_equal(Dg, Eg) else "mixed"
        states = [
            (y, z)
            for y in range(K + 1)
            for z in ([0] if mode == "zero" else [y] if mode == "same" else range(y + 1))
        ]
        onehot, load = (len(rhs), len(rhs) + 1)
        rhs += [1, 0]
        destination = None
        if mode == "mixed":
            destination = len(rhs)
            rhs.append(0)
        for j in np.flatnonzero(Dg):
            rows.append(load)
            columns.append(int(j))
            values.append(1)
        if destination is not None:
            for j in np.flatnonzero(Eg):
                rows.append(destination)
                columns.append(int(j))
                values.append(1)
        start = len(cost)
        for y, z in states:
            u = F(y, K)
            cost.append(z * group["alpha"] * u * u + y * group["rho"] * max(F(0), u - eta) ** 2)
            column = len(cost) - 1
            rows.append(onehot)
            columns.append(column)
            values.append(1)
            if y:
                rows.append(load)
                columns.append(column)
                values.append(-y)
            if destination is not None and z:
                rows.append(destination)
                columns.append(column)
                values.append(-z)
        group.update(
            mode=mode,
            states=states,
            start=start,
            stop=len(cost),
            onehot_row=onehot,
            load_row=load,
            destination_row=destination,
        )
    A = coo_matrix((values, (rows, columns)), shape=(len(rhs), len(cost)), dtype=np.int64).tocsr()
    upper = np.ones(len(cost))
    for j in range(n):
        upper[j] = min(float(model.demand[owner[j]]), float(np.min(model.capacity[D[:, j] == 1])))
    return {
        "A": A,
        "rhs": np.array(rhs, dtype=int),
        "cost": cost,
        "constant": constant,
        "groups": groups,
        "n": n,
        "upper": upper,
        "center": center,
        "owner": owner,
    }


def state_vector(model, built, raw):
    h = [F(v) for v in raw]
    if (
        len(h) != built["n"]
        or any((v.denominator != 1 for v in h))
        or (not bounds().is_exact_feasible(model, h))
    ):
        raise ValueError("An exactly feasible integer flow is required")
    x = np.zeros(len(built["cost"]), dtype=int)
    x[: built["n"]] = [int(v) for v in h]
    for group in built["groups"]:
        y = sum((v * int(d) for v, d in zip(h, group["D"])))
        z = sum((v * int(e) for v, e in zip(h, group["E"])))
        state = group["states"].index((int(y), int(z)))
        x[group["start"] + state] = 1
    if not np.array_equal(built["A"] @ x, built["rhs"]):
        raise ValueError("Finite-state encoding failed")
    return x


def integer_candidate(model, raw):
    raw = np.asarray(raw, float)
    if raw.shape != model.base.shape or not np.all(np.isfinite(raw)):
        return {"exact_integer_feasible": False, "reason": "Invalid returned flow"}
    nearest = np.rint(raw)
    if np.max(abs(raw - nearest)) > 1e-07:
        return {"exact_integer_feasible": False, "reason": "Returned flow is fractional"}
    h = [F(int(v)) for v in nearest]
    if not bounds().is_exact_feasible(model, h):
        return {
            "exact_integer_feasible": False,
            "reason": "Rounded flow violates original constraints",
        }
    cost = bounds().exact_original_cost(model, h)
    return {
        "exact_integer_feasible": True,
        "flow_integer": [int(v) for v in h],
        "cost_exact": str(cost),
        "cost_upper": bounds().upward(cost),
    }


def columns_product(A, dual):
    A = A.tocsc()
    return [
        sum(
            (
                int(v) * dual[int(r)]
                for r, v in zip(
                    A.indices[A.indptr[j] : A.indptr[j + 1]], A.data[A.indptr[j] : A.indptr[j + 1]]
                )
            ),
            F(0),
        )
        for j in range(A.shape[1])
    ]


def repair_dual(model, built, raw, scale):
    raw = np.asarray(raw, float)
    if (
        raw.shape != built["rhs"].shape
        or not np.all(np.isfinite(raw))
        or (not np.isfinite(scale))
        or (scale <= 0)
    ):
        raise ValueError("A finite LP equality-dual vector and positive scale are required")
    dual = [F(float(v)) * F(float(scale)) for v in raw]
    lhs = columns_product(built["A"], dual)
    for w, row in enumerate(model.conservation):
        dual[w] += min([F(0)] + [built["cost"][j] - lhs[j] for j in np.flatnonzero(row)])
    lhs = columns_product(built["A"], dual)
    for group in built["groups"]:
        dual[group["onehot_row"]] += min(
            [F(0)] + [built["cost"][j] - lhs[j] for j in range(group["start"], group["stop"])]
        )
    lhs = columns_product(built["A"], dual)
    if not all((a <= c for a, c in zip(lhs, built["cost"]))):
        raise ValueError("Exact LP dual repair failed")
    lower = built["constant"] + sum((F(int(b)) * v for b, v in zip(built["rhs"], dual)), F(0))
    return {
        "passed": True,
        "dual_exact": [str(v) for v in dual],
        "lower_exact": str(lower),
        "lower": bounds().downward(lower),
    }


def gap_bounds(continuous_lower, continuous_upper, integer_lower, integer_upper):
    LC, UC, LI, UI = map(F, (continuous_lower, continuous_upper, integer_lower, integer_upper))
    if not 0 < LC <= UC or not LC <= LI <= UI:
        raise ValueError("Invalid continuous or integer cost interval")
    low = max(F(0), LI / UC - 1)
    high = UI / LC - 1
    return {
        "continuous_lower": bounds().downward(LC),
        "continuous_upper": bounds().upward(UC),
        "integer_lower": bounds().downward(LI),
        "integer_upper": bounds().upward(UI),
        "integerization_loss_lower_exact": str(low),
        "integerization_loss_upper_exact": str(high),
        "integerization_loss_percent_lower": bounds().downward(100 * low),
        "integerization_loss_percent_upper": bounds().upward(100 * high),
        "integer_optimality_width_exact": str(UI - LI),
        "integer_optimality_width": bounds().upward(UI - LI),
        "integer_global_exact": LI == UI,
        "continuous_global_exact": LC == UC,
    }
