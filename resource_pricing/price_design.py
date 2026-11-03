from fractions import Fraction as F
import numpy as np
import sympy as sp

PRIME = 2147483647


def integer_data(R, H):
    arrays = []
    for value in (R, H):
        a = np.asarray(value)
        if a.ndim != 2 or not np.all(np.isfinite(a)) or (not np.all(a == np.floor(a))):
            raise ValueError("An exact integer matrix is required")
        arrays.append(a.astype(np.int64))
    R, H = arrays
    if R.shape[1] != H.shape[1] or np.any(R < 0):
        raise ValueError("Nonnegative price map with matching choices required")
    if (
        not np.all((H == 0) | (H == 1))
        or not np.all(H.sum(axis=0) == 1)
        or (not np.all(H.sum(axis=1) > 0))
    ):
        raise ValueError("H must assign every choice to one nonempty OD")
    return (R, H)


def common_charge(R, H):
    charges = R.sum(axis=0)
    values = [int(charges[np.flatnonzero(row)[0]]) for row in H]
    common = all((np.all(charges[row == 1] == value) for row, value in zip(H, values)))
    return (bool(common), values)


def modular_rank(A):
    A = np.array(A, dtype=np.int64, copy=True) % PRIME
    rank = 0
    for column in range(A.shape[1]):
        candidates = np.flatnonzero(A[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        A[[rank, pivot]] = A[[pivot, rank]]
        A[rank] = A[rank] * pow(int(A[rank, column]), -1, PRIME) % PRIME
        for row in range(rank + 1, len(A)):
            if A[row, column]:
                A[row] = (A[row] - A[row, column] * A[rank]) % PRIME
        rank += 1
        if rank == min(A.shape):
            break
    return rank


def analyze(R, H):
    R, H = integer_data(R, H)
    A = np.vstack([R, H])
    unique = list(dict.fromkeys((tuple((int(x) for x in row)) for row in A)))
    reduced, pivots = sp.Matrix(unique).rref(normalize_last=True)
    rank = len(pivots)
    free = [j for j in range(A.shape[1]) if j not in pivots]
    nullspace = []
    for column in free:
        v = [sp.S.Zero] * A.shape[1]
        v[column] = sp.S.One
        for row, pivot in enumerate(pivots):
            v[pivot] = -reduced[row, column]
        nullspace.append([str(x) for x in v])
    common, values = common_charge(R, H)
    return {
        "rank": rank,
        "choices": A.shape[1],
        "price_parameters": len(R),
        "pivots": list(pivots),
        "free_columns": free,
        "nullspace": nullspace,
        "common_positive_charge": common,
        "common_charge_by_OD": values,
        "universal_representation_condition": bool(common and rank == A.shape[1]),
        "uses_target_cost": False,
        "new_optimization_calls": 0,
    }


def verify(R, H, output):
    R, H = integer_data(R, H)
    A = np.vstack([R, H])
    n, rank = (A.shape[1], output["rank"])
    assert output["choices"] == n and output["price_parameters"] == len(R)
    assert modular_rank(A) == rank
    free = output["free_columns"]
    assert len(set(free)) == len(free) == n - rank
    assert sorted(free + output["pivots"]) == list(range(n))
    assert len(output["nullspace"]) == n - rank
    for column, raw in zip(free, output["nullspace"]):
        v = [F(x) for x in raw]
        assert len(v) == n
        assert all((v[j] == (1 if j == column else 0) for j in free))
        assert all((sum((int(row[j]) * v[j] for j in np.flatnonzero(row)), F(0)) == 0 for row in A))
    common, values = common_charge(R, H)
    assert output["common_positive_charge"] == common
    assert output["common_charge_by_OD"] == values
    assert output["universal_representation_condition"] == (common and rank == n)
    assert not output["uses_target_cost"] and output["new_optimization_calls"] == 0
    return True
