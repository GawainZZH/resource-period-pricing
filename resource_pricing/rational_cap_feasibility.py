from fractions import Fraction as F
import json
import sys
import numpy as np
from resource_pricing.minimum_caps import affine, exact_matvec, exact_check, exact_dual_check

sys.set_int_max_str_digits(0)


def repair_stationarity(R, Dactive, H, entry, c, positive, theta, lam, phi):
    from scipy import sparse

    R, Dactive, H = [np.asarray(v, dtype=np.int64) for v in (R, Dactive, H)]
    theta = [max(F(0), F(float(v))) for v in theta]
    lam = [max(F(0), F(float(v))) for v in lam]
    phi = [F(float(v)) for v in phi]
    price = exact_matvec(sparse.csr_matrix(R.T), theta)
    normals = exact_matvec(sparse.csr_matrix(Dactive.T), lam)
    demand = exact_matvec(sparse.csr_matrix(H.T), phi)
    reduced = [cost + t + l + p for cost, t, l, p in zip(c, price, normals, demand)]
    active = [j for j, v in enumerate(reduced) if positive[j] or float(v) < 1e-07]
    A = np.column_stack([R[entry].T, H.T])
    correction = affine(A[active], [-reduced[j] for j in active], np.zeros(A.shape[1]))
    if correction is None:
        return (None, {"reason": "Exact stationarity correction is inconsistent"})
    for i, v in zip(entry, correction[: len(entry)]):
        theta[i] += v
    phi = [v + d for v, d in zip(phi, correction[len(entry) :])]
    neutral = R[entry].sum(axis=0)
    owner = H.argmax(axis=0)
    by_owner = [int(neutral[np.flatnonzero(row)[0]]) for row in H]
    if not np.array_equal(neutral, np.asarray(by_owner)[owner]):
        raise ValueError("Entrance shift is not neutral within OD")
    shift = max([F(0)] + [-theta[i] for i in entry])
    for i in entry:
        theta[i] += shift
    phi = [v - F(w) * shift for v, w in zip(phi, by_owner)]
    return (
        theta + lam + phi + [max(theta, default=F(0))],
        {
            "exact_stationarity_rows": len(active),
            "neutral_shift_exact": str(shift),
            "maximum_correction_exact": str(max(map(abs, correction), default=F(0))),
            "neutral_entry_charge_per_OD": by_owner,
            "additional_optimizer_calls": 0,
        },
    )
