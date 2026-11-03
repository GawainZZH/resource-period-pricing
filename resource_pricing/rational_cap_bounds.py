from fractions import Fraction as F
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse
from resource_pricing.minimum_caps import affine, exact_check, exact_dual_check, exact_matvec

sys.set_int_max_str_digits(0)


def basis_vector(lp, raw):
    basic_col = [j for j, status in enumerate(raw["col_basis"]) if status.endswith("kBasic")]
    cap_active = [
        j for j in range(lp.k) if not raw["row_basis"][lp.meta["core_rows"] + j].endswith("kBasic")
    ]
    remaining = [j for j in basic_col if j not in cap_active]
    B = np.zeros((lp.n, len(remaining)), dtype=np.int64)
    for k, j in enumerate(remaining):
        B[j, k] = 1
    if cap_active:
        if lp.n - 1 not in remaining:
            return (None, {"reason": "Positive cap is not basic"})
        position = remaining.index(lp.n - 1)
        for j in cap_active:
            B[j, position] = 1
    active_core = [
        i
        for i in range(lp.meta["core_rows"])
        if lp.equality[i] or not raw["row_basis"][i].endswith("kBasic")
    ]
    A = np.asarray(lp.A @ B)
    z = affine(A[active_core], [lp.b[i] for i in active_core], np.asarray(raw["x"])[remaining])
    return (
        None if z is None else exact_matvec(sparse.csr_matrix(B), z),
        {
            "basic_columns": len(basic_col),
            "basis_cap_equalities": len(cap_active),
            "basis_core_equalities": len(active_core),
            "reduced_variables": len(remaining),
        },
    )
