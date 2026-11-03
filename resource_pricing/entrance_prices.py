from fractions import Fraction as F
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import resource_pricing.exact_prices as core
import resource_pricing.price_design as structure


def construct(R, H, c):
    R, H = structure.integer_data(R, H)
    common, constants = structure.common_charge(R, H)
    if not common:
        raise ValueError("The prescribed positive direction must be OD-constant")
    A = np.column_stack([R.T, -H.T])
    value, rank = core.solve_affine(A, [-F(v) for v in c], np.zeros(A.shape[1]))
    if value is None or rank != len(c):
        raise ValueError("Full-map representation failed the proven rank condition")
    theta0, nu0 = (value[: len(R)], value[len(R) :])
    shift = max([F(0)] + [-v for v in theta0])
    theta = [v + shift for v in theta0]
    nu = [v + F(a) * shift for v, a in zip(nu0, constants)]
    out = {
        "theta_initial_exact": [str(v) for v in theta0],
        "nu_initial_exact": [str(v) for v in nu0],
        "shift_exact": str(shift),
        "theta_exact": [str(v) for v in theta],
        "nu_exact": [str(v) for v in nu],
        "common_charge_by_OD": constants,
        "rank": rank,
        "max_price_per_minute": core.bounds().upward(max(theta, default=F(0))),
    }
    check(R, H, c, out)
    out["passed"] = True
    return out


def check(R, H, c, out):
    R, H = structure.integer_data(R, H)
    common, constants = structure.common_charge(R, H)
    assert common and constants == out["common_charge_by_OD"]
    theta0 = [F(v) for v in out["theta_initial_exact"]]
    nu0 = [F(v) for v in out["nu_initial_exact"]]
    theta = [F(v) for v in out["theta_exact"]]
    nu = [F(v) for v in out["nu_exact"]]
    shift = F(out["shift_exact"])
    assert len(theta0) == len(theta) == len(R) and len(nu0) == len(nu) == len(H)
    assert len(c) == R.shape[1] and out["rank"] == len(c)
    assert shift == max([F(0)] + [-v for v in theta0])
    assert theta == [v + shift for v in theta0] and all((v >= 0 for v in theta))
    assert nu == [v + F(a) * shift for v, a in zip(nu0, constants)]
    for j, cost in enumerate(c):
        assert F(cost) + core.dot(R[:, j], theta0) == core.dot(H[:, j], nu0)
        assert F(cost) + core.dot(R[:, j], theta) == core.dot(H[:, j], nu)
    assert out["max_price_per_minute"] == core.bounds().upward(max(theta, default=F(0)))
    return True
