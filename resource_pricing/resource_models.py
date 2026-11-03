from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.linalg import null_space, qr, solve
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LinearResourceNetwork:
    conservation: np.ndarray
    demand: np.ndarray
    delta: np.ndarray
    capacity: np.ndarray
    base_cost: np.ndarray
    delay_slope: np.ndarray
    periods: np.ndarray

    def load(self, flow: np.ndarray) -> np.ndarray:
        return self.delta @ flow

    def private_cost(self, flow: np.ndarray) -> np.ndarray:
        return self.base_cost + self.delta.T @ (self.delay_slope * self.load(flow))

    def delay_externality(self, flow: np.ndarray) -> np.ndarray:
        return self.delay_slope * self.load(flow)

    def system_gradient(self, flow: np.ndarray) -> np.ndarray:
        return self.private_cost(flow) + self.delta.T @ self.delay_externality(flow)

    def total_cost(self, flow: np.ndarray) -> float:
        return float(flow @ self.private_cost(flow))


def boundary_network(node_period: int = 0, background: float = 1.0) -> LinearResourceNetwork:
    if node_period < 0 or background < 0:
        raise ValueError("Periods and background demand must be nonnegative.")
    return LinearResourceNetwork(
        conservation=np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        demand=np.array([1.0, background]),
        delta=np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        capacity=np.array([2.0, 3.0]),
        base_cost=np.array([0.0, 1.5, 0.0]),
        delay_slope=np.array([0.0, 1.0]),
        periods=np.array([0, node_period]),
    )


def boundary_objective_derivative(x: float) -> float:
    return 0.5 + 2.0 * x


def boundary_price_response(node_price=Fraction(0)) -> tuple[Fraction, Fraction, Fraction]:
    """Exact equilibrium; common corridor prices cancel between target choices."""
    try:
        theta = Fraction(node_price)
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError("The node price must be finite and nonnegative.") from exc
    if theta < 0:
        raise ValueError("The node price must be finite and nonnegative.")
    x = max(Fraction(1, 2) - theta, Fraction(0))
    return (x, 1 - x, Fraction(1))


def boundary_class_value(price_class, *, node_period=0, node_cap=None) -> dict:
    """Global values for the existing two-choice background-flow examples."""
    if price_class not in ("omitted_node", "complete"):
        raise ValueError("Unknown boundary price class.")
    if not isinstance(node_period, int) or node_period < 0:
        raise ValueError("The node period must be a nonnegative integer.")
    cap = None
    if node_cap is not None:
        try:
            cap = Fraction(node_cap)
        except (ValueError, OverflowError, TypeError) as exc:
            raise ValueError("The node cap must be finite and nonnegative.") from exc
        if cap < 0:
            raise ValueError("The node cap must be finite and nonnegative.")
    theta = (
        Fraction(0)
        if price_class == "omitted_node"
        else Fraction(1, 2) if cap is None else min(cap, Fraction(1, 2))
    )
    flow = boundary_price_response(theta)
    x = flow[0]
    so_cost = Fraction(5, 2)
    cost = so_cost + x / 2 + x * x
    return {
        "node_price": theta,
        "flow": flow,
        "system_cost": cost,
        "so_cost": so_cost,
        "selection_robust_cost": cost,
        "gcr": (3 - cost) / Fraction(1, 2),
        "minimum_so_gap": x,
    }


def twin_path_network() -> LinearResourceNetwork:
    return LinearResourceNetwork(
        conservation=np.array([[1.0, 1.0, 1.0]]),
        demand=np.array([1.0]),
        delta=np.array([[1.0, 1.0, 0.0]]),
        capacity=np.array([2.0]),
        base_cost=np.array([0.0, 0.0, 1.0]),
        delay_slope=np.array([1.0]),
        periods=np.array([1]),
    )


def primal_gap(network: LinearResourceNetwork, flow: np.ndarray, cost: np.ndarray) -> float:
    solved = linprog(
        cost,
        A_ub=network.delta,
        b_ub=network.capacity,
        A_eq=network.conservation,
        b_eq=network.demand,
        bounds=(0.0, None),
        method="highs-ds",
    )
    if not solved.success:
        raise RuntimeError(solved.message)
    return float(cost @ flow - solved.fun)


def minimum_gap(
    network: LinearResourceNetwork,
    flow: np.ndarray,
    design: np.ndarray,
    price_bounds: list[tuple[float | None, float | None]] | None = None,
) -> dict:
    n_prices = design.shape[1]
    cost = network.private_cost(flow)
    objective = np.concatenate([design.T @ flow, network.capacity, network.demand])
    inequality = np.hstack([-design, -network.delta.T, -network.conservation.T])
    bounds = (
        (price_bounds if price_bounds is not None else [(0.0, None)] * n_prices)
        + [(0.0, None)] * network.delta.shape[0]
        + [(None, None)] * network.conservation.shape[0]
    )
    solved = linprog(
        objective,
        A_ub=inequality,
        b_ub=cost,
        bounds=bounds,
        method="highs-ds",
        options={"dual_feasibility_tolerance": 1e-09, "primal_feasibility_tolerance": 1e-09},
    )
    if not solved.success:
        raise RuntimeError(solved.message)
    price = design @ solved.x[:n_prices]
    value = float(cost @ flow + solved.fun)
    direct = primal_gap(network, flow, cost + price)
    return {
        "success": True,
        "gap": value,
        "direct_gap": direct,
        "duality_error": abs(value - direct),
        "parameters": solved.x[:n_prices].tolist(),
    }


def mapping_compatibility(
    conservation: np.ndarray,
    delta: np.ndarray,
    design: np.ndarray,
    *,
    rcond: float = 1e-11,
    tolerance: float = 1e-10,
) -> dict:
    matrices = [np.asarray(matrix, dtype=float) for matrix in (conservation, delta, design)]
    conservation, delta, design = matrices
    if any((matrix.ndim != 2 or not np.all(np.isfinite(matrix)) for matrix in matrices)):
        raise ValueError("Finite two-dimensional matrices are required.")
    if conservation.shape[1] != delta.shape[1] or design.shape[0] != delta.shape[1]:
        raise ValueError("The flow dimensions of the three matrices must match.")
    constraints = np.vstack([conservation, delta])
    constraints = constraints[np.any(constraints != 0, axis=1)]
    basis = null_space(constraints, rcond=rcond)
    error = float(np.linalg.norm(design.T @ basis, ord="fro"))
    scale = max(1.0, float(np.linalg.norm(design, ord="fro")))
    return {
        "nullity": int(basis.shape[1]),
        "relative_defect": error / scale,
        "compatible": bool(error / scale <= tolerance),
        "rank_tolerance": rcond,
        "compatibility_tolerance": tolerance,
    }


def _modular_determinant(matrix: np.ndarray, prime: int = 1000003) -> int:
    work = np.asarray(matrix, dtype=np.int64).copy() % prime
    value = 1
    for k in range(len(work)):
        nonzero = np.flatnonzero(work[k:, k])
        if not len(nonzero):
            return 0
        pivot_row = k + int(nonzero[0])
        if pivot_row != k:
            work[[k, pivot_row]] = work[[pivot_row, k]]
            value = -value
        pivot = int(work[k, k])
        value = value * pivot % prime
        inverse = pow(pivot, -1, prime)
        factors = work[k + 1 :, k] * inverse % prime
        work[k + 1 :, k:] = (work[k + 1 :, k:] - factors[:, None] * work[k, k:][None, :]) % prime
    return int(value % prime)


def integer_kernel_certificate(conservation: np.ndarray, delta: np.ndarray) -> dict:
    constraints = np.vstack([conservation, delta])
    if not np.all(np.isfinite(constraints)) or not np.array_equal(
        constraints, np.round(constraints)
    ):
        raise ValueError(
            "Exact verification requires integer occupation and conservation matrices."
        )
    constraints = constraints[np.any(constraints != 0, axis=1)].astype(np.int64)
    _, triangular, column_order = qr(constraints.astype(float), mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    rank = int(np.sum(diagonal > 1e-11 * max(1.0, float(diagonal.max(initial=0.0)))))
    pivot_columns = column_order[:rank]
    free_columns = column_order[rank:]
    _, _, row_order = qr(
        constraints[:, pivot_columns].T.astype(float), mode="economic", pivoting=True
    )
    pivot_rows = row_order[:rank]
    minor = constraints[np.ix_(pivot_rows, pivot_columns)]
    prime = 1000003
    determinant = _modular_determinant(minor, prime)
    if determinant == 0:
        raise RuntimeError(
            "The selected minor is not verified nonsingular modulo the chosen prime."
        )
    coefficients = (
        solve(
            minor.astype(float),
            -constraints[np.ix_(pivot_rows, free_columns)].astype(float),
            check_finite=False,
        )
        if len(free_columns)
        else np.empty((rank, 0))
    )
    basis = np.zeros((constraints.shape[1], len(free_columns)), dtype=np.int64)
    for j, free_column in enumerate(free_columns):
        fractions = [
            Fraction(float(value)).limit_denominator(10000) for value in coefficients[:, j]
        ]
        denominator = math.lcm(*(value.denominator for value in fractions))
        entries = [value.numerator * (denominator // value.denominator) for value in fractions]
        if max([denominator, *(abs(entry) for entry in entries)]) >= np.iinfo(np.int64).max // max(
            1, constraints.shape[1]
        ):
            raise OverflowError(
                "Rational reconstruction exceeds the safe exact integer product range."
            )
        basis[pivot_columns, j] = entries
        basis[free_column, j] = denominator
    product_bound = (
        constraints.shape[1]
        * int(np.max(np.abs(constraints), initial=0))
        * int(np.max(np.abs(basis), initial=0))
    )
    if product_bound >= np.iinfo(np.int64).max:
        raise OverflowError("The exact kernel product may overflow int64.")
    if np.any(constraints @ basis != 0):
        raise RuntimeError("Rational reconstruction failed the exact integer kernel identity.")
    return {
        "verified": True,
        "rank": rank,
        "nullity": int(basis.shape[1]),
        "basis": basis,
        "pivot_columns": pivot_columns,
        "pivot_rows": pivot_rows,
        "prime": prime,
        "minor_determinant_mod_prime": determinant,
    }


def masked_temporal_counterexample() -> dict:
    network = LinearResourceNetwork(
        conservation=np.array([[1.0, 1.0]]),
        demand=np.array([1.0]),
        delta=np.array([[1.0, 1.0]]),
        capacity=np.array([2.0]),
        base_cost=np.zeros(2),
        delay_slope=np.ones(1),
        periods=np.array([1]),
    )
    design = np.array([[0.0], [1.0]])
    h1, h2 = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    return {
        "system_cost_1": network.total_cost(h1),
        "system_cost_2": network.total_cost(h2),
        "minimum_gap_1": minimum_gap(network, h1, design, [(1.0, 2.0)])["gap"],
        "minimum_gap_2": minimum_gap(network, h2, design, [(1.0, 2.0)])["gap"],
        "price_domain": "[1,2]",
    }


def nonmonotone_resource_costs(instance, flow: float) -> np.ndarray:
    """Embed the existing scalar example in two disjoint resource costs."""
    from resource_pricing.equilibrium_selection import private_cost_difference, system_objective

    if not 0.0 <= flow <= 1.0:
        raise ValueError("The flow must belong to [0,1].")
    difference = private_cost_difference(instance, flow, 0.0)
    objective = system_objective(instance, flow)
    offset = 1.0 + instance.interaction_scale
    return np.array(
        [offset + objective + (1.0 - flow) * difference, offset + objective - flow * difference]
    )


def audit_existing_mappings(
    instances: int = 200, seed: int = 20260625, witness_dir: Path | None = None
) -> list[dict]:
    import pandas as pd
    from resource_pricing.restricted_pricing import build_price_parameterization
    from resource_pricing.controlled_networks import _config_stream, _constraints, generate_instance

    source = ROOT / "data/class_gaps/initial/minimum_primal_gap_constructed_instances.csv"
    old = pd.read_csv(source).set_index("Seed")
    rows = []
    for config in _config_stream(instances, seed):
        instance = generate_instance(config)
        if config.seed not in old.index:
            raise ValueError(f"Instance {config.seed} is absent from the manuscript data.")
        previous = old.loc[config.seed]
        expected = {
            "Alternatives": instance.n_alt,
            "Resources": instance.n_resources,
            "Capacity configuration": config.bottleneck_type,
            "Pressure": config.pressure,
            "Reliability": config.reliability,
        }
        if any((previous[field] != value for field, value in expected.items())):
            raise ValueError("Regenerated metadata differ from the manuscript data.")
        conservation = np.asarray(_constraints(instance)[0].A)
        delta = instance.delta.reshape(-1, instance.n_alt)
        exact = integer_kernel_certificate(conservation, delta)
        if witness_dir is not None:
            witness_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                witness_dir / f"kernel_{config.seed}.npz",
                conservation=conservation.astype(np.int64),
                delta=delta.astype(np.int64),
                basis=exact["basis"],
                pivot_rows=exact["pivot_rows"],
                pivot_columns=exact["pivot_columns"],
                prime=exact["prime"],
                determinant=exact["minor_determinant_mod_prime"],
            )
        for price_class in ("corridor", "node", "contemporaneous"):
            design = build_price_parameterization(instance, price_class).design
            result = mapping_compatibility(conservation, delta, design)
            if not np.array_equal(design, np.round(design)):
                raise ValueError("The price design is not integer-valued.")
            exact_product = design.T.astype(np.int64) @ exact["basis"]
            rows.append(
                {
                    "seed": config.seed,
                    "price_class": price_class,
                    "alternatives": instance.n_alt,
                    "positive_delay_coefficients": bool(np.all(instance.alpha > 0)),
                    "source_metadata_verified": True,
                    "exact_rank_verified": exact["verified"],
                    "exact_compatible": bool(np.all(exact_product == 0)),
                    **result,
                }
            )
    return rows


def run_audit(out_dir: Path, instances: int = 200) -> dict:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    boundary = []
    for node_period in (0, 2):
        network = boundary_network(node_period)
        flow = np.array([0.0, 1.0, 1.0])
        boundary.append(
            {
                "node_period": node_period,
                "so_cost": network.total_cost(flow),
                "so_first_order_gap": primal_gap(network, flow, network.system_gradient(flow)),
                "node_load": float(network.load(flow)[1]),
                "node_externality": float(network.delay_externality(flow)[1]),
                "restricted_minimum_gap": minimum_gap(network, flow, network.delta[:1].T)["gap"],
                "complete_minimum_gap": minimum_gap(network, flow, network.delta.T)["gap"],
            }
        )
    rows = audit_existing_mappings(instances, witness_dir=out_dir / "integer_kernel_witnesses")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "existing_price_mapping_audit.csv", index=False)
    from resource_pricing.equilibrium_selection import (
        private_cost_difference,
        selection_instances,
        system_objective,
    )

    embedding = []
    for instance in selection_instances():
        difference_errors, objective_errors, all_costs = ([], [], [])
        for x in np.linspace(0.0, 1.0, 101):
            costs = nonmonotone_resource_costs(instance, float(x))
            all_costs.extend(costs.tolist())
            difference_errors.append(
                abs(costs[0] - costs[1] - private_cost_difference(instance, x, 0.0))
            )
            objective_errors.append(
                abs(
                    x * costs[0]
                    + (1.0 - x) * costs[1]
                    - (1.0 + instance.interaction_scale)
                    - system_objective(instance, x)
                )
            )
        embedding.append(
            {
                "instance": instance.name,
                "minimum_resource_cost": min(all_costs),
                "maximum_cost_difference_error": max(difference_errors),
                "maximum_aggregate_objective_error": max(objective_errors),
            }
        )
    pd.DataFrame(embedding).to_csv(out_dir / "nonmonotone_resource_embedding.csv", index=False)
    summary = {
        "boundary_examples": boundary,
        "masked_temporal_example": masked_temporal_counterexample(),
        "nonmonotone_resource_embedding": embedding,
        "matrix_checks": {
            price_class: {
                "instances": int(len(group)),
                "compatible": int(group["compatible"].sum()),
                "exact_compatible": int(group["exact_compatible"].sum()),
                "exact_rank_verified": int(group["exact_rank_verified"].sum()),
                "nontrivial_nullspace": int((group["nullity"] > 0).sum()),
                "largest_nullity": int(group["nullity"].max()),
                "largest_relative_defect": float(group["relative_defect"].max()),
            }
            for price_class, group in table.groupby("price_class")
        },
        "interpretation": "Matrix compatibility is a sufficient condition, not a classification of SO decentralization.",
    }
    (out_dir / "mathematical_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
