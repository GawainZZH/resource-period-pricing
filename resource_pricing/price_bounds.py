import argparse
import csv
import io
import json
from fractions import Fraction
from itertools import product
from math import isqrt

NETWORKS = ("linear_contemporaneous", "quadratic_delayed")
PRICE_CLASSES = ("corridor", "node", "contemporaneous")
CAPS = (Fraction(0), Fraction(1, 4), Fraction(1, 2))
ROOT_BITS = 80


def sqrt_dyadic_bounds(value: Fraction, bits: int = 80) -> tuple[Fraction, Fraction]:
    """Enclose sqrt(value) by integer-square-root arithmetic only."""
    if not isinstance(value, (int, Fraction)) or value < 0:
        raise ValueError("The radicand must be an exact nonnegative rational.")
    if not isinstance(bits, int) or bits < 1:
        raise ValueError("The dyadic precision must be a positive integer.")
    value = Fraction(value)
    scale = 2**bits
    numerator = isqrt(value.numerator * scale**2 // value.denominator)
    lower = Fraction(numerator, scale)
    upper = lower if lower**2 == value else Fraction(numerator + 1, scale)
    return (lower, upper)


def covered_rows(network: str, price_class: str) -> tuple[int, ...]:
    """Return every independently priceable row, not a marginal-price ray."""
    if network not in NETWORKS or price_class not in PRICE_CLASSES:
        raise ValueError("Only the approved two networks and three classes are allowed.")
    if price_class == "node":
        return (1,)
    if price_class == "contemporaneous" and network == NETWORKS[0]:
        return (0, 1)
    return (0,)


def equilibrium_response(network: str, node_price: Fraction) -> dict:
    """Symbolic unique equilibrium and rational enclosures, never endpoint flows."""
    if network not in NETWORKS:
        raise ValueError("Unknown analytical network.")
    if not isinstance(node_price, (int, Fraction)) or not 0 <= node_price <= Fraction(1, 2):
        raise ValueError("The exact effective node price must lie in [0, 1/2].")
    theta = Fraction(node_price)
    if network == NETWORKS[0]:
        x = Fraction(1, 2) - theta
        return {
            "exact_x": str(x),
            "rational_x": x,
            "x_bounds": (x, x),
            "root_radicand": None,
            "root_bounds": None,
            "bounds_are_equilibria": False,
        }
    radicand = Fraction(3, 2) - theta
    lower, upper = sqrt_dyadic_bounds(radicand, ROOT_BITS)
    rational_x = lower - 1 if lower == upper else None
    return {
        "exact_x": str(rational_x) if rational_x is not None else f"sqrt({radicand}) - 1",
        "rational_x": rational_x,
        "x_bounds": (lower - 1, upper - 1),
        "root_radicand": radicand,
        "root_bounds": (lower, upper),
        "bounds_are_equilibria": False,
    }


def class_certificate(network: str, price_class: str, cap: Fraction) -> dict:
    """Proved global value over the entire covered-row nonnegative price box."""
    rows = covered_rows(network, price_class)
    if not isinstance(cap, (int, Fraction)) or cap not in CAPS:
        raise ValueError("Only exact caps 0, 1/4 and 1/2 are approved.")
    cap = Fraction(cap)
    visible = 1 in rows
    theta = cap if visible else Fraction(0)
    response = equilibrium_response(network, theta)
    if network == NETWORKS[0]:
        value = 3 - Fraction(3, 2) * theta + theta**2
        cost_lower = cost_upper = value
        exact_cost = str(value)
    elif theta == 0:
        cost_lower = cost_upper = Fraction(3)
        exact_cost = "3"
    else:
        root_lower, root_upper = response["root_bounds"]
        cost_lower = 3 - theta * root_upper
        cost_upper = 3 - theta * root_lower
        exact_cost = (
            str(cost_lower)
            if cost_lower == cost_upper
            else f"3 - ({theta})*sqrt({response['root_radicand']})"
        )
    return {
        "network": network,
        "price_class": price_class,
        "cap": cap,
        "covered_rows": rows,
        "node_visible": visible,
        "optimal_price_vector": (Fraction(0), theta),
        "effective_node_price": theta,
        "optimal_corridor_price_set": "[0, cap]" if 0 in rows else "{0}",
        "equilibrium": response,
        "exact_cost": exact_cost,
        "cost_bounds": (cost_lower, cost_upper),
        "so_cost": Fraction(5, 2),
        "due_cost": Fraction(3),
        "minimum_so_gap": Fraction(1, 2) - theta,
        "gcr_bounds": (2 * (3 - cost_upper), 2 * (3 - cost_lower)),
        "remaining_loss_bounds": (cost_lower - Fraction(5, 2), cost_upper - Fraction(5, 2)),
        "equilibrium_unique": True,
        "selection_gap": Fraction(0),
        "globality": "analytic_monotonicity_over_complete_price_box",
    }


def all_certificates() -> list[dict]:
    """The frozen batch: exactly two networks by three classes by three caps."""
    return [
        class_certificate(network, cls, cap)
        for network, cls, cap in product(NETWORKS, PRICE_CLASSES, CAPS)
    ]


def render_artifacts() -> dict[str, str]:
    """Render deterministic UTF-8 file contents without writing any filesystem path."""
    certificates = all_certificates()
    networks = []
    for name, r, offset in ((NETWORKS[0], 1, 0), (NETWORKS[1], 2, 2)):
        networks.append(
            {
                "name": name,
                "r": r,
                "paths": {
                    "p1": ["O", "A", "B", "D"],
                    "p2": ["O", "A", "bypass", "D"],
                    "p0": ["S", "B", "D"],
                },
                "path_order": ["p1", "p2", "p0"],
                "H": [[1, 1, 0], [0, 0, 1]],
                "Delta": [[1, 1, 0], [1, 0, 1]],
                "K": [2, 3],
                "demands": [1, 1],
                "fixed_costs": ["0", "3/2", "0"],
                "resource_row_order": ["corridor_A", "node_B"],
                "departure_offsets": [0, 0, 0],
                "resource_offsets": [0, offset],
                "node_delay": f"y^{r}",
                "corridor_delay": 0,
                "reliability": 0,
                "background_flow": "1 (single path, independent of payment)",
                "feasible_flow": "(x, 1-x, 1), 0 <= x <= 1",
            }
        )
    payload = {
        "schema_version": 1,
        "batch": "bounded_independent_C1_20261017",
        "method": "analytical proof and exact Fraction/integer-square-root arithmetic",
        "root_bits": ROOT_BITS,
        "solver_calls": {"lp": 0, "outer_optimization": 0, "equilibrium": 0},
        "networks": networks,
        "price_sets": {
            "complete_covered_row_box": True,
            "units": "generalized_cost",
            "covered_price_bounds": "0 <= theta_i <= cap, independently for every covered row",
            "uncovered_prices": "zero",
            "caps": CAPS,
            "covered_rows": {
                name: {cls: covered_rows(name, cls) for cls in PRICE_CLASSES} for name in NETWORKS
            },
            "common_corridor_charge": "arbitrary in its allowed interval; cancels in target difference",
            "not_a_truncated_marginal_signal": True,
        },
        "equilibrium_bounds_semantics": "Rational enclosure endpoints are not asserted to be equilibrium flows; exact_x defines the feasible symbolic equilibrium witness. Coincidence is possible for rational equilibria. Background flow is exactly one.",
        "scope": "Prescribed analytical networks only; not a global certificate for the 200-instance outer optimization or the protected 40-flight case (capacity feasibility value 38/40).",
        "domains": certificates,
    }
    json_text = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            default=lambda value: {"numerator": value.numerator, "denominator": value.denominator},
        )
        + "\n"
    )
    stream = io.StringIO(newline="")
    columns = [
        "network",
        "price_class",
        "cap",
        "covered_rows",
        "node_visible",
        "corridor_price",
        "node_price",
        "exact_x",
        "x_lower",
        "x_upper",
        "bounds_are_equilibria",
        "exact_cost",
        "cost_lower",
        "cost_upper",
        "bound_width",
        "so_cost",
        "due_cost",
        "minimum_so_gap",
        "gcr_lower",
        "gcr_upper",
        "remaining_loss_lower",
        "remaining_loss_upper",
        "equilibrium_unique",
        "selection_gap",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for cert in certificates:
        response = cert["equilibrium"]
        writer.writerow(
            {
                "network": cert["network"],
                "price_class": cert["price_class"],
                "cap": cert["cap"],
                "covered_rows": ";".join((str(i) for i in cert["covered_rows"])),
                "node_visible": str(cert["node_visible"]).lower(),
                "corridor_price": "0",
                "node_price": cert["effective_node_price"],
                "exact_x": response["exact_x"],
                "x_lower": response["x_bounds"][0],
                "x_upper": response["x_bounds"][1],
                "bounds_are_equilibria": "false",
                "exact_cost": cert["exact_cost"],
                "cost_lower": cert["cost_bounds"][0],
                "cost_upper": cert["cost_bounds"][1],
                "bound_width": cert["cost_bounds"][1] - cert["cost_bounds"][0],
                "so_cost": cert["so_cost"],
                "due_cost": cert["due_cost"],
                "minimum_so_gap": cert["minimum_so_gap"],
                "gcr_lower": cert["gcr_bounds"][0],
                "gcr_upper": cert["gcr_bounds"][1],
                "remaining_loss_lower": cert["remaining_loss_bounds"][0],
                "remaining_loss_upper": cert["remaining_loss_bounds"][1],
                "equilibrium_unique": "true",
                "selection_gap": "0",
            }
        )
    return {"bounds.json": json_text, "bounds.csv": stream.getvalue()}
