"""Numerical studies of restricted resource-period pricing."""

from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_inputs(name):
    return json.loads((DATA / "inputs" / (name + ".json")).read_text(encoding="utf-8"))


def accepted_rows(frame):
    return frame[frame["success"].astype(str).str.lower() == "true"].copy()


def saved_summary():
    rows = []
    for name, path in (
        ("constructed", "component_prices/controlled_synthetic_instances.csv"),
        ("random", "random_networks/random_geometric_validation_instances.csv"),
    ):
        frame = accepted_rows(pd.read_csv(DATA / path))
        rows.append(
            dict(
                sample=name,
                instances=len(frame),
                mean_poa=frame["PoA"].mean(),
                corridor_gcr=frame["Corridor-only recovery"].mean(),
                node_gcr=frame["Node-only recovery"].mean(),
                all_resource_gcr=frame["Resource-externality recovery"].mean(),
                contemporaneous_gcr=frame["Contemporaneous-only recovery"].mean(),
            )
        )
    return pd.DataFrame(rows)


def exact_examples():
    from resource_pricing.resource_models import boundary_class_value

    omitted = boundary_class_value("omitted_node")
    complete = boundary_class_value("complete", node_cap=Fraction(1, 2))

    def loss(value):
        return str(value["system_cost"] - value["so_cost"])

    return dict(
        omitted_node_loss=loss(omitted),
        complete_node_loss=loss(complete),
        omitted_node=omitted,
        complete_node=complete,
    )


def generated_instances(count):
    from resource_pricing.controlled_networks import StudyConfig, generate_instance

    cases = load_inputs("constructed")
    for config in cases[:count]:
        yield generate_instance(StudyConfig(**config))


def write_json(path, value):
    def convert(v):
        if isinstance(v, Fraction):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    path.write_text(json.dumps(value, default=convert, indent=2), encoding="utf-8")


def class_gaps(count, output, recompute_so=False):
    from resource_pricing.convex_programs import (
        from_instance,
        minimum_class_gap,
        solve_conic,
        polish_solution,
        refine_active_face,
    )
    from resource_pricing.price_class_gap import build_gap_price_parameterization

    rows = []
    for inst in generated_instances(count):
        problem = from_instance(inst)
        if recompute_so:
            conic = solve_conic(problem, 1e-10)
            warm = polish_solution(problem, conic["flow"])
            flow = refine_active_face(problem, warm["flow"])["flow"]
        else:
            path = DATA / "class_gaps/constructed" / f"flows_{inst.config.seed}.npz"
            with np.load(path, allow_pickle=False) as arrays:
                flow = arrays["flow_2"]
        for name in ("corridor", "node", "contemporaneous", "all_resource"):
            design = build_gap_price_parameterization(inst, name).design
            result = minimum_class_gap(problem, flow, design)
            result.pop("theta", None)
            rows.append(dict(seed=inst.config.seed, price_class=name, **result))
    pd.DataFrame(rows).to_csv(output / "class_gaps.csv", index=False)


def nominal_comparisons(output, count=None):
    from resource_pricing.nominal_reservations import (
        build_operational_case,
        solve_reservation,
        scale_problem,
        integer_projection,
    )
    from resource_pricing.coverage import resource_coverage_gap

    settings = load_inputs("reservation_grids")
    rows = []
    for grid in settings["resource_minutes"][:count]:
        case = build_operational_case(grid_minutes=grid)
        due = solve_reservation(case.problem)
        optimum = solve_reservation(case.problem, system=True)
        write_json(output / f"grid_{grid}_due.json", due)
        write_json(output / f"grid_{grid}_so.json", optimum)
        loads = case.problem.occupation @ optimum["flow"] / case.problem.capacity
        marginal = 2 * case.problem.alpha * loads**2
        resource_kind = case.resource_kind[case.row_resource]
        departures = 6 * case.choices[:, 2]
        contemporaneous = (case.row_start[:, None] >= departures) & (
            case.row_start[:, None] < departures + 6
        )
        for name in ("corridor", "node", "contemporaneous"):
            gap = resource_coverage_gap(case.problem, optimum["flow"], case.price_map(name))
            if name == "contemporaneous":
                price = (case.problem.occupation * contemporaneous).T @ marginal
            else:
                chosen = resource_kind == name
                price = case.problem.occupation[chosen].T @ marginal[chosen]
            priced = solve_reservation(case.problem, price=price)
            gcr = (due["objective"] - priced["objective"]) / (
                due["objective"] - optimum["objective"]
            )
            rows.append(
                dict(
                    resource_minutes=grid,
                    price_class=name,
                    component_gcr=gcr,
                    **{k: v for k, v in gap.items() if np.isscalar(v)},
                )
            )
    pd.DataFrame(rows).to_csv(output / "nominal_class_gaps.csv", index=False)
    baseline = build_operational_case(grid_minutes=3).problem
    continuous = solve_reservation(baseline, system=True)
    integer_rows = []
    for scale in settings["capacity_multipliers"][:count]:
        problem = scale_problem(baseline, scale)
        flow = scale * continuous["flow"]
        projected = integer_projection(problem, flow)
        lower = scale * (continuous["objective"] - continuous["gap_upper"])
        row = dict(scale=scale, success=projected["success"], continuous_lower=lower)
        if projected["success"]:
            row.update(
                integer_cost_upper=projected["objective_upper"],
                relative_cost_bound=(projected["objective_upper"] - lower) / max(1, abs(lower)),
            )
        integer_rows.append(row)
    pd.DataFrame(integer_rows).to_csv(output / "integer_cost_bounds.csv", index=False)


def analytical_families(output, count=None):
    from resource_pricing.family_prices import make_family
    from resource_pricing.analytical_family import system_optimum, total_cost, global_node_optimum

    cases = json.loads((DATA / "inputs/exact_classes/family_cases.json").read_text())
    rows = []
    for case in cases[:count]:
        family = make_family(case["family"])
        so = system_optimum(family)
        so_cost = total_cost(family, so)
        for cap in case["caps"]:
            interval = tuple(map(Fraction, cap["net_price_interval"]))
            result = global_node_optimum(family, *interval)
            rows.append(
                dict(
                    case=case["id"],
                    normalized_cap=cap["normalized_cap"],
                    rate_cap=cap["rate_cap"],
                    so_cost=str(so_cost),
                    node_cost=str(result["cost"]),
                    cost_loss=str(result["cost"] - so_cost),
                    net_price=str(result["p"]),
                )
            )
    pd.DataFrame(rows).to_csv(output / "exact_family_values.csv", index=False)


def pressure_comparisons(output, count=None):
    from resource_pricing.controlled_networks import StudyConfig, generate_instance, solve_instance

    settings = load_inputs("pressure")
    configs = load_inputs("adaptive")
    rows = []
    for identity in settings["instance_ids"][:count]:
        base = generate_instance(StudyConfig(**configs[identity - 1]))
        for pressure in settings["pressure_values"]:
            instance = replace(
                base,
                capacity=base.capacity * settings["reference_pressure"] / pressure,
                config=replace(base.config, pressure=pressure),
            )
            rows.append(dict(network_id=identity, **solve_instance(instance, maxiter=220)))
    pd.DataFrame(rows).to_csv(output / "fixed_network_pressure.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        choices=[
            "summarize",
            "examples",
            "constructed",
            "random",
            "class-gaps",
            "restricted-prices",
            "adaptive",
            "reliability",
            "pressure",
            "scalability",
            "selection",
            "reservations",
            "families",
            "statistics",
        ],
    )
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--recompute-so", action="store_true")
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--outer-iterations", type=int, default=24)
    args = parser.parse_args()
    output = (args.out or ROOT / "results" / args.study).resolve()
    if output == DATA.resolve() or DATA.resolve() in output.parents:
        parser.error("Choose an output directory outside data/.")
    if args.instances is not None and args.instances < 1:
        parser.error("--instances must be positive.")
    output.mkdir(parents=True, exist_ok=True)
    if args.study == "summarize":
        frame = saved_summary()
        frame.to_csv(output / "pricing_summary.csv", index=False)
        print(frame.to_string(index=False))
    elif args.study == "examples":
        result = exact_examples()
        write_json(output / "exact_examples.json", result)
        print(json.dumps({k: result[k] for k in ("omitted_node_loss", "complete_node_loss")}))
    elif args.study == "constructed":
        from resource_pricing.controlled_networks import solve_instance

        rows = [
            solve_instance(inst, maxiter=args.maxiter or 220)
            for inst in generated_instances(args.instances or 200)
        ]
        pd.DataFrame(rows).to_csv(output / "constructed_instances.csv", index=False)
    elif args.study == "random":
        from resource_pricing.random_networks import (
            build_random_geometric_instance,
            solve_instance,
            _posthoc_class,
            _posthoc_subclass,
        )

        rows = []
        for config in load_inputs("random")[: args.instances or 300]:
            inst = build_random_geometric_instance(seed=config["seed"], pressure=config["pressure"])
            row = solve_instance(inst, maxiter=args.maxiter or 150)
            if row.get("success"):
                row["Post-hoc bottleneck class"] = _posthoc_class(pd.Series(row))
                row["Post-hoc bottleneck subclass"] = _posthoc_subclass(pd.Series(row))
            rows.append(row)
        pd.DataFrame(rows).to_csv(output / "random_instances.csv", index=False)
    elif args.study == "class-gaps":
        class_gaps(args.instances or 200, output, args.recompute_so)
    elif args.study == "restricted-prices":
        from resource_pricing.restricted_pricing import solve_best_in_class_instance

        rows = []
        prices = []
        allocations = []
        for inst in generated_instances(args.instances or 200):
            result = solve_best_in_class_instance(
                inst,
                max_outer_iterations=args.outer_iterations,
                max_inner_iterations=args.maxiter or 240,
                multistart=2,
            )
            row, resource_prices, flows = result
            rows.append(row)
            prices.extend(resource_prices)
            allocations.extend(flows)
        pd.DataFrame(rows).to_csv(output / "restricted_prices.csv", index=False)
        pd.DataFrame(prices).to_csv(output / "resource_prices.csv", index=False)
        pd.DataFrame(allocations).to_csv(output / "allocations.csv", index=False)
    elif args.study == "adaptive":
        from resource_pricing.adaptive_pricing import run_adaptive_pricing_trajectories

        run_adaptive_pricing_trajectories(
            out_dir=output, seed=20260626, maxiter=args.maxiter or 160, update_steps=8
        )
    elif args.study == "reliability":
        from resource_pricing.reliability_sensitivity import evaluate_network, summarize

        source = pd.read_csv(DATA / "component_prices/controlled_synthetic_instances.csv")
        rows = []
        for i, row in source.iloc[: args.instances or 200].iterrows():
            rows.extend(
                evaluate_network(
                    int(i),
                    row.to_dict(),
                    load_inputs("reliability")["thresholds"],
                    args.maxiter or 220,
                )
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "reliability_instances.csv", index=False)
        summarize(frame).to_csv(output / "reliability_summary.csv", index=False)
    elif args.study == "scalability":
        from resource_pricing.scalability import run_active_capacity_scalability_multiscale_panel

        settings = load_inputs("scalability")
        run_active_capacity_scalability_multiscale_panel(
            out_dir=output,
            scales=tuple(settings["scales"]),
            resource_targets=dict(zip(settings["scales"], settings["resources"])),
            instances_per_scale=args.instances or 5,
            seed=settings["seed"],
            maxiter=args.maxiter or settings["maxiter"],
        )
    elif args.study == "selection":
        from resource_pricing.equilibrium_selection import run_selection_robust_study

        run_selection_robust_study(out_dir=output, instances=args.instances or 12, grid_size=4001)
    elif args.study == "reservations":
        nominal_comparisons(output, args.instances)
    elif args.study == "families":
        analytical_families(output, args.instances)
    elif args.study == "pressure":
        pressure_comparisons(output, args.instances)
    elif args.study == "statistics":
        from resource_pricing.statistical_analysis import run_audit

        run_audit(
            source=DATA / "random_networks/random_geometric_validation_instances.csv",
            out_dir=output,
            draws=2000,
        )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
