from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import Bounds, LinearConstraint, minimize, linprog

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/component_prices"
BOTTLENECK_TYPES = ["corridor", "node", "mixed", "delayed-node"]
RELIABILITY_LEVELS = ["loose", "medium", "tight"]


@dataclass(frozen=True)
class StudyConfig:
    n_od: int
    paths_per_od: int
    n_dep: int
    bottleneck_type: str
    pressure: float
    reliability: str
    window_width: int
    seed: int
    reliability_threshold: float | None = None


@dataclass
class SyntheticInstance:
    config: StudyConfig
    n_od: int
    paths_per_od: int
    n_dep: int
    n_res: int
    n_resources: int
    n_alt: int
    resource_types: np.ndarray
    delta: np.ndarray
    capacity: np.ndarray
    alpha: np.ndarray
    demand: np.ndarray
    od_of_alt: np.ndarray
    path_of_alt: np.ndarray
    dep_of_alt: np.ndarray
    planned_time: np.ndarray
    schedule_penalty: np.ndarray
    deadline: np.ndarray
    rho_late: float
    base_load: np.ndarray
    peak_pressure: float
    delayed_intensity: float
    reliability_threshold: float | None = None


def _softmax_profile(periods: np.ndarray, center: float, width: float) -> np.ndarray:
    raw = np.exp(-0.5 * ((periods - center) / max(width, 1e-06)) ** 2)
    return raw / raw.sum()


def generate_instance(cfg: StudyConfig) -> SyntheticInstance:
    if cfg.bottleneck_type not in BOTTLENECK_TYPES:
        raise ValueError(f"Unknown bottleneck type: {cfg.bottleneck_type}")
    if cfg.reliability not in RELIABILITY_LEVELS:
        raise ValueError(f"Unknown reliability level: {cfg.reliability}")
    if cfg.window_width > cfg.n_dep:
        raise ValueError("window_width cannot exceed n_dep")
    rng = np.random.default_rng(cfg.seed)
    max_lag = 3 if cfg.bottleneck_type == "delayed-node" else 2
    n_res = cfg.n_dep + max_lag
    n_corridors = max(6, cfg.paths_per_od * 2 + cfg.n_od)
    n_nodes = max(4, cfg.n_od + 2)
    n_resources = n_corridors + n_nodes
    resource_types = np.array(["corridor"] * n_corridors + ["node"] * n_nodes)
    demand = rng.uniform(4.0, 8.0, size=cfg.n_od)
    demand *= 1.0 + 0.1 * (cfg.pressure - 0.75)
    alts: list[tuple[int, int, int]] = []
    od_pref_center = rng.integers(
        max(2, cfg.n_dep // 3), max(3, 2 * cfg.n_dep // 3 + 1), size=cfg.n_od
    )
    for w in range(cfg.n_od):
        half = cfg.window_width // 2
        start = int(np.clip(od_pref_center[w] - half, 0, cfg.n_dep - cfg.window_width))
        window = range(start, start + cfg.window_width)
        for p in range(cfg.paths_per_od):
            for t in window:
                alts.append((w, p, t))
    n_alt = len(alts)
    od_of_alt = np.array([a[0] for a in alts], dtype=int)
    path_of_alt = np.array([a[1] for a in alts], dtype=int)
    dep_of_alt = np.array([a[2] for a in alts], dtype=int)
    delta = np.zeros((n_resources, n_res, n_alt), dtype=float)
    planned_time = np.zeros(n_alt, dtype=float)
    schedule_penalty = np.zeros(n_alt, dtype=float)
    deadline = np.zeros(n_alt, dtype=float)
    od_path_specs: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for w in range(cfg.n_od):
        for p in range(cfg.paths_per_od):
            c1 = int((2 * p + w) % n_corridors)
            c2 = int((2 * p + w + 1) % n_corridors)
            node = int(n_corridors + (p + 2 * w) % n_nodes)
            lag_node = 2
            if cfg.bottleneck_type == "delayed-node" and p % 2 == 0:
                lag_node = 3
            elif p == cfg.paths_per_od - 1:
                lag_node = 1
            od_path_specs[w, p] = (c1, c2, node, lag_node)
    rel_slack = {"loose": 5.5, "medium": 3.5, "tight": 2.2}[cfg.reliability]
    for j, (w, p, t) in enumerate(alts):
        c1, c2, node, lag_node = od_path_specs[w, p]
        delta[c1, t, j] = 1.0
        if t + 1 < n_res:
            delta[c2, t + 1, j] = 1.0
        if t + lag_node < n_res:
            delta[node, t + lag_node, j] = 1.0
        planned_time[j] = 2.0 + 0.45 * p + rng.uniform(0.0, 0.25)
        center = od_pref_center[w]
        schedule_penalty[j] = 0.35 * abs(t - center)
        deadline[j] = center + planned_time[j] + rel_slack
    preferred = np.zeros(n_alt)
    for w in range(cfg.n_od):
        mask = od_of_alt == w
        periods = dep_of_alt[mask]
        profile = _softmax_profile(periods, od_pref_center[w], max(cfg.window_width / 4, 1.0))
        path_weight = np.exp(-0.55 * path_of_alt[mask])
        weights = profile * path_weight
        weights = weights / weights.sum()
        preferred[mask] = demand[w] * weights
    base_load = np.tensordot(delta, preferred, axes=([2], [0]))
    capacity = np.maximum(1.5, base_load / max(cfg.pressure, 1e-06))
    corridor_mask = resource_types == "corridor"
    node_mask = resource_types == "node"
    if cfg.bottleneck_type == "corridor":
        capacity[corridor_mask, :] *= 0.78
        capacity[node_mask, :] *= 1.35
    elif cfg.bottleneck_type == "node":
        capacity[corridor_mask, :] *= 1.35
        capacity[node_mask, :] *= 0.78
    elif cfg.bottleneck_type == "mixed":
        capacity[corridor_mask, :] *= 0.92
        capacity[node_mask, :] *= 0.92
    elif cfg.bottleneck_type == "delayed-node":
        capacity[corridor_mask, :] *= 1.2
        capacity[node_mask, :] *= 0.74
        capacity[node_mask, :max_lag] *= 1.45
    capacity = np.maximum(capacity, 1.0)
    alpha = np.where(corridor_mask, 0.65, 0.9)
    if cfg.bottleneck_type == "corridor":
        alpha[corridor_mask] *= 1.45
    if cfg.bottleneck_type in {"node", "delayed-node"}:
        alpha[node_mask] *= 1.55
    delayed_occ = delta[:, cfg.n_dep :, :].sum()
    total_occ = delta.sum()
    delayed_intensity = float(delayed_occ / max(total_occ, 1e-12))
    peak_pressure = float(np.max(base_load / capacity))
    return SyntheticInstance(
        config=cfg,
        n_od=cfg.n_od,
        paths_per_od=cfg.paths_per_od,
        n_dep=cfg.n_dep,
        n_res=n_res,
        n_resources=n_resources,
        n_alt=n_alt,
        resource_types=resource_types,
        delta=delta,
        capacity=capacity,
        alpha=alpha,
        demand=demand,
        od_of_alt=od_of_alt,
        path_of_alt=path_of_alt,
        dep_of_alt=dep_of_alt,
        planned_time=planned_time,
        schedule_penalty=schedule_penalty,
        deadline=deadline,
        rho_late=1.8,
        base_load=base_load,
        peak_pressure=peak_pressure,
        delayed_intensity=delayed_intensity,
        reliability_threshold=cfg.reliability_threshold,
    )


def build_metropolitan_corridor_instance(
    seed: int = 20260625,
) -> tuple[SyntheticInstance, dict[str, object]]:
    """Construct a fixed geometry-driven low-altitude corridor instance."""
    rng = np.random.default_rng(seed)
    zone_coords = np.array(
        [
            [-4.2, 1.8],
            [-4.4, -0.4],
            [-3.2, -2.8],
            [-1.0, 4.0],
            [1.6, 3.4],
            [4.0, 1.4],
            [4.2, -1.2],
            [1.8, -3.4],
            [-1.1, -4.1],
            [-3.5, -2.0],
            [-0.3, 2.1],
            [0.5, -2.2],
        ],
        dtype=float,
    )
    merge_coords = np.array([[-1.0, 0.9], [1.0, 0.9], [-0.9, -0.9], [1.1, -0.9]], dtype=float)
    od_pairs = [
        (0, 5, "morning"),
        (1, 6, "morning"),
        (2, 4, "morning"),
        (3, 7, "morning"),
        (9, 5, "morning"),
        (5, 0, "evening"),
        (6, 1, "evening"),
        (4, 2, "evening"),
        (7, 3, "evening"),
        (8, 10, "evening"),
    ]
    merge_options = [
        [0, 1, 2],
        [2, 3, 1],
        [2, 0, 3],
        [0, 1, 3],
        [2, 1, 0],
        [1, 0, 3],
        [3, 2, 0],
        [0, 2, 1],
        [1, 3, 0],
        [2, 3, 1],
    ]
    n_od = len(od_pairs)
    paths_per_od = 3
    n_dep = 16
    window_width = 6
    max_lag = 5
    n_res = n_dep + max_lag
    edge_to_idx: dict[tuple[str, int, str, int], int] = {}

    def corridor_idx(edge: tuple[str, int, str, int]) -> int:
        if edge not in edge_to_idx:
            edge_to_idx[edge] = len(edge_to_idx)
        return edge_to_idx[edge]

    for w, (origin, dest, _) in enumerate(od_pairs):
        for merge in merge_options[w]:
            corridor_idx(("z", origin, "m", merge))
            corridor_idx(("m", merge, "z", dest))
    n_corridors = len(edge_to_idx)
    n_service_nodes = len(zone_coords)
    n_merge_nodes = len(merge_coords)
    n_nodes = n_service_nodes + n_merge_nodes
    n_resources = n_corridors + n_nodes
    resource_types = np.array(["corridor"] * n_corridors + ["node"] * n_nodes)
    merge_resource_ids = [n_corridors + n_service_nodes + k for k in range(n_merge_nodes)]
    service_resource_ids = [n_corridors + k for k in range(n_service_nodes)]
    alts: list[tuple[int, int, int]] = []
    od_centers = np.zeros(n_od, dtype=int)
    for w, (_, _, peak) in enumerate(od_pairs):
        center = 5 if peak == "morning" else 11
        center += int(rng.integers(-1, 2))
        od_centers[w] = int(np.clip(center, 2, n_dep - 3))
        start = int(np.clip(od_centers[w] - window_width // 2, 0, n_dep - window_width))
        for p in range(paths_per_od):
            for t in range(start, start + window_width):
                alts.append((w, p, t))
    n_alt = len(alts)
    od_of_alt = np.array([a[0] for a in alts], dtype=int)
    path_of_alt = np.array([a[1] for a in alts], dtype=int)
    dep_of_alt = np.array([a[2] for a in alts], dtype=int)
    delta = np.zeros((n_resources, n_res, n_alt), dtype=float)
    planned_time = np.zeros(n_alt, dtype=float)
    schedule_penalty = np.zeros(n_alt, dtype=float)
    deadline = np.zeros(n_alt, dtype=float)
    path_edges: list[tuple[int, int, int, int, int, int]] = []
    for j, (w, p, t) in enumerate(alts):
        origin, dest, peak = od_pairs[w]
        merge = merge_options[w][p]
        c_in = edge_to_idx["z", origin, "m", merge]
        c_out = edge_to_idx["m", merge, "z", dest]
        merge_res = n_corridors + n_service_nodes + merge
        service_res = n_corridors + dest
        in_dist = float(np.linalg.norm(zone_coords[origin] - merge_coords[merge]))
        out_dist = float(np.linalg.norm(merge_coords[merge] - zone_coords[dest]))
        merge_lag = 2 + (1 if p == 0 and peak == "morning" else 0)
        service_lag = merge_lag + 2
        delta[c_in, t, j] = 1.0
        delta[merge_res, min(t + merge_lag, n_res - 1), j] = 1.0
        delta[c_out, min(t + merge_lag + 1, n_res - 1), j] = 1.0
        delta[service_res, min(t + service_lag, n_res - 1), j] = 0.85
        planned_time[j] = 1.55 + 0.18 * (in_dist + out_dist) + 0.28 * p
        schedule_penalty[j] = 0.42 * abs(t - od_centers[w])
        deadline[j] = od_centers[w] + planned_time[j] + (2.4 if peak == "morning" else 3.0)
        path_edges.append((origin, merge, dest, c_in, c_out, merge_res))
    base_demand = np.array([8.4, 7.8, 6.9, 7.1, 8.0, 7.7, 7.3, 6.8, 7.0, 6.4])
    demand = base_demand * (1.0 + rng.normal(0.0, 0.035, size=n_od))
    demand = np.maximum(demand, 5.8)
    preferred = np.zeros(n_alt)
    for w in range(n_od):
        mask = od_of_alt == w
        periods = dep_of_alt[mask]
        profile = _softmax_profile(periods, od_centers[w], 1.35)
        path_weight = np.array([1.0, 0.82, 0.68])[path_of_alt[mask]]
        weights = profile * path_weight
        weights = weights / weights.sum()
        preferred[mask] = demand[w] * weights
    base_load = np.tensordot(delta, preferred, axes=([2], [0]))
    capacity_floor = np.zeros((n_resources, n_res))
    for edge, idx in edge_to_idx.items():
        _, a, _, b = edge
        if edge[0] == "z":
            dist = float(np.linalg.norm(zone_coords[a] - merge_coords[b]))
        else:
            dist = float(np.linalg.norm(merge_coords[a] - zone_coords[b]))
        capacity_floor[idx, :] = max(2.4, 4.4 - 0.18 * dist)
    for z, idx in enumerate(service_resource_ids):
        capacity_floor[idx, :] = 2.15 + 0.15 * (z % 3)
    for k, idx in enumerate(merge_resource_ids):
        capacity_floor[idx, :] = 2.45 - 0.1 * (k % 2)
    capacity = np.maximum(capacity_floor, base_load / 0.9)
    capacity[np.array(merge_resource_ids), :] *= 0.82
    capacity[np.array(service_resource_ids), :] *= 0.96
    capacity[:n_corridors, :] *= 1.05
    capacity = np.maximum(capacity, 1.25)
    alpha = np.full(n_resources, 0.58)
    alpha[np.array(service_resource_ids)] = 0.88
    alpha[np.array(merge_resource_ids)] = 1.15
    delayed_occ = delta[:, n_dep:, :].sum()
    total_occ = delta.sum()
    delayed_intensity = float(delayed_occ / max(total_occ, 1e-12))
    peak_pressure = float(np.max(base_load / capacity))
    cfg = StudyConfig(
        n_od=n_od,
        paths_per_od=paths_per_od,
        n_dep=n_dep,
        bottleneck_type="metropolitan-merge",
        pressure=0.9,
        reliability="tight",
        window_width=window_width,
        seed=seed,
    )
    inst = SyntheticInstance(
        config=cfg,
        n_od=n_od,
        paths_per_od=paths_per_od,
        n_dep=n_dep,
        n_res=n_res,
        n_resources=n_resources,
        n_alt=n_alt,
        resource_types=resource_types,
        delta=delta,
        capacity=capacity,
        alpha=alpha,
        demand=demand,
        od_of_alt=od_of_alt,
        path_of_alt=path_of_alt,
        dep_of_alt=dep_of_alt,
        planned_time=planned_time,
        schedule_penalty=schedule_penalty,
        deadline=deadline,
        rho_late=2.0,
        base_load=base_load,
        peak_pressure=peak_pressure,
        delayed_intensity=delayed_intensity,
    )
    meta = {
        "zone_coords": zone_coords,
        "merge_coords": merge_coords,
        "od_pairs": od_pairs,
        "merge_options": merge_options,
        "edge_to_idx": edge_to_idx,
        "path_edges": path_edges,
        "n_corridors": n_corridors,
        "n_service_nodes": n_service_nodes,
        "n_merge_nodes": n_merge_nodes,
    }
    return (inst, meta)


def _flat_load(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    return np.tensordot(inst.delta, h, axes=([2], [0]))


def _delay_by_resource(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    y = _flat_load(inst, h)
    return inst.alpha[:, None] * (y / inst.capacity) ** 2


def _path_delay(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    return np.tensordot(inst.delta, _delay_by_resource(inst, h), axes=([0, 1], [0, 1]))


def _reliability_threshold(inst: SyntheticInstance) -> float:
    if inst.reliability_threshold is not None:
        return float(inst.reliability_threshold)
    return {"loose": 0.88, "medium": 0.74, "tight": 0.6}[inst.config.reliability]


def _reliability_by_resource(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    y = _flat_load(inst, h)
    utilization = y / inst.capacity
    excess = np.maximum(utilization - _reliability_threshold(inst), 0.0)
    return inst.rho_late * excess**2


def _reliability(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    return np.tensordot(inst.delta, _reliability_by_resource(inst, h), axes=([0, 1], [0, 1]))


def private_cost(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    return inst.planned_time + inst.schedule_penalty + _path_delay(inst, h) + _reliability(inst, h)


def total_cost(inst: SyntheticInstance, h: np.ndarray) -> float:
    return float(np.dot(h, private_cost(inst, h)))


def due_potential(inst: SyntheticInstance, h: np.ndarray, toll: np.ndarray | None = None) -> float:
    y = _flat_load(inst, h)
    base = float(np.dot(h, inst.planned_time + inst.schedule_penalty))
    delay_potential = float(np.sum(inst.alpha[:, None] * y**3 / (3.0 * inst.capacity**2)))
    rel_threshold = _reliability_threshold(inst)
    rel_excess = np.maximum(y / inst.capacity - rel_threshold, 0.0)
    reliability = float(np.sum(inst.rho_late * inst.capacity * rel_excess**3 / 3.0))
    toll_part = 0.0 if toll is None else float(np.dot(h, toll))
    return base + delay_potential + reliability + toll_part


def _external_reliability_component(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    signal = _reliability_externality_signal(inst, h)
    return np.tensordot(inst.delta, signal, axes=([0, 1], [0, 1]))


def _reliability_externality_signal(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    y = _flat_load(inst, h)
    utilization = y / inst.capacity
    excess = np.maximum(utilization - _reliability_threshold(inst), 0.0)
    return 2.0 * inst.rho_late * y * excess / inst.capacity


def _delay_externality_component(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    y = _flat_load(inst, h)
    signal = 2.0 * inst.alpha[:, None] * y**2 / inst.capacity**2
    return np.tensordot(inst.delta, signal, axes=([0, 1], [0, 1]))


def total_cost_gradient(inst: SyntheticInstance, h: np.ndarray) -> np.ndarray:
    return (
        private_cost(inst, h)
        + _delay_externality_component(inst, h)
        + _external_reliability_component(inst, h)
    )


def due_potential_gradient(
    inst: SyntheticInstance, h: np.ndarray, toll: np.ndarray | None = None
) -> np.ndarray:
    grad = private_cost(inst, h)
    if toll is not None:
        grad = grad + toll
    return grad


def _constraints(
    inst: SyntheticInstance,
) -> tuple[LinearConstraint, LinearConstraint, Bounds, np.ndarray, np.ndarray]:
    a_eq = np.zeros((inst.n_od, inst.n_alt))
    for j, w in enumerate(inst.od_of_alt):
        a_eq[w, j] = 1.0
    eq = LinearConstraint(a_eq, inst.demand, inst.demand)
    a_ub = inst.delta.reshape((inst.n_resources * inst.n_res, inst.n_alt))
    b_ub = inst.capacity.reshape(inst.n_resources * inst.n_res)
    cap = LinearConstraint(a_ub, -np.inf * np.ones_like(b_ub), b_ub)
    bounds = Bounds(np.zeros(inst.n_alt), np.full(inst.n_alt, np.inf))
    return (eq, cap, bounds, a_ub, b_ub)


def _initial_flow(inst: SyntheticInstance) -> np.ndarray:
    h = np.zeros(inst.n_alt)
    for w in range(inst.n_od):
        idx = np.where(inst.od_of_alt == w)[0]
        h[idx] = inst.demand[w] / len(idx)
    return h


def _minimize_flow(
    inst: SyntheticInstance,
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    maxiter: int,
) -> tuple[np.ndarray, bool, str]:
    eq, cap, bounds, _, _ = _constraints(inst)
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=[eq, cap],
        options={"ftol": 1e-08, "maxiter": maxiter, "disp": False},
    )
    return (np.maximum(res.x, 0.0), bool(res.success), str(res.message))


def solve_core_equilibria(
    inst: SyntheticInstance, maxiter: int = 220
) -> tuple[np.ndarray, np.ndarray]:
    x0 = _initial_flow(inst)
    due_h, _, _ = _minimize_flow(
        inst,
        lambda x: due_potential(inst, x),
        lambda x: due_potential_gradient(inst, x),
        x0,
        maxiter,
    )
    so_h, _, _ = _minimize_flow(
        inst, lambda x: total_cost(inst, x), lambda x: total_cost_gradient(inst, x), due_h, maxiter
    )
    return (due_h, so_h)


def _resource_tolls(inst: SyntheticInstance, h: np.ndarray) -> dict[str, np.ndarray]:
    y = _flat_load(inst, h)
    delay_signal = 2.0 * inst.alpha[:, None] * y**2 / inst.capacity**2
    reliability_signal = _reliability_externality_signal(inst, h)
    delay = np.tensordot(inst.delta, delay_signal, axes=([0, 1], [0, 1]))
    reliability = np.tensordot(inst.delta, reliability_signal, axes=([0, 1], [0, 1]))
    delay_plus_reliability = delay + reliability
    corridor_signal = delay_signal.copy()
    corridor_signal[inst.resource_types != "corridor", :] = 0.0
    node_signal = delay_signal.copy()
    node_signal[inst.resource_types != "node", :] = 0.0
    corridor = np.tensordot(inst.delta, corridor_signal, axes=([0, 1], [0, 1]))
    node = np.tensordot(inst.delta, node_signal, axes=([0, 1], [0, 1]))
    contemp = np.zeros(inst.n_alt)
    delayed = np.zeros(inst.n_alt)
    for j, t in enumerate(inst.dep_of_alt):
        contemp[j] = float(np.sum(inst.delta[:, t, j] * delay_signal[:, t]))
        if t + 1 < inst.n_res:
            delayed[j] = float(np.sum(inst.delta[:, t + 1 :, j] * delay_signal[:, t + 1 :]))
    return {
        "resource": delay,
        "reliability": reliability,
        "delay_plus_reliability": delay_plus_reliability,
        "corridor": corridor,
        "node": node,
        "contemporaneous": contemp,
        "delayed": delayed,
    }


def _local_exchange_basis(
    inst: SyntheticInstance, h: np.ndarray, active_tol: float = 1e-05, flow_tol: float = 1e-07
) -> np.ndarray:
    eq, _, _, a_ub, b_ub = _constraints(inst)
    active = b_ub - a_ub @ h <= active_tol * np.maximum(1.0, np.abs(b_ub))
    rows = [eq.A]
    if np.any(active):
        rows.append(a_ub[active])
    zero_flow = h <= flow_tol
    if np.any(zero_flow):
        rows.append(np.eye(inst.n_alt)[zero_flow])
    tangent_constraints = np.vstack(rows)
    return null_space(tangent_constraints)


def _price_design_matrix(inst: SyntheticInstance, price_class: str) -> np.ndarray:
    flat_delta = inst.delta.reshape((inst.n_resources * inst.n_res, inst.n_alt)).T
    if price_class == "all_resource":
        return flat_delta
    if price_class == "corridor":
        mask = np.repeat(inst.resource_types == "corridor", inst.n_res)
        return flat_delta[:, mask]
    if price_class == "node":
        mask = np.repeat(inst.resource_types == "node", inst.n_res)
        return flat_delta[:, mask]
    if price_class == "contemporaneous":
        design = np.zeros_like(flat_delta)
        for j, t in enumerate(inst.dep_of_alt):
            for r in range(inst.n_resources):
                design[j, r * inst.n_res + t] = inst.delta[r, t, j]
        keep = np.any(np.abs(design) > 1e-12, axis=0)
        return design[:, keep]
    raise ValueError(f"Unknown price class: {price_class}")


def _projection_residual(
    inst: SyntheticInstance,
    h_so: np.ndarray,
    marginal_toll: np.ndarray,
    price_class: str,
    normalized: bool = True,
) -> float:
    basis = _local_exchange_basis(inst, h_so)
    if basis.size == 0 or basis.shape[1] == 0:
        return 0.0
    target = basis.T @ marginal_toll
    design = basis.T @ _price_design_matrix(inst, price_class)
    if design.size == 0:
        residual = float(np.linalg.norm(target))
    else:
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = float(np.linalg.norm(target - design @ coef))
    if normalized:
        return residual / max(float(np.linalg.norm(target)), 1e-12)
    return residual


def _projection_residuals(
    inst: SyntheticInstance, h_so: np.ndarray, marginal_toll: np.ndarray, normalized: bool = True
) -> dict[str, float]:
    return {
        "corridor": _projection_residual(inst, h_so, marginal_toll, "corridor", normalized),
        "node": _projection_residual(inst, h_so, marginal_toll, "node", normalized),
        "contemporaneous": _projection_residual(
            inst, h_so, marginal_toll, "contemporaneous", normalized
        ),
        "all_resource": _projection_residual(inst, h_so, marginal_toll, "all_resource", normalized),
    }


def _solve_priced_due(
    inst: SyntheticInstance, toll: np.ndarray, x0: np.ndarray, maxiter: int
) -> tuple[np.ndarray, bool]:
    h, ok, _ = _minimize_flow(
        inst,
        lambda x, tau=toll: due_potential(inst, x, tau),
        lambda x, tau=toll: due_potential_gradient(inst, x, tau),
        x0,
        maxiter,
    )
    return (h, ok)


def _iterative_estimated_toll(
    inst: SyntheticInstance,
    initial_h: np.ndarray,
    toll_key: str,
    maxiter: int,
    update_steps: int = 8,
    damping: float = 0.65,
    tolerance: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, bool, int, float]:
    toll = np.zeros(inst.n_alt)
    h = initial_h.copy()
    best_h = h.copy()
    best_toll = toll.copy()
    best_cost = total_cost(inst, best_h)
    ok_all = True
    relative_change = 0.0
    steps_done = 0
    base_estimate = _resource_tolls(inst, initial_h)[toll_key]
    positive_base = base_estimate[base_estimate > 0.0]
    upper = (
        max(float(np.quantile(positive_base, 0.9)) * 1.2, 1e-09) if positive_base.size else 1e-09
    )
    base_steps = np.array([0.9, 0.65, 0.45, 0.3, 0.2, 0.12, 0.07, 0.035])
    for step in range(1, update_steps + 1):
        estimated = _resource_tolls(inst, best_h)[toll_key]
        accepted: tuple[np.ndarray, np.ndarray, float] | None = None
        trial_best = best_cost
        for base_step in base_steps:
            step_size = min(damping, float(base_step / np.sqrt(step)))
            candidate_toll = (1.0 - step_size) * best_toll + step_size * estimated
            candidate_toll = np.clip(candidate_toll, 0.0, upper)
            candidate_h, ok = _solve_priced_due(inst, candidate_toll, best_h, maxiter)
            ok_all = ok_all and ok
            candidate_cost = total_cost(inst, candidate_h)
            if candidate_cost < trial_best - 1e-07:
                trial_best = candidate_cost
                accepted = (candidate_h, candidate_toll, candidate_cost)
        if accepted is None:
            relative_change = 0.0
            break
        candidate_h, candidate_toll, candidate_cost = accepted
        relative_change = float(
            np.linalg.norm(candidate_toll - best_toll) / (1.0 + np.linalg.norm(best_toll))
        )
        best_h, best_toll, best_cost = (candidate_h, candidate_toll, candidate_cost)
        steps_done = step
        if relative_change <= tolerance:
            break
    return (best_h, best_toll, ok_all, steps_done, relative_change)


def _vi_gap(inst: SyntheticInstance, h: np.ndarray, cost: np.ndarray) -> float:
    eq, _, _, a_ub, b_ub = _constraints(inst)
    res = linprog(
        c=cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=eq.A,
        b_eq=inst.demand,
        bounds=[(0, None)] * inst.n_alt,
        method="highs-ds",
    )
    if not res.success:
        return np.nan
    return max(0.0, float(np.dot(cost, h) - res.fun))


def _departure_redistribution(inst: SyntheticInstance, h: np.ndarray) -> float:
    observed = np.zeros(inst.n_dep)
    preferred = np.zeros(inst.n_dep)
    h0 = _initial_flow(inst)
    for j, t in enumerate(inst.dep_of_alt):
        observed[t] += h[j]
        preferred[t] += h0[j]
    total = max(float(np.sum(h)), 1e-12)
    return float(0.5 * np.sum(np.abs(observed - preferred)) / total)


def _route_shift(inst: SyntheticInstance, h_a: np.ndarray, h_b: np.ndarray) -> float:
    route_a = np.zeros((inst.n_od, inst.paths_per_od))
    route_b = np.zeros((inst.n_od, inst.paths_per_od))
    for j, (w, p) in enumerate(zip(inst.od_of_alt, inst.path_of_alt)):
        route_a[w, p] += h_a[j]
        route_b[w, p] += h_b[j]
    total = max(float(np.sum(h_a)), float(np.sum(h_b)), 1e-12)
    return float(0.5 * np.sum(np.abs(route_a - route_b)) / total)


def _departure_shift(inst: SyntheticInstance, h_a: np.ndarray, h_b: np.ndarray) -> float:
    dep_a = np.zeros((inst.n_od, inst.n_dep))
    dep_b = np.zeros((inst.n_od, inst.n_dep))
    for j, (w, t) in enumerate(zip(inst.od_of_alt, inst.dep_of_alt)):
        dep_a[w, t] += h_a[j]
        dep_b[w, t] += h_b[j]
    total = max(float(np.sum(h_a)), float(np.sum(h_b)), 1e-12)
    return float(0.5 * np.sum(np.abs(dep_a - dep_b)) / total)


def _resource_period_shift(inst: SyntheticInstance, h_a: np.ndarray, h_b: np.ndarray) -> float:
    y_a = _flat_load(inst, h_a)
    y_b = _flat_load(inst, h_b)
    total = max(float(np.sum(y_a)), float(np.sum(y_b)), 1e-12)
    return float(0.5 * np.sum(np.abs(y_a - y_b)) / total)


def _binding_frequency(
    inst: SyntheticInstance, h: np.ndarray, resource_type: str | None = None, tol: float = 1e-05
) -> float:
    load = _flat_load(inst, h)
    binding = load >= (1.0 - tol) * inst.capacity
    if resource_type is not None:
        mask = inst.resource_types == resource_type
        binding = binding[mask, :]
    return float(np.mean(binding)) if binding.size else 0.0


def _capacity_violation(inst: SyntheticInstance, h: np.ndarray) -> float:
    return float(np.max(np.maximum(_flat_load(inst, h) - inst.capacity, 0.0)))


def solve_instance(
    inst: SyntheticInstance, maxiter: int = 220
) -> dict[str, float | int | str | bool]:
    x0 = _initial_flow(inst)
    due_h, due_ok, due_msg = _minimize_flow(
        inst,
        lambda x: due_potential(inst, x),
        lambda x: due_potential_gradient(inst, x),
        x0,
        maxiter,
    )
    so_h, so_ok, so_msg = _minimize_flow(
        inst, lambda x: total_cost(inst, x), lambda x: total_cost_gradient(inst, x), due_h, maxiter
    )
    tolls = _resource_tolls(inst, so_h)
    full_mc_toll = total_cost_gradient(inst, so_h) - private_cost(inst, so_h)
    corridor_h, corridor_ok = _solve_priced_due(inst, tolls["corridor"], due_h, maxiter)
    node_h, node_ok = _solve_priced_due(inst, tolls["node"], due_h, maxiter)
    resource_h, resource_ok = _solve_priced_due(inst, tolls["resource"], due_h, maxiter)
    delay_reliability_h, delay_reliability_ok = _solve_priced_due(
        inst, tolls["delay_plus_reliability"], due_h, maxiter
    )
    contemp_h, contemp_ok = _solve_priced_due(inst, tolls["contemporaneous"], due_h, maxiter)
    full_mc_h, full_mc_ok = _solve_priced_due(inst, full_mc_toll, due_h, maxiter)
    due_estimated_tolls = _resource_tolls(inst, due_h)
    due_est_resource_h, due_est_resource_ok = _solve_priced_due(
        inst, due_estimated_tolls["resource"], due_h, maxiter
    )
    due_est_delay_reliability_h, due_est_delay_reliability_ok = _solve_priced_due(
        inst, due_estimated_tolls["delay_plus_reliability"], due_h, maxiter
    )
    (
        iter_resource_h,
        iter_resource_toll,
        iter_resource_ok,
        iter_resource_steps,
        iter_resource_change,
    ) = _iterative_estimated_toll(inst, due_h, "resource", maxiter)
    (
        iter_delay_reliability_h,
        iter_delay_reliability_toll,
        iter_delay_reliability_ok,
        iter_delay_reliability_steps,
        iter_delay_reliability_change,
    ) = _iterative_estimated_toll(inst, due_h, "delay_plus_reliability", maxiter)
    due_phi = total_cost(inst, due_h)
    so_phi = total_cost(inst, so_h)
    corridor_phi = total_cost(inst, corridor_h)
    node_phi = total_cost(inst, node_h)
    resource_phi = total_cost(inst, resource_h)
    delay_reliability_phi = total_cost(inst, delay_reliability_h)
    contemp_phi = total_cost(inst, contemp_h)
    full_mc_phi = total_cost(inst, full_mc_h)
    due_est_resource_phi = total_cost(inst, due_est_resource_h)
    due_est_delay_reliability_phi = total_cost(inst, due_est_delay_reliability_h)
    iter_resource_phi = total_cost(inst, iter_resource_h)
    iter_delay_reliability_phi = total_cost(inst, iter_delay_reliability_h)
    denom = max(due_phi - so_phi, 1e-09)
    full_support_gap = _vi_gap(inst, so_h, private_cost(inst, so_h) + full_mc_toll)
    projection = _projection_residuals(inst, so_h, full_mc_toll, normalized=True)
    resource_toll_payment = float(np.dot(resource_h, tolls["resource"]))
    delayed_toll_share = (
        float(np.dot(resource_h, tolls["delayed"])) / resource_toll_payment
        if abs(resource_toll_payment) > 1e-10
        else 0.0
    )
    so_delay_toll_mass = max(float(np.dot(so_h, tolls["resource"])), 1e-10)
    corridor_delay_share = float(np.dot(so_h, tolls["corridor"]) / so_delay_toll_mass)
    node_delay_share = float(np.dot(so_h, tolls["node"]) / so_delay_toll_mass)
    support_resource = _vi_gap(inst, so_h, private_cost(inst, so_h) + tolls["resource"])
    support_delay_reliability = _vi_gap(
        inst, so_h, private_cost(inst, so_h) + tolls["delay_plus_reliability"]
    )
    support_contemp = _vi_gap(inst, so_h, private_cost(inst, so_h) + tolls["contemporaneous"])
    support_due_est_resource = _vi_gap(
        inst, so_h, private_cost(inst, so_h) + due_estimated_tolls["resource"]
    )
    support_due_est_delay_reliability = _vi_gap(
        inst, so_h, private_cost(inst, so_h) + due_estimated_tolls["delay_plus_reliability"]
    )
    support_iter_resource = _vi_gap(inst, so_h, private_cost(inst, so_h) + iter_resource_toll)
    support_iter_delay_reliability = _vi_gap(
        inst, so_h, private_cost(inst, so_h) + iter_delay_reliability_toll
    )
    due_residual = _vi_gap(inst, due_h, private_cost(inst, due_h))
    corridor_residual = _vi_gap(
        inst, corridor_h, private_cost(inst, corridor_h) + tolls["corridor"]
    )
    node_residual = _vi_gap(inst, node_h, private_cost(inst, node_h) + tolls["node"])
    resource_residual = _vi_gap(
        inst, resource_h, private_cost(inst, resource_h) + tolls["resource"]
    )
    delay_reliability_residual = _vi_gap(
        inst,
        delay_reliability_h,
        private_cost(inst, delay_reliability_h) + tolls["delay_plus_reliability"],
    )
    contemp_residual = _vi_gap(
        inst, contemp_h, private_cost(inst, contemp_h) + tolls["contemporaneous"]
    )
    full_mc_residual = _vi_gap(inst, full_mc_h, private_cost(inst, full_mc_h) + full_mc_toll)
    due_est_resource_residual = _vi_gap(
        inst,
        due_est_resource_h,
        private_cost(inst, due_est_resource_h) + due_estimated_tolls["resource"],
    )
    due_est_delay_reliability_residual = _vi_gap(
        inst,
        due_est_delay_reliability_h,
        private_cost(inst, due_est_delay_reliability_h)
        + due_estimated_tolls["delay_plus_reliability"],
    )
    iter_resource_residual = _vi_gap(
        inst, iter_resource_h, private_cost(inst, iter_resource_h) + iter_resource_toll
    )
    iter_delay_reliability_residual = _vi_gap(
        inst,
        iter_delay_reliability_h,
        private_cost(inst, iter_delay_reliability_h) + iter_delay_reliability_toll,
    )
    return {
        "success": bool(
            due_ok
            and so_ok
            and corridor_ok
            and node_ok
            and resource_ok
            and delay_reliability_ok
            and contemp_ok
            and full_mc_ok
            and due_est_resource_ok
            and due_est_delay_reliability_ok
            and iter_resource_ok
            and iter_delay_reliability_ok
        ),
        "DUE message": due_msg,
        "SO message": so_msg,
        "OD count": inst.config.n_od,
        "Paths per OD": inst.config.paths_per_od,
        "Departure periods": inst.config.n_dep,
        "Window width": inst.config.window_width,
        "Bottleneck type": inst.config.bottleneck_type,
        "Pressure target": inst.config.pressure,
        "Realized peak pressure": inst.peak_pressure,
        "Reliability": inst.config.reliability,
        "Seed": inst.config.seed,
        "Alternatives": inst.n_alt,
        "Resources": inst.n_resources,
        "DUE cost": due_phi,
        "SO cost": so_phi,
        "Corridor-only cost": corridor_phi,
        "Node-only cost": node_phi,
        "Resource-externality cost": resource_phi,
        "Delay-plus-reliability cost": delay_reliability_phi,
        "Contemporaneous-only cost": contemp_phi,
        "Full marginal-cost cost": full_mc_phi,
        "DUE-estimated resource cost": due_est_resource_phi,
        "DUE-estimated delay+reliability cost": due_est_delay_reliability_phi,
        "Iterative resource cost": iter_resource_phi,
        "Iterative delay+reliability cost": iter_delay_reliability_phi,
        "PoA": due_phi / max(so_phi, 1e-09),
        "Corridor-only recovery": (due_phi - corridor_phi) / denom,
        "Node-only recovery": (due_phi - node_phi) / denom,
        "Resource-externality recovery": (due_phi - resource_phi) / denom,
        "Delay-plus-reliability recovery": (due_phi - delay_reliability_phi) / denom,
        "Contemporaneous-only recovery": (due_phi - contemp_phi) / denom,
        "Full marginal-cost recovery": (due_phi - full_mc_phi) / denom,
        "DUE-estimated resource recovery": (due_phi - due_est_resource_phi) / denom,
        "DUE-estimated delay+reliability recovery": (due_phi - due_est_delay_reliability_phi)
        / denom,
        "Adaptive resource recovery": (due_phi - iter_resource_phi) / denom,
        "Adaptive delay+reliability recovery": (due_phi - iter_delay_reliability_phi) / denom,
        "Iterative resource recovery": (due_phi - iter_resource_phi) / denom,
        "Iterative delay+reliability recovery": (due_phi - iter_delay_reliability_phi) / denom,
        "Adaptive resource iterations": iter_resource_steps,
        "Adaptive delay+reliability iterations": iter_delay_reliability_steps,
        "Adaptive resource toll change": iter_resource_change,
        "Adaptive delay+reliability toll change": iter_delay_reliability_change,
        "Iterative resource iterations": iter_resource_steps,
        "Iterative delay+reliability iterations": iter_delay_reliability_steps,
        "Iterative resource toll change": iter_resource_change,
        "Iterative delay+reliability toll change": iter_delay_reliability_change,
        "Delayed toll share": delayed_toll_share,
        "Corridor externality share": corridor_delay_share,
        "Node externality share": node_delay_share,
        "Corridor-only omitted share": node_delay_share,
        "Node-only omitted share": corridor_delay_share,
        "Corridor-only recovery loss": 1.0 - (due_phi - corridor_phi) / denom,
        "Node-only recovery loss": 1.0 - (due_phi - node_phi) / denom,
        "Corridor projection residual": projection["corridor"],
        "Node projection residual": projection["node"],
        "Contemporaneous projection residual": projection["contemporaneous"],
        "All-resource projection residual": projection["all_resource"],
        "Maximum restricted projection residual": max(
            projection["corridor"], projection["node"], projection["contemporaneous"]
        ),
        "Delayed occupation intensity": inst.delayed_intensity,
        "DUE departure redistribution": _departure_redistribution(inst, due_h),
        "SO departure redistribution": _departure_redistribution(inst, so_h),
        "Priced departure redistribution": _departure_redistribution(inst, resource_h),
        "DUE-estimated resource departure redistribution": _departure_redistribution(
            inst, due_est_resource_h
        ),
        "Iterative resource departure redistribution": _departure_redistribution(
            inst, iter_resource_h
        ),
        "DUE-SO route shift": _route_shift(inst, due_h, so_h),
        "DUE-SO departure shift": _departure_shift(inst, due_h, so_h),
        "DUE-SO resource-period shift": _resource_period_shift(inst, due_h, so_h),
        "DUE binding frequency": _binding_frequency(inst, due_h),
        "SO binding frequency": _binding_frequency(inst, so_h),
        "Priced binding frequency": _binding_frequency(inst, resource_h),
        "DUE corridor binding frequency": _binding_frequency(inst, due_h, "corridor"),
        "SO corridor binding frequency": _binding_frequency(inst, so_h, "corridor"),
        "Priced corridor binding frequency": _binding_frequency(inst, resource_h, "corridor"),
        "DUE node binding frequency": _binding_frequency(inst, due_h, "node"),
        "SO node binding frequency": _binding_frequency(inst, so_h, "node"),
        "Priced node binding frequency": _binding_frequency(inst, resource_h, "node"),
        "Resource SO support gap": support_resource,
        "Delay-plus-reliability SO support gap": support_delay_reliability,
        "Contemporaneous SO support gap": support_contemp,
        "Full MC SO support gap": full_support_gap,
        "DUE-estimated resource SO support gap": support_due_est_resource,
        "DUE-estimated delay+reliability SO support gap": support_due_est_delay_reliability,
        "Iterative resource SO support gap": support_iter_resource,
        "Iterative delay+reliability SO support gap": support_iter_delay_reliability,
        "Adaptive resource SO support gap": support_iter_resource,
        "Adaptive delay+reliability SO support gap": support_iter_delay_reliability,
        "DUE VI residual": due_residual,
        "Corridor-only VI residual": corridor_residual,
        "Node-only VI residual": node_residual,
        "Resource-externality VI residual": resource_residual,
        "Delay-plus-reliability VI residual": delay_reliability_residual,
        "Contemporaneous-only VI residual": contemp_residual,
        "Full MC VI residual": full_mc_residual,
        "DUE-estimated resource VI residual": due_est_resource_residual,
        "DUE-estimated delay+reliability VI residual": due_est_delay_reliability_residual,
        "Iterative resource VI residual": iter_resource_residual,
        "Iterative delay+reliability VI residual": iter_delay_reliability_residual,
        "Adaptive resource VI residual": iter_resource_residual,
        "Adaptive delay+reliability VI residual": iter_delay_reliability_residual,
        "Maximum capacity violation": max(
            _capacity_violation(inst, due_h),
            _capacity_violation(inst, so_h),
            _capacity_violation(inst, corridor_h),
            _capacity_violation(inst, node_h),
            _capacity_violation(inst, resource_h),
            _capacity_violation(inst, delay_reliability_h),
            _capacity_violation(inst, contemp_h),
            _capacity_violation(inst, full_mc_h),
            _capacity_violation(inst, due_est_resource_h),
            _capacity_violation(inst, due_est_delay_reliability_h),
            _capacity_violation(inst, iter_resource_h),
            _capacity_violation(inst, iter_delay_reliability_h),
        ),
    }


def _config_stream(
    target_instances: int, seed: int, max_alternatives: int | None = None
) -> list[StudyConfig]:
    rng = np.random.default_rng(seed)
    configs: list[StudyConfig] = []
    od_choices = [3, 5, 10]
    path_choices = [3, 5]
    dep_choices = [12, 24]
    pressure_choices = [0.6, 0.75, 0.85, 0.95]
    window_choices = [4, 6, 8]
    grid: list[tuple[int, int, int, str, float, str, int]] = []
    for n_od in od_choices:
        for paths in path_choices:
            for n_dep in dep_choices:
                for bottleneck in BOTTLENECK_TYPES:
                    for pressure in pressure_choices:
                        for reliability in RELIABILITY_LEVELS:
                            for window in window_choices:
                                alternatives = n_od * paths * window
                                if window <= n_dep and (
                                    max_alternatives is None or alternatives <= max_alternatives
                                ):
                                    grid.append(
                                        (
                                            n_od,
                                            paths,
                                            n_dep,
                                            bottleneck,
                                            pressure,
                                            reliability,
                                            window,
                                        )
                                    )
    if not grid:
        raise ValueError("No configurations satisfy the requested max_alternatives bound.")
    order = rng.permutation(len(grid))
    for i in range(target_instances):
        item = grid[int(order[i % len(grid)])]
        configs.append(
            StudyConfig(
                n_od=item[0],
                paths_per_od=item[1],
                n_dep=item[2],
                bottleneck_type=item[3],
                pressure=item[4],
                reliability=item[5],
                window_width=item[6],
                seed=seed * 100000 + i,
            )
        )
    return configs


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Bottleneck type", "Pressure target", "Reliability"]
    metrics = [
        "PoA",
        "Resource-externality recovery",
        "Delay-plus-reliability recovery",
        "Contemporaneous-only recovery",
        "Full marginal-cost recovery",
        "Corridor-only recovery",
        "Node-only recovery",
        "Delayed toll share",
        "Corridor externality share",
        "Node externality share",
        "Corridor-only omitted share",
        "Node-only omitted share",
        "Corridor-only recovery loss",
        "Node-only recovery loss",
        "SO departure redistribution",
        "DUE-SO route shift",
        "DUE-SO departure shift",
        "DUE-SO resource-period shift",
        "DUE binding frequency",
        "SO binding frequency",
        "Priced binding frequency",
        "Resource SO support gap",
        "Delay-plus-reliability SO support gap",
        "Contemporaneous SO support gap",
    ]
    summary = results.groupby(group_cols, as_index=False)[metrics].agg(["mean", "median", "std"])
    summary.columns = [
        " ".join([c for c in col if c]).strip() for col in summary.columns.to_flat_index()
    ]
    return summary


def make_figures(results: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 300,
        }
    )
    agg = results.groupby(["Bottleneck type", "Pressure target"], as_index=False)[
        "Resource-externality recovery"
    ].mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    for b in BOTTLENECK_TYPES:
        part = agg[agg["Bottleneck type"] == b]
        ax.plot(part["Pressure target"], part["Resource-externality recovery"], marker="o", label=b)
    ax.set_xlabel("Demand-capacity pressure")
    ax.set_ylabel("Mean pricing recovery")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(fig_dir / "controlled_recovery_by_pressure.png", bbox_inches="tight")
    plt.close(fig)
    temp = results[
        ["Delayed toll share", "Contemporaneous SO support gap", "Contemporaneous-only recovery"]
    ].copy()
    temp = temp.dropna()
    temp["Contemporaneous unrecovered share"] = 1.0 - temp["Contemporaneous-only recovery"]
    temp["Delayed toll share bin"] = pd.qcut(temp["Delayed toll share"], q=5, duplicates="drop")
    bin_stats = (
        temp.groupby("Delayed toll share bin", observed=True)
        .agg(
            x=("Delayed toll share", "mean"),
            gap_mean=("Contemporaneous SO support gap", "mean"),
            gap_q25=("Contemporaneous SO support gap", lambda s: s.quantile(0.25)),
            gap_q75=("Contemporaneous SO support gap", lambda s: s.quantile(0.75)),
            unrecovered_mean=("Contemporaneous unrecovered share", "mean"),
            unrecovered_q25=("Contemporaneous unrecovered share", lambda s: s.quantile(0.25)),
            unrecovered_q75=("Contemporaneous unrecovered share", lambda s: s.quantile(0.75)),
        )
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    axes[0].errorbar(
        bin_stats["x"],
        bin_stats["gap_mean"],
        yerr=[
            bin_stats["gap_mean"] - bin_stats["gap_q25"],
            bin_stats["gap_q75"] - bin_stats["gap_mean"],
        ],
        marker="o",
        capsize=3,
        color="#2F5D8C",
    )
    axes[0].set_xlabel("Delayed toll share")
    axes[0].set_ylabel("Contemporaneous-only SO support gap")
    axes[0].text(0.02, 0.95, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[1].errorbar(
        bin_stats["x"],
        bin_stats["unrecovered_mean"],
        yerr=[
            bin_stats["unrecovered_mean"] - bin_stats["unrecovered_q25"],
            bin_stats["unrecovered_q75"] - bin_stats["unrecovered_mean"],
        ],
        marker="s",
        capsize=3,
        color="#8A4B2A",
    )
    axes[1].set_xlabel("Delayed toll share")
    axes[1].set_ylabel("Unrecovered efficiency loss")
    axes[1].text(0.02, 0.95, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")
    fig.savefig(fig_dir / "controlled_temporal_displacement_gap.png", bbox_inches="tight")
    plt.close(fig)
    omitted = pd.concat(
        [
            pd.DataFrame(
                {
                    "Omitted resource externality share": results["Corridor-only omitted share"],
                    "Recovery loss": results["Corridor-only recovery loss"],
                    "Restriction": "corridor-only",
                }
            ),
            pd.DataFrame(
                {
                    "Omitted resource externality share": results["Node-only omitted share"],
                    "Recovery loss": results["Node-only recovery loss"],
                    "Restriction": "node-only",
                }
            ),
        ],
        ignore_index=True,
    ).dropna()
    omitted["Share bin"] = pd.qcut(
        omitted["Omitted resource externality share"], q=5, duplicates="drop"
    )
    omitted_stats = (
        omitted.groupby(["Restriction", "Share bin"], observed=True)
        .agg(
            x=("Omitted resource externality share", "mean"),
            loss_mean=("Recovery loss", "mean"),
            loss_q25=("Recovery loss", lambda s: s.quantile(0.25)),
            loss_q75=("Recovery loss", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    colors = {"corridor-only": "#2F5D8C", "node-only": "#8A4B2A"}
    markers = {"corridor-only": "o", "node-only": "s"}
    for restriction, part in omitted_stats.groupby("Restriction", sort=False):
        part = part.sort_values("x")
        ax.errorbar(
            part["x"],
            part["loss_mean"],
            yerr=[part["loss_mean"] - part["loss_q25"], part["loss_q75"] - part["loss_mean"]],
            marker=markers[restriction],
            capsize=3,
            color=colors[restriction],
            label=restriction,
        )
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Omitted-resource externality share")
    ax.set_ylabel("Recovery loss")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "controlled_omitted_resource_loss.png", bbox_inches="tight")
    plt.close(fig)
    agg2 = results.groupby("Bottleneck type", as_index=False)[
        [
            "Corridor-only recovery",
            "Node-only recovery",
            "Resource-externality recovery",
            "Delay-plus-reliability recovery",
            "Full marginal-cost recovery",
        ]
    ].mean()
    x = np.arange(len(agg2))
    width = 0.16
    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    ax.bar(x - 2.0 * width, agg2["Corridor-only recovery"], width, label="Corridor-only")
    ax.bar(x - 1.0 * width, agg2["Node-only recovery"], width, label="Node-only")
    ax.bar(x, agg2["Resource-externality recovery"], width, label="Delay")
    ax.bar(x + 1.0 * width, agg2["Delay-plus-reliability recovery"], width, label="Delay+rel.")
    ax.bar(x + 2.0 * width, agg2["Full marginal-cost recovery"], width, label="MC bench.")
    ax.set_xticks(x, agg2["Bottleneck type"], rotation=20, ha="right")
    ax.set_ylabel("Mean recovery")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(fig_dir / "controlled_pricing_mechanism_by_bottleneck.png", bbox_inches="tight")
    plt.close(fig)
    agg3 = results.groupby("Window width", as_index=False)[
        ["SO departure redistribution", "Priced departure redistribution"]
    ].mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    ax.plot(agg3["Window width"], agg3["SO departure redistribution"], marker="o", label="SO")
    ax.plot(
        agg3["Window width"],
        agg3["Priced departure redistribution"],
        marker="o",
        label="priced DUE",
    )
    ax.set_xlabel("Departure-window width")
    ax.set_ylabel("Departure-time redistribution")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "controlled_departure_window_redistribution.png", bbox_inches="tight")
    plt.close(fig)


def _plot_metropolitan_network(meta: dict[str, object], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    zone_coords = np.asarray(meta["zone_coords"])
    merge_coords = np.asarray(meta["merge_coords"])
    od_pairs = list(meta["od_pairs"])
    merge_options = list(meta["merge_options"])
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.scatter(
        zone_coords[:, 0], zone_coords[:, 1], s=48, color="#2F5D8C", label="service zone", zorder=3
    )
    ax.scatter(
        merge_coords[:, 0],
        merge_coords[:, 1],
        s=70,
        marker="s",
        color="#8A4B2A",
        label="merge/crossing",
        zorder=4,
    )
    for i, (x, y) in enumerate(zone_coords):
        ax.text(x + 0.08, y + 0.08, f"V{i + 1}", fontsize=8)
    for i, (x, y) in enumerate(merge_coords):
        ax.text(x + 0.08, y + 0.08, f"M{i + 1}", fontsize=8, fontweight="bold")
    drawn: set[tuple[int, int, str]] = set()
    for w, (origin, dest, peak) in enumerate(od_pairs):
        for merge in merge_options[w]:
            key1 = (origin, merge, "in")
            key2 = (dest, merge, "out")
            if key1 not in drawn:
                drawn.add(key1)
                ax.plot(
                    [zone_coords[origin, 0], merge_coords[merge, 0]],
                    [zone_coords[origin, 1], merge_coords[merge, 1]],
                    color="#7E9FBC",
                    linewidth=0.9,
                    alpha=0.62,
                )
            if key2 not in drawn:
                drawn.add(key2)
                ax.plot(
                    [merge_coords[merge, 0], zone_coords[dest, 0]],
                    [merge_coords[merge, 1], zone_coords[dest, 1]],
                    color="#B98C6A",
                    linewidth=0.9,
                    alpha=0.62,
                )
    for origin, dest, peak in od_pairs:
        color = "#1F5A3D" if peak == "morning" else "#7A3C6A"
        ax.annotate(
            "",
            xy=zone_coords[dest],
            xytext=zone_coords[origin],
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "alpha": 0.42},
        )
    ax.set_xlabel("East-west coordinate")
    ax.set_ylabel("North-south coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(fig_dir / "metropolitan_low_altitude_network.png", bbox_inches="tight")
    plt.close(fig)


def run_metropolitan_case(
    out_dir: Path = OUT, seed: int = 20260625, maxiter: int = 260
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    inst, meta = build_metropolitan_corridor_instance(seed=seed)
    row = solve_instance(inst, maxiter=maxiter)
    row.update(
        {
            "Service zones": 12,
            "Merge/crossing nodes": 4,
            "Corridor resources": int(meta["n_corridors"]),
            "Service-node resources": int(meta["n_service_nodes"]),
            "Morning OD pairs": sum((1 for _, _, peak in meta["od_pairs"] if peak == "morning")),
            "Evening OD pairs": sum((1 for _, _, peak in meta["od_pairs"] if peak == "evening")),
        }
    )
    result = pd.DataFrame([row])
    result.to_csv(out_dir / "metropolitan_structured_case.csv", index=False)
    _plot_metropolitan_network(meta, out_dir)
    return result


def run_batch(
    out_dir: Path = OUT,
    target_instances: int = 240,
    seed: int = 20260625,
    maxiter: int = 220,
    max_alternatives: int | None = None,
    progress_every: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str | bool]] = []
    for i, cfg in enumerate(_config_stream(target_instances, seed, max_alternatives), start=1):
        inst = generate_instance(cfg)
        try:
            row = solve_instance(inst, maxiter=maxiter)
        except Exception as exc:
            row = {
                "success": False,
                "OD count": cfg.n_od,
                "Paths per OD": cfg.paths_per_od,
                "Departure periods": cfg.n_dep,
                "Window width": cfg.window_width,
                "Bottleneck type": cfg.bottleneck_type,
                "Pressure target": cfg.pressure,
                "Reliability": cfg.reliability,
                "Seed": cfg.seed,
                "error": str(exc),
            }
        rows.append(row)
        if progress_every > 0 and (i % progress_every == 0 or i == target_instances):
            ok_so_far = sum((1 for item in rows if item.get("success") is True))
            print({"processed": i, "successful": ok_so_far, "target": target_instances}, flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "controlled_synthetic_instances.csv", index=False)
    ok = results[results["success"] == True].copy()
    summary = summarize_results(ok) if len(ok) else pd.DataFrame()
    summary.to_csv(out_dir / "controlled_synthetic_summary.csv", index=False)
    if len(ok):
        make_figures(ok, out_dir)
    return (results, summary)
