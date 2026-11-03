from __future__ import annotations
import csv
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/reservation_inputs"
STAGES = ROOT / "data/protected_reservations/protected_stage_intervals.csv"
ARRAY_SHA256 = "e59e2a425b66a11708d4be426a7a6fcb6b11f8c03ce4ef0900c286d9c984301e"
STAGE_SHA256 = "55d242dacbdc7f3e85f080ea0f752a2d77d7e8e9f2e9ae5e466e5f65b79aa529"


@dataclass(frozen=True)
class ProtectedCost:
    occupation: np.ndarray
    capacity: np.ndarray
    alpha: np.ndarray
    conservation: np.ndarray
    demand: np.ndarray
    choices: np.ndarray
    row_resource: np.ndarray
    row_start: np.ndarray
    destination: np.ndarray
    destination_start: np.ndarray
    destination_nominal: np.ndarray
    departure_penalty: np.ndarray
    rho: float
    threshold: float
    resource_grid: float = 3.0
    departure_period: float = 6.0

    def __post_init__(self):
        array_names = {f.name for f in fields(self)} - {
            "rho",
            "threshold",
            "resource_grid",
            "departure_period",
        }
        for name in array_names:
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Nonfinite array: {name}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        D = self.occupation
        if D.ndim != 2 or min(D.shape) < 1:
            raise ValueError("Occupation must be a nonempty matrix")
        m, n = D.shape
        for name in ("capacity", "alpha", "row_resource", "row_start"):
            if getattr(self, name).shape != (m,):
                raise ValueError(f"Invalid resource dimension: {name}")
        for name in ("destination_start", "destination_nominal", "departure_penalty"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"Invalid choice dimension: {name}")
        if (
            self.destination.shape != D.shape
            or self.conservation.ndim != 2
            or self.conservation.shape[1] != n
            or (self.demand.shape != (self.conservation.shape[0],))
            or (self.choices.shape != (n, 3))
        ):
            raise ValueError("Inconsistent matrix dimensions")
        for name in ("occupation", "destination", "conservation"):
            a = getattr(self, name)
            if not np.all((a == 0) | (a == 1)):
                raise ValueError(f"Expected binary matrix: {name}")
        if (
            np.any(self.destination > D)
            or np.any(self.destination.sum(axis=0) == 0)
            or (not np.all(self.conservation.sum(axis=0) == 1))
            or np.any(self.capacity <= 0)
            or np.any(self.alpha < 0)
            or np.any(self.demand < 0)
            or np.any(self.destination_nominal <= 0)
            or np.any(self.departure_penalty < -1e-10)
        ):
            raise ValueError("Invalid incidence, capacity or cost data")
        for name in ("choices", "row_resource"):
            a = getattr(self, name)
            if np.any(a < 0) or not np.all(a == np.rint(a)):
                raise ValueError(f"Expected nonnegative integers: {name}")
        if not np.array_equal(self.choices[:, 0], self.conservation.argmax(axis=0)):
            raise ValueError("Choice ownership differs from conservation")
        scalars = [self.rho, self.threshold, self.resource_grid, self.departure_period]
        if (
            not np.all(np.isfinite(scalars))
            or self.rho < 0
            or (not 0 <= self.threshold <= 1)
            or (min(self.resource_grid, self.departure_period) <= 0)
        ):
            raise ValueError("Invalid risk parameters or grid lengths")
        if (
            np.any(self.row_start < 0)
            or np.any(self.destination_start < 0)
            or np.any(self.destination_start < self.departure_period * self.choices[:, 2])
            or (
                not np.allclose(
                    self.row_start / self.resource_grid,
                    np.rint(self.row_start / self.resource_grid),
                    rtol=0,
                    atol=1e-10,
                )
            )
        ):
            raise ValueError("Invalid reservation times")
        base = (
            self.destination_start
            + self.destination_nominal
            - self.departure_period * self.choices[:, 2]
            + self.departure_penalty
        )
        base.setflags(write=False)
        object.__setattr__(self, "base", base)

    def _vector(self, value):
        a = np.asarray(value, dtype=float)
        if a.shape != self.base.shape or not np.all(np.isfinite(a)):
            raise ValueError("Expected one finite scalar per choice")
        return a

    def _resource_terms(self, h):
        y = self.occupation @ self._vector(h)
        u = y / self.capacity
        v = np.maximum(u - self.threshold, 0)
        return (
            y,
            self.alpha * u**2,
            self.rho * v**2,
            2 * self.alpha * u / self.capacity,
            2 * self.rho * v / self.capacity,
        )

    def components(self, h):
        _, delay, risk, _, _ = self._resource_terms(h)
        destination_delay = self.destination.T @ delay
        completed = self.destination_start + self.destination_nominal + destination_delay
        elapsed = completed - self.departure_period * self.choices[:, 2]
        reliability = self.occupation.T @ risk
        return {
            "arrival_time": self.destination_start.copy(),
            "completion_time": completed,
            "elapsed_minutes": elapsed,
            "destination_delay_minutes": destination_delay,
            "departure_penalty": self.departure_penalty.copy(),
            "reliability_penalty": reliability,
            "generalized_cost": elapsed + self.departure_penalty + reliability,
        }

    def private(self, h):
        _, delay, risk, _, _ = self._resource_terms(h)
        return self.base + self.destination.T @ delay + self.occupation.T @ risk

    def cost(self, h):
        h = self._vector(h)
        return float(h @ self.private(h))

    def jacobian(self, h):
        _, _, _, delay_prime, risk_prime = self._resource_terms(h)
        D = self.occupation
        return self.destination.T @ (delay_prime[:, None] * D) + D.T @ (risk_prime[:, None] * D)

    def jacobian_vector(self, h, v):
        _, _, _, delay_prime, risk_prime = self._resource_terms(h)
        dv = self.occupation @ self._vector(v)
        return self.destination.T @ (delay_prime * dv) + self.occupation.T @ (risk_prime * dv)

    def jacobian_transpose_vector(self, h, v):
        _, _, _, delay_prime, risk_prime = self._resource_terms(h)
        v = self._vector(v)
        return self.occupation.T @ (
            delay_prime * (self.destination @ v) + risk_prime * (self.occupation @ v)
        )

    def resource_externality(self, h):
        y, _, _, delay_prime, risk_prime = self._resource_terms(h)
        return delay_prime * (self.destination @ self._vector(h)) + risk_prime * y

    def externality(self, h):
        return self.occupation.T @ self.resource_externality(h)

    def gradient(self, h):
        return self.private(h) + self.externality(h)

    def feasibility(self, h):
        h = self._vector(h)
        return {
            "nonnegativity_violation": float(np.maximum(-h, 0).max()),
            "capacity_violation": float(np.maximum(self.occupation @ h - self.capacity, 0).max()),
            "demand_violation": float(np.abs(self.conservation @ h - self.demand).max()),
        }


def _verify_hash(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Frozen input hash differs: {path.name}")


def load_three_channel(evidence_dir=EVIDENCE, stages_path=STAGES):
    """Read the single frozen case without importing historical constructors."""
    archive = Path(evidence_dir) / "three_channel_protected.npz"
    stages_path = Path(stages_path)
    _verify_hash(archive, ARRAY_SHA256)
    _verify_hash(stages_path, STAGE_SHA256)
    with np.load(archive, allow_pickle=False) as stored:
        a = {name: stored[name].copy() for name in stored.files}
    D, resources, starts = (a["occupation"], a["row_resource"], a["row_start"])
    if D.shape != (1345, 180) or a["conservation"].shape != (10, 180):
        raise ValueError("Unexpected frozen configuration dimensions")
    terminal = (resources >= 114) & (resources < 126)
    if not np.all(a["capacity"][terminal] == 3):
        raise ValueError("Expected three terminal channels")
    with stages_path.open(encoding="utf-8-sig", newline="") as stream:
        stages = list(csv.DictReader(stream))
    if len(stages) != 1260:
        raise ValueError("Expected seven stages for each choice")
    names = (
        "origin",
        "inbound_entry",
        "inbound",
        "merge",
        "outbound_entry",
        "outbound",
        "destination",
    )
    grouped = [[] for _ in range(180)]
    for row in stages:
        grouped[int(row["choice"])].append(row)
    rebuilt = np.zeros_like(D)
    destination = np.zeros_like(D)
    dst_start, dst_nominal = (np.zeros(180), np.zeros(180))
    for j, route in enumerate(grouped):
        if tuple((row["stage"] for row in route)) != names:
            raise ValueError("Invalid stage order")
        for row in route:
            start, end = (float(row["start_minutes"]), float(row["end_minutes"]))
            nominal, resource = (float(row["nominal_minutes"]), int(row["resource"]))
            if (
                end <= start
                or nominal <= 0
                or start < 6 * a["choices"][j, 2]
                or (max(abs(start / 3 - round(start / 3)), abs(end / 3 - round(end / 3))) > 1e-10)
            ):
                raise ValueError("Invalid stage interval")
            mask = (resources == resource) & (starts >= start) & (starts < end)
            if int(mask.sum()) != round((end - start) / 3):
                raise ValueError("Stage interval has missing capacity rows")
            rebuilt[mask, j] = 1
            if row["stage"] == "destination":
                destination[mask, j] = 1
                dst_start[j], dst_nominal[j] = (start, nominal)
        serial = [route[k] for k in (0, 2, 3, 5, 6)]
        if any(
            (
                float(left["end_minutes"]) != float(right["start_minutes"])
                for left, right in zip(serial[:-1], serial[1:])
            )
        ):
            raise ValueError("Protected handoffs differ")
        for entry, transit in ((1, 2), (4, 5)):
            if float(route[entry]["start_minutes"]) != float(
                route[transit]["start_minutes"]
            ) or float(route[entry]["end_minutes"]) > float(route[transit]["end_minutes"]):
                raise ValueError("Entrance/transit overlap differs")
    if not np.array_equal(rebuilt, D):
        raise ValueError("Saved occupation differs from protected stage intervals")
    return ProtectedCost(
        occupation=D,
        capacity=a["capacity"],
        alpha=a["alpha"],
        conservation=a["conservation"],
        demand=a["demand"],
        choices=a["choices"],
        row_resource=resources,
        row_start=starts,
        destination=destination,
        destination_start=dst_start,
        destination_nominal=dst_nominal,
        departure_penalty=a["departure_penalty"],
        rho=0.1 * 3,
        threshold=0.7,
    )
