from fractions import Fraction as F
import numpy as np


def integer_vector(values):
    values = tuple((F(float(v)) if isinstance(v, (float, np.floating)) else F(v) for v in values))
    if any((v.denominator != 1 or v < 0 for v in values)):
        raise ValueError("Nonnegative integer request flows required")
    return tuple((int(v) for v in values))


class AtomicGame:

    def __init__(self, model):
        arrays = [np.asarray(v) for v in (model.occupation, model.destination, model.conservation)]
        if any((a.ndim != 2 or not np.all((a == 0) | (a == 1)) for a in arrays)):
            raise ValueError("Binary occupation, destination and ownership matrices required")
        D, E, H = arrays
        if (
            D.shape != E.shape
            or D.shape[1] != H.shape[1]
            or np.any(E > D)
            or (not np.all(H.sum(axis=0) == 1))
        ):
            raise ValueError("Inconsistent matrix dimensions or resource selection")
        self.D, self.E, self.H = [a.astype(np.int64) for a in arrays]
        self.K = np.asarray(integer_vector(model.capacity), dtype=np.int64)
        self.q = np.asarray(integer_vector(model.demand), dtype=np.int64)
        if self.K.shape != (len(D),) or self.q.shape != (len(H),) or np.any(self.K <= 0):
            raise ValueError("Positive integer capacities with matching demands required")
        self.n = D.shape[1]
        if len(model.base) != self.n or len(model.alpha) != len(D):
            raise ValueError("Invalid cost dimensions")
        self.base = tuple((F(float(v)) for v in model.base))
        alpha = tuple((F(float(v)) for v in model.alpha))
        rho, eta = (F(float(model.rho)), F(float(model.threshold)))
        if min(alpha, default=F(0)) < 0 or rho < 0 or (not 0 <= eta <= 1):
            raise ValueError("Nondecreasing resource costs required")
        self.delay = tuple(
            (
                tuple((a * F(y, int(k)) ** 2 for y in range(int(k) + 1)))
                for a, k in zip(alpha, self.K)
            )
        )
        self.risk = tuple(
            (
                tuple((rho * max(F(0), F(y, int(k)) - eta) ** 2 for y in range(int(k) + 1)))
                for k in self.K
            )
        )
        self.used = tuple((tuple((int(r) for r in np.flatnonzero(D[:, j]))) for j in range(self.n)))
        self.destination = tuple(
            (tuple((int(r) for r in np.flatnonzero(E[:, j]))) for j in range(self.n))
        )
        owner = H.argmax(axis=0)
        self.peers = tuple((tuple((int(j) for j in np.flatnonzero(H[int(w)]))) for w in owner))

    def flow(self, values):
        h = integer_vector(values)
        if len(h) != self.n:
            raise ValueError("Invalid request flow dimension")
        return h

    def feasible(self, values):
        h = np.array(self.flow(values), dtype=np.int64)
        return bool(np.array_equal(self.H @ h, self.q) and np.all(self.D @ h <= self.K))

    def load(self, values):
        h = self.flow(values)
        if not self.feasible(h):
            raise ValueError("An exactly capacity-feasible integer allocation required")
        return self.D @ np.array(h, dtype=np.int64)

    def private_at_load(self, j, y):
        return (
            self.base[j]
            + sum((self.delay[r][int(y[r])] for r in self.destination[j]), F(0))
            + sum((self.risk[r][int(y[r])] for r in self.used[j]), F(0))
        )

    def private(self, values):
        y = self.load(values)
        return tuple((self.private_at_load(j, y) for j in range(self.n)))

    def total(self, values):
        h = self.flow(values)
        return sum((v * c for v, c in zip(h, self.private(h))), F(0))

    def scan(self, values, charges=None):
        h, y = (self.flow(values), self.load(values))
        charges = tuple((F(v) for v in charges)) if charges is not None else (F(0),) * self.n
        if len(charges) != self.n:
            raise ValueError("One fixed charge per choice required")
        current = tuple((self.private_at_load(i, y) + charges[i] for i in range(self.n)))
        directions, best = ([], None)
        feasible, blocked = (0, 0)
        for i in range(self.n):
            if h[i] == 0:
                continue
            for j in self.peers[i]:
                if i == j:
                    continue
                after = y - self.D[:, i] + self.D[:, j]
                bad = np.flatnonzero(after > self.K)
                row = dict(from_=i, to=j, multiplicity=h[i], feasible=not len(bad))
                if len(bad):
                    row["violated_rows"] = [int(v) for v in bad]
                    blocked += 1
                else:
                    new_cost = self.private_at_load(j, after) + charges[j]
                    gain = current[i] - new_cost
                    row.update(old_cost=current[i], new_cost=new_cost, gain=gain)
                    feasible += 1
                    if gain > 0 and (best is None or gain > best["gain"]):
                        best = row.copy()
                directions.append(row)
        return dict(
            equilibrium=best is None,
            best=best,
            directions=directions,
            possible_directions=len(directions),
            feasible_directions=feasible,
            blocked_directions=blocked,
        )


def improve(game, initial, max_moves, checkpoint=None, charges=None):
    h = game.flow(initial)
    seen, moves, states = (set(), [], [h])
    for step in range(max_moves + 1):
        if h in seen:
            return dict(status="cycle", flow=h, moves=moves, states=states, final_scan=None)
        seen.add(h)
        scan = game.scan(h, charges)
        if checkpoint:
            checkpoint(step, h, scan)
        if scan["equilibrium"]:
            return dict(status="equilibrium", flow=h, moves=moves, states=states, final_scan=scan)
        if step == max_moves:
            return dict(status="move_limit", flow=h, moves=moves, states=states, final_scan=scan)
        move = scan["best"]
        new = list(h)
        new[move["from_"]] -= 1
        new[move["to"]] += 1
        h = tuple(new)
        moves.append(move)
        states.append(h)
    raise AssertionError("Finite move budget must terminate")


def strict_cost_gate(baseline_cost, target_cost):
    return F(baseline_cost) > F(target_cost)
