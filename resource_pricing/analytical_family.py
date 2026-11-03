from dataclasses import dataclass
from fractions import Fraction

ZERO = Fraction(0)
ScalarOrTuple = Fraction | tuple[Fraction, ...]
Flow = tuple[Fraction, ...]
__all__ = [
    "Family",
    "private_costs",
    "total_cost",
    "equilibrium",
    "system_optimum",
    "global_node_optimum",
    "capacity_threshold",
]


def _fraction(value, name):
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be a Fraction")


def _vector(value, n, name):
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of Fractions")
    if len(value) != n:
        raise ValueError(f"{name} must have length n")
    for item in value:
        _fraction(item, name)


def _at(value, w):
    return value[w] if isinstance(value, tuple) else value


@dataclass(frozen=True)
class Family:
    n: int
    Q: Flow
    F: ScalarOrTuple
    S: ScalarOrTuple
    beta: Flow
    delta: Flow
    theta: Flow
    L: Flow
    gamma: Fraction
    K: Fraction

    def __post_init__(self):
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive integer")
        for name in ("Q", "beta", "delta", "theta", "L"):
            _vector(getattr(self, name), self.n, name)
        for name in ("F", "S"):
            value = getattr(self, name)
            if isinstance(value, tuple):
                _vector(value, self.n, name)
            else:
                _fraction(value, name)
        for name in ("gamma", "K"):
            _fraction(getattr(self, name), name)
        if any((t <= 0 for row in (self.Q, self.beta, self.delta) for t in row)):
            raise ValueError("Q, beta and delta must be strictly positive")
        if self.gamma < 0 or any((t < 0 for t in self.theta)):
            raise ValueError("gamma and theta must be nonnegative")
        if not ZERO < self.K < sum(self.Q, ZERO):
            raise ValueError("capacity must satisfy 0 < K < sum(Q)")
        for w in range(self.n):
            if _at(self.F, w) + self.beta[w] * self.Q[w] + self.gamma * self.K >= self.L[w]:
                raise ValueError("fast completion bound must be strictly below each deadline")


def _flow(f, x):
    _vector(x, f.n, "x")
    if any((z < 0 or z > q for z, q in zip(x, f.Q))) or sum(x, ZERO) > f.K:
        raise ValueError("x must satisfy 0 <= x <= Q and sum(x) <= K")


def private_costs(f: Family, x: Flow) -> tuple[Flow, Flow]:
    """Return unpriced fast and slow costs, including slow deadline penalties."""
    _flow(f, x)
    u = sum(x, ZERO)
    fast = tuple((_at(f.F, w) + f.beta[w] * x[w] + f.gamma * u for w in range(f.n)))
    slow = []
    for w in range(f.n):
        arrival = _at(f.S, w) + f.delta[w] * (f.Q[w] - x[w])
        slow.append(arrival + f.theta[w] * max(arrival - f.L[w], ZERO))
    return (fast, tuple(slow))


def total_cost(f: Family, x: Flow) -> Fraction:
    """System cost, excluding price transfers."""
    fast, slow = private_costs(f, x)
    return sum((z * cf + (q - z) * cs for z, q, cf, cs in zip(x, f.Q, fast, slow)), ZERO)


def _coefficients(f, w, x):
    late = _at(f.S, w) + f.delta[w] * (f.Q[w] - x) > f.L[w]
    eta = f.delta[w] * (1 + f.theta[w] * late)
    a = f.beta[w] + eta
    d = _at(f.S, w) - _at(f.F, w) + eta * f.Q[w] + f.theta[w] * late * (_at(f.S, w) - f.L[w])
    return (a, d, eta)


@dataclass(frozen=True)
class _Piece:
    lo: Fraction
    hi: Fraction
    slope: Fraction
    intercept: Fraction

    def derivative(self, x):
        return self.slope * x + self.intercept


@dataclass(frozen=True)
class _Branch:
    lo: Fraction | None
    hi: Fraction | None
    r: Flow
    b: Flow

    @property
    def A(self):
        return sum(self.r, ZERO)

    @property
    def B(self):
        return sum(self.b, ZERO)

    def contains(self, mu):
        return (self.lo is None or self.lo <= mu) and (self.hi is None or mu <= self.hi)

    def flow(self, mu):
        return tuple((r - b * mu for r, b in zip(self.r, self.b)))

    def price(self, mu, coupling):
        return (1 + coupling * self.B) * mu - coupling * self.A

    def parameter(self, price, coupling):
        return (price + coupling * self.A) / (1 + coupling * self.B)


def _pieces(f, w, system):
    q = f.Q[w]
    deadline = (_at(f.S, w) + f.delta[w] * q - f.L[w]) / f.delta[w]
    edges = [ZERO, q]
    if f.theta[w] > 0 and ZERO < deadline < q:
        edges.insert(1, deadline)
    result = []
    for lo, hi in zip(edges, edges[1:]):
        a, d, eta = _coefficients(f, w, (lo + hi) / 2)
        slope, intercept = (2 * a, -d - q * eta) if system else (a, -d)
        result.append(_Piece(lo, hi, slope, intercept))
    return tuple(result)


def _inverse_line(pieces, mu):
    """Local affine inverse r-b*mu; derivative jumps give b=0, not a root."""
    if mu <= -pieces[-1].derivative(pieces[-1].hi):
        return (pieces[-1].hi, ZERO)
    if mu >= -pieces[0].derivative(pieces[0].lo):
        return (pieces[0].lo, ZERO)
    for part in pieces:
        root = (-mu - part.intercept) / part.slope
        if part.lo <= root <= part.hi:
            return (-part.intercept / part.slope, 1 / part.slope)
    for left, right in zip(pieces, pieces[1:]):
        if -right.derivative(right.lo) <= mu <= -left.derivative(left.hi):
            return (left.hi, ZERO)
    raise AssertionError("monotone inverse failed to cover a parameter")


def _sample(lo, hi):
    if lo is None:
        return hi - 1
    if hi is None:
        return lo + 1
    return (lo + hi) / 2


def _atlas(f, system=False):
    pieces = tuple((_pieces(f, w, system) for w in range(f.n)))
    breaks = sorted(
        {-part.derivative(edge) for row in pieces for part in row for edge in (part.lo, part.hi)}
    )
    edges = [None, *breaks, None]
    result = []
    for lo, hi in zip(edges, edges[1:]):
        sample = _sample(lo, hi)
        lines = tuple((_inverse_line(row, sample) for row in pieces))
        result.append(_Branch(lo, hi, tuple((r for r, _ in lines)), tuple((b for _, b in lines))))
    return tuple(result)


def _capacity(atlas, K):
    hits = []
    for branch in atlas:
        if branch.B:
            mu = (branch.A - K) / branch.B
            if branch.contains(mu):
                hits.append((mu, branch.flow(mu)))
        elif branch.A == K:
            assert branch.hi is not None
            hits.append((branch.hi, branch.flow(branch.hi)))
    if not hits:
        raise AssertionError("capacity level must be attained by continuous U")
    return max(hits, key=lambda pair: pair[0])


def capacity_threshold(f: Family) -> tuple[Fraction, Flow]:
    """Return (p_K, x_K), using the upper edge if the inverse level is flat."""
    mu, x = _capacity(_atlas(f), f.K)
    return (mu - f.gamma * f.K, x)


def _constrained_response(atlas, K, coupling, price):
    mu_K, x_K = _capacity(atlas, K)
    if price <= mu_K - coupling * K:
        return x_K
    for branch in atlas:
        mu = branch.parameter(price, coupling)
        if branch.contains(mu):
            return branch.flow(mu)
    raise AssertionError("strictly increasing price map must have a root")


def equilibrium(f: Family, p: Fraction) -> Flow:
    """Unique constrained priced equilibrium for any finite rational net price."""
    _fraction(p, "p")
    return _constrained_response(_atlas(f), f.K, f.gamma, p)


def system_optimum(f: Family) -> Flow:
    """Unique global SO, including bounds, capacity and nonsmooth deadlines."""
    return _constrained_response(_atlas(f, system=True), f.K, 2 * f.gamma, ZERO)


def _quadratic(f, branch, mu):
    """Coefficients of Phi(v(mu)); mu selects a branch, not a fitting sample."""
    c2 = f.gamma * branch.B**2
    c1 = -2 * f.gamma * branch.A * branch.B
    for w, (r, b, x) in enumerate(zip(branch.r, branch.b, branch.flow(mu))):
        a, d, eta = _coefficients(f, w, x)
        c2 += a * b * b
        c1 += (d + f.Q[w] * eta - 2 * a * r) * b
    return (c2, c1)


def global_node_optimum(f: Family, p_lo: Fraction, p_hi: Fraction) -> dict:
    """Exact complete-node-class minimum on the nonempty closed price interval.

    Candidate dictionaries contain kind, p, x, cost and interval. Constant and
    capacity candidates represent their whole interval; other candidates are
    endpoints or stationary points of the indicated quadratic branch. Duplicate
    shared endpoints are retained. Every tied minimum remains in candidates.
    """
    _fraction(p_lo, "p_lo")
    _fraction(p_hi, "p_hi")
    if p_lo > p_hi:
        raise ValueError("price interval must be nonempty: p_lo <= p_hi")
    atlas = _atlas(f)
    mu_K, x_K = _capacity(atlas, f.K)
    p_K = mu_K - f.gamma * f.K
    candidates = []

    def add(kind, p, x, interval):
        candidates.append(dict(kind=kind, p=p, x=x, cost=total_cost(f, x), interval=interval))

    if p_lo <= p_K:
        add("capacity", p_lo, x_K, (p_lo, min(p_hi, p_K)))
    for branch in atlas:
        lo, hi = (max(p_lo, p_K), p_hi)
        if branch.lo is not None:
            lo = max(lo, branch.price(branch.lo, f.gamma))
        if branch.hi is not None:
            hi = min(hi, branch.price(branch.hi, f.gamma))
        if lo > hi or hi <= p_K:
            continue
        interval = (lo, hi)
        if not branch.B:
            assert lo > p_K
            add("constant", lo, branch.flow(branch.parameter(lo, f.gamma)), interval)
            continue
        for p in sorted({lo, hi}):
            if p > p_K:
                add("endpoint", p, branch.flow(branch.parameter(p, f.gamma)), interval)
        sample = branch.parameter((lo + hi) / 2, f.gamma)
        c2, c1 = _quadratic(f, branch, sample)
        assert c2 > 0
        stationary_mu = -c1 / (2 * c2)
        stationary_p = branch.price(stationary_mu, f.gamma)
        if lo < stationary_p < hi:
            add("stationary", stationary_p, branch.flow(stationary_mu), interval)
    if not candidates:
        raise AssertionError("nonempty price interval must contribute a candidate")
    best = min(candidates, key=lambda row: (row["cost"], row["p"], row["x"]))
    return dict(cost=best["cost"], x=best["x"], p=best["p"], candidates=candidates)
