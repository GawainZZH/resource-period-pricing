from fractions import Fraction as F

CHOICES = ("F1", "S1", "F2", "S2")
DEADLINES = (F(113, 20), F(113, 20), F(227, 40), F(227, 40))
BASE = (F(4), F(28, 5), F(4), F(28, 5))
SLOT = F(1, 2)


def build_inputs():
    rows = []
    resources = []

    def add(name, kind, first, last, occupation, delay_slope):
        physical_name = "B" if kind == "node" else name
        resources.append(
            dict(
                name=name,
                physical_resource=physical_name,
                kind=kind,
                interval=[str(first * SLOT), str(last * SLOT)],
                occupation=occupation,
                delay_slope=str(delay_slope),
            )
        )
        for s in range(first, last):
            rows.append(
                dict(
                    resource=name,
                    physical_resource=physical_name,
                    kind=kind,
                    period=s,
                    start=str(s * SLOT),
                    end=str((s + 1) * SLOT),
                    occupation=occupation,
                    delay_slope=str(delay_slope),
                )
            )

    for j, label in enumerate(CHOICES):
        add("I_" + label, "corridor", 0, 2, [int(i == j) for i in range(4)], F(1, 20))
    add("R", "corridor", 1, 2, [1, 0, 0, 1], F(1, 20))
    for j, label in enumerate(CHOICES):
        add(
            "C_" + label,
            "corridor",
            1,
            9 if j in (0, 2) else 12,
            [int(i == j) for i in range(4)],
            F(0),
        )
    add("B_fast", "node", 7, 11, [1, 0, 1, 0], F(1, 4))
    add("B_slow", "node", 11, 12, [0, 1, 0, 1], F(0))
    return dict(
        design="crossing_corridor_deadline_network_v1",
        status="static_candidate",
        choice_order=list(CHOICES),
        departure_times=["0"] * 4,
        departure_step="1/2",
        resource_step="1/2",
        horizon=["0", "6"],
        demand=["1", "1"],
        conservation=[[1, 1, 0, 0], [0, 0, 1, 1]],
        resources=resources,
        rows=rows,
        node_resource_identity={"B_fast": "B", "B_slow": "B"},
        deadlines=[str(t) for t in DEADLINES],
        lateness_rates=["1/2"] * 4,
        baseline_completion=[str(t) for t in BASE],
        node_capacity_gate=dict(
            high="11/10",
            slow_capacity="2",
            formula="chi(k)=min(1,max(0,(2-k)/(2-11/10))); d_B=eta*y*chi(k)/4",
        ),
        stages=dict(
            initial_duration="1/2",
            initial_total_delay="eta*h_j/10",
            crossing_choices=["F1", "S2"],
            crossing_duration="1/4",
            crossing_delay="eta*(h_F1+h_S2)/20",
            connector_durations=["16/5", "101/20", "69/20", "24/5"],
            terminal_base_duration="1/20",
            fast_terminal_delay="eta*(h_F1+h_F2)",
            slow_terminal_delay="0",
            fixed_handoffs=[],
            reserved_capacity_release="at protected interval end",
        ),
        states=[
            dict(id="B", capacity="1", eta="1"),
            dict(id="K-", capacity="9/10", eta="1"),
            dict(id="K+", capacity="11/10", eta="1"),
            dict(id="C-", capacity="1", eta="9/10"),
            dict(id="C+", capacity="1", eta="11/10"),
        ],
        caps=["0", "1/4", "1"],
        rate_unit="generalized cost per reserved position per minute",
        price_matrix="resource_step * transpose(occupation rows selected by class)",
        price_classes=["corridor", "node", "contemporaneous"],
        topology=dict(
            nodes=["O1", "O2", "U", "V", "J1", "J2", "D"],
            edges=[
                ["I_F1", "O1", "U"],
                ["I_S2", "O2", "U"],
                ["R", "U", "V"],
                ["C_F1", "V", "D"],
                ["C_S2", "V", "D"],
                ["I_S1", "O1", "J1"],
                ["C_S1", "J1", "D"],
                ["I_F2", "O2", "J2"],
                ["C_F2", "J2", "D"],
            ],
            candidate_paths=[
                ["I_F1", "R", "C_F1", "B"],
                ["I_S1", "C_S1", "B"],
                ["I_F2", "C_F2", "B"],
                ["I_S2", "R", "C_S2", "B"],
            ],
            candidate_scope="Two declared paths per OD, not all topological walks",
        ),
        reports_ceiling=45,
        price_domains_ceiling=35,
        task_ceiling=40,
        objective="optimistic minimum system cost over complete class and all VI equilibria",
        scientific_optimization_authorized=False,
    )


def flows(x):
    return (x[0], 1 - x[0], x[1], 1 - x[1])


def vertices(k):
    out = {(F(0), F(0)), (min(k, F(1)), F(0)), (F(0), min(k, F(1)))}
    if k > 1:
        out.update({(F(1), k - 1), (k - 1, F(1))})
    return tuple(sorted(out))


def arrivals(x, eta=F(1)):
    x1, x2 = x
    u, v = (x1 + x2, x1 + 1 - x2)
    return (
        4 + eta * (x1 / 10 + u + v / 20),
        F(28, 5) + eta * (1 - x1) / 10,
        4 + eta * (x2 / 10 + u),
        F(28, 5) + eta * ((1 - x2) / 10 + v / 20),
    )


def deadline_flags(x, eta=F(1)):
    return tuple((a > L for a, L in zip(arrivals(x, eta), DEADLINES)))


def costs(x, eta=F(1), penalty=True):
    return tuple(
        (
            a + (F(1, 2) * max(F(0), a - L) if penalty else 0)
            for a, L in zip(arrivals(x, eta), DEADLINES)
        )
    )


def system_cost(x, eta=F(1), penalty=True):
    return sum((h * c for h, c in zip(flows(x), costs(x, eta, penalty))))


def gradient(x, eta=F(1), penalty=True):
    a = arrivals(x, eta)
    if penalty and any((v == L for v, L in zip(a, DEADLINES))):
        raise ValueError("A deadline equality requires directional derivatives.")
    jac = ((F(23, 20), F(19, 20)), (-F(1, 10), F(0)), (F(1), F(11, 10)), (F(1, 20), -F(3, 20)))
    h, c = (flows(x), costs(x, eta, penalty))
    out = [c[0] - c[1], c[2] - c[3]]
    for j in range(4):
        weight = F(3, 2) if penalty and a[j] > DEADLINES[j] else F(1)
        for i in range(2):
            out[i] += h[j] * weight * eta * jac[j][i]
    return tuple(out)


def capacity(row, state):
    if row["resource"] == "B_fast":
        return F(state["capacity"])
    return F(2) if row["resource"] in {"R", "B_slow"} else F(1)


def matrix_arrivals(data, state, x):
    h = flows(x)
    result = list(BASE)
    eta = F(state["eta"])
    for row in data["rows"]:
        delta = row["occupation"]
        y = sum((h[j] * delta[j] for j in range(4)))
        if row["kind"] == "node":
            k = capacity(row, state)
            gate = min(F(1), max(F(0), (2 - k) / (2 - F(11, 10))))
            delay = eta * y * gate / 4
        else:
            delay = eta * y * F(row["delay_slope"])
        for j in range(4):
            result[j] += delta[j] * delay
    return tuple(result)


def verify_stages(x, eta):
    h = flows(x)
    times = []
    for j in range(4):
        initial_end = F(1, 2) + eta * h[j] / 10
        assert 0 < initial_end <= 1
        connector_start = initial_end
        if j in (0, 3):
            assert connector_start >= F(1, 2)
            connector_start += F(1, 4) + eta * (h[0] + h[3]) / 20
            assert connector_start <= 1
        connector_duration = (F(16, 5), F(101, 20), F(69, 20), F(24, 5))[j]
        service_start = connector_start + connector_duration
        assert connector_start >= F(1, 2)
        assert service_start <= (F(9, 2) if j in (0, 2) else F(6))
        node_delay = eta * (h[0] + h[2]) if j in (0, 2) else F(0)
        completion = service_start + F(1, 20) + node_delay
        left, right = (F(7, 2), F(11, 2)) if j in (0, 2) else (F(11, 2), F(6))
        assert left <= service_start <= completion <= right
        times.append(completion)
    assert tuple(times) == arrivals(x, eta)
    return tuple(times)


def covered_rows(data, kind):
    if kind == "corridor":
        return [i for i, r in enumerate(data["rows"]) if r["kind"] == "corridor"]
    if kind == "node":
        return [i for i, r in enumerate(data["rows"]) if r["kind"] == "node"]
    if kind == "contemporaneous":
        return [i for i, r in enumerate(data["rows"]) if r["period"] == 0]
    raise ValueError(kind)


def price_differences(data, theta):
    tau = [
        sum((SLOT * t * r["occupation"][j] for t, r in zip(theta, data["rows"]))) for j in range(4)
    ]
    return (tau[0] - tau[1], tau[2] - tau[3])


def reverse_prices(data, kind, cap, target):
    if cap < 0:
        raise ValueError("Negative cap.")
    d1, d2 = target
    theta = [F(0)] * len(data["rows"])
    if kind == "node":
        if d1 != d2 or not -cap / 2 <= d1 <= 2 * cap:
            raise ValueError("Outside full node price image.")
        for i in covered_rows(data, kind):
            r = data["rows"][i]
            theta[i] = max(F(0), d1) / 2 if r["resource"] == "B_fast" else max(F(0), -d1) * 2
    elif kind == "contemporaneous":
        if any((abs(d) > cap / 2 for d in target)):
            raise ValueError("Outside contemporaneous price image.")
        rates = (max(F(0), d1) * 2, max(F(0), -d1) * 2, max(F(0), d2) * 2, max(F(0), -d2) * 2)
        for i in covered_rows(data, kind):
            j = data["rows"][i]["occupation"].index(1)
            theta[i] = rates[j]
    elif kind == "corridor":
        low, high = (-F(13, 2) * cap, 5 * cap)
        zlo = max(F(0), d1 - high, low - d2)
        zhi = min(cap / 2, d1 - low, high - d2)
        if zlo > zhi:
            raise ValueError("Outside full corridor zonotope.")
        z = zlo
        private = (d1 - z, d2 + z)
        rates = (
            max(F(0), private[0]) / 5,
            max(F(0), -private[0]) / F(13, 2),
            max(F(0), private[1]) / 5,
            max(F(0), -private[1]) / F(13, 2),
        )
        for i in covered_rows(data, kind):
            r = data["rows"][i]
            theta[i] = 2 * z if r["resource"] == "R" else rates[r["occupation"].index(1)]
    else:
        raise ValueError(kind)
    assert all((0 <= t <= cap for t in theta))
    assert price_differences(data, theta) == target
    return theta
