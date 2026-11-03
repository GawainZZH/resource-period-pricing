# Resource-Period Pricing

Numerical code and supporting data for **Resource-Period Pricing Characterization for Capacity-Constrained Low-Altitude Corridor Networks**.

The study examines the resource and temporal coverage needed to implement system-optimal capacity reservations. The numerical analyses compare specified marginal-cost prices with optimized restricted prices, evaluate price-class primal gaps, and examine price bounds, gradual updates and indivisible flight requests.

## Installation

Python 3.13 is recommended.

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the reported summaries

```bash
python reproduce.py summarize
python reproduce.py examples
python -m pytest tests
```

`summarize` calculates the principal pricing comparisons from the saved results. `examples` evaluates the exact background-flow examples using rational arithmetic. Generated files are written to `results/`; the data accompanying the article remain in `data/`.

## Numerical studies

```bash
# Constructed networks
python reproduce.py constructed --instances 200

# Random geometric networks
python reproduce.py random --instances 300 --maxiter 150

# Complete price-class gaps at the saved refined system optima
python reproduce.py class-gaps --instances 200

# Recompute the system optima before evaluating the class gaps
python reproduce.py class-gaps --instances 200 --recompute-so

# Optimization within corridor, node and contemporaneous price classes
python reproduce.py restricted-prices --instances 200

# Gradual price formation in 24 instances
python reproduce.py adaptive

# Seven reliability thresholds on the same 200 networks
python reproduce.py reliability --instances 200

# Capacity pressure on twelve fixed networks
python reproduce.py pressure

# Seven network sizes, with five instances per size
python reproduce.py scalability

# Nonmonotone equilibrium-selection examples
python reproduce.py selection

# Six resource-period lengths in the operational network
python reproduce.py reservations

# Exact node-class values in the analytical reservation networks
python reproduce.py families

# Network-clustered inference and whole-network bootstrap
python reproduce.py statistics
```

For an initial numerical example, run `python reproduce.py constructed --instances 1`. The `--out` option selects a separate result directory. Restricted-price optimization and full sensitivity studies involve repeated equilibrium solutions and require more computation than summary reconstruction.

The random-network data contain all 300 generated instances. Reported aggregate comparisons retain the 295 instances satisfying the original numerical acceptance criteria. The saved selection is used for summary reconstruction; a new solver run reports its own convergence results.

## Data

- `data/inputs/`: numerical specifications for the constructed and random networks, adaptive pricing, sensitivity studies and analytical price classes.
- `data/component_prices/`: the 200 constructed instances and the illustrative-network comparison.
- `data/restricted_prices/`: computed prices, allocations and system costs within the three restricted classes.
- `data/class_gaps/`: refined SO flows, direct class gaps and associated numerical solutions.
- `data/random_networks/`, `data/random_restricted_prices/` and `data/random_class_gaps/`: random-network pricing results and the complete-class comparisons.
- `data/adaptive_pricing/` and `data/matched_updates/`: observed-flow price updates, matched relaxation comparisons and cost intervals.
- `data/reliability/` and `data/scalability/`: matched-network sensitivity and network-size comparisons.
- `data/exact_classes/` and `data/two_class_values/`: exact reservation responses and restricted-price cost values.
- `data/reservation_inputs/`, `data/protected_reservations/` and `data/nominal_reservations/`: occupation matrices, capacity constraints and reservation allocations.
- `data/minimum_price_caps/`, `data/individual_responses/` and `data/reservation_sensitivity/`: minimum-cap calculations, integer responses and the six-setting appendix comparisons.

The input specifications preserve the reported study populations. Constructed networks contain 36–400 route and departure choices. The resource-period sensitivity uses 1, 1.2, 1.5, 2, 3 and 6 minutes; the integer comparison uses capacity multipliers 1, 2, 3, 5, 7 and 10. The protected three-channel case contains 40 requests and 180 choices.

## Measures

For system cost $\Phi$, the DUE–SO gap closure ratio is

$$
\mathrm{GCR}(\tau)=
\frac{\Phi(h^{\mathrm{DUE}})-\Phi(h^\tau)}
{\Phi(h^{\mathrm{DUE}})-\Phi(h^{\mathrm{SO}})}.
$$

A value of 1 corresponds to the SO cost; a negative value denotes a cost above the unpriced DUE. The price of anarchy is $\Phi(h^{\mathrm{DUE}})/\Phi(h^{\mathrm{SO}})$. The primal gap at the SO evaluates a specified price, whereas the minimum price-class gap optimizes that condition across the available price set.

CSV fields named `recovery` or `Recovery` store GCR values. Fields named `SO support gap` store the primal gap at the SO. In the restricted-price search files, `stationary price GCR` stores the GCR of the computed price–equilibrium pair; the numerical search is distinct from the exact class values obtained for the analytical networks.

## Methods

`resource_pricing/` contains the occupation models, equilibrium and SO solvers, price mappings, coverage LPs, exact reservation responses and integer allocation methods. Exact calculations employ rational arithmetic. Convex resource-cost calculations employ SciPy and CVXPY; the LP and MILP calculations employ HiGHS. Statistical inference resamples whole network instances.

The files in `data/` contain the results accompanying the article and supplementary material. New computations are saved separately. Numerical precision and solver settings are specified in the numerical routines and study inputs.
