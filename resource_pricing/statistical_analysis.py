from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from numbers import Real
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, t
import statsmodels
from statsmodels.regression.linear_model import OLS
from resource_pricing import random_networks as original

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/random_networks/random_geometric_validation_instances.csv"
OUTPUT = ROOT / "results/statistical_analysis"
STANDARDIZED_TERMS = {
    "projection_residual": "Projection residual",
    "due_so_gap": "DUE-SO gap",
    "delayed_intensity": "Delayed intensity",
    "pressure": "Pressure",
    "resource_concentration": "Resource concentration",
}
PRICE_CLASSES = {"corridor", "node", "contemporaneous"}
BOOTSTRAP_SEED = 20261008


@dataclass
class RegressionData:
    panel: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    terms: list[str]
    standardization: dict[str, dict[str, float]]


def _validate_arrays(x, y, groups):
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    groups = np.asarray(groups, dtype=object)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have compatible matrix/vector shapes")
    n, k = x.shape
    if k == 0 or n <= k:
        raise ValueError("OLS inference requires n > k > 0")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite")
    if groups.ndim != 1 or len(groups) != n:
        raise ValueError("group labels must be a vector of length n")
    if pd.isna(groups).any() or any((isinstance(g, Real) and (not np.isfinite(g)) for g in groups)):
        raise ValueError("group labels must be nonmissing and finite")
    codes, labels = pd.factorize(groups, sort=True)
    if len(labels) < 2:
        raise ValueError("cluster inference requires at least two groups")
    if np.linalg.matrix_rank(x) < k:
        raise ValueError("design matrix is rank deficient")
    return (x, y, groups, codes, labels)


def build_regression_data(results: pd.DataFrame) -> RegressionData:
    """Reuse original row construction, finite-row filter, and population scaling."""
    if "Seed" not in results:
        raise ValueError("source must include Seed")
    ok = results.loc[results["success"] == True]
    seeds = ok["Seed"].to_numpy()
    if pd.isna(seeds).any() or any((isinstance(g, Real) and (not np.isfinite(g)) for g in seeds)):
        raise ValueError("successful network Seed values must be nonmissing and finite")
    if ok["Seed"].duplicated().any():
        raise ValueError("each successful network must have a unique Seed")
    raw = original._mechanism_regression_panel(results)
    if ok.empty or raw.empty or len(raw) % len(ok):
        raise ValueError("original regression row construction is empty or inconsistent")
    cleaned = raw.replace([np.inf, -np.inf], np.nan)
    keep = cleaned.notna().all(axis=1)
    panel = cleaned.loc[keep].copy()
    panel["Seed"] = np.repeat(seeds, len(raw) // len(ok))[keep.to_numpy()]
    parts, terms, standardization = ([np.ones((len(panel), 1))], ["Intercept"], {})
    for column, name in STANDARDIZED_TERMS.items():
        values = panel[column].to_numpy(dtype=float)
        standardization[column] = {
            "mean": float(np.nanmean(values)),
            "std_ddof0": float(np.nanstd(values)),
        }
        parts.append(original._standardized_column(panel[column])[:, None])
        terms.append(name)
    if panel["price_class"].nunique() >= 3 and len(panel) >= 12:
        for price_class, name in [
            ("node", "Node price class"),
            ("contemporaneous", "Contemporaneous price class"),
        ]:
            parts.append((panel["price_class"].to_numpy() == price_class).astype(float)[:, None])
            terms.append(name)
    return RegressionData(
        panel,
        np.hstack(parts),
        panel["recovery_loss"].to_numpy(dtype=float),
        panel["Seed"].to_numpy(),
        terms,
        standardization,
    )


def cluster_ols(x, y, groups) -> dict:
    """OLS with an independently calculated finite-corrected cluster sandwich."""
    x, y, groups, codes, labels = _validate_arrays(x, y, groups)
    n, k = x.shape
    g = len(labels)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    block_scores = np.zeros((g, k))
    np.add.at(block_scores, codes, x * residual[:, None])
    correction = g / (g - 1) * (n - 1) / (n - k)
    covariance = correction * bread @ (block_scores.T @ block_scores) @ bread
    return {
        "coefficient": beta,
        "covariance": covariance,
        "correction": correction,
        "observations": n,
        "parameters": k,
        "clusters": g,
    }


def network_pairs_bootstrap(
    x, y, groups, price_classes, *, draws=2000, seed=BOOTSTRAP_SEED
) -> pd.DataFrame:
    """Resample G whole three-class blocks; do not restandardize or redraw failures."""
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    x, y, groups, codes, labels = _validate_arrays(x, y, groups)
    classes = np.asarray(price_classes)
    if classes.ndim != 1 or len(classes) != len(y):
        raise ValueError("price classes must have length n")
    blocks = [np.flatnonzero(codes == g) for g in range(len(labels))]
    if any((len(block) != 3 or set(classes[block]) != PRICE_CLASSES for block in blocks)):
        raise ValueError("each network block must include exactly the three distinct price classes")
    if x.shape[1] < 2:
        raise ValueError("design must include the projection slope at column 1")
    rng = np.random.default_rng(seed)
    samples = []
    for draw in range(draws):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[g] for g in chosen])
        sampled_x, sampled_y = (x[indices], y[indices])
        beta, _, rank, _ = np.linalg.lstsq(sampled_x, sampled_y, rcond=None)
        identified = rank == x.shape[1]
        samples.append(
            {
                "draw": draw + 1,
                "projection_slope": float(beta[1]) if identified else np.nan,
                "rank": int(rank),
                "observations": len(indices),
                "unique_networks": len(np.unique(chosen)),
                "status": "ok" if identified else "rank_deficient",
            }
        )
    return pd.DataFrame(samples)


def run_audit(source=SOURCE, out_dir=OUTPUT, *, draws=2000) -> dict:
    """Verify historical results before saving inference artifacts."""
    source, out_dir = (Path(source), Path(out_dir))
    source_bytes = source.read_bytes()
    results = pd.read_csv(io.BytesIO(source_bytes))
    successes = int((results["success"] == True).sum())
    if len(results) != 300 or successes != 295:
        raise ValueError("expected 300 rows / 295 successes in frozen source")
    data = build_regression_data(results)
    if data.x.shape != (885, 8) or len(np.unique(data.groups)) != 295:
        raise ValueError(
            "frozen source must yield 885 observations, 8 parameters, and 295 Seed clusters"
        )
    old, old_summary = original._summarize_mechanism_regression(results)
    reference_path = source.with_name("random_geometric_mechanism_regression.csv")
    stored = pd.read_csv(reference_path)
    if old["term"].tolist() != stored["term"].tolist():
        raise ValueError("historical regression term ordering changed")
    np.testing.assert_allclose(
        old[["coefficient", "robust_se", "robust_t"]],
        stored[["coefficient", "robust_se", "robust_t"]],
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        old.loc[1, ["coefficient", "robust_t"]].to_numpy(dtype=float),
        [0.03486883493915299, 2.3604579783134736],
        rtol=1e-11,
        atol=1e-12,
    )
    fit = cluster_ols(data.x, data.y, data.groups)
    np.testing.assert_array_equal(fit["coefficient"], old["coefficient"].to_numpy())
    reference = OLS(data.y, data.x).fit(
        cov_type="cluster", cov_kwds={"groups": data.groups, "use_correction": True}, use_t=True
    )
    covariance = fit["covariance"]
    np.testing.assert_allclose(covariance, reference.cov_params(), rtol=1e-10, atol=1e-12)
    cluster_se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    beta = fit["coefficient"]
    cluster_t = np.divide(beta, cluster_se, out=np.full_like(beta, np.nan), where=cluster_se > 0)
    cluster_df = fit["clusters"] - 1
    cluster_p = 2 * t.sf(np.abs(cluster_t), df=cluster_df)
    np.testing.assert_allclose(cluster_p, reference.pvalues, rtol=1e-10, atol=1e-12)
    cluster_critical = t.ppf(0.975, df=cluster_df)
    regression = pd.DataFrame(
        {
            "term": data.terms,
            "coefficient": beta,
            "hc1_se": old["robust_se"].to_numpy(),
            "hc1_t": old["robust_t"].to_numpy(),
            "hc1_p_normal": 2 * norm.sf(np.abs(old["robust_t"])),
            "cluster_se": cluster_se,
            "cluster_t": cluster_t,
            "cluster_p_t": cluster_p,
            "cluster_ci95_low": beta - cluster_critical * cluster_se,
            "cluster_ci95_high": beta + cluster_critical * cluster_se,
            "statsmodels_cluster_se": reference.bse,
            "observations": fit["observations"],
            "clusters": fit["clusters"],
            "parameters": fit["parameters"],
            "cluster_reference_df": cluster_df,
            "r2": old_summary["mechanism_regression_r2"],
        }
    )
    samples = network_pairs_bootstrap(
        data.x, data.y, data.groups, data.panel["price_class"], draws=draws
    )
    valid = samples.loc[samples["status"] == "ok", "projection_slope"].to_numpy()
    interval = np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None
    summary = {
        "source": {
            "path": source.resolve().as_posix(),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(results),
            "successes": successes,
            "failed_networks": len(results) - successes,
            "successful_seeds": results.loc[results["success"] == True, "Seed"].tolist(),
            "failed_seeds": results.loc[results["success"] != True, "Seed"].tolist(),
            "historical_regression_path": reference_path.resolve().as_posix(),
            "historical_regression_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "original_helper_path": Path(original.__file__).resolve().as_posix(),
            "original_helper_sha256": hashlib.sha256(
                Path(original.__file__).read_bytes()
            ).hexdigest(),
        },
        "regression": {
            "observations": fit["observations"],
            "parameters": fit["parameters"],
            "clusters": fit["clusters"],
            "terms": data.terms,
            "filter_policy": "Original success == True, original long rows, replace +/-inf with NaN then dropna; Seed never adds exclusions",
            "class_reference": "corridor",
            "observations_per_network": 3,
            "standardization": data.standardization,
            "standardization_policy": "Original population std (ddof=0); std <= 1e-12 maps to zeros; response and class dummies unscaled",
            "r2": old_summary["mechanism_regression_r2"],
            "cluster_covariance": covariance.tolist(),
            "finite_cluster_correction_formula": "G/(G-1)*(n-1)/(n-k)",
            "finite_cluster_correction": fit["correction"],
            "cluster_reference_distribution": "Student t with G-1 degrees of freedom",
            "cluster_reference_df": cluster_df,
            "hc1_reference_distribution": "standard normal (comparison only)",
        },
        "projection": {
            "coefficient": float(beta[1]),
            "hc1_se": float(old.loc[1, "robust_se"]),
            "hc1_t": float(old.loc[1, "robust_t"]),
            "cluster_se": float(cluster_se[1]),
            "cluster_t": float(cluster_t[1]),
            "cluster_p_t": float(cluster_p[1]),
            "cluster_ci95": regression.loc[1, ["cluster_ci95_low", "cluster_ci95_high"]].tolist(),
        },
        "statsmodels_check": {
            "passed": True,
            "cov_type": "cluster",
            "use_correction": True,
            "use_t": True,
            "rtol": 1e-10,
            "atol": 1e-12,
            "max_abs_covariance_difference": float(
                np.max(np.abs(covariance - reference.cov_params()))
            ),
            "max_abs_coefficient_difference": float(np.max(np.abs(beta - reference.params))),
        },
        "bootstrap": {
            "draws": int(draws),
            "seed": BOOTSTRAP_SEED,
            "rng": "numpy.random.default_rng / PCG64",
            "resampling_policy": "295 Seed blocks sampled with replacement per draw; all three class observations preserved",
            "standardization_policy": "frozen X from the original full regression sample; no within-draw restandardization",
            "valid_draws": len(valid),
            "rank_deficient_draws": int(samples["status"].eq("rank_deficient").sum()),
            "failure_policy": "Record rank-deficient draws as NaN; no redraw; interval uses identified draws only",
            "percentile_policy": "numpy.percentile, linear interpolation, [2.5, 97.5]",
            "percentile_95_interval": interval,
        },
        "environment": {
            "executable": sys.executable,
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "interpretation": "Conditional association in the frozen simulated-network design, not causality or theoretical necessity",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    regression.to_csv(out_dir / "regression.csv", index=False)
    samples.to_csv(out_dir / "bootstrap_slopes.csv", index=False)
    (out_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
