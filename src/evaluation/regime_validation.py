"""
regime_validation.py — Validate that regime-conditioned generation actually
produces distributional shifts.

Two tests:
1. conditional_distribution_shift_test — KS / energy distance between
   regime-g and regime-not-g generated samples.
2. roundtrip_identifiability — classifier recovers intended regime from
   generated sample statistics.

IMPORTANT: stagflation has only 42 rows (0.67% support). When executing
on real data, stagflation must be labelled "insufficient support —
validation underpowered" rather than reported as a clean pass/fail.
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Tuple


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size between two samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan')
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return float('inf') if np.mean(x) != np.mean(y) else 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance between two 1-d samples (Székely-Rizzo).

    E = 2 * E||X - Y|| - E||X - X'|| - E||Y - Y'||
    """
    x, y = x.ravel(), y.ravel()
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return float('nan')
    xy = 0.0
    for xi in x:
        xy += np.sum(np.abs(xi - y))
    xy *= 2.0 / (n * m)
    xx = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            xx += np.abs(x[i] - x[j])
    xx *= 2.0 / (n * (n - 1)) if n > 1 else 0.0
    yy = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            yy += np.abs(y[i] - y[j])
    yy *= 2.0 / (m * (m - 1)) if m > 1 else 0.0
    return float(xy - xx - yy)


def conditional_distribution_shift_test(
    samples_g: np.ndarray,
    samples_not_g: np.ndarray,
    statistic: str = "realized_vol",
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """KS test + energy distance + Cohen's d for regime-conditional samples.

    Parameters
    ----------
    samples_g : np.ndarray, shape (n_paths, n_steps)
        Generated returns conditioned on regime g.
    samples_not_g : np.ndarray, shape (n_paths, n_steps)
        Generated returns conditioned on NOT-g.
    statistic : str
        Summary statistic to compare ('realized_vol' or 'mean_return').
    n_samples : int
        Number of bootstrap samples for the comparison (subsamples paths).
    rng : np.random.Generator, optional

    Returns
    -------
    dict with keys: ks_stat, ks_pvalue, energy_dist, cohens_d, n_g, n_not_g
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_g = min(len(samples_g), n_samples)
    n_not_g = min(len(samples_not_g), n_samples)
    if n_g < 10 or n_not_g < 10:
        return {'ks_stat': float('nan'), 'ks_pvalue': float('nan'),
                'energy_dist': float('nan'), 'cohens_d': float('nan'),
                'n_g': n_g, 'n_not_g': n_not_g,
                'warning': 'insufficient support — validation underpowered'}
    idx_g = rng.choice(len(samples_g), n_g, replace=False)
    idx_not_g = rng.choice(len(samples_not_g), n_not_g, replace=False)
    if statistic == "realized_vol":
        stat_g = np.std(samples_g[idx_g], axis=1, ddof=1)
        stat_not_g = np.std(samples_not_g[idx_not_g], axis=1, ddof=1)
    elif statistic == "mean_return":
        stat_g = np.mean(samples_g[idx_g], axis=1)
        stat_not_g = np.mean(samples_not_g[idx_not_g], axis=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    ks_stat, ks_pvalue = stats.ks_2samp(stat_g, stat_not_g)
    edist = energy_distance(stat_g, stat_not_g)
    d = cohens_d(stat_g, stat_not_g)
    return {
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'energy_dist': edist,
        'cohens_d': d,
        'n_g': n_g,
        'n_not_g': n_not_g,
    }


def roundtrip_identifiability(
    real_features: np.ndarray,
    real_labels: np.ndarray,
    generated_features: np.ndarray,
    generated_labels: np.ndarray,
    classifier_type: str = "logistic",
    cv_folds: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Train classifier on REAL data, evaluate on GENERATED samples.

    If the classifier recovers the intended regime labels
    significantly above chance, the conditioning is effective.

    Parameters
    ----------
    real_features : np.ndarray, shape (n_real, n_features)
        Real test-set summary statistics per path (e.g. realized_vol, mean_return, kurtosis).
    real_labels : np.ndarray, shape (n_real,)
        True regime labels for real data.
    generated_features : np.ndarray, shape (n_gen, n_features)
        Summary statistics for each generated path.
    generated_labels : np.ndarray, shape (n_gen,)
        Intended regime labels for generated paths.
    classifier_type : str
        'logistic' or 'rf' (RandomForest).
    cv_folds : int
        Cross-validation folds for training.
    rng : np.random.Generator

    Returns
    -------
    dict with: train_accuracy, train_chance, test_accuracy, test_chance,
               n_real, n_gen
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_classes = len(np.unique(real_labels))
    chance = 1.0 / n_classes
    if classifier_type == "logistic":
        clf = LogisticRegression(max_iter=2000, random_state=int(rng.integers(0, 2**31)))
    elif classifier_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=int(rng.integers(0, 2**31)))
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    # Train on real data
    clf.fit(real_features, real_labels)
    train_acc = float(np.mean(clf.predict(real_features) == real_labels))
    # Evaluate on generated
    gen_pred = clf.predict(generated_features)
    test_acc = float(np.mean(gen_pred == generated_labels))
    return {
        'train_accuracy': train_acc,
        'train_chance': chance,
        'test_accuracy': test_acc,
        'test_chance': chance,
        'n_real': len(real_features),
        'n_gen': len(generated_features),
    }


def regime_validation_report(
    samples_by_regime: Dict[str, Dict[str, np.ndarray]],
    regime_dims: List[str],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Dict]:
    """Run the full regimen validation suite over all regime dimensions.

    Parameters
    ----------
    samples_by_regime : dict of dict
        samples_by_regime[dim][label] = np.ndarray shape (n_paths, n_steps)
    regime_dims : list of str
        e.g. ['vol_regime', 'market_regime', 'macro_regime']

    Returns
    -------
    dict of results per regime dimension.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    results = {}
    for dim in regime_dims:
        labels = list(samples_by_regime[dim].keys())
        dim_results = {}
        for label in labels:
            samples_g = samples_by_regime[dim][label]
            samples_not_g = np.concatenate(
                [samples_by_regime[dim][l] for l in labels if l != label],
                axis=0,
            )
            n_g = len(samples_g)
            if n_g < 50:
                dim_results[label] = {
                    'warning': 'insufficient support — validation underpowered',
                    'n_g': n_g,
                    'n_not_g': len(samples_not_g),
                }
                continue
            shift = conditional_distribution_shift_test(
                samples_g, samples_not_g, statistic='realized_vol',
                rng=rng,
            )
            dim_results[label] = shift
        results[dim] = dim_results
    return results