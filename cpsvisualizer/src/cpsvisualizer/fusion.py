"""
Coordinate-Aware Visual-Numerical Fusion Enhancement for CPS-Visualizer.

Implements three enhancement strategies operating on the same spatial coordinates:

  Method A — Visual Enhancement:
    Apply image-processing transforms (log, histogram equalisation) to amplify
    patterns that the human eye struggles to discern in raw count matrices.

  Method B — Statistical Enhancement:  
    Compute per-pixel statistical significance scores (z-score, robust z-score
    using median absolute deviation) that highlight pixels whose count values
    deviate meaningfully from the background distribution. Unlike visual
    enhancement, this preserves the exact numerical magnitude of deviations.

  Method C — Coordinate-Fused Enhancement:
    Combine A and B at each (i,j) pixel via a tunable fusion parameter α∈[0,1].
    Three fusion modes are provided: additive, multiplicative, and power-weighted.
    The AODA engine can automatically search for the α that maximises DPS.

Key insight (Reviewer Comment #2): visual enhancement alone may over-emphasise
subjective patterns while neglecting statistically significant but visually
subtle features. By operating both channels on the same coordinate grid and
fusing them, the method provides stronger statistical evidence for observed
differences than either channel alone.

Author: CPS-Visualizer team
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, median_filter
from itertools import product

from cpsvisualizer.core import (
    DISTANCE_FUNCTIONS, log_transform, equalize_hist, standardize,
    compute_pairwise_matrix,
)


# ══════════════════════════════════════════════════════════════════════════════
# Per-pixel statistical significance transforms
# ══════════════════════════════════════════════════════════════════════════════

def global_zscore(data):
    """Global z-score: (x - mean) / std, applied element-wise to the full matrix."""
    mu = np.mean(data)
    sd = np.std(data)
    if sd == 0:
        return np.zeros_like(data)
    return (data - mu) / sd


def robust_zscore(data):
    """Robust z-score using median and MAD (median absolute deviation).
    Resistant to outliers; recommended for LA-ICP-MS data with hot spots."""
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    if mad == 0:
        mad = np.std(data) or 1.0
    return (data - med) / (1.4826 * mad)  # 1.4826 = consistency factor for normal


def local_zscore(data, window=5):
    """Local z-score: (pixel - local_mean) / local_std within a sliding window.
    Highlights features that are anomalous relative to their immediate
    surroundings rather than the global distribution."""
    mu = uniform_filter(data.astype(np.float64), size=window)
    sq = uniform_filter(data.astype(np.float64)**2, size=window)
    sd = np.sqrt(np.maximum(sq - mu**2, 1e-10))
    return (data - mu) / np.maximum(sd, 1e-10)


def local_robust(data, window=5):
    """Local robust score: (pixel - local_median) / local_MAD within a window.
    Most robust to hot-spot artefacts in LA-ICP-MS data."""
    med = median_filter(data, size=window)
    ad = np.abs(data - med.astype(np.float64))
    mad = median_filter(ad, size=window)
    mad = np.maximum(mad, 1e-10)
    return (data - med.astype(np.float64)) / (1.4826 * mad)


# Registry of statistical transforms
STAT_ENHANCE = {
    'zscore':        global_zscore,
    'robust_zscore': robust_zscore,
    'local_zscore':  local_zscore,
    'local_robust':  local_robust,
}

STAT_ENHANCE_NAMES = list(STAT_ENHANCE.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Fusion functions: combine visual (V) and statistical (S) at each pixel
# ══════════════════════════════════════════════════════════════════════════════

def fuse_additive(V, S, alpha=0.5):
    """Linear interpolation: F = α·V + (1-α)·S."""
    return alpha * V + (1.0 - alpha) * S


def fuse_multiplicative(V, S, alpha=0.5):
    """Geometric mean style: F = V^α · S^(1-α).  
    Assumes both V and S are positive; clips to avoid domain errors."""
    eps = 1e-10
    Vp = np.maximum(V, eps)
    Sp = np.maximum(S, eps)
    return (Vp ** alpha) * (Sp ** (1.0 - alpha))


def fuse_exponential(V, S, alpha=0.5):
    """Exponential weighting: F = exp(α·log(V+ε) + (1-α)·log(S+ε)).
    Numerically equivalent to multiplicative but handles zeros gracefully."""
    eps = 1e-10
    return np.exp(alpha * np.log(np.maximum(V, eps)) +
                  (1.0 - alpha) * np.log(np.maximum(S, eps)))


FUSION_FUNCTIONS = {
    'add':         fuse_additive,
    'multiply':    fuse_multiplicative,
    'exp':         fuse_exponential,
}

FUSION_NAMES = list(FUSION_FUNCTIONS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation: compare Method A, B, C on a dataset
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_triple_enhancement(
    df_list,
    df_name_list,
    stat_method='robust_zscore',
    fusion_mode='multiply',
    alpha=0.5,
    metric_funcs=None,
    n_jobs=1,
):
    """Evaluate Method A (visual), B (statistical), and C (fused) on the same
    dataset and return their discrimination power scores.

    Parameters
    ----------
    df_list : list of pd.DataFrame
    df_name_list : list of str
    stat_method : str, one of {'zscore', 'robust_zscore', 'local_zscore', 'local_robust'}
    fusion_mode : str, one of {'add', 'multiply', 'exp'}
    alpha : float in [0, 1]
    metric_funcs : list of callable, optional

    Returns
    -------
    dict with keys 'visual', 'statistical', 'fused' containing DPS dataframes
    """
    from cpsvisualizer.adaptive import discrimination_power_score

    if metric_funcs is None:
        metric_funcs = DISTANCE_FUNCTIONS

    # Method A: visual enhancement (log + equalize)
    visual_data = []
    for df in df_list:
        arr = df.values.copy()
        arr = log_transform(arr)
        arr = equalize_hist(arr)
        visual_data.append(pd.DataFrame(arr))

    # Method B: statistical enhancement
    stat_func = STAT_ENHANCE[stat_method]
    stat_data = []
    for df in df_list:
        arr = stat_func(df.values.copy().astype(np.float64))
        stat_data.append(pd.DataFrame(arr))

    # Method C: fused (coordinate-aware combination)
    fusion_func = FUSION_FUNCTIONS[fusion_mode]
    fused_data = []
    for vdf, sdf in zip(visual_data, stat_data):
        V = vdf.values
        S = sdf.values
        # Normalise S to [0, 1] range before fusion so both channels are comparable
        S_norm = (S - S.min()) / (S.max() - S.min() + 1e-10)
        F = fusion_func(V, S_norm, alpha)
        fused_data.append(pd.DataFrame(F))

    results = {}
    for label, data in [('visual', visual_data), ('statistical', stat_data),
                         ('fused', fused_data)]:
        dps_rows = []
        for metric in metric_funcs:
            mat = compute_pairwise_matrix(data, df_name_list, metric)
            dps = discrimination_power_score(mat)
            dps_rows.append({
                'method': label,
                'metric': metric.__name__,
                'dps': dps,
            })
        results[label] = pd.DataFrame(dps_rows).sort_values('dps', ascending=False)

    return results


def optimise_alpha(df_list, df_name_list, stat_method='robust_zscore',
                   fusion_mode='multiply', alphas=None, n_jobs=1):
    """Find the optimal α parameter for visual-statistical fusion.

    Searches α ∈ [0, 1] for the value that maximises the top-1 DPS.
    """
    if alphas is None:
        alphas = np.linspace(0, 1, 21)  # 0.0, 0.05, ..., 1.0

    best_alpha = 0.5
    best_dps = -1.0
    results = []

    for a in alphas:
        ev = evaluate_triple_enhancement(
            df_list, df_name_list,
            stat_method=stat_method, fusion_mode=fusion_mode, alpha=a,
            n_jobs=n_jobs,
        )
        top_dps = ev['fused']['dps'].iloc[0]
        results.append({'alpha': a, 'dps': top_dps})
        if top_dps > best_dps:
            best_dps = top_dps
            best_alpha = a

    return best_alpha, best_dps, pd.DataFrame(results)
