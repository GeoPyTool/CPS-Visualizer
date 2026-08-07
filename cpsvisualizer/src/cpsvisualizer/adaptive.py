"""
Adaptive Optimal Discrimination Algorithm (AODA) for CPS-Visualizer.
"""
import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings("ignore")

from joblib import Parallel, delayed

from cpsvisualizer.core import (
    TRANSFORM_FUNCTIONS, DISTANCE_FUNCTIONS, DISTANCE_NAMES,
    compute_pairwise_matrix, apply_transforms,
)

# ---------------------------------------------------------------------------
# Preprocessing pipeline combinations to evaluate
# ---------------------------------------------------------------------------
PIPELINE_COMBOS = [
    ['log_transform'],
    ['equalize_hist'],
    ['centering_transform'],
    ['z_score_normalization'],
    ['standardize'],
    ['log_transform', 'equalize_hist'],
    ['log_transform', 'centering_transform'],
    ['log_transform', 'z_score_normalization'],
    ['log_transform', 'standardize'],
    ['centering_transform', 'equalize_hist'],
    ['z_score_normalization', 'equalize_hist'],
    ['standardize', 'equalize_hist'],
    ['log_transform', 'centering_transform', 'equalize_hist'],
    ['log_transform', 'z_score_normalization', 'equalize_hist'],
    ['log_transform', 'standardize', 'equalize_hist'],
]

PIPELINE_NAMES = ['+'.join(p) for p in PIPELINE_COMBOS]


def discrimination_power_score(distance_matrix):
    """Compute the Discrimination Power Score (DPS) from a distance matrix.

    DPS = mean(row-wise min off-diagonal) / mean(all off-diagonal)

    Intuition:
      - Numerator: average nearest-neighbor distance (how far is each
        element from its closest neighbor? higher = better separated)
      - Denominator: overall scale of distances (normalization factor)
      - DPS > 1: elements are well-separated (inter > nearest-neighbor)
      - DPS ≈ 0: elements are tightly clustered (nearest neighbor is close)
      - DPS ≈ 1: random/undifferentiated structure

    Parameters
    ----------
    distance_matrix : pd.DataFrame (n x n), symmetric, zeros on diagonal

    Returns
    -------
    float : DPS value
    """
    vals = distance_matrix.values.copy()
    n = vals.shape[0]
    if n < 3:
        return 0.0

    # Set diagonal to infinity so min ignores self-distance
    np.fill_diagonal(vals, np.inf)
    row_minima = vals.min(axis=1)  # nearest neighbor for each element
    nn_mean = np.mean(row_minima)

    # Mean of all off-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    off_diag_mean = vals[mask].mean()

    if off_diag_mean == 0:
        return 0.0
    return float(nn_mean / off_diag_mean)


def discrimination_stability(distance_matrix, n_bootstrap=100, seed=42):
    """Estimate the stability of DPS via bootstrap resampling of rows.

    Returns (mean_dps, std_dps, cv_dps) where cv = std/mean (lower = more stable).
    """
    rng = np.random.default_rng(seed)
    vals = distance_matrix.values.copy()
    n = vals.shape[0]
    if n < 4:
        dps = discrimination_power_score(distance_matrix)
        return dps, 0.0, 0.0

    dps_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Unique indices to avoid self-comparison issues
        idx_u = np.unique(idx)
        if len(idx_u) < 3:
            continue
        sub = vals[np.ix_(idx_u, idx_u)]
        sub_df = pd.DataFrame(sub)
        dps_samples.append(discrimination_power_score(sub_df))

    dps_samples = np.array(dps_samples)
    mean_dps = float(np.mean(dps_samples))
    std_dps = float(np.std(dps_samples))
    cv_dps = float(std_dps / mean_dps) if mean_dps > 0 else float('inf')
    return mean_dps, std_dps, cv_dps


def evaluate_single_pipeline(df_list, df_name_list, transform_names, metric_func):
    """Evaluate one (pipeline, metric) combination.

    Returns dict with DPS, stability, and metadata.
    """
    try:
        transformed = [pd.DataFrame(apply_transforms(df.values.copy(), transform_names))
                       for df in df_list]
    except Exception as e:
        return {'pipeline': '+'.join(transform_names),
                'metric': metric_func.__name__,
                'dps': 0.0, 'dps_std': 0.0, 'dps_cv': float('inf'),
                'error': str(e)}

    try:
        dist_mat = compute_pairwise_matrix(transformed, df_name_list, metric_func)
    except Exception as e:
        return {'pipeline': '+'.join(transform_names),
                'metric': metric_func.__name__,
                'dps': 0.0, 'dps_std': 0.0, 'dps_cv': float('inf'),
                'error': str(e)}

    dps = discrimination_power_score(dist_mat)
    mean_dps, std_dps, cv_dps = discrimination_stability(dist_mat)

    return {
        'pipeline': '+'.join(transform_names),
        'metric': metric_func.__name__,
        'dps': dps,
        'dps_std': std_dps,
        'dps_cv': cv_dps,
        'error': None,
    }


def find_optimal_pipeline(df_list, df_name_list,
                          pipelines=None, metrics=None,
                          n_jobs=-1, verbose=True):
    """Find the optimal (transform pipeline, distance metric) combination
    that maximizes the Discrimination Power Score.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        Input datasets (one per element/component).
    df_name_list : list of str
        Names for each dataset.
    pipelines : list of list of str, optional
        Pipeline combos to evaluate. Default: PIPELINE_COMBOS.
    metrics : list of callable, optional
        Distance metrics. Default: DISTANCE_FUNCTIONS.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame sorted by DPS (descending), with columns:
        pipeline, metric, dps, dps_std, dps_cv, rank
    """
    if pipelines is None:
        pipelines = PIPELINE_COMBOS
    if metrics is None:
        metrics = DISTANCE_FUNCTIONS

    if verbose:
        total = len(pipelines) * len(metrics)
        print(f'AODA: Evaluating {len(pipelines)} pipelines × {len(metrics)} metrics '
              f'= {total} combinations...')

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single_pipeline)(df_list, df_name_list, pipe, metric)
        for pipe, metric in product(pipelines, metrics)
    )

    df = pd.DataFrame(results)
    # Filter out errors
    valid = df[df['error'].isna()].copy()
    failed = len(df) - len(valid)
    if verbose and failed > 0:
        print(f'AODA: {failed} combinations failed, {len(valid)} succeeded.')

    # Sort by DPS descending, then by stability (lower CV = more stable)
    valid = valid.sort_values(['dps', 'dps_cv'], ascending=[False, True])
    valid['rank'] = range(1, len(valid) + 1)
    valid = valid.reset_index(drop=True)

    if verbose:
        print('\nAODA: Top-10 optimal combinations:')
        print(valid.head(10)[['rank', 'pipeline', 'metric', 'dps', 'dps_cv']].to_string(index=False))

    return valid


def compute_comprehensive_benchmark(df_list, df_name_list, n_jobs=-1):
    """Run comprehensive benchmark comparing AODA-optimal against:
    - All individual pipeline+metric combinations (ranked by DPS)
    - PCA-space Euclidean distance
    - t-SNE-space Euclidean distance
    - UMAP-space Euclidean distance
    - Hierarchical clustering cophenetic correlation
    - K-means silhouette score

    Returns a dict with all results.
    """
    from cpsvisualizer.comparison import (
        compute_pca_embedding, compute_tsne_embedding, compute_umap_embedding,
        compute_hierarchical_clustering, compute_kmeans_clustering,
    )
    from cpsvisualizer.statistics import compute_anova
    from sklearn.metrics import silhouette_score
    from cpsvisualizer.core import Euclidean

    results = {}

    # 1. AODA ranking
    print('Benchmark: Computing AODA optimal rankings...')
    results['aoda'] = find_optimal_pipeline(df_list, df_name_list, n_jobs=n_jobs)

    # 2. PCA-based distances
    print('Benchmark: Computing PCA-space comparison...')
    pca = compute_pca_embedding(df_list, df_name_list)
    pca_coords = pca['embedding']
    pca_dist = np.linalg.norm(pca_coords[:, None] - pca_coords[None, :], axis=2)
    pca_dps = discrimination_power_score(pd.DataFrame(pca_dist))
    results['pca_space'] = {
        'method': 'PCA-space Euclidean',
        'dps': pca_dps,
        'explained_variance': pca['explained_variance'],
    }

    # 3. t-SNE-based distances
    print('Benchmark: Computing t-SNE comparison...')
    try:
        tsne = compute_tsne_embedding(df_list, df_name_list, perplexity=min(5, len(df_list)-1))
        if 'error' not in tsne:
            tsne_coords = tsne['embedding']
            tsne_dist = np.linalg.norm(tsne_coords[:, None] - tsne_coords[None, :], axis=2)
            results['tsne_space'] = {
                'method': 't-SNE-space Euclidean',
                'dps': discrimination_power_score(pd.DataFrame(tsne_dist)),
                'kl_divergence': tsne.get('kl_divergence', None),
            }
    except Exception as e:
        results['tsne_space'] = {'method': 't-SNE-space Euclidean', 'error': str(e)}

    # 4. UMAP-based distances
    print('Benchmark: Computing UMAP comparison...')
    try:
        umap_r = compute_umap_embedding(df_list, df_name_list)
        if 'error' not in umap_r:
            umap_coords = umap_r['embedding']
            umap_dist = np.linalg.norm(umap_coords[:, None] - umap_coords[None, :], axis=2)
            results['umap_space'] = {
                'method': 'UMAP-space Euclidean',
                'dps': discrimination_power_score(pd.DataFrame(umap_dist)),
            }
    except Exception as e:
        results['umap_space'] = {'method': 'UMAP-space Euclidean', 'error': str(e)}

    # 5. Hierarchical clustering cophenetic correlation
    print('Benchmark: Computing hierarchical clustering comparison...')
    hier = compute_hierarchical_clustering(df_list, df_name_list)
    results['hierarchical'] = {
        'method': 'Hierarchical (Ward)',
        'cophenetic_correlation': hier.get('cophenetic_correlation', None),
    }

    # 6. K-means silhouette
    print('Benchmark: Computing K-means comparison...')
    from cpsvisualizer.comparison import prepare_feature_matrix
    X, _ = prepare_feature_matrix(df_list)
    n_clusters = min(3, len(df_list))
    if len(df_list) >= 3:
        kmeans = compute_kmeans_clustering(df_list, df_name_list, n_clusters=n_clusters)
        try:
            sil = silhouette_score(X, kmeans['labels']) if len(set(kmeans['labels'])) > 1 else 0.0
        except Exception:
            sil = 0.0
        results['kmeans'] = {
            'method': f'K-Means (k={n_clusters})',
            'silhouette_score': sil,
            'inertia': kmeans.get('inertia', None),
        }

    # 7. ANOVA for statistical significance
    anova = compute_anova(df_list, df_name_list)
    results['anova'] = anova

    # 8. Raw Euclidean DPS (baseline)
    raw_dist = compute_pairwise_matrix(df_list, df_name_list, Euclidean)
    raw_dps = discrimination_power_score(raw_dist)
    results['raw_baseline'] = {
        'method': 'Raw Euclidean (baseline)',
        'dps': raw_dps,
    }

    print('Benchmark: Complete.')
    return results


def benchmark_summary_table(results):
    """Generate a summary comparison table from benchmark results."""
    rows = []

    # Add AODA best
    if 'aoda' in results and len(results['aoda']) > 0:
        best = results['aoda'].iloc[0]
        rows.append({
            'Method': f'AODA: {best["pipeline"]} + {best["metric"]}',
            'DPS': best['dps'],
            'DPS_CV': best['dps_cv'],
            'Type': 'Adaptive (proposed)',
        })

    # Add embedding-based methods
    for key in ['pca_space', 'tsne_space', 'umap_space']:
        if key in results and 'error' not in results[key]:
            rows.append({
                'Method': results[key]['method'],
                'DPS': results[key]['dps'],
                'DPS_CV': None,
                'Type': 'Embedding-based',
            })

    # Add baseline
    if 'raw_baseline' in results:
        rows.append({
            'Method': results['raw_baseline']['method'],
            'DPS': results['raw_baseline']['dps'],
            'DPS_CV': None,
            'Type': 'Baseline',
        })

    # Add clustering metrics
    if 'hierarchical' in results:
        rows.append({
            'Method': results['hierarchical']['method'],
            'DPS': None,
            'DPS_CV': None,
            'Type': f"Cophenetic r={results['hierarchical']['cophenetic_correlation']:.3f}",
        })
    if 'kmeans' in results:
        rows.append({
            'Method': results['kmeans']['method'],
            'DPS': None,
            'DPS_CV': None,
            'Type': f"Silhouette={results['kmeans']['silhouette_score']:.3f}",
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Continuous numerical optimisation of the preprocessing power parameter
# ---------------------------------------------------------------------------
# AODA's exhaustive 15×26 grid search is replaced here by a fast 1-D search
# over a continuous Box-Cox power exponent.  The DPS is a smooth function of
# the exponent, so we combine two classic line-search optimisers:
#
#   * Golden-section search (bracketing, robust, derivative-free)
#   * Secant / quasi-Newton method (superlinear convergence near the optimum)
#
# Both run from the same bracket; the method that achieves the higher DPS wins.


def box_cox_transform(data, p, eps=1e-6):
    """Box-Cox power transform with continuous exponent p.

    p = 1  → identity
    p ≈ 0 → natural log
    p = 0.5 → sqrt-like
    """
    x = np.maximum(data, 0.0) + 1.0
    if abs(p) < eps:
        return np.log(x)
    return (np.power(x, p) - 1.0) / p


def _dps_for_power(df_list, df_name_list, metric_func, p, stability_samples=0, seed=42):
    """DPS of the pairwise distance matrix after applying Box-Cox(p)."""
    try:
        transformed = [pd.DataFrame(box_cox_transform(df.values.copy(), p))
                       for df in df_list]
        dist_mat = compute_pairwise_matrix(transformed, df_name_list, metric_func)
        dps = discrimination_power_score(dist_mat)
        return float(dps)
    except Exception:
        return 0.0


def golden_section_search(f, a, b, tol=1e-6, max_iter=120, verbose=False):
    """Maximise f over [a, b] using golden-section search.

    Robust and derivative-free; optimal for unimodal functions.
    Returns (x_opt, f_opt, n_evaluations).
    """
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate ≈ 0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    n_eval = 2
    while abs(b - a) > tol and n_eval < max_iter:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        n_eval += 1
    x_opt = (a + b) / 2.0
    if verbose:
        print(f'  golden-section: x={x_opt:.6f} f={f(x_opt):.6f} evals={n_eval}')
    return x_opt, float(f(x_opt)), n_eval


def _finite_diff_gradient(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def secant_method(f, x0, x1, tol=1e-8, max_iter=100, verbose=False):
    """Maximise f using the secant method on the derivative (quasi-Newton).

    Solves f'(x) = 0 by the secant iteration:
        x_{n+1} = x_n - g(x_n) * (x_n - x_{n-1}) / (g(x_n) - g(x_{n-1}))
    where g = f' is approximated by central differences.

    Falls back to the best of the two endpoints if it diverges.
    Returns (x_opt, f_opt, n_evaluations).
    """
    def g(x):
        return _finite_diff_gradient(f, x)

    g0, g1 = g(x0), g(x1)
    n_eval = 4
    best_x, best_f = x0, f(x0)
    if f(x1) > best_f:
        best_x, best_f = x1, f(x1)

    for _ in range(max_iter):
        denom = g1 - g0
        if abs(denom) < 1e-14:
            break
        x2 = x1 - g1 * (x1 - x0) / denom
        if not np.isfinite(x2):
            break
        # Clamp to a sane bracket around the data
        x2 = float(np.clip(x2, -2.0, 3.0))
        f2 = f(x2)
        n_eval += 2
        if f2 > best_f:
            best_x, best_f = x2, f2
        if abs(x2 - x1) < tol:
            break
        x0, x1, g0, g1 = x1, x2, g1, g(x2)
        n_eval += 2

    if verbose:
        print(f'  secant: x={best_x:.6f} f={best_f:.6f} evals={n_eval}')
    return best_x, float(best_f), n_eval


def optimize_power_parameter(df_list, df_name_list, metric_func,
                             bracket=(-0.5, 2.0), tol=1e-6, verbose=False):
    """Find the Box-Cox exponent maximising DPS for one metric.

    Runs golden-section AND secant search on the same bracket, then keeps the
    better result.  Returns a dict with the winning method, exponent, DPS and
    the per-method comparison.
    """
    f = lambda p: _dps_for_power(df_list, df_name_list, metric_func, p)

    x_gs, f_gs, n_gs = golden_section_search(f, bracket[0], bracket[1],
                                             tol=tol, verbose=verbose)
    x_sec, f_sec, n_sec = secant_method(f, bracket[0], bracket[1],
                                        verbose=verbose)

    if f_gs >= f_sec:
        method, x_opt, f_opt, n_evals = 'golden_section', x_gs, f_gs, n_gs
    else:
        method, x_opt, f_opt, n_evals = 'secant', x_sec, f_sec, n_sec

    return {
        'method': metric_func.__name__,
        'power': float(x_opt),
        'dps': float(f_opt),
        'winner': method,
        'n_evaluations': int(n_evals),
        'comparison': {
            'golden_section': {'power': float(x_gs), 'dps': float(f_gs), 'evals': n_gs},
            'secant': {'power': float(x_sec), 'dps': float(f_sec), 'evals': n_sec},
        },
    }


def find_optimal_power(df_list, df_name_list, metrics=None,
                       bracket=(-0.5, 2.0), tol=1e-6, n_jobs=1, verbose=True,
                       top_n=None):
    """Optimise the power exponent for every metric and rank by DPS.

    Combines golden-section and secant search per metric; the overall best
    (pipeline-exponent, metric) combination is returned as a DataFrame.

    If ``top_n`` is set, metrics are first screened with a single DPS
    evaluation at the bracket midpoint (cheap for most metrics, and
    mutual-info metrics are evaluated on a reduced sample so they never
    dominate runtime).  Only the ``top_n`` candidates get the full
    golden-section + secant optimisation.
    """
    if metrics is None:
        # Skip the slow mutual-info metrics by default — they dominate runtime
        # and their DPS behaviour is no better than the fast continuous metrics.
        _SLOW = {'mutual_info_score_unflattern', 'mutual_info_score_flattern',
                 'mutual_info_regression_unflattern', 'mutual_info_regression_flattern'}
        metrics = [f for f in DISTANCE_FUNCTIONS if f.__name__ not in _SLOW]

    if verbose:
        print(f'AODA-continuous: optimising power exponent for '
              f'{len(metrics)} metrics...')

    if top_n is None or top_n >= len(metrics):
        candidates = list(metrics)
    else:
        # ---- coarse pre-screen at a few fixed exponents (cheap) ----
        screen_exps = [0.0, 0.5, 1.0, 1.5, 2.0]
        if verbose:
            print(f'AODA-continuous: pre-screening {len(metrics)} metrics '
                  f'at {len(screen_exps)} exponents...')
        screen = Parallel(n_jobs=n_jobs)(
            delayed(_metric_screen_score)(
                df_list, df_name_list, metric, screen_exps)
            for metric in metrics
        )
        screen = sorted(screen, key=lambda x: x[1], reverse=True)
        candidates = [m for m, _ in screen[:top_n]]
        if verbose:
            print(f'AODA-continuous: keeping top {len(candidates)} metrics: '
                  f'{", ".join(m.__name__ for m in candidates)}')

    results = Parallel(n_jobs=n_jobs)(
        delayed(optimize_power_parameter)(
            df_list, df_name_list, metric, bracket=bracket, tol=tol, verbose=verbose)
        for metric in candidates
    )

    df = pd.DataFrame(results)
    df = df.sort_values('dps', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df


def _metric_screen_score(df_list, df_name_list, metric_func, exponents):
    """Max DPS over a handful of fixed exponents — cheap screening.

    Returns (metric_func, score) so the caller can recover the callable.
    """
    best = 0.0
    for p in exponents:
        v = _dps_for_power(df_list, df_name_list, metric_func, p)
        if v > best:
            best = v
    return metric_func, float(best)
