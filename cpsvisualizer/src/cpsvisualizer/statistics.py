"""
Statistical analysis module for CPS-Visualizer.
Provides PCA, correlation analysis, ANOVA, and uncertainty quantification
for LA-ICP-MS surface scan data.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal
from scipy.spatial.distance import pdist, squareform
import warnings


def compute_pca(df_list, df_name_list, n_components=2):
    """Perform PCA on flattened dataset vectors and return transformed
    coordinates, explained variance ratios, and loadings.

    Args:
        df_list: list of DataFrames (each is a 2D scan matrix)
        df_name_list: dataset names
        n_components: number of PC dimensions to retain

    Returns:
        dict with keys: 'coords' (ndarray nx2), 'explained_variance' (list),
        'loadings' (ndarray), 'pca' (fitted PCA object)
    """
    X = np.array([df.values.ravel() for df in df_list])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
    coords = pca.fit_transform(X_scaled)
    return {
        'coords': coords,
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'loadings': pca.components_,
        'pca_object': pca,
        'scaled_data': X_scaled,
        'names': df_name_list,
    }


def compute_pearson_correlation_matrix(df_list, df_name_list):
    """Compute pairwise Pearson correlation between all datasets."""
    X = np.array([df.values.ravel() for df in df_list])
    n = len(df_list)
    corr_mat = np.zeros((n, n))
    pval_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, p = pearsonr(X[i], X[j])
            corr_mat[i, j] = r
            pval_mat[i, j] = p
    corr_df = pd.DataFrame(corr_mat, index=df_name_list, columns=df_name_list)
    pval_df = pd.DataFrame(pval_mat, index=df_name_list, columns=df_name_list)
    return corr_df, pval_df


def compute_spearman_correlation_matrix(df_list, df_name_list):
    """Compute pairwise Spearman rank correlation between all datasets."""
    X = np.array([df.values.ravel() for df in df_list])
    n = len(df_list)
    corr_mat = np.zeros((n, n))
    pval_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, p = spearmanr(X[i], X[j])
            corr_mat[i, j] = r
            pval_mat[i, j] = p
    corr_df = pd.DataFrame(corr_mat, index=df_name_list, columns=df_name_list)
    pval_df = pd.DataFrame(pval_mat, index=df_name_list, columns=df_name_list)
    return corr_df, pval_df


def compute_anova(df_list, df_name_list):
    """One-way ANOVA across all datasets testing whether means differ.

    Uses Kruskal-Wallis H-test (non-parametric) as fallback if normality
    is violated.
    """
    X = [df.values.ravel() for df in df_list]
    f_stat, anova_p = f_oneway(*X)
    h_stat, kruskal_p = kruskal(*X)
    return {
        'anova_f_statistic': float(f_stat),
        'anova_p_value': float(anova_p),
        'kruskal_h_statistic': float(h_stat),
        'kruskal_p_value': float(kruskal_p),
        'names': df_name_list,
    }


def compute_uncertainty(df):
    """Estimate measurement uncertainty statistics for a single dataset.

    Returns mean, std, relative std (RSD%), median, MAD, IQR.
    """
    flat = df.values.ravel()
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat, ddof=1))
    rsd = float(std_val / mean_val * 100) if mean_val != 0 else float('inf')
    median_val = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median_val)))
    q1 = float(np.percentile(flat, 25))
    q3 = float(np.percentile(flat, 75))
    iqr = float(q3 - q1)
    return {
        'mean': mean_val,
        'std': std_val,
        'rsd_percent': rsd,
        'median': median_val,
        'mad': mad,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
    }


def compute_uncertainty_all(df_list, df_name_list):
    """Compute uncertainty statistics for all datasets."""
    results = {}
    for name, df in zip(df_name_list, df_list):
        results[name] = compute_uncertainty(df)
    return results


def compute_descriptive_statistics(df):
    """Compute comprehensive descriptive statistics for a dataset.

    Includes: count, mean, std, min, 25%, 50%, 75%, max, skewness, kurtosis.
    """
    flat = df.values.ravel()
    return {
        'count': len(flat),
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat, ddof=1)),
        'min': float(np.min(flat)),
        'q25': float(np.percentile(flat, 25)),
        'q50': float(np.median(flat)),
        'q75': float(np.percentile(flat, 75)),
        'max': float(np.max(flat)),
        'skewness': float(stats.skew(flat)),
        'kurtosis': float(stats.kurtosis(flat)),
    }


def compute_descriptive_statistics_all(df_list, df_name_list):
    """Compute descriptive statistics for all datasets."""
    results = {}
    for name, df in zip(df_name_list, df_list):
        results[name] = compute_descriptive_statistics(df)
    return results
