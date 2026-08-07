"""
Comparison module for CPS-Visualizer.

Provides comparison with established dimensionality reduction and
clustering methods: PCA, t-SNE, UMAP, hierarchical clustering,
and K-means clustering. Demonstrates the added value of the
CPS-Visualizer framework against existing approaches.

Addresses Reviewer #1's request for comprehensive validation.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform
import warnings


try:
    import umap as umap_module
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def _prepare_feature_matrix(df_list):
    """Convert list of DataFrames into a feature matrix (n_samples x n_features)."""
    X = np.array([df.values.ravel() for df in df_list])
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def compute_pca_embedding(df_list, df_name_list, n_components=2):
    """PCA dimensionality reduction for comparison visualization."""
    X, scaler = _prepare_feature_matrix(df_list)
    pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
    embedding = pca.fit_transform(X)
    return {
        'embedding': embedding,
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'names': df_name_list,
        'method': 'PCA',
    }


def compute_tsne_embedding(df_list, df_name_list, n_components=2,
                           perplexity=30, random_state=42):
    """t-SNE embedding for nonlinear structure discovery."""
    X, _ = _prepare_feature_matrix(df_list)
    n = X.shape[0]
    if n < 2:
        return {'embedding': np.zeros((n, n_components)), 'names': df_name_list,
                'method': 't-SNE', 'error': 'Need at least 2 samples for t-SNE'}
    actual_perplexity = min(perplexity, n - 1)
    tsne = TSNE(n_components=n_components, perplexity=actual_perplexity,
                random_state=random_state, init='pca', learning_rate='auto')
    embedding = tsne.fit_transform(X)
    return {
        'embedding': embedding,
        'kl_divergence': float(tsne.kl_divergence_),
        'names': df_name_list,
        'method': 't-SNE',
    }


def compute_umap_embedding(df_list, df_name_list, n_components=2,
                           n_neighbors=15, min_dist=0.1, random_state=42):
    """UMAP embedding for manifold learning-based comparison."""
    if len(df_list) < 2:
        return {
            'embedding': np.zeros((len(df_list), n_components)),
            'names': df_name_list,
            'method': 'UMAP',
            'error': 'Need at least 2 samples for UMAP',
        }
    if not UMAP_AVAILABLE:
        return {
            'embedding': np.zeros((len(df_list), n_components)),
            'names': df_name_list,
            'method': 'UMAP',
            'error': 'umap-learn package not installed. '
                     'Install with: pip install umap-learn',
        }
    X, _ = _prepare_feature_matrix(df_list)
    reducer = umap_module.UMAP(
        n_components=n_components, n_neighbors=min(n_neighbors, len(df_list) - 1),
        min_dist=min_dist, random_state=random_state
    )
    embedding = reducer.fit_transform(X)
    return {
        'embedding': embedding,
        'names': df_name_list,
        'method': 'UMAP',
    }


def compute_hierarchical_clustering(df_list, df_name_list, method='ward',
                                    metric='euclidean'):
    """Hierarchical clustering with dendrogram linkage.

    Args:
        method: 'ward', 'average', 'complete', 'single'
        metric: distance metric
    """
    X, _ = _prepare_feature_matrix(df_list)
    if len(df_list) < 2:
        return {
            'linkage': None,
            'cophenetic_correlation': 0.0,
            'names': df_name_list,
            'method': f'Hierarchical ({method})',
            'distance_metric': metric,
        }
    Z = linkage(X, method=method, metric=metric)
    c, _ = cophenet(Z, pdist(X))
    return {
        'linkage': Z,
        'cophenetic_correlation': float(c),
        'names': df_name_list,
        'method': f'Hierarchical ({method})',
        'distance_metric': metric,
    }


def compute_kmeans_clustering(df_list, df_name_list, n_clusters=3,
                              random_state=42, n_init=10):
    """K-means clustering for grouping similar element distributions."""
    X, _ = _prepare_feature_matrix(df_list)
    n_clusters = min(max(n_clusters, 1), len(df_list))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                    n_init=n_init)
    labels = kmeans.fit_predict(X)
    return {
        'labels': labels.tolist(),
        'centroids': kmeans.cluster_centers_,
        'inertia': float(kmeans.inertia_),
        'names': df_name_list,
        'n_clusters': n_clusters,
        'method': 'K-Means',
    }


def compute_all_comparisons(df_list, df_name_list):
    """Run all comparison methods and return comprehensive results."""
    results = {}
    results['pca'] = compute_pca_embedding(df_list, df_name_list)
    results['tsne'] = compute_tsne_embedding(df_list, df_name_list)
    results['umap'] = compute_umap_embedding(df_list, df_name_list)
    results['hierarchical'] = compute_hierarchical_clustering(
        df_list, df_name_list
    )
    results['kmeans'] = compute_kmeans_clustering(df_list, df_name_list)
    return results


def compute_method_comparison_metrics(df_list, df_name_list,
                                      distance_func, distance_funcs_dict=None):
    """Compare CPS-Visualizer distance metrics against classical methods.

    Computes a distance matrix using the custom distance function and
    compares it against Euclidean distance, correlation distance, and
    cosine distance of the flattened data.

    Returns a dictionary of distance matrices for comparison.
    """
    from cpsvisualizer.core import compute_pairwise_matrix, Euclidean as core_euc

    comparison = {}
    X = [df.values.ravel() for df in df_list]
    n = len(df_list)

    custom_matrix = compute_pairwise_matrix(df_list, df_name_list, distance_func)
    euclidean_matrix = compute_pairwise_matrix(df_list, df_name_list, core_euc)

    dist_pdist = squareform(pdist(np.array(X), metric='euclidean'))
    dist_corr = squareform(pdist(np.array(X), metric='correlation'))
    dist_cosine = squareform(pdist(np.array(X), metric='cosine'))

    euc_df = pd.DataFrame(dist_pdist, index=df_name_list, columns=df_name_list)
    corr_df = pd.DataFrame(dist_corr, index=df_name_list, columns=df_name_list)
    cos_df = pd.DataFrame(dist_cosine, index=df_name_list, columns=df_name_list)

    comparison['custom'] = custom_matrix
    comparison['euclidean_scipy'] = euc_df
    comparison['correlation_scipy'] = corr_df
    comparison['cosine_scipy'] = cos_df

    return comparison
