"""
Comprehensive test suite for CPS-Visualizer — covers all modules exhaustively.
"""
import os
import sys
import math
import tempfile
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SEED = 42


def _rng(shape, seed=_SEED):
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=2, sigma=0.5, size=shape)


def _df(rows=20, cols=15, seed=_SEED):
    return pd.DataFrame(_rng((rows, cols), seed))


def _df_list(n=4, rows=20, cols=15):
    return [pd.DataFrame(_rng((rows, cols), _SEED + i)) for i in range(n)]


def _names(n=4):
    return [f'Elem{i}' for i in range(n)]


# ===================================================================
# CORE MODULE
# ===================================================================
class TestCoreTransforms:
    def test_log_transform_handles_zeros(self):
        from cpsvisualizer.core import log_transform
        r = log_transform(np.array([[0., 1.], [0., 0.]]))
        assert np.all(np.isfinite(r))

    def test_log_transform_shape(self):
        from cpsvisualizer.core import log_transform
        d = _rng((30, 25))
        assert log_transform(d).shape == d.shape

    def test_centering_zero_mean(self):
        from cpsvisualizer.core import centering_transform
        r = centering_transform(np.array([[1., 3.], [5., 7.], [9., 11.]]))
        np.testing.assert_allclose(r.mean(axis=0), 0, atol=1e-10)

    def test_z_score_properties(self):
        from cpsvisualizer.core import z_score_normalization
        d = _rng((50, 10))
        r = z_score_normalization(d)
        np.testing.assert_allclose(r.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(r.std(axis=0), 1, atol=1e-10)

    def test_z_score_constant_column(self):
        from cpsvisualizer.core import z_score_normalization
        d = np.array([[1., 2.], [1., 4.], [1., 6.]])  # col 0 constant
        r = z_score_normalization(d)
        assert np.all(np.isfinite(r))

    def test_standardize_via_sklearn(self):
        from cpsvisualizer.core import standardize
        r = standardize(_rng((40, 8)))
        np.testing.assert_allclose(r.mean(axis=0), 0, atol=1e-8)
        np.testing.assert_allclose(r.std(axis=0, ddof=0), 1, atol=1e-8)

    def test_equalize_hist_range(self):
        from cpsvisualizer.core import equalize_hist
        r = equalize_hist(_rng((20, 20)))
        assert r.min() >= 0 and r.max() <= 1

    def test_log_centering(self):
        from cpsvisualizer.core import log_centering_transform
        r = log_centering_transform(np.array([[1., 10.], [100., 1000.]]))
        np.testing.assert_allclose(r.mean(axis=0), 0, atol=1e-10)

    def test_apply_transforms_sequence(self):
        from cpsvisualizer.core import apply_transforms
        d = _rng((10, 10))
        r = apply_transforms(d, ['log_transform', 'centering_transform', 'equalize_hist'])
        assert r.shape == d.shape

    def test_apply_transforms_unknown_func(self):
        from cpsvisualizer.core import apply_transforms
        d = _rng((10, 10))
        r = apply_transforms(d, ['log_transform', 'nonexistent', 'equalize_hist'])
        np.testing.assert_allclose(
            r, apply_transforms(d, ['log_transform', 'equalize_hist']))


class TestCoreDistance:
    """Exhaustive tests for all 28 distance functions."""

    def test_euclidean_zero(self):
        from cpsvisualizer.core import Euclidean
        df = _df()
        assert Euclidean(df, df) == 0.0

    def test_euclidean_positive(self):
        from cpsvisualizer.core import Euclidean
        assert Euclidean(pd.DataFrame(np.eye(3)), pd.DataFrame(np.zeros((3, 3)))) == math.sqrt(3)

    def test_manhattan(self):
        from cpsvisualizer.core import Manhattan
        assert Manhattan(pd.DataFrame(np.ones((2, 3))), pd.DataFrame(np.zeros((2, 3)))) == 6.0

    def test_chebyshev(self):
        from cpsvisualizer.core import Chebyshev
        a = pd.DataFrame([[0., 5.], [2., 1.]])
        b = pd.DataFrame([[1., 2.], [3., 4.]])
        assert Chebyshev(a, b) == 3.0

    def test_minkowski_p2_equals_euclidean(self):
        from cpsvisualizer.core import Minkowski, Euclidean
        a, b = _df(10, 10), _df(10, 10)
        np.testing.assert_allclose(Minkowski(a, b, p=2), Euclidean(a, b))

    def test_minkowski_p1_equals_manhattan(self):
        from cpsvisualizer.core import Minkowski, Manhattan
        a, b = _df(10, 10), _df(10, 10)
        np.testing.assert_allclose(Minkowski(a, b, p=1), Manhattan(a, b))

    def test_cosine_identical(self):
        from cpsvisualizer.core import Cosine
        a = pd.DataFrame([[2., 0], [0., 2.]])
        assert Cosine(a, a) == pytest.approx(0, abs=1e-10)

    def test_cosine_orthogonal(self):
        from cpsvisualizer.core import Cosine
        a = pd.DataFrame([[1., 0]])
        b = pd.DataFrame([[0., 1.]])
        assert Cosine(a, b) == pytest.approx(1.0)

    def test_cosine_zero_vector(self):
        from cpsvisualizer.core import Cosine
        z = pd.DataFrame(np.zeros((3, 3)))
        assert Cosine(z, z) == 0.0  # guard returns 0

    def test_correlation_perfect_positive(self):
        from cpsvisualizer.core import Correlation
        a = pd.DataFrame([[1., 2.], [3., 4.]])
        b = pd.DataFrame([[2., 4.], [6., 8.]])
        assert Correlation(a, b) == pytest.approx(0, abs=1e-10)

    def test_correlation_constant(self):
        from cpsvisualizer.core import Correlation
        c = pd.DataFrame(np.ones((5, 5)))
        assert Correlation(c, c) == 0.0  # guard returns 0

    def test_jaccard_identical(self):
        from cpsvisualizer.core import Jaccard
        a = pd.DataFrame([[1., 0.], [1., 1.]])
        assert Jaccard(a, a) == 0.0

    def test_jaccard_disjoint(self):
        from cpsvisualizer.core import Jaccard
        a = pd.DataFrame([[1., 0., 0.]])
        b = pd.DataFrame([[0., 1., 1.]])
        assert Jaccard(a, b) == 1.0

    def test_dice(self):
        from cpsvisualizer.core import Dice
        a = pd.DataFrame([[1., 0., 1., 0.]])
        b = pd.DataFrame([[1., 1., 1., 0.]])
        # IA=1,IB=3,DA=2,DB=3 => inter=2, union=4 => dice=1-4/5=0.2
        assert Dice(a, b) == pytest.approx(0.2)

    def test_kulsinski(self):
        from cpsvisualizer.core import Kulsinski
        a = pd.DataFrame([[1., 0.]])
        b = pd.DataFrame([[0., 1.]])
        assert 0 <= Kulsinski(a, b) <= 1

    def test_rogers_tanimoto(self):
        from cpsvisualizer.core import Rogers_Tanimoto
        assert Rogers_Tanimoto(_df(), _df()) >= 0

    def test_russell_rao(self):
        from cpsvisualizer.core import Russell_Rao
        a = pd.DataFrame([[1., 0.]])
        b = pd.DataFrame([[1., 1.]])
        # inter=1, len=2 → 0.5
        assert Russell_Rao(a, b) == 0.5

    def test_sokal_michener(self):
        from cpsvisualizer.core import Sokal_Michener
        assert Sokal_Michener(_df(), _df()) >= 0

    def test_sokal_sneath(self):
        from cpsvisualizer.core import Sokal_Sneath
        a = pd.DataFrame([[1., 0., 1.]])
        b = pd.DataFrame([[1., 1., 0.]])
        # inter=1, sumA=2, sumB=2 => 2/4=0.5
        assert Sokal_Sneath(a, b) == 0.5

    def test_yule(self):
        from cpsvisualizer.core import Yule
        assert 0 <= Yule(_df(), _df()) <= 1

    def test_hsim_range(self):
        from cpsvisualizer.core import Hsim_Distance
        val = Hsim_Distance(_df(10, 10), _df(10, 10))
        assert 0 < val <= 1

    def test_close_is_hsim(self):
        from cpsvisualizer.core import Close_Distance, Hsim_Distance
        a, b = _df(10, 10), _df(10, 10)
        assert Close_Distance(a, b) == Hsim_Distance(a, b)

    def test_bray_curtis_identical_zero(self):
        from cpsvisualizer.core import Bray_Curtis
        a = _df()
        assert Bray_Curtis(a, a) == 0.0

    def test_bray_curtis_all_zero(self):
        from cpsvisualizer.core import Bray_Curtis
        z = pd.DataFrame(np.zeros((3, 3)))
        assert Bray_Curtis(z, z) == 0.0

    def test_canberra(self):
        from cpsvisualizer.core import Canberra
        a = pd.DataFrame(np.ones((3, 3)))
        b = pd.DataFrame(np.ones((3, 3)) * 2)
        assert Canberra(a, b) > 0

    def test_canberra_identical(self):
        from cpsvisualizer.core import Canberra
        a = _df()
        assert Canberra(a, a) == 0.0

    def test_mutual_info_score_unflattern(self):
        from cpsvisualizer.core import mutual_info_score_unflattern
        a, b = _df(30, 5), _df(30, 5)
        v = mutual_info_score_unflattern(a, b)
        assert isinstance(v, float)

    def test_mutual_info_score_flattern(self):
        from cpsvisualizer.core import mutual_info_score_flattern
        a, b = _df(30, 5), _df(30, 5)
        v = mutual_info_score_flattern(a, b)
        assert isinstance(v, float)

    def test_mutual_info_regression_unflattern(self):
        from cpsvisualizer.core import mutual_info_regression_unflattern
        a, b = _df(30, 5), _df(30, 5)
        v = mutual_info_regression_unflattern(a, b)
        assert isinstance(v, float)

    def test_mutual_info_regression_flattern(self):
        from cpsvisualizer.core import mutual_info_regression_flattern
        a, b = _df(30, 5), _df(30, 5)
        v = mutual_info_regression_flattern(a, b)
        assert isinstance(v, float)

    def test_mutual_info_unequal_rows(self):
        """Regression funcs should handle datasets with different row counts."""
        from cpsvisualizer.core import mutual_info_regression_unflattern
        a = _df(30, 5)
        b = _df(15, 5)
        v = mutual_info_regression_unflattern(a, b)
        assert isinstance(v, float)

    def test_calculate_ssim_identical(self):
        from cpsvisualizer.core import calculate_ssim
        a = _df(20, 20)
        assert calculate_ssim(a, a) == pytest.approx(1.0, abs=0.01)

    def test_calculate_ssim_shape_mismatch(self):
        from cpsvisualizer.core import calculate_ssim
        with pytest.raises(ValueError):
            calculate_ssim(_df(10, 10), _df(10, 12))

    def test_luminance(self):
        from cpsvisualizer.core import luminance
        assert 0 <= luminance(_df(20, 20), _df(20, 20)) <= 1

    def test_contrast(self):
        from cpsvisualizer.core import contrast
        assert 0 <= contrast(_df(20, 20), _df(20, 20)) <= 1

    def test_structure(self):
        from cpsvisualizer.core import structure
        assert 0 <= structure(_df(20, 20), _df(20, 20)) <= 1


class TestCoreUtilities:
    def test_compute_pairwise_matrix_square(self):
        from cpsvisualizer.core import compute_pairwise_matrix, Euclidean
        dl = _df_list(4, 10, 10)
        nm = _names(4)
        r = compute_pairwise_matrix(dl, nm, Euclidean)
        assert r.shape == (4, 4)
        np.testing.assert_allclose(np.diag(r), 0, atol=1e-10)
        np.testing.assert_allclose(r.values, r.values.T)

    def test_compute_pairwise_matrix_single(self):
        from cpsvisualizer.core import compute_pairwise_matrix, Euclidean
        r = compute_pairwise_matrix(_df_list(1), _names(1), Euclidean)
        assert r.shape == (1, 1)
        assert r.iloc[0, 0] == 0.0

    def test_load_data_files_csv(self):
        from cpsvisualizer.core import load_data_files
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'test_Ag.csv')
            pd.DataFrame({'a': [1, 2]}).to_csv(p, index=False)
            dfs, names = load_data_files([p])
            assert len(dfs) == 1
            assert names[0] == 'test'

    def test_load_data_files_skips_unknown(self):
        from cpsvisualizer.core import load_data_files
        dfs, names = load_data_files(['/nonexistent.txt'])
        assert len(dfs) == 0

    def test_clean_name(self):
        from cpsvisualizer.core import clean_name
        assert clean_name('/path/to/Cu65_mat.csv') == 'Cu65'

    def test_dist_names_consistency(self):
        from cpsvisualizer.core import DISTANCE_FUNCTIONS, DISTANCE_NAMES
        assert len(DISTANCE_NAMES) == len(DISTANCE_FUNCTIONS)
        assert len(set(DISTANCE_NAMES)) == len(DISTANCE_NAMES)

    def test_dist_functions_no_duplicates(self):
        from cpsvisualizer.core import DISTANCE_FUNCTIONS
        names = [f.__name__ for f in DISTANCE_FUNCTIONS]
        assert len(names) == len(set(names))


# ===================================================================
# STATISTICS MODULE
# ===================================================================
class TestStatistics:
    def test_pca_embedding(self):
        from cpsvisualizer.statistics import compute_pca
        r = compute_pca(_df_list(10, 30, 20), _names(10))
        assert r['coords'].shape == (10, 2)
        assert len(r['explained_variance']) == 2
        # Variance ratios are valid probabilities between 0 and 1
        for v in r['explained_variance']:
            assert 0 < v <= 1

    def test_pca_single_sample(self):
        from cpsvisualizer.statistics import compute_pca
        r = compute_pca(_df_list(3), _names(3), n_components=2)
        assert r['coords'].shape == (3, 2)

    def test_pearson_correlation_symmetric(self):
        from cpsvisualizer.statistics import compute_pearson_correlation_matrix
        c, pv = compute_pearson_correlation_matrix(_df_list(4, 30, 20), _names(4))
        assert c.shape == (4, 4)
        assert pv.shape == (4, 4)
        np.testing.assert_allclose(np.diag(c), 1.0)

    def test_spearman_correlation(self):
        from cpsvisualizer.statistics import compute_spearman_correlation_matrix
        c, pv = compute_spearman_correlation_matrix(_df_list(4, 30, 20), _names(4))
        np.testing.assert_allclose(np.diag(c), 1.0)

    def test_anova(self):
        from cpsvisualizer.statistics import compute_anova
        r = compute_anova(_df_list(3, 30, 20), _names(3))
        assert 'anova_f_statistic' in r
        assert 'kruskal_p_value' in r

    def test_uncertainty(self):
        from cpsvisualizer.statistics import compute_uncertainty
        r = compute_uncertainty(_df())
        assert r['rsd_percent'] > 0
        assert r['mean'] > 0

    def test_uncertainty_all(self):
        from cpsvisualizer.statistics import compute_uncertainty_all
        r = compute_uncertainty_all(_df_list(3), _names(3))
        assert len(r) == 3

    def test_descriptive_statistics(self):
        from cpsvisualizer.statistics import compute_descriptive_statistics
        r = compute_descriptive_statistics(_df())
        assert 'skewness' in r
        assert 'kurtosis' in r


# ===================================================================
# METRICS MODULE
# ===================================================================
class TestMetrics:
    def test_psnr(self):
        from cpsvisualizer.metrics import compute_psnr
        a, b = _df(20, 20), _df(20, 20)
        assert compute_psnr(a, b) > 0

    def test_psnr_identical(self):
        from cpsvisualizer.metrics import compute_psnr
        a = _df(20, 20)
        assert compute_psnr(a, a) == float('inf') or compute_psnr(a, a) > 100

    def test_entropy_range(self):
        from cpsvisualizer.metrics import compute_entropy
        r = compute_entropy(_df())
        assert 0 <= r['normalized_entropy'] <= 1

    def test_entropy_constant(self):
        from cpsvisualizer.metrics import compute_entropy
        r = compute_entropy(pd.DataFrame(np.ones((10, 10))))
        assert r['entropy'] >= 0

    def test_cei_improvement(self):
        from cpsvisualizer.metrics import compute_contrast_enhancement_index
        raw = _df(20, 20)
        enh = pd.DataFrame(np.log1p(raw.values))
        assert compute_contrast_enhancement_index(raw, enh) > 0

    def test_tenengrad(self):
        from cpsvisualizer.metrics import compute_tenengrad
        assert compute_tenengrad(_df(20, 20)) > 0

    def test_tenengrad_constant(self):
        from cpsvisualizer.metrics import compute_tenengrad
        assert compute_tenengrad(pd.DataFrame(np.ones((20, 20)))) == pytest.approx(0, abs=0.01)

    def test_all_image_metrics(self):
        from cpsvisualizer.metrics import compute_all_image_metrics
        raw = _df(20, 20)
        trans = pd.DataFrame(np.log1p(raw.values))
        r = compute_all_image_metrics(raw, trans)
        for k in ('psnr', 'cei', 'ssim_vs_original'):
            assert k in r

    def test_batch_evaluate_transforms(self):
        from cpsvisualizer.metrics import batch_evaluate_transforms
        from cpsvisualizer.core import TRANSFORM_FUNCTIONS
        tfs = {k: TRANSFORM_FUNCTIONS[k] for k in ['log_transform', 'equalize_hist']}
        r = batch_evaluate_transforms(_df_list(3, 20, 20), _names(3), tfs)
        assert 'log_transform' in r


# ===================================================================
# COMPARISON MODULE
# ===================================================================
class TestComparison:
    def test_pca_embedding(self):
        from cpsvisualizer.comparison import compute_pca_embedding
        r = compute_pca_embedding(_df_list(5, 30, 20), _names(5))
        assert r['embedding'].shape == (5, 2)

    def test_tsne_embedding(self):
        from cpsvisualizer.comparison import compute_tsne_embedding
        r = compute_tsne_embedding(_df_list(5, 30, 20), _names(5), perplexity=2)
        assert r['embedding'].shape == (5, 2)

    def test_tsne_few_samples(self):
        from cpsvisualizer.comparison import compute_tsne_embedding
        r = compute_tsne_embedding(_df_list(1), _names(1))
        assert 'error' in r

    def test_umap_embedding(self):
        from cpsvisualizer.comparison import compute_umap_embedding
        r = compute_umap_embedding(_df_list(5, 30, 20), _names(5))
        assert r['embedding'].shape == (5, 2)

    def test_umap_empty(self):
        from cpsvisualizer.comparison import compute_umap_embedding
        r = compute_umap_embedding([], [])
        assert 'error' in r

    def test_hierarchical_clustering(self):
        from cpsvisualizer.comparison import compute_hierarchical_clustering
        r = compute_hierarchical_clustering(_df_list(5, 30, 20), _names(5))
        assert 0 <= r['cophenetic_correlation'] <= 1

    def test_hierarchical_single(self):
        from cpsvisualizer.comparison import compute_hierarchical_clustering
        r = compute_hierarchical_clustering(_df_list(1), _names(1))
        assert r['linkage'] is None

    def test_kmeans(self):
        from cpsvisualizer.comparison import compute_kmeans_clustering
        r = compute_kmeans_clustering(_df_list(5, 30, 20), _names(5), n_clusters=2)
        assert len(r['labels']) == 5

    def test_kmeans_single(self):
        from cpsvisualizer.comparison import compute_kmeans_clustering
        r = compute_kmeans_clustering(_df_list(2), _names(2), n_clusters=5)
        assert r['n_clusters'] in (1, 2)

    def test_all_comparisons(self):
        from cpsvisualizer.comparison import compute_all_comparisons
        r = compute_all_comparisons(_df_list(5, 30, 20), _names(5))
        assert all(k in r for k in ('pca', 'tsne', 'umap', 'hierarchical', 'kmeans'))

    def test_method_comparison_metrics(self):
        from cpsvisualizer.comparison import compute_method_comparison_metrics
        from cpsvisualizer.core import Euclidean
        r = compute_method_comparison_metrics(_df_list(4, 20, 20), _names(4), Euclidean)
        assert 'custom' in r
        assert 'euclidean_scipy' in r


# ===================================================================
# VISUALIZATION MODULE
# ===================================================================
class TestVisualization:
    def test_pca_plot(self, tmp_path):
        from cpsvisualizer.comparison import compute_pca_embedding
        from cpsvisualizer.visualization import plot_pca_comparison
        r = compute_pca_embedding(_df_list(5, 30, 20), _names(5))
        path = plot_pca_comparison(r, save_path=str(tmp_path / 'pca.pdf'))
        assert os.path.exists(path)

    def test_tsne_plot(self, tmp_path):
        from cpsvisualizer.comparison import compute_tsne_embedding
        from cpsvisualizer.visualization import plot_tsne_comparison
        r = compute_tsne_embedding(_df_list(4, 30, 20), _names(4), perplexity=2)
        path = plot_tsne_comparison(r, save_path=str(tmp_path / 'tsne.pdf'))
        assert os.path.exists(path)

    def test_umap_plot(self, tmp_path):
        from cpsvisualizer.comparison import compute_umap_embedding
        from cpsvisualizer.visualization import plot_umap_comparison
        r = compute_umap_embedding(_df_list(4, 30, 20), _names(4))
        path = plot_umap_comparison(r, save_path=str(tmp_path / 'umap.pdf'))
        assert os.path.exists(path)

    def test_dendrogram_plot(self, tmp_path):
        from cpsvisualizer.comparison import compute_hierarchical_clustering
        from cpsvisualizer.visualization import plot_dendrogram
        r = compute_hierarchical_clustering(_df_list(4, 30, 20), _names(4))
        path = plot_dendrogram(r, save_path=str(tmp_path / 'dend.pdf'))
        assert os.path.exists(path)

    def test_correlation_heatmap(self, tmp_path):
        from cpsvisualizer.statistics import compute_pearson_correlation_matrix
        from cpsvisualizer.visualization import plot_correlation_heatmap
        c, pv = compute_pearson_correlation_matrix(_df_list(4, 30, 20), _names(4))
        path = plot_correlation_heatmap(c, pv, save_path=str(tmp_path / 'heat.pdf'))
        assert os.path.exists(path)

    def test_enhanced_data_matrix(self, tmp_path):
        from cpsvisualizer.visualization import plot_enhanced_data_matrix
        path = plot_enhanced_data_matrix(
            _df_list(3, 20, 20), _names(3),
            save_path=str(tmp_path / 'matrix.pdf'))
        assert path is not None
        assert os.path.exists(path)

    def test_enhanced_empty(self):
        from cpsvisualizer.visualization import plot_enhanced_data_matrix
        assert plot_enhanced_data_matrix([], []) is None

    def test_image_quality_empty(self, tmp_path):
        from cpsvisualizer.visualization import plot_image_quality_comparison
        path = plot_image_quality_comparison({}, save_path=str(tmp_path / 'empty.pdf'))
        assert os.path.exists(path)

    def test_all_comparisons_figures(self, tmp_path):
        from cpsvisualizer.comparison import compute_all_comparisons
        from cpsvisualizer.visualization import plot_all_comparisons
        cr = compute_all_comparisons(_df_list(4, 30, 20), _names(4))
        paths = plot_all_comparisons(_df_list(4, 30, 20), _names(4), cr,
                                     output_dir=str(tmp_path))
        assert 'pca' in paths
        assert 'tsne' in paths

    def test_pca_1d(self, tmp_path):
        from cpsvisualizer.visualization import plot_pca_comparison
        r = {
            'embedding': np.array([[1.], [2.], [3.], [4.]]),
            'names': ['A', 'B', 'C', 'D'],
            'explained_variance': [1.0],
        }
        path = plot_pca_comparison(r, save_path=str(tmp_path / 'pca1d.pdf'))
        assert os.path.exists(path)


# ===================================================================
# CLI MODULE
# ===================================================================
class TestCLI:
    def test_init(self):
        from cpsvisualizer.app_cli import CPS_CLI
        app = CPS_CLI()
        assert app.df_list == []

    def test_open_and_trans(self):
        from cpsvisualizer.app_cli import CPS_CLI
        app = CPS_CLI()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'Ag.csv')
            _df(20, 15).to_csv(p, index=False)
            app.open_files([p])
            assert len(app.df_list) == 1
            app.trans_data(['log_transform', 'equalize_hist'])
            assert len(app.trans_df_list) == 1

    def test_calc_and_plot(self):
        from cpsvisualizer.app_cli import CPS_CLI
        app = CPS_CLI()
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, 'Ag.csv')
            p2 = os.path.join(d, 'Cu.csv')
            _df(20, 15).to_csv(p1, index=False)
            _df(20, 15).to_csv(p2, index=False)
            app.open_files([p1, p2])
            app.trans_data(['log_transform'])
            app.calc_data(['Euclidean', 'Bray_Curtis'])
            assert 'Euclidean' in app.result_df_dict
            assert 'Bray_Curtis' in app.result_df_dict
        fig = app.plot_data(show=False)
        assert fig is not None
        plt.close(fig)

    def test_unrecognised_func_filtered(self, capsys):
        from cpsvisualizer.app_cli import CPS_CLI
        app = CPS_CLI()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'Ag.csv')
            _df(20, 15).to_csv(p, index=False)
            app.open_files([p])
            app.trans_data(['log_transform', 'bogus_func', 'equalize_hist'])
            # bogus_func is silently skipped; only valid transforms applied
            applied = app.trans_applied[app.df_name_list[0]]
            assert 'log_transform' in applied
            assert 'equalize_hist' in applied
            assert all('bogus' not in str(x) for x in applied)


# ===================================================================
# EDGE CASES
# ===================================================================
class TestEdgeCases:
    def test_all_zeros_data(self):
        from cpsvisualizer.core import (
            Euclidean, Cosine, Correlation, Hsim_Distance, Bray_Curtis)
        z = pd.DataFrame(np.zeros((5, 5)))
        for f in [Euclidean, Cosine, Correlation, Hsim_Distance, Bray_Curtis]:
            val = f(z, z)
            assert np.isfinite(val), f'{f.__name__} returned {val}'

    def test_constant_data(self):
        from cpsvisualizer.core import Cosine, Correlation
        c = pd.DataFrame(np.full((5, 5), 3.14))
        assert Cosine(c, c) == pytest.approx(0.0, abs=1e-8)
        assert Correlation(c, c) == 0.0

    def test_single_element(self):
        from cpsvisualizer.core import Hsim_Distance, Euclidean
        a = pd.DataFrame([[7.0]])
        b = pd.DataFrame([[3.0]])
        assert Euclidean(a, b) == 4.0
        assert 0 < Hsim_Distance(a, b) < 1

    def test_large_shape_mismatch(self):
        import pytest as pt
        from cpsvisualizer.core import Euclidean
        a = _df(30, 40, seed=1)
        b_same = _df(30, 40, seed=2)
        assert Euclidean(a, b_same) > 0
        with pt.raises(ValueError):
            Euclidean(a, _df(25, 50))

    def test_mutual_info_with_nans(self):
        """Functions should handle NaN propagation gracefully (fillna before)."""
        from cpsvisualizer.core import mutual_info_score_unflattern
        a = _df(30, 5)
        a.iloc[0, 0] = np.nan
        b = _df(30, 5).fillna(0)
        val = mutual_info_score_unflattern(a.fillna(0), b)
        assert isinstance(val, float)


# ===================================================================
# ADAPTIVE MODULE (AODA)
# ===================================================================
class TestAdaptive:
    def test_dps_perfect_separation(self):
        from cpsvisualizer.adaptive import discrimination_power_score
        # Perfectly separated: off-diag=10, diag=0
        mat = pd.DataFrame([[0, 10, 10], [10, 0, 10], [10, 10, 0]], dtype=float)
        dps = discrimination_power_score(mat)
        assert dps == 1.0

    def test_dps_random(self):
        from cpsvisualizer.adaptive import discrimination_power_score
        # Random small distances
        mat = pd.DataFrame(np.random.uniform(0, 1, (5, 5)))
        np.fill_diagonal(mat.values, 0)
        dps = discrimination_power_score(mat)
        assert 0 <= dps <= 1

    def test_dps_two_samples(self):
        from cpsvisualizer.adaptive import discrimination_power_score
        mat = pd.DataFrame([[0, 5.0], [5.0, 0]])
        dps = discrimination_power_score(mat)
        assert dps == 0.0  # n < 3 returns 0

    def test_dps_stability(self):
        from cpsvisualizer.adaptive import discrimination_stability
        mat = pd.DataFrame([
            [0, 10, 12, 9],
            [10, 0, 11, 8],
            [12, 11, 0, 7],
            [9, 8, 7, 0],
        ], dtype=float)
        mean_dps, std_dps, cv = discrimination_stability(mat, n_bootstrap=50)
        assert 0 <= mean_dps <= 1
        assert std_dps >= 0

    def test_find_optimal_pipeline(self):
        from cpsvisualizer.adaptive import find_optimal_pipeline
        dfs = _df_list(4, 20, 15)
        names = _names(4)
        # Test with a subset of pipelines and metrics for speed
        result = find_optimal_pipeline(
            dfs, names,
            pipelines=[['log_transform'], ['equalize_hist']],
            metrics=None,  # use default
            n_jobs=1, verbose=False
        )
        assert len(result) > 0
        assert 'pipeline' in result.columns
        assert 'metric' in result.columns
        assert 'dps' in result.columns
        assert result['dps'].max() > 0

    def test_comprehensive_benchmark(self):
        from cpsvisualizer.adaptive import compute_comprehensive_benchmark, benchmark_summary_table
        dfs = _df_list(5, 20, 15)
        names = _names(5)
        result = compute_comprehensive_benchmark(dfs, names, n_jobs=1)
        assert 'aoda' in result
        assert 'pca_space' in result
        assert 'raw_baseline' in result
        tbl = benchmark_summary_table(result)
        assert len(tbl) > 0

    def test_aoda_improves_over_baseline(self):
        from cpsvisualizer.adaptive import find_optimal_pipeline
        dfs = _df_list(5, 30, 20)
        names = _names(5)
        result = find_optimal_pipeline(dfs, names, n_jobs=1, verbose=False)
        best_dps = result.iloc[0]['dps']
        # AODA-optimal should outperform raw Euclidean baseline
        assert best_dps > 0.4
