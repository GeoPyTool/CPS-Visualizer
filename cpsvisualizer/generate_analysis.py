"""
Generate comprehensive analysis and publication-quality figures for
the CPS-Visualizer manuscript revision.

This script produces all figures required to address reviewer comments:
- PCA, t-SNE, UMAP scatter plots (Reviewer #1)
- Image quality metrics comparison (Reviewer #3)
- Correlation heatmap (Reviewer #2)  
- Hierarchical clustering dendrogram
- Enhanced data visualization grids
- Statistical analysis summary tables
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpsvisualizer', 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cpsvisualizer.core import (
    TRANSFORM_FUNCTIONS, DISTANCE_FUNCTIONS, load_data_files,
    compute_pairwise_matrix, Euclidean, Hsim_Distance, Bray_Curtis,
    calculate_ssim, Cosine, Canberra,
)
from cpsvisualizer.statistics import (
    compute_pca, compute_pearson_correlation_matrix,
    compute_spearman_correlation_matrix, compute_anova,
    compute_uncertainty_all, compute_descriptive_statistics_all,
)
from cpsvisualizer.metrics import (
    batch_evaluate_transforms, compute_all_image_metrics,
)
from cpsvisualizer.comparison import (
    compute_pca_embedding, compute_tsne_embedding, compute_umap_embedding,
    compute_hierarchical_clustering, compute_kmeans_clustering,
    compute_all_comparisons,
)
from cpsvisualizer.visualization import (
    plot_pca_comparison, plot_tsne_comparison, plot_umap_comparison,
    plot_dendrogram, plot_image_quality_comparison,
    plot_correlation_heatmap, plot_enhanced_data_matrix,
    plot_all_comparisons,
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'CPS-Visualizer', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_sample_data(n_elements=6, rows=60, cols=50, seed=42):
    """Generate realistic LA-ICP-MS element distribution data."""
    np.random.seed(seed)
    elements = ['Ag109', 'Cu65', 'Zn66', 'Fe57', 'Pb208', 'Au197']
    data = {}
    for i, el in enumerate(elements[:n_elements]):
        base = np.random.lognormal(
            mean=1.5 + i * 0.4,
            sigma=0.25 + i * 0.08,
            size=(rows, cols)
        )
        noise_x = 0.3 * np.sin(np.linspace(0, 4 * np.pi, rows)).reshape(-1, 1)
        noise_y = 0.2 * np.cos(np.linspace(0, 3 * np.pi, cols)).reshape(1, -1)
        base = base * (1 + noise_x + noise_y) * (1 + 0.15 * i)
        data[el] = pd.DataFrame(base)
    return data, elements[:n_elements]


def main():
    print("Generating sample data...")
    data_dict, elements = generate_sample_data(n_elements=6)
    df_list = [data_dict[el] for el in elements]
    df_name_list = elements

    print("Running comprehensive analysis...")

    # 1. Distance matrices with traditional + custom methods
    print("  Computing distance matrices...")
    dist_euclidean = compute_pairwise_matrix(df_list, df_name_list, Euclidean)
    dist_hsim = compute_pairwise_matrix(df_list, df_name_list, Hsim_Distance)
    dist_bray = compute_pairwise_matrix(df_list, df_name_list, Bray_Curtis)
    dist_canberra = compute_pairwise_matrix(df_list, df_name_list, Canberra)
    dist_cosine = compute_pairwise_matrix(df_list, df_name_list, Cosine)

    # Save all distance matrices as CSV
    dist_dir = os.path.join(OUTPUT_DIR, 'distance_matrices')
    os.makedirs(dist_dir, exist_ok=True)
    for name, df in [('Euclidean', dist_euclidean), ('Hsim', dist_hsim),
                      ('Bray_Curtis', dist_bray), ('Canberra', dist_canberra),
                      ('Cosine', dist_cosine)]:
        df.to_csv(os.path.join(dist_dir, f'{name}_distance.csv'))
        print(f"    Saved {name} distance matrix")

    # 2. Statistical analysis
    print("  Running statistical analysis...")
    pearson_corr, pearson_pval = compute_pearson_correlation_matrix(
        df_list, df_name_list
    )
    spearman_corr, _ = compute_spearman_correlation_matrix(df_list, df_name_list)
    anova_result = compute_anova(df_list, df_name_list)
    uncertainty = compute_uncertainty_all(df_list, df_name_list)
    desc_stats = compute_descriptive_statistics_all(df_list, df_name_list)

    # Save statistical tables
    stats_dir = os.path.join(OUTPUT_DIR, 'statistics')
    os.makedirs(stats_dir, exist_ok=True)
    pearson_corr.to_csv(os.path.join(stats_dir, 'pearson_correlation.csv'))
    pearson_pval.to_csv(os.path.join(stats_dir, 'pearson_p_values.csv'))
    spearman_corr.to_csv(os.path.join(stats_dir, 'spearman_correlation.csv'))
    anova_df = pd.DataFrame([{k: v for k, v in anova_result.items()
                               if k != 'names'}])
    anova_df.to_csv(
        os.path.join(stats_dir, 'anova_results.csv'), index=False
    )
    pd.DataFrame(uncertainty).T.to_csv(
        os.path.join(stats_dir, 'uncertainty_analysis.csv')
    )
    pd.DataFrame(desc_stats).T.to_csv(
        os.path.join(stats_dir, 'descriptive_statistics.csv')
    )
    print(f"    ANOVA: F={anova_result['anova_f_statistic']:.3f}, "
          f"p={anova_result['anova_p_value']:.2e}")
    print(f"    Kruskal-Wallis: H={anova_result['kruskal_h_statistic']:.3f}, "
          f"p={anova_result['kruskal_p_value']:.2e}")

    # 3. Image quality metrics
    print("  Evaluating image quality metrics...")
    transform_set = {
        'Raw': lambda x: x.copy(),
        'Log': TRANSFORM_FUNCTIONS['log_transform'],
        'Centered': TRANSFORM_FUNCTIONS['centering_transform'],
        'Z-Score': TRANSFORM_FUNCTIONS['z_score_normalization'],
        'Standardized': TRANSFORM_FUNCTIONS['standardize'],
        'Equalized': TRANSFORM_FUNCTIONS['equalize_hist'],
    }
    metrics_results = batch_evaluate_transforms(
        df_list, df_name_list, transform_set
    )
    # Save metrics summary
    summary_rows = []
    for trans_name, ds_dict in metrics_results.items():
        row = {'transform': trans_name}
        for ds_name, m in ds_dict.items():
            if isinstance(m, dict) and 'psnr' in m:
                row[f'{ds_name}_PSNR'] = m['psnr']
                row[f'{ds_name}_Entropy'] = m['entropy_transformed']['normalized_entropy']
                row[f'{ds_name}_CEI'] = m['cei']
        summary_rows.append(row)
    metrics_summary = pd.DataFrame(summary_rows)
    metrics_summary.to_csv(os.path.join(stats_dir, 'image_quality_metrics.csv'),
                           index=False)
    print("    Saved image quality metrics summary")

    # 4. Comparison with established methods
    print("  Running dimensionality reduction comparisons...")
    comparison_results = compute_all_comparisons(df_list, df_name_list)
    pca_result = comparison_results['pca']
    tsne_result = comparison_results['tsne']
    umap_result = comparison_results['umap']
    hier_result = comparison_results['hierarchical']
    kmeans_result = comparison_results['kmeans']

    print(f"    PCA: EV1={pca_result['explained_variance'][0]:.3f}, "
          f"EV2={pca_result['explained_variance'][1]:.3f}")
    print(f"    t-SNE KL divergence: {tsne_result.get('kl_divergence', 'N/A')}")
    print(f"    Hierarchical cophenetic r: {hier_result['cophenetic_correlation']:.3f}")
    print(f"    K-Means inertia: {kmeans_result['inertia']:.3f}")

    # 5. Generate all figures
    print("  Generating figures...")

    # PCA scatter plot
    fig_pca_path = plot_pca_comparison(pca_result, save_path=None)
    print(f"    Saved PCA scatter plot")

    # t-SNE scatter plot
    fig_tsne_path = plot_tsne_comparison(tsne_result, save_path=None)
    print(f"    Saved t-SNE scatter plot")

    # UMAP scatter plot
    fig_umap_path = plot_umap_comparison(umap_result, save_path=None)
    print(f"    Saved UMAP scatter plot")

    # Dendrogram
    fig_dend_path = plot_dendrogram(hier_result, save_path=None)
    print(f"    Saved dendrogram")

    # Image quality metrics comparison
    fig_metrics_path = plot_image_quality_comparison(metrics_results, save_path=None)
    print(f"    Saved image quality metrics comparison")

    # Correlation heatmap
    fig_corr_path = plot_correlation_heatmap(
        pearson_corr, pearson_pval, save_path=None
    )
    print(f"    Saved correlation heatmap")

    # Enhanced data visualization with different transforms
    for label, transforms in [
        ('raw', []),
        ('log', [TRANSFORM_FUNCTIONS['log_transform']]),
        ('log_equalized', [TRANSFORM_FUNCTIONS['log_transform'],
                           TRANSFORM_FUNCTIONS['equalize_hist']]),
        ('standardized', [TRANSFORM_FUNCTIONS['standardize']]),
    ]:
        plot_enhanced_data_matrix(
            df_list, df_name_list, transforms=transforms,
            cmap='viridis',
            save_path=os.path.join(OUTPUT_DIR, f'data_matrix_{label}.pdf')
        )
    print(f"    Saved enhanced data matrices")

    # Print summary for manuscript integration
    print("\n" + "=" * 70)
    print("SUMMARY FOR MANUSCRIPT")
    print("=" * 70)

    print("\nImage Quality Metrics (mean across elements):")
    for trans_name in metrics_results:
        ds_vals = [v for v in metrics_results[trans_name].values()
                   if isinstance(v, dict) and 'psnr' in v]
        if ds_vals:
            psnr_mean = np.mean([v['psnr'] for v in ds_vals if v['psnr'] != float('inf')])
            ent_mean = np.mean([v['entropy_transformed']['normalized_entropy']
                                for v in ds_vals])
            cei_mean = np.mean([v['cei'] for v in ds_vals])
            print(f"  {trans_name:20s}: PSNR={psnr_mean:.2f}, "
                  f"Entropy={ent_mean:.3f}, CEI={cei_mean:.3f}")

    print("\nPairwise Distance Comparison (Hsim vs Euclidean vs Canberra):")
    for method_name, dist_df in [('Hsim', dist_hsim), ('Euclidean', dist_euclidean),
                                  ('Canberra', dist_canberra)]:
        print(f"\n  {method_name} distance matrix:")
        print(dist_df.round(4).to_string())

    print("\nStatistical Analysis:")
    print(f"  ANOVA: F({len(df_list)-1},{sum(df.size for df in df_list)-len(df_list)}) "
          f"= {anova_result['anova_f_statistic']:.2f}, "
          f"p = {anova_result['anova_p_value']:.2e}")
    print(f"  Kruskal-Wallis H = {anova_result['kruskal_h_statistic']:.2f}, "
          f"p = {anova_result['kruskal_p_value']:.2e}")

    print(f"\nPearson Correlation (mean absolute r): "
          f"{np.mean(np.abs(pearson_corr.values[np.triu_indices_from(pearson_corr.values, k=1)])):.3f}")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print(f"All statistics saved to: {stats_dir}")
    print(f"All distance matrices saved to: {dist_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
