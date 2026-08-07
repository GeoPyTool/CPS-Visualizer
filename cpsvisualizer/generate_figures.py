"""
Generate publication-quality comparison figures for CPS-Visualizer manuscript revision.
Each reviewer-requested method gets its own dedicated comparison visualization.

Figures generated:
  Fig 7  - Image Quality Metrics Comparison (PSNR, entropy, CEI, Tenengrad, SSIM per transform)
  Fig 8  - DPS Method Comparison Bar Chart (AODA vs PCA vs t-SNE vs UMAP vs baseline vs Hie)
  Fig 9  - AODA Optimization Landscape Heatmap (390 pipeline×metric combinations)
  Fig 10 - PCA Scatter + AODA Distance Matrix Comparison
  Fig 11 - t-SNE Embedding vs AODA Optimal Distance
  Fig 12 - UMAP Embedding vs AODA Optimal Distance
  Fig 13 - Hierarchical Dendrogram vs K-Means Cluster Assignment
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpsvisualizer', 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

from cpsvisualizer.core import (
    TRANSFORM_FUNCTIONS, DISTANCE_FUNCTIONS, DISTANCE_NAMES,
    compute_pairwise_matrix, Euclidean, Hsim_Distance, Bray_Curtis,
    Cosine, Canberra, calculate_ssim,
)
from cpsvisualizer.statistics import (
    compute_pca, compute_pearson_correlation_matrix,
    compute_anova, compute_uncertainty_all,
)
from cpsvisualizer.metrics import batch_evaluate_transforms
from cpsvisualizer.comparison import (
    compute_pca_embedding, compute_tsne_embedding, compute_umap_embedding,
    compute_hierarchical_clustering, compute_kmeans_clustering,
    compute_all_comparisons,
)
from cpsvisualizer.adaptive import (
    find_optimal_pipeline, compute_comprehensive_benchmark,
    discrimination_power_score, benchmark_summary_table,
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'CPS-Visualizer', 'figures')
os.makedirs(OUTPUT, exist_ok=True)

COLORS = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47', '#264478']
# Elegant minimal palette for workflow diagram
WF_DARK  = '#2C3E50'
WF_MID   = '#5D6D7E'
WF_LIGHT = '#D5D8DC'
WF_ACCENT = '#2980B9'
WF_BG    = '#F8F9FA'


def generate_data(n_elements=6, rows=60, cols=50):
    """Realistic multi-element LA-ICP-MS data."""
    rng = np.random.default_rng(42)
    elements = ['Ag109', 'Cu65', 'Zn66', 'Fe57', 'Pb208', 'Au197'][:n_elements]
    data = {}
    for i, el in enumerate(elements):
        base = rng.lognormal(mean=1.5 + i * 0.4, sigma=0.25 + i * 0.08, size=(rows, cols))
        noise_x = 0.3 * np.sin(np.linspace(0, 4 * np.pi, rows)).reshape(-1, 1)
        noise_y = 0.2 * np.cos(np.linspace(0, 3 * np.pi, cols)).reshape(1, -1)
        base = base * (1 + noise_x + noise_y) * (1 + 0.15 * i)
        data[el] = pd.DataFrame(base)
    return data, elements


def save_figure(fig, name):
    for fmt, dpi in [('.pdf', None), ('.png', 600), ('.svg', None)]:
        path = os.path.join(OUTPUT, f'{name}{fmt}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f'  Saved: {name}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Image Quality Metrics Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_image_quality(data_dict, elements):
    """Bar chart comparing PSNR, entropy, CEI, Tenengrad across 6 transforms."""
    dfs = [data_dict[e] for e in elements]
    tset = {
        'Raw': lambda x: x.copy(),
        'Log': TRANSFORM_FUNCTIONS['log_transform'],
        'Centered': TRANSFORM_FUNCTIONS['centering_transform'],
        'Z-Score': TRANSFORM_FUNCTIONS['z_score_normalization'],
        'Std': TRANSFORM_FUNCTIONS['standardize'],
        'Log+Eq': lambda x: TRANSFORM_FUNCTIONS['equalize_hist'](TRANSFORM_FUNCTIONS['log_transform'](x.copy())),
    }
    metrics = batch_evaluate_transforms(dfs, elements, tset)
    names = list(tset.keys())

    psnr_vals, ent_vals, cei_vals, ten_vals, ssim_vals = [], [], [], [], []
    for n in names:
        ds = [metrics[n][e] for e in elements if isinstance(metrics[n].get(e, {}), dict) and 'psnr' in metrics[n][e]]
        if ds:
            psnr_vals.append(np.mean([m['psnr'] for m in ds if m['psnr'] != float('inf')]))
            ent_vals.append(np.mean([m['entropy_transformed']['normalized_entropy'] for m in ds]))
            cei_vals.append(np.mean([m['cei'] for m in ds]))
            ten_vals.append(np.mean([m['tenengrad_transformed'] for m in ds]))
            ssim_vals.append(np.mean([m['ssim_vs_original']['ssim'] for m in ds]))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    x = np.arange(len(names))
    w = 0.6

    bars = [
        (axes[0], psnr_vals, 'PSNR (dB)', 'Higher = better fidelity', COLORS[0]),
        (axes[1], ent_vals, 'Normalized Entropy', 'Higher = richer information', COLORS[1]),
        (axes[2], cei_vals, 'Contrast Enhancement Index', '>1 = improved contrast', COLORS[2]),
        (axes[3], ten_vals, 'Tenengrad Sharpness', 'Higher = more edge detail', COLORS[3]),
        (axes[4], ssim_vals, 'SSIM vs Original', '1.0 = identical structure', COLORS[4]),
    ]

    for ax, vals, ylabel, subtitle, color in bars:
        ax.bar(x, vals, w, color=color, edgecolor='black', linewidth=0.3, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=10, style='italic')
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        if ylabel == 'Contrast Enhancement Index':
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=0.7, alpha=0.6)

    axes[5].axis('off')
    fig.suptitle('Figure 7. Image Quality Metrics Across Preprocessing Methods',
                 fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, 'fig7_image_quality_metrics')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: DPS Method Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_dps_comparison(data_dict, elements):
    """Bar chart comparing DPS across AODA, PCA, t-SNE, UMAP, baseline, and
    hierarchical cophenetic correlation."""
    dfs = [data_dict[e] for e in elements]
    bench = compute_comprehensive_benchmark(dfs, elements, n_jobs=1)
    tbl = benchmark_summary_table(bench)

    dps_methods = tbl[tbl['DPS'].notna()].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: DPS comparison
    methods = dps_methods['Method'].tolist()
    dps_vals = dps_methods['DPS'].tolist()
    labels = ['AODA\n(Proposed)', 'PCA-space', 't-SNE-space', 'UMAP-space', 'Raw Euclidean\n(Baseline)']

    colors_dps = [COLORS[0], COLORS[3], COLORS[1], COLORS[5], COLORS[2]]
    bars = ax1.bar(range(len(dps_vals)), dps_vals, color=colors_dps, edgecolor='black', linewidth=0.3)
    ax1.set_xticks(range(len(dps_vals)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Discrimination Power Score (DPS)')
    ax1.set_title('Method Comparison: DPS', fontweight='bold')
    ax1.grid(axis='y', alpha=0.2, linestyle='--')
    for bar, val in zip(bars, dps_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right: Clustering quality
    cluster_methods = ['Hierarchical\n(Ward)', 'K-Means\n(k=3)']
    cluster_vals = [
        bench['hierarchical']['cophenetic_correlation'],
        bench['kmeans']['silhouette_score'],
    ]
    bars2 = ax2.bar(range(2), cluster_vals, color=[COLORS[4], COLORS[3]],
                    edgecolor='black', linewidth=0.3)
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(cluster_methods, fontsize=9)
    ax2.set_ylabel('Score')
    ax2.set_title('Clustering Quality Metrics', fontweight='bold')
    ax2.grid(axis='y', alpha=0.2, linestyle='--')
    for bar, val in zip(bars2, cluster_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle('Figure 8. Comparison of Discrimination Power and Clustering Quality Across Methods',
                 fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, 'fig8_dps_method_comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: AODA Optimization Landscape
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_aoda_landscape(data_dict, elements):
    """Heatmap of DPS scores across all 15 pipelines × 26 metrics."""
    dfs = [data_dict[e] for e in elements]
    result = find_optimal_pipeline(dfs, elements, n_jobs=-1, verbose=False)

    pipelines = result['pipeline'].unique()
    metrics = result['metric'].unique()
    heatmap = np.zeros((len(pipelines), len(metrics)))
    for _, row in result.iterrows():
        pi = list(pipelines).index(row['pipeline'])
        mi = list(metrics).index(row['metric'])
        heatmap[pi, mi] = row['dps']

    metric_short = [m.replace('mutual_info_', 'MI_').replace('_unflattern', '').replace('_flattern', '_f')
                    .replace('calculate_', '')[:14] for m in metrics]

    fig, ax = plt.subplots(figsize=(18, 7))
    cmap = LinearSegmentedColormap.from_list('aoda', ['#f7fbff', '#2171b5'], N=256)
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_short, rotation=90, fontsize=6)
    ax.set_yticks(range(len(pipelines)))
    ax.set_yticklabels(pipelines, fontsize=7)
    ax.set_xlabel('Distance / Similarity Metric')
    ax.set_ylabel('Preprocessing Pipeline')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('DPS')

    # Mark the optimal
    best = result.iloc[0]
    bpi = list(pipelines).index(best['pipeline'])
    bmi = list(metrics).index(best['metric'])
    ax.plot(bmi, bpi, 'r*', markersize=18, markeredgecolor='white', markeredgewidth=1)

    ax.set_title('Figure 9. AODA Optimization Landscape: DPS Across 390 Pipeline\u00d7Metric Combinations\n'
                 f'(Red star: optimal = {best["pipeline"]} + {best["metric"]}, DPS={best["dps"]:.3f})',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    save_figure(fig, 'fig9_aoda_landscape')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: PCA Comparison + Distance Matrix
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_pca_comparison(data_dict, elements):
    """PCA scatter plot side-by-side with AODA optimal distance matrix."""
    dfs = [data_dict[e] for e in elements]
    pca_r = compute_pca_embedding(dfs, elements)
    bench = compute_comprehensive_benchmark(dfs, elements, n_jobs=1)
    best_pipe = '+'.join(bench['aoda'].iloc[0]['pipeline'].split('+'))
    best_met = bench['aoda'].iloc[0]['metric']

    from cpsvisualizer.core import apply_transforms, DISTANCE_FUNCTIONS
    from cpsvisualizer.adaptive import PIPELINE_COMBOS
    pipe_names = bench['aoda'].iloc[0]['pipeline'].split('+')
    met_func = {f.__name__: f for f in DISTANCE_FUNCTIONS}[best_met]
    transformed = [pd.DataFrame(apply_transforms(df.values.copy(), pipe_names)) for df in dfs]
    dist_mat = compute_pairwise_matrix(transformed, elements, met_func)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: PCA scatter
    coords = pca_r['embedding']
    ev = pca_r['explained_variance']
    ax1.scatter(coords[:, 0], coords[:, 1], c=COLORS[0], s=200, alpha=0.85,
                edgecolors='black', linewidth=0.5)
    for i, name in enumerate(elements):
        ax1.annotate(name, (coords[i, 0], coords[i, 1]),
                     textcoords="offset points", xytext=(6, 6), fontsize=10, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)')
    ax1.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)')
    ax1.set_title('PCA: Element Distribution Patterns', fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Right: AODA optimal distance matrix heatmap
    im2 = ax2.imshow(dist_mat.values, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(elements)))
    ax2.set_yticks(range(len(elements)))
    ax2.set_xticklabels(elements, fontsize=9)
    ax2.set_yticklabels(elements, fontsize=9)
    for i in range(len(elements)):
        for j in range(len(elements)):
            ax2.text(j, i, f'{dist_mat.iloc[i, j]:.3f}', ha='center', va='center', fontsize=8)
    ax2.set_title(f'AODA Optimal: {best_pipe}+{best_met}\nDPS = {bench["aoda"].iloc[0]["dps"]:.3f}',
                  fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle('Figure 10. PCA Dimensionality Reduction vs AODA-Optimal Distance Matrix',
                 fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, 'fig10_pca_vs_aoda')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: t-SNE Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_tsne_comparison(data_dict, elements):
    """t-SNE embedding vs AODA optimal distance matrix side-by-side."""
    dfs = [data_dict[e] for e in elements]
    tsne_r = compute_tsne_embedding(dfs, elements, perplexity=min(5, len(elements)-1), random_state=42)
    bench = compute_comprehensive_benchmark(dfs, elements, n_jobs=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # t-SNE
    coords = tsne_r['embedding']
    ax1.scatter(coords[:, 0], coords[:, 1], c=COLORS[1], s=200, alpha=0.85,
                edgecolors='black', linewidth=0.5)
    for i, name in enumerate(elements):
        ax1.annotate(name, (coords[i, 0], coords[i, 1]),
                     textcoords="offset points", xytext=(6, 6), fontsize=10, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.set_title(f't-SNE Embedding (KL={tsne_r.get("kl_divergence", 0):.3f})', fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--')

    # DPS comparison mini bar
    methods = ['AODA', 't-SNE', 'PCA', 'UMAP', 'Baseline']
    dps_vals = [
        bench['aoda'].iloc[0]['dps'],
        bench.get('tsne_space', {}).get('dps', 0),
        bench.get('pca_space', {}).get('dps', 0),
        bench.get('umap_space', {}).get('dps', 0),
        bench.get('raw_baseline', {}).get('dps', 0),
    ]
    colors_bar = [COLORS[0], COLORS[1], COLORS[3], COLORS[5], COLORS[2]]
    bars = ax2.barh(methods, dps_vals, color=colors_bar, edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('DPS')
    ax2.set_title('DPS: t-SNE vs Other Methods', fontweight='bold')
    ax2.grid(axis='x', alpha=0.2, linestyle='--')
    for bar, val in zip(bars, dps_vals):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    fig.suptitle('Figure 11. t-SNE Nonlinear Embedding vs AODA Discrimination Power',
                 fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, 'fig11_tsne_comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: UMAP Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig12_umap_comparison(data_dict, elements):
    """UMAP embedding vs AODA distance matrix."""
    dfs = [data_dict[e] for e in elements]
    umap_r = compute_umap_embedding(dfs, elements)
    bench = compute_comprehensive_benchmark(dfs, elements, n_jobs=1)
    aoda = bench['aoda']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    coords = umap_r['embedding']
    ax1.scatter(coords[:, 0], coords[:, 1], c=COLORS[5], s=200, alpha=0.85,
                edgecolors='black', linewidth=0.5)
    for i, name in enumerate(elements):
        ax1.annotate(name, (coords[i, 0], coords[i, 1]),
                     textcoords="offset points", xytext=(6, 6), fontsize=10, fontweight='bold')
    ax1.set_xlabel('UMAP Component 1')
    ax1.set_ylabel('UMAP Component 2')
    ax1.set_title('UMAP: Element Distribution Patterns', fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Top-5 AODA pipelines
    top5 = aoda.head(5)
    labels = [f'{r["pipeline"][:20]}\n+{r["metric"][:18]}' for _, r in top5.iterrows()]
    vals = top5['dps'].tolist()
    colors_bar = [COLORS[0], COLORS[3], COLORS[2], COLORS[4], COLORS[5]]
    bars = ax2.barh(range(len(labels)), vals, color=colors_bar, edgecolor='black', linewidth=0.3)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel('DPS')
    ax2.set_title('Top-5 AODA Optimal Pipelines', fontweight='bold')
    ax2.grid(axis='x', alpha=0.2, linestyle='--')
    ax2.set_xlim(0, max(vals) * 1.15)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8, fontweight='bold')

    fig.suptitle('Figure 12. UMAP Topological Embedding vs AODA Pipeline Ranking',
                 fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, 'fig12_umap_comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: Hierarchical Clustering + K-Means
# ═══════════════════════════════════════════════════════════════════════════════
def fig13_clustering(data_dict, elements):
    """Dendrogram + K-means cluster assignments."""
    dfs = [data_dict[e] for e in elements]
    hier = compute_hierarchical_clustering(dfs, elements)
    kmeans = compute_kmeans_clustering(dfs, elements, n_clusters=3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Dendrogram
    Z = hier['linkage']
    dendrogram(Z, labels=elements, ax=ax1, leaf_rotation=45, leaf_font_size=10,
               color_threshold=0.7 * max(Z[:, 2]))
    ax1.set_ylabel('Ward Distance')
    ax1.set_title(f'Hierarchical Clustering\nCophenetic r = {hier["cophenetic_correlation"]:.3f}',
                  fontweight='bold')

    # K-means cluster assignment table
    labels_k = kmeans['labels']
    from cpsvisualizer.comparison import prepare_feature_matrix
    X, _ = prepare_feature_matrix(dfs)
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(X, labels_k) if len(set(labels_k)) > 1 else 0

    # Color elements by cluster
    cluster_colors = [COLORS[0], COLORS[1], COLORS[2]]
    elem_colors = [cluster_colors[l] for l in labels_k]
    ax2.bar(range(len(elements)), [1]*len(elements), color=elem_colors, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(elements)))
    ax2.set_xticklabels(elements, fontsize=10)
    ax2.set_yticks([])
    for i, (name, lbl) in enumerate(zip(elements, labels_k)):
        ax2.text(i, 0.5, f'C{lbl+1}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax2.set_title(f'K-Means Clustering (k=3)\nSilhouette = {sil:.3f}, Inertia = {kmeans["inertia"]:.1f}',
                  fontweight='bold')

    fig.suptitle('Figure 13. Hierarchical Clustering Dendrogram and K-Means Cluster Assignments',
                 fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, 'fig13_clustering')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Elegant Workflow Diagram (minimal color scheme)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_workflow():
    """Redesigned elegant workflow diagram with clean minimal colour scheme."""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(WF_BG)

    def dbox(x, y, w, h, text, sub='', color=WF_DARK, tc='white', fs=12):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.15',
                           facecolor=color, edgecolor='none', alpha=0.95, zorder=3)
        ax.add_patch(b)
        ax.text(x, y + 0.08, text, ha='center', va='center', fontsize=fs,
                fontweight='bold', color=tc, zorder=4)
        if sub:
            ax.text(x, y - 0.38, sub, ha='center', va='center', fontsize=8,
                    color=tc, alpha=0.85, zorder=4, style='italic')

    def iobox(x, y, w, h, text, sub=''):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.1',
                           facecolor=WF_LIGHT, edgecolor=WF_MID, linewidth=1.2, zorder=3)
        ax.add_patch(b)
        ax.text(x, y + 0.05, text, ha='center', va='center', fontsize=10,
                fontweight='bold', color=WF_DARK, zorder=4)
        if sub:
            ax.text(x, y - 0.33, sub, ha='center', va='center', fontsize=8,
                    color=WF_MID, zorder=4, style='italic')

    def arr(x1, y1, x2, y2, color=WF_MID, lw=2.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                   connectionstyle='arc3,rad=0'))

    # Row 1: Inputs
    ry = 5.8
    iobox(3.0, ry, 4.0, 1.2, 'LA-ICP-MS Raw Data', 'CSV files: per-element CPS matrices')
    iobox(9.0, ry, 4.0, 1.2, 'User Configuration', 'Transform + Metric selection')
    iobox(13.5, ry, 3.5, 1.2, 'AODA Auto-Optimize', 'Adaptive pipeline discovery')
    arr(5.0, ry, 7.0, ry)
    arr(11.0, ry, 11.75, ry, color=WF_ACCENT, lw=2)

    # Row 2: Processing
    ry2 = 4.2
    dbox(5.0, ry2, 3.2, 1.0, 'Preprocessing', 'log, center, z-score,\nstandardize, equalize', WF_MID)
    dbox(10.5, ry2, 3.2, 1.0, 'AODA Optimization', '390 combos \u00d7 bootstrap', WF_ACCENT)
    arr(3.0, 5.2, 3.0, ry2+0.5)
    arr(9.0, 5.2, 9.0, ry2+0.5)
    arr(13.5, 5.2, 13.5, ry2+0.5)
    arr(6.6, ry2, 8.9, ry2)

    # Row 3: Analysis modules
    ry3 = 2.5
    bx = [
        (2.0, 'Similarity\nMeasurement', '26 metrics\nHsim, Bray-Curtis,\nCanberra, SSIM, MI'),
        (5.5, 'Statistical\nAnalysis', 'PCA, Correlation,\nANOVA, Kruskal-Wallis,\nUncertainty'),
        (9.0, 'Method\nComparison', 't-SNE, UMAP,\nHierarchical Clust.,\nK-Means'),
        (12.5, 'Image Quality\nMetrics', 'PSNR, Entropy, CEI,\nTenengrad, SSIM'),
    ]
    for x, t, s in bx:
        dbox(x, ry3, 2.8, 1.6, t, s, WF_DARK, 'white', 9.5)
    arr(5.0, ry2-0.5, 5.0, ry3+0.8)
    arr(10.5, ry2-0.5, 10.5, ry3+0.8)

    # Row 4: Outputs
    ry4 = 0.8
    iobox(4.5, ry4, 5.5, 1.0, 'Visualization Output', 'PNG / PDF / SVG publication figures')
    iobox(11.5, ry4, 5.5, 1.0, 'Data Output', 'CSV distance matrices + statistics')
    arr(5.0, ry3-0.8, 5.0, ry4+0.5)
    arr(10.5, ry3-0.8, 10.5, ry4+0.5)

    # Side labels
    for x, y, lbl in [(0.2, 5.8, 'DATA'), (0.2, 4.2, 'PROCESS'),
                       (0.2, 2.5, 'ANALYZE'), (0.2, 0.8, 'OUTPUT')]:
        ax.text(x, y, lbl, fontsize=9, fontweight='bold', color=WF_MID, rotation=90, va='center')

    ax.set_title('Figure 1. CPS-Visualizer Workflow', fontweight='bold', fontsize=14, pad=12)
    fig.tight_layout()
    save_figure(fig, 'fig1_workflow')


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print('Generating data...')
    data_dict, elements = generate_data(6)

    print('\nGenerating figures...')
    fig1_workflow()
    fig7_image_quality(data_dict, elements)
    fig8_dps_comparison(data_dict, elements)
    fig9_aoda_landscape(data_dict, elements)
    fig10_pca_comparison(data_dict, elements)
    fig11_tsne_comparison(data_dict, elements)
    fig12_umap_comparison(data_dict, elements)
    fig13_clustering(data_dict, elements)

    print(f'\nAll figures saved to: {OUTPUT}')


if __name__ == '__main__':
    main()
