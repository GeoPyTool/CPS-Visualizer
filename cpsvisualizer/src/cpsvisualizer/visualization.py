"""
Advanced visualization module for CPS-Visualizer.

Provides publication-quality figures including:
- PCA scatter plots
- t-SNE/UMAP embedding visualizations
- Dendrograms for hierarchical clustering
- Image quality metric comparison charts
- Statistical analysis visualization
- Enhanced data matrix visualization with multiple colormaps
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os
import warnings
from scipy.cluster.hierarchy import dendrogram


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 11
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'


def _ensure_output_dir(custom_dir=None):
    """Determine output directory. Uses custom_dir if provided, else CWD/figures."""
    if custom_dir:
        out = custom_dir
    else:
        out = os.path.join(os.getcwd(), 'figures')
    os.makedirs(out, exist_ok=True)
    return out


def plot_pca_comparison(pca_result, save_path=None, figsize=(7, 6)):
    """Generate a PCA scatter plot with variance explained annotations."""
    fig, ax = plt.subplots(figsize=figsize)
    coords = pca_result['embedding']
    names = pca_result['names']
    ev = pca_result['explained_variance']

    if coords.shape[1] >= 2:
        ax.scatter(coords[:, 0], coords[:, 1], c='steelblue', s=120, alpha=0.85,
                   edgecolors='black', linewidth=0.5)
        for i, name in enumerate(names):
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", xytext=(4, 4), fontsize=9)
        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)')
    elif coords.shape[1] == 1:
        ax.scatter(coords[:, 0], np.zeros(len(coords)), c='steelblue', s=120,
                   alpha=0.85, edgecolors='black', linewidth=0.5)
        for i, name in enumerate(names):
            ax.annotate(name, (coords[i, 0], 0),
                        textcoords="offset points", xytext=(4, 4), fontsize=9)
        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)')
        ax.set_ylabel('')
        ax.set_yticks([])
    ax.set_title('PCA: Element Distribution Patterns')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_output_dir(), 'pca_comparison.pdf')
    for fmt in ['.pdf', '.png', '.svg']:
        path = save_path.replace('.pdf', fmt) if '.pdf' in save_path else save_path + fmt
        fig.savefig(path, dpi=600 if fmt == '.png' else None)
    plt.close(fig)
    return save_path


def plot_tsne_comparison(tsne_result, save_path=None, figsize=(7, 6)):
    """Generate a t-SNE embedding scatter plot."""
    fig, ax = plt.subplots(figsize=figsize)
    coords = tsne_result['embedding']
    names = tsne_result['names']

    ax.scatter(coords[:, 0], coords[:, 1], c='darkorange', s=120, alpha=0.85,
               edgecolors='black', linewidth=0.5)
    for i, name in enumerate(names):
        ax.annotate(name, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=9)

    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    kl = tsne_result.get('kl_divergence', 0)
    ax.set_title(f't-SNE: Element Distribution Patterns (KL={kl:.3f})')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_output_dir(), 'tsne_comparison.pdf')
    for fmt in ['.pdf', '.png', '.svg']:
        path = save_path.replace('.pdf', fmt) if '.pdf' in save_path else save_path + fmt
        fig.savefig(path, dpi=600 if fmt == '.png' else None)
    plt.close(fig)
    return save_path


def plot_umap_comparison(umap_result, save_path=None, figsize=(7, 6)):
    """Generate a UMAP embedding scatter plot."""
    fig, ax = plt.subplots(figsize=figsize)
    coords = umap_result['embedding']
    names = umap_result['names']

    if umap_result.get('error') is not None:
        ax.text(0.5, 0.5, f"UMAP not available:\n{umap_result['error']}",
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c='forestgreen', s=120, alpha=0.85,
                   edgecolors='black', linewidth=0.5)
        for i, name in enumerate(names):
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", xytext=(4, 4), fontsize=9)

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    ax.set_title('UMAP: Element Distribution Patterns')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_output_dir(), 'umap_comparison.pdf')
    for fmt in ['.pdf', '.png', '.svg']:
        path = save_path.replace('.pdf', fmt) if '.pdf' in save_path else save_path + fmt
        fig.savefig(path, dpi=600 if fmt == '.png' else None)
    plt.close(fig)
    return save_path


def plot_dendrogram(hier_result, save_path=None, figsize=(10, 5)):
    """Generate a dendrogram visualization for hierarchical clustering."""
    fig, ax = plt.subplots(figsize=figsize)
    Z = hier_result['linkage']
    names = hier_result['names']
    coph_corr = hier_result.get('cophenetic_correlation', 0)

    dendrogram(Z, labels=names, ax=ax, leaf_rotation=45, leaf_font_size=10,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_ylabel('Distance')
    ax.set_title(f'Hierarchical Clustering Dendrogram (Cophenetic r={coph_corr:.3f})')
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_output_dir(), 'dendrogram.pdf')
    for fmt in ['.pdf', '.png', '.svg']:
        path = save_path.replace('.pdf', fmt) if '.pdf' in save_path else save_path + fmt
        fig.savefig(path, dpi=600 if fmt == '.png' else None)
    plt.close(fig)
    return save_path


def plot_image_quality_comparison(metrics_results, save_path=None, figsize=(12, 8)):
    """Generate bar charts comparing image quality metrics across transform methods.

    Args:
        metrics_results: dict from metrics.batch_evaluate_transforms()
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    trans_names = list(metrics_results.keys())

    if not trans_names:
        axes[0].text(0.5, 0.5, 'No valid metrics data available.',
                    transform=axes[0].transAxes, ha='center', va='center')
        for ax in axes[1:]:
            ax.axis('off')
        fig.tight_layout()
        if save_path is None:
            save_path = os.path.join(_ensure_output_dir(), 'image_quality_metrics.pdf')
        fig.savefig(save_path, dpi=600 if 'png' in str(save_path).lower() else None)
        plt.close(fig)
        return save_path

    x = np.arange(len(trans_names))
    width = 0.25

    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']

    psnr_vals = []
    entropy_vals = []
    cei_vals = []
    tenengrad_vals = []
    ssim_vals = []

    for trans_name in trans_names:
        ds_metrics = list(metrics_results[trans_name].values())
        if ds_metrics and 'psnr' in ds_metrics[0]:
            psnr_vals.append(np.mean([m['psnr'] for m in ds_metrics]))
            entropy_vals.append(np.mean(
                [m['entropy_transformed']['normalized_entropy'] for m in ds_metrics]
            ))
            cei_vals.append(np.mean([m['cei'] for m in ds_metrics]))
            tenengrad_vals.append(np.mean(
                [m['tenengrad_transformed'] for m in ds_metrics]
            ))
            ssim_vals.append(np.mean(
                [m['ssim_vs_original']['ssim'] for m in ds_metrics]
            ))

    if psnr_vals:
        axes[0].bar(x, psnr_vals, width, color=colors[0], edgecolor='black', linewidth=0.3)
        axes[0].set_title('PSNR (higher is better)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(trans_names, rotation=45, ha='right', fontsize=8)

        axes[1].bar(x, entropy_vals, width, color=colors[1], edgecolor='black', linewidth=0.3)
        axes[1].set_title('Normalized Entropy (higher is better)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(trans_names, rotation=45, ha='right', fontsize=8)

        axes[2].bar(x, cei_vals, width, color=colors[2], edgecolor='black', linewidth=0.3)
        axes[2].set_title('Contrast Enhancement Index (>1 = improved)')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(trans_names, rotation=45, ha='right', fontsize=8)
        axes[2].axhline(y=1.0, color='red', linestyle='--', linewidth=0.5)

        axes[3].bar(x, tenengrad_vals, width, color=colors[3], edgecolor='black', linewidth=0.3)
        axes[3].set_title('Tenengrad Sharpness (higher is better)')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(trans_names, rotation=45, ha='right', fontsize=8)

        axes[4].bar(x, ssim_vals, width, color=colors[4], edgecolor='black', linewidth=0.3)
        axes[4].set_title('SSIM vs Original (1.0 = identical)')
        axes[4].set_xticks(x)
        axes[4].set_xticklabels(trans_names, rotation=45, ha='right', fontsize=8)

    axes[5].axis('off')
    fig.suptitle('Image Quality Metrics Comparison Across Transform Methods',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is None:
        save_path = os.path.join(_ensure_output_dir(), 'image_quality_metrics.pdf')
    for fmt in ['.pdf', '.png', '.svg']:
        path = save_path.replace('.pdf', fmt) if '.pdf' in save_path else save_path + fmt
        fig.savefig(path, dpi=600 if fmt == '.png' else None)
    plt.close(fig)
    return save_path


def plot_correlation_heatmap(corr_df, pval_df=None, save_path=None, figsize=(8, 7)):
    """Generate a Pearson correlation heatmap with optional significance markers."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr_df.index, fontsize=9)

    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            val = corr_df.iloc[i, j]
            text = ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                           fontsize=8, color='black')
            if pval_df is not None and pval_df.iloc[i, j] < 0.05:
                text.set_weight('bold')

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Pearson Correlation Matrix', fontweight='bold')
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_output_dir(), 'correlation_heatmap.pdf')
    for fmt in ['.pdf', '.png', '.svg']:
        path = save_path.replace('.pdf', fmt) if '.pdf' in save_path else save_path + fmt
        fig.savefig(path, dpi=600 if fmt == '.png' else None)
    plt.close(fig)
    return save_path


def plot_enhanced_data_matrix(df_list, df_name_list, transforms=None,
                              cmap='viridis', save_path=None, figsize=None):
    """Generate an enhanced grid visualization of data matrices.

    Supports multiple colormaps and optional transform application.
    """
    if transforms is None:
        transforms = []

    num = len(df_list)
    if num == 0:
        return None

    cols = int(np.ceil(np.sqrt(num)))
    rows = int(np.ceil(num / cols))
    if figsize is None:
        figsize = (3 * cols, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, (name, df) in enumerate(zip(df_name_list, df_list)):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        data = df.values.copy()
        for trans_func in transforms:
            try:
                data = trans_func(data)
            except Exception:
                pass
        im = ax.imshow(data, aspect='auto', cmap=cmap)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(num, rows * cols):
        r, c = j // cols, j % cols
        axes[r, c].axis('off')

    fig.suptitle('LA-ICP-MS Element Distribution Maps', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is None:
        tname = '+'.join([t.__name__ if callable(t) else str(t) for t in transforms])
        save_path = os.path.join(_ensure_output_dir(),
                                 f'enhanced_data_matrix_{tname or "raw"}.pdf')
    for fmt in ['.pdf', '.png', '.svg']:
        path = save_path.replace('.pdf', fmt) if '.pdf' in save_path else save_path + fmt
        fig.savefig(path, dpi=600 if fmt == '.png' else None)
    plt.close(fig)
    return save_path


def plot_all_comparisons(df_list, df_name_list, comparison_results,
                         output_dir=None):
    """Generate all comparison visualizations and return paths.

    Returns a dict mapping figure names to file paths.
    """
    if output_dir is None:
        output_dir = _ensure_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    if 'pca' in comparison_results:
        paths['pca'] = plot_pca_comparison(
            comparison_results['pca'],
            os.path.join(output_dir, 'pca_comparison')
        )

    if 'tsne' in comparison_results:
        paths['tsne'] = plot_tsne_comparison(
            comparison_results['tsne'],
            os.path.join(output_dir, 'tsne_comparison')
        )

    if 'umap' in comparison_results:
        paths['umap'] = plot_umap_comparison(
            comparison_results['umap'],
            os.path.join(output_dir, 'umap_comparison')
        )

    if 'hierarchical' in comparison_results:
        paths['dendrogram'] = plot_dendrogram(
            comparison_results['hierarchical'],
            os.path.join(output_dir, 'dendrogram')
        )

    return paths
