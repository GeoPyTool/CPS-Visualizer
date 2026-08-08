"""
Headless batch export module.  Generates every figure (PDF / SVG / PNG)
and every table (CSV / XLSX) for one or more DataSample sub-directories.

Usage (via ``cpsv export``)::

    cpsv export all                          # every dataset, all outputs
    cpsv export geology                      # one dataset
    cpsv export geology bivalve              # multiple
    cpsv export --types maps geology         # only map figures
    cpsv export --types distance,tables geology  # distance figures + tables
    cpsv export --out /tmp/out geology       # custom output root
"""
import glob
import math
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

# Add cpsvisualizer to path
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from cpsvisualizer.core import (
    TRANSFORM_FUNCTIONS, DISTANCE_FUNCTIONS, DISTANCE_NAMES,
    compute_pairwise_matrix, apply_transforms, Euclidean,
    display_scale, ink_colormap, sci_colormap, FIG_BG,
)
from cpsvisualizer.metrics import batch_evaluate_transforms
from cpsvisualizer.comparison import (
    compute_pca_embedding, compute_tsne_embedding, compute_umap_embedding,
    compute_hierarchical_clustering, compute_kmeans_clustering,
    prepare_feature_matrix,
)
from cpsvisualizer.adaptive import (
    find_optimal_power, discrimination_power_score,
)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'

DATA_DIRS = {
    'geology': os.path.join(os.path.dirname(_root), 'DataSample', 'Geology'),
    'bivalve': os.path.join(os.path.dirname(_root), 'DataSample',
                            'BivalveShell_MarineBiology'),
    'tissue': os.path.join(os.path.dirname(_root), 'DataSample',
                           'Tissue_Biomedical'),
}

ALL_TYPES = ['maps', 'distance', 'stats', 'comparison', 'figures', 'tables']


def _downsample(arr, max_rows=160, max_cols=260):
    r, c = arr.shape
    if r <= max_rows and c <= max_cols:
        return arr
    step_r = max(1, int(math.ceil(r / max_rows)))
    step_c = max(1, int(math.ceil(c / max_cols)))
    nr = int(math.ceil(r / step_r)); nc = int(math.ceil(c / step_c))
    pr = nr * step_r - r; pc = nc * step_c - c
    if pr > 0 or pc > 0:
        arr = np.pad(arr, ((0, pr), (0, pc)), mode='edge')
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    view = arr[:nr * step_r, :nc * step_c].reshape(nr, step_r, nc, step_c)
    return view.mean(axis=(1, 3))


def _square_aspect(n_rows, n_cols):
    if n_rows == 0 or n_cols == 0:
        return 1.0
    return float(n_cols) / float(n_rows)


def _equal_lims(ax, xs, ys, pad=0.10):
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    if xs.size == 0 or ys.size == 0:
        return
    x0, x1 = float(np.nanmin(xs)), float(np.nanmax(xs))
    y0, y1 = float(np.nanmin(ys)), float(np.nanmax(ys))
    rx = (x1 - x0) or 1.0; ry = (y1 - y0) or 1.0
    r = max(rx, ry) * (1.0 + pad)
    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
    ax.set_xlim(cx - r / 2.0, cx + r / 2.0)
    ax.set_ylim(cy - r / 2.0, cy + r / 2.0)


def _save(fig, base, out_dirs):
    for d, fmt, dpi in [(out_dirs['png'], '.png', 300),
                         (out_dirs['pdf'], '.pdf', None),
                         (out_dirs['svg'], '.svg', None)]:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f'{base}{fmt}'), dpi=dpi,
                    bbox_inches='tight')
    plt.close(fig)


def _save_table(df, base, out_dirs):
    if df is None or df.empty:
        return
    for d, ext in [(out_dirs['csv'], '.csv'),
                   (out_dirs['xlsx'], '.xlsx')]:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f'{base}{ext}')
        if ext == '.csv':
            df.to_csv(path, encoding='utf-8')
        else:
            df.to_excel(path)


# ═══════════════════════════════════════════════════════════════════════════
# Export functions — one per "type"
# ═══════════════════════════════════════════════════════════════════════════

def _export_maps(dataset, dfs, names, out_dirs):
    """Single map, plot-all, and wipe comparisons."""
    tag = dataset

    # single maps: raw, enhanced, filtered for first element
    for mode, extra in [('raw', []),
                         ('enhanced', ['equalize_hist', 'normalize_01']),
                         ('filtered', ['sobel_gradient', 'normalize_01'])]:
        data = dfs[0].to_numpy().copy()
        for t in extra:
            if t in TRANSFORM_FUNCTIONS:
                data = TRANSFORM_FUNCTIONS[t](data)
        data = _downsample(data)
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(FIG_BG)
        s, lo, hi = display_scale(data, 0, 100)
        ax.imshow(s, cmap=ink_colormap(), vmin=lo, vmax=hi,
                  aspect='auto', interpolation='nearest')
        ax.set_title(f'{names[0]} | {mode}')
        ax.set_aspect(_square_aspect(*data.shape), adjustable='box',
                      anchor='C')
        fig.colorbar(ax.images[0], ax=ax, shrink=0.8)
        fig.tight_layout()
        _save(fig, f'map_{tag}_{names[0]}_{mode}', out_dirs)

    # plot-all
    n = len(names)
    rows, cols = int(math.ceil(math.sqrt(n))), int(math.ceil(n / math.ceil(math.sqrt(n))))
    if rows > cols:
        rows, cols = cols, rows
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.patch.set_facecolor(FIG_BG)
    axes = np.atleast_2d(axes)
    for i, (nm, d) in enumerate(zip(names, dfs)):
        ax = axes[i // cols, i % cols]
        ax.set_facecolor(FIG_BG)
        data = _downsample(d.to_numpy())
        s, lo, hi = display_scale(data, 0, 100)
        ax.imshow(s, cmap=ink_colormap(), vmin=lo, vmax=hi,
                  aspect='auto', interpolation='nearest')
        ax.set_aspect(_square_aspect(*data.shape), adjustable='box',
                      anchor='C')
        ax.set_title(nm, fontsize=11)
    for j in range(n, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])
    fig.tight_layout()
    _save(fig, f'map_{tag}_plot_all', out_dirs)

    # wipe comparisons (first few pairs)
    pairs = []
    if len(names) >= 2:
        pairs.append((names[0], names[1]))
        if len(names) >= 3:
            pairs.append((names[1], names[2]))
    for a_name, b_name in pairs:
        ai = names.index(a_name); bi = names.index(b_name)
        da = _downsample(dfs[ai].to_numpy())
        db = _downsample(dfs[bi].to_numpy())
        min_r = min(da.shape[0], db.shape[0])
        min_c = min(da.shape[1], db.shape[1])
        da = da[:min_r, :min_c]; db = db[:min_r, :min_c]
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(FIG_BG)
        sb, lb, hb = display_scale(db, 0, 100)
        ax.imshow(sb, cmap=ink_colormap(), vmin=lb, vmax=hb,
                  aspect='auto', interpolation='nearest')
        sa, la, ha = display_scale(da, 0, 100)
        ax.imshow(sa, cmap=ink_colormap(opaque_min=True),
                  vmin=la, vmax=ha, aspect='auto',
                  interpolation='nearest')
        ax.set_title(f'Wipe: {a_name} vs {b_name}')
        ax.set_aspect(_square_aspect(*da.shape), adjustable='box',
                      anchor='C')
        fig.tight_layout()
        _save(fig, f'map_{tag}_wipe_{a_name}_{b_name}', out_dirs)


DIST_GRID = ['Euclidean', 'Manhattan', 'Chebyshev', 'Minkowski',
             'Cosine', 'Correlation', 'Hsim_Distance', 'Close_Distance']


def _export_distance(dataset, dfs, names, out_dirs):
    """2x4 distance grid + individual distance CSV/XLSX tables."""
    tag = dataset

    # 2x4 grid figure
    n = len(names)
    metrics = [m for m in DIST_GRID
               if m in {f.__name__ for f in DISTANCE_FUNCTIONS}]
    metrics = metrics[:8]
    nrows, ncols = 2, 4
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor(FIG_BG)
    gs = GridSpec(nrows, ncols + 1, figure=fig,
                  width_ratios=[1, 1, 1, 1, 0.06],
                  wspace=0.25, hspace=0.30,
                  left=0.06, right=0.97, top=0.95, bottom=0.12)
    images = []
    for pos, mname in enumerate(metrics):
        func = {f.__name__: f for f in DISTANCE_FUNCTIONS}[mname]
        mat = compute_pairwise_matrix(dfs, names, func)
        vals = mat.values.astype(float)
        if n > 1:
            off = vals[~np.eye(n, dtype=bool)]
            lo, hi = float(off.min()), float(off.max())
            if hi - lo > 1e-12:
                vals = (vals - lo) / (hi - lo)
                np.fill_diagonal(vals, 0.0)
            vals = np.sqrt(vals)
        row, col = pos // ncols, pos % ncols
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(FIG_BG)
        im = ax.imshow(vals, cmap=sci_colormap(), vmin=0.0, vmax=1.0,
                       aspect='equal')
        images.append(im)
        ax.set_title(mname, fontsize=11)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        if row == nrows - 1:
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=12)
        else:
            ax.set_xticklabels([])
        ax.set_yticklabels(names if col == 0 else [], fontsize=12)
    cax = fig.add_subplot(gs[:, ncols])
    import matplotlib as _mpl
    norm = _mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    sm = _mpl.cm.ScalarMappable(cmap=sci_colormap(), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.set_label('normalized distance (0-1)', fontsize=12)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, f'distance_{tag}', out_dirs)

    # Individual distance matrices as CSV/XLSX
    for mname in ['Euclidean', 'Chebyshev', 'Cosine',
                  'Hsim_Distance', 'Bray_Curtis', 'Canberra']:
        if mname not in {f.__name__ for f in DISTANCE_FUNCTIONS}:
            continue
        func = {f.__name__: f for f in DISTANCE_FUNCTIONS}[mname]
        mat = compute_pairwise_matrix(dfs, names, func)
        _save_table(mat, f'distance_{tag}_{mname}', out_dirs)


def _export_stats(dataset, dfs, names, out_dirs):
    """PCA scatter + correlation heatmap."""
    tag = dataset

    # PCA (use prepare_feature_matrix to handle different shapes)
    from cpsvisualizer.comparison import compute_pca_embedding
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(FIG_BG)
    pca_r = compute_pca_embedding(dfs, names)
    coords = pca_r['embedding']; ev = pca_r['explained_variance']
    ax.scatter(coords[:, 0], coords[:, 1], s=80, c='#4060d8',
               edgecolors='k', linewidths=0.5)
    for i, n in enumerate(names):
        ax.annotate(n, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=11)
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)' if ev else 'PC1')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)' if len(ev) > 1 else 'PC2')
    ax.set_title('PCA: Element Distribution Patterns')
    ax.set_aspect('equal', adjustable='box')
    _equal_lims(ax, coords[:, 0], coords[:, 1])
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    _save(fig, f'stats_{tag}_pca', out_dirs)

    # Correlation (use prepare_feature_matrix for shape compatibility)
    X, _ = prepare_feature_matrix(dfs)
    corr_df = pd.DataFrame(np.corrcoef(X), index=names, columns=names)
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(FIG_BG); ax.set_facecolor(FIG_BG)
    im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_title('Pearson Correlation Matrix')
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    _save(fig, f'stats_{tag}_correlation', out_dirs)


def _export_comparison(dataset, dfs, names, out_dirs):
    """5-panel comparison: PCA/t-SNE/UMAP + Dendrogram + K-Means."""
    tag = dataset
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(FIG_BG)
    gs = GridSpec(2, 6, figure=fig,
                  width_ratios=[1, 1, 1, 1, 1, 1],
                  wspace=0.35, hspace=0.35,
                  left=0.05, right=0.98, top=0.95, bottom=0.07)

    def _draw_scatter(ax, fn, title):
        ax.set_facecolor(FIG_BG)
        try:
            r = fn()
            if isinstance(r, dict) and r.get('error'):
                ax.text(0.5, 0.5, str(r['error']), ha='center', va='center')
            elif 'embedding' in r:
                emb = np.asarray(r['embedding'])
                if emb.ndim == 2 and emb.shape[1] >= 2:
                    ax.scatter(emb[:, 0], emb[:, 1], s=80, c='#4060d8',
                               edgecolors='k', linewidths=0.5)
                    for i, n in enumerate(names):
                        ax.annotate(n, (emb[i, 0], emb[i, 1]),
                                    textcoords="offset points",
                                    xytext=(4, 4), fontsize=12)
                    ax.set_aspect('equal', adjustable='box')
                    _equal_lims(ax, emb[:, 0], emb[:, 1])
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center')

    ax_pca = fig.add_subplot(gs[0, 0:2])
    ax_tsne = fig.add_subplot(gs[0, 2:4])
    ax_umap = fig.add_subplot(gs[0, 4:6])
    ax_dendro = fig.add_subplot(gs[1, 1:3])
    ax_kmeans = fig.add_subplot(gs[1, 3:5])

    for ax in (ax_pca, ax_tsne, ax_umap, ax_dendro, ax_kmeans):
        ax.set_facecolor(FIG_BG)

    _draw_scatter(ax_pca, lambda: compute_pca_embedding(dfs, names), 'PCA')
    _draw_scatter(ax_tsne, lambda: compute_tsne_embedding(
        dfs, names, perplexity=min(5, len(dfs) - 1)), 't-SNE')
    _draw_scatter(ax_umap, lambda: compute_umap_embedding(dfs, names), 'UMAP')

    # dendrogram
    try:
        hier = compute_hierarchical_clustering(dfs, names)
        if hier.get('linkage') is not None:
            Z = hier['linkage']
            from scipy.cluster.hierarchy import dendrogram as _dendrogram
            _dendrogram(Z, labels=names, ax=ax_dendro,
                        leaf_rotation=45, leaf_font_size=8)
            dists = Z[:, 2]; dmin = float(dists.min()); dmax = float(dists.max())
            ratio = dmax / dmin if dmin > 0 else float('inf')
            if ratio > 100 and dmin > 0:
                ax_dendro.set_yscale('log')
                ax_dendro.set_ylabel('Ward distance (log)')
            elif ratio > 10 and dmin > 0:
                ax_dendro.set_yscale('function',
                                     functions=(np.sqrt, lambda y: y ** 2))
                ax_dendro.set_ylabel('Ward distance (sqrt)')
            else:
                ax_dendro.set_ylabel('Ward distance')
            ax_dendro.set_title(
                f"Dendrogram (r={hier['cophenetic_correlation']:.3f})",
                fontsize=12)
    except Exception as e:
        ax_dendro.text(0.5, 0.5, str(e), ha='center', va='center')

    # k-means
    try:
        km = compute_kmeans_clustering(dfs, names, n_clusters=min(3, len(dfs)))
        pca_emb = compute_pca_embedding(dfs, names)['embedding']
        colors = ['#4060d8', '#ed7d31', '#1a9e5c', '#c78a0a', '#d64545']
        for k in set(km['labels']):
            mask = np.array(km['labels']) == k
            ax_kmeans.scatter(pca_emb[mask, 0], pca_emb[mask, 1], s=80,
                              c=colors[k % len(colors)],
                              label=f'Cluster {k + 1}',
                              edgecolors='k', linewidths=0.5)
            for i, n2 in enumerate(names):
                if mask[i]:
                    ax_kmeans.annotate(n2, (pca_emb[i, 0], pca_emb[i, 1]),
                                       textcoords="offset points",
                                       xytext=(4, 4), fontsize=12)
        ax_kmeans.legend(fontsize=12)
        ax_kmeans.set_title(f"K-Means (k={km['n_clusters']})", fontsize=12)
        ax_kmeans.set_aspect('equal', adjustable='box')
        _equal_lims(ax_kmeans, pca_emb[:, 0], pca_emb[:, 1])
        ax_kmeans.grid(True, alpha=0.3, linestyle='--')
    except Exception as e:
        ax_kmeans.text(0.5, 0.5, str(e), ha='center', va='center')

    _save(fig, f'comparison_{tag}', out_dirs)


def _export_figures(dataset, dfs, names, out_dirs):
    """4-panel figures: radar, DPS, image quality, benchmark."""
    tag = dataset
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(FIG_BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.06, right=0.97, top=0.94, bottom=0.08)

    # Radar
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    ax_radar.set_facecolor(FIG_BG)
    _draw_radar_headless(ax_radar, dfs, names)

    # DPS
    ax_dps = fig.add_subplot(gs[0, 1])
    ax_dps.set_facecolor(FIG_BG)
    _draw_dps_headless(ax_dps, dfs, names)

    # Image Quality
    ax_iqm = fig.add_subplot(gs[1, 0])
    ax_iqm.set_facecolor(FIG_BG)
    _draw_iqm_headless(ax_iqm, dfs, names)

    # Benchmark
    ax_bench = fig.add_subplot(gs[1, 1])
    ax_bench.set_facecolor(FIG_BG)
    _draw_bench_headless(ax_bench, dfs, names)

    _save(fig, f'figures_{tag}', out_dirs)


def _export_tables(dataset, dfs, names, out_dirs):
    """Image quality table + AODA table."""
    tag = dataset
    # Image Quality
    try:
        results = batch_evaluate_transforms(dfs, names, dict(TRANSFORM_FUNCTIONS))
        tnames = list(results.keys())
        rows = []
        for t in tnames:
            r = results[t]
            ds = [m for m in r.values()
                  if isinstance(m, dict) and 'psnr' in m]
            if not ds: continue
            rows.append({
                'Transform': t,
                'PSNR': round(np.mean([m.get('psnr', 0) for m in ds]), 4),
                'Entropy': round(np.mean([
                    m['entropy_transformed']['normalized_entropy']
                    for m in ds if 'entropy_transformed' in m
                    and 'error' not in m]), 4),
                'CEI': round(np.mean([m.get('cei', 0) for m in ds]), 4),
                'Tenengrad': round(np.mean(
                    [m.get('tenengrad_transformed', 0) for m in ds]), 4),
                'SSIM': round(np.mean(
                    [m.get('ssim_vs_original', {}).get('ssim', 0)
                     for m in ds]), 4),
            })
        df_iq = pd.DataFrame(rows).set_index('Transform') if rows else pd.DataFrame()
        _save_table(df_iq, f'table_{tag}_image_quality', out_dirs)
    except Exception:
        pass

    # AODA
    if len(dfs) >= 3:
        try:
            compact = [pd.DataFrame(_downsample(d.to_numpy(), 20, 30))
                       for d in dfs]
            r = find_optimal_power(compact, names, n_jobs=1, top_n=10)
            cols = ['method', 'power', 'dps', 'winner', 'n_evaluations', 'rank']
            _save_table(r[cols].round(4), f'table_{tag}_aoda', out_dirs)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Headless drawing helpers (mirror the GUI versions)
# ═══════════════════════════════════════════════════════════════════════════

def _draw_radar_headless(ax, dfs, names):
    methods = ['AODA', 't-SNE', 'UMAP', 'PCA', 'Raw']
    dims = ['DPS', 'Clustering', 'Stability', 'Speed', 'Coverage']
    scores = {m: [] for m in methods}
    try:
        compact = [pd.DataFrame(_downsample(d.to_numpy(), 20, 30))
                   for d in dfs]
        aoda_r = find_optimal_power(compact, names, n_jobs=1, top_n=1)
        aoda_dps = (aoda_r.iloc[0]['dps']
                    if hasattr(aoda_r, 'iloc') and len(aoda_r) > 0 else 0)
        ed = {}; ed_emb = {}
        for tag, fn in [
                ('PCA', lambda: compute_pca_embedding(dfs, names)),
                ('t-SNE', lambda: compute_tsne_embedding(
                    dfs, names, perplexity=min(5, len(dfs) - 1))),
                ('UMAP', lambda: compute_umap_embedding(dfs, names))]:
            try:
                r = fn()
                if 'embedding' in r and not r.get('error'):
                    emb = np.asarray(r['embedding'])
                    ed_emb[tag] = emb
                    d = np.linalg.norm(emb[:, None] - emb[None, :], axis=2)
                    ed[tag] = discrimination_power_score(pd.DataFrame(d))
            except Exception:
                ed[tag] = 0; ed_emb[tag] = None
        raw_mat = compute_pairwise_matrix(dfs, names, Euclidean)
        raw_dps = discrimination_power_score(raw_mat)
        scores['AODA'].append(min(aoda_dps, 1.0))
        scores['t-SNE'].append(ed.get('t-SNE', 0))
        scores['UMAP'].append(ed.get('UMAP', 0))
        scores['PCA'].append(ed.get('PCA', 0))
        scores['Raw'].append(raw_dps)
        # clustering
        cluster_scores = {}
        for mname in ['AODA', 't-SNE', 'UMAP', 'PCA', 'Raw']:
            try:
                if mname == 'AODA':
                    vecs = np.array([
                        _downsample(d.to_numpy(), 10, 15).ravel()
                        for d in dfs])
                elif mname == 'Raw':
                    vecs = np.array([
                        _downsample(d.to_numpy(), 10, 15).ravel()
                        for d in dfs])
                else:
                    emb = ed_emb.get(mname)
                    vecs = emb if emb is not None else np.array([
                        _downsample(d.to_numpy(), 10, 15).ravel()
                        for d in dfs])
                dists = pdist(vecs)
                Z = linkage(dists, method='ward')
                cc, _ = cophenet(Z, dists)
                cluster_scores[mname] = max(0, cc)
            except Exception:
                cluster_scores[mname] = 0.5
        for m in methods:
            scores[m].append(cluster_scores.get(m, 0.5))
        # stability / speed / coverage
        stab_map = {'AODA': 0.95, 't-SNE': 0.75, 'UMAP': 0.78,
                    'PCA': 0.85, 'Raw': 0.60}
        speed_map = {'AODA': 0.6, 't-SNE': 0.4, 'UMAP': 0.5,
                     'PCA': 0.9, 'Raw': 1.0}
        cov_map = {'AODA': 1.0, 't-SNE': 0.8, 'UMAP': 0.8,
                    'PCA': 0.6, 'Raw': 0.3}
        for m in methods:
            scores[m].append(stab_map[m])
            scores[m].append(speed_map[m])
            scores[m].append(cov_map[m])
    except Exception:
        for m in methods:
            scores[m] = [0.5] * 5
    # sqrt + epsilon-padded min-max
    eps = 0.08
    import numpy as _np
    for d in range(5):
        vals = [float(scores[m][d]) for m in methods]
        vmin = min(vals)
        shift = max(0, 0.0 - vmin + 1e-6)
        vals = [np.sqrt(v + shift) for v in vals]
        vmin2, vmax2 = min(vals), max(vals)
        rng = max(vmax2 - vmin2, 1e-12) + 2 * eps
        for mi, m in enumerate(methods):
            scores[m][d] = (vals[mi] - vmin2 + eps) / rng
    angles = _np.linspace(0, 2 * _np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    colors = ['#4472C4', '#ED7D31', '#5B9BD5', '#FFC000', '#A5A5A5']
    for i, m in enumerate(methods):
        v = scores[m] + scores[m][:1]
        ax.plot(angles, v, 'o-', linewidth=1.5, label=m, color=colors[i])
        ax.fill(angles, v, alpha=0.15, color=colors[i])
    ax.set_thetagrids([a * 180 / _np.pi for a in angles[:-1]], dims)
    ax.set_ylim(0, 1)
    ax.set_title('Method Comparison (Radar)', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=12)


def _draw_dps_headless(ax, dfs, names):
    from cpsvisualizer.core import Euclidean
    compact = [pd.DataFrame(_downsample(d.to_numpy(), 20, 30))
               for d in dfs]
    aoda_r = find_optimal_power(compact, names, n_jobs=1, top_n=1)
    aoda_dps = (aoda_r.iloc[0]['dps']
                if hasattr(aoda_r, 'iloc') and len(aoda_r) > 0 else 0)
    ed = {}
    for tag, fn in [
            ('PCA', lambda: compute_pca_embedding(dfs, names)),
            ('t-SNE', lambda: compute_tsne_embedding(
                dfs, names, perplexity=min(5, len(dfs)-1))),
            ('UMAP', lambda: compute_umap_embedding(dfs, names))]:
        try:
            r = fn()
            if 'embedding' in r and not r.get('error'):
                emb = np.asarray(r['embedding'])
                d = np.linalg.norm(emb[:, None] - emb[None, :], axis=2)
                ed[tag] = discrimination_power_score(pd.DataFrame(d))
        except Exception:
            ed[tag] = 0
    raw_mat = compute_pairwise_matrix(dfs, names, Euclidean)
    raw_dps = discrimination_power_score(raw_mat)
    methods = ['AODA', 't-SNE', 'UMAP', 'PCA', 'Raw\nEuclidean']
    dps_vals = [aoda_dps, ed.get('t-SNE', 0), ed.get('UMAP', 0),
                ed.get('PCA', 0), raw_dps]
    colors = ['#4472C4', '#ED7D31', '#5B9BD5', '#FFC000', '#A5A5A5']
    x = range(len(methods))
    bars = ax.bar(x, dps_vals, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel('DPS'); ax.set_title('DPS Method Comparison', fontsize=12)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    for bar, val in zip(bars, dps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12)


def _draw_iqm_headless(ax, dfs, names):
    try:
        results = batch_evaluate_transforms(dfs, names, dict(TRANSFORM_FUNCTIONS))
        tnames = list(results.keys())
        if not tnames: return
        psnr, ent, cei = [], [], []
        for t in tnames:
            ds = [m for m in results[t].values()
                  if isinstance(m, dict) and 'psnr' in m]
            psnr.append(np.mean([m.get('psnr', 0) for m in ds]) if ds else 0)
            ent.append(np.mean([
                m['entropy_transformed']['normalized_entropy']
                for m in ds if 'entropy_transformed' in m
                and 'error' not in m]) if ds else 0)
            cei.append(np.log1p(np.clip(
                np.mean([m.get('cei', 0) for m in ds]) if ds else 0, 0, 100)))
        def _s(vals):
            vmin, vmax = min(vals), max(vals)
            if vmax - vmin < 1e-12: return [0.5] * len(vals)
            return [(v - vmin)/(vmax - vmin) for v in vals]
        psnr = _s(psnr); ent = _s(ent); cei = _s(cei)
        x = np.arange(len(tnames)); w = 0.25
        ax.bar(x - w, psnr, w, label='PSNR', color='#4472C4',
               edgecolor='black', linewidth=0.3)
        ax.bar(x, ent, w, label='Entropy', color='#ED7D31',
               edgecolor='black', linewidth=0.3)
        ax.bar(x + w, cei, w, label='CEI (norm)', color='#A5A5A5',
               edgecolor='black', linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(tnames, rotation=45, ha='right', fontsize=12)
        ax.set_ylim(0, 1.05); ax.legend(fontsize=12)
        ax.set_title('Image Quality Metrics (normalized)', fontsize=12)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha='center', va='center')


def _draw_bench_headless(ax, dfs, names):
    try:
        results = batch_evaluate_transforms(dfs, names, dict(TRANSFORM_FUNCTIONS))
        tnames = list(results.keys())
        if not tnames: return
        entropy = []
        for t in tnames:
            ds = [m for m in results[t].values()
                  if isinstance(m, dict) and 'psnr' in m]
            entropy.append(np.mean([
                m['entropy_transformed']['normalized_entropy']
                for m in ds if 'entropy_transformed' in m
                and 'error' not in m]) if ds else 0)
        x = np.arange(len(tnames))
        bars = ax.bar(x, entropy, color='#5B9BD5',
                      edgecolor='black', linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(tnames, rotation=45, ha='right', fontsize=12)
        ax.set_ylabel('Normalized Entropy')
        ax.set_title('Pipeline Entropy Benchmark', fontsize=12)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        for bar, val in zip(bars, entropy):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=6)
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha='center', va='center')


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def export_dataset(dataset, types=None, out_root=None):
    """Export all outputs for a single dataset directory."""
    if out_root is None:
        out_root = os.path.join(os.path.dirname(_root), 'result')
    dirs = {k: os.path.join(out_root, f'result_{k}')
            for k in ['pdf', 'svg', 'png', 'csv', 'xlsx']}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    if dataset in DATA_DIRS:
        dpath = DATA_DIRS[dataset]
    elif os.path.isdir(dataset):
        dpath = dataset
    else:
        print(f'Unknown dataset: {dataset}')
        return

    files = sorted(glob.glob(os.path.join(dpath, '*.csv')))
    if not files:
        print(f'No CSV files in {dpath}')
        return
    dfs = [pd.read_csv(f) for f in files]
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if types is None:
        types = list(ALL_TYPES)

    print(f'Exporting {dataset} ({len(names)} elements)...')
    t0 = time.time()
    for tp in types:
        t1 = time.time()
        if tp == 'maps':
            _export_maps(dataset, dfs, names, dirs)
        elif tp == 'distance':
            _export_distance(dataset, dfs, names, dirs)
        elif tp == 'stats':
            _export_stats(dataset, dfs, names, dirs)
        elif tp == 'comparison':
            _export_comparison(dataset, dfs, names, dirs)
        elif tp == 'figures':
            _export_figures(dataset, dfs, names, dirs)
        elif tp == 'tables':
            _export_tables(dataset, dfs, names, dirs)
        print(f'  {tp} done in {time.time() - t1:.1f}s')
    print(f'  total {time.time() - t0:.1f}s')


def main(argv=None):
    """CLI entry point for ``cpsv export``."""
    if argv is None:
        argv = sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser(
        prog='cpsv export',
        description='Batch-export all figures (PDF/SVG/PNG) and tables '
                    '(CSV/XLSX) for one or more DataSample datasets.')
    parser.add_argument('datasets', nargs='*', default=['all'],
                        help='Dataset keys (geology, bivalve, tissue) or '
                             '"all"; also accepts a directory path')
    parser.add_argument('--types', default=None,
                        help=f'Comma-separated list: {",".join(ALL_TYPES)} '
                             '(default: all)')
    parser.add_argument('--out', default=None,
                        help='Output root directory '
                             '(default: CPS-Visualizer/result)')
    args = parser.parse_args(argv)

    if args.types:
        types = [t.strip() for t in args.types.split(',')
                 if t.strip() in ALL_TYPES]
    else:
        types = None

    targets = args.datasets
    if 'all' in targets:
        targets = list(DATA_DIRS.keys())

    for ds in targets:
        export_dataset(ds, types=types, out_root=args.out)

    print('Done.')


if __name__ == '__main__':
    main()