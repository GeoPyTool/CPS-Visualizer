"""
CPS-Visualizer CLI - command-line interface for batch-processing
LA-ICP-MS surface scan data with transforms, distance computation, and
vector-graphics output.
"""
import argparse
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpsvisualizer.core import (
    TRANSFORM_FUNCTIONS, DISTANCE_FUNCTIONS, DISTANCE_NAMES,
    load_data_files, apply_transforms, compute_pairwise_matrix,
    display_scale, ink_colormap, FIG_BG,
)
from cpsvisualizer.statistics import (
    compute_pca, compute_pearson_correlation_matrix,
    compute_spearman_correlation_matrix, compute_anova,
    compute_uncertainty_all, compute_descriptive_statistics_all,
)
from cpsvisualizer.metrics import (
    compute_psnr, compute_entropy, compute_contrast_enhancement_index,
    compute_tenengrad, compute_all_image_metrics, batch_evaluate_transforms,
)
from cpsvisualizer.comparison import (
    compute_pca_embedding, compute_tsne_embedding, compute_umap_embedding,
    compute_hierarchical_clustering, compute_kmeans_clustering,
    compute_all_comparisons, compute_method_comparison_metrics,
)

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 'truetype'

TRANSFORM_NAMES = list(TRANSFORM_FUNCTIONS.keys())
DISTANCE_DISPATCH = {f.__name__: f for f in DISTANCE_FUNCTIONS}


def _subplot_grid(n):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _square_aspect(n_rows, n_cols):
    """Matplotlib aspect that renders the whole plot as a square."""
    if n_rows == 0 or n_cols == 0:
        return 1.0
    return float(n_cols) / float(n_rows)


def _downsample_cli(arr, max_rows=120, max_cols=200):
    """Vectorized block-average downsample."""
    import numpy as np
    r, c = arr.shape
    if r <= max_rows and c <= max_cols:
        return np.ascontiguousarray(arr)
    sr = max(1, int(math.ceil(r / max_rows)))
    sc = max(1, int(math.ceil(c / max_cols)))
    nr = int(math.ceil(r / sr))
    nc = int(math.ceil(c / sc))
    pr = nr * sr - r
    pc = nc * sc - c
    if pr > 0 or pc > 0:
        arr = np.pad(arr, ((0, pr), (0, pc)), mode='edge')
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(
        arr[:nr * sr, :nc * sc].reshape(nr, sr, nc, sc).mean(axis=(1, 3)))


def _clean_name(path):
    return os.path.splitext(os.path.basename(path))[0].split('_')[0]


class CPS_CLI:
    """Headless batch processor for CPS-Visualizer."""

    def __init__(self):
        self.df_list = []
        self.df_name_list = []
        self.trans_df_list = []
        self.result_df_dict = {}
        self.trans_applied = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def open_files(self, paths):
        """Load CSV / Excel files and store with cleaned names."""
        self.df_list, self.df_name_list = load_data_files(paths)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def trans_data(self, func_names=None):
        """Apply a sequence of transforms to every loaded dataset."""
        if func_names is None:
            func_names = TRANSFORM_NAMES
        self.trans_df_list.clear()
        self.trans_applied.clear()
        for name, df in zip(self.df_name_list, self.df_list):
            arr = df.to_numpy()
            applied = []
            for fn in func_names:
                if fn in TRANSFORM_FUNCTIONS:
                    try:
                        arr = TRANSFORM_FUNCTIONS[fn](arr)
                        applied.append(fn)
                    except Exception as e:
                        applied.append(f'{fn}(FAILED: {e})')
            self.trans_applied[name] = applied
            self.trans_df_list.append(pd.DataFrame(arr))

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------
    def calc_data(self, func_names=None):
        """Compute pairwise distance matrices for selected metrics."""
        if func_names is None:
            func_names = DISTANCE_NAMES
        out_dir = os.getcwd()
        # downsample + align all datasets to a common shape so pairwise
        # distance functions work even when input matrices differ in size
        aligned = self._align_datasets(self.df_list)
        for fn in func_names:
            if fn not in DISTANCE_DISPATCH:
                continue
            if fn not in self.result_df_dict:
                self.result_df_dict[fn] = compute_pairwise_matrix(
                    aligned, self.df_name_list, DISTANCE_DISPATCH[fn])
            path = os.path.join(out_dir, f'{fn}.csv')
            try:
                self.result_df_dict[fn].to_csv(path, encoding='utf-8')
                print(f'{fn} -> {path}')
            except Exception as e:
                print(f'{fn} save failed: {e}')

    @staticmethod
    def _align_datasets(df_list, max_rows=120, max_cols=200):
        """Downsample and pad all datasets to a common shape."""
        import numpy as np
        from cpsvisualizer.app_cli import _downsample_cli
        downsampled = [pd.DataFrame(_downsample_cli(df.to_numpy(), max_rows, max_cols))
                       for df in df_list]
        if not downsampled:
            return downsampled
        common_cols = max(d.shape[1] for d in downsampled)
        aligned = []
        for d in downsampled:
            if d.shape[1] < common_cols:
                pad = pd.DataFrame(np.zeros((d.shape[0], common_cols - d.shape[1])))
                aligned.append(pd.concat([d, pad], axis=1).iloc[:, :common_cols])
            else:
                aligned.append(d)
        return aligned

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_data(self, show=True):
        """Display (or save) the transformed data matrices in a grid."""
        n = len(self.df_name_list)
        if n == 0:
            return
        rows, cols = _subplot_grid(n)
        figsize = (3 * cols, 3 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor(FIG_BG)
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        for i, name in enumerate(self.df_name_list):
            r, c = i // cols, i % cols
            ax = axes[r, c]
            arr = self.trans_df_list[i].to_numpy()
            s, lo, hi = display_scale(arr)
            ax.set_facecolor(FIG_BG)
            ax.imshow(s, cmap=ink_colormap(), vmin=lo, vmax=hi,
                      aspect=_square_aspect(*arr.shape))
            info = '\n'.join(self.trans_applied.get(name, []))
            ax.set_title(f'{name}\n{info}', fontsize=9)
        # Hide unused subplots
        for j in range(n, rows * cols):
            r, c = j // cols, j % cols
            fig.delaxes(axes[r, c])
        fig.canvas.manager.set_window_title('CPS Data Visualization') if hasattr(
            fig.canvas, 'manager') else None
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def silent_plot(self):
        """Save the current figure as PNG (600 dpi), PDF, and SVG."""
        base = os.path.abspath('CPS_Data_Visualization')
        plt.savefig(f'{base}.png', dpi=600)
        plt.savefig(f'{base}.pdf')
        plt.savefig(f'{base}.svg')
        print(f'PNG -> {base}.png')
        print(f'PDF -> {base}.pdf')
        print(f'SVG -> {base}.svg')
        plt.close()


def main(data_files=None, functions=None, mode='show'):
    """Entry point for the CPS-Visualizer CLI.

    Usage::

        import cpsvisualizer
        cpsvisualizer.cli('Ag.csv Cu.csv Zn.csv', 'log_transform equalize_hist Euclidean', 'silent')

    Or from the command line::

        python -m cpsvisualizer --cli "file1.csv file2.csv" "func1 func2" silent
    """
    app = CPS_CLI()

    if data_files is None or functions is None:
        if len(sys.argv) < 4:
            print("Usage: cpsvisualizer.cli('file1.csv file2.csv', 'trans1 trans2 metric1', 'silent|show')")
            sys.exit(1)
        data_files = sys.argv[1]
        functions = sys.argv[2]
        mode = sys.argv[3] if len(sys.argv) > 3 else 'show'

    paths = list(dict.fromkeys(data_files.split()))
    funcs = list(dict.fromkeys(functions.split()))

    trans = [f for f in funcs if f in TRANSFORM_FUNCTIONS]
    dists = [f for f in funcs if f in DISTANCE_DISPATCH]
    unknown = [f for f in funcs if f not in TRANSFORM_FUNCTIONS and f not in DISTANCE_DISPATCH]
    if unknown:
        print(f'Unknown functions (ignored): {unknown}')

    print(f'Data files       : {paths}')
    print(f'Transforms       : {trans}')
    print(f'Distance metrics : {dists}')
    print(f'Mode             : {mode}')

    app.open_files(paths)
    app.trans_data(trans)
    app.calc_data(dists)
    app.plot_data(show=(mode == 'show'))

    if mode == 'silent':
        app.silent_plot()


if __name__ == '__main__':
    main()
