"""Regenerate the GUI screenshots headlessly (QT_QPA_PLATFORM=offscreen).

The Map-Viewer / distance / statistics / quality / AODA screenshots use
the Geology pyrite dataset (same-shape element matrices).  The
11_comparison screenshot is generated once per DataSample sub-directory
(Geology, Bivalve shell, Tissue) so each shows a same-type,
cross-element comparison.
"""
import glob
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'cpsvisualizer', 'src'))

import pandas as pd
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from cpsvisualizer.app import CPSVisualizer

OUT = os.path.dirname(os.path.abspath(__file__))

GEO_DIR = os.path.join(ROOT, 'DataSample', 'Geology')
BIV_DIR = os.path.join(ROOT, 'DataSample', 'BivalveShell_MarineBiology')
TIS_DIR = os.path.join(ROOT, 'DataSample', 'Tissue_Biomedical')

geo_files = sorted(glob.glob(os.path.join(GEO_DIR, '*.csv')))


def load(paths):
    dfs = [pd.read_csv(p) for p in paths]
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return dfs, names


w = CPSVisualizer()
w.resize(1600, 1000)
w.df_list, w.df_name_list = load(geo_files)
w._data_sel.addItems(w.df_name_list)
w._fill_picks()
w._on_cmp_changed()
app.processEvents()


def grab(name):
    app.processEvents()
    w.grab().save(os.path.join(OUT, name))
    print('saved', name)


w._tabs.setCurrentIndex(0)
w._map_mode_sel.setCurrentText('single')
w._render_sel.setCurrentText('raw')
w._view_idx = 0
w._render_map()
grab('01_map_single_raw.png')

w._render_sel.setCurrentText('enhanced')
grab('02_map_enhanced.png')

w._render_sel.setCurrentText('filtered')
grab('03_map_filtered.png')

w._step_map(1)
grab('04_map_next.png')

w._render_sel.setCurrentText('raw')
w._map_mode_sel.setCurrentText('wipe')
w._pick_a.setCurrentText('Fe')
w._pick_b.setCurrentText('Pb')
w._wipe_slider.setValue(50)
grab('05_wipe_similar_fepb.png')

w._pick_a.setCurrentText('Ag')
w._pick_b.setCurrentText('Fe')
grab('06_wipe_different_agfe.png')

w._pick_a.setCurrentText('Au')
w._pick_b.setCurrentText('Zn')
grab('07_wipe_different_auzn.png')

w._map_mode_sel.setCurrentText('single')
w._tabs.setCurrentIndex(0)

# 08_plot_all: one element-map grid per DataSample sub-directory
# (Geology, Bivalve shell, Tissue).  Ceramics is a single sample x
# element table (not surface-scan matrices) and is skipped.
PLOT_ALL_DIRS = [
    ('geology', GEO_DIR),
    ('bivalve', BIV_DIR),
    ('tissue', TIS_DIR),
]

def _load_into(paths):
    w._on_clear()
    w.df_list, w.df_name_list = load(paths)
    w._data_sel.addItems(w.df_name_list)
    w._fill_picks()
    w._on_cmp_changed()
    app.processEvents()

for tag, directory in PLOT_ALL_DIRS:
    files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    if not files:
        continue
    _load_into(files)
    w._on_plot_all()
    grab(f'08_plot_all_{tag}.png')

# Reload Geology so the remaining screenshots (09-13) use the
# same dataset as the single/wipe maps above.
_load_into(geo_files)
w._render_sel.setCurrentText('raw')
w._view_idx = 0

# 09_distance: 2x3 grid of six representative normalized distance heatmaps,
# one figure per DataSample sub-directory (Geology, Bivalve, Tissue).
w._tabs.setCurrentIndex(1)
for tag, directory in PLOT_ALL_DIRS:
    files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    if not files:
        continue
    _load_into(files)
    w._on_cmp_changed()
    grab(f'09_distance_{tag}.png')

# Reload Geology for the remaining analysis screenshots.
_load_into(geo_files)
w._render_sel.setCurrentText('raw')
w._view_idx = 0

# Statistics / Image Quality / AODA: keep the Geology dataset loaded
# (same as the Map-Viewer screenshots above).
w._tabs.setCurrentIndex(2)
w._data_sel.clearSelection()
w._analysis_tabs.setCurrentIndex(0)
w._analysis_selected(0)
grab('10_statistics_pca.png')

w._analysis_tabs.setCurrentIndex(2)
w._analysis_selected(2)
grab('12_image_quality.png')
grab('13_aoda.png')

# Figures tab
w._analysis_tabs.setCurrentIndex(3)
w._analysis_selected(3)
grab('14_figures.png')

# 11_comparison: one screenshot per DataSample sub-directory (Geology,
# Bivalve shell, Tissue).  Ceramics_Archaeology is a single sample x
# element table (not surface-scan matrices) and needs >=2 datasets for
# a comparison, so it is skipped.
SUBDIRS = [
    ('geology', GEO_DIR),
    ('bivalve', BIV_DIR),
    ('tissue', TIS_DIR),
]

w._analysis_tabs.setCurrentIndex(1)
for tag, directory in SUBDIRS:
    files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    if len(files) < 2:
        continue
    w._on_clear()
    w.df_list, w.df_name_list = load(files)
    w._data_sel.addItems(w.df_name_list)
    w._fill_picks()
    w._on_cmp_changed()
    app.processEvents()
    w._data_sel.clearSelection()
    # bivalve needs Blended; others use Auto
    w._dendro_mode.setCurrentText('Blended' if tag == 'bivalve' else 'Auto')
    w._analysis_selected(1)
    grab(f'11_comparison_{tag}.png')