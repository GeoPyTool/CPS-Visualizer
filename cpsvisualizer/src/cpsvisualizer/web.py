"""
CPS-Visualizer Web — Flask-based interactive web interface.

Drag-and-drop CSV/XLSX files, choose transforms and distance metrics,
and explore elegant Plotly visualizations with overlay modes.

Run::

    python -c "import cpsvisualizer; cpsvisualizer.web()"
    # then open http://127.0.0.1:5005

Or::

    python -m cpsvisualizer --web
"""
import io
import os
import math
import json
import base64
import warnings
import traceback
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_file, render_template, Response

try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False

from cpsvisualizer.core import (
    TRANSFORM_FUNCTIONS, DISTANCE_FUNCTIONS, DISTANCE_NAMES,
    apply_transforms, compute_pairwise_matrix, clean_name, load_data_files,
)
from cpsvisualizer.statistics import (
    compute_pca, compute_pearson_correlation_matrix,
    compute_spearman_correlation_matrix, compute_anova,
    compute_uncertainty_all, compute_descriptive_statistics_all,
)
from cpsvisualizer.metrics import (
    compute_all_image_metrics, batch_evaluate_transforms,
)
from cpsvisualizer.comparison import (
    compute_pca_embedding, compute_tsne_embedding, compute_umap_embedding,
    compute_hierarchical_clustering, compute_kmeans_clustering,
    compute_all_comparisons,
)
from cpsvisualizer.adaptive import (
    find_optimal_pipeline, compute_comprehensive_benchmark,
    benchmark_summary_table, PIPELINE_COMBOS,
    find_optimal_power, optimize_power_parameter,
)

warnings.filterwarnings("ignore")

# Repository root (parent of the package) — hosts bundled sample data
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# ---------------------------------------------------------------------------
# In-memory session store (simple, single-user local server)
# ---------------------------------------------------------------------------
_STORE = {
    'datasets': {},      # name -> {'values': ndarray, 'shape': [r,c]}
    'order': [],         # preserve upload order
    '_version': 0,       # increments on upload/clear/remove to invalidate cache
    '_cache': {},        # endpoint result cache keyed by (version, endpoint, args...)
}


def _cache_key(endpoint, *args):
    return (endpoint, _STORE['_version']) + args


def _cache_get(key):
    return _STORE['_cache'].get(key, _MISSING)


_MISSING = object()


def _cache_put(key, value):
    # Simple LRU-ish cap: keep cache bounded
    _STORE['_cache'][key] = value
    if len(_STORE['_cache']) > 256:
        _STORE['_cache'] = dict(list(_STORE['_cache'].items())[-192:])


def _invalidate():
    _STORE['_version'] += 1
    _STORE['_cache'].clear()


def _to_jsonable(obj):
    """Recursively convert numpy / pandas objects to JSON-safe types.
    Optimised for Python 3.12+ — uses isinstance checks and direct float()."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.values.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _downsample(arr, max_rows=360, max_cols=720):
    """Vectorized block-average downsample for fast browser rendering.
    Every output cell is the mean of its source block, preserving data shape.
    Returns a C-contiguous array for best downstream performance."""
    r, c = arr.shape
    if r <= max_rows and c <= max_cols:
        return np.ascontiguousarray(arr)
    step_r = max(1, int(math.ceil(r / max_rows)))
    step_c = max(1, int(math.ceil(c / max_cols)))
    if step_r == 1 and step_c == 1:
        return np.ascontiguousarray(arr)
    nr = int(math.ceil(r / step_r))
    nc = int(math.ceil(c / step_c))
    # Pad to a multiple of the block size using edge replication,
    # then reshape + mean — fully vectorized.
    pr = nr * step_r - r
    pc = nc * step_c - c
    if pr > 0 or pc > 0:
        arr = np.pad(arr, ((0, pr), (0, pc)), mode='edge')
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    view = arr[:nr * step_r, :nc * step_c].reshape(nr, step_r, nc, step_c)
    return np.ascontiguousarray(view.mean(axis=(1, 3)))


def _df_from_store(name):
    return pd.DataFrame(_STORE['datasets'][name]['values'])


def _df_list_all():
    return [pd.DataFrame(_STORE['datasets'][n]['values']) for n in _STORE['order']]


def _names_all():
    return list(_STORE['order'])


def create_app(host='127.0.0.1', port=5005, debug=False):
    app = Flask(__name__, static_folder='static', template_folder='templates')
    if _HAS_CORS:
        CORS(app)
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256MB

    # ------------------------------------------------------------------
    # Routes — pages
    # ------------------------------------------------------------------
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/health')
    def health():
        return jsonify({'ok': True})

    # ------------------------------------------------------------------
    # API — methods catalog
    # ------------------------------------------------------------------
    @app.route('/api/methods', methods=['GET'])
    def methods():
        return jsonify({
            'transforms': list(TRANSFORM_FUNCTIONS.keys()),
            'distances': DISTANCE_NAMES,
            'pipelines': ['+'.join(p) for p in PIPELINE_COMBOS],
            'samples': [
                {'id': 'geology', 'label': 'Geological (Ag/Au/Cu/Fe/Pb/Zn)', 'dir': 'DataSample/Geology'},
                {'id': 'bivalve', 'label': 'Marine Biology — bivalve shell element/Ca (Zenodo)', 'dir': 'DataSample/BivalveShell_MarineBiology'},
                {'id': 'archaeology', 'label': 'Archaeology — medieval ceramics trace elements (Zenodo)', 'dir': 'DataSample/Ceramics_Archaeology'},
                {'id': 'biomedical', 'label': 'Biomedical — LA-ICP-MS tissue imaging (Zenodo)', 'dir': 'DataSample/Tissue_Biomedical'},
            ],
        })

    @app.route('/api/sample/<sample_id>', methods=['POST'])
    def sample_load(sample_id):
        """Load a bundled sample suite from the repository data directories."""
        samples = {
            'geology': os.path.join(_REPO_ROOT, 'DataSample', 'Geology'),
            'bivalve': os.path.join(_REPO_ROOT, 'DataSample', 'BivalveShell_MarineBiology'),
            'archaeology': os.path.join(_REPO_ROOT, 'DataSample', 'Ceramics_Archaeology'),
            'biomedical': os.path.join(_REPO_ROOT, 'DataSample', 'Tissue_Biomedical'),
        }
        folder = samples.get(sample_id)
        if not folder or not os.path.isdir(folder):
            return jsonify({'error': f'Unknown sample {sample_id}'}), 404
        added = []
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith('.csv'):
                continue
            try:
                df = pd.read_csv(os.path.join(folder, fn))
            except Exception as e:
                return jsonify({'error': f'Failed to read {fn}: {e}'}), 400
            name = clean_name(fn)
            base, k = name, 1
            while name in _STORE['datasets']:
                k += 1
                name = f'{base}_{k}'
            arr = df.values.astype(float)
            _STORE['datasets'][name] = {'values': arr.tolist(), 'shape': list(arr.shape)}
            if name not in _STORE['order']:
                _STORE['order'].append(name)
            added.append({'name': name, 'shape': list(arr.shape)})
        _invalidate()
        return jsonify({'added': added, 'all': _names_all()})

    # ------------------------------------------------------------------
    # API — upload (multipart files)
    # ------------------------------------------------------------------
    @app.route('/api/upload', methods=['POST'])
    def upload():
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        added = []
        for f in files:
            fn = f.filename
            if not fn:
                continue
            ext = os.path.splitext(fn)[1].lower()
            try:
                if ext == '.csv':
                    df = pd.read_csv(io.BytesIO(f.read()))
                elif ext in ('.xls', '.xlsx'):
                    df = pd.read_excel(io.BytesIO(f.read()))
                else:
                    continue
            except Exception as e:
                return jsonify({'error': f'Failed to read {fn}: {e}'}), 400
            name = clean_name(fn)
            # de-duplicate name
            base, k = name, 1
            while name in _STORE['datasets']:
                k += 1
                name = f'{base}_{k}'
            arr = df.values.astype(float)
            _STORE['datasets'][name] = {'values': arr.tolist(), 'shape': list(arr.shape)}
            if name not in _STORE['order']:
                _STORE['order'].append(name)
            added.append({'name': name, 'shape': list(arr.shape)})
        _invalidate()
        return jsonify({'added': added, 'all': _names_all()})

    @app.route('/api/datasets', methods=['GET'])
    def datasets():
        out = []
        for n in _STORE['order']:
            d = _STORE['datasets'][n]
            out.append({'name': n, 'shape': d['shape']})
        return jsonify(out)

    @app.route('/api/clear', methods=['POST'])
    def clear():
        _STORE['datasets'].clear()
        _STORE['order'].clear()
        _invalidate()
        return jsonify({'ok': True})

    @app.route('/api/remove/<name>', methods=['POST'])
    def remove(name):
        if name in _STORE['datasets']:
            del _STORE['datasets'][name]
            _STORE['order'].remove(name)
        _invalidate()
        return jsonify({'all': _STORE['order']})

    # ------------------------------------------------------------------
    # API — transform preview (returns processed matrix as lists)
    # ------------------------------------------------------------------
    @app.route('/api/transform', methods=['POST'])
    def transform():
        body = request.get_json(force=True)
        name = body.get('name')
        transforms = body.get('transforms', [])
        if name not in _STORE['datasets']:
            return jsonify({'error': f'Unknown dataset {name}'}), 404
        arr = np.array(_STORE['datasets'][name]['values'], dtype=float)
        try:
            out = apply_transforms(arr, transforms)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        return jsonify({'values': out.tolist(), 'shape': list(out.shape)})

    # ------------------------------------------------------------------
    # API — distance matrix
    # ------------------------------------------------------------------
    @app.route('/api/distance', methods=['POST'])
    def distance():
        body = request.get_json(force=True)
        metric = body.get('metric')
        names = body.get('names', _names_all())
        transforms = body.get('transforms', [])
        if metric not in {f.__name__ for f in DISTANCE_FUNCTIONS}:
            return jsonify({'error': f'Unknown metric {metric}'}), 400
        if not names:
            return jsonify({'error': 'No datasets selected'}), 400
        key = _cache_key('distance', metric, tuple(names), tuple(transforms))
        cached = _cache_get(key)
        if cached is not _MISSING:
            return jsonify(cached)
        try:
            dfs = []
            for n in names:
                arr = np.array(_STORE['datasets'][n]['values'], dtype=float)
                arr = apply_transforms(arr, transforms)
                # downsample so all datasets share a comparable, fast size
                arr = _downsample(arr, max_rows=120, max_cols=200)
                dfs.append(pd.DataFrame(arr))
            # align all datasets to a common shape (pad columns) so the
            # flattened pairwise metrics stay well-defined
            if dfs:
                common_cols = max(d.shape[1] for d in dfs)
                aligned = []
                for d in dfs:
                    if d.shape[1] < common_cols:
                        pad = pd.DataFrame(np.zeros((d.shape[0], common_cols - d.shape[1])))
                        aligned.append(pd.concat([d, pad], axis=1).iloc[:, :common_cols])
                    else:
                        aligned.append(d)
                dfs = aligned
            func = next(f for f in DISTANCE_FUNCTIONS if f.__name__ == metric)
            mat = compute_pairwise_matrix(dfs, names, func)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        payload = {
            'matrix': mat.values.tolist(),
            'labels': list(mat.index),
            'metric': metric,
        }
        _cache_put(key, payload)
        return jsonify(payload)

    @app.route('/api/distance/all', methods=['POST'])
    def distance_all():
        body = request.get_json(force=True)
        names = body.get('names', _names_all())
        transforms = body.get('transforms', [])
        if not names:
            return jsonify({'error': 'No datasets selected'}), 400
        key = _cache_key('distance_all', tuple(names), tuple(transforms))
        cached = _cache_get(key)
        if cached is not _MISSING:
            return jsonify(cached)
        dfs = []
        for n in names:
            arr = np.array(_STORE['datasets'][n]['values'], dtype=float)
            arr = apply_transforms(arr, transforms)
            # aggressive downsample: 60x80 is enough for distance heatmaps
            arr = _downsample(arr, max_rows=60, max_cols=80)
            dfs.append(pd.DataFrame(arr))
        out = {}
        from joblib import Parallel, delayed
        def _compute_one(func):
            try:
                mat = compute_pairwise_matrix(dfs, names, func)
                return func.__name__, mat.values.tolist()
            except Exception as e:
                return func.__name__, {'error': str(e)}
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(_compute_one)(func) for func in DISTANCE_FUNCTIONS
        )
        for name, value in results:
            out[name] = value
        result = {'labels': names, 'matrices': out}
        _cache_put(key, result)
        return jsonify(result)

    # ------------------------------------------------------------------
    # API — statistics
    # ------------------------------------------------------------------
    @app.route('/api/statistics', methods=['POST'])
    def statistics():
        body = request.get_json(force=True)
        names = body.get('names', _names_all())
        transforms = body.get('transforms', [])
        if not names:
            return jsonify({'error': 'No datasets'}), 400
        key = _cache_key('statistics', tuple(names), tuple(transforms))
        cached = _cache_get(key)
        if cached is not _MISSING:
            return jsonify(cached)
        dfs = []
        for n in names:
            arr = np.array(_STORE['datasets'][n]['values'], dtype=float)
            arr = apply_transforms(arr, transforms)
            arr = _downsample(arr, max_rows=100, max_cols=150)
            dfs.append(pd.DataFrame(arr))
        result = {}
        try:
            pca = compute_pca(dfs, names)
            result['pca'] = {
                'coords': pca['coords'].tolist(),
                'explained_variance': pca['explained_variance'],
                'names': pca['names'],
            }
        except Exception as e:
            result['pca'] = {'error': str(e)}
        try:
            corr, pv = compute_pearson_correlation_matrix(dfs, names)
            result['pearson'] = {'matrix': corr.values.tolist(),
                                 'pvalues': pv.values.tolist(), 'labels': names}
        except Exception as e:
            result['pearson'] = {'error': str(e)}
        try:
            corr, pv = compute_spearman_correlation_matrix(dfs, names)
            result['spearman'] = {'matrix': corr.values.tolist(),
                                  'pvalues': pv.values.tolist(), 'labels': names}
        except Exception as e:
            result['spearman'] = {'error': str(e)}
        try:
            result['anova'] = compute_anova(dfs, names)
        except Exception as e:
            result['anova'] = {'error': str(e)}
        try:
            result['uncertainty'] = compute_uncertainty_all(dfs, names)
        except Exception as e:
            result['uncertainty'] = {'error': str(e)}
        try:
            result['descriptive'] = compute_descriptive_statistics_all(dfs, names)
        except Exception as e:
            result['descriptive'] = {'error': str(e)}
        jsonable = _to_jsonable(result)
        _cache_put(key, jsonable)
        return jsonify(jsonable)

    # ------------------------------------------------------------------
    # API — image quality metrics
    # ------------------------------------------------------------------
    @app.route('/api/metrics', methods=['POST'])
    def metrics():
        body = request.get_json(force=True)
        names = body.get('names', _names_all())
        if not names:
            return jsonify({'error': 'No datasets'}), 400
        key = _cache_key('metrics', tuple(names))
        cached = _cache_get(key)
        if cached is not _MISSING:
            return jsonify(cached)
        dfs = []
        for n in names:
            arr = np.array(_STORE['datasets'][n]['values'], dtype=float)
            dfs.append(pd.DataFrame(_downsample(arr, max_rows=100, max_cols=150)))
        try:
            results = batch_evaluate_transforms(dfs, names, TRANSFORM_FUNCTIONS)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        # summarize per transform
        summary = {}
        for tname, ds_metrics in results.items():
            psnr_vals, ent_vals, cei_vals, teng_vals, ssim_vals = [], [], [], [], []
            for ds, m in ds_metrics.items():
                if 'error' in m:
                    continue
                psnr_vals.append(m.get('psnr', 0))
                ent_vals.append(m['entropy_transformed']['normalized_entropy'])
                cei_vals.append(m.get('cei', 0))
                teng_vals.append(m.get('tenengrad_transformed', 0))
                ssim_vals.append(m['ssim_vs_original']['ssim'])
            summary[tname] = {
                'psnr': float(np.mean(psnr_vals)) if psnr_vals else 0,
                'entropy': float(np.mean(ent_vals)) if ent_vals else 0,
                'cei': float(np.mean(cei_vals)) if cei_vals else 0,
                'tenengrad': float(np.mean(teng_vals)) if teng_vals else 0,
                'ssim': float(np.mean(ssim_vals)) if ssim_vals else 0,
            }
        jsonable = _to_jsonable(summary)
        _cache_put(key, jsonable)
        return jsonify(jsonable)

    # ------------------------------------------------------------------
    # API — comparison (PCA, t-SNE, UMAP, clustering)
    # ------------------------------------------------------------------
    @app.route('/api/comparison', methods=['POST'])
    def comparison():
        body = request.get_json(force=True)
        names = body.get('names', _names_all())
        transforms = body.get('transforms', [])
        if not names:
            return jsonify({'error': 'No datasets'}), 400
        key = _cache_key('comparison', tuple(names), tuple(transforms))
        cached = _cache_get(key)
        if cached is not _MISSING:
            return jsonify(cached)
        dfs = []
        for n in names:
            arr = np.array(_STORE['datasets'][n]['values'], dtype=float)
            arr = apply_transforms(arr, transforms)
            # aggressive downsample for fast t-SNE/UMAP
            arr = _downsample(arr, max_rows=30, max_cols=50)
            dfs.append(pd.DataFrame(arr))
        out = {'names': names}
        # Each comparison computed independently so one failure doesn't break all
        try:
            pca = compute_pca_embedding(dfs, names)
            out['pca'] = {'embedding': pca['embedding'].tolist(),
                          'explained_variance': pca['explained_variance']}
        except Exception as e:
            out['pca'] = {'error': str(e), 'embedding': [], 'explained_variance': []}
        try:
            tsne = compute_tsne_embedding(dfs, names)
            out['tsne'] = {'embedding': tsne['embedding'].tolist(),
                           'kl_divergence': tsne.get('kl_divergence', 0),
                           'error': tsne.get('error')}
        except Exception as e:
            out['tsne'] = {'embedding': [], 'error': str(e)}
        try:
            umap_r = compute_umap_embedding(dfs, names, init='random')
            out['umap'] = {'embedding': umap_r['embedding'].tolist(),
                           'error': umap_r.get('error')}
        except Exception as e:
            out['umap'] = {'embedding': [], 'error': str(e)}
        try:
            hier = compute_hierarchical_clustering(dfs, names)
            if hier.get('linkage') is not None:
                from scipy.cluster.hierarchy import dendrogram
                ddata = dendrogram(hier['linkage'], labels=names, no_plot=True)
                out['dendrogram'] = {
                    'x': ddata['dcoord'],
                    'y': ddata['icoord'],
                    'leaves': ddata['leaves'],
                    'labels': names,
                }
                out['cophenetic'] = hier.get('cophenetic_correlation', 0)
            else:
                out['dendrogram'] = None
                out['cophenetic'] = 0
        except Exception as e:
            out['dendrogram'] = None
            out['cophenetic'] = 0
            out['hier_error'] = str(e)
        try:
            km = compute_kmeans_clustering(dfs, names)
            out['kmeans'] = {
                'labels': km['labels'],
                'n_clusters': km['n_clusters'],
                'inertia': km['inertia'],
            }
        except Exception as e:
            out['kmeans'] = {'error': str(e), 'labels': [], 'n_clusters': 0, 'inertia': 0}
        jsonable = _to_jsonable(out)
        _cache_put(key, jsonable)
        return jsonify(jsonable)

    # ------------------------------------------------------------------
    # API — AODA adaptive optimal pipeline
    # ------------------------------------------------------------------
    @app.route('/api/adaptive', methods=['POST'])
    def adaptive():
        body = request.get_json(force=True)
        names = body.get('names', _names_all())
        if not names:
            return jsonify({'error': 'No datasets'}), 400
        key = _cache_key('adaptive', tuple(names))
        cached = _cache_get(key)
        if cached is not _MISSING:
            return jsonify(cached)
        dfs = []
        for n in names:
            arr = np.array(_STORE['datasets'][n]['values'], dtype=float)
            # minimal downsample for AODA — just enough for MI to be fast
            arr = _downsample(arr, max_rows=20, max_cols=30)
            dfs.append(pd.DataFrame(arr))
        try:
            # Golden-section + secant hybrid: continuous power optimisation.
            # Pre-screen all metrics cheaply, then fully optimise the top 8.
            result = find_optimal_power(dfs, names, n_jobs=1, verbose=False, top_n=8)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        payload = _to_jsonable({
            'method': 'continuous',
            'rankings': result[['method', 'power', 'dps', 'winner', 'n_evaluations', 'rank']]
                .to_dict('records'),
        })
        _cache_put(key, payload)
        return jsonify(payload)

    @app.route('/api/benchmark', methods=['POST'])
    def benchmark():
        body = request.get_json(force=True)
        names = body.get('names', _names_all())
        if not names:
            return jsonify({'error': 'No datasets'}), 400
        key = _cache_key('benchmark', tuple(names))
        cached = _cache_get(key)
        if cached is not _MISSING:
            return jsonify(cached)
        dfs = []
        for n in names:
            arr = np.array(_STORE['datasets'][n]['values'], dtype=float)
            arr = _downsample(arr, max_rows=20, max_cols=30)
            dfs.append(pd.DataFrame(arr))
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            try:
                result = compute_comprehensive_benchmark(dfs, names, n_jobs=-1)
            except Exception as e:
                return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400
        tbl = benchmark_summary_table(result)
        payload = {
            'summary': tbl.to_dict('records'),
            'log': buf_out.getvalue(),
        }
        _cache_put(key, payload)
        return jsonify(payload)

    # ------------------------------------------------------------------
    # API — overlay: composite multi-element visualization data
    # ------------------------------------------------------------------
    @app.route('/api/overlay', methods=['POST'])
    def overlay():
        body = request.get_json(force=True)
        names = body.get('names', [])
        transforms = body.get('transforms', [])
        mode = body.get('mode', 'rgb')  # rgb | alpha | difference | ratio
        if not names:
            return jsonify({'error': 'No datasets selected'}), 400
        if len(names) < 1:
            return jsonify({'error': 'Need at least 1 dataset'}), 400
        arrs = []
        for n in names:
            arr = np.array(_STORE['datasets'][n]['values'], dtype=float)
            arr = apply_transforms(arr, transforms)
            arr = _downsample(arr, max_rows=160, max_cols=260)
            arrs.append(arr)
        # align to min shape
        min_r = min(a.shape[0] for a in arrs)
        min_c = min(a.shape[1] for a in arrs)
        arrs = [a[:min_r, :min_c] for a in arrs]

        def norm01(a):
            mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
            return (a - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(a)

        normed = [norm01(a) for a in arrs]
        out = {}
        if mode == 'rgb':
            # map up to 3 elements to R, G, B
            rgb = np.zeros((min_r, min_c, 3))
            for i in range(min(3, len(normed))):
                rgb[..., i] = normed[i]
            out['image'] = (rgb * 255).clip(0, 255).astype(int).tolist()
            out['channels'] = names[:3]
        elif mode == 'alpha':
            # stack with alpha-blend (mean of normalized)
            blended = np.mean(np.stack(normed), axis=0)
            out['image'] = (blended * 255).clip(0, 255).astype(int).tolist()
            out['channels'] = names
        elif mode == 'difference' and len(normed) >= 2:
            diff = normed[0] - normed[1]
            out['image'] = diff.tolist()
            out['channels'] = names[:2]
        elif mode == 'ratio' and len(normed) >= 2:
            ratio = normed[0] / (normed[1] + 1e-12)
            out['image'] = ratio.tolist()
            out['channels'] = names[:2]
        else:
            out['image'] = (normed[0] * 255).astype(int).tolist()
            out['channels'] = [names[0]]
        out['shape'] = [min_r, min_c]
        out['mode'] = mode
        return jsonify(_to_jsonable(out))

    @app.route('/api/preview', methods=['POST'])
    def preview():
        """Return raw + transformed matrices for selected datasets."""
        body = request.get_json(force=True)
        names = body.get('names', _names_all())
        transforms = body.get('transforms', [])
        out = {}
        for n in names:
            arr = np.array(_STORE['datasets'][n]['values'], dtype=float)
            raw = _downsample(arr.copy(), max_rows=160, max_cols=260)
            trans = _downsample(apply_transforms(arr, transforms), max_rows=160, max_cols=260)
            out[n] = {
                'raw': raw.tolist(),
                'transformed': trans.tolist(),
                'shape': list(raw.shape),
            }
        return jsonify(_to_jsonable(out))

    return app


def main(host='127.0.0.1', port=5005, debug=False):
    app = create_app(host, port, debug)
    print(f'CPS-Visualizer Web → http://{host}:{port}')
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()