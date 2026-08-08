"""
Shared core module for CPS-Visualizer.
Contains all distance metrics, transform functions, and data utilities.
Used by both GUI (app.py) and CLI (app_cli.py) to eliminate code duplication.
"""
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity as ssim
from skimage import exposure


def log_transform(data):
    """Apply natural log transform: log1p(data - min + 1) to handle negatives and zeros."""
    offset = max(0, -data.min()) + 1
    return np.log1p(data + offset)


def centering_transform(data):
    """Center data by subtracting column-wise mean."""
    return data - np.mean(data, axis=0)


def log_centering_transform(data):
    """Apply log transform followed by centering."""
    offset = max(0, -data.min()) + 1
    log_data = np.log1p(data + offset)
    return log_data - np.mean(log_data, axis=0)


def z_score_normalization(data):
    """Z-score normalization: (x - mean) / std per column."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std


def standardize(data):
    """Standardize using sklearn StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def equalize_hist(data):
    """Histogram equalization for contrast enhancement."""
    return exposure.equalize_hist(data)


# ---------------------------------------------------------------------------
# Spatial filters (for denoising / edge emphasis) — shared by all interfaces
# ---------------------------------------------------------------------------
def gaussian_filter(data, sigma=1.0):
    """Gaussian smoothing to reduce noise."""
    from scipy.ndimage import gaussian_filter as _gf
    return _gf(np.nan_to_num(data.astype(float), nan=0.0), sigma=sigma)


def median_filter(data, size=3):
    """Median filter — edge-preserving noise reduction."""
    from scipy.ndimage import median_filter as _mf
    return _mf(np.nan_to_num(data.astype(float), nan=0.0), size=size)


def sobel_gradient(data):
    """Sobel gradient magnitude — emphasises sharp boundaries."""
    from scipy.ndimage import sobel
    d = np.nan_to_num(data.astype(float), nan=0.0)
    gx = sobel(d, axis=0)
    gy = sobel(d, axis=1)
    return np.hypot(gx, gy)


def unsharp_mask(data, radius=2.0, amount=1.0):
    """Unsharp masking — boosts local contrast / detail."""
    from skimage.filters import unsharp_mask as _um
    d = np.nan_to_num(data.astype(float), nan=0.0)
    return _um(d, radius=radius, amount=amount)


def normalize_01(data):
    """Min-max normalise to [0, 1]."""
    d = np.asarray(data, dtype=float)
    mn, mx = np.nanmin(d), np.nanmax(d)
    if mx - mn < 1e-12:
        return np.zeros_like(d)
    return (d - mn) / (mx - mn)


def clip_percentile(data, low=1.0, high=99.0):
    """Clip extreme values to the [low, high] percentile range (contrast stretch)."""
    d = np.asarray(data, dtype=float)
    lo, hi = np.percentile(d, [low, high])
    return np.clip(d, lo, hi)


FIG_BG = '#F5F7FA'


def ink_colormap(opaque_min=False, name='cps_ink'):
    """Grayscale ink colormap: the minimum value is colourless (transparent)
    and only the deepest values render as dark grayscale. With
    ``opaque_min=True`` the ramp starts at opaque white instead (used for
    overlay layers that must hide what is beneath them)."""
    from matplotlib.colors import LinearSegmentedColormap
    low = (1.0, 1.0, 1.0, 1.0) if opaque_min else (0.0, 0.0, 0.0, 0.0)
    cmap = LinearSegmentedColormap.from_list(
        name, [low, (0.0, 0.0, 0.0, 1.0)], N=256)
    cmap.set_under((0.0, 0.0, 0.0, 0.0))
    return cmap


def sci_colormap(name='cps_viridis'):
    """Perceptually-uniform sequential colormap (viridis-based) with
    transparent minimum — the gold standard for scientific heatmaps in
    top journals (Nature, Science, etc.).  Colour-blind friendly and
    monotonically increasing in lightness."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap('viridis')
    colors = list(base(np.linspace(0, 1, 256)))
    colors[0] = (0, 0, 0, 0)          # fully transparent
    for i in range(1, 10):
        r, g, b, _ = base(i / 255)
        colors[i] = (r, g, b, i / 20)  # fade in
    cmap = LinearSegmentedColormap.from_list(name, colors, N=256)
    cmap.set_under((0, 0, 0, 0))
    return cmap


def display_scale(data, low=1.0, high=99.0):
    """Robust display scaling: shift to non-negative, log1p-compress and
    window to the [low, high] percentile so outliers cannot dominate the
    colour mapping. Returns ``(scaled, vmin, vmax)``."""
    d = np.nan_to_num(np.asarray(data, dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    if d.size == 0:
        return d, 0.0, 1.0
    s = np.log1p(d - d.min())
    lo, hi = np.percentile(s, [low, high])
    if hi - lo < 1e-12:
        lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        lo, hi = 0.0, 1.0
    return s, lo, hi


TRANSFORM_FUNCTIONS = {
    'log_transform': log_transform,
    'centering_transform': centering_transform,
    'log_centering_transform': log_centering_transform,
    'z_score_normalization': z_score_normalization,
    'standardize': standardize,
    'equalize_hist': equalize_hist,
    'gaussian_filter': gaussian_filter,
    'median_filter': median_filter,
    'sobel_gradient': sobel_gradient,
    'unsharp_mask': unsharp_mask,
    'normalize_01': normalize_01,
    'clip_percentile': clip_percentile,
}


def apply_transforms(data, func_names):
    """Apply a sequence of transform functions to the data."""
    result = data.copy()
    for name in func_names:
        if name in TRANSFORM_FUNCTIONS:
            result = TRANSFORM_FUNCTIONS[name](result)
    return result


def clean_name(file_path):
    """Extract a clean dataset name from a file path."""
    tmp_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(tmp_name)[0]
    cleaned = name_without_ext.split('_')[0]
    return cleaned


def load_data_files(file_paths):
    """Load CSV or Excel files into a list of DataFrames with cleaned names."""
    df_list = []
    df_name_list = []
    for file_path in file_paths:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            continue
        df_list.append(df)
        df_name_list.append(clean_name(file_path))
    return df_list, df_name_list


# ---------------------------------------------------------------------------
# Distance and similarity metric functions
# ---------------------------------------------------------------------------

def Euclidean(df_A, df_B):
    return float(np.linalg.norm(df_A.values.ravel() - df_B.values.ravel()))


def Manhattan(df_A, df_B):
    return float(np.sum(np.abs(df_A.values.ravel() - df_B.values.ravel())))


def Chebyshev(df_A, df_B):
    return float(np.max(np.abs(df_A.values.ravel() - df_B.values.ravel())))


def Minkowski(df_A, df_B, p=2):
    """Generalized Minkowski distance. p=2 is Euclidean, p=1 is Manhattan."""
    diff = np.abs(df_A.values.ravel() - df_B.values.ravel())
    return float(np.sum(diff ** p) ** (1 / p))


def Cosine(df_A, df_B):
    A = df_A.values.ravel()
    B = df_B.values.ravel()
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    if denom == 0:
        return 0.0
    return float(1 - np.dot(A, B) / denom)


def Correlation(df_A, df_B):
    A = df_A.values.ravel()
    B = df_B.values.ravel()
    A_mean = A - np.mean(A)
    B_mean = B - np.mean(B)
    denom = np.linalg.norm(A_mean) * np.linalg.norm(B_mean)
    if denom == 0:
        return 0.0
    return float(1 - np.dot(A_mean, B_mean) / denom)


def Jaccard(df_A, df_B):
    A = df_A.values.ravel().astype(bool)
    B = df_B.values.ravel().astype(bool)
    intersection = np.sum(A & B)
    union = np.sum(A | B)
    return float(1 - intersection / union) if union > 0 else 1.0


def Dice(df_A, df_B):
    A = df_A.values.ravel().astype(bool)
    B = df_B.values.ravel().astype(bool)
    intersection = np.sum(A & B)
    denom = np.sum(A) + np.sum(B)
    return float(1 - (2 * intersection) / denom) if denom > 0 else 1.0


def Kulsinski(df_A, df_B):
    A = df_A.values.ravel().astype(bool)
    B = df_B.values.ravel().astype(bool)
    intersection = np.sum(A & B)
    n = len(A)
    sum_diff = np.sum(A != B)
    return float((n - intersection + sum_diff) / (n + sum_diff))


def Rogers_Tanimoto(df_A, df_B):
    A = df_A.values.ravel().astype(bool)
    B = df_B.values.ravel().astype(bool)
    n = len(A)
    return float((np.sum(A != B) + np.sum(~A & ~B)) / n)


def Russell_Rao(df_A, df_B):
    A = df_A.values.ravel().astype(bool)
    B = df_B.values.ravel().astype(bool)
    return float(np.sum(A & B) / len(A))


def Sokal_Michener(df_A, df_B):
    A = df_A.values.ravel().astype(bool)
    B = df_B.values.ravel().astype(bool)
    n = len(A)
    return float((np.sum(A == B) + np.sum(~A & ~B)) / n)


def Sokal_Sneath(df_A, df_B):
    A = df_A.values.ravel().astype(bool)
    B = df_B.values.ravel().astype(bool)
    intersection = np.sum(A & B)
    denom = np.sum(A) + np.sum(B)
    return float((2 * intersection) / denom) if denom > 0 else 0.0


def Yule(df_A, df_B):
    A = df_A.values.ravel().astype(bool)
    B = df_B.values.ravel().astype(bool)
    n = len(A)
    return float((np.sum(A & ~B) + np.sum(~A & B)) / n)


def Hsim_Distance(df_A, df_B):
    """Exponential soft-min distance: mean(exp(-|a-b|))."""
    a = df_A.values.ravel()
    b = df_B.values.ravel()
    differences = np.abs(a - b)
    exp_values = np.exp(-differences)
    return float(np.sum(exp_values) / min(len(a), len(b)))


def Close_Distance(df_A, df_B):
    """Exponential soft-min distance, identical to Hsim_Distance.
    Provided as an alternative name for backward compatibility."""
    return Hsim_Distance(df_A, df_B)


def mutual_info_score_unflattern(df_A, df_B):
    labels_true = df_A.values if isinstance(df_A, pd.DataFrame) else df_A
    labels_pred = df_B.values if isinstance(df_B, pd.DataFrame) else df_B
    n_features = labels_true.shape[1]
    mi_scores = Parallel(n_jobs=1, backend='threading')(
        delayed(mutual_info_score)(labels_true[:, i], labels_pred[:, i])
        for i in range(n_features)
    )
    return float(np.mean(mi_scores))


def mutual_info_score_flattern(df_A, df_B):
    data_A = df_A.values
    data_B = df_B.values
    min_len = min(data_A.shape[0], data_B.shape[0])
    truncated_A = data_A[:min_len, :].flatten()
    truncated_B = data_B[:min_len, :].flatten()
    return float(mutual_info_score(truncated_A, truncated_B))


def _pad_to_match(data_A, data_B):
    """Repeat the smaller dataset to match the larger one's row count."""
    if data_A.shape[0] > data_B.shape[0]:
        data_B_repeated = np.tile(
            data_B, (int(np.ceil(data_A.shape[0] / data_B.shape[0])), 1)
        )[:data_A.shape[0], :]
        return data_A, data_B_repeated
    else:
        data_A_repeated = np.tile(
            data_A, (int(np.ceil(data_B.shape[0] / data_A.shape[0])), 1)
        )[:data_B.shape[0], :]
        return data_A_repeated, data_B


def mutual_info_regression_unflattern(df_A, df_B):
    data_A = df_A.values
    data_B = df_B.values
    min_columns = min(data_A.shape[1], data_B.shape[1])
    data_A_rep, data_B_rep = _pad_to_match(data_A, data_B)
    mi_list = Parallel(n_jobs=1, backend='threading')(
        delayed(mutual_info_regression)(
            data_A_rep[:, i].reshape(-1, 1), data_B_rep[:, i]
        )
        for i in range(min_columns)
    )
    return float(np.mean([mi_r[0] for mi_r in mi_list]))


def mutual_info_regression_flattern(df_A, df_B):
    data_A = df_A.values
    data_B = df_B.values
    data_A_rep, data_B_rep = _pad_to_match(data_A, data_B)
    flattened_A = data_A_rep.flatten()
    flattened_B = data_B_rep.flatten()
    mi_r = mutual_info_regression(flattened_A.reshape(-1, 1), flattened_B)
    return float(mi_r[0])


def _ssim_data_range(img1, img2, method='max_range'):
    if method == 'max_range':
        dr1 = img1.max() - img1.min()
        dr2 = img2.max() - img2.min()
        return max(dr1, dr2)
    else:
        return max(img1.max(), img2.max()) - min(img1.min(), img2.min())


def calculate_ssim(df_A, df_B, method='max_range'):
    if df_A.shape != df_B.shape:
        raise ValueError("The shape of both dataframes must be the same")
    img1 = df_A.fillna(0).values
    img2 = df_B.fillna(0).values
    data_range = _ssim_data_range(img1, img2, method)
    ssim_value, _ = ssim(img1, img2, full=True, data_range=data_range)
    return ssim_value


def _ssim_components(img1, img2, method='max_range'):
    data_range = _ssim_data_range(img1, img2, method)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    C3 = C2 / 2
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1, sigma2 = np.std(img1), np.std(img2)
    covariance = np.mean((img1 - mu1) * (img2 - mu2))
    luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)
    structure = (covariance + C3) / (sigma1 * sigma2 + C3)
    return luminance, contrast, structure


def luminance(df_A, df_B, method='max_range'):
    if df_A.shape != df_B.shape:
        raise ValueError("The shape of both dataframes must be the same")
    return _ssim_components(df_A.values, df_B.values, method)[0]


def contrast(df_A, df_B, method='max_range'):
    if df_A.shape != df_B.shape:
        raise ValueError("The shape of both dataframes must be the same")
    return _ssim_components(df_A.values, df_B.values, method)[1]


def structure(df_A, df_B, method='max_range'):
    if df_A.shape != df_B.shape:
        raise ValueError("The shape of both dataframes must be the same")
    return _ssim_components(df_A.values, df_B.values, method)[2]


# New continuous-valued distance metrics (not binary)
def Bray_Curtis(df_A, df_B):
    """Bray-Curtis dissimilarity for continuous data."""
    A = df_A.values.ravel()
    B = df_B.values.ravel()
    numerator = np.sum(np.abs(A - B))
    denominator = np.sum(np.abs(A) + np.abs(B))
    return float(numerator / denominator) if denominator > 0 else 0.0


def Canberra(df_A, df_B):
    """Canberra distance: sum(|a-b|/(|a|+|b|))."""
    A = df_A.values.ravel()
    B = df_B.values.ravel()
    denom = np.abs(A) + np.abs(B)
    denom[denom == 0] = 1e-10
    return float(np.sum(np.abs(A - B) / denom))


DISTANCE_FUNCTIONS = [
    Euclidean, Manhattan, Chebyshev, Minkowski,
    Cosine, Correlation, Jaccard, Dice,
    Kulsinski, Rogers_Tanimoto, Russell_Rao,
    Sokal_Michener, Sokal_Sneath, Yule,
    Hsim_Distance, Close_Distance,
    mutual_info_score_unflattern, mutual_info_score_flattern,
    mutual_info_regression_unflattern, mutual_info_regression_flattern,
    calculate_ssim, luminance, contrast, structure,
    Bray_Curtis, Canberra,
]

DISTANCE_NAMES = [f.__name__ for f in DISTANCE_FUNCTIONS]


def _resample_1d(vec, length):
    """Resample a 1-D vector to ``length`` evenly-spaced samples (linear)."""
    vec = np.asarray(vec, dtype=float).ravel()
    vec = vec[np.isfinite(vec)]
    if vec.size == 0:
        return np.zeros(length, dtype=float)
    if vec.size == 1:
        return np.full(length, float(vec[0]))
    if length <= 1:
        return np.array([float(vec.mean())])
    src = np.linspace(0.0, 1.0, num=vec.size)
    dst = np.linspace(0.0, 1.0, num=length)
    return np.interp(dst, src, vec)


def compute_pairwise_matrix(df_list, df_name_list, func):
    """Compute an n x n symmetric pairwise distance/similarity matrix.

    Datasets may differ in shape; each matrix is flattened and resampled
    to the median flattened length before the metric is applied, so the
    matrix is well-defined across heterogeneous inputs.

    Args:
        df_list: list of pandas DataFrames
        df_name_list: list of string labels
        func: callable that takes (df_a, df_b) and returns a float

    Returns:
        pandas DataFrame with row/column labels
    """
    n = len(df_list)
    vecs = [df.values.ravel() for df in df_list]
    if vecs:
        length = int(np.median([v.size for v in vecs]))
        if length < 1:
            length = 1
        aligned = [pd.DataFrame(_resample_1d(v, length).reshape(-1, 1)) for v in vecs]
    else:
        aligned = []
    results = np.zeros((n, n))
    for i in range(n):
        results[i, i] = func(aligned[i], aligned[i])
        for j in range(i + 1, n):
            val = func(aligned[i], aligned[j])
            results[i, j] = val
            results[j, i] = val
    return pd.DataFrame(results, index=df_name_list, columns=df_name_list)


def compute_all_distances(df_list, df_name_list,
                          func_list=None, n_jobs=1):
    """Compute all distance matrices for a set of functions."""
    if func_list is None:
        func_list = DISTANCE_FUNCTIONS
    result_dict = {}
    for func in func_list:
        result_dict[func.__name__] = compute_pairwise_matrix(
            df_list, df_name_list, func
        )
    return result_dict
