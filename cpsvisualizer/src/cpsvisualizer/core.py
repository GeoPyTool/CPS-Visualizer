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
    """Apply natural log transform with log1p to handle zeros safely."""
    return np.log1p(data)


def centering_transform(data):
    """Center data by subtracting column-wise mean."""
    return data - np.mean(data, axis=0)


def log_centering_transform(data):
    """Apply log transform followed by centering."""
    log_data = np.log1p(data)
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


TRANSFORM_FUNCTIONS = {
    'log_transform': log_transform,
    'centering_transform': centering_transform,
    'log_centering_transform': log_centering_transform,
    'z_score_normalization': z_score_normalization,
    'standardize': standardize,
    'equalize_hist': equalize_hist,
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
    mi_scores = Parallel(n_jobs=-1)(
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
    mi_list = Parallel(n_jobs=-1)(
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


def compute_pairwise_matrix(df_list, df_name_list, func):
    """Compute an n x n symmetric pairwise distance/similarity matrix.

    Args:
        df_list: list of pandas DataFrames
        df_name_list: list of string labels
        func: callable that takes (df_a, df_b) and returns a float

    Returns:
        pandas DataFrame with row/column labels
    """
    n = len(df_list)
    arrays = [df.values.ravel() for df in df_list]
    results = np.zeros((n, n))
    for i in range(n):
        A_arr = arrays[i]
        A_shape = df_list[i].shape
        for j in range(i, n):
            B_arr = arrays[j]
            B_shape = df_list[j].shape
            val = func(
                pd.DataFrame(A_arr.reshape(A_shape)),
                pd.DataFrame(B_arr.reshape(B_shape)),
            )
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
