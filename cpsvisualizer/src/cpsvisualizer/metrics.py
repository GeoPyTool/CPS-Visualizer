"""
Quantitative image quality metrics module for CPS-Visualizer.

Implements PSNR, entropy, contrast enhancement index (CEI),
multi-scale SSIM, and composite quality scores for objective
evaluation of LA-ICP-MS scan visualization quality.

Addresses Reviewer #3's request for quantitative performance evaluation.
"""
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from skimage import exposure
from scipy.stats import entropy as scipy_entropy


def compute_psnr(data, reference=None, data_range=None):
    """Peak Signal-to-Noise Ratio.

    If reference is None, uses the raw data as reference and
    treats the difference as noise (useful for comparing
    transformed vs raw data quality).
    """
    img = data.values if isinstance(data, pd.DataFrame) else data
    if reference is not None:
        ref = reference.values if isinstance(reference, pd.DataFrame) else reference
    else:
        ref = img.copy()
    if data_range is None:
        data_range = max(ref.max() - ref.min(), img.max() - img.min())
    return float(peak_signal_noise_ratio(ref, img, data_range=data_range))


def compute_entropy(data, bins=256):
    """Compute Shannon entropy of image intensity distribution.

    Higher entropy indicates richer information content.
    Optionally returns normalized entropy (divided by log2(bins)).
    """
    img = data.values.ravel() if isinstance(data, pd.DataFrame) else data.ravel()
    hist, _ = np.histogram(img, bins=bins, density=True)
    hist = hist[hist > 0]
    ent = float(scipy_entropy(hist, base=2))
    max_ent = np.log2(bins)
    normalized = ent / max_ent if max_ent > 0 else 0.0
    return {'entropy': ent, 'normalized_entropy': normalized,
            'max_possible': max_ent}


def compute_contrast_enhancement_index(original, enhanced):
    """Contrast Enhancement Index (CEI) comparing enhanced vs original.

    CEI = contrast_enhanced / contrast_original
    where contrast = std / mean (coefficient of variation).

    CEI > 1 indicates improved contrast.
    """
    orig = original.values.ravel() if isinstance(original, pd.DataFrame) else original.ravel()
    enh = enhanced.values.ravel() if isinstance(enhanced, pd.DataFrame) else enhanced.ravel()
    contrast_orig = np.std(orig) / (np.abs(np.mean(orig)) + 1e-10)
    contrast_enh = np.std(enh) / (np.abs(np.mean(enh)) + 1e-10)
    return float(contrast_enh / contrast_orig) if contrast_orig > 0 else 1.0


def compute_tenengrad(data):
    """Tenengrad sharpness measure using Sobel gradient magnitude.

    Higher values indicate sharper images with more edge information.
    """
    from scipy.ndimage import sobel
    img = data.values if isinstance(data, pd.DataFrame) else data
    gx = sobel(img, axis=0)
    gy = sobel(img, axis=1)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def compute_multiscale_ssim(data_a, data_b, data_range=None):
    """Multi-scale SSIM (if available) or standard SSIM with component breakdown.

    For data matrices, provides SSIM with luminance, contrast,
    and structure decomposition.
    """
    img1 = data_a.values if isinstance(data_a, pd.DataFrame) else data_a
    img2 = data_b.values if isinstance(data_b, pd.DataFrame) else data_b
    if data_range is None:
        dr1 = img1.max() - img1.min()
        dr2 = img2.max() - img2.min()
        data_range = max(dr1, dr2)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    C3 = C2 / 2
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1, sigma2 = np.std(img1), np.std(img2)
    cov = np.mean((img1 - mu1) * (img2 - mu2))
    lum = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    con = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)
    stru = (cov + C3) / (sigma1 * sigma2 + C3)
    ssim_val = lum * con * stru
    return {
        'ssim': float(ssim_val),
        'luminance': float(lum),
        'contrast': float(con),
        'structure': float(stru),
        'data_range': float(data_range),
    }


def compute_all_image_metrics(original, transformed, data_range=None):
    """Compute comprehensive image quality metrics comparing original
    to transformed data.

    Args:
        original: DataFrame or ndarray - raw data
        transformed: DataFrame or ndarray - processed data
        data_range: optional float

    Returns:
        dict with PSNR, entropy, CEI, Tenengrad, and SSIM scores
    """
    metrics = {}
    metrics['psnr'] = compute_psnr(transformed, original, data_range)
    metrics['entropy_original'] = compute_entropy(original)
    metrics['entropy_transformed'] = compute_entropy(transformed)
    metrics['cei'] = compute_contrast_enhancement_index(original, transformed)
    metrics['tenengrad_original'] = compute_tenengrad(original)
    metrics['tenengrad_transformed'] = compute_tenengrad(transformed)
    metrics['ssim_vs_original'] = compute_multiscale_ssim(
        original, transformed, data_range
    )
    return metrics


def batch_evaluate_transforms(df_list, df_name_list, transform_functions):
    """Evaluate multiple transform methods across all datasets.

    Args:
        df_list: list of DataFrames
        df_name_list: list of names
        transform_functions: dict of {name: callable} transform functions

    Returns:
        dict of {transform_name: {dataset_name: metrics_dict}}
    """
    results = {}
    for trans_name, trans_func in transform_functions.items():
        results[trans_name] = {}
        for ds_name, df in zip(df_name_list, df_list):
            try:
                transformed = trans_func(df.values.copy())
                transformed = pd.DataFrame(transformed)
                metrics = compute_all_image_metrics(df, transformed)
                results[trans_name][ds_name] = metrics
            except Exception as e:
                results[trans_name][ds_name] = {'error': str(e)}
    return results
