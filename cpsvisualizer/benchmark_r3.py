"""
【R3】Comprehensive quantitative performance benchmark for CPS-Visualizer transforms.
Measures all reviewer-requested metrics (PSNR, entropy, CEI, Tenengrad, SSIM)
plus runtime and operation counts on standard LA-ICP-MS data sizes.
Outputs publication-ready table and figure.
"""
import time, os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpsvisualizer', 'src'))

from cpsvisualizer.core import TRANSFORM_FUNCTIONS, apply_transforms
from cpsvisualizer.metrics import (
    compute_psnr, compute_entropy, compute_contrast_enhancement_index,
    compute_tenengrad, compute_all_image_metrics,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'CPS-Visualizer', 'figures')
os.makedirs(OUT, exist_ok=True)

COLORS = ['#4472C4','#ED7D31','#A5A5A5','#FFC000','#5B9BD5','#70AD47','#264478','#9B59B6','#E74C3C','#1ABC9C']

# ── Generate realistic test data ─────────────────────────────────────────────
RNG = np.random.default_rng(42)
ELEMENTS = ['Ag109','Cu65','Zn66','Fe57','Pb208','Au197']
ROWS, COLS = 60, 50  # standard LA-ICP-MS scan size
SIZES = [(30,25), (40,35), (50,45), (60,50), (80,60), (100,80)]  # multiple sizes for scaling analysis

datasets = {}
for i, el in enumerate(ELEMENTS):
    base = RNG.lognormal(mean=1.5+i*0.4, sigma=0.25+i*0.08, size=(ROWS, COLS))
    nx = 0.3 * np.sin(np.linspace(0, 4*np.pi, ROWS)).reshape(-1,1)
    ny = 0.2 * np.cos(np.linspace(0, 3*np.pi, COLS)).reshape(1,-1)
    datasets[el] = pd.DataFrame(base * (1+nx+ny) * (1+0.15*i))

PIPELINES = [
    ('Raw',               []),
    ('Log',               ['log_transform']),
    ('Centered',          ['centering_transform']),
    ('Z-Score',           ['z_score_normalization']),
    ('Standardized',      ['standardize']),
    ('Equalized',         ['equalize_hist']),
    ('Log + Equalized',   ['log_transform','equalize_hist']),
    ('Log + Centered',    ['log_transform','centering_transform']),
    ('Centered + Eq',     ['centering_transform','equalize_hist']),
    ('Z-Score + Eq',      ['z_score_normalization','equalize_hist']),
]

# ── Benchmark single size ────────────────────────────────────────────────────
def benchmark_size(rows, cols, datasets_dict, pipelines, n_repeat=5):
    N = rows * cols
    results = []
    for name, pipe in pipelines:
        psnr_l, ent_l, cei_l, ten_l, ssim_l, time_l = [], [], [], [], [], []
        for _ in range(n_repeat):
            for el, df in datasets_dict.items():
                raw = df.iloc[:rows, :cols]
                t0 = time.perf_counter()
                trans = pd.DataFrame(apply_transforms(raw.values.copy(), pipe)) if pipe else raw.copy()
                t1 = time.perf_counter()
                time_l.append(t1 - t0)
                m = compute_all_image_metrics(raw, trans)
                psnr_l.append(m['psnr'] if m['psnr'] != float('inf') else 999.0)
                ent_l.append(m['entropy_transformed']['normalized_entropy'])
                cei_l.append(m['cei'] if m['cei'] < 1e6 else float('nan'))
                ten_l.append(m['tenengrad_transformed'])
                ssim_l.append(m['ssim_vs_original']['ssim'])
        results.append({
            'Transform': name, 'Steps': len(pipe), 'Pixels': N,
            'PSNR_dB':  round(np.nanmean(psnr_l), 2),
            'Entropy':  round(np.nanmean(ent_l), 4),
            'CEI':      round(np.nanmean(cei_l), 3),
            'Tenengrad':round(np.nanmean(ten_l), 1),
            'SSIM':     round(np.nanmean(ssim_l), 4),
            'Time_us':  round(np.mean(time_l) * 1e6, 1),
            'Time_us_std': round(np.std(time_l) * 1e6, 1),
        })
    return pd.DataFrame(results)

# ── Run benchmarks ───────────────────────────────────────────────────────────
print('【R3】Benchmarking quantitative performance...')
df_main = benchmark_size(ROWS, COLS, datasets, PIPELINES, n_repeat=10)
df_main = df_main.sort_values('Entropy', ascending=False)

# Print table
print('\nTable 1. Quantitative performance of preprocessing transforms.')
print(f'Data: {ROWS}×{COLS} pixels ({ROWS*COLS} points) × {len(ELEMENTS)} elements, n=10 repeats.')
print(f'{"Transform":<22s} {"Steps":>5s} {"PSNR":>7s} {"Entropy":>8s} {"CEI":>8s} {"Tenengrad":>10s} {"SSIM":>8s} {"Time(μs)":>12s}')
print('-' * 90)
for _, r in df_main.iterrows():
    cei_str = f'{r["CEI"]:.3f}' if not np.isnan(r['CEI']) else 'N/A'
    print(f'{r["Transform"]:<22s} {r["Steps"]:>5d} {r["PSNR_dB"]:>7.2f} {r["Entropy"]:>8.4f} {cei_str:>8s} {r["Tenengrad"]:>10.1f} {r["SSIM"]:>8.4f} {r["Time_us"]:>7.1f} ± {r["Time_us_std"]:.1f}')

# ── Scaling analysis ─────────────────────────────────────────────────────────
print('\nScaling analysis (Log+Eq, varying matrix size):')
scale_rows = []
for rows, cols in SIZES:
    df_s = benchmark_size(rows, cols, datasets, [('Log+Eq', ['log_transform','equalize_hist'])], n_repeat=3)
    r = df_s.iloc[0]
    scale_rows.append({'Pixels': rows*cols, 'Time_us': r['Time_us'], 'Entropy': r['Entropy']})
    print(f'  {rows}×{cols} = {rows*cols:>5d} px → {r["Time_us"]:.0f} μs, entropy={r["Entropy"]:.4f}')

# Save table
df_main.to_csv(os.path.join(OUT, 'benchmark_metrics.csv'), index=False)
print(f'\nBenchmark table saved to {OUT}/benchmark_metrics.csv')

# ── Generate figure ──────────────────────────────────────────────────────────
valid = df_main[df_main['CEI'].notna() & (df_main['CEI'] < 100)].copy()
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
x = np.arange(len(valid))
names = valid['Transform'].tolist()
colors = COLORS[:len(names)]

metrics_plot = [
    (axes[0], valid['Entropy'].values,   'Normalized Entropy', 'Higher = richer information', 0, 1),
    (axes[1], valid['CEI'].values,       'Contrast Enhancement Index', '>1 = improved contrast', 0.5, max(valid['CEI'])*1.1),
    (axes[2], valid['Tenengrad'].values, 'Tenengrad Sharpness', 'Higher = more edge detail', 0, max(valid['Tenengrad'])*1.1),
    (axes[3], valid['SSIM'].values,      'SSIM vs Original', '1.0 = identical structure', 0, 1),
    (axes[4], valid['Time_us'].values,   'Time (μs)', 'Wall-clock, 3000 px × 6 elem', 0, max(valid['Time_us'])*1.1),
    (axes[5], valid['PSNR_dB'].values,   'PSNR (dB)', 'Higher = better fidelity', min(valid['PSNR_dB'])*0.9, max(valid['PSNR_dB'])*1.05),
]

for ax, vals, ylabel, subtitle, ymin, ymax in metrics_plot:
    ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.3, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(subtitle, fontsize=9, style='italic')
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', alpha=0.2, linestyle='--')

fig.suptitle('Figure 7. R3 Quantitative Performance Evaluation of Preprocessing Transforms',
             fontweight='bold', fontsize=14)
fig.tight_layout(rect=[0,0,1,0.95])
for fmt in ['.pdf','.png','.svg']:
    fig.savefig(os.path.join(OUT, f'fig7_benchmark{fmt}'), dpi=600 if fmt=='.png' else None)
plt.close(fig)
print(f'Figure saved to {OUT}/fig7_benchmark.*')

# ── Print LaTeX-ready table for manuscript ───────────────────────────────────
print('\n=== LATEX TABLE FOR MANUSCRIPT ===')
for _, r in df_main.iterrows():
    print(f'{r["Transform"]} & {r["Steps"]} & {r["PSNR_dB"]:.1f} & {r["Entropy"]:.4f} & {r["CEI"]:.3f} & {r["Tenengrad"]:.1f} & {r["SSIM"]:.4f} & {r["Time_us"]:.0f}\\\\pm{r["Time_us_std"]:.0f} \\\\')
