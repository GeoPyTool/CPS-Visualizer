# CPS-Visualizer Fusion: Visual–Numerical Coordinate-Wise Enhancement

## Overview

The Fusion module implements **coordinate-wise visual–numerical fusion** as described in the paper. At every pixel coordinate \((i, j)\) of the LA-ICP-MS count matrix, two independent channels are computed and combined:

1. **Visual Enhancement Channel** \(V(i,j)\) — image-processing transforms that amplify spatial patterns visible to the human eye.
2. **Statistical Significance Channel** \(S(i,j)\) — per-pixel statistical scores that quantify how far each pixel deviates from the background distribution.

Both channels live on the same coordinate grid, enabling direct pixel-level fusion. The key insight (Reviewer Comment #2) is that pure visual enhancement may over-emphasise subjective patterns while neglecting statistically significant but visually subtle features; operating both channels on the same grid provides stronger quantitative evidence.

---

## Pipeline

### Step 1 — Data Loading

Up to three trace-element surface-scan matrices are selected (Element A / B / C). All share the same spatial coordinate system (identical `R × C` acquisition grid). A **Base** element (default: Fe) provides the geological structural context for the overlay panel.

### Step 2 — Visual Enhancement (Channel V)

For each selected element, the raw CPS count matrix undergoes:

\[
V = \text{equalize\_hist}\big(\log_{\text{transform}}(\text{raw})\big)
\]

- `log_transform(raw)` applies \(\ln(1 + \text{data})\) to compress the wide dynamic range of LA-ICP-MS counts (often 10⁶–10⁸ spans).
- `equalize_hist` performs histogram equalisation via the CDF mapping \(V(i,j) = \frac{\text{CDF}(x_{ij}) - \text{CDF}_{\min}}{1 - \text{CDF}_{\min}}\), spreading pixel intensities uniformly across the full 8-bit greyscale range.

The result is a contrast-maximised image where fine spatial structures become visible.

### Step 3 — Structural Contour Extraction

A reference crystal boundary is extracted from the **Fe** (iron) element, which serves as the pyrite matrix. The processing chain is:

1. **Heavy Gaussian smoothing** (\(\sigma = 6.0\)) — removes internal textural variation while preserving the macroscopic crystal outline.
2. **Downsampling** (\(80 \times 80\)) for computational efficiency.
3. **Otsu auto-thresholding** — finds the optimal intensity threshold that maximises inter-class variance:
   \[
   \sigma^2_B(t) = \omega_0(t)\,\omega_1(t)\,\big[\mu_0(t) - \mu_1(t)\big]^2
   \]
4. **Morphological cleanup** — binary closing (×2, 5×5 structure) fills gaps; binary opening (×1) removes isolated noise; `binary_fill_holes` ensures a solid mask.
5. **Largest connected component** — only the primary crystal body is retained.
6. **Final Gaussian smoothing** (\(\sigma = 2.0\)) on the binary mask before contour extraction.
7. **Contour tracing** via `skimage.measure.find_contours` at \(0.5\) level.

The resulting contour is a smooth, closed polygon representing the pyrite crystal boundary.

### Step 4 — Distribution Trajectory

For each element, the main spatial distribution path is computed as a **row-wise weighted centroid**:

\[
\bar{x}_r = \frac{\sum_{c=1}^{C} c \cdot d(r,c)}{\sum_{c=1}^{C} d(r,c)},\quad r = 1,\dots,R
\]

where \(d(r,c)\) is the CPS value at row \(r\), column \(c\). The trajectory \((\bar{x}_r, r)\) is smoothed with a moving-average filter (window = 5% of rows) to yield a continuous, interpretable path.

### Step 5 — Fractal Dimension (Box-Counting)

The structural complexity of each element's distribution is quantified via **box-counting fractal dimension**:

1. Apply threshold \(\mu + \sigma\) (mean + 1 standard deviation) to binarise the image.
2. At box sizes \(s \in \{2,4,8,\dots,2^{k}\}\), count the number \(N(s)\) of boxes containing at least one foreground pixel.
3. Fit \(\log N(s) = -D_f \cdot \log s + C\) via linear regression.
4. The slope \(D_f\) is the box-counting fractal dimension.

Higher \(D_f\) values indicate more spatially complex, space-filling distributions.

### Step 6 — Visualisation Layout (2 × 4 Grid)

The Fusion tab displays three element rows × four columns, all panels maintaining square aspect ratio:

| Column | Name | Content |
|--------|------|---------|
| 0 | **Raw** | Original data scaled via `display_scale` (log₁ₚ + percentile window 1–99). Greyscale ink colormap (colourless at minimum, dark at maximum). |
| 1 | **Enhanced + Contour** | The same data at reduced opacity (35%) with the **Fe crystal boundary** (white outline) and the element's **distribution trajectory** (white trace) overlaid. |
| 2 | **Original + Trace** | Full-opacity raw data with the Fe crystal boundary (white outline) and the element's trajectory (coloured, RGB per element) overlaid. |
| 3 | **Overlay (synthesis)** | The **Base element** grayscale map as background (default: Fe), with: (a) white Fe crystal boundary contour; (b) coloured trajectory traces from all selected elements (Red = A, Green = B, Blue = C); (c) fractal dimension comparison legend in the corner. |

### Step 7 — User Controls

| Control | Function |
|---------|----------|
| **A / B / C** | Element selectors (up to 3 trace elements) |
| **Colour buttons** ■ | Per-element RGB colour picker (QColorDialog) for trajectory trace colour |
| **Base** | Background element for the overlay panel (default: Fe) |
| **Res** | Super-resolution upsampling (1× / 2× / 4×) via cubic spline interpolation |

---

## Mathematical Summary

For element \(k\) with raw count matrix \(\mathbf{D}_k \in \mathbb{R}^{R \times C}\):

\[
\begin{aligned}
\text{Visual:}\quad & V_k = \text{histeq}\big(\log(1 + \mathbf{D}_k)\big) \\[4pt]
\text{Contour:}\quad & \mathcal{C}_{\text{Fe}} = \text{Otsu}\big(G_{\sigma=6}(\mathbf{D}_{\text{Fe}})\big) \\[4pt]
\text{Trajectory:}\quad & \mathbf{T}_k(r) = \frac{\sum_{c} c \cdot D_k(r,c)}{\sum_{c} D_k(r,c)} \\[4pt]
\text{Fractal:}\quad & D_f^{(k)} \text{ from box-counting on } \mathbf{D}_k > \mu_k + \sigma_k
\end{aligned}
\]

The overlay panel at coordinate \((i,j)\) renders:

\[
\text{Overlay}(i,j) = \underbrace{\mathbf{D}_{\text{base}}(i,j)}_{\text{grayscale bg}} \;\oplus\; \underbrace{\mathcal{C}_{\text{Fe}}(i,j)}_{\text{white contour}} \;\oplus\; \bigoplus_{k} \underbrace{\mathbf{T}_k(i,j)}_{\text{coloured traces}}
\]

---

## File Location

Generated by: `cpsvisualizer/src/cpsvisualizer/app.py`

Screenshot: `screenshot/15_fusion.png`
