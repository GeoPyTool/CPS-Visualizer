# Metric Selection Methodology for CPS-Visualizer

*Why binary (Boolean) set-similarity metrics were excluded, and the
mathematical reason they collapse to constant values on LA-ICP-MS data.*

---

## 1. Problem statement

The full metric suite exposed 26 distance / similarity measures.  In the
AODA continuous optimisation, a subset of these — the *Boolean set metrics*
(Jaccard, Dice, Kulsinski, Rogers-Tanimoto, Russell-Rao, Sokal-Michener,
Sokal-Sneath, Yule) — degenerated to **constant values** (all 0 or all 1) and
therefore carried **no discriminatory information**.  They were removed from
the reported rankings.

This document explains *why* they degenerate, which is a direct consequence of
two mathematical facts:

1.  LA-ICP-MS data are **continuous, non-negative intensity fields** spanning
    several orders of magnitude (in the sample set: 35.7 – 6.86×10⁹ cps,
    skewness ≈ 43);
2.  The Boolean metrics operate on the **binary support**
    (is-zero / is-nonzero) of the data, throwing away every intensity value.

---

## 2. The two failure regimes

### 2.1 After block-average downsampling: all-zero or all-one support

The AODA search and the fast web previews downsample each 2-D scan by
block-averaging (`_downsample`).  A block average of cells that are "mostly
background zero + a few bright pixels" is **always strictly positive**:

$$
\bar{x}_{\text{block}} = \frac{1}{k}\sum_{i=1}^{k} x_i > 0
\quad\text{whenever at least one } x_i > 0.
$$

In the sample set ~20% of raw cells are zero, but after 5×10 block averaging
the fraction of exactly-zero cells drops to **0.0%** (verified empirically).
Consequently the Boolean support becomes the constant-one mask:

$$
\mathbb{1}_{[x>0]}(i,j) \equiv 1 \quad \forall (i,j),
$$

i.e. `a = |A ∩ B| = n`, `b = c = 0`, `d = 0` in the contingency table.  Every
Boolean metric then collapses to a fixed value (Table 1).

**Table 1 — Boolean metrics after block-average downsampling (n = number of cells).**

| Metric            | Formula (distance d / similarity s)   | Collapsed value |
|-------------------|----------------------------------------|-----------------|
| Jaccard           | d = 1 − a/(a+b+c)                      | **0**           |
| Dice              | d = 1 − 2a/(2a+b+c)                    | **0**           |
| Kulsinski         | d = (b+c+n−a)/(n+b+c)                  | **0**           |
| Rogers-Tanimoto   | d = (b+c+d)/(a+d+b+c)                  | **0**           |
| Yule              | d = 1 − (ad−bc)/(ad+bc)                | **0** (0/0 → guard 0) |
| Russell-Rao       | s = a/n                                | **1**           |
| Sokal-Michener    | s = (a+d)/n                            | **1**           |
| Sokal-Sneath      | s = a/(a+2(b+c))                       | **1**           |

Since every off-diagonal entry of the pairwise matrix equals the same constant,
`range = max − min = 0` and the DPS (mean nearest-neighbour / mean off-diagonal)
becomes **trivially 1** (or 0), carrying no information.

### 2.2 On raw sparse data: loss of intensity contrast

Even without downsampling, the Boolean metrics still discard the CPS intensity.
All of them are functions of the single 2×2 contingency table

$$
\begin{array}{c|cc}
 & B & \neg B\\ \hline
 A & a & b\\
 \neg A & c & d
\end{array}
\qquad
a=|A\cap B|,\ b=|A\setminus B|,\ c=|B\setminus A|,\ d=|\neg A\cap\neg B|.
$$

They are **invariant under any strictly increasing monotone transformation**
of the positive intensities, because the sets A, B only depend on the sign of
the data, never on its magnitude.  This is the decisive mathematical
shortcoming: LA-ICP-MS contrast lives almost entirely in the intensity
*ratio*, not in the support.

Counter-example (verified numerically): two arrays sharing the *identical*
binary support (same 70% nonzero mask) but differing 50× in intensity give

| Metric | Value |
|--------|-------|
| Jaccard distance  | **0.0000** (identical support ⇒ identical "sets") |
| Euclidean distance| 6.4×10⁴ (captures the 50× intensity gap) |
| Bray-Curtis       | 0.9586 (captures the relative intensity gap) |

In other words the Boolean family cannot distinguish a 100 cps grain from a
10⁷ cps grain as long as both are non-zero — exactly the signal LA-ICP-MS
imaging is meant to resolve.

---

## 3. Why the remaining metrics keep their contrast

The metrics retained in the AODA rankings operate on the *continuous* values:

* **Norm-based** — Euclidean, Manhattan, Chebyshev, Minkowski: respond to the
  absolute intensity gap $\sum_i |x_i-y_i|^p$.
* **Angular / correlation** — Cosine, Correlation: respond to the relative
  intensity profile (scale invariant).
* **Ratio-based dissimilarity** — Bray-Curtis
  $\dfrac{\sum|x_i-y_i|}{\sum(x_i+y_i)}$ and Canberra
  $\sum\dfrac{|x_i-y_i|}{|x_i|+|y_i|}$: normalise by the local magnitude, so a
  1-cps difference on a 100-cps background counts the same as a 10⁴-cps
  difference on a 10⁶-cps background.
* **Soft distances** — Hsim / Close $= \frac{1}{n}\sum \frac{1}{1+|x_i-y_i|}$:
  continuous, always distinct unless the matrices are identical.
* **Information-theoretic** — mutual-information variants: measure statistical
  dependence of the continuous columns, not the support.
* **Structural** — SSIM (luminance / contrast / structure) and PSNR-based
  quality metrics: defined on continuous intensities.

All of these returned a full range of 15 distinct pairwise values on the six
element maps, i.e. genuine, non-degenerate contrast.

---

## 4. Recommendation

For LA-ICP-MS surface-scan comparison, use only **continuous-valued** metrics:

* Defaults: **Euclidean, Bray-Curtis, Cosine, Hsim_Distance, Canberra, SSIM**
* For compositional (relative-abundance) analysis: **Bray-Curtis, Cosine,
  Correlation**
* For structural pattern matching: **SSIM, luminance, contrast, structure**
* For statistical dependence: **mutual-information variants**

The Boolean set metrics are excluded because, on this data type, they either
collapse to constants (block-averaged / dense positive data) or reduce to the
trivial support mask and are blind to the multi-order-of-magnitude intensity
contrast that defines LA-ICP-MS images.

---

## 5. Reproducibility

The empirical statements above were produced with the bundled `DataSample`
(Ag, Au, Cu, Fe, Pb, Zn — six 126×1991 LA-ICP-MS element maps):

```python
from cpsvisualizer.core import DISTANCE_FUNCTIONS, compute_pairwise_matrix
# pairwise matrices over the downsampled data, off-diagonal uniqueness test
# -> 8 Boolean metrics give a single constant; 18 continuous metrics give 15
#    distinct values each.
```
