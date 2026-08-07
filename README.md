# CPS-Visualizer

CPS-Visualizer is a Python package that calculates and visualizes CPS (counts per second) for LA-ICP-MS scan data.
It provides a command-line interface (CLI) and a graphical user interface (GUI) for users to easily calculate and visualize CPS data. The package is designed to be user-friendly and easy to use, with a simple and intuitive interface.

## Features

- Support for surface data visualization for LA-ICP-MS scan data
- Export CPS data to CSV files
- Support for multiple data files as multiple components
- Support for various data processing methods, such as log_transform, centering_transform, z_score_normalization, standardize, and equalize_hist
- Support for various distance metrics, such as Euclidean, Manhattan, Chebyshev, Minkowski, Cosine, Correlation, Jaccard, Dice, Kulsinski, Rogers-Tanimoto, Russell-Rao, Sokal-Michener, Sokal-Sneath, Yule, Hsim_Distance, Close_Distance, Mutual Information, and SSIM (structural similarity index)

## Preprocessing Functions for More Intuitive Visualization

We have developed a list of preprocessing functions that can be used to transform the data for more intuitive visualization.

* log_transform:

$$
Log(data) = \ln(1 + data)
$$

`log_data = np.log1p(data)`. The mathematical meaning is to compute the natural logarithm of `1 + data`. This function is more accurate when dealing with small values close to zero than calculating `np.log(data)` directly. Specifically, it returns `ln(1 + data)`, where `ln` denotes the natural logarithm (the logarithm with e as its base).

* centering_transform:

$$
Centered(data) = data - Mean(data)
$$

`centered_data = data - np.mean(data, axis=0)`. This function subtracts the mean of each column from the corresponding column in the input data. The result is a new array where each element is the corresponding element in the input data minus the mean of its column.

* z_score_normalization:

$$
Normalized_{z\_score}(data) = \frac{data - \mu}{\sigma}
$$

`normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)`. This function normalizes the data by subtracting the mean and dividing by the standard deviation of each column. The result is a new array where each element is the corresponding element in the input data minus the mean of its column, divided by the standard deviation of its column.

* standardize:

$$
Standardized_{(Min-Max)}(data) = \frac{data - \min(data)}{\max(data) - \min(data)}
$$

`standardized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))`. This function standardizes the data by subtracting the minimum value and dividing by the range (maximum value - minimum value) of each column. The result is a new array where each element is the corresponding element in the input data minus the minimum value of its column, divided by the range of its column.

* equalize_hist:

$$
\mathrm{Equalized\_data}(x, y) = \frac{CDF(data(x, y)) - CDF_{\min}}{1 - CDF_{\min}}
$$

Where:
- $data(x, y)$ is the pixel value at position $(x, y)$.
- $CDF$ is the cumulative distribution function of the histogram of the data.
- $CDF_{\min}$ is the minimum non-zero value of the CDF.

`equalized_data = exposure.equalize_hist(data)`. This function applies histogram equalization to the data. Histogram equalization is a technique used to improve the contrast of an image by redistributing the intensity values of the pixels. It works by creating a histogram of the pixel intensities and then equalizing the histogram so that the pixel intensities are distributed evenly across the range of possible values. The result is a new array where each element is the corresponding element in the input data after histogram equalization.

## Similarity Metrics

We have developed a list of similarity metrics that can be used to calculate the similarity between data points.

### Traditional Distance Metrics

* Euclidean

$$
d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

`euclidean_distance = np.sqrt(np.sum((data1 - data2) ** 2))`. This function calculates the Euclidean distance between data points. The Euclidean distance is the "ordinary" straight-line distance between two points in Euclidean space. It is calculated by taking the square root of the sum of the squared differences between the corresponding elements of the two data points.

* Manhattan

$$
d(x, y) = \sum_{i=1}^{n}|x_i - y_i|
$$

`manhattan_distance = np.sum(np.abs(data1 - data2))`. This function calculates the Manhattan distance between data points. The Manhattan distance is the sum of the absolute differences between the corresponding elements of the two data points. It is also known as the L1 norm or the taxicab distance.

* Chebyshev

$$
d(x, y) = \max_{i}|x_i - y_i|
$$

`chebyshev_distance = np.max(np.abs(data1 - data2))`. This function calculates the Chebyshev distance between data points. The Chebyshev distance is the maximum absolute difference between the corresponding elements of the two data points. It is also known as the L∞ norm or the chessboard distance.

* Minkowski

$$
d(x, y) = \Big(\sum_{i=1}^{n}|x_i - y_i|^p\Big)^{1/p}
$$

`minkowski_distance = np.sum(np.abs(data1 - data2) ** p) ** (1/p)`. This function calculates the Minkowski distance between data points. The Minkowski distance is a generalized metric that can be used to measure the distance between two points in a normed vector space. It is calculated by taking the pth root of the sum of the pth powers of the absolute differences between the corresponding elements of the two data points.

* Cosine

$$
\cos(\theta) = \frac{x \cdot y}{\|x\|\|y\|}
$$

`cosine_similarity = 1 - spatial.distance.cosine(data1, data2)`. This function calculates the cosine similarity between data points. The cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. It is calculated by taking the dot product of the two data points and dividing it by the product of the magnitudes of the two data points.

* Correlation

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

`correlation_coefficient = np.corrcoef(data1, data2)[0, 1]`. This function calculates the correlation coefficient between data points. The correlation coefficient is a measure of the linear relationship between two variables. It is calculated by taking the covariance of the two data points and dividing it by the product of the standard deviations of the two data points.

* Jaccard

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{a}{a + b + c}
$$

`jaccard_similarity = spatial.distance.jaccard(data1, data2)`. This function calculates the Jaccard similarity between data points. The Jaccard similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the size of the union of the two sets.

* Dice

$$
D(A, B) = \frac{2|A \cap B|}{|A| + |B|} = \frac{2a}{2a + b + c}
$$

`dice_similarity = 2 * np.sum(data1 * data2) / (np.sum(data1 ** 2) + np.sum(data2 ** 2))`. This function calculates the Dice similarity between data points. The Dice similarity is a measure of similarity between two sets. It is calculated by taking twice the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets.

* Kulsinski

$$
K(A, B) = \frac{|A \triangle B| + n - |A \cap B|}{|A \cup B| + n - |A \cap B|} = \frac{b + c + n - a}{a + b + c + n - a}
$$

`kulsinski_distance = spatial.distance.kulsinski(data1, data2)`. This function calculates the Kulsinski distance between data points. The Kulsinski distance is a measure of dissimilarity between two sets. It is calculated by taking the size of the union of the two sets and subtracting the size of the intersection of the two sets.

* Rogers-Tanimoto

$$
RT(A, B) = \frac{|A \cap B| + |\overline{A} \cap \overline{B}|}{|A \cup B| + |\overline{A} \cup \overline{B}|} = \frac{a + d}{a + d + 2(b + c)}
$$

`rogers_tanimoto_similarity = 1 - spatial.distance.rogerstanimoto(data1, data2)`. This function calculates the Rogers-Tanimoto similarity between data points. The Rogers-Tanimoto similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets minus the size of the intersection of the two sets.

* Russell-Rao

$$
RR(A, B) = \frac{|A \cap B|}{n} = \frac{a}{a + b + c + d}
$$

`russell_rao_similarity = np.sum(np.minimum(data1, data2)) / np.sum(data1 + data2)`. This function calculates the Russell-Rao similarity between data points. The Russell-Rao similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets.

* Sokal-Michener

$$
SM(A, B) = \frac{|A \cap B| + |\overline{A} \cap \overline{B}|}{n} = \frac{a + d}{a + b + c + d}
$$

`sokal_michener_similarity = np.sum(np.minimum(data1, data2)) / np.sum(np.maximum(data1, data2))`. This function calculates the Sokal-Michener similarity between data points. The Sokal-Michener similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets minus the size of the intersection of the two sets.

* Sokal-Sneath

$$
SS(A, B) = \frac{|A \cap B|}{|A \cap B| + 2|A \triangle B|} = \frac{a}{a + 2(b + c)}
$$

`sokal_sneath_similarity = np.sum(np.minimum(data1, data2)) / np.sum(data1 + data2)`. This function calculates the Sokal-Sneath similarity between data points. The Sokal-Sneath similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets minus the size of the intersection of the two sets.

* Yule

$$
Y(A, B) = \frac{|A \cap B| \cdot |\overline{A} \cap \overline{B}| - |A \setminus B| \cdot |B \setminus A|}{|A \cap B| \cdot |\overline{A} \cap \overline{B}| + |A \setminus B| \cdot |B \setminus A|} = \frac{ad - bc}{ad + bc}
$$

`yule_coefficient = spatial.distance.yule(data1, data2)`. This function calculates the Yule coefficient between data points. The Yule coefficient is a measure of dissimilarity between two sets. It is calculated by taking the size of the union of the two sets and subtracting the size of the intersection of the two sets.

Where $a = |A \cap B|$, $b = |A \setminus B|$, $c = |B \setminus A|$, $d = |\overline{A} \cap \overline{B}|$, and $n = a + b + c + d$.

### High Dimensional Distance

These are functions that calculate distances between high-dimensional data points, which have been partially incorporated into the `GeoPyTool` application.

* Hsim_Distance

$$
\mathrm{Hsim}(x_i, x_j)=\frac{\sum_{k=1}^n \frac{1}{1+|x_{ik}-x_{jk}|}}{n}
$$

* Close_Distance

$$
\mathrm{Close}(x_i, x_j)=\frac{\sum_{k=1}^n e^{-|x_{ik}-x_{jk}|}}{n}
$$

### Mutual Information

The mutual information is a measure of the amount of information that one random variable contains about another random variable.

The function with the `_flattern` suffix computes the mutual information directly after flattening the matrix data, without considering the structural information of the matrix form of the original data; the function with the `_unflattern` suffix computes the mutual information of the matrix by columns and then takes the average value, considering the structural information of the matrix.

* mutual_info_regression

Including `mutual_info_regression_flattern` and `mutual_info_regression_unflattern`, used for regression tasks to measure the dependency between continuous features and a continuous target variable.

* mutual_info_score

Including `mutual_info_score_flattern` and `mutual_info_score_unflattern`, used for classification tasks to measure the dependency between two categorical variables.

### Structural Similarity

The structural similarity index (SSIM) is a method for measuring the similarity between two images. It is a measure of the structural information in the images, and is designed to be robust to changes in brightness, contrast, and noise.

* calculate_ssim

This function calculates the structural similarity index (SSIM) between two matrices as images. The SSIM is a measure of the structural information in the images, and is designed to be robust to changes in brightness, contrast, and noise. The function takes two images as input, and returns the SSIM between the two images.

* luminance

This function only returns the luminance difference between two matrices as images. The luminance is a measure of the brightness of an image, and is calculated as the average value of the pixel intensities in the image.

* contrast

This function only returns the contrast difference between two images. The contrast is a measure of the difference in brightness between the lightest and darkest parts of an image.

* structure

This function only returns the structural difference between two images. The structure is a measure of the difference in the shape and texture of an image.

### On the Selection of Metrics (why Boolean set metrics are excluded)

A full mathematical treatment is given in [METHODOLOGY.md](./METHODOLOGY.md).
In short, the Boolean set-similarity metrics — **Jaccard, Dice, Kulsinski,
Rogers-Tanimoto, Russell-Rao, Sokal-Michener, Sokal-Sneath, Yule** — are
excluded from the AODA rankings because they collapse to **constant values**
(all 0 or all 1) on LA-ICP-MS data and therefore carry no discriminatory power.

There are two complementary reasons:

1. **After block-average downsampling the support becomes constant.**
   LA-ICP-MS scans are sparse (≈20% zeros); a block average of a mostly-zero
   window is strictly positive, so the zero fraction drops to 0.0%.  Every
   Boolean mask `𝟙[x>0]` then equals the all-ones mask, giving `a=n, b=c=d=0`.
   Substituting into the contingency-table formulas yields a single constant:
   `Jaccard=Dice=Kulsinski=Rogers-Tanimoto=Yule → 0`, and
   `Russell-Rao=Sokal-Michener=Sokal-Sneath → 1`.

2. **Even on raw sparse data the Boolean metrics are intensity-blind.**
   All eight metrics are functions of the single 2×2 contingency table
   `{a,b,c,d}` and are invariant under any strictly increasing monotone
   transform of the positive intensities.  Two maps with *identical* binary
   support but a 50× intensity gap give Jaccard distance = 0.0000, whereas
   Euclidean gives 6.4×10⁴ and Bray-Curtis 0.9586 — i.e. the Boolean family
   cannot tell a 100 cps grain from a 10⁷ cps grain as long as both are non-zero.

For LA-ICP-MS imaging, **continuous-valued** metrics (Euclidean, Manhattan,
Chebyshev, Minkowski, Cosine, Correlation, Bray-Curtis, Canberra,
Hsim/Close, mutual-information variants, SSIM family) are retained because
they operate on the multi-order-of-magnitude intensity field that defines the
signal, and all returned 15 distinct pairwise values on the sample data.

## Installation

The package is available on PyPI and can be installed using pip. It is compatible with Python 3.12 and above.
Developed with Python and PySide6, theoretically it should work on any platform that supports Python and PySide6.
However, due to the limitations of our current development environment, we have only tested the package on Windows 11 and Ubuntu 24.04.

### Additional Steps on Ubuntu

If you are using Ubuntu, you may need to install some additional dependencies.

```bash
sudo apt update
sudo apt install libxcb-cursor0
sudo apt install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0
```

### Windows Installation

If you are using Windows 11 and above, you can download the packaged file from the following link: [https://pan.baidu.com/s/1F-RFVtzELEoOlSAkViuSsA?pwd=cugb](https://pan.baidu.com/s/1F-RFVtzELEoOlSAkViuSsA?pwd=cugb) with code `cugb`.

There are two files in the link above, `CPS-Visualizer-1.0.msi` and `CPS-Visualizer-1.0.zip`.
`CPS-Visualizer-1.0.msi` can be double-clicked to install.
`CPS-Visualizer-1.0.zip` can be extracted to a folder, and you can run the `CPS-Visualizer-1.0.exe` file in the folder.

### Installation from PyPI

To use this application, Python 3.12 or above is required, which can be downloaded from the official website. Python installation related resources and instructions can be found at https://www.python.org/downloads/.

```bash
pip install cpsvisualizer
```

### Editable Install from Source

If you want to develop or run from source, clone the repo and install in editable mode inside the `cpsvisualizer` directory:

```bash
git clone https://github.com/GeoPyTool/CPS-Visualizer.git
cd CPS-Visualizer/cpsvisualizer
pip install -e ./
```

This registers a system-level `cpsv` command (no need to call `python -c ...` any more).

## Usage

This package provides three interfaces — a command-line interface (CLI, the default), a graphical user interface (GUI), and a web interface — all reachable through the single `cpsv` command:

```bash
cpsv --help
```

```
cpsv - CPS-Visualizer command line

Usage:
  cpsv [DATA_FILES] [FUNCTIONS] [MODE]      Run the batch CLI (default).
  cpsv gui                                  Launch the desktop GUI.
  cpsv web [--host HOST] [--port PORT]      Launch the web interface.

CLI arguments:
  DATA_FILES    space-separated list of CSV/XLSX data files (one quoted arg)
  FUNCTIONS    space-separated transforms and distance metrics (one quoted arg)
  MODE         show (default) | silent   (silent saves PNG/PDF/SVG)

Web options:
  --host HOST   bind address (default 127.0.0.1)
  --port PORT   bind port    (default 5005)

Examples:
  cpsv "Ag.csv Cu.csv" "log_transform equalize_hist Euclidean" silent
  cpsv gui
  cpsv web --port 6789
```

### Web Interface

The web interface provides an elegant browser-based experience with drag-and-drop file upload and interactive Plotly visualizations.

```bash
cpsv web --port 6789
```

Then open <http://127.0.0.1:6789> in your browser. `--host` (default `127.0.0.1`) and `--port` (default `5005`) are both optional.

Features in the web version:

- Drag-and-drop CSV/XLSX files (each file is one element/component)
- **Dark / light themes** with a header toggle — follows your OS preference, remembers your choice (like the GeoPyTool blog)
- **Map Viewer** — full-scan element maps at the true data aspect ratio (every cell is a strict square, no stretching):
  - flip through datasets one by one with prev / next
  - render as **Raw**, **Enhanced** (equalise + normalise) or **Filtered** (sobel/unsharp)
  - **Wipe compare**: pick any two datasets, drag the divider left / right to reveal one map over the other
- 12 processing transforms including spatial filters (gaussian / median / sobel / unsharp, plus normalise / percentile-clip)
- Select any of the 26 distance metrics and view the pairwise heatmap
- Overlay modes: RGB composite, alpha blend, difference, ratio
- Statistics: PCA scatter, Pearson/Spearman correlation heatmap, descriptive stats, ANOVA
- Comparison: PCA, t-SNE, UMAP embeddings, dendrogram, K-means clustering
- Image quality metrics: PSNR, entropy, CEI, Tenengrad, SSIM across all transforms
- **AODA continuous optimisation** — Box-Cox power exponent is tuned by combining golden-section search and the secant (quasi-Newton) method; whichever achieves the higher Discrimination Power Score wins
- Result caching makes repeated tab visits instantaneous

The GUI and CLI share the same transforms, filters and square-cell rendering:
the GUI adds a **Wipe Compare** action (select two datasets, drag the slider),
and both use `aspect='equal'` so element maps keep their true proportions.

The GUI mirrors the Web Map Viewer — full-scan element maps at the true data
aspect ratio (every cell a strict square), step through datasets with the
**<** / **>** buttons, render as **raw / enhanced / filtered**, and compare any
two datasets by switching the mode to **wipe** and dragging the slider.  The
GUI follows the native system theme automatically (no forced light/dark).

There is a **Load Sample** button in the header if you want to try without your own data.

### Adaptive Optimisation (AODA)

The adaptive search treats preprocessing as a continuous Box-Cox power transform
and maximises the Discrimination Power Score (DPS) with two classic line-search
optimisers:

- **Golden-section search** — robust, derivative-free bracketing on a unimodal DPS curve
- **Secant method** — quasi-Newton root-finding on the DPS derivative for superlinear convergence

Both run on the same bracket; the method that reaches the higher DPS is kept and
labelled as the winner in the UI. A cheap multi-exponent pre-screen ranks the
distance metrics first, so only the top candidates get the full optimisation —
this keeps the whole search fast even on large LA-ICP-MS matrices.

### Graphical User Interface (GUI)

After the installation, launch the GUI with:

```bash
cpsv gui
```

Then there will come the GUI, which will look like this:

![GUI](./images/gui.png)

The GUI is really quite straightforward, just check it out and you will be able to use it.

### Command-Line Interface (CLI)

Alternatively, you can run the application from the command line (`cpsv` defaults to the CLI):

```bash
cd path/to/data/files # always cd to the location of your data files first
cpsv "Ag.csv Cu.csv Zn.csv Fe.csv" "log_transform equalize_hist Euclidean Yule" silent   # silent mode (saves PNG/PDF/SVG)
cpsv "Ag.csv Cu.csv Zn.csv Fe.csv" "log_transform equalize_hist Euclidean Yule" show     # show the plot
```

As shown above, the command line interface takes three arguments: the path to the data files, the processing methods, and the mode (silent or show).

The processing methods can be selected from the following set of commands, and the order in which they are listed is the order in which the corresponding processing methods are applied, so be sure to pay attention to the order.

The available methods for converting data are shown in the table below:

`log_transform`, `centering_transform`, `z_score_normalization`, `standardize`, `equalize_hist`

The method of calculating the distance for each pair of data can be selected from the list below:

`Euclidean`, `Manhattan`, `Chebyshev`, `Minkowski`, `Cosine`, `Correlation`, `Jaccard`, `Dice`, `Kulsinski`, `Rogers_Tanimoto`, `Russell_Rao`, `Sokal_Michener`, `Sokal_Sneath`, `Yule`, `mutual_info_regression_flattern`, `mutual_info_regression_unflattern`, `mutual_info_score_flattern`, `mutual_info_score_unflattern`, `calculate_ssim`, `luminance`, `contrast`, `structure`, `Hsim_Distance`, `Close_Distance`

The last option can be 'silent' or 'show', the former means save the plots as png, pdf, and svg files directly, the latter means show the plots in the GUI and require the user to save the plots manually.

### Output of the CLI

The CLI silent mode will output the following information to the console:

```bash
(base) hadoop@hadoop:~$ cd Desktop
(base) hadoop@hadoop:~/Desktop$ cpsv "Ag.csv Cu.csv Zn.csv Fe.csv" "log_transform equalize_hist Euclidean Yule" silent
Data Files are :  ['Ag.csv', 'Cu.csv', 'Zn.csv', 'Fe.csv']
Trans Functions are: ['log_transform', 'equalize_hist']
Distance Calculations are: ['Euclidean', 'Yule']
Plot Option is :  silent
log_transform success on Ag
equalize_hist success on Ag
log_transform success on Cu
equalize_hist success on Cu
log_transform success on Zn
equalize_hist success on Zn
log_transform success on Fe
equalize_hist success on Fe
Euclidean file is saved to /home/hadoop/Desktop/Euclidean.csv
Yule file is saved to /home/hadoop/Desktop/Yule.csv
PNG file saved at: /home/hadoop/Desktop/CPS_Data_Visualization.png
PDF file saved at: /home/hadoop/Desktop/CPS_Data_Visualization.pdf
SVG file saved at: /home/hadoop/Desktop/CPS_Data_Visualization.svg
```

The CLI show mode will output the following information to the console:

```bash
(base) hadoop@hadoop:~$ cd Desktop
(base) hadoop@hadoop:~/Desktop$ cpsv "Ag.csv Cu.csv Zn.csv Fe.csv" "log_transform equalize_hist Euclidean Yule" show
Data Files are :  ['Ag.csv', 'Cu.csv', 'Zn.csv', 'Fe.csv']
Trans Functions are: ['log_transform', 'equalize_hist']
Distance Calculations are: ['Euclidean', 'Yule']
Plot Option is :  silent
log_transform success on Ag
equalize_hist success on Ag
log_transform success on Cu
equalize_hist success on Cu
log_transform success on Zn
equalize_hist success on Zn
log_transform success on Fe
equalize_hist success on Fe
Euclidean file is saved to /home/hadoop/Desktop/Euclidean.csv
Yule file is saved to /home/hadoop/Desktop/Yule.csv
```

And there will come a plot window to show the results.

![CLI_show](./images/cli_show.png)

## License

This project is licensed under the GNU Affero General Public License V3 - see the [LICENSE](LICENSE) file for details.