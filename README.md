# CPS-Visualizer

CPS-Visualizer is a Python package that calculates and visualizes CPS (counts per second) for ICPMS scan data.
It provides a command-line interface (CLI) and a graphical user interface (GUI) for users to easily calculate and visualize CPS data. The package is designed to be user-friendly and easy to use, with a simple and intuitive interface.

## Features

- Support for surface data visualization for ICPMS scan data
- Export CPS data to CSV files
- Support for multiple data files as multiple components
- Support for various data processing methods, such as log_transform,centering_transform,z_score_normalization,standardize and equalize_hist
- Support for various distance metrics, such as Euclidean, Manhattan, Chebyshev, Minkowski, Cosine, Correlation, Jaccard, Dice, Kulsinski, Rogers-Tanimoto, Russell-Rao, Sokal-Michener, Sokal-Sneath, Yule,Hsim_Distance,Close_Distance, Mutual Information and SSIM (structural similarity index)

## Preprocessing Function for More Intutitive Visualization

We have developed a list of preprocessing functions that can be used to transform the data for more intuitive visualization.

* log_transform: 

        `log_data = np.log1p(data)`. The mathematical meaning of  is to compute the natural logarithm of `1 + data`. This function is more accurate when dealing with small values close to zero than calculating `np.log( data)` directly. Specifically, it returns `ln(1 + data)`, where `ln`denotes the natural logarithm (the logarithm with e as its base).

* centering_transform

        `centered_data = data - np.mean(data, axis=0)`. This function subtracts the mean of each column from the corresponding column in the input data. The result is a new array where each element is the corresponding element in the input data minus the mean of its column.

* z_score_normalization

        `normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)`. This function normalizes the data by subtracting the mean and dividing by the standard deviation of each column. The result is a new array where each element is the corresponding element in the input data minus the mean of its column, divided by the standard deviation of its column.

* standardize

        `standardized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))`. This function standardizes the data by subtracting the minimum value and dividing by the range (maximum value - minimum value) of each column. The result is a new array where each element is the corresponding element in the input data minus the minimum value of its column, divided by the range of its column.

* equalize_hist

        `equalized_data = exposure.equalize_hist(data)`. This function applies histogram equalization to the data. Histogram equalization is a technique used to improve the contrast of an image by redistributing the intensity values of the pixels. It works by creating a histogram of the pixel intensities and then equalizing the histogram so that the pixel intensities are distributed evenly across the range of possible values. The result is a new array where each element is the corresponding element in the input data after histogram equalization.

## Similarity Metrics

We have developed a list of similarity metrics that can be used to calculate the similarity between data points.

### Traditional Distance Metrics

* Euclidean

        `euclidean_distance = np.sqrt(np.sum((data1 - data2) ** 2))`. This function calculates the Euclidean distance between data points. The Euclidean distance is the "ordinary" straight-line distance between two points in Euclidean space. It is calculated by taking the square root of the sum of the squared differences between the corresponding elements of the two data points.

* Manhattan

        `manhattan_distance = np.sum(np.abs(data1 - data2))`. This function calculates the Manhattan distance between data points. The Manhattan distance is the sum of the absolute differences between the corresponding elements of the two data points. It is also known as the L1 norm or the taxicab distance.

* Chebyshev

        `chebyshev_distance = np.max(np.abs(data1 - data2))`. This function calculates the Chebyshev distance between data points. The Chebyshev distance is the maximum absolute difference between the corresponding elements of the two data points. It is also known as the L∞ norm or the chessboard distance.

* Minkowski

        `minkowski_distance = np.sum(np.abs(data1 - data2) ** p) ** (1/p)`. This function calculates the Minkowski distance between data points. The Minkowski distance is a generalized metric that can be used to measure the distance between two points in a normed vector space. It is calculated by taking the pth root of the sum of the pth powers of the absolute differences between the corresponding elements of the two data points.

* Cosine

        `cosine_similarity = 1 - spatial.distance.cosine(data1, data2)`. This function calculates the cosine similarity between data points. The cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. It is calculated by taking the dot product of the two data points and dividing it by the product of the magnitudes of the two data points.

* Correlation

        `correlation_coefficient = np.corrcoef(data1, data2)[0, 1]`. This function calculates the correlation coefficient between data points. The correlation coefficient is a measure of the linear relationship between two variables. It is calculated by taking the covariance of the two data points and dividing it by the product of the standard deviations of the two data points.
        
* Jaccard

        `jaccard_similarity = spatial.distance.jaccard(data1, data2)`. This function calculates the Jaccard similarity between data points. The Jaccard similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the size of the union of the two sets.

* Dice

        `dice_similarity = 2 * np.sum(data1 * data2) / (np.sum(data1 ** 2) + np.sum(data2 ** 2))`. This function calculates the Dice similarity between data points. The Dice similarity is a measure of similarity between two sets. It is calculated by taking twice the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets.

* Kulsinski

        `kulsinski_distance = spatial.distance.kulsinski(data1, data2)`. This function calculates the Kulsinski distance between data points. The Kulsinski distance is a measure of dissimilarity between two sets. It is calculated by taking the size of the union of the two sets and subtracting the size of the intersection of the two sets.

* Rogers_Tanimoto

        `rogers_tanimoto_similarity = 1 - spatial.distance.rogerstanimoto(data1, data2)`. This function calculates the Rogers-Tanimoto similarity between data points. The Rogers-Tanimoto similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets minus the size of the intersection of the two sets.

* Russell_Rao

        `russell_rao_similarity = np.sum(np.minimum(data1, data2)) / np.sum(data1 + data2)`. This function calculates the Russell-Rao similarity between data points. The Russell-Rao similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets.

* Sokal_Michener

        `sokal_michener_similarity = np.sum(np.minimum(data1, data2)) / np.sum(np.maximum(data1, data2))`. This function calculates the Sokal-Michener similarity between data points. The Sokal-Michener similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets minus the size of the intersection of the two sets.

* Sokal_Sneath

        `sokal_sneath_similarity = np.sum(np.minimum(data1, data2)) / np.sum(data1 + data2)`. This function calculates the Sokal-Sneath similarity between data points. The Sokal-Sneath similarity is a measure of similarity between two sets. It is calculated by taking the size of the intersection of the two sets and dividing it by the sum of the sizes of the two sets minus the size of the intersection of the two sets.

* Yule

        `yule_coefficient = spatial.distance.yule(data1, data2)`. This function calculates the Yule coefficient between data points. The Yule coefficient is a measure of dissimilarity between two sets. It is calculated by taking the size of the union of the two sets and subtracting the size of the intersection of the two sets.

### High Dimensional Distance

These are functions that calculate distances between high-dimensional data points. Which has been partially incorporated into the `GeoPyTool` application.

* Hsim_Distance

$$
\operatorname{Hsim}\left(x_i, \quad x_j\right)=\frac{\sum_{k=1}^n \frac{1}{1+\left|x_{i k}-x_{j k}\right|}}{n}
$$


* Close_Distance

$$
\operatorname{Close}\left(x_i, x_j\right)=\frac{\sum_{k=1}^n e^{-\left|x_{i k}-x_{j k}\right|}}{n}
$$

### Mutual Information

The mutual information is a measure of the amount of information that one random variable contains about another random variable.

The function with `_flattern`suffix is to compute the mutual information directly after flattening the matrix data, without considering the structural information of the matrix form of the original data; the function with `_unflattern`suffix is to compute the mutual information of the matrix by columns and then take the average value, considering the structural information of the matrix.

* mutual_info_regression

        Including `mutual_info_regression_flattern`and `mutual_info_regression_unflattern`, used for regression tasks to measure the dependency between continuous features and a continuous target variable.

* mutual_info_score

        Including `mutual_info_score_flattern` and `mutual_info_score_unflattern`, used for classification tasks to measure the dependency between two categorical variables.

### Structural Similarity

The structural similarity index (SSIM) is a method for measuring the similarity between two images. It is a measure of the structural information in the images, and is designed to be robust to changes in brightness, contrast, and noise.


* calculate_ssim

        This function calculates the structural similarity index (SSIM) between two matrix as images. The SSIM is a measure of the structural information in the images, and is designed to be robust to changes in brightness, contrast, and noise. The function takes two images as input, and returns the SSIM between the two images.

* luminance

        This function only return the luminance difference between two matrix as images. The luminance is a measure of the brightness of an image, and is calculated as the average value of the pixel intensities in the image.

* contrast

        This function only return the contrast difference between two images. The contrast is a measure of the difference in brightness between the lightest and darkest parts of an image.

* structure

        This function only return the structural difference between two images. The structure is a measure of the difference in the shape and texture of an image.


## Installation

The package is available on PyPI and can be installed using pip. It is compatible with Python 3.12 and above.
Developed with Python and Pyside6, theoretically it should work on any platform that supports Python and Pyside6.
However, due to the limitations of our current development environment, we have only tested the package on Windows 11 and Ubuntu 24.04.

### Additional Steps on Ubuntu

If you are using Ubuntu, you may need to install some additional dependencies.

```Bash
sudo apt update
sudo apt install libxcb-cursor0
sudo apt install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0
```

### Windows Installation

If you are using Windows 11 and above, you can download the packaged file from the following link: [https://pan.baidu.com/s/1F-RFVtzELEoOlSAkViuSsA?pwd=cugb](https://pan.baidu.com/s/1F-RFVtzELEoOlSAkViuSsA?pwd=cugb) with code `cugb`.

There are two files in the link above, `CPS-Visualizer-1.0.msi`and `CPS-Visualizer-1.0.zip`.
`CPS-Visualizer-1.0.msi`can be double clicked to install.
`CPS-Visualizer-1.0.zip`can be extracted to a folder and run the `CPS-Visualizer-1.0.exe`file in the folder.

### Installation with Pip

To use this application, python 3.12 or above is required, which can be downloaded from the official website. Python installation related resources and instructions can be found at https://www.python.org/downloads/.

When finished installing Python, you need to install some depencies using pip:

```Bash
pip install matplotlib numpy==1.26.4 pandas PySide6 scipy scikit-learn scikit-image
```

Then you can install the `cpsvisualizer`package using pip:

```Bash
pip install cpsvisualizer
```

## Usage

This package provides two interfaces: a command-line interface (CLI) and a graphical user interface (GUI).
You can choose to use either interface based on your needs.


### Graphical User Interface (GUI)


After the installation, you can run the application by executing the following commands to run a GUI:

```Bash
python -c "import cpsvisualizer;cpsvisualizer.gui()"
```

Then there will coms the gui, which will look like this:

![GUI](./images/gui.png)

The GUI is really quite straightforward, just check it out and you will be able to use it.

### Command-Line Interface (CLI)

Alternatively, you can run the application from the command line:

```Bash
cd path/to/data/files # always cd to the location of your data files first
python -c "import cpsvisualizer;cpsvisualizer.cli('Ag.csv Cu.csv Zn.csv Fe.csv', 'log_transform papa pupu pipi popo equalize_hist Euclidean Yule', 'silent')" # silent mode
python -c "import cpsvisualizer;cpsvisualizer.cli('Ag.csv Cu.csv Zn.csv Fe.csv', 'log_transform papa pupu pipi popo equalize_hist Euclidean Yule', 'show')" # show the plot
```

As shown above, the command line interface takes three arguments: the path to the data files, the processing methods, and the mode (silent or show).

The processing methods can be selected from the following set of commands, and the order in which they are listed is the order in which the corresponding processing methods are applied, so be sure to pay attention to the order.

The available methods for converting data are shown in the table below:
        `log_transform`,`centering_transform`,`z_score_normalization`,`standardize`,`equalize_hist`

The method of calculating the distance for each pair of data can be selected from the list below:
        `Euclidean`,`Manhattan`,`Chebyshev`,`Minkowski`,`Cosine`,`Correlation`,`Jaccard`,`Dice`,`Kulsinski`,`Rogers_Tanimoto`,`Russell_Rao`,`Sokal_Michener`,`Sokal_Sneath`,`Yule`,`mutual_info_regression_flattern`,`mutual_info_regression_unflattern`,`mutual_info_score_flattern`,`mutual_info_score_unflattern`,`calculate_ssim`,`luminance`,`contrast`,`structure`,`Hsim_Distance`,`Close_Distance`

The last opition can be 'silent' or 'show', the former means save the plots as png, pdf and svg files directly, the latter means show the plots in the GUI and require user to save the plots manually.

### Output of the CLI

The CLI silent mode will output the following information to the console:

```Bash
(base) hadoop@hadoop:~$ cd Desktop
(base) hadoop@hadoop:~/Desktop$ python -c "import cpsvisualizer;cpsvisualizer.cli('Ag.csv Cu.csv Zn.csv Fe.csv', 'log_transform papa pupu pipi popo equalize_hist Euclidean Yule', 'silent')"
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
Euclidean file is save to /home/hadoop/Desktop/Euclidean.csv
Yule file is save to /home/hadoop/Desktop/Yule.csv
PNG file saved at: /home/hadoop/Desktop/CPS_Data_Visualization.png
PDF file saved at: /home/hadoop/Desktop/CPS_Data_Visualization.pdf
SVG file saved at: /home/hadoop/Desktop/CPS_Data_Visualization.svg
```

The CLI show mode will output the following information to the console:

```Bash
(base) hadoop@hadoop:~$ cd Desktop
(base) hadoop@hadoop:~/Desktop$ python -c "import cpsvisualizer;cpsvisualizer.cli('Ag.csv Cu.csv Zn.csv Fe.csv', 'log_transform papa pupu pipi popo equalize_hist Euclidean Yule', 'show')"
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
Euclidean file is save to /home/hadoop/Desktop/Euclidean.csv
Yule file is save to /home/hadoop/Desktop/Yule.csv
```

And there will come a plot window to show the results.

![CLI_show](./images/cli_show.png)

## License

This project is licensed under the GNU Affero General Public License V3 - see the [LICENSE](LICENSE) file for details.
