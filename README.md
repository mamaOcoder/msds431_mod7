# Week 7 Assignment: Data Cleaning, Frames, and Pipelines

## Project Summary

The goal of this assignment is to experiment with Go for preparing data for analysis and modeling. We test the isolation forest method for detecting outliers in the Modified National Institute of Standards and Technology (MNIST) dataset. 

### MNIST Dataset
The MNIST dataset comprises 60 thousand training observations and 10 thousand test observations. Each observation image includes a 28-by-28 grid of pixel values. Black (value 255) represents a foreground pixel for a digit, and white (value 0) the background. The labels associated with the images represent digits are 0 through 9.

We use the [GoMNIST GitHub repository](https://github.com/petar/GoMNIST) for loading the dataset.

### Go Package Selection:
There were a handful of packages available on Github that implement a version of Isolation Forests. I decided to use the [e-XpertSolutions/go-iforest](https://github.com/e-XpertSolutions/go-iforest) package. This package was the most recently updated (Nov 2022) of all of the packages that I looked at and also has 27 stars. A close second option was [malaschitz/randomForest](https://github.com/malaschitz/randomForest) which was listed in Go Awesome. I actually started an implementation using the randomForest package but got a little lost as there was not a way to clearly define hyperparameters for the isolation forest and the output was not as straight-forward. I opted to switch to the go-iforest package primarily because I found the documentation easier to follow.

### Results/Conculsions
Overall, working with the go-iforest package was straight-forward. We did have to do some data manipulation to convert the data into a two dimensional array of the type float64, which was not difficult. The iforest.NewForest method accepts 3 parameters, treesNumber, subsampleSize, and outlierRatio. These correspond to n_estimators, max_samples and contamination, respectively, in Python's scikit-learn. One thing that I struggled with was selecting a value for contamination. In Python's scikit-learn's IsolationForest method, contamination is set to "auto" by default and I was unable to find documentation quickly for how to replicate how "auto" computes the value. This is likely a factor for why my results from Go differ from the Python results.

It is difficult to say how successful the tests were. We were able to successfully flag instances as anomalies quickly, however, comparing the results to the Python output was difficult and I was unable to produce similar results that would make me feel comfortable solely relying on the Go implementation. 

Comparing the results to the Python output was difficult. The score that is produced from the go-iforest method appears to be inverted, with lower scores indicating an anomaly and higher score indicating normality. Additionally, the scores have a smaller range, an example running produced a minimum score of -0.0198 and maximum score of 0.1635, whereas Python scores range between 0 and 1. It is possible that the method in Python normalizes the outputs. Ultimately, I do not think that the scores matter much as long as the algorithms are detecting the actual outliers. For this reason, I decided to focus on whether the image instance was flagged as an outlier or not.

In the original Python code (isolationForest.py) the anomaly label (flag) was not printed as output, so I added some code to print out the anomaly labels. I then load those results into my Go code, convert them to match Go's labels and compare. In my first runnings of the code, my results were flagging significantly less anomalies than the Python results. As I played with the outlierRatio (contamination) value, I was able to improve the mismatched anomaly flags, however, never saw better than 2709 mismatch labels.

Overall, the fact is that there is an immaturity to the Go packages for machine learning algorithms and without more time to really investigate how each package works, I would hesitate to adopt this for the firm's outlier/anomaly detection needs in their data science pipeline. 

## Files
### *main.go*
This file contains the bulk of the code for this assignment.

### *python_miller*
This folder contains the code provided as our [jump-start](https://github.com/ThomasWMiller/jump-start-mnist-iforest) for this assignment. I chose to just use the Python code for comparison, as I have more experience with Python than with R. In the *isolationForest.py* I added code to write the anomaly labels out to a file for comparison.

### *results/pythonScores.csv*
This CSV contains the output from the Python code for the computed scores.

### *results/pythonAnomalyLabels.csv*
This CSV contains the output from the added Python code for the anomaly labels (-1 = anomaly and 1 = normal).

### *results/labels.csv*
This CSV contains the output from the Python code for the digits associated with the images. Not used in isolation forests.

### *results/goResults.csv*
This CSV contains the anomaly score and the anomaly label for the Go code.

### *results/comparedResults.csv*
This CSV contains the Go anomaly score, Python anomaly score, Go anomaly label and the converted Python anomaly label.