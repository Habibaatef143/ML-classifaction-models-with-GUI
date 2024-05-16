# Machine Learning Project with GUI
This code snippet demonstrates the implementation of various machine learning classification and regression algorithms using Python.
It includes Support Vector Machines (SVM), Random Forest Classifier (RFC), K-Nearest Neighbors (KNN), and Linear Regression.

## Prerequisites

- Python 3.x
- Required libraries: pandas, numpy, scikit-learn, tkinter

## Getting Started

1. Clone the repository or download the code files.

2. Install the required libraries using the following command:pip install pandas numpy scikit-learn
3. Run the Python script `main.py` to launch the graphical user interface (GUI) application.

4. Choose the dataset by selecting the appropriate radio button:
- Diabetics
- Breast Cancer
- Salary Data

5. Enter the test size, which represents the proportion of the dataset to be used for testing the model.

6. Select the desired algorithm by clicking the corresponding button:
- Decision Tree
- SVM Algorithm
- Random Forest Algorithm
- K-Nearest Neighbors Algorithm
- Linear Regression Algorithm

7. The application will display the results, such as accuracy, precision, recall, error, and coefficients, depending on the selected algorithm and dataset.

## Functionality Overview

### Decision Tree Algorithm

The Decision Tree algorithm follows these steps:
1. Select random samples from the dataset.
2. Construct a decision tree for each sample.
3. Obtain a prediction result from each decision tree.
4. Perform voting for each predicted result.
5. Select the prediction result with the most votes as the final prediction.

### Random Forest Algorithm

The Random Forest algorithm has the following advantages:
- Highly accurate and robust due to multiple decision trees.
- Handles overfitting by averaging predictions.
- Applicable to both classification and regression problems.
- Can handle missing values.

However, it also has some limitations:
- Slower compared to individual decision trees.
- Difficult to interpret compared to a single decision tree.

### K-Nearest Neighbors (KNN) Algorithm

KNN Algorithm operates as follows:
- Computes the Euclidean distance between the new point and all rows in the dataset.
- Considers the k nearest neighbors based on the distance.
- Assigns the class label based on majority voting.

Advantages of KNN Algorithm:
- Lazy learner that requires no training.
- Easy to add data and implement.

Disadvantages of KNN Algorithm:
- Not suitable for big data or high-dimensional data.
- Sensitive to noise and missing values.
- Requires normalization and scaling.

### Linear Regression Algorithm

The Linear Regression algorithm performs the following steps:
1. Select the Linear Regression Algorithm.
2. Read the dataset based on the selected checkbox value.
3. Scale the dataset using MinMaxScaler.
4. Split the data into training and testing sets.
5. Train the Linear Regression model on the training data.
6. Predict the target variable for the test data.
7. Calculate the mean squared error and obtain the coefficients.
8. Display the results.

## Contributions

Contributions to improve the functionality, efficiency, or documentation of the code are welcome. If you find any issues or have suggestions, please feel free to open an issue or submit a pull request.

## License

This code is released under the [MIT License]
