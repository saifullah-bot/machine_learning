# Machine Learning Models on Various Datasets

This project demonstrates the application of multiple machine learning models on different datasets using **scikit-learn**. It includes classification and regression tasks on well-known datasets like Iris, Breast Cancer, Wine, Digits, Diabetes, California Housing, and 20 Newsgroups text data.

---

## Project Overview

* Implemented different machine learning algorithms including:

  * Support Vector Machine (SVM)
  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Decision Tree
  * Random Forest
  * Naive Bayes
  * Gradient Boosting
  * AdaBoost
  * Multi-layer Perceptron (MLP) Neural Network
  * Linear Regression
  * Support Vector Regression (SVR)

* Used various datasets from `sklearn.datasets` and other sources:

  * Iris (classification)
  * Breast Cancer (classification)
  * Wine (classification)
  * Digits (classification)
  * Diabetes (regression)
  * California Housing (regression)
  * 20 Newsgroups (text classification)

* Evaluated models using metrics like:

  * Accuracy
  * Mean Squared Error (MSE)
  * Classification Report
  * Confusion Matrix

* Preprocessing steps such as train/test splitting and feature scaling (StandardScaler).

* Visualization of feature importance using **matplotlib** and **seaborn** for Random Forest on the Breast Cancer dataset.

---

## Installation

Make sure you have Python 3.x installed. Install the required Python packages using:

```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

---

## Usage

The scripts perform the following steps for each dataset:

1. Load dataset from `sklearn.datasets`.
2. Split dataset into training and testing sets.
3. Initialize and train the chosen machine learning model.
4. Predict on test data.
5. Evaluate the model using appropriate metrics.
6. (Optional) Visualize feature importance or confusion matrix.

---

## Example: Support Vector Machine on Iris Dataset

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("SVM Model Accuracy:", accuracy_score(y_test, y_pred))
```

---

## Key Results Summary

| Dataset            | Model                    | Metric             | Result |
| ------------------ | ------------------------ | ------------------ | ------ |
| Iris               | SVM (RBF Kernel)         | Accuracy           | \~97%  |
| Breast Cancer      | Logistic Regression      | Accuracy           | \~96%  |
| Wine               | Random Forest            | Accuracy           | \~98%  |
| Digits             | KNN                      | Accuracy           | \~98%  |
| Diabetes           | Decision Tree Regression | Mean Squared Error | \~3000 |
| California Housing | SVR                      | Mean Squared Error | \~0.3  |
| 20 Newsgroups      | Naive Bayes              | Accuracy           | \~90%  |

*Note: Exact values depend on random splits and parameters.*

---

## Visualization

* Feature importance bar chart for Random Forest on Breast Cancer dataset.
* Confusion matrix and classification reports printed for model evaluation.

---

## References

* [scikit-learn documentation](https://scikit-learn.org/stable/)
* [Matplotlib documentation](https://matplotlib.org/)
* [Seaborn documentation](https://seaborn.pydata.org/)

