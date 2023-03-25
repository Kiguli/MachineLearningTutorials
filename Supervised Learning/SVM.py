"""Support vector machines (SVMs) are a set of supervised learning methods used for classification,
regression and outliers detection.

The advantages of support vector machines are:
- Effective in high dimensional spaces.
- Still effective in cases where number of dimensions is greater than the number of samples.
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided,
    but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:
- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions
    and regularization term is crucial.
- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold
    cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray)
 and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data,
  it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or
  scipy.sparse.csr_matrix (sparse) with dtype=float64."""

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# scikit-learn comes with datasets which are ready to be used e.g. breast cancer, iris, etc.
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

# setup features and labels, then split into training and testing data
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.1)

# print(X_train)
# print(y_train)
classes = ["malignant", "benign"]

# default kernel is rbf, also are poly, sigmoid, linear,
# C is how many values allowed in the margin of the SVM, 1 would be a hard margin,
# 2 allows double the number of points in the margin
clf = svm.SVC(kernel="rbf", C=2)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_predict)

print(acc)
