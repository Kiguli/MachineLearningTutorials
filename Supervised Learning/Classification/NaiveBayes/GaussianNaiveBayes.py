'''
Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem
 with the “naive” assumption of conditional independence between every pair of features given the value
 of the class variable.

In spite of their apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many
 real-world situations, famously document classification and spam filtering. They require a small amount of training
 data to estimate the necessary parameters.

Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods.
 The decoupling of the class conditional feature distributions means that each distribution can be independently
 estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of
 dimensionality.

GaussianNB implements the Gaussian Naive Bayes algorithm for classification.
The likelihood of the features is assumed to be Gaussian.
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(f"Number of mislabeled points out of a total {X_test.shape[0]} points :{(y_test != y_pred).sum()}")
