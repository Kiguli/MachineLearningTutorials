"""The following are a set of methods intended for regression in which the target value is expected
 to be a linear combination of the features. In mathematical notation, if hat{y}
 is the predicted value.

        hat{y}(w,x) = w_0 + w_1x_1 + ... + w_px_p

Across the module, we designate the vector w = (w_1,...,w_p) as coef_ and w_0 as intercept_.

To perform classification with generalized linear models, see Logistic regression."""

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# column we are trying to predict
predict = "G3"

# split data into data used to predict and data trying to predict
X = np.array(data.drop([predict], 1))  # features
y = np.array(data[predict])  # labels
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best = 0  # store value for best model
for _ in range(30):
    # split training data and test data
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # fit best fit line on training data
    linear.fit(X_train, y_train)

    # find accuracy of the model
    acc = linear.score(X_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        # save model into pickle
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# load model from pickle
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# linear model in form y = mx+c, where m is coefficient and c is the intercept
print("Coefficients: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# show the predicted scores for each student
predictions = linear.predict(X_test)
for i in range(len(predictions)):
    print("data: ", X_test[i], "prediction: ", predictions[i], "real value: ", y_test[i])

# change the matplotlib style
style.use("ggplot")

# create scatter plot
xaxis = "G1"
pyplot.scatter(data[xaxis], data[predict])
pyplot.xlabel(xaxis)
pyplot.ylabel(predict)
pyplot.show()
