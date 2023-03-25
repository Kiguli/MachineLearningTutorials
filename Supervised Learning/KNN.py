"""sklearn.neighbors provides functionality for unsupervised and supervised neighbors-based learning
 methods. Unsupervised nearest neighbors is the foundation of many other learning methods, notably
 manifold learning and spectral clustering. Supervised neighbors-based learning comes in two flavors:
 classification for data with discrete labels, and regression for data with continuous labels.

The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance
 to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest
  neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance
   can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based
    methods are known as non-generalizing machine learning methods, since they simply “remember” all of its
    training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).

Despite its simplicity, nearest neighbors has been successful in a large number of classification and regression
problems, including handwritten digits and satellite image scenes. Being a non-parametric method, it is often
successful in classification situations where the decision boundary is very irregular.

The classes in sklearn.neighbors can handle either NumPy arrays or scipy.sparse matrices as input. For dense
matrices, a large number of possible distance metrics are supported. For sparse matrices, arbitrary Minkowski
metrics are supported for searches.

There are many learning routines which rely on nearest neighbors at their core. One example is kernel density
estimation, discussed in the density estimation section."""

import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("car.data")
print(data.head())

# change non-numerical columns to numerical values (e.g. vhigh -> 3)
# returned as numpy arrays
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# column we are trying to predict
predict = "class"

# set features and labels
X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(cls)  # label

# separate training and test data
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# set neighbours
model = KNeighborsClassifier(n_neighbors=9)
# fit data to model
model.fit(X_train, y_train)
# calculate model accuracy
acc = model.score(X_test, y_test)
print(acc)

predictions = model.predict(X_test)

names = ["unacc", "acc", "good", "vgood"]
for i in range(len(predictions)):
    print("data: ", X_test[i], "prediction: ", names[predictions[i]], "actual: ", names[y_test[i]])
    n = model.kneighbors([X_test[i]], 9, True)  # view 9 nearest neighbours
    print("N: ", n)
