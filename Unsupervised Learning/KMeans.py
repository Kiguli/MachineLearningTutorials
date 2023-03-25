"""The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance,
minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below). This algorithm requires
 the number of clusters to be specified. It scales well to large numbers of samples and has been used across a large
  range of application areas in many fields.

The k-means algorithm divides a set of N samples X into K disjoint clusters C, each described by the mean mu_j
of the samples in the cluster. The means are commonly called the cluster “centroids”; note that they are not, in
general, points from X, although they live in the same space.

 K-means is often referred to as Lloyd’s algorithm. In basic terms, the algorithm has three steps. The first step
 chooses the initial centroids, with the most basic method being to choose k samples from the dataset X. After
 initialization, K-means consists of looping between the two other steps. The first step assigns each sample to its
  nearest centroid. The second step creates new centroids by taking the mean value of all the samples assigned
  to each previous centroid. The difference between the old and the new centroids are computed and the algorithm
  repeats these last two steps until this value is less than a threshold. In other words, it repeats until the
  centroids do not move significantly."""

import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# scale digits down to be between -1 and 1
digits = load_digits()
X = scale(digits.data)
y = digits.target

# number of clusters = number of different digits (0-9)
k = len(np.unique(y))
samples, features = X.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


# n_clusters = number of clusters,
# init = 'random' first location or 'k-means++' based on empirical probability distribution of the data
# n_init = number of times K-means algorithm is run with different centroid seeds.
#   default = 10 for random and 1 for k-means++. ('auto' or int).
clf = KMeans(n_clusters=k, init='random', n_init=10)

# score the approach using various methods
bench_k_means(clf, "1", X)
