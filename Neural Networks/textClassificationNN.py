import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# create training and testing data, limit number of words to 10,000 from dataset
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
