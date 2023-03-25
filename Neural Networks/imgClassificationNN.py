"""
A neural network is a supervised machine learning algorithm. We can train neural networks to solve classification or
regression problems. Yet, utilizing neural networks for a machine learning problem has its pros and cons.

Building a neural network model requires answering lots of architecture-oriented questions. Depending on the complexity
 of the problem and available data, we can train neural networks with different sizes and depths. Furthermore, we need
 to preprocess our input features, initialize the weights, add bias if needed, and choose appropriate activation
 functions.

  One method for improving network generalization is to use a network that is just large enough to provide an adequate
   fit. The larger network you use, the more complex the functions the network can create. If you use a small enough
   network, it will not have enough power to overfit the data. Another is early stopping to prevent the data being
   used too much and not generalising.
"""
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# scale images to have values under 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# show the images in the dataset as their array values
# print(train_images[7])

# show the images in the dataset as an image
# can change colour-maps e.g. black and white by setting cmap=plt.cm.binary inside function
# plt.imshow(train_images[7])
# plt.show()

# create the neural network
model = keras.Sequential([
    # flattens input to 1D shape representing image
    keras.layers.Flatten(input_shape=(28, 28)),
    # create dense layer approx 15-20% size of input, activation function can be relu, sigmoid, etc.
    keras.layers.Dense(128, activation="relu"),
    # output layer, 10 clothing items, softmax sets probability of each result, and they all sum to 1.
    keras.layers.Dense(10, activation="softmax")
])

# loss function is what we try and minimise, metrics are what we monitor
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# fit training data to the model, epochs are number of partitions data is split into to run the model over
model.fit(train_images, train_labels, epochs=5)

# test model on test data and get accuracy scores
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Acc: ", test_acc)

# validate predictions by printing some images
prediction = model.predict(test_images)
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + class_names[test_labels[i]])
    plt.title('Prediction: ' + class_names[np.argmax(prediction[i])])
    plt.show()
