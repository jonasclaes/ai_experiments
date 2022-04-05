from tabnanny import verbose
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from keras import layers

###############################################################################
# Load dataset                                                                #
###############################################################################
mnist = keras.datasets.mnist
(_, _), (test_images, test_labels) = mnist.load_data()

# Input shape is a grayscale image of 28 pixels by 28 pixels.
input_shape = (28, 28, 1)

# Number of output classes (possibilities)
num_classes = 10

# Make sure the shape is (28, 28, 1)
test_images = np.expand_dims(test_images, -1)

###############################################################################
# Keras preprocessing                                                         #
###############################################################################
# Convert class vectors to binary class matrices
converted_testing_labels = keras.utils.to_categorical(test_labels, num_classes)

###############################################################################
# Inference                                                                   #
###############################################################################

model = keras.models.load_model('model')

###############################################################################
# Show predictions                                                            #
###############################################################################
predictions = model.predict(test_images)
plt.figure(figsize=(15,15))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    label = plt.xlabel(str(test_labels[i]) + " (a) = " + str(np.argmax(predictions[i])) + "(p)")
    if test_labels[i] == np.argmax(predictions[i]):
        label.set_color("green")
    else:
        label.set_color("red")
plt.show()
