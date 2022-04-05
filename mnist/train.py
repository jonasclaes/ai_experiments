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
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Input shape is a grayscale image of 28 pixels by 28 pixels.
input_shape = (28, 28, 1)

# Number of output classes (possibilities)
num_classes = 10

# Make sure the shape is (28, 28, 1)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

###############################################################################
# Show dataset                                                                #
###############################################################################
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(train_labels[i])
# plt.show()

###############################################################################
# Keras preprocessing                                                         #
###############################################################################
# Convert class vectors to binary class matrices
converted_training_labels = keras.utils.to_categorical(train_labels, num_classes)

# Convert class vectors to binary class matrices
converted_testing_labels = keras.utils.to_categorical(test_labels, num_classes)

###############################################################################
# Keras model                                                                 #
###############################################################################
def build_model():
    # Define a sequential model
    model = keras.Sequential(
        [
            # Add an input layer in the input shape
            keras.Input(shape=input_shape),

            # Add an image rescaling layer
            layers.Rescaling(scale=1./255),

            # Add a convolutional layer
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),

            # Add a max pooling layer
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Add a convolutional layer
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),

            # Add a max pooling layer
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Add a convolutional layer
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),

            # Add a max pooling layer
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Flatten the layers
            layers.Flatten(),

            # Add a dropout
            layers.Dropout(0.4),

            # Add an output layer
            layers.Dense(num_classes, activation="softmax")
        ]
    )

    # Summarize the model
    model.summary()
    
    # Compile model
    model.compile(
        optimizer = "adam",
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])
    return model

###############################################################################
# Training                                                                    #
###############################################################################

# Build the model
model = build_model()

# Configure training params
batch_size = 128
epochs = 35

# Train the model
history = model.fit(train_images, converted_training_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

###############################################################################
# Evaluation                                                                  #
###############################################################################
loss, accuracy = model.evaluate(test_images, converted_testing_labels, verbose=0)
print("Test loss: %.2f" % loss)
print("Test accuracy: %.2f" % accuracy)

model.save('model')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# ###############################################################################
# # Show predictions                                                            #
# ###############################################################################
# predictions = model.predict(normalized_testing_data)
# plt.figure(figsize=(15,15))
# for i in range(100):
#     plt.subplot(10,10,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images[i], cmap=plt.cm.binary)
#     label = plt.xlabel(str(test_labels[i]) + " (a) = " + str(np.argmax(predictions[i])) + "(p)")
#     if test_labels[i] == np.argmax(predictions[i]):
#         label.set_color("green")
#     else:
#         label.set_color("red")
# plt.show()
