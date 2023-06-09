# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Link to MNIST tutorial: https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb
# Link to dropout turorial
import tensorflow as tf
import numpy as np
#from keras import Sequential
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt

# 10 different numbers.
num_classes = 10
# Sets the dropout ratio.
dropout_prob = 0.5

# loads in the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
pixel_count = 784
# Reshapes the images from 28x28 to 784 array
x_train = x_train.reshape(60000, pixel_count)
x_test = x_test.reshape(10000, pixel_count)

# For preprocssing MNIST: https://www.kaggle.com/code/damienbeneschi/mnist-eda-preprocessing-classifiers/notebook

# Changes input from scalar RGB value to binary White or Black
x_train_trimmed = np.floor(x_train / 128).astype(int)
x_test_trimmed = np.floor(x_test / 128).astype(int)
x_all = np.append(x_train_trimmed, x_test_trimmed, axis=0)
y_all = np.append(y_train, y_test, axis=0)
print("x train: ", x_train_trimmed.shape)
print("x test: ", x_test_trimmed.shape)
print("x all: ", x_all.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)
print("y all: ", y_all.shape)


# Remove Pixles of constant intensity to process faster
def remove_constant_intensity(all_x, trimmed_x_train, trimmed_x_test):
    trimmed_x = np.copy(all_x)
    dropped_pix = []
    for inv_col in range(784):
        # starts dropping from the end of the array to not change remaining indexes when dropping
        col = 783 - inv_col
        if trimmed_x[:, col].max() == 0 or trimmed_x[:, col].min() == 1:
            trimmed_x_train = np.delete(trimmed_x_train, col, 1)
            trimmed_x_test = np.delete(trimmed_x_test, col, 1)
            dropped_pix.append(col)

    print('dropped pixels: ', dropped_pix)
    return trimmed_x_train, trimmed_x_test, dropped_pix


x_train_trimmed, x_test_trimmed, pix_dropped = remove_constant_intensity(x_all, x_train_trimmed, x_test_trimmed)
print("x trimmed", x_train_trimmed.shape)
pixel_count = x_train_trimmed.shape[1]


# Create the Models
def getDropoutModel():
    print(pixel_count)
    dropout_model = Sequential([
        Dense(4096, input_shape=(pixel_count,), activation="relu"),
        Dropout(dropout_prob),
        Dense(4096, activation="relu"),
        Dropout(dropout_prob),
        Dense(10, activation="softmax")])
    dropout_model.compile(loss='sparse_categorical_crossentropy',
                          metrics=[metrics.SparseCategoricalAccuracy()],
                          optimizer='adam')
    return dropout_model, "dropoutCheckpoints"


def getBaseModel():
    base_model = Sequential([
        Dense(4096, input_shape=(pixel_count,), activation="relu"),
        Dense(4096, activation="relu"),
        Dense(10, activation="softmax")])
    base_model.compile(loss='sparse_categorical_crossentropy',
                       metrics=[metrics.SparseCategoricalAccuracy()],
                       optimizer='adam')
    return base_model, "baseCheckpoints"

#Saves the model to cp-epoch-{The Epoch Number}-loss-{The Loss Value}.keras
baseModel, baseCheckpointPath = getBaseModel()
baseCheckpointPath = baseCheckpointPath + "/cp-epoch-{epoch:02d}-loss-{val_loss:.2f}.keras"

dropoutModel, dropoutCheckpointPath = getDropoutModel()
dropoutCheckpointPath = dropoutCheckpointPath + "/cp256-epoch-{epoch:02d}-loss-{val_loss:.2f}.keras"

base_callback_checkpoint = ModelCheckpoint(filepath=baseCheckpointPath,
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_weights_only=True,
                                           save_best_only=False)

dropout_callback_checkpoint = ModelCheckpoint(filepath=dropoutCheckpointPath,
                                              monitor='val_accuracy',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=False)
def baseFit():
    baseModel.fit(x_train_trimmed,
                y_train,
                epochs=10,
                batch_size=256,
                validation_data=(x_test_trimmed, y_test),
                verbose=1,
                callbacks=[base_callback_checkpoint])
    return baseModel

def dropoutFit():
    dropoutModel.fit(x_train_trimmed,
                    y_train,
                    epochs=10,
                    batch_size=256,
                    validation_data=(x_test_trimmed, y_test),
                    verbose=1,
                    callbacks=[dropout_callback_checkpoint])
    return dropoutModel

#Train the models
#baseModel = baseFit()
#dropoutModel = dropoutFit()

#Load the models from previously trained checkpoints
baseModel.load_weights('baseCheckpoints/cp-epoch-10-loss-0.12.keras')
dropoutModel.load_weights('dropoutCheckpoints/cp256-epoch-10-loss-0.08.keras')

#Evaluate the models
baseScore = baseModel.evaluate(x_test_trimmed, y_test,
                               verbose=1)

dropoutScore = dropoutModel.evaluate(x_test_trimmed, y_test,
                                     verbose=1)

print('Base Test loss:', baseScore[0])
print('Base Test accuracy:', baseScore[1])

print('Dropout Test loss:', dropoutScore[0])
print('Dropout Test accuracy:', dropoutScore[1])

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
#predictions = np.around(dropoutModel.predict(x_test_trimmed)).astype(int)
predicted_classes = np.argmax(dropoutModel.predict(x_test_trimmed), axis=1)
# Check which items we got right / wrong
correct_predictions = np.nonzero(predicted_classes == y_test)[0]
incorrect_predictions = np.nonzero(predicted_classes != y_test)[0]

plt.figure()
#Show the first nine correct predictions
for i, correct in enumerate(correct_predictions[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
plt.tight_layout()
plt.show()

#Show the first nine incorrect predictions
plt.figure()
for i, incorrect in enumerate(incorrect_predictions[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
plt.tight_layout()
plt.show()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
