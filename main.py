# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Link to MNIST tutorial: https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb
# Link to dropout turorial
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Dense
from keras.utils import np_utils

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
x_train = np.floor(x_train / 128).astype(int)
x_test = np.floor(x_test / 128).astype(int)
x_all = np.append(x_train, x_test, axis=0)
y_all = np.append(y_train, y_test, axis=0)
print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
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


x_train_trimmed, x_test_trimmed, pix_dropped = remove_constant_intensity(x_all, x_train, x_test)
print("x trimmed", x_train_trimmed.shape)
pixel_count = x_train_trimmed.shape[1]


# To save the model https://www.tensorflow.org/tutorials/keras/save_and_load
def getDropoutModel():
    print(pixel_count)
    dropout_model = Sequential([
        Dense(4096, input_shape=(pixel_count,), activation="relu"),
        Dropout(dropout_prob),
        Dense(4096, activation="relu"),
        Dropout(dropout_prob),
        Dense(10, activation="softmax")])
    dropout_model.compile(loss='sparse_categorical_crossentropy', metrics=[metrics.SparseCategoricalAccuracy()],
                          optimizer='adam')
    return dropout_model, "dropoutCheckpoints"


def getBaseModel():
    base_model = Sequential([
        Dense(4096, input_shape=(pixel_count,), activation="relu"),
        Dense(4096, activation="relu"),
        Dense(10, activation="softmax")])
    base_model.compile(loss='sparse_categorical_crossentropy', metrics=[metrics.SparseCategoricalAccuracy()],
                       optimizer='adam')
    return base_model, "baseCheckpoints"


baseModel, baseCheckpointPath = getBaseModel()
baseCheckpointPath = baseCheckpointPath + "/cp-epoch-{epoch:02d}-loss-{val_loss:.2f}.keras"

dropoutModel, dropoutCheckpointPath = getDropoutModel()
dropoutCheckpointPath = dropoutCheckpointPath + "/cp-epoch-{epoch:02d}-loss-{val_loss:.2f}.keras"

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

baseModel.fit(x_train_trimmed,
              y_train,
              epochs=10,
              batch_size=256,
              validation_data=(x_test_trimmed, y_test),
              verbose=1,
              callbacks=[base_callback_checkpoint])

baseScore = baseModel.evaluate(x_test_trimmed, y_test, verbose=1)

print('Base Test loss:', baseScore[0])
print('Base Test accuracy:', baseScore[1])

dropoutModel.fit(x_train_trimmed,
                 y_train,
                 epochs=10,
                 batch_size=32,
                 validation_data=(x_test_trimmed, y_test),
                 verbose=1,
                 callbacks=[dropout_callback_checkpoint])

baseScore = baseModel.evaluate(x_test_trimmed, y_test,
                               verbose=1)

dropoutScore = dropoutModel.evaluate(x_test_trimmed, y_test,
                                     show_accuracy=True, verbose=1)

print('Base Test loss:', baseScore[0])
print('Base Test accuracy:', baseScore[1])

print('Dropout Test loss:', dropoutScore[0])
print('Dropout Test accuracy:', dropoutScore[1])


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
