# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Link to MNIST tutorial: https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb
# Link to dropout turorial
import tensorflow as tf;
import numpy as np;
from keras.datasets import mnist;
from keras.models import Sequential;
from keras.layers.core import Dropout, Activation, Dense;
from keras.utils import np_utils;

#10 different numbers.
num_classes = 10;
#Sets the dropout ratio.
dropout_prob = 0.2;

#loads in the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data();

#Reshapes the images from 28x28 to 784 array
x_train = x_train.reshape(60000, 784);
x_test = x_test.reshape(10000, 784);


# For preprocssing MNIST: https://www.kaggle.com/code/damienbeneschi/mnist-eda-preprocessing-classifiers/notebook


#Changes imput from scalar RGB value to binary White or Black
x_train = x_train / 255;
x_test = x_test / 255;
#Remove Pixles of constant intensity to process faster

    

def runDropout():
    dropout_model = Sequential();
    model.add(Dense(4096, input_shape=(784,), Activation = "relu"));
    model.add(Dropout(dropout_prob));
    model.add(Dense(4096, input_shape=(784,), Activation = "relu"));
    model.add(Dropout(dropout_prob));


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
