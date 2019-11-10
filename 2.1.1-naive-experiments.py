# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Step 0: Create a new conda env for deep learning
#
# One env per *project* seems like the theoretical right way to use conda. But in practice, I know I won't follow that convention for smaller projects.
#
# Compromise: I'll compartmentalize my envs in the style of AWS's deep-learning AMI -- just the `tensorflow_p36` combination for now:
#
#     ====================================================
#            __|  __|_  )
#            _|  (     /   Deep Learning AMI  (Ubuntu)
#           ___|\___|___|
#     ====================================================
#
#     Welcome to Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-1062-aws x86_64v)
#
#     Please use one of the following commands to start the required environment with the framework of your choice:
#     for MXNet(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _______________________________ source activate mxnet_p36
#     for MXNet(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _______________________________ source activate mxnet_p27
#     for TensorFlow(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _____________________ source activate tensorflow_p36 <<<
#     for TensorFlow(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _____________________ source activate tensorflow_p27
#     for Theano(+Keras2) with Python3 (CUDA 9.0) _______________________________________________ source activate theano_p36
#     for Theano(+Keras2) with Python2 (CUDA 9.0) _______________________________________________ source activate theano_p27
#     for PyTorch with Python3 (CUDA 9.0 and Intel MKL) ________________________________________ source activate pytorch_p36
#     for PyTorch with Python2 (CUDA 9.0 and Intel MKL) ________________________________________ source activate pytorch_p27
#     for CNTK(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _________________________________ source activate cntk_p36
#     for CNTK(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _________________________________ source activate cntk_p27
#     for Caffe2 with Python2 (CUDA 9.0) ________________________________________________________ source activate caffe2_p27
#     for Caffe with Python2 (CUDA 8.0) __________________________________________________________ source activate caffe_p27
#     for Caffe with Python3 (CUDA 8.0) __________________________________________________________ source activate caffe_p35
#     for Chainer with Python2 (CUDA 9.0 and Intel iDeep) ______________________________________ source activate chainer_p27
#     for Chainer with Python3 (CUDA 9.0 and Intel iDeep) ______________________________________ source activate chainer_p36
#     for base Python2 (CUDA 9.0) __________________________________________________________________ source activate python2
#     for base Python3 (CUDA 9.0) __________________________________________________________________ source activate python3

# Following the DL AMI offerings, I'm gonna stick with python 3.6. For what it's worth, Tensorflow 2.0 doesn't support 3.7 yet, at the time of this writing. Online discussions suggest that docker images are the way to go for TF, but I'll see how far conda gets me.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as keras_tf
keras_tf.__version__

import keras as keras_itself
keras_itself.__version__

keras.backend.backend()

# # A first look at a neural network
#
# This notebook contains the code samples found in Chapter 2, Section 1 of [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). Note that the original text features far more content, in particular further explanations and figures: in this notebook, you will only find source code and related comments.
#
# ----
#
# We will now take a look at a first concrete example of a neural network, which makes use of the Python library Keras to learn to classify 
# hand-written digits. Unless you already have experience with Keras or similar libraries, you will not understand everything about this 
# first example right away. You probably haven't even installed Keras yet. Don't worry, that is perfectly fine. In the next chapter, we will 
# review each element in our example and explain them in detail. So don't worry if some steps seem arbitrary or look like magic to you! 
# We've got to start somewhere.
#
# The problem we are trying to solve here is to classify grayscale images of handwritten digits (28 pixels by 28 pixels), into their 10 
# categories (0 to 9). The dataset we will use is the MNIST dataset, a classic dataset in the machine learning community, which has been 
# around for almost as long as the field itself and has been very intensively studied. It's a set of 60,000 training images, plus 10,000 test 
# images, assembled by the National Institute of Standards and Technology (the NIST in MNIST) in the 1980s. You can think of "solving" MNIST 
# as the "Hello World" of deep learning -- it's what you do to verify that your algorithms are working as expected. As you become a machine 
# learning practitioner, you will see MNIST come up over and over again, in scientific papers, blog posts, and so on.

# The MNIST dataset comes pre-loaded in Keras, in the form of a set of four Numpy arrays:

# +
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# -

train_images.shape

test_images.shape

import matplotlib.pyplot as plt

train_labels[:10]

# `train_images` and `train_labels` form the "training set", the data that the model will learn from. The model will then be tested on the 
# "test set", `test_images` and `test_labels`. Our images are encoded as Numpy arrays, and the labels are simply an array of digits, ranging 
# from 0 to 9. There is a one-to-one correspondence between the images and the labels.
#
# Let's have a look at the training data:

train_images.shape

len(train_labels)

train_labels


def show_digits(images, labels, preds, nrows, ncols, i0=0):
    n = nrows * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    
    for i, ax in enumerate(axes.flatten()):
        idx = i0 + i
        
        ax.imshow(images[idx], cmap='Greys_r')
        
        title_string = f'image {idx}\nlabel={labels[idx]}'
        if preds is not None:
            title_string += f', pred={preds[idx]}'
        ax.set_title(title_string)
        
    plt.tight_layout()


show_digits(train_images, train_labels, None, 2, 5, 0)

# Let's have a look at the test data:

test_images.shape

len(test_labels)

test_labels

show_digits(test_images, test_labels, None, 2, 5, 0)

# Our workflow will be as follows: first we will present our neural network with the training data, `train_images` and `train_labels`. The 
# network will then learn to associate images and labels. Finally, we will ask the network to produce predictions for `test_images`, and we 
# will verify if these predictions match the labels from `test_labels`.
#
# Let's build our network -- again, remember that you aren't supposed to understand everything about this example just yet.

# +
# # Note to self: how much has the keras API changed since the writing of this book?
# from keras import models
# from keras import layers

# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation='softmax'))
# -

# 2019-11-07: let's try to modernize the above code cell based on
# https://www.tensorflow.org/guide/keras
from tensorflow import keras
keras.__version__

import keras as keras_direct_import
keras_direct_import.__version__

# +
from tensorflow.keras import layers
model = keras.Sequential()

# Densely connected layer with 512 units
# What happens if we omit the input shape?
# model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(512, activation='relu'))

# Another densely connected layer with 10 (output) units
model.add(layers.Dense(10, activation='softmax'))
# -

#
# The core building block of neural networks is the "layer", a data-processing module which you can conceive as a "filter" for data. Some 
# data comes in, and comes out in a more useful form. Precisely, layers extract _representations_ out of the data fed into them -- hopefully 
# representations that are more meaningful for the problem at hand. Most of deep learning really consists of chaining together simple layers 
# which will implement a form of progressive "data distillation". A deep learning model is like a sieve for data processing, made of a 
# succession of increasingly refined data filters -- the "layers".
#
# Here our network consists of a sequence of two `Dense` layers, which are densely-connected (also called "fully-connected") neural layers. 
# The second (and last) layer is a 10-way "softmax" layer, which means it will return an array of 10 probability scores (summing to 1). Each 
# score will be the probability that the current digit image belongs to one of our 10 digit classes.
#
# To make our network ready for training, we need to pick three more things, as part of "compilation" step:
#
# * A loss function: the is how the network will be able to measure how good a job it is doing on its training data, and thus how it will be 
# able to steer itself in the right direction.
# * An optimizer: this is the mechanism through which the network will update itself based on the data it sees and its loss function.
# * Metrics to monitor during training and testing. Here we will only care about accuracy (the fraction of the images that were correctly 
# classified).
#
# The exact purpose of the loss function and the optimizer will be made clear throughout the next two chapters.

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#
# Before training, we will preprocess our data by reshaping it into the shape that the network expects, and scaling it so that all values are in 
# the `[0, 1]` interval. Previously, our training images for instance were stored in an array of shape `(60000, 28, 28)` of type `uint8` with 
# values in the `[0, 255]` interval. We transform it into a `float32` array of shape `(60000, 28 * 28)` with values between 0 and 1.

train_images.shape

# +
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# -

# We also need to categorically encode the labels, a step which we explain in chapter 3:

train_labels[0:5]

# +
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# -

train_labels[0:5]

import pandas as pd
pd.DataFrame(train_labels[0:5].astype(int))

# We are now ready to train our network, which in Keras is done via a call to the `fit` method of the network: 
# we "fit" the model to its training data.

history = model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_split=0.1)

# Two quantities are being displayed during training: the "loss" of the network over the training data, and the accuracy of the network over 
# the training data.
#
# We quickly reach an accuracy of 0.989 (i.e. 98.9%) on the training data. Now let's check that our model performs well on the test set too:

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.grid()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

#
# Our test set accuracy turns out to be 97.8% -- that's quite a bit lower than the training set accuracy. 
# This gap between training accuracy and test accuracy is an example of "overfitting", 
# the fact that machine learning models tend to perform worse on new data than on their training data. 
# Overfitting will be a central topic in chapter 3.
#
# This concludes our very first example -- you just saw how we could build and a train a neural network to classify handwritten digits, in 
# less than 20 lines of Python code. In the next chapter, we will go in detail over every moving piece we just previewed, and clarify what is really 
# going on behind the scenes. You will learn about "tensors", the data-storing objects going into the network, about tensor operations, which 
# layers are made of, and about gradient descent, which allows our network to learn from its training examples.

# <div style='background:#f4f4ff'>
#
# # Epilogue: miscellaneous experiments
#
# </div>

# ## Sanity checks by hand

train_images.shape

preds = model.predict(train_images[0:10, :]).argmax(axis=1)

preds

show_digits(train_images[:10].reshape(10, 28, 28), 
            train_labels[:10].argmax(axis=1), 
            preds, 2, 5)

from keras.utils import plot_model
plot_model(model, rankdir='LR')

# Looks about right.
