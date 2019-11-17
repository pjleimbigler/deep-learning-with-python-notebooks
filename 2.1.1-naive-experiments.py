# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
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
#     Welcome to Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-1062-aws x86_64v)
#
#     Please use one of the following commands to start the required environment with the framework of your choice:
#     for MXNet(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _______________________________ source activate mxnet_p36
#     for MXNet(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _______________________________ source activate mxnet_p27
#     for TensorFlow(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _____________________ source activate tensorflow_p36 <<<
#     for TensorFlow(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _____________________ source activate tensorflow_p27
#     for Theano(+Keras2) with Python3 (CUDA 9.0) _______________________________________________ source activate theano_p36
#     [...]

# Following the DL AMI offerings, I'm gonna stick with python 3.6. For what it's worth, Tensorflow 2.0 doesn't support 3.7 yet, at the time of this writing. Online discussions suggest that docker images are the way to go for TF, but I'll see how far conda gets me.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
keras.__version__

# +
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# -

train_images.shape

train_labels[:10]


def show_digits(images, labels, preds, nrows, ncols, i0=0):
    n = nrows * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    
    for i, ax in enumerate(axes.flatten()):
        idx = i0 + i
        
        ax.imshow(images[idx], cmap='Greys_r')
        
        title_string = f'image {idx}\nlabel={labels[idx]}'
        c = 'k'
        
        if preds is not None:
            title_string += f', pred={preds[idx]}'
            if preds[idx] != labels[idx]:
                c='r'
        ax.set_title(title_string, c=c)
        
    plt.tight_layout()


show_digits(train_images, train_labels, None, 2, 5, 0)

test_images.shape

test_labels

show_digits(test_images, test_labels, None, 2, 5, 0)

# +
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# -

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# +
from tensorflow.keras.utils import to_categorical

train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

train_labels_cat[0:5]
# -

history = model.fit(train_images, train_labels_cat, epochs=20, batch_size=128, validation_split=0.1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.grid()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels_cat)

print('test_acc:', test_acc)

preds = model.predict(train_images[0:10, :]).argmax(axis=1)
preds

show_digits(train_images[:10].reshape(10, 28, 28), 
            train_labels[:10], 
            preds, 2, 5)
