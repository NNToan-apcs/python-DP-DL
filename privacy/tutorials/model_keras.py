from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, InputLayer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
def get_model_stucture(mode, inputShape):
    # input(mode)
    # input(mode == "cifar10_vgg")
    if( mode == "mnist"):
      model = get_mnist_cnn(inputShape)
    elif(mode == "cifar10"):
      # https://keras.io/examples/cifar10_cnn/
      model = get_cifar10_cnn(inputShape)
    elif(mode == "softmax"):
      model = get_softmax(inputShape)
    elif(mode == "cifar10_vgg"):
      # "https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py"
      model = get_cifer10_vgg(inputShape)
    elif(mode == "cifar10_mlleak"):
      model = get_cifar10_mlleak(inputShape)
    else:
      model = None
    return model

def get_cifer10_mlleak(inputShape):
  # net = dict()
  #   net['input'] = lasagne.layers.InputLayer((None, n_in))
  #   net['fc'] = lasagne.layers.DenseLayer(
  #       net['input'],
  #       num_units=n_hidden,
  #       nonlinearity=lasagne.nonlinearities.tanh)
  #   net['output'] = lasagne.layers.DenseLayer(
  #       net['fc'],
  #       num_units=n_out,
  #       nonlinearity=lasagne.nonlinearities.softmax)
  model = Sequential()
  model.add(Conv2D(64, (3,3), inputshape=inputShape))
  model.add(Activation('tanh'))
  model.add(Dense(10))
  model.add(Activation('softmax'))
  return model

def get_cifer10_vgg(inputShape):
  weight_decay = 0.0005
  x_shape = inputShape
  num_classes = 10
  model = Sequential()
  model.add(Conv2D(64, (3, 3), padding='same',
                    input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))

  model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))


  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))


  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.4))

  model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
  model.add(Activation('relu'))
  model.add(BatchNormalization())

  model.add(Dropout(0.5))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  return model

def get_mnist_cnn(inputShape):
    model = tf.keras.Sequential([
      Conv2D(16, 8,
          strides=2,
          padding='same',
          activation='relu',
          input_shape=inputShape),
      MaxPool2D(2, 1),
      Conv2D(32, 4,
          strides=2,
          padding='valid',
          activation='relu'),
      MaxPool2D(2, 1),
      Flatten(),
      Dense(32, activation='relu'),
      Dense(10)])
    return model

def get_cifar10_cnn(inputShape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

def get_softmax(inputShape):
    model = tf.keras.Sequential([
      InputLayer(input_shape=inputShape),
      Dense(2, activation='softmax')
    ])
    return model
   