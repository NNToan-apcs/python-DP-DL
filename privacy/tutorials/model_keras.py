from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dropout, Activation, Flatten, InputLayer, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
def get_model_stucture(mode, inputShape):
    # input(mode == "cifar10_vgg")
    if( mode == "mnist_cnn"):
      model = get_mnist_cnn(inputShape)
    elif(mode == "cifar10_cnn"):
      # https://keras.io/examples/cifar10_cnn/
      model = get_cifar10_cnn(inputShape)
    elif(mode == "softmax"):
      model = get_softmax(inputShape)
    elif(mode == "cifar10_vgg"):
      # "https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py"
      model = get_cifar10_vgg(inputShape)
    elif(mode == "cifar10_mlleak"):
      model = get_cifar10_mlleak(inputShape)
    elif(mode == "cifar10_resnet20"):
      #https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
      model = get_cifar10_resnet20(inputShape)  
    elif(mode == "cifar10_resnet32"):
      #https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
      model = get_cifar10_resnet32(inputShape) 
    elif(mode == "cifar10_resnet56"):
      #https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
      model = get_cifar10_resnet56(inputShape) 
    else:
      model = None
    return model

def get_cifar10_resnet32(inputShape):
  depth = 32
  model = resnet_v1(input_shape=inputShape, depth=depth)
  return model

def get_cifar10_resnet20(inputShape):
  depth = 20
  model = resnet_v1(input_shape=inputShape, depth=depth)
  return model

def get_cifar10_resnet56(inputShape):
  depth = 56
  model = resnet_v1(input_shape=inputShape, depth=depth)
  return model
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
  """ResNet Version 1 Model builder [a]
  Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
  Last ReLU is after the shortcut connection.
  At the beginning of each stage, the feature map size is halved (downsampled)
  by a convolutional layer with strides=2, while the number of filters is
  doubled. Within each stage, the layers have the same number filters and the
  same number of filters.
  Features maps sizes:
  stage 0: 32x32, 16
  stage 1: 16x16, 32
  stage 2:  8x8,  64
  The Number of parameters is approx the same as Table 6 of [a]:
  ResNet20 0.27M
  ResNet32 0.46M
  ResNet44 0.66M
  ResNet56 0.85M
  ResNet110 1.7M
  # Arguments
      input_shape (tensor): shape of input image tensor
      depth (int): number of core convolutional layers
      num_classes (int): number of classes (CIFAR10 has 10)
  # Returns
      model (Model): Keras model instance
  """
  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
  # Start model definition.
  num_filters = 16
  num_res_blocks = int((depth - 2) / 6)

  inputs = Input(shape=input_shape)
  x = resnet_layer(inputs=inputs)
  # Instantiate the stack of residual units
  for stack in range(3):
    for res_block in range(num_res_blocks):
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(inputs=x,
                      num_filters=num_filters,
                      strides=strides)
      y = resnet_layer(inputs=y,
                      num_filters=num_filters,
                      activation=None)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match
        # changed dims
        x = resnet_layer(inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False)
      x = tf.keras.layers.add([x, y])
      x = Activation('relu')(x)
    num_filters *= 2

  # Add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x = AveragePooling2D(pool_size=8)(x)
  y = Flatten()(x)
  outputs = Dense(num_classes,
                  activation='softmax',
                  kernel_initializer='he_normal')(y)

  # Instantiate model.
  model = Model(inputs=inputs, outputs=outputs)
  return model



def get_cifar10_mlleak(inputShape):
  
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
  n_hidden = 64
  n_out = 10
  model = Sequential()
  model.add(InputLayer(input_shape=inputShape))
  model.add(Dense(n_hidden))
  model.add(Activation('tanh'))
  model.add(Flatten())
  model.add(Dense(n_out))
  model.add(Activation('softmax'))
  return model

def get_cifar10_vgg(inputShape):
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
      MaxPooling2D(2, 1),
      Conv2D(32, 4,
          strides=2,
          padding='valid',
          activation='relu'),
      MaxPooling2D(2, 1),
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
   