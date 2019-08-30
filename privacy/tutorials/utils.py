from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import tensorflow as tf  
import numpy as np


def load_dataset(train_feat, train_label, test_feat=None, test_label=None):
    train_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
    train_y = np.genfromtxt(train_label, dtype='int32')
    min_y = np.min(train_y)
    train_y -= min_y
    if test_feat is not None and test_label is not None:
        test_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
        test_y = np.genfromtxt(train_label, dtype='int32')
        test_y -= min_y
    else:
        test_x = None
        test_y = None
    return train_x, train_y, test_x, test_y


def load_trained_indices():
    fname = MODEL_PATH + 'data_indices.npz'
    with np.load(fname) as f:
        indices = [f['arr_%d' % i] for i in range(len(f.files))]
    return indices


def get_data_indices(data_size, target_train_size=int(1e4), sample_target_data=True):
    train_indices = np.arange(data_size)
    if sample_target_data:
        target_data_indices = np.random.choice(train_indices, target_train_size, replace=False)
        shadow_indices = np.setdiff1d(train_indices, target_data_indices)
    else:
        target_data_indices = train_indices[:target_train_size]
        shadow_indices = train_indices[target_train_size:]
    return target_data_indices, shadow_indices


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
  test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels


def load_cifar10():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.cifar10.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

#   train_data = np.array(train_data, dtype=np.float32)
#   test_data = np.array(test_data, dtype=np.float32)

#   train_labels = np.array(train_labels, dtype=np.int32)
#   test_labels = np.array(test_labels, dtype=np.int32)

  num_classes = 10
  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)
  train_data = train_data.astype('float32')
  test_data = test_data.astype('float32')
  train_data  /= 255
  test_data /= 255
  return train_data, train_labels, test_data, test_labels