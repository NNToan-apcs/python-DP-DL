# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training a CNN on MNIST with differentially private SGD optimizer."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from distutils.version import LooseVersion


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys, os, errno
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer, DPGradientDescentOptimizer

from model_keras import get_model_stucture
from utils import load_mnist_keras, load_cifar10

if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer
else:
  GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 200, 'Batch size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer('softmax_epochs', 1, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 200, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS

def create_folder_if_not_exist(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

def save_model_stats(filename, history):
  create_folder_if_not_exist(filename)
  f = open(filename, "w")
  for item in history.keys():
    f.write(str(item) + ": " + str(history[item]) + "\n")
  f.close()

def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def train(dataset, model_name, mode='nn'):
  with tf.device('gpu:0'):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load training and test data.
    train_data, train_labels, test_data, test_labels = dataset
    logfile = FLAGS.dataset + "_dp" if FLAGS.dpsgd else FLAGS.dataset
    if mode == "softmax":
      FLAGS.learning_rate = 0.01
      epochs = FLAGS.softmax_epochs
      logfile = logfile + "\\softmax_" + str(epochs) + "\\" + model_name
      num_classes = 2
      train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
      test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)
    else:
      epochs = FLAGS.epochs
      logfile = logfile + "\\sgd_" + str(epochs) + "\\" + model_name
    # Create logs path
    if os.path.exists(".\\logs\\" + logfile):
        print("FOLDER IS EXISTS")
        i=1
        while os.path.exists(".\\logs\\" + logfile + '_' + str(i)):
          i+=1
        logfile = logfile + '_' + str(i) 

    
    tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir=".\\logs\\{}".format(logfile))
    if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
      raise ValueError('Number of microbatches should divide evenly batch_size')
    
    # Define a sequential Keras model
    input_shape = train_data[1].shape
    # input(input_shape)
    model = get_model_stucture(mode, input_shape)
    
    if FLAGS.dpsgd:
      optimizer = DPGradientDescentGaussianOptimizer(
      # optimizer = DPGradientDescentOptimizer(
          l2_norm_clip=FLAGS.l2_norm_clip,
          noise_multiplier=FLAGS.noise_multiplier,
          num_microbatches=FLAGS.microbatches,
          learning_rate=FLAGS.learning_rate)
      # Compute vector of per-example loss rather than its mean over a minibatch.
      loss = tf.keras.losses.CategoricalCrossentropy(
          from_logits=True, reduction=tf.losses.Reduction.NONE)
    else:
      optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
      # input(train_labels[0])
      # input(type(train_labels[0]))
      # input(type(train_labels[0]) == "numpy.ndarray")
      # if (model_name=="mnist"):
      loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
      # else:
      #   loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)
    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train model with Keras
    if FLAGS.dpsgd:
      history = model.fit(train_data, train_labels,
                  epochs=epochs,
                  validation_data=(test_data, test_labels),
                  batch_size=FLAGS.batch_size)
      # list all data in history
      print(history.history.keys())
      # summarize history for accuracy
      plt.plot(history.history['acc'])
      plt.plot(history.history['val_acc'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
      # summarize history for loss
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
    else:
      history = model.fit(train_data, train_labels,
              epochs=epochs,
              validation_data=(test_data, test_labels),
              batch_size=FLAGS.batch_size,
              callbacks=[tb_callbacks])
    # Compute the privacy budget expended.
    model_stats_dir = ".\\model_stats\\" + logfile + ".txt"
    save_model_stats(model_stats_dir, history.history)
    if FLAGS.dpsgd:
      eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
      f = open(model_stats_dir, "a")
      f.write("Eps: " +str(eps))
      f.close()
      print('For delta=1e-5, the current epsilon is: %.2f' % eps)
    else:
      print('Trained with vanilla non-private SGD optimizer')

    # serialize model to JSON
    model_json = model.to_json()
    create_folder_if_not_exist(".\\logs\\" + logfile +".json")
    with open(".\\logs\\" + logfile +".json", "w") as json_file:
      json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(".\\logs\\" + logfile +".h5")
    print("Saved model to disk")
    return model

def main(unused_argv):
  # dataset = load_mnist()
  # x,y,x_test,y_test = dataset
  # train(dataset,'mnist')


  dataset = load_cifar10()
  x,y,x_test,y_test = dataset
  # y = tf.one_hot(y, depth=10) 
  # input(y)
  train(dataset,'cifar10')
  
  # later...
  
  # load json and create model
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  

  loaded_model = tf.keras.models.model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("model.h5")
  # print("Loaded model from disk")
  # # predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)
  print(loaded_model.predict(x)) 
  # results = loaded_model.evaluate(x_test, y_test, batch_size=128)
  # print(results)
if __name__ == '__main__':
  app.run(main)
