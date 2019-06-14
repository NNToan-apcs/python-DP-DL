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
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from absl import app
from absl import flags

from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf
from privacy.analysis import privacy_ledger
from privacy.analysis.rdp_accountant import compute_rdp_from_ledger
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer

if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer
else:
  GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'strong_composition', False, 'If True, Use strong composition theorem to compute epsilon. If False, '
    'Use RDP.')
flags.DEFINE_boolean(
    'dpsgd', False, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_boolean(
    'moment_accountant', True, 'If True, compute eps using moment_accountant. If False, '
    'compute eps using strong composition.')
flags.DEFINE_float('learning_rate', .25, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.7,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.5, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_float('delta', 1e-5, 'target delta')
flags.DEFINE_integer('epochs', 5, 'Number of epochs')
# flags.DEFINE_integer('epochs', 15, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 256, 'Number of microbatches '
    '(must evenly divide batch_size)')

flags.DEFINE_string('record_dir', "./record_data" , 'Model records dir')
# modeldir =  "/home/toan/Desktop/DL_models/" # UBUNTU
modeldir =  "D:/DL_models/" # Window
modelName = "mnist_sgd_10"
if os.path.exists(modeldir + modelName):
  i=1
  while os.path.exists(modeldir + modelName + '_' + str(i)):
    i+=1
  flags.DEFINE_string('model_dir', modeldir +  modelName + '_' + str(i), 'Model directory')
  flags.DEFINE_string('record_file', modelName + '_' + str(i) + ".txt" , 'Model records file')
else:
  flags.DEFINE_string('model_dir', modeldir +  modelName, 'Model directory')
  flags.DEFINE_string('record_file', modelName + ".txt" , 'Model records file')
class EpsilonPrintingTrainingHook(tf.estimator.SessionRunHook):
  """Training hook to print current value of epsilon after an epoch."""

  def __init__(self, ledger):
    """Initalizes the EpsilonPrintingTrainingHook.

    Args:
      ledger: The privacy ledger.
    """
    self._samples, self._queries = ledger.get_unformatted_ledger()

  def end(self, session):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    samples = session.run(self._samples)
    queries = session.run(self._queries)
    formatted_ledger = privacy_ledger.format_ledger(samples, queries)
    rdp = compute_rdp_from_ledger(formatted_ledger, orders)
    eps = get_privacy_spent(orders, rdp, target_delta=FLAGS.delta)[0]
    print('***************************************************')
    f=open(FLAGS.record_dir + "/" + FLAGS.record_file, "a+")
    f.write('For delta='+ str(FLAGS.delta) + ', the current epsilon is: %.2f \n' % eps)
    print('For delta='+ str(FLAGS.delta) + ', the current epsilon is: %.2f' % eps)
    f.close()
    print('***************************************************')


def cnn_model_fn(features, labels, mode):
  """Model function for a CNN."""

  # Define CNN architecture using tf.keras.layers.
  input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
  y = tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu').apply(input_layer)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu').apply(y)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Flatten().apply(y)
  y = tf.keras.layers.Dense(32, activation='relu').apply(y)
  logits = tf.keras.layers.Dense(10).apply(y)
  logits = tf.identity(logits, name='output')
  # Calculate loss as a vector (to support microbatches in DP-SGD).
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:

    if FLAGS.dpsgd:
      ledger = privacy_ledger.PrivacyLedger(
          population_size=60000,
          selection_probability=(FLAGS.batch_size / 60000))

      # Use DP version of GradientDescentOptimizer. Other optimizers are
      # available in dp_optimizer. Most optimizers inheriting from
      # tf.train.Optimizer should be wrappable in differentially private
      # counterparts by calling dp_optimizer.optimizer_from_args().
      optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
          l2_norm_clip=FLAGS.l2_norm_clip,
          noise_multiplier=FLAGS.noise_multiplier,
          num_microbatches=FLAGS.microbatches,
          ledger=ledger,
          learning_rate=FLAGS.learning_rate)
      training_hooks = [
          EpsilonPrintingTrainingHook(ledger)
      ]
      opt_loss = vector_loss
    else:
      optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
      training_hooks = []
      opt_loss = scalar_loss
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      train_op=train_op,
                                      training_hooks=training_hooks)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(input=logits, axis=1))
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)
  


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.
  assert train_labels.ndim == 1
  assert test_labels.ndim == 1

  return train_data, train_labels, test_data, test_labels

def train(dataset):
    # BLACKBOX MODEL TRAINING 
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
      raise ValueError('Number of microbatches should divide evenly batch_size')

    # Load training and test data.
    train_data, train_labels, test_data, test_labels = dataset
    # Instantiate the tf.Estimator.
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir=FLAGS.model_dir)

    # Create tf.Estimator input functions for the training and test data.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.epochs,
        shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    # Training loop.
    steps_per_epoch = len(train_data) // FLAGS.batch_size
    # Clean records
    f=open(FLAGS.record_dir + "/" + FLAGS.record_file, "w+")
    f.write("learning rate: " + str(FLAGS.learning_rate)+ "\n")
    f.write("noise multiplier: " + str(FLAGS.noise_multiplier) + "\n")
    f.write("Clipping norm: " + str(FLAGS.l2_norm_clip) + "\n")
    f.write("Batch size: " + str(float(FLAGS.batch_size)) + "\n")
    f.write("delta: 10e-5 \n")
    f.close()
    for epoch in range(1, FLAGS.epochs + 1):
      f=open(FLAGS.record_dir + "/" + FLAGS.record_file, "a+")
      print("-------------------------------------------")
      f.write("EPOCH: " + str(epoch) +"/" + str(FLAGS.epochs) + "\n")
      print("EPOCH", epoch,"/", FLAGS.epochs)
      print("-------------------------------------------")
      f.close()
      # Train the model for one epoch.
      mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)

      # Evaluate the model and print results
      eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
      test_accuracy = eval_results['accuracy']
      f=open(FLAGS.record_dir + "/" + FLAGS.record_file, "a+")
      f.write('Test accuracy after %d epochs is: %.3f \n' % (epoch, 100*test_accuracy))
      print('Test accuracy after %d epochs is: %.3f' % (epoch, 100*test_accuracy))
      f.close()
      print("----------------------------------")
    
    return FLAGS.model_dir


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_mnist()

  # Instantiate the tf.Estimator.
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                            model_dir=FLAGS.model_dir)

  # Create tf.Estimator input functions for the training and test data.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_data},
      y=train_labels,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.epochs,
      shuffle=True)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)

  # Training loop.
  steps_per_epoch = 60000 // FLAGS.batch_size
  # Clean records
  f=open(FLAGS.record_dir + "/" + FLAGS.record_file, "w+")
  f.write("learning rate: " + str(FLAGS.learning_rate)+ "\n")
  f.write("noise multiplier: " + str(FLAGS.noise_multiplier) + "\n")
  f.write("Clipping norm: " + str(FLAGS.l2_norm_clip) + "\n")
  f.write("Batch size: " + str(float(FLAGS.batch_size)) + "\n")
  f.write("delta: 10e-5 \n")
  f.close()
  for epoch in range(1, FLAGS.epochs + 1):
    f=open(FLAGS.record_dir + "/" + FLAGS.record_file, "a+")
    print("-------------------------------------------")
    f.write("EPOCH: " + str(epoch) +"/" + str(FLAGS.epochs) + "\n")
    print("EPOCH", epoch,"/", FLAGS.epochs)
    print("-------------------------------------------")
    f.close()
    # Train the model for one epoch.
    mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    test_accuracy = eval_results['accuracy']
    f=open(FLAGS.record_dir + "/" + FLAGS.record_file, "a+")
    f.write('Test accuracy after %d epochs is: %.3f \n' % (epoch, 100*test_accuracy))
    print('Test accuracy after %d epochs is: %.3f' % (epoch, 100*test_accuracy))
    f.close()
    print("----------------------------------")
    


if __name__ == '__main__':
  app.run(main)
