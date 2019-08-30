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


import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
# from model import cnn_model_fn, softmax_model_fn, cifar_10_cnn_model_fn
from utils import load_mnist, load_cifar10

if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer
else:
  GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

flags.DEFINE_boolean(
    'dpsgd', False, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 10, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS


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


def get_model_stucture(model_name):
  if( model_name == "mnist"):
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu',
                             input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
  elif(model_name == "cifar10"):
    # https://keras.io/examples/cifar10_cnn/
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                    input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
  else:
    model = None
  return model

def train(dataset, model_name, mode='nn'):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')
  model = get_model_stucture(model_name)
  # Load training and test data.
  train_data, train_labels, test_data, test_labels = dataset
  input(train_labels)
  # Define a sequential Keras model
  
  if FLAGS.dpsgd:
    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
  else:
    optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    # if (model_name=="mnist"):
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # else:
    #   loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)
  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Train model with Keras
  model.fit(train_data, train_labels,
            epochs=FLAGS.epochs,
            validation_data=(test_data, test_labels),
            batch_size=FLAGS.batch_size)

  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
  else:
    print('Trained with vanilla non-private SGD optimizer')

  # serialize model to JSON
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")
  print("Saved model to disk")
  # return model

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







































def main(unused_argv):
  # Use to train the MNIST classifier
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')
  FLAGS.epochs = 35
  FLAGS.model_dir = "../DL_models/mnist_sgd_35"
  FLAGS.record_file = "mnist_sgd_35.txt"
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
    
    # Save file
    f=open(FLAGS.record_dir + "/" + FLAGS.record_file, "a+")
    f.write('Test accuracy after %d epochs is: %.3f \n' % (epoch, 100*test_accuracy))
    f.close()
    print("----------------------------------")
    


if __name__ == '__main__':
  app.run(main)
