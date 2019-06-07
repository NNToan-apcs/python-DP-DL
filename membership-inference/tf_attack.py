
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys

from PIL import Image
from absl import app
from tensorflow.contrib import predictor
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

def cnn_model_fn(features, labels, mode):
  """Model function for a CNN."""
  
  print("----------------------MODE----------------------------------------")
  print("Mode = ", mode)
  print("------------------------------------------------------------------")
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

  if mode == tf.estimator.ModeKeys.PREDICT:
    predicted_classes = tf.argmax(logits, 1)
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': 100*tf.nn.softmax(logits),
        # 'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  
  

  # Add evaluation metrics (for EVAL mode).
  if mode == tf.estimator.ModeKeys.EVAL:
    # Calculate loss as a vector (to support microbatches in DP-SGD).
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    # Define mean of loss across minibatch (for reporting through tf.Estimator).
    scalar_loss = tf.reduce_mean(vector_loss)
    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(input=logits, axis=1))
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)
  


# def main(unused_argv):
#     # Load model: mnist_sgd_60 mnist_dpsgd_60 model_test
#     if len(sys.argv) >= 2:
#         model_dir = sys.argv[1]
#         export_dir="D:\DL_models\\" + model_dir
#     else:
#         export_dir="D:\DL_models\mnist_sgd_60"
#     predict_datas = []
#     if len(sys.argv) >= 3:
#         for i in range(2, len(sys.argv)):
#             input_image = sys.argv[i]
#             img = Image.open('C:/Users/nntoa/Desktop/mnist_test_data/' + input_image +'.png').convert("L")
#             img = img.resize((28,28))
#             im2arr = np.array(img)
#             im2arr = im2arr.reshape(1,28,28,1)
#             im2arr = im2arr.astype('float32')
#             # Normalize
#             im2arr /= 255
#             predict_datas.append(im2arr)
#     else:
#         img = Image.open('C:/Users/nntoa/Desktop/mnist_test_data/seven.png').convert("L")
#         img = img.resize((28,28))
#         img.show()
#         im2arr = np.array(img)
#         im2arr = im2arr.reshape(1,28,28,1)
#         im2arr = im2arr.astype('float32')
#         # Normalize
#         im2arr /= 255
#         predict_datas.append(im2arr)
    
#     # test
    
#     mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
#                                             model_dir=export_dir)
#     # Load training and test data.
#     train_data, train_labels, test_data, test_labels = load_mnist()

#     # Create tf.Estimator input functions for the training and test data.
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={'x': test_data},
#         y=test_labels,
#         num_epochs=1,
#         shuffle=False)

#     eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#     print("---------------------------------------------------------")
#     print("Current model's accuracy:", 100*eval_results['accuracy'] )
#     print("---------------------------------------------------------")

#     # Load own Image
    
    
#     # img.show()
    
#     # Predict Image
#     pred_input_fn = tf.estimator.inputs.numpy_input_fn(
#         # x={'x': np.array([test_data[0:10]])},
#         x={'x': np.array(predict_datas)},
#         y=None, 
#         batch_size=1,
#         num_epochs=1,
#         shuffle=False,
#         num_threads=1)

#     predict_results = mnist_classifier.predict(pred_input_fn) 

#     for idx, prediction in enumerate(predict_results):
#         if( idx == 0):
#             print("id - labels - probabilities")
#         # Get the indices of maximum element in numpy array
#         label = prediction['class_ids'][0]
        
#         print(idx, "-",  prediction['class_ids'], "-", prediction['probabilities'])

def main(unused_argv):

if __name__ == '__main__':
    app.run(main)
