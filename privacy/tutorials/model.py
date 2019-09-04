import tensorflow as tf

from absl import flags
from distutils.version import LooseVersion



if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer
else:
  GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', False, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_boolean(
    'moment_accountant', True, 'If True, compute eps using moment_accountant. If False, '
    'compute eps using strong composition.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_float('delta', 1e-5, 'target delta')

flags.DEFINE_integer('epochs', 1, 'Number of epochs')
flags.DEFINE_integer('soft_max_epochs', 1, 'Number of epochs')

# flags.DEFINE_integer('epochs', 15, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 256, 'Number of microbatches '
    '(must evenly divide batch_size)')

def cifar_10_cnn_model_fn(features, labels, mode):
    with tf.device('/gpu:0'):

      #       # 1st Convolutional Layer
      # conv1 = tf.layers.conv2d(
      #     inputs=images, filters=64, kernel_size=[5, 5], padding='same',
      #     activation=tf.nn.relu, name='conv1')
      # pool1 = tf.layers.max_pooling2d(
      #     inputs=conv1, pool_size=[3, 3], strides=2, name='pool1')
      # norm1 = tf.nn.lrn(
      #     pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

      # # 2nd Convolutional Layer
      # conv2 = tf.layers.conv2d(
      #     inputs=norm1, filters=64, kernel_size=[5, 5], padding='same',
      #     activation=tf.nn.relu, name='conv2')
      # norm2 = tf.nn.lrn(
      #     conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
      # pool2 = tf.layers.max_pooling2d(
      #     inputs=norm2, pool_size=[3, 3], strides=2, name='pool2')

      # # Flatten Layer
      # shape = pool2.get_shape()
      # pool2_ = tf.reshape(pool2, [-1, shape[1]*shape[2]*shape[3]])

      # # 1st Fully Connected Layer
      # dense1 = tf.layers.dense(
      #     inputs=pool2_, units=384, activation=tf.nn.relu, name='dense1')

      # # 2nd Fully Connected Layer
      # dense2 = tf.layers.dense(
      #     inputs=dense1, units=192, activation=tf.nn.relu, name='dense2')

      # # 3rd Fully Connected Layer (Logits)
      # logits = tf.layers.dense(
      #     inputs=dense2, units=NUM_CLASSES, activation=tf.nn.relu, name='logits')
  
      # 32 output- Conv2d
      # input(features['x'][1:])
      # input_layer = tf.reshape(features['x'],[])
      input_layer = features['x']
      input(input_layer.shape)
      y = tf.keras.layers.Conv2D(32, (3,3),
                                padding='same',
                                activation='relu').apply(input_layer)
      input(y.shape)
      y = tf.keras.layers.Conv2D(32, (3,3),
                                activation='relu').apply(y)
      input(y.shape)
      y = tf.keras.layers.MaxPool2D(pool_size=(2, 2)).apply(y)
      input(y.shape)
      # 64 output
      y = tf.keras.layers.Conv2D(64, (3,3),
                                padding='same',
                                activation='relu').apply(features['x'])
      input(y.shape)                                
      y = tf.keras.layers.Conv2D(64, (3,3),
                                activation='relu').apply(y)
      input(y.shape)
      y = tf.keras.layers.MaxPool2D(pool_size=(2, 2)).apply(y)
      input(y.shape)
      # Flatten
      y = tf.keras.layers.Flatten().apply(y)
      input(y.shape)
      # 512 relu output
      
      y = tf.keras.layers.Dense(512, activation='relu').apply(y)
      input(y.shape)
      y = tf.keras.layers.Dense(32, activation='relu').apply(y)
      input(y.shape)
      # softmax
      logits = tf.keras.layers.Dense(10).apply(y)
      if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_classes = tf.argmax(logits, 1)
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': 100*tf.nn.softmax(logits),
            # 'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
      # Calculate loss as a vector (to support microbatches in DP-SGD).
      
      print(labels.shape)
      input(logits.shape)
      
      input("DEBUG")
      # vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      #     labels=tf.argmax(tf.cast(labels, dtype=tf.int32),1), logits=logits)
      vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
      # vector_loss = tf.nn.softmax_cross_entropy_with_logits(
      #     labels=tf.argmax(labels,1), logits=logits)
      
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
          # optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
          optimizer = GradientDescentOptimizer(learning_rate=0.001)
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

def cnn_model_fn(features, labels, mode):
  """Model function for a CNN."""
  with tf.device('/gpu:0'):
    # Define CNN architecture using tf.keras.layers.
    
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    # input(features['x'].shape)
    # input(input_layer)
    # input(input_layer.shape)
    y = tf.keras.layers.Conv2D(16, 8,
                              strides=2,
                              padding='same',
                              activation='relu').apply(input_layer)
    # input(y.shape)
    y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    # input(y.shape)
    y = tf.keras.layers.Conv2D(32, 4,
                              strides=2,
                              padding='valid',
                              activation='relu').apply(y)
    # input(y.shape)
    y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    # input(y.shape)
    y = tf.keras.layers.Flatten().apply(y)
    # input(y.shape)
    y = tf.keras.layers.Dense(32, activation='relu').apply(y)
    # input(y.shape)
    logits = tf.keras.layers.Dense(10).apply(y)
    if mode == tf.estimator.ModeKeys.PREDICT:
      predicted_classes = tf.argmax(logits, 1)
      predictions = {
          'class_ids': predicted_classes[:, tf.newaxis],
          'probabilities': 100*tf.nn.softmax(logits),
          # 'logits': logits,
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # Calculate loss as a vector (to support microbatches in DP-SGD).
    
    # print(labels.shape)
    # input(logits.shape)
      
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
  






def softmax_model_fn(features, labels, mode):
  """Model function for a Softmax regression."""
  with tf.device('/gpu:0'):
    # y = tf.keras.layers.Dense(64, activation='relu').apply(features['x'])
    y = tf.keras.layers.Dense(10, activation='relu').apply(features['x'])
    y = tf.keras.layers.Flatten().apply(y)
    logits = tf.keras.layers.Dense(2).apply(y)

    # logits = tf.keras.layers.Dense(2, activation='softmax').apply(features['x'])

    # logits = tf.keras.layers.Dense(2).apply(features['x'])
    # input_layer = input_layer = tf.reshape(features['x'], [-1, 10, 1, 1])
    # y = tf.keras.layers.Conv2D(16, 8,
    #                           strides=1,
    #                           padding='same',
    #                           activation='relu').apply(input_layer)
    # y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    # y = tf.keras.layers.Conv2D(32, 4,
    #                           strides=2,
    #                           padding='valid',
    #                           activation='relu').apply(y)
    # y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
    # y = tf.keras.layers.Flatten().apply(y)
    # y = tf.keras.layers.Dense(32, activation='relu').apply(y)
    # logits = tf.keras.layers.Dense(2).apply(y)

    if mode == tf.estimator.ModeKeys.PREDICT:
      
      predicted_classes = tf.argmax(logits, 1)
      predictions = {
          'class_ids': predicted_classes[:, tf.newaxis],
          'probabilities': 100*tf.nn.softmax(logits),
      }
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    vector_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.one_hot(labels, depth=2), logits=logits)
    scalar_loss = tf.reduce_mean(vector_loss)

    # vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=labels, logits=logits)
    # # Define mean of loss across minibatch (for reporting through tf.Estimator).
    # scalar_loss = tf.reduce_mean(vector_loss)

    # Configure the training op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:
      # optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
      optimizer = GradientDescentOptimizer(learning_rate=0.0001)
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
    









# import re
# # import tensorflow as tf


# tf.logging.set_verbosity(tf.logging.INFO)


# IMAGE_HEIGHT = 32
# IMAGE_WIDTH = 32
# IMAGE_DEPTH = 3
# NUM_CLASSES = 10


# def parse_record(serialized_example):
#   """Parsing CIFAR-10 dataset that is saved in TFRecord format."""
#   features = tf.parse_single_example(
#     serialized_example,
#     features={
#       'image': tf.FixedLenFeature([], tf.string),
#       'label': tf.FixedLenFeature([], tf.int64),
#     })

#   image = tf.decode_raw(features['image'], tf.uint8)
#   image.set_shape([IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH])
#   image = tf.reshape(image, [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
#   image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

#   label = tf.cast(features['label'], tf.int32)
#   label = tf.one_hot(label, NUM_CLASSES)

#   return image, label


# def preprocess_image(image, is_training=False):
#   """Preprocess a single image of layout [height, width, depth]."""
#   if is_training:
#     # Resize the image to add four extra pixels on each side.
#     image = tf.image.resize_image_with_crop_or_pad(
#         image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

#     # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
#     image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

#     # Randomly flip the image horizontally.
#     image = tf.image.random_flip_left_right(image)

#   # Subtract off the mean and divide by the variance of the pixels.
#   image = tf.image.per_image_standardization(image)
#   return image


# def generate_input_fn(filenames, mode=tf.estimator.ModeKeys.EVAL, batch_size=1):
#   """Input function for Estimator API."""
#   def _input_fn():
#     dataset = tf.data.TFRecordDataset(filenames=filenames)

#     is_training = (mode == tf.estimator.ModeKeys.TRAIN)
#     if is_training:
#       buffer_size = batch_size * 2 + 1
#       dataset = dataset.shuffle(buffer_size=buffer_size)

#     dataset = dataset.map(parse_record)
#     dataset = dataset.map(
#       lambda image, label: (preprocess_image(image, is_training), label))

#     dataset = dataset.repeat()
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(2 * batch_size)

#     images, labels = dataset.make_one_shot_iterator().get_next()

#     features = {'images': images}
#     return features, labels

#   return _input_fn


# def get_feature_columns():
#   """Define feature columns."""
#   feature_columns = {
#     'images': tf.feature_column.numeric_column(
#         'images', (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)),
#   }
#   return feature_columns


# def serving_input_fn():
#   """Define serving function."""
#   receiver_tensor = {
#       'images': tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
#   }
#   features = {
#       'images': tf.map_fn(preprocess_image, receiver_tensor['images'])
#   }
#   return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


# def inference(images):
#   # 1st Convolutional Layer
#   conv1 = tf.layers.conv2d(
#       inputs=images, filters=64, kernel_size=[5, 5], padding='same',
#       activation=tf.nn.relu, name='conv1')
#   pool1 = tf.layers.max_pooling2d(
#       inputs=conv1, pool_size=[3, 3], strides=2, name='pool1')
#   norm1 = tf.nn.lrn(
#       pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

#   # 2nd Convolutional Layer
#   conv2 = tf.layers.conv2d(
#       inputs=norm1, filters=64, kernel_size=[5, 5], padding='same',
#       activation=tf.nn.relu, name='conv2')
#   norm2 = tf.nn.lrn(
#       conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
#   pool2 = tf.layers.max_pooling2d(
#       inputs=norm2, pool_size=[3, 3], strides=2, name='pool2')

#   # Flatten Layer
#   shape = pool2.get_shape()
#   pool2_ = tf.reshape(pool2, [-1, shape[1]*shape[2]*shape[3]])

#   # 1st Fully Connected Layer
#   dense1 = tf.layers.dense(
#       inputs=pool2_, units=384, activation=tf.nn.relu, name='dense1')

#   # 2nd Fully Connected Layer
#   dense2 = tf.layers.dense(
#       inputs=dense1, units=192, activation=tf.nn.relu, name='dense2')

#   # 3rd Fully Connected Layer (Logits)
#   logits = tf.layers.dense(
#       inputs=dense2, units=NUM_CLASSES, activation=tf.nn.relu, name='logits')

#   return logits


# def cifar_10_cnn_model_fn(features, labels, mode, params):
#   # Create the input layers from the features
#   feature_columns = list(get_feature_columns().values())

#   images = tf.feature_column.input_layer(
#     features=features, feature_columns=feature_columns)

#   images = tf.reshape(
#     images, shape=(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

#   # Calculate logits through CNN
#   logits = inference(images)

#   if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
#     predicted_indices = tf.argmax(input=logits, axis=1)
#     probabilities = tf.nn.softmax(logits, name='softmax_tensor')

#   if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
#     global_step = tf.train.get_or_create_global_step()
#     label_indices = tf.argmax(input=labels, axis=1)
#     loss = tf.losses.softmax_cross_entropy(
#         onehot_labels=labels, logits=logits)
#     tf.summary.scalar('cross_entropy', loss)

#   if mode == tf.estimator.ModeKeys.PREDICT:
#     predictions = {
#         'classes': predicted_indices,
#         'probabilities': probabilities
#     }
#     export_outputs = {
#         'predictions': tf.estimator.export.PredictOutput(predictions)
#     }
#     return tf.estimator.EstimatorSpec(
#         mode, predictions=predictions, export_outputs=export_outputs)

#   if mode == tf.estimator.ModeKeys.TRAIN:
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#     train_op = optimizer.minimize(loss, global_step=global_step)
#     return tf.estimator.EstimatorSpec(
#         mode, loss=loss, train_op=train_op)

#   if mode == tf.estimator.ModeKeys.EVAL:
#     eval_metric_ops = {
#         'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
#     }
#     return tf.estimator.EstimatorSpec(
#         mode, loss=loss, eval_metric_ops=eval_metric_ops)