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
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_float('delta', 1e-5, 'target delta')

flags.DEFINE_integer('epochs', 30, 'Number of epochs')
flags.DEFINE_integer('soft_max_epochs', 20, 'Number of epochs')

# flags.DEFINE_integer('epochs', 15, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 256, 'Number of microbatches '
    '(must evenly divide batch_size)')

def cifar_100_cnn_model_fn(features, labels, mode):
    with tf.device('/gpu:0'):
      # 128 output- Conv2d
      y = tf.keras.layers.Conv2D(128, (3,3),
                                padding='same',
                                activation='relu').apply(features['x'])
      y = tf.keras.layers.Conv2D(128, (3,3),
                                activation='relu').apply(y)
      y = tf.keras.layers.MaxPool2D(2, (2,2)).apply(y)
      # 256 output
      y = tf.keras.layers.Conv2D(256, (3,3),
                                padding='same',
                                activation='relu').apply(features['x'])
      y = tf.keras.layers.Conv2D(256, (3,3),
                                activation='relu').apply(y)
      y = tf.keras.layers.MaxPool2D(2, (2,2)).apply(y)
      # 512 output
      y = tf.keras.layers.Conv2D(512, (3,3),
                                padding='same',
                                activation='relu').apply(features['x'])
      y = tf.keras.layers.Conv2D(512, (3,3),
                                activation='relu').apply(y)
      y = tf.keras.layers.MaxPool2D(2, (2,2)).apply(y)
      # Flatten
      y = tf.keras.layers.Flatten().apply(y)
      # 1024 relu output
      y = tf.keras.layers.Dense(1024, activation='relu').apply(y)
      # softmax
      logits = tf.keras.layers.Dense(100, activation='softmax').apply(y)
      if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_classes = tf.argmax(logits, 1)
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': 100*tf.nn.softmax(logits),
            # 'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
      # Calculate loss as a vector (to support microbatches in DP-SGD).
      
      vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.argmax(tf.cast(labels, dtype=tf.int32),1), logits=logits)
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
                    labels=tf.argmax(tf.cast(labels, dtype=tf.int32),1),
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
  






def softmax_model_fn(features, labels, mode):
  """Model function for a Softmax regression."""
  with tf.device('/gpu:0'):
    y = tf.keras.layers.Dense(64, activation='relu').apply(features['x'])
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
    

