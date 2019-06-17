
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# from PIL import Image
from absl import app

from dpsgd_classifier import train as train_model

from utils import load_trained_indices, get_data_indices, load_mnist
import argparse
MODEL_PATH = './model/'
DATA_PATH = './data/'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# UTILS FUNCTIONS

def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')

def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]

def load_data(data_name):
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y, test_x, test_y

def save_data():
    print( '-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')
    MODEL_PATH = './model/'
    DATA_PATH = './data/'
    
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    # x, y, test_x, test_y = load_dataset(args.train_feat, args.train_label, args.test_feat, args.train_label)
    x, y, test_x, test_y = load_mnist()
    if test_x is None:
        print( 'Splitting train/test data with ratio {}/{}'.format(1 - args.test_ratio, args.test_ratio))
        x, test_x, y, test_y = train_test_split(x, y, test_size=args.test_ratio, stratify=y)

    # need to partition target and shadow model data
    assert len(x) > 2 * args.target_data_size

    target_data_indices, shadow_indices = get_data_indices(len(x), target_train_size=args.target_data_size)
    np.savez(MODEL_PATH + 'data_indices.npz', target_data_indices, shadow_indices)

    # target model's data
    print( 'Saving data for target model')
    train_x, train_y = x[target_data_indices], y[target_data_indices]
    size = len(target_data_indices)
    if size < len(test_x):
        test_x = test_x[:size]
        test_y = test_y[:size]
    # save target data
    np.savez(DATA_PATH + 'target_data.npz', train_x, train_y, test_x, test_y)

    # shadow model's data
    target_size = len(target_data_indices)
    shadow_x, shadow_y = x[shadow_indices], y[shadow_indices]
    shadow_indices = np.arange(len(shadow_indices))

    for i in range(args.n_shadow):
        print( 'Saving data for shadow model {}'.format(i))
        shadow_i_indices = np.random.choice(shadow_indices, 2 * target_size, replace=False)
        shadow_i_x, shadow_i_y = shadow_x[shadow_i_indices], shadow_y[shadow_i_indices]
        train_x, train_y = shadow_i_x[:target_size], shadow_i_y[:target_size]
        test_x, test_y = shadow_i_x[target_size:], shadow_i_y[target_size:]
        np.savez(DATA_PATH + 'shadow{}_data.npz'.format(i), train_x, train_y, test_x, test_y)

# MODEL STRUCTURE

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

# TRAINING FUNCTIONS

def train_target_model(dataset, save=True):
    train_x, train_y, test_x, test_y = dataset
    batchSize=100
    
    modelDir = train_model(dataset, "target")
    target_model = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                        model_dir=modelDir)
    # test data for attack model
    attack_x, attack_y = [], []
    # data used in training, label is 1
    for batch in iterate_minibatches(train_x, train_y, batchSize, False):
        pred_input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x={'x': batch[0]},
        y=None, 
        batch_size=batchSize,
        num_epochs=1,
        shuffle=False,
        num_threads=1)

        predict_results = target_model.predict(pred_input_fn_train) 
        attack_x.append( [list(item[1]['probabilities']) for item in list(enumerate(predict_results))])
        attack_y.append(np.ones(batchSize))

    # data used in training, label is 0
    for batch in iterate_minibatches(test_x, test_y, batchSize, False):
        pred_input_fn_test = tf.estimator.inputs.numpy_input_fn(
        x={'x': batch[0]},
        y=None, 
        batch_size=batchSize,
        num_epochs=1,
        shuffle=False,
        num_threads=1)

        predict_results = target_model.predict(pred_input_fn_test) 
        attack_x.append( [list(item[1]['probabilities']) for item in list(enumerate(predict_results))])
        attack_y.append(np.ones(batchSize))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    if save:
        np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)
        # np.savez(MODEL_PATH + 'target_model.npz', *target_model)

    classes = np.concatenate([train_y, test_y])
    return attack_x, attack_y, classes


def train_shadow_models(dataset, save=True):
    # for attack model
    attack_x, attack_y = [], []
    classes = []
    batchSize = 100
    n_shadow = 10
    for i in range(n_shadow):
        print( 'Training shadow model {}'.format(i))
        data = load_data('shadow{}_data.npz'.format(i))
        train_x, train_y, test_x, test_y = data
        modelDir = train_model(dataset, "shadow")
        shadow_model = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                        model_dir=modelDir)
        # Predict 
        attack_i_x, attack_i_y = [], []
        # data used in training, label is 1
        # data used in training, label is 1
        for batch in iterate_minibatches(train_x, train_y, batchSize, False):
            pred_input_fn_train = tf.estimator.inputs.numpy_input_fn(
            x={'x': batch[0]},
            y=None, 
            batch_size=batchSize,
            num_epochs=1,
            shuffle=False,
            num_threads=1)

            predict_results = shadow_model.predict(pred_input_fn_train) 
            attack_i_x.append( [list(item[1]['probabilities']) for item in list(enumerate(predict_results))])
            attack_i_y.append(np.ones(batchSize))

        # data used in training, label is 0
        for batch in iterate_minibatches(test_x, test_y, batchSize, False):
            pred_input_fn_test = tf.estimator.inputs.numpy_input_fn(
            x={'x': batch[0]},
            y=None, 
            batch_size=batchSize,
            num_epochs=1,
            shuffle=False,
            num_threads=1)

            predict_results = shadow_model.predict(pred_input_fn_test) 
            attack_i_x.append( [list(item[1]['probabilities']) for item in list(enumerate(predict_results))])
            attack_i_y.append(np.ones(batchSize))
        attack_x += attack_i_x
        attack_y += attack_i_y
        classes.append(np.concatenate([train_y, test_y]))
    # train data for attack model
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = np.concatenate(classes)
    if save:
        np.savez(MODEL_PATH + 'attack_train_data.npz', attack_x, attack_y)

    return attack_x, attack_y, classes

# def train_attack_model(classes, dataset=None):
#     if dataset is None:
#         dataset = load_attack_data()

#     train_x, train_y, test_x, test_y = dataset

#     train_classes, test_classes = classes
#     train_indices = np.arange(len(train_x))
#     test_indices = np.arange(len(test_x))
#     unique_classes = np.unique(train_classes)

#     true_y = []
#     pred_y = []
#     for c in unique_classes:
#         print( 'Training attack model for class {}...'.format(c))
#         c_train_indices = train_indices[train_classes == c]
#         c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
#         c_test_indices = test_indices[test_classes == c]
#         c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
#         c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
#         modelDir = train_model(c_dataset)
#         target_model = tf.estimator.Estimator(model_fn=cnn_model_fn,
#                                         model_dir=modelDir)
#         true_y.append(c_test_y)
#         pred_y.append(c_pred_y)

#     print( '-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
#     true_y = np.concatenate(true_y)
#     pred_y = np.concatenate(pred_y)
#     print( 'Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)))
#     print( classification_report(true_y, pred_y))

# MAIN PROGRAMMES
# TODO
def attack_experiment(unused_argv):
    # print( '-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    # dataset = load_data('target_data.npz')
    dataset = load_mnist()
    print('-'*10 + "A"*10 + '-'*10)
    print("TRAINING TARGET MODEL")
    print('-'*10 + "A"*10 + '-'*10)
    attack_test_x, attack_test_y, test_classes = train_target_model(dataset)
    print('-'*10 + "B"*10 + '-'*10)
    print("TRAINING SHADOW MODELS")
    print('-'*10 + "B"*10 + '-'*10)
    train_shadow_models(dataset)
    print('-'*10 + "C"*10 + '-'*10)
    print("TRAINING ATTACK MODEL")
    print('-'*10 + "C"*10 + '-'*10)
    # train_attack_model(dataset)
def main(unused_argv):
    dataset = load_mnist()
    # TODO: LOAD model from estimators
    model = train_model(dataset)
    train_data, train_labels, test_data, test_labels = dataset

    # Create tf.Estimator input functions for the training and test data.
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = model.evaluate(input_fn=eval_input_fn)
    print("---------------------------------------------------------")
    print("Current model's accuracy:", 100*eval_results['accuracy'] )
    print("---------------------------------------------------------")

    # attack_x, attack_y = [], []
    
    print('*'*10 + "Output" + '*'*10 )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('train_feat', type=str)
    # parser.add_argument('train_label', type=str)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    # if test not give, train test split configuration
    parser.add_argument('--test_ratio', type=float, default=0.3)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=10)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))   # number of data point used in target model
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=100)
    parser.add_argument('--target_n_hidden', type=int, default=50)
    parser.add_argument('--target_epochs', type=int, default=50)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-6)

    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='softmax')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=50)
    parser.add_argument('--attack_epochs', type=int, default=50)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

    # parse configuration
    args = parser.parse_args()
    print( vars(args))
    if args.save_data:
        save_data()
    else:
        app.run(attack_experiment)
        # app.run(main)
    
