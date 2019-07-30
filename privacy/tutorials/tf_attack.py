
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# from PIL import Image
from absl import app
from sklearn.model_selection import train_test_split

from dpsgd_classifier import train as train_model
from utils import load_trained_indices, get_data_indices, load_mnist, load_dataset
from model import cnn_model_fn, softmax_model_fn, cifar_10_cnn_model_fn

import argparse

from absl import flags
from sklearn.metrics import classification_report, accuracy_score

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mnist' , 'dataset name')

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

def load_attack_classes(class_name):
    with np.load(MODEL_PATH + class_name) as f:
        classes = [f['arr_%d' % i] for i in range(len(f.files))]
    return classes[0] 


def save_data():
    print( '-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')
    MODEL_PATH = './model/'
    DATA_PATH = './data/'
    
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    # Choosing dataset
    if "local" in args.dataset:
        x, y, test_x, test_y = load_dataset(args.train_feat, args.train_label, args.test_feat, args.train_label)
    elif "mnist" in args.dataset:
        x, y, test_x, test_y = load_mnist()
    elif "cifar10" in args.dataset:
        x, y, test_x, test_y = load_cifar_10()
    
    # x, y, test_x, test_y = load_cifar_10()
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


# TRAINING FUNCTIONS

def train_target_model(dataset, save=True):
    train_x, train_y, test_x, test_y = dataset
    batchSize=100
    
    modelDir = train_model(dataset, "target", 'nn')
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
        attack_y.append(np.zeros(batchSize))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = np.concatenate([train_y, test_y])

    if save:
        np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)
        np.savez(MODEL_PATH + 'attack_test_classes.npz', classes)
        # np.savez(MODEL_PATH + 'target_model.npz', *target_model)


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
        modelDir = train_model(dataset, "shadow", 'nn')
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
            attack_i_y.append(np.zeros(batchSize))
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
        np.savez(MODEL_PATH + 'attack_train_classes.npz', classes)

    return attack_x, attack_y, classes

def train_attack_model(classes=None, dataset=None):
    if dataset is None:
        dataset = load_attack_data()
    if classes is None:
        train_classes = load_attack_classes('attack_train_classes.npz')

        test_classes = load_attack_classes('attack_test_classes.npz')


    else:
        train_classes, test_classes = classes
    train_x, train_y, test_x, test_y = dataset

    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)

    true_y = []
    pred_y = []
    for c in unique_classes:
        print( 'Training attack model for class {}...'.format(c))
        
        c_train_indices = train_indices[train_classes == c]

        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]

        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        f=open("./attack_data/c_train_x_"+ str(c) + ".txt", "w+")
        for idx in range(len(c_train_x)):
            f.write(str(idx) + " - " +str(c_train_x[idx])+"\n")
        f.close()
        f=open("./attack_data/c_train_y"+ str(c) + ".txt", "w+")
        for idx in range(len(c_train_y)):
            f.write(str(idx) + " - " +str(c_train_y[idx])+"\n")
        f.close()
        f=open("./attack_data/c_test_x"+ str(c) + ".txt", "w+")
        for idx in range(len(c_test_x)):
            f.write(str(idx) + " - " +str(c_test_x[idx])+"\n")
        f.close()
        f=open("./attack_data/c_test_y"+ str(c) + ".txt", "w+")
        for idx in range(len(c_test_y)):
            f.write(str(idx) + " - " +str(c_test_y[idx])+"\n")
        f.close()
        #################
        # print("A"*20)

        # zero = [item for item in c_test_y if item==0]
        # one = [item for item in c_test_y if item==1]
       
        # print("-"*20)
        # print(len(c_test_y))

        # print("-"*20)
        # # print(one)
        # print("-"*20)
        # print(len(one))
        # print("-"*20)
        # # print(zero)
        # print("-"*20)
        # print(len(zero))
        # print("-"*20)
        # print(len(zero)/len(one))
        # print("A"*20)
        # input()
        # ##################


        # arr = []
        # for i in range(0,len(c_test_y)):
        #     if(c_test_y[i]==0):
        #        arr.append( np.array([i,[1.0,0.0]]))
        #     elif (c_test_y[i]==1):
        #         arr.append( np.array([i,[0.0,1.0]]))
        # c_test_y = np.array(arr)
        # print(c_test_y.shape)

        # arr = []
        # for i in range(0,len(c_train_y)):
        #     if(c_train_y[i]==0):
        #         arr.append( np.array([i,[1.0,0.0]]))
        #     elif (c_train_y[i]==1):
        #         arr.append( np.array([i,[0.0,1.0]]))
        # c_train_y = np.array(arr)
        # print(c_train_y.shape)


        modelDir = train_model(c_dataset, "class_" + str(c) + "_attack", 'softmax')
        # Train attack model with class c
        attack_model = tf.estimator.Estimator(model_fn=softmax_model_fn,
                                        model_dir=modelDir)


        # EVALUATION
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': c_test_x},
        y=c_test_y,
        num_epochs=1,
        shuffle=False)
        eval_results = attack_model.evaluate(input_fn=eval_input_fn)
        test_accuracy = eval_results['accuracy']
        
        
        # PREDICTION
        pred_input_fn_test = tf.estimator.inputs.numpy_input_fn(
            x={'x': c_test_x},
            y=None, 
            batch_size=100,
            num_epochs=1,
            shuffle=False,
            num_threads=1)
        pred = attack_model.predict(pred_input_fn_test) 
        f=open("./evaluate/evaluate_class_"+ str(c)+ ".txt", "w+")
        
        for index, item in enumerate(pred):
            # print(index, type(index))
            # print(item['class_ids'], type(item['class_ids']))
            # print(item['probabilities'], type(item['class_ids']))
            if(item['class_ids'][0] != c_test_y[index]):
                f.write(str(index) + " " + str(item['class_ids']) + " / " + str(c_test_y[index]) + " " + str(item['probabilities']) + " WRONG PREDICTION \n")
            else:
                f.write(str(index) + " " + str(item['class_ids']) + " / " + str(c_test_y[index]) + " " + str(item['probabilities']) + "\n")
        f.close()  

        print('Test accuracy for class %d is: %.3f' % (c, 100*test_accuracy))
        f=open("./record_data/"  + "attack_record_class_" + str(c), "a+")
        f.write('Test accuracy for class %d is: %.3f \n' % (c, 100*test_accuracy))
        f.close()

    print( '-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    


    
    # true_y = np.concatenate(true_y)
    # pred_y = np.concatenate(pred_y)
    # print( 'Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)))
    # print( classification_report(true_y, pred_y))

# MAIN PROGRAMMES
# TODO

def load_cifar_10():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.cifar10.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32)
  test_data = np.array(test_data, dtype=np.float32)

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)
  print(train_data.shape)
  print(test_data.shape)
  print(train_labels.shape)
  print(test_labels.shape)
  print(train_data[0])
  print(train_labels[0])
#   input()
  return train_data, train_labels, test_data, test_labels
def attack_experiment(unused_argv):
    print( '-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    dataset = load_data('target_data.npz')
    # dataset = load_mnist()
    # dataset = load_cifar_10()
    print('-'*10 + "A"*10 + '-'*10)
    print("TRAINING TARGET MODEL")
    print('-'*10 + "A"*10 + '-'*10)
    attack_test_x, attack_test_y, test_classes = train_target_model(dataset)
    print('-'*10 + "B"*10 + '-'*10)
    print("TRAINING SHADOW MODELS")
    print('-'*10 + "B"*10 + '-'*10)
    attack_train_x, attack_train_y, train_classes = train_shadow_models(dataset)
    print('-'*10 + "C"*10 + '-'*10)
    print("TRAINING ATTACK MODEL")
    print('-'*10 + "C"*10 + '-'*10)
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    classes = (train_classes, test_classes)
    train_attack_model(classes , dataset)
    # input()
    # train_attack_model()

# def main(unused_argv):
    
    
        

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
    parser.add_argument('--n_shadow', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="mnist")
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
    if args.save_data:

        save_data() # Check dataset which is used by this function before running!!!
    else:
        FLAGS.dataset = args.dataset
        app.run(attack_experiment)
        # app.run(main)
    
    
