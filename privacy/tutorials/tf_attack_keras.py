
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# from PIL import Image
from absl import app
from sklearn.model_selection import train_test_split

from dpsgd_classifier_keras import train as train_model
from utils import load_trained_indices, get_data_indices, load_mnist_keras, load_dataset, load_cifar10

import argparse

from absl import flags
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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
        print("USING LOCAL DATASET")
        x, y, test_x, test_y = load_dataset(args.train_feat, args.train_label, args.test_feat, args.train_label)
    elif "mnist" in args.dataset:
        print("USING MNIST DATASET")
        x, y, test_x, test_y = load_mnist_keras()
    elif "cifar10" in args.dataset:
        print("USING CIFAR10 DATASET")
        x, y, test_x, test_y = load_cifar10()

    
    # x, y, test_x, test_y = load_cifar10()
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
    print("TRAINING TARGET MODEL WITH " + FLAGS.dataset + "DATASET")
    target_model = train_model(dataset, "target", FLAGS.dataset)

    # test data for attack model
    attack_x, attack_y = [], []
    # data used in training, label is 1
    for batch in iterate_minibatches(train_x, train_y, batchSize, False):
        predict_results = target_model.predict(batch[0])
        attack_x.append(predict_results)
        attack_y.append(np.ones(batchSize))


    # data used in training, label is 0
    for batch in iterate_minibatches(test_x, test_y, batchSize, False):
        predict_results = target_model.predict(batch[0]) 
        attack_x.append(predict_results)
        attack_y.append(np.zeros(batchSize))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = np.concatenate([train_y, test_y])

    if save:
        np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)
        np.savez(MODEL_PATH + 'attack_test_classes.npz', classes)

    return attack_x, attack_y, classes


def train_shadow_models(save=True):
    # for attack model
    attack_x, attack_y = [], []
    classes = []
    batchSize = 100
    n_shadow = 10
    for i in range(n_shadow):
        print( 'Training shadow model {} for dataset {}'.format(i,FLAGS.dataset))
        dataset = load_data('shadow{}_data.npz'.format(i))
        train_x, train_y, test_x, test_y = dataset
        shadow_model = train_model(dataset, "shadow", FLAGS.dataset)
        # Predict 
        attack_i_x, attack_i_y = [], []
        # data used in training, label is 1
        
        for batch in iterate_minibatches(train_x, train_y, batchSize, False):
            predict_results = shadow_model.predict(batch[0]) 
            attack_x.append(predict_results)
            attack_i_y.append(np.ones(batchSize))

        # data used in training, label is 0
        for batch in iterate_minibatches(test_x, test_y, batchSize, False):
            predict_results = shadow_model.predict(batch[0]) 
            attack_x.append(predict_results)
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
        print("AAAA")
        train_classes = load_attack_classes('attack_train_classes.npz')
        # input(np.argmax(train_classes))
        test_classes = load_attack_classes('attack_test_classes.npz')
        # input(np.argmax(test_classes))
    else:
        train_classes, test_classes = classes
    train_x, train_y, test_x, test_y = dataset
    train_classes = [np.argmax(item) for item in train_classes]
    test_classes =[np.argmax(item) for item in test_classes]
    
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)
    
    true_y = []
    pred_y = []
    for c in unique_classes:
        # input(c)
        print( 'Training attack model for class {}...'.format(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        # input(c_train_indices.shape)
        
        # input(train_indices.shape)
        # input(test_indices.shape)
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]

        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        # input(c_train_x.shape)
        # input(c_train_y.shape)
        # input(c_test_x.shape)
        # input(c_test_y.shape)
        # f=open("./attack_data/c_train_x_"+ str(c) + ".txt", "w+")
        # for idx in range(len(c_train_x)):
        #     f.write(str(idx) + " - " +str(c_train_x[idx])+"\n")
        # f.close()
        # f=open("./attack_data/c_train_y"+ str(c) + ".txt", "w+")
        # for idx in range(len(c_train_y)):
        #     f.write(str(idx) + " - " +str(c_train_y[idx])+"\n")
        # f.close()
        # f=open("./attack_data/c_test_x"+ str(c) + ".txt", "w+")
        # for idx in range(len(c_test_x)):
        #     f.write(str(idx) + " - " +str(c_test_x[idx])+"\n")
        # f.close()
        # f=open("./attack_data/c_test_y"+ str(c) + ".txt", "w+")
        # for idx in range(len(c_test_y)):
        #     f.write(str(idx) + " - " +str(c_test_y[idx])+"\n")
        # f.close()


        # Train attack model with class c
        
        attack_model = train_model(c_dataset, "class_" + str(c) + "_attack", 'softmax')
        pred = attack_model.predict(c_test_x) 
        f=open("./evaluate/evaluate_class_"+ str(c)+ ".txt", "w+")
        pred_y = [np.argmax(item) for item in pred]
        for index in range(0,len(pred)):
            # print(index, type(index))
            # print(np.argmax(pred[index]),type(pred[index]))
            # print(c_test_y[index],type(c_test_y[index]))
            # input()
            if(np.argmax(pred[index]) != c_test_y[index]):
                f.write(str(index) + " " + str(np.argmax(pred[index])) + " / " + str(c_test_y[index]) + " " + str(pred[index]) + " WRONG PREDICTION \n")
            else:
                f.write(str(index) + " " + str(np.argmax(pred[index])) + " / " + str(c_test_y[index]) + " " + str(pred[index]) + "\n")
        f.close()  
        
        
        # input()
    print( '-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')

def attack_experiment(unused_argv):
    print( '-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    dataset = load_data('target_data.npz')
    
    print('-'*10 + "A"*10 + '-'*10)
    print("TRAINING TARGET MODEL")
    print('-'*10 + "A"*10 + '-'*10)
    attack_test_x, attack_test_y, test_classes = train_target_model(dataset)
    print('-'*10 + "B"*10 + '-'*10)
    print("TRAINING SHADOW MODELS")
    print('-'*10 + "B"*10 + '-'*10)
    attack_train_x, attack_train_y, train_classes = train_shadow_models()
    print('-'*10 + "C"*10 + '-'*10)
    print("TRAINING ATTACK MODEL")
    print('-'*10 + "C"*10 + '-'*10)
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    classes = (train_classes, test_classes)
    train_attack_model(classes , dataset)
    # input()
    # train_attack_model()


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
    parser.add_argument('--dataset', type=str, default="cifar10")
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
    
    
