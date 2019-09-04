'''
Created on 15 Nov 2017

@author: ahmed.salem
'''

import sys

sys.dont_write_bytecode = True


from classifier import train_model, iterate_minibatches, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import theano.tensor as T
import lasagne
import theano
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse
from sklearn.datasets import fetch_20newsgroups

import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

import imp
from sklearn.datasets import fetch_lfw_people

from sklearn.datasets import fetch_mldata

np.random.seed(21312)
MODEL_PATH = './model/'
DATA_PATH = './data/'
train = 'false'



def train_target_model(dataset, name,epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, model='nn', save=True):

    print("deeplearning.py -- train_target_model")
    
    returnDouble=True
    train_x, train_y, test_x, test_y = dataset

    print("begin traning:")
    output_layer = train_model(dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, l2_ratio=l2_ratio)
    # test data for attack model
    attack_x, attack_y = [], []
    isCorrect = []
    if model=='cnn' or model=='Droppcnn'or model=='Droppcnn2' :
        input_var = T.tensor4('x')
    elif model=='cnn2':
        input_var = T.tensor4('x')
    else:
        input_var = T.matrix('x')

    prob = lasagne.layers.get_output(output_layer, input_var, deterministic=True)

    prob_fn = theano.function([input_var], prob)
    # data used in training, label is 1
    for batch in iterate_minibatches(train_x, train_y, batch_size, False):
        attack_x.append(prob_fn(batch[0]))
        predicted = np.argmax(prob_fn(batch[0]), axis=1)
        isCorrect.append((batch[1] == predicted).astype('float32'))
        attack_y.append(np.zeros(len(batch[0])))
    # data not used in training, label is 0
    for batch in iterate_minibatches(test_x, test_y, batch_size, False):
        attack_x.append(prob_fn(batch[0]))
        predicted = np.argmax(prob_fn(batch[0]), axis=1)
        isCorrect.append((batch[1] == predicted).astype('float32'))

        attack_y.append(np.ones(len(batch[0])))
        
    #print len(attack_y)
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    isCorrect = np.concatenate(isCorrect)
    #print('total length  ' + str(sum(attack_y)))
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    if save:
        np.savez(MODEL_PATH + 'attack_test_data'+str(name)+'.npz', attack_x)
        np.savez(MODEL_PATH + 'attack_test_label'+str(name)+'.npz', attack_y)
        np.savez(MODEL_PATH + 'target_model'+str(name)+'.npz', *lasagne.layers.get_all_param_values(output_layer))

    classes = np.concatenate([train_y, test_y])
    if(returnDouble):
        return attack_x, attack_y, output_layer,prob_fn
    else:
        return attack_x, attack_y, output_layer
def load_data(data_name):
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y, test_x, test_y
