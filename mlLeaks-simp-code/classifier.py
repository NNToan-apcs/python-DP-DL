'''
Created on 15 Nov 2017

@author: ahmed.salem
'''

import sys

sys.dont_write_bytecode = True


from sklearn.metrics import classification_report, accuracy_score
import theano.tensor as T
import numpy as np
import lasagne
import theano
import argparse
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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




def get_cnn_model(n_in, n_hidden, n_out):
    net = dict()
    #print(n_in)
    net['input'] = lasagne.layers.InputLayer(shape=(None, 3, 32, 32))
    
    net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], num_filters=32, filter_size=(5, 5),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    
    net['maxPool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
     
     
    net['conv2'] = lasagne.layers.Conv2DLayer(
            net['maxPool1'], num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    net['maxPool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))
    
    
    
    net['fc'] = lasagne.layers.DenseLayer(
        net['maxPool2'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    
    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def get_cnn2_model(n_in, n_hidden, n_out):
    net = dict()
    face = False
    if (face):
        net['input'] = lasagne.layers.InputLayer(shape=(None, 1, 62, 47))
    else:
        net['input'] = lasagne.layers.InputLayer(shape=(None, 1, 28, 28))
    
    net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], num_filters=32, filter_size=(5, 5),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    
    net['maxPool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
     
     
    net['conv2'] = lasagne.layers.Conv2DLayer(
            net['maxPool1'], num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    net['maxPool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))
    
    
    
    net['fc'] = lasagne.layers.DenseLayer(
        net['maxPool2'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    
    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net


def get_nn_model(n_in, n_hidden, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, n_in))
    net['fc'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def get_Dropnn_model(n_in, n_hidden, n_out,drop1,drop2):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, n_in))
    
    
    net['inputDropOut'] = lasagne.layers.DropoutLayer(net['input'], p=drop1)



    net['fc'] = lasagne.layers.DenseLayer(
        net['inputDropOut'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    
    net['fcDropOut'] = lasagne.layers.DropoutLayer(net['fc'], p=drop2)
    
    
    
    net['output'] = lasagne.layers.DenseLayer(
        net['fcDropOut'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net
def get_Droppcnn_model(n_in, n_hidden, n_out,drop1,drop2,drop3):
    net = dict()
    net['input'] = lasagne.layers.InputLayer(shape=(None, 3, 32, 32))
    
    net['inputDropOut'] = lasagne.layers.DropoutLayer(net['input'], p=drop1)


    net['conv1'] = lasagne.layers.Conv2DLayer(net['inputDropOut'], num_filters=32, filter_size=(5, 5),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    
    
    net['maxPool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
     
    net['conv2'] = lasagne.layers.Conv2DLayer(
            net['maxPool1'], num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    
    
    net['maxPool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))
    
    
    #net['maxPool2DropOut'] = lasagne.layers.DropoutLayer(net['maxPool2'], p=drop1)

    
    
    net['fc'] = lasagne.layers.DenseLayer(
        net['maxPool2'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    
    net['fcDropOut'] = lasagne.layers.DropoutLayer(net['fc'], p=drop2)

    
    net['output'] = lasagne.layers.DenseLayer(
        net['fcDropOut'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def get_Droppcnn2_model(n_in, n_hidden, n_out,drop1,drop2,drop3):
    net = dict()
    face = False
    if (face):
        net['input'] = lasagne.layers.InputLayer(shape=(None, 1, 62, 47))
    else:
        net['input'] = lasagne.layers.InputLayer(shape=(None, 1, 28, 28))
        
    net['inputDropOut'] = lasagne.layers.DropoutLayer(net['input'], p=drop1)

    net['conv1'] = lasagne.layers.Conv2DLayer(net['inputDropOut'], num_filters=32, filter_size=(5, 5),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    
    
    net['maxPool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
     
    net['conv2'] = lasagne.layers.Conv2DLayer(
            net['maxPool1'], num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    
    
    net['maxPool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))
    
    
    #net['maxPool2DropOut'] = lasagne.layers.DropoutLayer(net['maxPool2'], p=drop1)

    
    
    net['fc'] = lasagne.layers.DenseLayer(
        net['maxPool2'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    
    net['fcDropOut'] = lasagne.layers.DropoutLayer(net['fc'], p=drop2)

    
    net['output'] = lasagne.layers.DenseLayer(
        net['fcDropOut'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def get_softmax_model(n_in, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, n_in))
    net['output'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net


def train_model(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='cnn', l2_ratio=1e-7,
          rtn_layer=True):
    print("classfier.py -- train_model")

    myTest = (model == 'softmax')
    
    train_x, train_y, test_x, test_y = dataset
    #print(train_x)
    n_in = train_x.shape[1]
    silent = False
    #print(train_x.shape)
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)
    if (not myTest):
        print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    if model=='cnn' or model=='cnn2' or model=='Droppcnn' or  model=='Droppcnn2':
        input_var = T.tensor4('x')
    else:
        input_var = T.matrix('x')
    target_var = T.ivector('y')
    if model == 'cnn':
        
        print('Using conv neural network...')
        net = get_cnn_model(n_in, n_hidden, n_out)
        
    elif model == 'cnn2':
        print('Using conv neural network...')
        net = get_cnn2_model(n_in, n_hidden, n_out)
    elif model == 'nn':
        print('Using neural network...')
        net = get_nn_model(n_in, n_hidden, n_out)
    elif model == 'Dropnn':
        print('Using neural network with Dropout')
        net = get_Dropnn_model(n_in, n_hidden, n_out,0.75,0.5)
    elif model == 'Droppcnn':
        print('Using neural network with Dropout')
        net = get_Droppcnn_model(n_in, n_hidden, n_out,0.2,0.5,0.5)
        
    elif model == 'Droppcnn2':
        print('Using Conv neural network with Dropout')
        net = get_Droppcnn2_model(n_in, n_hidden, n_out,0.5,0.5,0.5)
    else:
        if (not myTest and not silent):
            print('Using softmax regression...')
        net = get_softmax_model(n_in, n_out)

    net['input'].input_var = input_var
    
    output_layer = net['output']
    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_ratio * lasagne.regularization.regularize_network_params(output_layer,
                                                                                 lasagne.regularization.l2)
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = theano.function([input_var], test_prediction)
    if (not myTest and not silent):
        print('Training...')
    counter = 1
    for epoch in range(epochs):
        print(epoch)
        loss = 0
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            #input_batch = (np.reshape(input_batch,(len(input_batch),3,32,32)))
            #print("classfier.py -- func -- train_model -- iterate_minibatches")

            loss += train_fn(input_batch, target_batch)

        loss = round(loss, 4)
        if (not myTest and not silent):
            print('Epoch {}, train loss {}'.format(epoch, loss))
        
        
        counter = counter +1
    pred_y = []
    for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
        #input_batch = (np.reshape(input_batch,(len(input_batch),3,32,32)))
        pred = test_fn(input_batch)
        pred_y.append(np.argmax(pred, axis=1))
    pred_y = np.concatenate(pred_y)
    if (not myTest and not silent):
        print(classification_report(train_y, pred_y))

    if test_x is not None:
        if (not myTest and not silent):
            print('Testing...')
        pred_y = []

        if batch_size > len(test_y):
            batch_size = len(test_y)

        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            #input_batch = (np.reshape(input_batch,(len(input_batch),3,32,32)))
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
        pred_y = np.concatenate(pred_y)
        if (not myTest and not silent):
            print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    

    print(classification_report(test_y, pred_y))

    if ( myTest):
        pres= round(precision_score(test_y, pred_y,average='weighted'),4)
        recall= round(recall_score(test_y, pred_y,average='weighted'),4)
        accuracy= round(accuracy_score(test_y, pred_y),5)
        print("precision",pres)
        print("recall",recall)
        print("accuracy",accuracy)
        return pres,recall, accuracy
        # return classification_report(test_y, pred_y) #precision_recall_fscore_support(test_y, pred_y,average=None)
        
    if rtn_layer:
        
        return output_layer
    else:
        return pred_y


def load_dataset(train_feat, train_label, test_feat=None, test_label=None):
    
    
    train_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
    train_y = np.genfromtxt(train_label, dtype='int32')
    
    min_y = np.min(train_y)
    train_y -= min_y
    if test_feat is not None and test_label is not None:
        test_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
        test_y = np.genfromtxt(train_label, dtype='int32')
        test_y -= min_y
    else:
        test_x = None
        test_y = None
    return train_x, train_y, test_x, test_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_feat', type=str)
    parser.add_argument('train_label', type=str)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--model', type=str, default='nn')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print(vars(args))
    dataset = load_dataset(args.train_feat, args.train_label, args.test_feat, args.train_label)
    train_model(dataset,
          model=args.model,
          learning_rate=args.learning_rate,
          batch_size=args.batch_size,
          n_hidden=args.n_hidden,
          epochs=args.epochs)


if __name__ == '__main__':
    main()
