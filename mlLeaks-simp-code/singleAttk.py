'''
Created on 5 Dec 2018

@author: Wentao Liu, Ahmed.Salem
'''
# import os
# os.environ['THEANO_FLAGS'] = 'optimizer=None'
import sys
sys.dont_write_bytecode = True

import numpy as np

import pickle
from sklearn.model_selection import train_test_split
import random
import lasagne

# import tensorflow as tf

import deeplearning as dp
import classifier
# import tensorflow as tf
def clipDataTopTest(dataToClip, top=3):
	res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]

	return np.array(res)

def readData(data_path):
	for i in range(5):
		# f = open(data_path + '/data_batch_' + str(i + 1) +'.bin', 'rb')
		f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
	
		train_data_dict = pickle.load(f,encoding='bytes')
		# input(type(train_data_dict[b'data']))
		f.close()
		#print(train_data_dict)
		if i == 0:
			X = train_data_dict[b"data"]
			y = train_data_dict[b"labels"]
			continue
		X = np.concatenate((X , train_data_dict[b"data"]),   axis=0)
		y = np.concatenate((y , train_data_dict[b"labels"]), axis=0)

	f = open(data_path + '/test_batch', 'rb')
	test_data_dict = pickle.load(f,encoding='bytes')
	f.close()

	XTest = np.array(test_data_dict[b"data"])
	yTest = np.array(test_data_dict[b"labels"])
		
	return X, y, XTest, yTest

def trainTarget(model, X, y,
				X_test=[], y_test =[],
				toShuffle=True,
				test_size=0.5, 
				inepochs=50, batch_size=300,
				learning_rate=0.001):
	print("singleAttk.py -- trainTarget")

	X_train = X
	y_train = y
	if(toShuffle):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

	dataset = (X_train.astype(np.float32),
			   y_train.astype(np.int32),
			   X_test.astype(np.float32),
			   y_test.astype(np.int32))

	attack_x, attack_y, theModel, modelToUse = dp.train_target_model(dataset=dataset,name='Newshalf1', epochs=inepochs, batch_size=batch_size,learning_rate=learning_rate,
				   n_hidden=128,l2_ratio = 1e-07,model='cnn',save=False)

	return attack_x, attack_y, theModel, modelToUse

def load_data(data_name):
	with np.load( data_name) as f:
		train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]

	return train_x, train_y

def trainAttackSoftmax(X_train, y_train, X_test, y_test):
	dataset = (X_train.astype(np.float32),
			   y_train.astype(np.int32),
			   X_test.astype(np.float32),
			   y_test.astype(np.int32))

	output = classifier.train_model(dataset=dataset,
									epochs=50,
									batch_size=10,
									learning_rate=0.01,
									n_hidden=64,
									l2_ratio = 1e-6,
									model='softmax')

	return output

def preprocessesCIFAR(X):
	#X = np.dstack((X[:, :1024], X[:, 1024:2048], X[:, 2048:]))
	X = np.dstack((X[:, :1], X[:, 1:2], X[:, 2:]))
	X = X.reshape((X.shape[0], 32, 32, 3)).transpose(0,3,1,2)

	offset = np.mean(X, 0)
	scale = np.std(X, 0).clip(min=1)
	X = (X - offset) / scale
	
	return X.astype(np.float32)

def preprocess_for_save(toTrainData, toTestData):
	def reshape_for_save(raw_data):
		raw_data = np.dstack((raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
		raw_data = raw_data.reshape((raw_data.shape[0], 32, 32, 3)).transpose(0,3,1,2)
		return raw_data.astype(np.float32)

	offset = np.mean(reshape_for_save(toTrainData), 0)
	scale  = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

	def rescale(raw_data):
		return (reshape_for_save(raw_data) - offset) / scale

	return rescale(toTrainData), rescale(toTestData)

# def load_cifar_10():
#   """Loads MNIST and preprocesses to combine training and validation data."""
#   train, test = tf.keras.datasets.cifar10.load_data()
#   train_data, train_labels = train
#   test_data, test_labels = test

#   train_data = np.array(train_data, dtype=np.float32)
#   test_data = np.array(test_data, dtype=np.float32)

#   train_labels = np.array(train_labels, dtype=np.int32)
#   test_labels = np.array(test_labels, dtype=np.int32)
#   print(train_data.shape)
#   print(test_data.shape)
#   print(train_labels.shape)
#   print(test_labels.shape)
#   print(train_data[0])
#   print(train_labels[0])

print("preprocessing for CIFAR 10")

cifar_path = './data/cifar-10-batches-py-official'

dataX, dataY, _, _ = readData(cifar_path)

c = zip(dataX, dataY)  
c = list(c)
# input(type(c))
# input(len(c))
random.shuffle(c)
dataX, dataY = zip(*c)
print("len:", len(dataX))

cluster = 10520
print("cluster:", cluster)

toTrainData  = np.array(dataX[:cluster])
toTrainLabel = np.array(dataY[:cluster])

shadowData  = np.array(dataX[cluster:cluster*2])
shadowLabel = np.array(dataY[cluster:cluster*2])

toTestData  = np.array(dataX[cluster*2:cluster*3])
toTestLabel = np.array(dataY[cluster*2:cluster*3])

shadowTestData  = np.array(dataX[cluster*3:cluster*4])
shadowTestLabel = np.array(dataY[cluster*3:cluster*4])

toTrainDataSave, toTestDataSave    = preprocess_for_save(toTrainData, toTestData)
shadowDataSave, shadowTestDataSave = preprocess_for_save(shadowData, shadowTestData)

export_path = './data/exported_data'

print("saving cifar model: (training, testing) X (target, shadow)")

np.savez(export_path + '/CIFAR10_targetTrain.npz', toTrainDataSave, toTrainLabel)
np.savez(export_path + '/CIFAR10_targetTest.npz',  toTestDataSave, toTestLabel)
np.savez(export_path + '/CIFAR10_shadowTrain.npz', shadowDataSave, shadowLabel)
np.savez(export_path + '/CIFAR10_shadowTest.npz',  shadowTestDataSave, shadowTestLabel)

print("preprocess finished\n\n")

#if testing == 'trainStacking':

num_epoch = 20

print("training NN for Target (epoch: {})".format(num_epoch))
targetTrain, targetTrainLabel  = load_data(export_path + '/CIFAR10_targetTrain.npz')
targetTest,  targetTestLabel   = load_data(export_path + '/CIFAR10_targetTest.npz')

targetTrainNN = preprocessesCIFAR(targetTrain)
targetTestNN = preprocessesCIFAR(targetTest)
targetX, targetY, targetModelToStore, targetModelNN = trainTarget('cnn',targetTrainNN, targetTrainLabel, X_test=targetTestNN, y_test=targetTestLabel, toShuffle= False, inepochs=num_epoch, batch_size=100) 

np.savez(export_path + '/CIFAR10_targetModel.npz', *lasagne.layers.get_all_param_values(targetModelToStore))

print("target model saved\n")

print("training NN for Shadow (epoch: {})".format(num_epoch))

shadowTrainRaw, shadowTrainLabel  = load_data(export_path + '/CIFAR10_shadowTrain.npz')
targetTestRaw,  shadowTestLabel   = load_data(export_path + '/CIFAR10_shadowTest.npz')
shadowTrainNN = preprocessesCIFAR(shadowTrainRaw)
shadowTestNN  = preprocessesCIFAR(targetTestRaw)
shadowX, shadowY, shadowModelToStore, shadowModelNN = trainTarget('cnn', shadowTrainNN, shadowTrainLabel, X_test=shadowTestNN, y_test=shadowTestLabel, toShuffle= False, inepochs=num_epoch, batch_size=100) 

np.savez(export_path + '/CIFAR10_shadowModel.npz', *lasagne.layers.get_all_param_values(shadowModelToStore))

print("shadow model saved\n")

print("attacking")

targetX = clipDataTopTest(targetX)
shadowX = clipDataTopTest(shadowX)

temp = trainAttackSoftmax(targetX, targetY, shadowX, shadowY)  
print("RESULT")
print(temp)
