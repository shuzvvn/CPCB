#!/usr/bin/python3

# ML_hw4_Q6.1.py
# /mnt/c/Users/vivia/CPCB/ML/hw4
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/adaboost.py

# modules
# import pandas as pd
import numpy as np
#import math
# import statistics
# import matplotlib.pyplot as plt



# Suggested structure for the programming assignment: Consider writing the following functions:
# 1. read data: This function reads the data from the input file.
def read_data(file):
	data_f = open(file)
	data_a = np.loadtxt(data_f, delimiter=",")
	data_X = np.delete(data_a, -1, axis=1)
	data_y = data_a[:,2]
	return data_X, data_y


# 2. weak classifier: This function finds the best weak classifier, which is a 1-level decision tree. Defining N as number of training points and D = 2 as the number of features per data point, the inputs to the function should be the input data (N×D array), the true labels (N×1 array) and current point weights ω(t) i ∈ Ωt (N ×1 array) and you should return the best weak classifier with the best split based on the weighted error. Some of the things to include in the output can be: best feature index, best split value, label, value of αt and predicted labels by the best weak classifier. Note: αt should be a (T × 1 array, for t = 1 . . . T, and T denotes the number of iterations that you choose to run the boosting algorithm)
def weak_classifier(data_X, data_y, weights):
	n_samples, d_features = data_X.shape
	epsilon_min = float('inf') # minimizing the weighted training error
	classifier = {}

	# iterate through all possible decision stumps
	# potential dividing line comes from the training data values
	for d in range(d_features):
		X_d = data_X[:, d]
		pdls = np.unique(X_d) # list of potential dividing line
		for pdl in pdls:
			label = 1
			pred = np.full(n_samples, -1)
			pred[X_d > (pdl - 0.000001)] = 1

			# Compute the weighted training error of ℎt (epsilon)
			epsilon_h = sum(weights[data_y != pred])
			# determine which side should be classified as +1
			if epsilon_h > 0.5:
				label = -1
				epsilon_h = 1 - epsilon_h

			# update the best weak classifier
			if epsilon_h < epsilon_min:
				pred = pred * label
				epsilon_min = epsilon_h
				classifier = {'dimension':d, 'dividing_line':pdl, 'label':label, 'pred': pred}
				
	# Compute the importance of ℎt
	alpha = (np.log((1 - epsilon_min) / epsilon_min)) / 2
	classifier['alpha'] = alpha

	return classifier

# 3. update weights: This function computes the updated weights Ωt+1, The inputs to this function should be current weights Ωt (N × 1 array), the value of αt, the true target values (N × 1 array) and the predicted target values (N × 1 array). And the function should output the updated distribution Ωt+1.
def update_weights(weights, alpha, data_y, pred):
	weights *= np.exp(-alpha * data_y * pred)
	weights /= np.sum(weights)
	return weights


# 4. adaboost predict: This function returns the predicted labels for each weak classifier. The inputs: the input data (N ×D array), the array of weak classifiers (T ×3 array) and the array of αt for t = 1 . . . T(T × 1 array) and output the predicted labels (N × 1 array)
def adaboost_predict(data_X, hlist):
	n_samples = data_X.shape[0]
	# returns the predicted labels for each weak classifier
	for clf in hlist:
		X_d = data_X[:, clf['dimension']]
		pred = np.full(n_samples, -1)
		if clf['label'] == 1:
			pred[X_d > clf['dividing_line']] = 1
		else:
			pred[X_d < clf['dividing_line']] = 1
		clf['pred'] = pred
	# output: an aggregated hypothesis
	pred_votes = [clf['alpha'] * clf['pred'] for clf in hlist]
	pred = np.sign(np.sum(pred_votes, axis = 0))
	return pred


# 5. eval model: This function evaluates the model with test data by measuring the accuracy. Assuming we have M test points, the inputs should be the test data (M × D array), the true labels for test data (M × 1 array), the array of weak classifiers (T × 3 array), and the array of αt for t = 1 . . . T (T × 1 array). The function should output: the predicted labels (M × 1 array) and the accuracy.
def eval_model(X_test, y_test, hlist):
	# returns the predicted labels for each weak classifier
	pred = adaboost_predict(X_test, hlist)
	accuracy = np.sum(y_test == pred) / len(y_test)
	return pred, accuracy


# 6. Adaboost train: This function trains the model by using AdaBoost algorithm.
# Inputs:
# - The number of iterations (T)
# - The input data for training data (N × D array)
# - The true labels for training data (N × 1 array)
# - The input features for test data (M × D array)
# - The true labels for test data (M × 1 array)
# Output:
# - The array of weak classifiers (T × 3 array)
# - The array of αt for t = 1 . . . T (T × 1 array)
def Adaboost_train(num_iter, X_train, y_train):
	# Initialize input weights: w1, ..., wn = 1/n
	n_samples = X_train.shape[0]
	weights = np.full(n_samples, (1/n_samples))
	# list of classifiers
	hlist = [] 

	for t in range(num_iter):
		# finds the best weak classifier
		clf = weak_classifier(X_train, y_train, weights)
		# Save classifier
		hlist.append(clf)
		# computes the updated weights
		weights = update_weights(weights, clf['alpha'], y_train, clf['pred'])
	print(list(weights))
	return hlist



def main():

	# Input: D(y = {-1, +1}), T
	num_iter = 10

	X_train, y_train = read_data("train_adaboost.csv")
	X_test, y_test = read_data("test_adaboost.csv")

	hlist = Adaboost_train(num_iter, X_train, y_train)

	final_pred, final_acc = eval_model(X_test, y_test, hlist)
	print(final_acc)





if __name__ == '__main__':
	main()
