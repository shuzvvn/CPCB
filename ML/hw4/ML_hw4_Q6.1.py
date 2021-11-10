#!/usr/bin/python3

# ML_hw4_Q6.1.py

# modules
import pandas as pd
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt



# Suggested structure for the programming assignment: Consider writing the following functions:
# 1. read data: This function reads the data from the input file.
def read_data(file):
	data_f = open(file)
	data_a = np.loadtxt(data_f, delimiter=",")
	data_X = np.delete(data_a, -1, axis=1)
	data_y = data_a[:,2]
	return data_X, data_y










def main():
	num_iter = 400

	X_train, y_train = read_data("train_adaboost.csv")
	X_test, y_test = read_data("test_adaboost.csv")

	print(X_train)

	print(y_test)

	# hlist, alphalist = train(num_iter, X_train, y_train, X_test, y_test)
	# final_pred, final_acc = eval_model(X_test, y_test, hlist, alphalist)










"""
# Init centers
u1 = np.array([5.3, 3.5])
u2 = np.array([5.1, 4.2])

DataPoints = [[5.5,3.1],
[5.1,4.8],
[6.3,3.0],
[5.5,4.4],
[6.8,3.5]]

DataPoints = np.array(DataPoints)


# assign each data point to it's nearest cluster center
c1 = []
c2 = []
for i in range(len(DataPoints)):
	if math.dist(DataPoints[i], u1) < math.dist(DataPoints[i], u2):
		c1 = c1 + [DataPoints[i]]
	else:
		c2 = c2 + [DataPoints[i]]

c1 = np.array(c1)
c2 = np.array(c2)

# recomputing centers
mean1 = np.mean(c1, axis=0)
mean2 = np.mean(c2, axis=0)



print(mean1)
print(mean2)

print(len(c1))
print(len(c2))
"""