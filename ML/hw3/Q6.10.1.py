#!/usr/bin/python3

# ML_hw3_Q5.5.1.py

# modules
import pandas as pd
import numpy as np
import math
import statistics


# Loading CSV Files

## training data ##
train_f = open("training_data_student/train.csv")
train_a = np.loadtxt(train_f, delimiter=",")

train_X = np.delete(train_a, -1, 1)
train_y = train_a[:,784]

## test data ##
test_f = open("training_data_student/test.csv")
test_a = np.loadtxt(test_f, delimiter=",")

test_X = np.delete(test_a, -1, 1)
test_y = test_a[:,784]


## init weights ##
# 1st layer
alpha1_f = open("params/alpha1.txt")
alpha1 = np.loadtxt(alpha1_f, delimiter=",")

beta1_f = open("params/beta1.txt")
beta1 = np.loadtxt(beta1_f, delimiter=",")
beta1 = beta1[:,None]

alpha_star = alpha1
alpha = np.hstack((beta1, alpha_star))


# 2nd layer
alpha2_f = open("params/alpha2.txt")
alpha2 = np.loadtxt(alpha2_f, delimiter=",")

beta2_f = open("params/beta2.txt")
beta2 = np.loadtxt(beta2_f, delimiter=",")
beta2 = beta2[:,None]

beta_star = alpha2
beta = np.hstack((beta2, beta_star))


### hyper-parameters
# learning rate
eta = 0.01
# number of epochs
epochs = 100

batch_size = 10

## training loop ##
print("epoch\ttrain_l\ttest_l\ttest_ac")
for epoch in range(epochs):
	## training ##
	l_train_loss = []
	total_t = int(len(train_X)/batch_size)
	for t in range(total_t):
		x = train_X[t:t+batch_size].T
		x = np.vstack(([1]*batch_size,x))

		# y: to one hot
		y_true = train_y[t:t+batch_size].T
		y_onehot = np.zeros((10,batch_size))
		for n in range(batch_size):
			y_onehot[int(y_true[n]), n] = 1
		y = y_onehot

		####### forward #######
		a = np.matmul(alpha,x) # 1st linear layer (pre-activation)

		z = 1/(1+np.exp(-a)) # hidden layer (Sigma activation)
		z = np.vstack(([1]*batch_size,z)) # add bias z0=1

		b = np.matmul(beta,z) # 2nd linear layer (pre-activation)

		y_hat = np.exp(b)/sum(np.exp(b)) # output layer (softmax activation)

		loss_v = -sum(y*np.log(y_hat))
		loss = statistics.mean(loss_v)
		l_train_loss.append(float(loss))

		####### backward #######
		### update weights: beta ###
		# deri of l by b (10, batch_size)
		d_l_b = -y + y_hat

		# deri of l by beta (batch_size, 257)
		d_l_beta = np.matmul(d_l_b, np.transpose(z))

		### update weights: alpha ###
		# deri of z by a (256, batch_size)
		d_z_a = np.exp(-a)/((1+np.exp(-a))**2)

		# deri of l by z (256, batch_size)
		beta_star = np.delete(beta, 0, 1)
		d_l_z = np.matmul(np.transpose(beta_star), d_l_b)

		# deri of l by a (256, batch_size)
		d_l_a = d_l_z * d_z_a

		# deri of l by alpha (256, 785)
		d_l_alpha = np.matmul(d_l_a, np.transpose(x)) 

		# update alpha (256, 785)
		alpha = alpha - ( eta * d_l_alpha )

		# update beta (10, 257)
		beta = beta - (eta * d_l_beta)

		#print('step', t, float(loss))

	# compute the average training loss for the epoch
	average_training_loss = statistics.mean(l_train_loss)


	## testing ##
	l_test_loss = []
	n_correct = 0
	for n in range(len(test_X)):
		# x: add bias x0=1
		x = test_X[n]
		x = np.vstack(([1],x[:, np.newaxis]))

		# y: to one hot
		y_true = int(test_y[n])
		y_onehot = [0]*10
		y_onehot[y_true] = 1
		y = np.array([y_onehot]).T

		####### forward #######
		a = np.matmul(alpha,x) # 1st linear layer (pre-activation)

		z = 1/(1+np.exp(-a)) # hidden layer (Sigma activation)
		z = np.vstack(([1],z)) # add bias z0=1

		b = np.matmul(beta,z) # 2nd linear layer (pre-activation)

		y_hat = np.exp(b)/sum(np.exp(b)) # output layer (softmax activation)

		loss = -sum(y*np.log(y_hat))
		l_test_loss.append(float(loss))

		# evaluate
		y_hat = y_hat.tolist()
		y_predict = y_hat.index(max(y_hat))
		if y_predict == y_true:
			n_correct += 1
	
	# compute the average test loss for the epoch
	average_test_loss = statistics.mean(l_test_loss)

	# compute test accuracy
	l_test_accuracy = n_correct / len(test_X)

	print('%i\t%.4f\t%.4f\t%.4f' %(epoch, average_training_loss, average_test_loss, l_test_accuracy))





# close the files
train_f.close()
test_f.close()

alpha1_f.close()
alpha2_f.close()
beta1_f.close()
beta2_f.close()