#!/usr/bin/python3

# ML_hw3_Q6.3.py

# modules
import pandas as pd
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt


def plot_image(vector, out_f_name, label=None):
	"""
	Takes a vector as input of size (784) and saves as an image
	"""
	image = np.asarray(vector).reshape(28, 28)
	plt.imshow(image, cmap='gray')
	if label:
		plt.title(label)
	plt.axis('off')
	plt.savefig(f'{out_f_name}.png', bbox_inches='tight')



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


### hyper-parameters ###
# learning rate
eta = 0.01
# number of epochs
epochs = 1

batch_size = 1

## training loop ##
print("epoch\ttrain_l\ttest_l\ttest_ac")
for epoch in range(epochs):
	## training ##
	total_t = int(len(train_X)/batch_size)
	for t in range(1):
		start_row = t*batch_size
		x = train_X[start_row:start_row+batch_size].T
		x = np.vstack(([1]*batch_size,x))

		# y: to one hot
		y_true = train_y[start_row:start_row+batch_size].T
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

		####### backward #######
		# deri of l by b (10, batch_size)
		d_l_b = -y + y_hat

		# deri of l by beta (batch_size, 257)
		d_l_beta = np.matmul(d_l_b, np.transpose(z))

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
		alpha = alpha - ( eta * d_l_alpha/batch_size )

		# update beta (10, 257)
		beta = beta - (eta * d_l_beta/batch_size)


	############################################ Training loss ############################################
	l_train_loss = []
	n_correct = 0
	for n in range(len(train_X)): # 
		# x: add bias x0=1
		x = train_X[n]
		x = np.vstack(([1],x[:, np.newaxis]))

		# y: to one hot
		y_true = int(train_y[n])
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
		l_train_loss.append(float(loss))

		# evaluate
		y_hat = y_hat.tolist()
		y_predict = y_hat.index(max(y_hat))
		if y_predict == y_true:
			n_correct += 1
	
	# compute the average train loss for the epoch
	average_training_loss = statistics.mean(l_train_loss)

	# compute train accuracy
	l_train_accuracy = n_correct / len(train_X)

	############################################ Test loss ############################################
	l_test_loss = []
	n_correct = 0
	for n in range(len(test_X)): # 
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


###### confusion matrix ###### 
train_confusion = np.zeros((10,10))

for n in range(len(train_X)):
	# x: add bias x0=1
	x = train_X[n]		
	x = np.vstack(([1],x[:, np.newaxis]))

	# y: to one hot
	y_true = int(train_y[n])
	y_onehot = [0]*10
	y_onehot[y_true] = 1
	y = np.array([y_onehot]).T

	####### forward #######
	a = np.matmul(alpha,x) # 1st linear layer (pre-activation)
	z = 1/(1+np.exp(-a)) # hidden layer (Sigma activation)
	z = np.vstack(([1],z)) # add bias z0=1
	b = np.matmul(beta,z) # 2nd linear layer (pre-activation)
	y_hat = np.exp(b)/sum(np.exp(b)) # output layer (oftmax activation)

	# evaluate
	y_hat_l = y_hat.tolist()
	y_predict = y_hat_l.index(max(y_hat_l))
	train_confusion[y_true, y_predict] += 1


test_confusion = np.zeros((10,10))

l_cate = [0,5,6,9]

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
	y_hat = np.exp(b)/sum(np.exp(b)) # output layer (oftmax activation)

	# evaluate
	y_hat_l = y_hat.tolist()
	y_predict = y_hat_l.index(max(y_hat_l))
	test_confusion[y_true, y_predict] += 1

	# for Q9
	if y_true in l_cate and y_predict != y_true:
		print(n, y_true, y_predict)
		l_cate.remove(y_true)
		plot_image(test_X[n], y_true, label=None)

# # save confusion matrix as csv
np.savetxt('train_confusion.csv', train_confusion, delimiter=',')
np.savetxt('test_confusion.csv', test_confusion, delimiter=',')


# close the files
train_f.close()
test_f.close()

alpha1_f.close()
alpha2_f.close()
beta1_f.close()
beta2_f.close()