#!/usr/bin/python3

# ML_hw2.2.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>
# 5.5 Training

# v1 2021/10/02


# start time
import time
print(time.strftime("%H:%M:%S", time.localtime()), "start")


# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Loading CSV Files
train_h = pd.read_csv("datasets_3_train.csv")
test_h = pd.read_csv("datasets_3_test.csv")

# transfer input as array
X_all_train = np.array(train_h.drop(['Sales'],1))
y_train = np.array(train_h['Sales'])


X_all_test = np.array(test_h.drop(['Sales'],1))
y_test = np.array(test_h['Sales'])



# functions:

# update b
def update_b(b, y_h, y, eta):
	b = b - (2*eta*(y_h - y))
	return b


# update W, no regularization
def update_W_noreg(W, X, y_h, y, eta):
	W = W - (2*eta*(y_h - y)*X)
	return W

# update W, L2 (ridge) regularization
def update_W_L2reg(W, X, y_h, y, eta, lambda_) :
	W = (1 - 2*lambda_*eta)*W - (2*eta*(y_h - y)*X)
	return W

# update W, L1 (Lasso) regularization
def update_W_L1reg(W, X, y_h, y, eta, lambda_) :
	Wjs = []
	for j in list(range(len(W))):
		if W[j] >= 0:
			W_j = W[j] - (2*eta*(y_h-y)*X[j]) - (eta*lambda_)
			Wjs.append(W_j)
		else:
			W_j = W[j] - (2*eta*(y_h-y)*X[j]) + (eta*lambda_)
			Wjs.append(W_j)
	W = np.array(Wjs)
	return W

# Evaluation
def evaluate(W,b):
	loss_sum = 0
	for i in list(range(len(X_all_test))):
		y_i = y_test[i]
		X_i = X_all_test[i]
		y_i_h = sum(W * X_i) + b
		loss_sum += (y_i_h - y_i)**2
	loss = loss_sum / len(X_all_test)
	return loss

######### Q5.5.1 #########

#### Training ####
# 50 epochs, no regularization, η = 0.01
eta = 0.01

# You must initialize the model parameters to zeros
W = np.array([0]*12) # wj for 12 features
b = 0

# loss values for plotting
loss_list = []

for epoch in list(range(50)):
	for i in list(range(len(X_all_train))): # i is row
		y_i = y_train[i]
		X_i = X_all_train[i]

		y_i_h = sum(W * X_i) + b

		loss = (y_i_h - y_i)**2
		loss_list.append(loss)

		# update parameters
		W = update_W_noreg(W, X_i, y_i_h, y_i, eta)
		b = update_b(b, y_i_h, y_i, eta)

plt.scatter(list(range(len(loss_list))), loss_list, s=30, facecolors='none', edgecolors='black')

# Add title and axis names
plt.title('50 epochs, no regularization, η = 0.01')
plt.xlabel('Step')
plt.ylabel('Training Loss')

plt.savefig('noreg_eta_0.01.png')
plt.clf()
print(time.strftime("%H:%M:%S", time.localtime()), "no regularization, η = 0.01")


#### Evalutation ####
print("test loss 1: %.5f" % (evaluate(W,b)))



######### Q5.5.2 #########
# 50 epochs, no regularization, η = 0.001
eta = 0.001

# You must initialize the model parameters to zeros
W = np.array([0]*12) # wj for 12 features
b = 0

# loss values for plotting
loss_list = []

for epoch in list(range(50)):
	for i in list(range(len(X_all_train))): # i is row
		y_i = y_train[i]
		X_i = X_all_train[i]

		y_i_h = sum(W * X_i) + b

		loss = (y_i_h - y_i)**2
		loss_list.append(loss)

		# update parameters
		W = update_W_noreg(W, X_i, y_i_h, y_i, eta)
		b = update_b(b, y_i_h, y_i, eta)

plt.scatter(list(range(len(loss_list))), loss_list, s=30, facecolors='none', edgecolors='black')

# Add title and axis names
plt.title('50 epochs, no regularization, η = 0.001')
plt.xlabel('Step')
plt.ylabel('Training Loss')

plt.savefig('noreg_eta_0.001.png')
plt.clf()
print(time.strftime("%H:%M:%S", time.localtime()), "no regularization, η = 0.001")

#### Evalutation ####
print("test loss 2: %.5f" % (evaluate(W,b)))


######### Q5.5.3 #########
# 50 epochs, L2 regularization, η = 0.001, λ = 0.1
eta = 0.001
lambda_ = 0.1

# You must initialize the model parameters to zeros
W = np.array([0]*12) # wj for 12 features
b = 0

# loss values for plotting
loss_list = []

for epoch in list(range(50)):
	for i in list(range(len(X_all_train))): # i is row
		y_i = y_train[i]
		X_i = X_all_train[i]

		y_i_h = sum(W * X_i) + b

		loss = (y_i_h - y_i)**2 + lambda_*sum(W**2)

		loss_list.append(loss)

		# update parameters
		W = update_W_L2reg(W, X_i, y_i_h, y_i, eta, lambda_)
		b = update_b(b, y_i_h, y_i, eta)

plt.scatter(list(range(len(loss_list))), loss_list, s=30, facecolors='none', edgecolors='black')
# Add title and axis names
plt.title('50 epochs, L2 regularization, η = 0.001, λ = 0.1')
plt.xlabel('Step')
plt.ylabel('Training Loss')

plt.savefig('L2.png')
plt.clf()
print(time.strftime("%H:%M:%S", time.localtime()), "L2 regularization, η = 0.001, λ = 0.1")

#### Evalutation ####
print("test loss 3: %.5f" % (evaluate(W,b)))


######### Q5.5.4 #########
# 50 epochs, L1 regularization, η = 0.001, λ = 0.1
eta = 0.001
lambda_ = 0.1

# You must initialize the model parameters to zeros
W = np.array([0]*12) # wj for 12 features
b = 0

# loss values for plotting
loss_list = []

for epoch in list(range(50)):
	for i in list(range(len(X_all_train))): # i is row
		y_i = y_train[i]
		X_i = X_all_train[i]

		y_i_h = sum(W * X_i) + b

		loss = (y_i_h - y_i)**2 + lambda_*sum(abs(W))
		loss_list.append(loss)

		# update parameters
		W = update_W_L1reg(W, X_i, y_i_h, y_i, eta, lambda_)
		b = update_b(b, y_i_h, y_i, eta)

plt.scatter(list(range(len(loss_list))), loss_list, s=30, facecolors='none', edgecolors='black')
# Add title and axis names
plt.title('50 epochs, L1 regularization, η = 0.001, λ = 0.1')
plt.xlabel('Step')
plt.ylabel('Training Loss')

plt.savefig('L1.png')
plt.clf()
print(time.strftime("%H:%M:%S", time.localtime()), "L1 regularization, η = 0.001, λ = 0.1")

#### Evalutation ####
print("test loss 4: %.5f" % (evaluate(W,b)))