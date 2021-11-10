#!/usr/bin/python3

# ML_hw3_Q5.5.1.py

# modules
import numpy as np
import math


x = np.array([1.0,1,0,1,0,1,0])
x = x[:, np.newaxis]

y = np.array([0,1,0])
y = y[:, np.newaxis]

# eta
eta = 1

# lamda
#lamda = 0.01

# init weights
alpha = np.array([[1,2,1,-1,-1,0,-2],[1,0,1,0,-1,1,3],[1,-1,2,1,3,1,-1],[1,1,3,4,2,-1,2]])
beta = np.array([[1,2,-2,2,1],[1,3,-1,1,2],[1,0,-1,0,1]])


for epoch in range(1):
	a = np.matmul(alpha,x)

	z = 1/(1+np.exp(-a))
	z = np.vstack(([1],z))

	b = np.matmul(beta,z)

	y_hat = np.exp(b)/sum(np.exp(b))

	# loss1 = -sum(y*np.log(y_hat))
	alpha_start = np.delete(alpha, 0, 1)
	loss = -sum(y*np.log(y_hat))

	### update weights beta ###
	# deri of l by b
	d_l_b = -y + y_hat

	# deri of l by beta
	d_l_beta = np.matmul(d_l_b, np.transpose(z))

	# update beta
	# beta = beta - (eta * d_l_beta)
	# L2
	beta = beta - (eta * (d_l_beta))
	print(beta)


	### update weights alpha ###
	# deri of z by a (4*1)
	d_z_a = np.exp(-a)/((1+np.exp(-a))**2)

	# deri of l by z (4*1)
	d_l_z = np.matmul(np.transpose(np.delete(beta, 0, 1)), d_l_b)

	# deri of l by a (4*1)
	d_l_a = d_l_z * d_z_a

	# deri of l by alpha (4*7)
	d_l_alpha = np.matmul(d_l_a, np.transpose(x)) 

	# update alpha
	#alpha = alpha - ( eta * d_l_alpha )
	# L2
	alpha = alpha - (eta * (d_l_alpha))
	print(alpha)