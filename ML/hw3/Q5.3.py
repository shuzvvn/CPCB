import numpy as np
import math
x = np.array([1.0,1,0,1,0,1,0])
x = x[:, np.newaxis]

y = np.array([0,1,0])
y = y[:, np.newaxis]

# eta
eta = 1


# init weights
alpha = np.array([[1,2,1,-1,-1,0,-2],[1,0,1,0,-1,1,3],[1,-1,2,1,3,1,-1],[1,1,3,4,2,-1,2]])
beta = np.array([[1,2,-2,2,1],[1,3,-1,1,2],[1,0,-1,0,1]])

a = np.matmul(alpha,x)
# array([[2.],
#        [2.],
#        [2.],
#        [5.]])


z = 1/(1+np.exp(-a))
z = np.vstack(([1],z))
# array([[1.        ],
#        [0.88079708],
#        [0.88079708],
#        [0.88079708],
#        [0.99330715]])


b = np.matmul(beta,z)
# array([[3.75490131],
#        [5.62900553],
#        [1.11251007]])


y_hat = np.exp(b)/sum(np.exp(b))
# array([[0.1318188 ],
#        [0.85879691],
#        [0.00938429]])


loss = -sum(y*np.log(y_hat))
# array([0.15222281])




# deri of l by b
d_l_b = -y + y_hat

# deri of l by beta
d_l_beta = np.matmul(d_l_b, np.transpose(z))

# update beta
beta = beta - (eta * d_l_beta)
# array([[ 0.8681812 ,  1.88389439, -2.11610561,  1.88389439,  0.86906344],
#        [ 1.14120309,  3.12437127, -0.87562873,  1.12437127,  2.14025804],
#        [ 0.99061571, -0.00826566, -1.00826566, -0.00826566,  0.99067852]])




# deri of z by a (4*1)
d_z_a = np.exp(-a)/((1+np.exp(-a))**2)

# deri of l by z (4*1)
d_l_z = np.matmul(np.transpose(np.delete(beta, 0, 1)), d_l_b)

# deri of l by a (4*1)
d_l_a = d_l_z * d_z_a



# update alpha
alpha = alpha - ( eta * np.matmul(d_l_a, np.transpose(x)) )



####################
a = np.matmul(alpha,x)
# array([[2.],
#        [2.],
#        [2.],
#        [5.]])


z = 1/(1+np.exp(-a))
z = np.vstack(([1],z))
# array([[1.        ],
#        [0.88079708],
#        [0.88079708],
#        [0.88079708],
#        [0.99330715]])


b = np.matmul(beta,z)
# array([[3.75490131],
#        [5.62900553],
#        [1.11251007]])


y_hat = np.exp(b)/sum(np.exp(b))
# array([[0.1318188 ],
#        [0.85879691],
#        [0.00938429]])