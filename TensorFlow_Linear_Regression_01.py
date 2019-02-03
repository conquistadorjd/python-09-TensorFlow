################################################################################################
#   name:   TensorFlow_Linear_Regression_01.py
#   desc:   Linear Regression using TensorFlow
#   date:   2019-02-03
#   Author: conquistadorjd
################################################################################################
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

print('*** Program Started ***')
########## Input Data Creation
n = 20
x = np.arange(-n/2,n/2,1,dtype=np.float64)

m_real = np.random.uniform(0.8,0.9,(n,))
b_real = np.random.uniform(5,10,(n,))
print('m_real', type(m_real[0]))
y = x*m_real +b_real 

########## Variables definition
m = tf.Variable(np.random.uniform(5,15,(1,)))
b = tf.Variable(np.random.uniform(5,15,(1,)))

########## display inout data and datatypes
print('x', x, type(x), type(x[0]))
print('y', y, type(y), type(y[0]))
print('m', m, type(m))
print('b', b, type(b))

########## Plot input to see the data
# plt.scatter(x,y,s=None, marker='o',color='g',edgecolors='g',alpha=0.9,label="Linear Relation")
# plt.grid(color='black', linestyle='--', linewidth=0.5,markevery=int)
# plt.legend(loc=2)
# plt.axis('scaled')
# plt.show() 

########## Compute model and loss
model = tf.add(tf.multiply(x,m), b)
loss = tf.reduce_mean(tf.pow(model - y, 2))

########## Use following model if you get TypeError
# model = tf.add(tf.multiply(x, tf.cast(m, tf.float64)), tf.cast(b, tf.float64))
# loss = tf.reduce_mean(tf.pow(model - tf.cast(y, tf.float64), 2))
###########################################################################################

# Create optimizer
learn_rate = 0.01 # you can use 0.1/0.01/0.001 to test the output
num_epochs = 500  # Test output accuracy for different epochs
num_batches = n
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

########## Initialize variables
init = tf.global_variables_initializer()

########## Launch session
with tf.Session() as sess:
    sess.run(init)
    print('*** Initialize')

    ########## This is where training happens
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            sess.run(optimizer)

    ########## Display and plot results
    print('m = ', sess.run(m))
    print('b = ', sess.run(b))

    x1 = np.linspace(-10,10,50)
    y1 = sess.run(m)*x1+sess.run(b)  

    plt.scatter(x,y,s=None, marker='o',color='g',edgecolors='g',alpha=0.9,label="Linear Relation")
    plt.grid(color='black', linestyle='--', linewidth=0.5,markevery=int)
    plt.legend(loc=2)
    plt.axis('scaled')

    plt.plot(x1, y1, 'r')
    plt.savefig('TensorFlow_Linear_Regression_01.png')
    plt.show()    

print('*** Program ended ***')