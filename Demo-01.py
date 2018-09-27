import tensorflow as tf
import numpy as np

### Create test data ###
# generate phony data
x_data = np.random.rand(100).astype(np.float32)
# y_data is our expectation
y_data = x_data * 0.25 + 0.5

### Create tf structure ###
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# Correspondding to y_data
y = x_data * weights + biases

# Get variance
# Computes the mean of elements across dimensions of a tensor. 
loss = tf.reduce_mean(tf.square(y - y_data))
# Optimizer that implements the gradient descent algorithm.
optimizer = tf.train.GradientDescentOptimizer(0.5)
# Train: Minimum variance
train = optimizer.minimize(loss)

# Init variables
init = tf.initialize_all_variables()

### Create session
session = tf.Session()
session.run(init)

for step in range(201):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(weights), session.run(biases))