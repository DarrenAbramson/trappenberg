"""A very simple MNIST classifier modified using the "tutorial for experts":
    modified from
    https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/index.html
    and
    https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html
    """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

# -----------Import Data-----------
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets(".", one_hot=True)

# Needed for iPython notebook since data structures are maintained across
# execution of code blocks
tf.reset_default_graph()

# -----------Helper Functions-----------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# -----------Placeholder for reading in images------------
x = tf.placeholder(tf.float32, [None, 784])

# -----------First convolutional layer-----------
# Reshape x to a 4d tensor, with the second and third dimensions corresponding
# to image width and height, and the final dimension corresponding to the number
# of color channels
x_image = tf.reshape(x, [-1,28,28,1])

# Convolution, followed by max pooling. The convolution will compute 32 features
# for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The
# first two dimensions are the patch size, the next is the number of input channels,
# and the last is the number of output channels. We will also have a bias vector
# with a component for each output channel.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# We then convolve x_image with the weight tensor, add the bias, apply the ReLU function,
# and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# -----------Second convolutional layer-----------
# Second layer: 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Output of logistic regression model
# y = tf.matmul(x, W) + b

# -----------Densely connected hidden layer between convolution and output-----------
# Each input to the densely connected layer represents a pixel in the original image.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# Reshape the tensor from the pooling layer into a batch of vectors, multiply by a
# weight matrix, add a bias, and apply a ReLU
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# # -----------Dropout-----------
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# -----------Densely connected readout layer between convolution and output-----------
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Placeholder for testing labels
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

trainValues = []
validValues = []

# Start time
from time import time
t0 = time()

numRounds = 20000
batchSize = 50

for i in range(numRounds):
    batch = mnist.train.next_batch(batchSize)
    fd = feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0}
    train_accuracy = accuracy.eval(session=sess, feed_dict=fd)
    if i%10 == 0:
        validValues.append(train_accuracy)
    if i%250 == 0:
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_accuracy = accuracy.eval(session=sess, feed_dict=fd)
    if i%10 == 0:
        trainValues.append(train_accuracy)
# Test trained model

# Print training time
print ("training time:", "\t\t", round(time()-t0, 3), "s")

print("test accuracy %g"%accuracy.eval(feed_dict={
                                       x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))

plt.plot([1-x for x in trainValues], 'b', alpha=.5, linewidth=2.0)
plt.plot([1-x for x in validValues], 'g', alpha=.5, linewidth=2.0)

plt.xlabel("Tens of batches of 50 images for stochastic gradient descent")
plt.title("MNIST performance of a deep convolutional network")
plt.ylabel("Error Rate")


plt.legend(('Training Error', 'Validation Error'), loc='upper right')
#legend = plt.legend(loc='upper center', shadow=True)


plt.show()