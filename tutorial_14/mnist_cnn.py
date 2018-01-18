#
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

data_dir = 'MNIST_data/'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
batch_size = 50

x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_ = tf.placeholder(shape=[None, 10], dtype=tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

# convolution neural network layer 1
conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.constant(0.1, shape=[32]))
conv1_output = tf.nn.conv2d(x_image, conv1_w, strides=[1, 1,1, 1], padding='SAME')
conv1_relu = tf.nn.relu(tf.add(conv1_output, conv1_bias))

# maxpool layer 1
maxpool_1_out = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# convolution neural network layer 2
conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.constant(0.1, shape=[64]))
conv2_output = tf.nn.conv2d(maxpool_1_out, conv2_w, strides=[1, 1, 1, 1], padding='SAME')
conv2_relu = tf.nn.relu(tf.add(conv2_output, conv2_bias))

# maxpool layer 2
maxpool_2_out = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# full connected layer 1 - fc1 7x7x64
w_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], dtype=tf.float32, stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_pool2_flat = tf.reshape(maxpool_2_out, [-1, 7*7*64])
output_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, w_fc1), b_fc1))

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(output_fc1, keep_prob)

# full connected layer 2 - fc2
w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], dtype=tf.float32, stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)

cross_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)
my_loss = tf.reduce_mean(cross_loss)

train_step = tf.train.AdamOptimizer(1e-4).minimize(my_loss)

correction_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
correction_prediction = tf.cast(correction_prediction, tf.float32)

vec_loss = []
train_acc = []
accuracy = tf.reduce_mean(correction_prediction)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(10000):
        batch = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if (i+1)%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            train_acc.append(train_accuracy)
            print('step %d, training accuracy %g' % ((i+1), train_accuracy))
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    curr_accuracy = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels, keep_prob: 0.5})
    print("Final accuracy : ", str(curr_accuracy))

# Plot train and test accuracy
plt.plot(train_acc, 'r--', label='Train Accuracy')
plt.title('Train Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
