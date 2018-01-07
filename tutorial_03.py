import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# fetch and feed
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.constant(8.0)
add = tf.add(node1, node2)
m1 = tf.multiply(add, node3)

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
m2 = tf.multiply(a, b)


x_input = np.random.rand(100)
y_input = x_input*0.1+0.2

W = tf.Variable(0.1, dtype=tf.float32)
k = tf.Variable(-0.1, dtype=tf.float32)
y = W*x_input+k
# reduce_sum is bad!!! because iteration is bigger
# loss = tf.reduce_sum(tf.square(y_input - y))
loss = tf.reduce_mean(tf.square(y_input - y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result = sess.run([add, m1])
    print(result)
    print(sess.run(m2, feed_dict={a:[1, 2, 3, 4], b:[2, 4, 6, 8]}))
    for i in range(1000):
        sess.run(train)
        if i % 100 == 0:
            print(" step : ", i, sess.run([W, k, loss]))
    curr_W, curr_k, curr_loss = sess.run([W, k, loss])
    print("W : %s, k : %s, loss : %s", curr_W, curr_k, curr_loss)
    plt.plot(x_input, y_input, label="line")
    curr_y = []
    for i in range(len(x_input)):
        curr_y.append(x_input[i]*curr_W+curr_k)
    plt.plot(x_input, curr_y, label="fit-line")
    plt.legend()
    plt.show()