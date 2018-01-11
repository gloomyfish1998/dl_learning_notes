# non-line regression
# 使用神经网络训练
# 使用tanh激活函数，x_data取值范围必须在[-1, 1]之间
# 使用前，首先需要对输入数据进行归一化

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-.8, .8, 500)[:,np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

#x_data = np.float32(x_data)
#y_data = np.float32(y_data)

x_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# neural network - layer 1, input
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
bais_L1 = tf.Variable(tf.zeros([1, 10]))
output_L1 = tf.matmul(x_input, Weight_L1) + bais_L1
result_L1 = tf.nn.tanh(output_L1)

# layer 2th
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
bais_L2 = tf.Variable(tf.zeros([1, 1]))
output_L2 = tf.matmul(result_L1, Weight_L2) + bais_L2
prediction = tf.nn.tanh(output_L2)

loss = tf.reduce_mean(tf.square(y_input - prediction))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train_step, feed_dict={x_input:x_data, y_input: y_data})
        if(i+1)%100 == 0:
            temp_loss = sess.run(loss, feed_dict={x_input: x_data, y_input: y_data})
            print("Loss : ", str(temp_loss))

    curr_prediction = sess.run(prediction, feed_dict={x_input:x_data})
    plt.figure()
    plt.title("non-line regression")
    plt.scatter(x_data, y_data)
    plt.xlabel("input data")
    plt.ylabel("prediction")
    plt.plot(x_data, curr_prediction, "r--", lw=2)
    plt.show()

