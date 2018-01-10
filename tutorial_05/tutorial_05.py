# 逻辑回归模拟 - 模拟二次函数，使用reduce_mean与梯度下降算法求解逻辑回归
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_input = np.linspace(-1., 1., 500)
y_input = x_input*x_input*.7 + .5*x_input+.3

X = tf.placeholder(dtype=tf.float32)
a = tf.Variable(-.3)
b = tf.Variable(.3)
c = tf.Variable(.1)
line_model = X*X*a + b*X+c

loss = tf.reduce_mean(tf.square(line_model - y_input))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)


def drawXYPlot(ydata):
    plt.plot(x_input, y_input, label="origin data")
    plt.plot(x_input, ydata, label="fit data")
    plt.legend()
    plt.show()


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(loss, feed_dict={X: x_input}))
    for i in range(2000):
        sess.run(train_step, feed_dict={X:x_input})
        if i % 100 == 0:
            print(sess.run(loss, feed_dict={X: x_input}))
    curr_a, curr_b, curr_c, y_data, curr_loss = sess.run([a, b, c, line_model, loss], feed_dict={X : x_input})
    print(" a : %s, b : %s, c : %s", curr_a, curr_b, curr_c)
    print("loss : ", curr_loss)
    drawXYPlot(y_data)
