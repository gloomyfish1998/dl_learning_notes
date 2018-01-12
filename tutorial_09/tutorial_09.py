# 神经网络激活函数 演示
# sigmoid relu激活函数使用，使用高斯分布生成数组

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size= 50
a1 = tf.Variable(tf.random_normal([1, 1]))
b1 = tf.Variable(tf.random_uniform([1, 1]))
a2 = tf.Variable(tf.random_normal([1, 1]))
b2 = tf.Variable(tf.random_uniform([1, 1]))

# 高斯分布随机数， 均值与方差是0.1
x = np.random.normal(2, 0.1, 500)[:,np.newaxis]
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

sigmod_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmod_activation, 0.75)))
loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

my_optimizer = tf.train.GradientDescentOptimizer(0.01)
train_sigmod = my_optimizer.minimize(loss1)
train_relu = my_optimizer.minimize(loss2)

print("\n try to optimzier sigmod and relu to 0.75")
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    loss_vec_sigmod = []
    loss_vec_relu = []
    for i in range(2000):
        rand_index = np.random.choice(len(x), size=batch_size)
        v_xvals = x[[rand_index]]
        sess.run(train_sigmod, feed_dict={x_data:v_xvals})
        sess.run(train_relu, feed_dict={x_data:v_xvals})

        loss_vec_sigmod.append(sess.run(loss1, feed_dict={x_data:v_xvals}))
        loss_vec_relu.append(sess.run(loss2, feed_dict={x_data:v_xvals}))

        sigmod_output = np.mean(sess.run(sigmod_activation, feed_dict={x_data:v_xvals}))
        relu_output = np.mean(sess.run(relu_activation, feed_dict={x_data:v_xvals}))

        if(i+1)%50 == 0:
            print("sigmod : %s, relu : %s", str(np.mean(sigmod_output)), str(np.mean(relu_output)))

    plt.plot(loss_vec_sigmod, "k--", label="Sigmoid Activation")
    plt.plot(loss_vec_relu, "r--", label="Relu Activation")
    plt.ylim([0, 1.0])
    plt.ylabel("Loss")
    plt.xlabel("Generation")
    plt.title("Loss per Generation")
    plt.legend(loc='upper right')
    plt.show()




