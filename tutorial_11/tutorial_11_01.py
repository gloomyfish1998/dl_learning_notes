# 实现神经网络不同层
#---------------------------------------
#
# 常见的层类型有：
#  (1) 卷积层
#  (2) 激活层
#  (3) 最大池化层
#  (4) 全连接层
#
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_size = 25
conv_size = 5
maxpool_size = 5
stride_size = 1 # 步长

seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)

data_1d = np.random.normal(size=data_size)

# placeholder
x_input_1d = tf.placeholder(shape=[data_size], dtype=tf.float32)

# ----- Convolution ----- #
def conv_layer_1d(input_1d, my_filter, stride):
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    convolution_output = tf.nn.conv2d(input=input_4d, filter=my_filter, strides=[1, 1, stride, 1], padding="VALID")
    convolution_output_1d = tf.squeeze(convolution_output) # 删除所有大小是1的维度
    return(convolution_output_1d)


my_filter = tf.Variable(tf.random_normal(shape=[1, conv_size, 1, 1]))
my_conv_output = conv_layer_1d(x_input_1d, my_filter, stride=stride_size)

def activation(input_1d):
    return (tf.nn.relu(input_1d))

my_activation_output = activation(my_conv_output)

def max_pool(input_1d, width, stride):
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1], strides=[1, 1, stride, 1], padding="VALID")

    pool_output_1d = tf.squeeze(pool_output)
    return (pool_output_1d)

my_max_pool_output = max_pool(my_activation_output, width=maxpool_size, stride=stride_size)


def fully_connected(input_layer, num_outputs):
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])
    input_layer_2d = tf.expand_dims(input_layer, 0)
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
    full_output_1d = tf.squeeze(full_output)
    return (full_output_1d)

my_full_output = fully_connected(my_max_pool_output, 5)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print("convolution layer : ", sess.run(my_conv_output, feed_dict={x_input_1d:data_1d}))
print("activation layer : ", sess.run(my_activation_output, feed_dict={x_input_1d:data_1d}))
print("max_pool_output : ", sess.run(my_max_pool_output, feed_dict={x_input_1d:data_1d}))
print("full connection : ", sess.run(my_full_output, feed_dict={x_input_1d:data_1d}))



