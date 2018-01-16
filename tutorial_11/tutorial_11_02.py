import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

conv_size = 2
conv_stride_size = 2
maxpool_size = 2
maxpool_stride_size = 1

seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)

gray_image_2d = cv.imread("D:/javaopencv/girl.png", cv.IMREAD_GRAYSCALE)
h, w = gray_image_2d.shape
print("height : %s width : %s channels : %s", str(h), str(w))
x_input_2d = tf.placeholder(shape=[h, w], dtype=tf.float32)


# define convolution layer
def conv_layer_2d(input_2d, my_filter, stride_size):
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,
                                    stride_size, stride_size, 1], padding="VALID")
    conv_output_2d = tf.squeeze(convolution_output)
    return (conv_output_2d)


my_filter = tf.Variable(tf.random_normal(shape=[conv_size, conv_size, 1, 1]))

my_convolution_output = conv_layer_2d(x_input_2d, my_filter, stride_size=conv_stride_size)


# define activation
def activation(input_1d):
    return (tf.nn.relu(input_1d))


my_activation_output = activation(my_convolution_output)


# max pool -----------------------------
def max_pool(input_2d, width, height, stride):
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1], strides=[1, stride, stride, 1], padding="VALID")

    pool_output_2d = tf.squeeze(pool_output)
    return (pool_output_2d)


my_max_pool_output = max_pool(my_activation_output, width=maxpool_size, height=maxpool_size, stride=maxpool_stride_size)


# full - connection - layer
def fully_connected(input_layer, num_outputs):
    flat_input = tf.reshape(input_layer, [-1])
    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])

    input_2d = tf.expand_dims(flat_input, 0)
    full_output = tf.add(tf.matmul(input_2d, weight), bias)
    full_output_2d = tf.squeeze(full_output)
    return (full_output_2d)


my_full_output = fully_connected(my_max_pool_output, 5)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print("convolution layer : \n", sess.run(my_convolution_output, feed_dict={x_input_2d:gray_image_2d}))

print("activation layer : \n", sess.run(my_activation_output, feed_dict={x_input_2d:gray_image_2d}))

print("max_pool layer : \n", sess.run(my_max_pool_output, feed_dict={x_input_2d:gray_image_2d}))

print("full connection layer : \n", sess.run(my_full_output, feed_dict={x_input_2d:gray_image_2d}))








