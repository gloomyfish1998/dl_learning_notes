# 损失函数演示
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_input = np.linspace(-1., 1., 500)
target = tf.constant(0.)

sess = tf.Session()

# (x - y)^2
l2_loss = tf.square(target - x_input)
l2_loss_output = sess.run(l2_loss)
print(l2_loss_output)

# abs(x - y)
l1_loss = tf.abs(target-x_input)
l1_loss_output = sess.run(l1_loss)
print(l1_loss_output)

# Pseudo-Huber loss
# L = delta^2 * (sqrt(1 + ((pred - actual)/delta)^2) - 1)
delta1 = tf.constant(.25)
phuber_1_y = tf.multiply(tf.square(delta1), tf.sqrt(1.+tf.square((target-x_input)/delta1)) - 1.)
phuber_1_out = sess.run(phuber_1_y)
print(phuber_1_out)

delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_input)/delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)

plt.plot(x_input, l2_loss_output, 'b-', label='L2 Loss')
plt.plot(x_input, l1_loss_output, 'r--', label='L1 Loss')
plt.plot(x_input, phuber_1_out, 'k-.', label='P-Huber Loss (0.25)')
plt.plot(x_input, phuber2_y_out, 'g:', label='P-Huber Loss (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()


# TF中的损失函数
print(" ------------------- Loss Function in TensorFlow ------------------- ")
x_input = tf.linspace(-3., 5., 500)
target = tf.constant(1.0)
targets = tf.fill([500,], 1.)
print(sess.run(targets))

# Hinge loss
# Use for predicting binary (-1, 1) classes
# L = max(0, 1 - (pred * actual))
hinge_y = tf.maximum(0., 1.-tf.multiply(target, x_input))
hinge_y_out = sess.run(hinge_y)

# Cross entropy loss
# L = -actual * (log(pred)) - (1-actual)(log(1-pred))
xentropy_y_vals = - tf.multiply(target, tf.log(1-x_input)) - tf.multiply((1-target), tf.log(1-x_input))
xentropy_y_out = sess.run(xentropy_y_vals)

x_val_input = tf.expand_dims(x_input, 1)
target_input = tf.expand_dims(targets, 1)
xentropy_sigmoid_y_vals = tf.nn.softmax_cross_entropy_with_logits(logits=x_val_input, labels=target_input)
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

# Weighted (softmax) cross entropy loss
# L = -actual * (log(pred)) * weights - (1-actual)(log(1-pred))
# or
# L = (1 - pred) * actual + (1 + (weights - 1) * pred) * log(1 + exp(-actual))
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_input, targets, weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

x_array= sess.run(x_input)
plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xentropy_sigmoid_y_out, 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(x_array, xentropy_weighted_y_out, 'g:', label='Weighted Cross Entropy Loss (x0.5)')
plt.ylim(-1.5, 3)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()


# Softmax entropy loss
# L = -actual * (log(softmax(pred))) - (1-actual)(log(1-softmax(pred)))
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits,
                                                           labels=target_dist)
print(sess.run(softmax_xentropy))



