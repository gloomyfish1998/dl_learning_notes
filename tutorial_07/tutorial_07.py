# Logistic Regression
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve logistic regression.
# y = sigmoid(Ax + b)
#
# We will use the low birth weight data, specifically:
#  y = 0 or 1 = low birth weight
#  x = demographic and medical history data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

# name of data file
birth_weight_file = 'D:/tensorflow/dataset/Logistic_data.csv'

birth_data = []
with open(birth_weight_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]

# Pull out target variable
y_vals = np.array([x[0] for x in birth_data])
# Pull out predictor variables (not id, not target, and not birthweight)
x_vals = np.array([x[1:8] for x in birth_data])

seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)

# split dataset as train and test
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

batch_size = 45;
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# line regression
A = tf.Variable(tf.random_normal([7, 1]))
b = tf.Variable(tf.random_normal([1, 1]))

model_output = tf.add(tf.matmul(x_data, A), b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# try to declare train method
optimizer = tf.train.GradientDescentOptimizer(0.05)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # actual prediction
    prediction = tf.round(tf.sigmoid(model_output))
    prediction_correct = tf.cast(tf.equal(y_target, prediction), tf.float32)
    accuracy = tf.reduce_mean(prediction_correct)

    # train stage
    loss_vec = []
    train_acc = []
    test_acc = []
    for i in range(3000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose(y_vals_train[rand_index]).reshape([batch_size, 1])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        temp_acc_train = sess.run(accuracy, feed_dict={x_data : x_vals_train, y_target : np.transpose(y_vals_train).reshape([len(x_vals_train), 1])})
        train_acc.append(temp_acc_train)

        temp_acc_test = sess.run(accuracy, feed_dict={x_data : x_vals_test, y_target : np.transpose(y_vals_test).reshape([len(x_vals_test), 1])})
        test_acc.append(temp_acc_test)

        if (i+1)%300 == 0:
            print("Loss : " + str(temp_loss))
            print("Train Accuracy : " + str(temp_acc_train))
            print("Test Accuracy : " + str(temp_acc_test))

    # plot data
    plt.plot(loss_vec, "k--")
    plt.title("Cross Entropy Loss per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Cross Entropy")
    plt.show()

    # Plot train and test accuracy
    plt.plot(train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(test_acc, 'r--', label='Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()




