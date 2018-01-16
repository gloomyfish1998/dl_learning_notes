# Multiple Layer Neural Network
# We will illustrate how to use a Multiple
# Layer Network in TensorFlow
# 低出生率数据训练与预测
# 七个关键指标数据
# - AGE 母亲年龄
# - LWT 上次月经时候体重
# - RACE 人种
# - SMOKE 吸烟历史
# - PTL 早产史
# - HT 高血压史
# - UI 子宫易激惹
# - BWT 出生时体重(0 - 体重大于2500克， 1 - 体重小于2500克)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

birth_weight_file = 'D:/tensorflow/birth_weight.csv'

# 读取低出生率数据
birth_data = []
with open(birth_weight_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]
y_vals = np.array([x[8] for x in birth_data])
print(y_vals.shape)

cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])
print(x_vals.shape)

batch_size = 100
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
# 训练数据与测试数据
x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]

x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]


# 归一化
def normalize_col(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)


x_vals_train = np.nan_to_num(normalize_col(x_vals_train))
x_vals_test = np.nan_to_num(normalize_col(x_vals_test))


def init_weight(shape, std_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=std_dev))
    return weight


def init_bias(shape, std_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=std_dev))
    return (bias)


x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


def full_connected(input_layer, weight, bias):
    layer = tf.add(tf.matmul(input_layer, weight), bias)
    return (tf.nn.relu(layer))


# 第一层
weight_1 = init_weight([7, 25], std_dev=10.0)
bias_1 = init_bias([25], std_dev=10.0)
layer_1 = full_connected(x_data, weight_1, bias_1)

# 第二层
weight_2= init_weight([25, 10], std_dev=10.0)
bias_2 = init_bias([10], std_dev=10.0)
layer_2 = full_connected(layer_1, weight_2, bias_2)

# 第三层
weight_3 = init_weight(shape=[10, 3], std_dev=10.0)
bias_3 = init_bias(shape=[3], std_dev=10.0)
layer_3 = full_connected(layer_2, weight_3, bias_3)

# 输出层
weight_4 = init_weight(shape=[3, 1], std_dev=10.0)
bias_4 = init_bias(shape=[1], std_dev=10.0)
final_output = full_connected(layer_3, weight_4, bias_4)

loss = tf.reduce_mean(tf.abs(y_target - final_output))
my_optimizer = tf.train.AdamOptimizer(0.05)
train_step = my_optimizer.minimize(loss)

init = tf.global_variables_initializer()
loss_vec = []
test_loss = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})

        # fetch loss
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_loss.append(test_temp_loss)

        if(i+1)%100 == 0:
            print("Generation : " + str(i+1) + " .Loss : " + str(temp_loss))

    curr_output = sess.run(final_output, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    print(curr_output)
    sum = 0
    for k in range(len(curr_output)):
        if curr_output[k]>2500 and y_vals_test[k]>2500:
            sum += 1
    accuracy = np.float32(sum) / len(curr_output)
    print("accuracy : ", accuracy)

    plt.plot(loss_vec, "k-", label="Train Loss")
    plt.plot(test_loss, "r--", label="Test Loss")
    plt.title("Loss (MSE) per Generation")
    plt.legend(loc="upper right")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.show()
