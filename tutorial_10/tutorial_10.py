# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
# 神经网络 隐藏层使用
# 该数据集包含了5个属性：
# Sepal.Length（花萼长度），单位是cm;
# Sepal.Width（花萼宽度），单位是cm;
# Petal.Length（花瓣长度），单位是cm;
# Petal.Width（花瓣宽度），单位是cm;
# - Iris Setosa（山鸢尾）
# - Iris Versicolour（杂色鸢尾）
# - 以及Iris Virginica(维吉尼亚鸢尾)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
x_data = iris.data
y_data = iris.target

# make it reproduce
seed = 2;
tf.set_random_seed(seed)
np.random.seed(seed)

train_indices = np.random.choice(len(x_data), round(len(x_data)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_data))) - set(train_indices)))
x_vals_train = x_data[train_indices]
x_vals_test = x_data[test_indices]
y_vals_train = y_data[train_indices]
y_vals_test = y_data[test_indices].reshape([len(test_indices),1])


# 归一化
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

x_input = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 隐藏层
hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[4, hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
hidden_layer = tf.add(tf.matmul(x_input, A1), b1)

# 输出层
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))
output_layer = tf.add(tf.matmul(hidden_layer, A2), b2)

# 损失函数
loss = tf.reduce_mean(tf.square(y_target-output_layer))

# 训练方法 - 梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.005)
train_step = optimizer.minimize(loss)

batch_size = 50
vec_loss = []
test_loss = []
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # 计算识别分类准确率
    prediction = tf.round(tf.nn.relu(output_layer)) # relu激活函数
    prediction_correct = tf.cast(tf.equal(y_target, prediction), tf.float32)
    accuracy = tf.reduce_mean(prediction_correct)

    for i in range(1500):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose(y_vals_train[rand_index]).reshape([batch_size, 1])
        sess.run(train_step, feed_dict={x_input:rand_x, y_target:rand_y})

        temp_loss = sess.run(loss, feed_dict={x_input:rand_x, y_target:rand_y})
        vec_loss.append(np.sqrt(temp_loss))

        test_temp_loss = sess.run(loss, feed_dict={x_input:x_vals_test, y_target:y_vals_test})
        test_loss.append(np.sqrt(test_temp_loss))


        if(i+1)%100 == 0:
            print("Generation : " + str(i+1) + " .Loss : " + str(temp_loss))
            print("Train Accuracy : ", sess.run(accuracy, feed_dict={x_input:x_vals_train, y_target:np.transpose(y_vals_train).reshape([len(y_vals_train), 1])}))

    # plot MSE
    plt.plot(vec_loss, "k--", label="Train Loss")
    plt.plot(test_loss, "r--", label="Test Loss")
    plt.title("Loss (MSE) per Generation")
    plt.legend(loc="upper right")
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.show()







