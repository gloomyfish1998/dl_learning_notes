# 半生有梦化飞烟，一事无成惊逝水
# MNIST neural network example
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

data_dir = 'MNIST_data/'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
batch_size = 200
x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# 神经网络第一层，激活函数sigmoid
W1 = tf.Variable(tf.random_normal(shape=[784, 200], dtype=tf.float32))
b1 = tf.Variable(tf.random_normal(shape=[200], dtype=tf.float32))
layer_1_output = tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1))

# 神经网络第二层，激活函数sigmoid
W2 = tf.Variable(tf.random_normal(shape=[200, 100], dtype=tf.float32))
b2 = tf.Variable(tf.random_normal(shape=[100], dtype=tf.float32))
layer_2_output = tf.nn.sigmoid(tf.add(tf.matmul(layer_1_output, W2), b2))

# 神经网络第三层，激活函数sigmoid
W3 = tf.Variable(tf.random_normal(shape=[100, 50], dtype=tf.float32))
b3 = tf.Variable(tf.random_normal(shape=[50], dtype=tf.float32))
layer_3_output = tf.nn.sigmoid(tf.add(tf.matmul(layer_2_output, W3), b3))

# 神经网络输出层，激活函数sigmoid
W4 = tf.Variable(tf.random_normal(shape=[50, 10], dtype=tf.float32))
b4 = tf.Variable(tf.random_normal(shape=[10], dtype=tf.float32))
y_output = tf.add(tf.matmul(layer_3_output, W4), b4)

# 损失函数与训练方法
my_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_output, labels=y_target))
# 梯度下降92%, 不加激活函数 87%
# my_optimizer = tf.train.GradientDescentOptimizer(0.5)
# 95% 准确率
my_optimizer = tf.train.AdamOptimizer(0.005)
train_step = my_optimizer.minimize(my_loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

vec_loss = []
test_acc = []
for i in range(4000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x:batch_x, y_target:batch_y})
    curr_loss = sess.run(my_loss, feed_dict={x:batch_x, y_target:batch_y})
    vec_loss.append(curr_loss)
    if(i+1)%100 == 0:
        curr_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_target: mnist.test.labels})
        test_acc.append(curr_accuracy)
        print("accuracy : " + str(curr_accuracy))

plt.plot(vec_loss, 'k-', label="Softmax cross entropy Loss")
plt.xlabel("Generation")
plt.legend(loc='upper right')
plt.title("损失函数-损失下降曲线")
plt.ylabel("Loss")
plt.show()

# Plot train and test accuracy
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()








