# 若知四海皆兄弟，何处相逢非故人
# mnist data set, handwriting image recognition
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

# Import data
data_dir = 'MNIST_data/'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
W = tf.Variable(tf.random_normal(shape=[784, 10], dtype=tf.float32))
b = tf.Variable(tf.random_normal(shape=[10], dtype=tf.float32))

y = tf.add(tf.matmul(x, W), b)

y_target = tf.placeholder(shape=[None, 10], dtype=tf.float32)
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_target))

my_optimizer = tf.train.GradientDescentOptimizer(0.5)
train_step = my_optimizer.minimize(cross_entropy_loss)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_size = 100
vec_loss = []
for i in range(4000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x:batch_x, y_target:batch_y})
    curr_loss = sess.run(cross_entropy_loss, feed_dict={x:batch_x, y_target:batch_y})
    vec_loss.append(curr_loss)

    if(i+1)%100 == 1:
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_target: mnist.test.labels}))

plt.plot(vec_loss, 'k-', label="Softmax cross entropy Loss")
plt.xlabel("Generation")
plt.legend(loc='upper right')
plt.title("损失函数-损失下降曲线")
plt.ylabel("Loss")
plt.show()


