import tensorflow as tf
import matplotlib.pyplot as plt

# 计数器，操作数
counter = tf.Variable(0, name='counter')
temp = tf.add(counter, 1)
updated = tf.assign(counter, temp)

# 线性回归
W = tf.Variable(.3, dtype=tf.float32)
b = tf.Variable(-.3, dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)
line_model = W*x+b

# 定义线性回归数据
x_train = [1, 2, 3, 4, 5, 6, 7, 8]
y_train = [4, 6, 8, 10, 11, 14, 16, 18]
# 全局初始化
init = tf.global_variables_initializer()


# 绘制直线
def drawFitLine(a, b):
    plt.plot(x_train, y_train, label="origin-line")
    new_y = []
    for i in range(len(x_train)):
        new_y.append(a*x_train[i]+b)
    plt.plot(x_train, new_y, label="fit-line")
    plt.legend()
    plt.show()

with tf.Session() as sess:
    sess.run(init)
    print("init counter : ", sess.run(counter))
    for i in range(100):
        sess.run(updated)
    print("Final counter : ", sess.run(counter))

    square_model = tf.square(line_model - y)
    loss = tf.reduce_mean(square_model)
    print(sess.run(loss, {x:x_train, y:y_train}))

    # 使用梯度下降求解线性回归问题
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    for i in range(1000):
        sess.run(train, {x:x_train, y:y_train})
        print("current loop : ", i)
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
    print("W : %s, b : %s, loss : %s", curr_W, curr_b, curr_loss)
    drawFitLine(curr_W, curr_b)