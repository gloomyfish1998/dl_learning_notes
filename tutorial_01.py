import tensorflow as tf;

#加法操作
node1 = tf.constant(3.3, dtype=tf.float32)
node2 = tf.constant(4.8, dtype=tf.float32)
result1 = tf.add(node1, node2)

#乘法操作
node3 = tf.constant([[3.2, 3.8]], dtype=tf.float32)
node4 = tf.constant([[4.5], [5.5]], dtype=tf.float32)
result2 = tf.matmul(node3, node4)

#除法操作
node5 = tf.constant([3.2, 3.8], dtype=tf.float32)
node6 = tf.constant([4.5, 5.5], dtype=tf.float32)
result3 = tf.divide(node5, node6)

#减法操作
node7 = tf.constant([15, 3], dtype=tf.float32)
node8 = tf.constant([4, 5], dtype=tf.float32)
result4 = tf.subtract(node7, node8)

#混合运算
node9 = tf.constant([12, 14], dtype=tf.float32)
node10 = tf.constant([8, 10], dtype=tf.float32)
node11 = tf.constant([3, 5], dtype=tf.float32)
m1 = tf.multiply(node9, node10)
m2 = tf.subtract(m1, node11)
m3 = tf.add(m2, 3)

sess = tf.Session()
print("\n")
print(sess.run([node1, node2]))
print("result : ", sess.run(result1))

print("\n")
print(sess.run([node3, node4]))
print("result : ", sess.run(result2))

print("\n")
print(sess.run([node5, node6]))
print("result : ", sess.run(result3))

print("\n")
print(sess.run([node7, node8]))
print("result : ", sess.run(result4))

print("\n")
print("result : ", sess.run(m1))
print("result : ", sess.run(m2))
print("result : ", sess.run(m3))

#计算线性方程
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
line_model = W*x+b
init = tf.global_variables_initializer();
sess.run(init)
print("\n");
print("line model \n")
print(sess.run(line_model, {x:[1, 2, 3, 4]}))



