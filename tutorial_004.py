import tensorflow as tf
import numpy as np
# 矩阵计算
sess = tf.Session()

# Identity matrix
identify = tf.diag([1.0, 1.0, 1.0])
print(sess.run(identify))

# 2x3 matrix
A = tf.truncated_normal([2, 3], dtype=tf.float32)
print(sess.run(A))

# 2x3 matrix with value
B = tf.fill([2, 3], 5)
print(sess.run(B))

# 3x2 random uniform matrix
C = tf.random_normal([3, 2], dtype=tf.float32)
print("C = \n", sess.run(C))
print("C = \n", sess.run(C))

# Matrix Multiplication
print(sess.run(tf.matmul(A, C)))


# 矩阵转置
D = tf.fill([5, 5], 4)
print(sess.run(D))
print(sess.run(tf.transpose(D)))

# 矩阵特征值
E = tf.random_normal([2, 2])
result = sess.run(E)
print("E : \n", result)
print("Eigen value : \n", sess.run(tf.self_adjoint_eigvals(result)))
print("Eigen Vector : \n", sess.run(tf.self_adjoint_eig(result)))
