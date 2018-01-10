### 二次多项式逻辑回归求解
使用梯度下降算法 与 最小化reduce_mean作为损失函数求解逻辑回归问题
使用feed_dict时候要注意，不接受tensor类型数据作为输入，接受数据类型包括
- np.array
- Scalar
- array
从sess.run中fetch数据时候要注意，fetch多个数据使用[]来表示

上述详细情况均可参见代码

### 创建随机数使用
tf.linspace 或者 np.linspace 二者的不同之处是
前面创建的数据不能作为placeholder的输入！！！
