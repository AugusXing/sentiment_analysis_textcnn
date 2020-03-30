#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File         :tmp_test.py
@Copyright    :AugusXing
@Date         :2020-03-30 
"""
import numpy as np
import tensorflow as tf

tmp = [
    [[1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4]],
    [[1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4]]
     ]
b = np.array(tmp)
t3 = tf.expand_dims(b, -1)
# print(t3.shape)
bias = [1, 2, 3]
input = tf.Variable(tf.random_normal([1,3,3,1]))
filter = tf.Variable(tf.random_normal([1,3,1,3]))
op1 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')
op2 = tf.nn.bias_add(op1, bias)
op3 = tf.nn.relu(op2, name="relu")
op4 = tf.nn.max_pool(
    op3,
    ksize=[1, 3, 1, 1],
    strides=[1, 1, 1, 1],
    padding='VALID',
    name="pool")
pooled_outputs = []
pooled_outputs.append(op4)
pooled_outputs.append(op4)
h_pool = tf.concat(pooled_outputs, 3)


if __name__ == "__main__":
    # https://www.jianshu.com/p/903291573b49
    # print(b.shape)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(op1))
        print(op1.shape)
        print(sess.run(op2))
        print(op2.shape)
        print(sess.run(op3))
        print(op3.shape)
        print(sess.run(op4))
        print(op4.shape)
        print(sess.run(pooled_outputs))
        print(h_pool.shape)
        print(sess.run(h_pool))


