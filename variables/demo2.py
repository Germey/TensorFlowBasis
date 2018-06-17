import tensorflow as tf

count = tf.Variable(3, name='count', dtype=tf.float32)
print(count)

# 不会提示重复，强制新建
count = tf.Variable(4, name='count', dtype=tf.float32)
print(count)

# 前面没有用过 get_variable() 创建，不会重复
count = tf.get_variable('count', shape=[], dtype=tf.float32)
print(count)
# 前面用过 get_variable() 创建，提示重复报错，需要 reuse
# count = tf.get_variable('count', shape=[])
# print(count)

with tf.variable_scope('', reuse=tf.AUTO_REUSE):
    count = tf.get_variable('count')
    print(count)
