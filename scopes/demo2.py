import tensorflow as tf

with tf.variable_scope('a_name_scope'):
    initializer = tf.constant_initializer(value=1)
    # variable_scope 对两种创建方式都是起作用的
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name)  # var1:0
    print(sess.run(var1))  # [ 1.]
    print(var2.name)  # a_name_scope/var2:0
    print(sess.run(var2))  # [ 2.]
    print(var21.name)  # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]
    print(var22.name)  # a_name_scope/var2_2:0
    print(sess.run(var22))  # [ 2.20000005]
