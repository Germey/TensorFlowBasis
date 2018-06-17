import tensorflow as tf

with tf.variable_scope('level1', reuse=tf.AUTO_REUSE) as level1:
    a = tf.get_variable('a', dtype=tf.float32, initializer=tf.constant(1, dtype=tf.float32))
    # level1.reuse_variables()
    print(a)
    # new a
    a = tf.Variable(1, dtype=tf.float32, name='a')
    print(a)
    # origin a
    a = tf.get_variable('a', dtype=tf.float32, shape=[])
    # new level
    with tf.variable_scope('level2', reuse=tf.AUTO_REUSE):
        # a = tf.get_variable('a', shape=[], dtype=tf.float32)
        # new level of b c
        b = tf.get_variable('b', shape=[], dtype=tf.float32)
        c = tf.get_variable('c', shape=[], dtype=tf.float32)
        
        with tf.variable_scope('level3') as level3:
            # new level3 d
            d = tf.get_variable('d', shape=[], dtype=tf.float32)

print(a)
print(b)
print(c)
print(d)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
