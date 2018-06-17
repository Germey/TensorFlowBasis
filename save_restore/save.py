import tensorflow as tf

W = tf.Variable([[2, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
b = tf.Variable([[2, 2, 3]], dtype=tf.float32, name='biases')

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save_path = saver.save(sess, 'model/test.ckpt')
    print('Saved to path:', save_path)
