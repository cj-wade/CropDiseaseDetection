import tensorflow as tf

noise = tf.random_normal(shape=[10, 10, 3], mean=0.0, stddev=0.25, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(noise)
    print(noise)