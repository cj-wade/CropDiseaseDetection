import tensorflow as tf
from picture.draw_picture import draw
from nets.lenet import inference, accuracy
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from time import *
import os

batch_size = 32
num_classes = 10
img_w = 28
img_h = 28
lr = 0.01
learning_decay = 0.94
decay_steps = 2000
train_epoch = 10001
display_step = 100
save_step = 2000
MODEL_SAVE_PATH = "D:\Python\PycharmProjects\CropDiseaseDetection\mnist_model\\"
MODEL_NAME = 'mnist_lenet_model.ckpt'

if __name__ == '__main__':
    begin_time = time()
    mnist = input_data.read_data_sets("D:\Python\PycharmProjects\CropDiseaseDetection\mnist_data", one_hot=True)
    x = tf.placeholder(tf.float32, [None, img_w, img_h, 1])
    y = tf.placeholder(tf.float32, [None, num_classes])

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # dataset = get_train_dataset()
    # iterator = dataset.make_initializable_iterator()
    # image_batch, label_batch = iterator.get_next()
    # label_batch = tf.one_hot(label_batch, depth=num_classes)

    # lenet
    pred = inference(x)

    # 损失值
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    # 学习率设置
    learning_rate = tf.train.exponential_decay(
        lr, global_step, decay_steps=decay_steps, decay_rate=learning_decay, staircase=True)
    # 训练步骤
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    # 准确率计算
    accuracy = accuracy(pred, y)

    # 储存loss 与 acc值，用于作图
    fig_loss = []
    fig_acc = []

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # sess.run(iterator.initializer)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        print('开始训练')
        for i in range(train_epoch):
            image, label = mnist.train.next_batch(batch_size)
            image = np.reshape(image, (batch_size, img_w, img_h, 1))
            l, _, acc = sess.run([loss, train_step, accuracy],
                                 feed_dict={x: image, y: label})
            if i % display_step == 0:
                fig_loss.append(l)
                fig_acc.append(acc)
                print('epoch:', '%04d' % (i + 1), 'loss:' '{:.4f}'.format(l), 'acc:', '%.4f' % acc)

            if i % save_step == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        # 计算训练用时
        end_time = time()
        run_time = end_time - begin_time
        m, s = divmod(run_time, 60)
        h, m = divmod(m, 60)
        print("训练用时：%d:%02d:%02d" % (h, m, s))

        # 画出损失值Loss 与 准确率acc的迭代趋势图
        draw(fig_loss, fig_acc)
