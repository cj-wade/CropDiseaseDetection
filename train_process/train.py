import sys
cur = '/mnt/PyCharm_Project_1'
sys.path.append(cur)
print("success")
import tensorflow as tf
# from keras import backend as K
# K.clear_session()
from picture.draw_picture import draw
from data_generation.data_generate import get_train_dataset, get_test_dataset
from nets.lenet import accuracy
from nets.net_choice import net
from time import *
import numpy as np
import os
from train_process.checkmate import BestCheckpointSaver
import tensorflow.contrib.slim as slim

num_classes = 61
img_w = img_h = 224
lr = 0.01
learning_decay = 0.94
decay_steps = 992
train_epoch = 40001
display_step = 100
val_step = 992
# save_step = 20000
save_path = ["/mnt/PyCharm_Project_1/crop_model/Lenet/normal",
             "/mnt/PyCharm_Project_1/crop_model/new/normal_Alexnet",
             "/mnt/PyCharm_Project_1/crop_model/Lenet/CSB",
             "/mnt/PyCharm_Project_1/crop_model/test/Re_CBAM_Alexnet",
             "D:\Python\PycharmProjects\crop_base\crop_model\InResNet_V2",
             "D:\Python\PycharmProjects\CropDiseaseDetection\crop_model",
             "/mnt/PyCharm_Project_1/crop_model/test/Vgg",
             "/mnt/PyCharm_Project_1/crop_model/test/SE_Vgg",
             "/mnt/PyCharm_Project_1/crop_model/new/Densenet",
             "/mnt/PyCharm_Project_1/crop_model/test/CBAM_Densenet"]
net_id = 4
MODEL_SAVE_PATH = save_path[net_id]
# MODEL_NAME = 'crop_test_InResNet_V2_model.ckpt'
TB_SAVE_PATH = "/mnt/PyCharm_Project_1/crop_model/Lenet/CSB/TB"



if __name__ == '__main__':
    val_acc = 0
    begin_time = time()
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, img_w, img_h, 3], name='x-input')
        y = tf.placeholder(tf.float32, [None, num_classes], name='y-input')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # 加载训练集
    train_dataset = get_train_dataset()
    train_iterator = train_dataset.make_initializable_iterator()
    image_batch, label_batch = train_iterator.get_next()
    label_batch = tf.one_hot(label_batch, depth=num_classes)

    # 加载验证集
    val_dataset = get_test_dataset()
    val_iterator = val_dataset.make_initializable_iterator()
    val_image_batch, val_label_batch_nohot = val_iterator.get_next()
    val_label_batch = tf.one_hot(val_label_batch_nohot, depth=num_classes)

    # 定义预测结果
    training_flag = tf.placeholder(tf.bool)
    # pred = densenet.DenseNet(x, training=training_flag).model
    pred = net(net_id, x, is_train=training_flag)
    prediction = tf.argmax(pred, axis=-1, output_type=tf.int32)

    # 损失值
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    # 学习率设置

    learning_rate = tf.train.exponential_decay(
        lr, global_step, decay_steps=decay_steps, decay_rate=learning_decay, staircase=True)
    # 训练步骤
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step)
    # train_step = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss, global_step)
    # train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

    # 准确率计算
    accuracy = accuracy(pred, y)

    # 准确率、损失值、学习速率加入TB
    train_acc_sum = tf.summary.scalar('accuracy_train', accuracy)
    # val_acc_sum = tf.summary.scalar('accuracy_val', val_acc)
    loss_sum = tf.summary.scalar('cross_entropy', loss)
    lr_sum = tf.summary.scalar('learning_rate', learning_rate)

    # 用于BN时，使BN操作在train_op之前
    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies([train_step, update_ops]):
        train_op = tf.no_op(name='train')

    # 将所有日志进行整合，便于初始化
    summary_train_op = tf.summary.merge([train_acc_sum, loss_sum, lr_sum])
    # summary_val_op = tf.summary.merge([val_acc_sum])
    # 储存loss 与 acc值，用于作图
    fig_loss = []
    fig_acc = []

    # GPU设置
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8  # GPU setting,最高不超过90%
    config.gpu_options.allow_growth = True  # 显存占用根据需求增长

    # saver = tf.train.Saver()
    best_ckpt_saver = BestCheckpointSaver(save_dir=MODEL_SAVE_PATH, num_to_keep=3)
    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(TB_SAVE_PATH, tf.get_default_graph())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        print('开始训练')
        start = time()
        train_time = []
        for i in range(train_epoch):
            image, label = sess.run([image_batch, label_batch])
            train_start = time()
            summary_str, l, _, acc = sess.run([summary_train_op, loss, train_step, accuracy],
                                              feed_dict={x: image, y: label, training_flag: True})
            train_time.append(time() - train_start)
            summary_writer.add_summary(summary_str, i)

            if i % display_step == 0:
                avg_traintime_per_step = sum(train_time) / display_step
                fig_loss.append(l)
                fig_acc.append(acc)
                train_time = []
                print('epoch:', '%04d' % (i + 1), 'loss:' '{:.4f}'.format(l), 'acc:', '%.4f' % acc)
                print("平均每个batch训练时间：", avg_traintime_per_step)

            # if i % save_step == 0:
            #     saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

            # 每训练完一轮数据集，测试一次，保存最优
            if i == 0 or ((i + 1) % val_step == 0) or (i == train_epoch - 1):
                test_results = []
                test_labels = []
                for j in range(142):
                    val_image, label = sess.run([val_image_batch, val_label_batch_nohot])
                    pred = sess.run([prediction], feed_dict={x: val_image, training_flag: False})
                    pred = [y for x in pred for y in x]
                    test_results.extend(pred)
                    test_labels.extend(label)
                correct = [float(y == y_) for (y, y_) in zip(test_results, test_labels)]
                acc_val = sum(correct) / len(correct)
                print("测试集准确率为：", acc_val)
                # 在session内定义的变量使用如下方法进行可视化
                summary = tf.Summary()
                summary.value.add(tag='accuarcy_val', simple_value=acc_val)
                summary_writer.add_summary(summary, i)
                best_ckpt_saver.handle(acc_val, sess, global_step)

            # if i % val_step == 0:
            #     val_image, val_label = sess.run([val_image_batch, val_label_batch])
            #     summary_str, val_acc = sess.run([summary_val_op, accuracy], feed_dict={x: val_image, y: val_label})
            #     # best_ckpt_saver.handle(val_acc, sess, global_step)
            #     summary_writer.add_summary(summary_str, i)
            #     print("测试集准确率为：", val_acc)

        summary_writer.close()

        # 计算训练用时
        end_time = time()
        run_time = end_time - begin_time
        m, s = divmod(run_time, 60)
        h, m = divmod(m, 60)
        print("训练用时：%d:%02d:%02d" % (h, m, s))

        # 画出损失值Loss 与 准确率acc的迭代趋势图
        draw(fig_loss, fig_acc)
