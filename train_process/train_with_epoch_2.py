import sys
cur = '/mnt/PyCharm_Project_1'
sys.path.append(cur)
print("success")
import tensorflow as tf
import keras
from picture.draw_picture import draw
from data_generation.data_generate import get_train_dataset, get_test_dataset, batch_size
from nets.lenet import accuracy
from nets.net_choice import net
from time import *
from train_process.checkmate import BestCheckpointSaver
import math
from train_process.log import log_creater

num_classes = 61
img_w = img_h = 224
lr = 0.0001
learning_decay = 0.94
decay_steps = 992
train_epoch = 80
trainSet_size = 31718
testSet_size = 4540
save_path = ["/mnt/PyCharm_Project_1/crop_model/Lenet/strong",
             "/mnt/PyCharm_Project_1/crop_model/AlexNet_224",
             "/mnt/PyCharm_Project_1/crop_model/Lenet/CSB",
             "/mnt/PyCharm_Project_1/crop_model/test/Re_CBAM_Alexnet",
             "/mnt/PyCharm_Project_1/crop_model/InResNet_V2/strong",
             "/mnt/PyCharm_Project_1/crop_model/test/Re_CBAM_Alexnet",
             "/mnt/PyCharm_Project_1/crop_model/test/Vgg",
             "/mnt/PyCharm_Project_1/crop_model/test/SE_Vgg",
             "/mnt/PyCharm_Project_1/crop_model/new/Densenet",
             "/mnt/PyCharm_Project_1/crop_model/test/CBAM_Densenet"]
net_id = 1
MODEL_SAVE_PATH = save_path[net_id]
TB_SAVE_PATH = "/mnt/PyCharm_Project_1/crop_model/AlexNet_224/TB"
# LOG_PATH = "/mnt/PyCharm_Project_1/crop_model/AlexNet_150/log"
# log_creater(LOG_PATH)


# 计算每个epoch的loss和acc，并保存历史前三最优模型和tensorboard
def cal_accuracy_loss(cal_image_batch, cal_label_batch, cal_label_batch_nohot, dataset_size, epoch, x, y, loss,
                      is_train="Train"):
    results = []
    labels = []
    losses = []
    for j in range(math.ceil(dataset_size / batch_size)):
        image, label, label_nohot = sess.run([cal_image_batch, cal_label_batch, cal_label_batch_nohot])
        pred, l = sess.run([prediction, loss], feed_dict={x: image, y: label, training_flag: False})
        results.extend(pred)
        labels.extend(label_nohot)
        losses.append(l)
    # 计算准确率
    correct = [float(y == y_) for (y, y_) in zip(results, labels)]
    acc = sum(correct) / len(correct)
    # 计算平均loss
    loss = sum(losses) / len(losses)

    print(is_train, "-", 'epoch:', '%04d' % (epoch), 'loss:' '{:.4f}'.format(loss), 'acc:', '%.4f' % acc)

    # 在session内定义的变量使用如下方法进行可视化
    summary = tf.Summary()
    if is_train == "Train":
        summary.value.add(tag='accuracy_train', simple_value=acc)
        summary.value.add(tag='loss_train', simple_value=loss)
    else:
        summary.value.add(tag='accuracy_test', simple_value=acc)
        summary.value.add(tag='loss_test', simple_value=loss)

        best_ckpt_saver.handle(acc, sess, global_step)

    summary_writer.add_summary(summary, epoch)

    return


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
    image_batch, label_batch_nohot = train_iterator.get_next()
    label_batch = tf.one_hot(label_batch_nohot, depth=num_classes)

    # 加载验证集
    val_dataset = get_test_dataset()
    val_iterator = val_dataset.make_initializable_iterator()
    val_image_batch, val_label_batch_nohot = val_iterator.get_next()
    val_label_batch = tf.one_hot(val_label_batch_nohot, depth=num_classes)

    # 定义预测结果
    training_flag = tf.placeholder(tf.bool)
    pred = net(net_id, x, is_train=training_flag)
    prediction = tf.argmax(pred, axis=-1, output_type=tf.int32)

    # 损失值
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

    # 学习率设置
    learning_rate = tf.train.exponential_decay(
        lr, global_step, decay_steps=decay_steps, decay_rate=learning_decay, staircase=True)
    # 训练步骤
    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step)
    # train_step = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss, global_step)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

    # 准确率计算
    accuracy = accuracy(pred, y)

    # 用于BN时，使BN操作在train_op之前
    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies([train_step, update_ops]):
        train_op = tf.no_op(name='train')

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
        # # 多线程写tensorboard日志文件
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess, coord)

        print('开始测试')
        start = time()
        # 计算每个epoch的loss和acc
        cal_accuracy_loss(image_batch, label_batch, label_batch_nohot, trainSet_size, 0, x, y, loss)  # 训练集的loss和acc
        cal_accuracy_loss(val_image_batch, val_label_batch, val_label_batch_nohot, testSet_size, 0, x, y, loss,
                          is_train="Test")  # 验证集的loss和acc
        print("开始训练")
        for i in range(train_epoch):
            train_time = []
            # 一个epoch的训练过程
            for j in range(math.ceil(trainSet_size / batch_size)):
                image, label = sess.run([image_batch, label_batch])
                train_start = time()
                sess.run([train_step], feed_dict={x: image, y: label, training_flag: True})
                train_time.append(time() - train_start)

            # 打印每个batch的训练时间
            avg_traintime_per_step = sum(train_time) / math.ceil(trainSet_size / batch_size)
            print("平均每个batch训练时间：", avg_traintime_per_step)

            # 计算每个epoch的loss和acc
            cal_accuracy_loss(image_batch, label_batch, label_batch_nohot, trainSet_size, i + 1, x, y,
                              loss)  # 训练集的loss和acc
            cal_accuracy_loss(val_image_batch, val_label_batch, val_label_batch_nohot, testSet_size, i + 1, x, y, loss,
                              is_train="Test")  # 验证集的loss和acc

        summary_writer.close()

        # 计算训练用时
        end_time = time()
        run_time = end_time - begin_time
        m, s = divmod(run_time, 60)
        h, m = divmod(m, 60)
        print("训练用时：%d:%02d:%02d" % (h, m, s))
