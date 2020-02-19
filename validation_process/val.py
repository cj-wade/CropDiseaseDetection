from time import *
import tensorflow as tf
from data_generation.data_generate import get_test_dataset
from nets import lenet, alexnet, SE_Lenet
from train_process import train
import numpy as np
import tensorflow.contrib.slim as slim

num_classes = 61
img_w = 224
img_h = 224


def crop_class(id_class):
    if 0 <= id_class < 6:
        return 0, 0, 6
    if 6 <= id_class < 9:
        return 1, 6, 9
    if 9 <= id_class < 17:
        return 2, 9, 17
    if 17 <= id_class < 24:
        return 3, 17, 24
    if 24 <= id_class < 27:
        return 4, 24, 27
    if 27 <= id_class < 30:
        return 5, 27, 30
    if 30 <= id_class < 33:
        return 6, 30, 33
    if 33 <= id_class < 37:
        return 7, 33, 37
    if 37 <= id_class < 41:
        return 8, 37, 41
    if 41 <= id_class < 61:
        return 9, 41, 61


if __name__ == '__main__':
    # 程序开始
    begin_time = time()
    # 每次导入模型前，先重置计算图
    tf.reset_default_graph()
    with tf.get_default_graph().as_default():

        # 获得测试的batch
        dataset = get_test_dataset()
        iterator = dataset.make_initializable_iterator()
        image_batch, label_batch = iterator.get_next()
        label_batch = tf.one_hot(label_batch, depth=num_classes)

        # 定义预测结果
        if train.net_id == 0:  # lenet
            logit = lenet.inference(image_batch)
        elif train.net_id == 1:  # Alexnet
            with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
                logit, _ = alexnet.alexnet_v2(image_batch)
        elif train.net_id == 2:  # SE_lenet
            logit = SE_Lenet.inference(image_batch)
        else:
            print("未选择神经网络！")

        # 将预测概率矩阵转化为概率最高的项
        prediction = tf.argmax(logit, 1)

        # 初始化saver
        saver = tf.train.Saver()

        # 获取预测结果
        test_results = []
        test_labels = []

        with tf.Session() as sess:
            # 初始化全局变量、本地变量、dataset迭代器
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)

            # 自动通过checkpoint找到最新的模型
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                while True:
                    try:
                        # 进行预测
                        pred, label = sess.run([prediction, label_batch])
                        labels = np.argmax(label, 1)
                        
                        # 将每一批预测结果和标签加入Top1_列表
                        test_results.extend(pred)
                        test_labels.extend(labels)

                    except tf.errors.OutOfRangeError:
                        break

                # 计算测试集Top_1准确率
                correct = [float(y == y_) for (y, y_) in zip(test_results, test_labels)]
                accuracy = sum(correct) / len(correct)
                print("在第%s轮训练后，测试集Top_1准确率为：%g" % (global_step, accuracy))

                # 计算测试用时
                end_time = time()
                run_time = end_time - begin_time
                print("测试用时：", run_time)
