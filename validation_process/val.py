# import sys
# cur = '/mnt/PyCharm_Project_1'
# sys.path.append(cur)
print("success")
from time import *
import tensorflow as tf
from data_generation.data_generate import get_test_dataset
from nets.net_choice import net
from train_process import checkmate, transfer_learning_train
import numpy as np
from heapq import nlargest
from validation_process.crop_disease_type import crop_class
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from nets import SE_inception_resnet_v2 as InRes_V2
import tensorflow.contrib.slim as slim

Model_SAVE_PATH = "/mnt/PyCharm_Project_1/crop_model/transfer_learning_InRes_V2/normal/"

num_classes = 61
img_w = 224
img_h = 224

if __name__ == '__main__':
    # 程序开始
    begin_time = time()
    # 每次导入模型前，先重置计算图
    tf.reset_default_graph()
    with tf.get_default_graph().as_default():

        # 定义网络输入
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, img_w, img_h, 3], name='x-input')
            y = tf.placeholder(tf.float32, [None, num_classes], name='y-input')

        # 获得测试的batch
        dataset = get_test_dataset(is_train=False)
        iterator = dataset.make_initializable_iterator()
        image_batch, label_batch_nohot = iterator.get_next()
        label_batch = tf.one_hot(label_batch_nohot, depth=num_classes)

        # 定义预测结果
        training_flag = tf.placeholder(tf.bool)
        # logit = net(transfer_learning_train.net_id, image_batch, is_train=training_flag)
        with slim.arg_scope(InRes_V2.inception_resnet_v2_arg_scope()):
            logit, _ = InRes_V2.inception_resnet_v2(x, is_training=training_flag)

        prediction = tf.argmax(logit, axis=-1, output_type=tf.int32)
        # 初始化saver
        saver = tf.train.Saver()

        # 初始化用于计算准确率的各变量
        top_1_acc = 0
        top_5_acc = 0
        crop_type_acc = 0
        disease_type_acc = 0
        severity_wrong = 0

        # GPU设置
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # GPU setting,最高不超过90%
        config.gpu_options.allow_growth = True  # 显存占用根据需求增长

        # session = InteractiveSession(config=config)
        with tf.Session(config=config) as sess:
            # 初始化全局变量、本地变量、dataset迭代器
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)

            # # 自动通过checkpoint找到最新的模型
            # ckpt = tf.train.get_checkpoint_state(Model_SAVE_PATH)
            # # print(train.MODEL_SAVE_PATH)
            # if ckpt and ckpt.model_checkpoint_path:
            #     # 加载模型
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            #     # 通过文件名得到模型保存时迭代的轮数
            #     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            best_checkpoint = checkmate.get_best_checkpoint(transfer_learning_train.MODEL_SAVE_PATH, select_maximum_value=True)
            global_step = best_checkpoint.split('/')[-1].split('-')[-1]
            saver.restore(sess, best_checkpoint)
            print("开始测试！！！")
            count = 0.0
            severity_count = 0.0
            test_results = []
            test_labels = []
            while True:
                try:
                        count += 1.0
                        val_image, label = sess.run([image_batch, label_batch_nohot])
                        pred = sess.run([prediction], feed_dict={x: val_image, training_flag: True})
                        pred = [y for x in pred for y in x]
                        test_results.extend(pred)
                        test_labels.extend(label)
                        print(pred)
                        print(label)
                        print(count)
                except tf.errors.OutOfRangeError:
                    break
            correct = [float(y == y_) for (y, y_) in zip(test_results, test_labels)]
            acc_val = sum(correct) / len(correct)
            print("测试集准确率为：", acc_val)

            # 计算测试用时
            end_time = time()
            run_time = end_time - begin_time
            print("测试用时：", run_time)
