import sys
cur = '/mnt/PyCharm_Project_1'
sys.path.append(cur)
print("success")
from time import *
import tensorflow as tf
from data_generation.data_generate import get_test_dataset
from nets.net_choice import net
from train_process import checkmate, train_with_epoch
import numpy as np
from heapq import nlargest
from validation_process.crop_disease_type import crop_class
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from nets import SE_inception_resnet_v2 as InRes_V2
import tensorflow.contrib.slim as slim

Model_SAVE_PATH = "D:\Python\PycharmProjects\CropDiseaseDetection\crop_model\Lenet"

num_classes = 61
img_w = img_h = 224


if __name__ == '__main__':
    # 程序开始
    begin_time = time()
    # 每次导入模型前，先重置计算图
    tf.reset_default_graph()
    with tf.get_default_graph().as_default():

        # 获得测试的batch
        dataset = get_test_dataset(is_train=False)
        iterator = dataset.make_initializable_iterator()
        image_batch, label_batch = iterator.get_next()
        label_batch = tf.one_hot(label_batch, depth=num_classes)

        # 定义预测结果
        training_flag = tf.placeholder(tf.bool)
        logit = net(train_with_epoch.net_id, image_batch, is_train=training_flag)
        # with slim.arg_scope(InRes_V2.inception_resnet_v2_arg_scope()):
        #     logit, _ = InRes_V2.inception_resnet_v2(image_batch, is_train=training_flag)
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
            best_checkpoint = checkmate.get_best_checkpoint(train_with_epoch.MODEL_SAVE_PATH, select_maximum_value=True)
            global_step = best_checkpoint.split('/')[-1].split('-')[-1]
            saver.restore(sess, best_checkpoint)
            print("开始测试！！！")
            count = 0.0
            severity_count = 0.0
            while True:
                try:
                    # 进行预测
                    pred, label = sess.run([logit, label_batch], feed_dict={training_flag: False})
                    pred = pred.flatten()
                    label = label.flatten()
                    label = np.argmax(label)

                    top_1_pred = np.argmax(pred)
                    top_5_pred = nlargest(5, range(len(pred)), pred.__getitem__)

                    pred_cropClass = crop_class(top_1_pred)
                    label_cropClass = crop_class(label)

                    crop_type_pred = pred_cropClass[0]
                    crop_type_label = label_cropClass[0]

                    disease_type_pred = pred_cropClass[1]
                    disease_type_label = label_cropClass[1]

                    if label in top_5_pred:  # top_5准确率
                        top_5_acc += 1
                    if crop_type_pred == crop_type_label:  # 物种准确率
                        crop_type_acc += 1
                    if disease_type_pred == disease_type_label:
                        disease_type_acc += 1  # 病害种类准确率
                        severity_count += 1.0
                        if top_1_pred == label:
                            top_1_acc += 1  # top_1准确率
                        else:
                            severity_wrong += 1  # 严重/正常错误率

                    count += 1.0
                    if count % 100 == 0:
                        print(count)
                except tf.errors.OutOfRangeError:
                    break

            # 计算准确率
            print("在第%s轮训练后：" % global_step)
            print("Top_1准确率：", top_1_acc / count)
            print("Top_5准确率：", top_5_acc / count)
            print("品种准确率：", crop_type_acc / count)
            print("病虫害种类准确率：", disease_type_acc / count)
            print("严重程度准确率：", 1 - (severity_wrong / severity_count))

            # 计算测试用时
            end_time = time()
            run_time = end_time - begin_time
            print("测试用时：", run_time)
