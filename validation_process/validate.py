from time import *
import tensorflow as tf
from data_generation.data_generate import get_test_dataset
from nets.net_choice import net
from train_process import train
import numpy as np
from heapq import nlargest
from validation_process.crop_disease_type import crop_class

num_classes = 61
img_w = 224
img_h = 224

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
        logit = net(train.net_id, image_batch, is_train=False)

        # 初始化saver
        saver = tf.train.Saver()

        # 初始化用于计算准确率的各变量
        top_1_acc = 0
        top_5_acc = 0
        crop_type_acc = 0
        disease_type_acc = 0
        severity_wrong = 0

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

                print("开始测试！！！")
                count = 0.0
                severity_count = 0.0
                while True:
                    try:
                        # 进行预测
                        pred, label = sess.run([logit, label_batch])
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
