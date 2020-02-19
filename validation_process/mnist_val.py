from time import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from nets.lenet import inference
from train_process import train_mnist
import numpy as np

batch_size = 32
num_classes = 10
img_w = 28
img_h = 28
display_step = 100

mnist = input_data.read_data_sets("D:\Python\PycharmProjects\CropDiseaseDetection\mnist_data", one_hot=True)

if __name__ == '__main__':
    begin_time = time()
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [None, img_w, img_h, 1])
        y = tf.placeholder(tf.float32, [None, num_classes])

        # 获得测试的batch
        image, label = mnist.validation.next_batch(batch_size)
        image = np.reshape(image, (image.shape[0], img_w, img_h, 1))

        # 定义预测结果
        logit = inference(x)
        prediction = tf.argmax(logit, axis=-1, output_type=tf.int32)

        # 初始化saver
        saver = tf.train.Saver()

        # 获取预测结果
        test_results = []
        test_labels = []

        with tf.Session() as sess:

            # 自动通过checkpoint找到最新的模型
            ckpt = tf.train.get_checkpoint_state(train_mnist.MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                while True:
                    try:
                        pred, label = sess.run([prediction, label])
                        test_results.extend(pred)
                        test_labels.extend(label)
                    except tf.errors.OutOfRangeError:
                        break

                # 计算测试集准确率
                correct = [float(y == y_) for (y, y_) in zip(test_results, test_labels)]
                accuracy = sum(correct) / len(correct)
                print("在第%s步训练后，测试集准确率为：%g" % (global_step, accuracy))

                # 计算测试用时
                end_time = time()
                run_time = end_time - begin_time
                m, s = divmod(run_time, 60)
                h, m = divmod(m, 60)
                print("测试用时：%d:%02d:%02d" % (h, m, s))
