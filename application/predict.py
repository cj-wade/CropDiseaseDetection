from time import *
import tensorflow as tf
from nets.lenet import inference
from nets import alexnet
from train_process import train
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

img_w = 224
img_h = 224
channels = 3


def predict_one_picture(img_path):
    tf.reset_default_graph()
    with tf.get_default_graph().as_default():
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [1, img_w, img_h, channels])

        # 对图像进行预处理，成为标准的网络输入
        img = tf.gfile.FastGFile(img_path, "rb").read()
        img = tf.image.decode_jpeg(img)
        if img.dtype != tf.float32:
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize_images(img, [img_w, img_h])

        # 定义预测结果
        # logit = inference(x)
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            logit, _ = alexnet.alexnet_v2(x)
        prediction = tf.argmax(logit, 1)

        # 初始化saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            # 自动通过checkpoint找到最新的模型
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                begin_time = time()
                image = sess.run([img])
                pred = sess.run([prediction], feed_dict={x: image})
                # 计算训练用时
                end_time = time()
                run_time = end_time - begin_time
                print("预测用时:", run_time)
                plt.imshow(img.eval())
                plt.title(pred[0][0])
                plt.show()
                return pred[0][0]
            else:
                print("未发现识别模型！")
                return -1


if __name__ == '__main__':
    img_path = 'D:/各种学科资料/病虫害识别/病虫害简介/辣椒疮痂病.jpg'
    result = predict_one_picture(img_path)
    print(result)
