import tensorflow as tf

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math


# 定义函数转化变量类型。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 读取mnist数据。
mnist = input_data.read_data_sets("D:\各种学科资料\病虫害识别\数据集\Mnist\MNIST_data", dtype=tf.uint8, one_hot=True)
tf.logging.set_verbosity(old_v)

images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

print(pixels)

# 输出TFRecord文件的地址。
filename = "D:\各种学科资料\病虫害识别\数据集\Mnistoutput.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'img_W': _int64_feature(math.sqrt(pixels)),
        'img_H': _int64_feature(math.sqrt(pixels)),
        'channels': _int64_feature(1),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()
print("TFRecord文件已保存。")
