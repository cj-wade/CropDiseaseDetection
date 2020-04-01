import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import json

'''
    将数据集转化为TF格式
'''
# 图形格式化大小
image_size = 224
channels = 3

# 源数据地址
train_source_dir = 'D:\各种学科资料\病虫害识别\数据集\AI挑战赛\AgriculturalDisease_trainingset\images\\'
train_source_label = 'D:\各种学科资料\病虫害识别\数据集\AI挑战赛\AgriculturalDisease_trainingset\AgriculturalDisease_train_annotations.json'

validation_source_dir = 'D:\各种学科资料\病虫害识别\数据集\AI挑战赛\AgriculturalDisease_validationset\images\\'
validation_source_label = 'D:\各种学科资料\病虫害识别\数据集\AI挑战赛\AgriculturalDisease_validationset\AgriculturalDisease_validation_annotations.json'

# 读出标签文件
with open(train_source_label, 'r', encoding='UTF-8') as f:
    train_load_dict = json.load(f)

with open(validation_source_label, 'r', encoding='UTF-8') as f:
    validation_load_dict = json.load(f)

# 输出的TF格式文件
train_file = "D:\各种学科资料\病虫害识别\数据集\AI挑战赛\AgriculturalDisease_trainingset\\train_grabcut.tfrecords"
validation_file = "D:\各种学科资料\病虫害识别\数据集\AI挑战赛\AgriculturalDisease_validationset\\validation_grabcut.tfrecords"
# 用于写入TFRecord文件
train_writer = tf.python_io.TFRecordWriter(train_file)
validation_writer = tf.python_io.TFRecordWriter(validation_file)


# 找到图片id对应的label
def find_label(image, dict):
    for item in dict:
        if item['image_id'] == image:
            return item['disease_class']
    print('can not find id')


# 生成字符型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':

    count = 0

    # 逐个读取文件夹中的文件
    for file in os.listdir(train_source_dir):
        # 对应图片
        img = cv2.imdecode(np.fromfile(train_source_dir + file, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))

        # 前背景分离
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, 220, 220)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]

        # 图片对应标签（int）
        label = find_label(file, train_load_dict)
        # 将图片转换为字符串
        image_raw = img.tobytes()
        # 获取图像尺寸
        img_W = 224
        img_H = 224
        # 图像通道数
        channels = 3

        # 将一个样例转化成Example Protocol Buffer，并将所有的信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'img_W': _int64_feature(img_W),
            # 'img_H': _int64_feature(img_H),
            # 'channels': _int64_feature(channels),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)}))

        # 将一个Example写入TFRecord文件中
        train_writer.write(example.SerializeToString())
        count = count + 1
        if count % 100 == 0:
            print(count)
    train_writer.close()

    for file in os.listdir(validation_source_dir):
        # 对应图片
        img = cv2.imdecode(np.fromfile(validation_source_dir + file, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))

        # 前背景分离
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (1, 1, 220, 220)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]

        # 图片对应标签（int）
        label = find_label(file, validation_load_dict)
        # 将图片转换为字符串
        image_raw = img.tobytes()
        # 获取图像尺寸
        img_W = 224
        img_H = 224
        # 图像通道数
        channels = 3

        # 将一个样例转化成Example Protocol Buffer，并将所有的信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'img_W': _int64_feature(img_W),
            # 'img_H': _int64_feature(img_H),
            # 'channels': _int64_feature(channels),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)}))

        # 将一个Example写入TFRecord文件中
        validation_writer.write(example.SerializeToString())
        count = count + 1
        if count % 100 == 0:
            print(count)
    validation_writer.close()
