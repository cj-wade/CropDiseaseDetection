import tensorflow as tf
import data_generation.data_preprocessing as dp

# 列举输入文件，训练集（测试集待补）


# 定义模型保存路径
MODEL_SAVE_PATH = "../save_model/"
MODEL_NAME = "model.ckpt"


# 定义parser方法解析TFRecord文件
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            # 'img_W': tf.FixedLenFeature([], tf.int64),
            # 'img_H': tf.FixedLenFeature([], tf.int64),
            # 'channels': tf.FixedLenFeature([], tf.int64)
        })

    # 从原始图像中解析出像素矩阵，并根据图像尺寸还原图像
    decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    decoded_image.set_shape([150528])
    decoded_image = tf.reshape(decoded_image, [224, 224, 3])
    label = features['label']

    return decoded_image, label


image_size = 224  # 定义输入层图片的大小，这里是224*224
batch_size = 32  # 定义组合数据batch的大小
shuffle_buffer = 2048  # 定义随机打乱数据时buffer的大小
NUM_EPOCHS = 0  # 指定数据集重复的次数


def get_train_dataset():
    train_files = tf.train.match_filenames_once(
        "/mnt/PyCharm_Project_1/DataSet/train.tfrecords")
    # D:\Python\PycharmProjects\CropDiseaseDetection\DataSet\\train.tfrecords
    # /mnt/PyCharm_Project_1/DataSet/train.tfrecords
    # 定义读取训练数据的数据集
    dataset = tf.data.TFRecordDataset(train_files)
    dataset = dataset.map(parser)

    # 加入预处理操作
    dataset = dataset.map(
        lambda image, label: (
            dp.parse_data(image, image_size, is_train=True), label
        )
    )

    # 先随机打乱(buffer=1000),再组合成batch(32一个batch)
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)
    dataset = dataset.repeat()
    return dataset


def get_test_dataset(is_train=True):
    test_files = tf.train.match_filenames_once(
        "/mnt/PyCharm_Project_1/DataSet/validation.tfrecords")
    # /mnt/PyCharm_Project_1/DataSet/validation.tfrecords
    # D:\Python\PycharmProjects\CropDiseaseDetection\DataSet\\validation.tfrecords
    # 定义读取训练数据的数据集

    dataset = tf.data.TFRecordDataset(test_files)
    dataset = dataset.map(parser)

    # 加入预处理操作
    dataset = dataset.map(
        lambda image, label: (
            dp.parse_data(image, image_size, is_train=False), label
        )
    )

    # 测试集直接组合成batch
    if is_train:
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
    else:
        dataset = dataset.batch(1)

    return dataset
