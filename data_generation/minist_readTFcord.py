import tensorflow as tf
# 读取文件。
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["D:\各种学科资料\病虫害识别\数据集\AI挑战赛\AgriculturalDisease_trainingset\\train.tfrecords"])
_,serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64),
        'img_W': tf.FixedLenFeature([],tf.int64),
        'img_H': tf.FixedLenFeature([],tf.int64),
        'channels': tf.FixedLenFeature([],tf.int64)
    })

# images = tf.decode_raw(features['image_raw'],tf.uint8)
# labels = tf.cast(features['label'],tf.int32)
# img_W = tf.cast(features['img_W'],tf.int32)
# img_H = tf.cast(features['img_H'],tf.int32)
# channels = tf.cast(features['channels'],tf.int32)

decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
label = features['label']
img_W = features['img_W']

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(10):
    print(sess.run(img_W))
    print(type(img_W))

