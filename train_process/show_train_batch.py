from data_generation import data_generate
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def PreWork():
    # 对预处理的数据进行可视化，查看预处理的效果
    IMG_W = 224
    IMG_H = 224
    BATCH_SIZE = 32
    CAPACITY = 64
    dataset = data_generate.get_train_dataset()
    iterator = dataset.make_initializable_iterator()
    image_batch, label_batch = iterator.get_next()
    label_batch = tf.one_hot(label_batch, depth=61)
    print(label_batch.shape)

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(iterator.initializer)
        i = 0
        try:
            while i < 3:
                # 提取出两个batch的图片并可视化。
                img, label = sess.run([image_batch, label_batch])  # 在会话中取出img和label
                # img = tf.cast(img, tf.uint8)
                '''
                1、range()返回的是range object，而np.arange()返回的是numpy.ndarray()
                range(start, end, step)，返回一个list对象，起始值为start，终止值为end，但不含终止值，步长为step。只能创建int型list。
                arange(start, end, step)，与range()类似，但是返回一个array对象。需要引入import numpy as np，并且arange可以使用float型数据。

                2、range()不支持步长为小数，np.arange()支持步长为小数

                3、两者都可用于迭代
                range尽可用于迭代，而np.nrange作用远不止于此，它是一个序列，可被当做向量使用。
                '''
                for j in np.arange(BATCH_SIZE):
                    # np.arange()函数返回一个有终点和起点的固定步长的排列
                    print(label[j])
                    print(img.shape)
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print('done!')

if __name__ == '__main__':
    PreWork()