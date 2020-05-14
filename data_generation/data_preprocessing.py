import tensorflow as tf
import numpy as np
import cv2

# 给图片加入高斯噪声
def gaussian_noise_layer(input_image, std):
    noise = tf.random_normal(shape=tf.shape(input_image), mean=0.0, stddev=std, dtype=tf.float32)
    noise_image = tf.cast(input_image, tf.float32) + noise  # noise/255.0
    noise_image = tf.clip_by_value(noise_image, 0, 10)  # tf.clip_by_value(noise_image, 0, 1.0)
    return noise_image


# 随机调整图片色彩
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # elif color_ordering == 1:
    #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
    #     image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    #     image = tf.image.random_hue(image, max_delta=0.2)
    # elif color_ordering == 2:
    #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
    #     image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    #     image = tf.image.random_hue(image, max_delta=0.2)
    # elif color_ordering == 3:
    #     image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
    #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    #     image = tf.image.random_hue(image, max_delta=0.2)
    # else:
    #     image = tf.image.random_hue(image, max_delta=0.2)
    #     image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    #     image = tf.image.random_brightness(image, max_delta=32. / 255.)
    #     image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return tf.clip_by_value(image, 0.0, 1.0)


# 对图像进行预处理
def parse_data(image, image_size, is_train):
    """
    导入数据，进行预处理,输出处理后的图像
    Args:
        输入的图像
    Returns:
        目标图像
        :param is_train:
        :param image:
        :param image_size:
    """
    # 图片归一化
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # # 数据预处理，或者数据增强，这一步根据需要自由发挥
    #
    # if not is_train:
    #     # 随机提取patch
    #     image = tf.image.resize_images(image, [image_size, image_size])

    if is_train:
        # 随机提取patch
        image = tf.image.resize_images(image, [image_size, image_size], method=np.random.randint(4)) # 仅改变图片大小，保留完整图片
        # 随机水平翻转图像
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # 随机裁剪
        # image = tf.image.resize_image_with_crop_or_pad(image, 150, 150)
        # 随机调整图片色彩
        image = distort_color(image)
        # 加噪声
        # image = gaussian_noise_layer(image, std=10/255)  # 测试(0.25), 0.5， 1， 2， 3， 5, 10

    # image = tf.image.resize_images(image, [image_size, image_size])  # 仅改变图片大小，保留完整图片
    # image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
    # 加噪声
    # image = gaussian_noise_layer(image, std=0.5)  # 测试(0.25), 0.5， 1， 2， 3， 5, 10
    # 标准化，即三通道都减去均值
    # image = tf.image.per_image_standardization(image)
    # tf.clip_by_value(image, 0.0, 1.0)

    return image
