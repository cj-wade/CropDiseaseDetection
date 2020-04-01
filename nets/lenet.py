# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.attention_module import se_block, spatial_attention, cbam_module, cbam_channel_block, cbam_spatial_block, se_cbam_spatial_module, reverse_cbam

class_num = 61


def inference(inputs, is_se=False, is_ccb=False, is_csb=False, is_cbam=False, is_se_cbam=False, is_re_cbam=False):
    # 第一层卷积
    net = slim.conv2d(inputs, 32, [5, 5], padding='SAME')
    if is_se:
        net = se_block(net, name="se_lenet1")
    if is_ccb:
        net = cbam_channel_block(net, name="1")
    if is_csb:
        net = cbam_spatial_block(net, name="1")
    if is_cbam:
        net = cbam_module(net, name="1")
    if is_se_cbam:
        net = se_cbam_spatial_module(net, name="1")
    if is_re_cbam:
        net = reverse_cbam(net, name="1")
    # 第二层池化
    net = slim.max_pool2d(net, 2, stride=2)


    # 类似地
    net = slim.conv2d(net, 64, [5, 5], padding='SAME')
    if is_se:
        net = se_block(net, name="se_lenet2")
    if is_ccb:
        net = cbam_channel_block(net, name="2")
    if is_csb:
        net = cbam_spatial_block(net, name="2")
    if is_cbam:
        net = cbam_module(net, name="2")
    if is_se_cbam:
        net = se_cbam_spatial_module(net, name="2")
    if is_re_cbam:
        net = reverse_cbam(net, name="2")
    net = slim.max_pool2d(net, 2, stride=2)


    net = slim.flatten(net)
    net = slim.fully_connected(net, 500)
    net = slim.fully_connected(net, class_num)

    return net


def accuracy(logits, labels):
    with tf.variable_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
