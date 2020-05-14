import tensorflow as tf
from nets.stn import spatial_transformer_network
from nets.cbam import convolutional_block_attention_module as cbam_module
from nets.cbam import cbam_channel_block, cbam_spatial_block, se_cbam_spatial_module, reverse_cbam, improved_cbam

slim = tf.contrib.slim
import numpy as np
from data_generation import data_generate


# SE模块
def se_block(input_feature, name, ratio=16):
    """Contains the implementation of Squeeze-and-Excitation block.
    As described in https://arxiv.org/abs/1709.01507.
    """
    print("SE")
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        scale = input_feature * excitation
    return scale


# 空间注意力机制
def spatial_attention(feature_map, K=1024, weight_decay=0.00004, scope="", reuse=None):
    """This method is used to add spatial attention to model.

    Parameters
    ---------------
    @feature_map: Which visual feature map as branch to use.
    @K: Map `H*W` units to K units. Now unused.
    @reuse: reuse variables if use multi gpus.

    Return
    ---------------
    @attended_fm: Feature map with Spatial Attention.
    """
    with tf.variable_scope(scope, 'SpatialAttention', reuse=reuse):
        # Tensorflow's tensor is in BHWC format. H for row split while W for column split.
        _, H, W, C = tuple([x for x in feature_map.get_shape()])
        # w_s = tf.get_variable("SpatialAttention_w_s", [C, 1],
        #                       dtype=tf.float32,
        #                       initializer=tf.initializers.orthogonal,
        #                       regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        w_s = tf.get_variable("SpatialAttention_w_s", [C, 1],
                              dtype=tf.float32,
                              initializer=tf.initializers.orthogonal)
        b_s = tf.get_variable("SpatialAttention_b_s", [1],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        spatial_attention_fm = tf.matmul(tf.reshape(feature_map, [-1, C]), w_s) + b_s
        spatial_attention_fm = tf.nn.sigmoid(tf.reshape(spatial_attention_fm, [-1, W * H]))
        #         spatial_attention_fm = tf.clip_by_value(tf.nn.relu(tf.reshape(spatial_attention_fm,
        #                                                                       [-1, W * H])),
        #                                                 clip_value_min = 0,
        #                                                 clip_value_max = 1)
        attention = tf.reshape(tf.concat([spatial_attention_fm] * C, axis=1), [-1, H, W, C])
        attended_fm = attention * feature_map
        return attended_fm


# 通道注意力机制
def channel_wise_attention(feature_map, K=1024, weight_decay=0.00004, scope='', reuse=None):
    """This method is used to add spatial attention to model.

    Parameters
    ---------------
    @feature_map: Which visual feature map as branch to use.
    @K: Map `H*W` units to K units. Now unused.
    @reuse: reuse variables if use multi gpus.

    Return
    ---------------
    @attended_fm: Feature map with Channel-Wise Attention.
    """
    with tf.variable_scope(scope, 'ChannelWiseAttention', reuse=reuse):
        # Tensorflow's tensor is in BHWC format. H for row split while W for column split.
        _, H, W, C = tuple([int(x) for x in feature_map.get_shape()])
        w_s = tf.get_variable("ChannelWiseAttention_w_s", [C, C],
                              dtype=tf.float32,
                              initializer=tf.initializers.orthogonal,
                              regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        b_s = tf.get_variable("ChannelWiseAttention_b_s", [C],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        transpose_feature_map = tf.transpose(tf.reduce_mean(feature_map, [1, 2], keep_dims=True),
                                             perm=[0, 3, 1, 2])
        channel_wise_attention_fm = tf.matmul(tf.reshape(transpose_feature_map,
                                                         [-1, C]), w_s) + b_s
        channel_wise_attention_fm = tf.nn.sigmoid(channel_wise_attention_fm)
        #         channel_wise_attention_fm = tf.clip_by_value(tf.nn.relu(channel_wise_attention_fm),
        #                                                      clip_value_min = 0,
        #                                                      clip_value_max = 1)
        attention = tf.reshape(tf.concat([channel_wise_attention_fm] * (H * W),
                                         axis=1), [-1, H, W, C])
        attended_fm = attention * feature_map
        return attended_fm


# cbam模块
# def cbam_module(inputs, reduction_ratio=0.5, name=""):
#     with tf.variable_scope("cbam_" + name, reuse=tf.AUTO_REUSE):
#         # 获取batch_siez和特征图的通道数
#         _, hidden_num = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[3]
#         batch_size = tf.shape(inputs)[0]
#         # hidden_num = tf.shape(inputs)[3]
#
#         # chennel-attention
#         # 分别进行一次全局最大池化和平均池化
#         maxpool_channel = tf.reduce_max(tf.reduce_max(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
#         avgpool_channel = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
#
#         # 将两个全局池化后的结果都拉成一维
#         maxpool_channel = tf.layers.Flatten()(maxpool_channel)
#         avgpool_channel = tf.layers.Flatten()(avgpool_channel)
#
#         # 最大池化的结果经过两层全连接层进行学习
#         mlp_1_max = tf.layers.dense(inputs=maxpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
#                                     reuse=None, activation=tf.nn.relu)
#         mlp_2_max = tf.layers.dense(inputs=mlp_1_max, units=hidden_num, name="mlp_2", reuse=None)
#         mlp_2_max = tf.reshape(mlp_2_max, [batch_size, 1, 1, hidden_num])
#
#         # 平均池化的结果经过两层全连接层进行学习
#         mlp_1_avg = tf.layers.dense(inputs=avgpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
#                                     reuse=True, activation=tf.nn.relu)
#         mlp_2_avg = tf.layers.dense(inputs=mlp_1_avg, units=hidden_num, name="mlp_2", reuse=True)
#         mlp_2_avg = tf.reshape(mlp_2_avg, [batch_size, 1, 1, hidden_num])
#
#         # 将两次池化的结果做tf.add(),并sigmoid，再乘上原始的特征图(即进行加权操作)，得到chennel-attention后的结果channel_refined_feature
#         channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
#         channel_refined_feature = inputs * channel_attention
#
#         # spatial-attention
#         # 分别进行一次全局最大池化和平均池
#         maxpool_spatial = tf.reduce_max(inputs, axis=3, keepdims=True)
#         avgpool_spatial = tf.reduce_mean(inputs, axis=3, keepdims=True)
#
#         # 将池化后的结果连起来
#         max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
#
#         # 使用7*7的卷积将上面的结果压缩成一维
#         conv_layer = tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same",
#                                       activation=None)
#         # 使用sigmiod训练
#         spatial_attention = tf.nn.sigmoid(conv_layer)
#
#         # 将来结果后chennel-attention之后的结果进行相乘，得到最终的输出
#         refined_feature = channel_refined_feature * spatial_attention
#
#     return refined_feature
#
#
# # 其中的通道模块
# def cbam_channel_block(inputs, reduction_ratio=0.5, name=""):
#     with tf.variable_scope("cbam_channnel_" + name, reuse=tf.AUTO_REUSE):
#         # 获取batch_siez和特征图的通道数
#         _, hidden_num = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[3]
#         batch_size = tf.shape(inputs)[0]
#         # hidden_num = tf.shape(inputs)[3]
#
#
#         # chennel-attention
#         # 分别进行一次全局最大池化和平均池化
#         maxpool_channel = tf.reduce_max(tf.reduce_max(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
#         avgpool_channel = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
#
#         # 将两个全局池化后的结果都拉成一维
#         maxpool_channel = tf.layers.Flatten()(maxpool_channel)
#         avgpool_channel = tf.layers.Flatten()(avgpool_channel)
#
#         # 最大池化的结果经过两层全连接层进行学习
#         mlp_1_max = tf.layers.dense(inputs=maxpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
#                                     reuse=None, activation=tf.nn.relu)
#         mlp_2_max = tf.layers.dense(inputs=mlp_1_max, units=hidden_num, name="mlp_2", reuse=None)
#         mlp_2_max = tf.reshape(mlp_2_max, [batch_size, 1, 1, hidden_num])
#
#         # 平均池化的结果经过两层全连接层进行学习
#         mlp_1_avg = tf.layers.dense(inputs=avgpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
#                                     reuse=True, activation=tf.nn.relu)
#         mlp_2_avg = tf.layers.dense(inputs=mlp_1_avg, units=hidden_num, name="mlp_2", reuse=True)
#         mlp_2_avg = tf.reshape(mlp_2_avg, [batch_size, 1, 1, hidden_num])
#
#         # 将两次池化的结果按通道做相加并sigmoid，再乘上原始的特征图(即进行加权操作)，得到chennel-attention后的结果channel_refined_feature
#         channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
#         channel_refined_feature = inputs * channel_attention
#
#     return channel_refined_feature
#
#
# # 其中的spatial模块
# def cbam_spatial_block(inputs, name=""):
#     with tf.variable_scope("cbam_spatial_" + name, reuse=tf.AUTO_REUSE):
#         # 获取batch_siez和特征图的通道数
#         # batch_size, hidden_num = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[3]
#
#         # spatial-attention
#         # 分别进行一次全局最大池化和平均池化
#         maxpool_spatial = tf.reduce_max(inputs, axis=3, keepdims=True)
#         avgpool_spatial = tf.reduce_mean(inputs, axis=3, keepdims=True)
#
#         # 将池化后的结果连起来
#         max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
#
#         # 使用7*7的卷积将上面的结果压缩成一维
#         conv_layer = tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same",
#                                       activation=None)
#         # 使用sigmiod训练
#         spatial_attention = tf.nn.sigmoid(conv_layer)
#
#         # 将来结果后chennel-attention之后的结果进行相乘，得到最终的输出
#         spatial_refined_feature = inputs * spatial_attention
#
#     return spatial_refined_feature
#
# # SENet+cbam_spatial_block
# def se_cbam_spatial_module(inputs, reduction_ratio=0.5, name=""):
#     with tf.variable_scope("se_cbam_spatial_" + name, reuse=tf.AUTO_REUSE):
#         channel_refined_feature = se_block(inputs, name=name)
#
#         # spatial-attention
#         # 分别进行一次全局最大池化和平均池化
#         maxpool_spatial = tf.reduce_max(inputs, axis=3, keepdims=True)
#         avgpool_spatial = tf.reduce_mean(inputs, axis=3, keepdims=True)
#
#         # 将池化后的结果连起来
#         max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
#
#         # 使用7*7的卷积将上面的结果压缩成一维
#         conv_layer = tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same",
#                                       activation=None)
#         # 使用sigmiod训练
#         spatial_attention = tf.nn.sigmoid(conv_layer)
#
#         # 将来结果后chennel-attention之后的结果进行相乘，得到最终的输出
#         refined_feature = channel_refined_feature * spatial_attention
#
#     return refined_feature