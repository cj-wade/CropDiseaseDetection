import tensorflow as tf
slim = tf.contrib.slim

# SE模块
def se_block(input_feature, name, ratio=16):
    """Contains the implementation of Squeeze-and-Excitation block.
    As described in https://arxiv.org/abs/1709.01507.
    """

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

def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


# CBAM
def convolutional_block_attention_module(feature_map, name="", inner_units_ratio=0.5):
    print("cbam!")
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope(name):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # feature map[0]:batch_size, [1]:height, [2]:width, [3]:channel_size
        # channel attention
        # # 做一次全局平均池化
        #         # channel_avg_weights = tf.nn.avg_pool(
        #         #     value=feature_map,
        #         #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #         #     strides=[1, 1, 1, 1],
        #         #     padding='VALID'
        #         # )
        #         # # 做一次全局最大池化
        #         # channel_max_weights = tf.nn.max_pool(
        #         #     value=feature_map,
        #         #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #         #     strides=[1, 1, 1, 1],
        #         #     padding='VALID'
        #         # )
        # 分别进行一次全局最大池化和平均池化
        channel_max_weights = tf.reduce_max(tf.reduce_max(feature_map, axis=1, keepdims=True), axis=2, keepdims=True)
        channel_avg_weights = tf.reduce_mean(tf.reduce_mean(feature_map, axis=1, keepdims=True), axis=2, keepdims=True)
        # 将两个全局池化后的结果都拉成一维
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        # 将两个一维的结果进行拼接
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

        #
        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=None
        )
        # 结果两次全连接得出的avg结果和max结果进行加和
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)

        # spatial attention
        # 分别沿着chennel维做一次平均池化/最大池化
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)

        # reshape成论文图中的两个平面
        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        # 将两个平面进行concat
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        # 用7*7的一维卷积将其再压缩成一个平面
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [7, 7],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        # 将其与前面attention的结果相乘
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention


# R_CBAM
def reverse_cbam(feature_map, name="", inner_units_ratio=0.5):
    print("reverse_cbam!")
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope(name):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # spatial attention
        # 分别沿着chennel维做一次平均池化/最大池化
        channel_wise_avg_pooling = tf.reduce_mean(feature_map, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map, axis=3)

        # reshape成论文图中的两个平面
        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        # 将两个平面进行concat
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        # 用7*7的一维卷积将其再压缩成一个平面
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [7, 7],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        # 将其与前面attention的结果相乘
        feature_map_with_spatial_attention = tf.multiply(feature_map, spatial_attention)

        # channel attention
        # 做一次全局平均池化
        # channel_avg_weights = tf.nn.avg_pool(
        #     value=feature_map_with_spatial_attention,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )
        # # 做一次全局最大池化
        # channel_max_weights = tf.nn.max_pool(
        #     value=feature_map_with_spatial_attention,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )

        # 分别进行一次全局最大池化和平均池化
        channel_max_weights = tf.reduce_max(tf.reduce_max(feature_map_with_spatial_attention, axis=1, keepdims=True), axis=2, keepdims=True)
        channel_avg_weights = tf.reduce_mean(tf.reduce_mean(feature_map_with_spatial_attention, axis=1, keepdims=True), axis=2, keepdims=True)

        # 将两个全局池化后的结果都拉成一维
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        # 将两个一维的结果进行拼接
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

        #
        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=None
        )
        # 结果两次全连接得出的avg结果和max结果进行加和
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_attention = tf.multiply(feature_map_with_spatial_attention, channel_attention)
    return feature_map_with_attention


# I_CBAM
def improved_cbam(feature_map, name="", inner_units_ratio=0.5):
    print("improved_cbam!")
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope(name):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # spatial attention
        # 分别沿着chennel维做一次平均池化/最大池化
        channel_wise_avg_pooling = tf.reduce_mean(feature_map, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map, axis=3)

        # reshape成论文图中的两个平面
        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        # 将两个平面进行concat
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        # 用7*7的一维卷积将其再压缩成一个平面
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [7, 7],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        # 将其与前面attention的结果相乘
        feature_map_with_spatial_attention = tf.multiply(feature_map, spatial_attention)

        # channel attention
        # 分别进行一次全局最大池化和平均池化
        channel_max_weights = tf.reduce_max(tf.reduce_max(feature_map, axis=1, keepdims=True), axis=2, keepdims=True)
        channel_avg_weights = tf.reduce_mean(tf.reduce_mean(feature_map, axis=1, keepdims=True), axis=2, keepdims=True)

        # 将两个全局池化后的结果都拉成一维
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        # 将两个一维的结果进行拼接
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

        #
        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=None
        )
        # 结果两次全连接得出的avg结果和max结果进行加和
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_attention = tf.multiply(feature_map_with_spatial_attention, channel_attention)
    return feature_map_with_attention


# cbam_channel_attention
def cbam_channel_block(feature_map, name, inner_units_ratio=0.5):
    print("cbam_channel_attention!")
    with tf.variable_scope(name):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # feature map[0]:batch_size, [1]:height, [2]:width, [3]:channel_size
        # channel attention
        # 做一次全局平均池化
        # channel_avg_weights = tf.nn.avg_pool(
        #     value=feature_map,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )
        # # 做一次全局最大池化
        # channel_max_weights = tf.nn.max_pool(
        #     value=feature_map,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )

        # 分别进行一次全局最大池化和平均池化
        channel_max_weights = tf.reduce_max(tf.reduce_max(feature_map, axis=1, keepdims=True), axis=2, keepdims=True)
        channel_avg_weights = tf.reduce_mean(tf.reduce_mean(feature_map, axis=1, keepdims=True), axis=2, keepdims=True)
        # 将两个全局池化后的结果都拉成一维
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        # 将两个一维的结果进行拼接
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

        #
        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=None
        )
        # 结果两次全连接得出的avg结果和max结果进行加和
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)

        return feature_map_with_channel_attention

# cbam_spatial_attention
def cbam_spatial_block(feature_map, name=""):
    print("cbam_spatial_attention")
    with tf.variable_scope(name):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # feature map[0]:batch_size, [1]:height, [2]:width, [3]:channel_size
        # spatial attention
        # 分别沿着chennel维做一次平均池化/最大池化
        channel_wise_avg_pooling = tf.reduce_mean(feature_map, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map, axis=3)

        # reshape成论文图中的两个平面
        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        # 将两个平面进行concat
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        # 用7*7的一维卷积将其再压缩成一个平面
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [7, 7],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        # 将其与前面attention的结果相乘
        feature_map_with_spatial_attention = tf.multiply(feature_map, spatial_attention)
        return feature_map_with_spatial_attention






# SE_CBAM
def se_cbam_spatial_module(feature_map, name=""):
    print("SE_CBAM")
    with tf.variable_scope(name):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # feature map[0]:batch_size, [1]:height, [2]:width, [3]:channel_size
        # 先进行一次SE操作
        feature_map_with_se_attention = se_block(feature_map, name=name)

        # spatial attention
        # 分别沿着chennel维做一次平均池化/最大池化
        channel_wise_avg_pooling = tf.reduce_mean(feature_map, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map, axis=3)

        # reshape成论文图中的两个平面
        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        # 将两个平面进行concat
        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        # 用7*7的一维卷积将其再压缩成一个平面
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [7, 7],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        # 将其与前面attention的结果相乘
        feature_map_with_attention = tf.multiply(feature_map_with_se_attention, spatial_attention)
        return feature_map_with_attention