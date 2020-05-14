import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from nets.attention_module import cbam_module

class_num = 61
# Hyperparameter
growth_k = 24
nb_block = 2  # how many (dense block + Transition Layer) ?
# init_learning_rate = 1e-4
# epsilon = 1e-4  # AdamOptimizer epsilon
dropout_rate = 0.2


# Momentum Optimizer will use
# nesterov_momentum = 0.9
# weight_decay = 1e-4


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding='SAME')
        return network


def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Linear(x):
    return tf.layers.dense(inputs=x, units=class_num, name='linear')


# DenseNet类
class DenseNet():
    def __init__(self, x, training, nb_blocks=nb_block, filters=growth_k, is_cbam=False):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.is_cbam = is_cbam
        self.model = self.Dense_net(x)

    # 一个dense_block中一个完整的卷积模块
    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            # 先对前面输入的feature之和进行一个[1,1]卷积的压缩，压缩后的大小为4倍的growth_k
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # 再进行一次[3*3]的卷积，输出的feature个数为growth_k
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            if self.is_cbam:
                x = cbam_module(x, name=scope + "cbam")
            # print(x)

            return x

    # transition层，放在不同的dense_blcok之间，减少参数的输入，将输入的通道数缩减至原来的1/2
    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')

            # https://github.com/taki0112/Densenet-Tensorflow/issues/10

            in_channel = x.get_shape()[-1]
            x = conv_layer(x, filter=int(in_channel) * 0.5, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    # input: 上一个dense_blcok/输入层的输入
    # nb_layers: 一个dense_block中bottleneck_layer的个数
    # layer_name: 本dense_block的名字
    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            # 建一个列表用于存放每一层的输入
            layers_concat = list()
            # 首先存放上一个dense_blcok/输入层的输入
            layers_concat.append(input_x)
            # 做一次完整的卷积
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            # 存放本dense_blcok第一次卷积的输出
            layers_concat.append(x)

            # 不断重复上述操作，直至搭建完nb_layers个dense_block模块
            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            # 将本dense_block模块中的所有卷积模块的输入进行concat，作为下一层的输入
            x = Concatenation(layers_concat)

            return x

    # 完整的DenseNet向前传播，实际上就是k=growth_k的DenseNet-201
    def Dense_net(self, input_x):
        # 将输入的224*224的图片利用[7*7]的卷积变成112*112，输出的feature个数为2倍的growth_k
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)

        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        """

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=16, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        # x = tf.reshape(x, [-1, 10])
        return x
