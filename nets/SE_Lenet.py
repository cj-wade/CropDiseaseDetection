import tensorflow.contrib.slim as slim
from nets.attention_module import se_block

class_num = 61


def inference(inputs, is_se=False):
    # 第一层卷积
    net = slim.conv2d(inputs, 32, [5, 5], padding='SAME')
    # 第二层池化
    net = slim.max_pool2d(net, 2, stride=2)
    if is_se:
        # SE操作
        net = se_block(net, name="se_lenet1")
    # 类似地
    net = slim.conv2d(net, 64, [5, 5], padding='SAME')
    net = slim.max_pool2d(net, 2, stride=2)
    if is_se:
        # SE操作
        net = se_block(net, name="se_lenet2")
    net = slim.flatten(net)
    net = slim.fully_connected(net, 500)
    net = slim.fully_connected(net, class_num)

    return net
