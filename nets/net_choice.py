import tensorflow.contrib.slim as slim
from nets import lenet, alexnet, SE_inception_resnet_v2, vgg


def net(net_id, x, is_train=True):
    # 定义预测结果
    if net_id == 0:  # Lenet
        pred = lenet.inference(x)
    elif net_id == 1:  # Alexnet
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            pred, _ = alexnet.alexnet_v2(x, is_training=is_train)
    elif net_id == 2:  # attention_Lenet
        pred = lenet.inference(x, is_cbam=True)
    elif net_id == 3:  # attention_Alexnet
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            pred, _ = alexnet.alexnet_v2(x, is_re_cbam=True, is_training=is_train)
    elif net_id == 4:  # InResNet_V2
        pred, _ = SE_inception_resnet_v2.inception_resnet_v2(x, is_training=is_train)
    elif net_id == 5:  # attention_InResNet_V2
        with slim.arg_scope(SE_inception_resnet_v2.inception_resnet_v2_arg_scope()):
            pred, _ = SE_inception_resnet_v2.inception_resnet_v2(x, attention_module='cbam', is_training=is_train)
    elif net_id == 6:  # Vgg
        with slim.arg_scope(vgg.vgg_arg_scope()):
            pred, _ = vgg.vgg_16(x, is_training=is_train)
    elif net_id == 7:  # attention_Vgg
        with slim.arg_scope(vgg.vgg_arg_scope()):
            pred, _ = vgg.vgg_16(x, is_cbam=True, is_training=is_train)
    # elif net_id == 8:  # DenseNet
    #     pred = densenet.DenseNet(x, training=is_train).model
    # elif net_id == 9:   # # attention_DenseNet
    #     pred = densenet.DenseNet(x, training=is_train, is_cbam=True).model
    else:
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            pred, _ = alexnet.alexnet_v2(x)

    return pred
