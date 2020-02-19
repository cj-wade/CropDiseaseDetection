import tensorflow.contrib.slim as slim
from nets import lenet, alexnet, SE_inception_resnet_v2


def net(net_id, x, is_train=True):
    # 定义预测结果
    if net_id == 0:  # Lenet
        pred = lenet.inference(x)
    elif net_id == 1:  # Alexnet
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            pred, _ = alexnet.alexnet_v2(x, is_train=is_train)
    elif net_id == 2:  # SE_Lenet
        pred = lenet.inference(x, is_se=True)
    elif net_id == 3:  # SE_Alexnet
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            pred, _ = alexnet.alexnet_v2(x, is_se=True, is_train=is_train)
    elif net_id == 4:  # InResNet_V2
        pred, _ = SE_inception_resnet_v2.inception_resnet_v2(x, is_train=is_train)
    elif net_id == 5:  # SE_InResNet_V2
        pred, _ = SE_inception_resnet_v2.inception_resnet_v2(x, attention_module='se_block', is_train=is_train)
    else:
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            pred, _ = alexnet.alexnet_v2(x)

    return pred