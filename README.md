# CropDiseaseDetection
深度学习-病害识别
#Package-data_generation：
##——TransforToTFRecord.py:
将训练和验证图像数据集转化为TFRecord格式。
##——data_preprocessing.py: 
图像预处理过程。
##——data_generate.py:
调用data_preprocessing和TransforToTFRecord文件中的函数和结果，产生输入神经网络的批次图像。

#Package-nets：
包含LeNet及tensorflow.slim封装下的具体实现的AlexNet、Vgg等网络。
包含论文中所用到的注意力模块实现文件。

#Package-train_process：
##——checkmate.py：
实现保存训练过程中前N最优模型的类——BestCheckpointSaver。
##——show_train_batch.py：
抽取批次训练集中的图片可视化，用于检测图像预处理的正确与否。
##——train.py：
以初始化(迭代轮次)的方式训练图片。
##——train_with_epoch.py：
以初始化(epoch)的方式训练图片。
##——transfer_learning_train.py：
以迁移学习(迭代轮次)inception_resnet_v2的方式训练图片。

#Package-train_process：
##——crop_disease_type.py：
包含用于计算各类准确率的辅助函数。
##——validate.py：
验证过程，得出病虫害识别模型的各类准确率指标。

#Package-application：
##——predict.py：
向前传播，实现预测一张图片的具体种类。

