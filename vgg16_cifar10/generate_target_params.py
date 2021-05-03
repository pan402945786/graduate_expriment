import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
from common.vgg import VGG


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型定义-ResNet
net = VGG('VGG16').to(device)
toLoad = {}
checkpoint = torch.load("./model/vgg16_cifar10_normal_train_finish_100_epochs.pth", map_location='cpu')
# checkpoint = torch.load("./model/vgg16_cifar10_reverse_reset_conv3_before_training.pth", map_location='cpu')
# checkpoint = torch.load("./model/vgg16_cifar10_normal_train_init.pth", map_location='cpu')
net.load_state_dict(checkpoint)
params = net.state_dict()
toLoad = params
# for k,v in params.items():
#     print(k)    #打印网络中的变量名
#     # print(v)
#     # break
# exit()
# print("1.0----135")
# print(params["conv1.0.weight"])
# print("1.1----135")
# print(params["conv1.1.weight"])
print("------------")
checkpoint = torch.load("./model/vgg16_cifar10_normal_train_init.pth", map_location='cpu')

initLayers = [
        "features.0.weight", "features.0.bias", "features.1.weight", "features.1.bias", "features.1.running_mean","features.1.running_var","features.1.num_batches_tracked",
        "features.3.weight", "features.3.bias", "features.4.weight", "features.4.bias", "features.4.running_mean","features.4.running_var","features.4.num_batches_tracked",
        "features.7.weight", "features.7.bias", "features.8.weight", "features.8.bias", "features.8.running_mean","features.8.running_var","features.8.num_batches_tracked",
        "features.10.weight", "features.10.bias", "features.11.weight", "features.11.bias", "features.11.running_mean","features.11.running_var","features.11.num_batches_tracked",
        "features.14.weight", "features.14.bias", "features.15.weight", "features.15.bias", "features.15.running_mean","features.15.running_var","features.15.num_batches_tracked",
        "features.17.weight", "features.17.bias", "features.18.weight", "features.18.bias", "features.18.running_mean", "features.18.running_var", "features.18.num_batches_tracked",
        "features.20.weight", "features.20.bias", "features.21.weight", "features.21.bias", "features.21.running_mean", "features.21.running_var", "features.21.num_batches_tracked",
        "features.24.weight", "features.24.bias", "features.25.weight", "features.25.bias", "features.25.running_mean", "features.25.running_var", "features.25.num_batches_tracked",
        "features.27.weight", "features.27.bias", "features.28.weight", "features.28.bias", "features.28.running_mean", "features.28.running_var", "features.28.num_batches_tracked",
        "features.30.weight", "features.30.bias", "features.31.weight", "features.31.bias", "features.31.running_mean", "features.31.running_var", "features.31.num_batches_tracked",
        "features.34.weight", "features.34.bias", "features.35.weight", "features.35.bias", "features.35.running_mean", "features.35.running_var", "features.35.num_batches_tracked",
        "features.37.weight", "features.37.bias", "features.38.weight", "features.38.bias", "features.38.running_mean", "features.38.running_var", "features.38.num_batches_tracked",
        "features.40.weight", "features.40.bias", "features.41.weight", "features.41.bias", "features.41.running_mean", "features.41.running_var", "features.41.num_batches_tracked",
        "classifier.weight", "classifier.bias",
              ]
fileName = "vgg16_cifar10_reverse_reset_conv14_with_running_mean_var_before_training.pth"
print(fileName)
for k in checkpoint.keys():
    if k in initLayers:
        toLoad[k] = checkpoint[k]
        print("added:" + k)
net.load_state_dict(toLoad)
params=net.state_dict()
print('Saving model......')
torch.save(net.state_dict(), '%s/%s' % ('./model', fileName))
exit()
