import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型定义-ResNet
net = ResNet18().to(device)
toLoad = {}
checkpoint = torch.load("./model/resnet18_cifar10_normal_train_finished_saving_60.pth", map_location='cpu')
net.load_state_dict(checkpoint)
params = net.state_dict()
toLoad = params
# for k,v in params.items():
#     print(k)    #打印网络中的变量名
#     print(v)
#     break
# print("1.0----135")
# print(params["conv1.0.weight"])
# print("1.1----135")
# print(params["conv1.1.weight"])
print("------------")
checkpoint = torch.load("./model/resnet18_cifar10_noraml_train_init.pth", map_location='cpu')

initLayers = [
              "fc.weight", "fc.bias",
              "layer4.1.left.3.weight", "layer4.1.left.4.weight", "layer4.1.left.4.bias",
              "layer4.1.left.0.weight", "layer4.1.left.1.weight", "layer4.1.left.1.bias",
              "layer4.0.left.3.weight", "layer4.0.left.4.weight", "layer4.0.left.4.bias", "layer4.0.shortcut.0.weight", "layer4.0.shortcut.1.weight", "layer4.0.shortcut.1.bias",
              "layer4.0.left.0.weight", "layer4.0.left.1.weight", "layer4.0.left.1.bias",
              "layer3.1.left.3.weight", "layer3.1.left.4.weight", "layer3.1.left.4.bias",
              "layer3.1.left.0.weight", "layer3.1.left.1.weight", "layer3.1.left.1.bias",
              "layer3.0.left.3.weight", "layer3.0.left.4.weight", "layer3.0.left.4.bias", "layer3.0.shortcut.0.weight", "layer3.0.shortcut.1.weight", "layer3.0.shortcut.1.bias",
              "layer3.0.left.0.weight", "layer3.0.left.1.weight", "layer3.0.left.1.bias",
              # "layer2.1.left.3.weight", "layer2.1.left.4.weight", "layer2.1.left.4.bias",
              # "layer2.1.left.0.weight", "layer2.1.left.1.weight", "layer2.1.left.1.bias",
              # "layer2.0.left.3.weight", "layer2.0.left.4.weight", "layer2.0.left.4.bias", "layer2.0.shortcut.0.weight", "layer2.0.shortcut.1.weight", "layer2.0.shortcut.1.bias",
              # "layer2.0.left.0.weight", "layer2.0.left.1.weight", "layer2.0.left.1.bias",
              # "layer1.1.left.3.weight", "layer1.1.left.4.weight", "layer1.1.left.4.bias",
              # "layer1.1.left.0.weight", "layer1.1.left.1.weight", "layer1.1.left.1.bias",
              # "layer1.0.left.3.weight", "layer1.0.left.4.weight", "layer1.0.left.4.bias",
              # "layer1.0.left.0.weight", "layer1.0.left.1.weight", "layer1.0.left.1.bias",
              # "conv1.0.weight", "conv1.1.weight", "conv1.1.bias",
              ]
fileName = "resnet18_cifar10_fc_conv8_before_training.pth"
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
