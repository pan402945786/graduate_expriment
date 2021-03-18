import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
from resnet_1 import ResNet18
import collections

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）



# 冻结层次
# frozenSet = [
# "conv1.0.weight","conv1.1.weight","conv1.1.bias","conv1.1.running_mean","conv1.1.running_var","conv1.1.num_batches_tracked",
# "layer1.0.left.0.weight","layer1.0.left.1.weight","layer1.0.left.1.bias","layer1.0.left.1.running_mean","layer1.0.left.1.running_var","layer1.0.left.1.num_batches_tracked",
# "layer1.0.left.3.weight","layer1.0.left.4.weight","layer1.0.left.4.bias","layer1.0.left.4.running_mean","layer1.0.left.4.running_var","layer1.0.left.4.num_batches_tracked",
# "layer1.1.left.0.weight","layer1.1.left.1.weight","layer1.1.left.1.bias","layer1.1.left.1.running_mean","layer1.1.left.1.running_var","layer1.1.left.1.num_batches_tracked",
# "layer1.1.left.3.weight","layer1.1.left.4.weight","layer1.1.left.4.bias","layer1.1.left.4.running_mean","layer1.1.left.4.running_var","layer1.1.left.4.num_batches_tracked",
# "layer2.0.left.0.weight","layer2.0.left.1.weight","layer2.0.left.1.bias","layer2.0.left.1.running_mean","layer2.0.left.1.running_var","layer2.0.left.1.num_batches_tracked",
# "layer2.0.left.3.weight","layer2.0.left.4.weight","layer2.0.left.4.bias","layer2.0.left.4.running_mean","layer2.0.left.4.running_var","layer2.0.left.4.num_batches_tracked",
# "layer2.0.shortcut.0.weight","layer2.0.shortcut.1.weight","layer2.0.shortcut.1.bias","layer2.0.shortcut.1.running_mean","layer2.0.shortcut.1.running_var","layer2.0.shortcut.1.num_batches_tracked",
# "layer2.1.left.0.weight","layer2.1.left.1.weight","layer2.1.left.1.bias","layer2.1.left.1.running_mean","layer2.1.left.1.running_var","layer2.1.left.1.num_batches_tracked",
# "layer2.1.left.3.weight","layer2.1.left.4.weight","layer2.1.left.4.bias","layer2.1.left.4.running_mean","layer2.1.left.4.running_var","layer2.1.left.4.num_batches_tracked",
# "layer3.0.left.0.weight","layer3.0.left.1.weight","layer3.0.left.1.bias","layer3.0.left.1.running_mean","layer3.0.left.1.running_var","layer3.0.left.1.num_batches_tracked",
# "layer3.0.left.3.weight","layer3.0.left.4.weight","layer3.0.left.4.bias","layer3.0.left.4.running_mean","layer3.0.left.4.running_var","layer3.0.left.4.num_batches_tracked",
# "layer3.0.shortcut.0.weight","layer3.0.shortcut.1.weight","layer3.0.shortcut.1.bias","layer3.0.shortcut.1.running_mean","layer3.0.shortcut.1.running_var","layer3.0.shortcut.1.num_batches_tracked",
# "layer3.1.left.0.weight","layer3.1.left.1.weight","layer3.1.left.1.bias","layer3.1.left.1.running_mean","layer3.1.left.1.running_var","layer3.1.left.1.num_batches_tracked",
# "layer3.1.left.3.weight","layer3.1.left.4.weight","layer3.1.left.4.bias","layer3.1.left.4.running_mean","layer3.1.left.4.running_var","layer3.1.left.4.num_batches_tracked"]
# params=net.state_dict()
# for k,v in params.items():
#     if k in frozenSet:
#         v.require_grad = False
    # print(k)    #打印网络中的变量名


# print('Saving model......')
# torch.save(net.state_dict(), '%s/init_model.pth' % (args.outf))
# exit()
toLoad = {}
checkpoint = torch.load("./model/resnet18_vggface100_normal_056_epoch.pth", map_location='cpu')
# print(checkpoint)
# checkpoint = torch.load("./model/init_model.pth", map_location='cpu')

d2 = collections.OrderedDict([(k.replace('module.', ''), v) for k, v in checkpoint.items()])
net.load_state_dict(d2)
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
checkpoint = torch.load("./model/init_model.pth", map_location='cpu')

# correct = 0
# total = 0
# count = 0
# for data in testloader:
#     count += 1
#     print("count:", count)
#     net.eval()
#     images, labels = data
#     images, labels = images.to(device), labels.to(device)
#     outputs = net(images)
#     # 取得分最高的那个类 (outputs.data的索引号)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
# print('net135测试分类准确率为：%.3f%%' % (100 * correct / total))

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
fileName = "resnet18_70epoch_reset_fc_conv8_before_training.pth"
print(fileName)
for k in checkpoint.keys():
    if k in initLayers:
        toLoad[k] = checkpoint[k]
        print("added:" + k)
net.load_state_dict(toLoad)
params=net.state_dict()
print('Saving model......')

torch.save(net.state_dict(), '%s/%s' % (args.outf, fileName))
# for k,v in params.items():
#     print(k)    #打印网络中的变量名
# print("1.0----added")
# print(params["conv1.0.weight"])
# print("1.1----added")
# print(params["conv1.1.weight"])

# correct = 0
# total = 0
# count = 0
# for data in testloader:
#     count += 1
#     print(count)
#     net.eval()
#     images, labels = data
#     images, labels = images.to(device), labels.to(device)
#     outputs = net(images)
#     # 取得分最高的那个类 (outputs.data的索引号)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
# print('冻结最后一层后测试分类准确率为：%.3f%%' % (100 * correct / total))
exit()

# for p in net.parameters():
#     print(p)
    # p.requires_grad: bool
    # p.data: Tensor

# for name, param in net.state_dict().items():
#     print(name)
    # print(param)
    # name: str
    # param: Tensor
# exit()
