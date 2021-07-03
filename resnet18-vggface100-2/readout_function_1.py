import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
from common.resnet_100kinds_vggface2 import ResNet18
from common.lr_scheduler_temp import ReduceLROnPlateau
from common.vgg_face2 import VGG_Faces2
from common import utils
from common.utils import generateParamsResnet18
import os
import time
from common.utils import trainFunc
import numpy as np


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
trainForgetFile = r"/train-20kinds-all.txt"
trainRetainFile = r"/train-80kinds-all.txt"
testForgetFile = r"/test-20kinds-all.txt"
testRetainFile = r"/test-80kinds-all.txt"
trainFile = r"/train_list_100.txt"
testFile = r"/test_list_100.txt"

# 2080机器
fileRoot = r'/home/ubuntu/ml/resnet18-vggface100-2'
dataRoot = r'/home/ubuntu/ml/resnet18_vggface2'
datasetRoot = r'/datasets/train'

# 1080机器
# fileRoot = r'/media/public/ml/resnet18-vggface100-2'
# dataRoot = r'/media/public/ml/resnet18_vggface2'
# datasetRoot = r'/datasets/data/root'

# 实验室台式机
# fileRoot = r'D:\ww2\graduate_expriment\resnet18-vggface100-2'
# dataRoot = r'D:\ww2\graduate_expriment\resnet18_vggface2'
# datasetRoot = r'\datasets\data\root'
# trainForgetFile = r"\train-20kinds-all.txt"
# trainRetainFile = r"\train-80kinds-all.txt"
# testForgetFile = r"\test-20kinds-all.txt"
# testRetainFile = r"\test-80kinds-all.txt"
# trainFile = r"\train_list_100.txt"
# testFile = r"\test_list_100.txt"

layeredParams = []

layeredParams.append(["conv1.0.weight", "conv1.1.weight", "conv1.1.bias"])
layeredParams.append(["layer1.0.left.0.weight", "layer1.0.left.1.weight", "layer1.0.left.1.bias", ])
layeredParams.append(["layer1.0.left.3.weight", "layer1.0.left.4.weight", "layer1.0.left.4.bias", ])
layeredParams.append(["layer1.1.left.0.weight", "layer1.1.left.1.weight", "layer1.1.left.1.bias", ])
layeredParams.append(["layer1.1.left.3.weight", "layer1.1.left.4.weight", "layer1.1.left.4.bias", ])

layeredParams.append(["layer2.0.left.0.weight", "layer2.0.left.1.weight", "layer2.0.left.1.bias", ])
layeredParams.append(["layer2.0.left.3.weight", "layer2.0.left.4.weight", "layer2.0.left.4.bias", "layer2.0.shortcut.0.weight", "layer2.0.shortcut.1.weight", "layer2.0.shortcut.1.bias", ])
layeredParams.append(["layer2.1.left.0.weight", "layer2.1.left.1.weight", "layer2.1.left.1.bias", ])
layeredParams.append(["layer2.1.left.3.weight", "layer2.1.left.4.weight", "layer2.1.left.4.bias",])

layeredParams.append(["layer3.0.left.0.weight", "layer3.0.left.1.weight", "layer3.0.left.1.bias",])
layeredParams.append(["layer3.0.left.3.weight", "layer3.0.left.4.weight", "layer3.0.left.4.bias", "layer3.0.shortcut.0.weight", "layer3.0.shortcut.1.weight", "layer3.0.shortcut.1.bias",])
layeredParams.append(["layer3.1.left.0.weight", "layer3.1.left.1.weight", "layer3.1.left.1.bias",])
layeredParams.append(["layer3.1.left.3.weight", "layer3.1.left.4.weight", "layer3.1.left.4.bias"])

layeredParams.append(["layer4.0.left.0.weight", "layer4.0.left.1.weight", "layer4.0.left.1.bias",])
layeredParams.append(["layer4.0.left.3.weight", "layer4.0.left.4.weight", "layer4.0.left.4.bias", "layer4.0.shortcut.0.weight", "layer4.0.shortcut.1.weight", "layer4.0.shortcut.1.bias",])
layeredParams.append(["layer4.1.left.0.weight", "layer4.1.left.1.weight", "layer4.1.left.1.bias",])
layeredParams.append(["layer4.1.left.3.weight", "layer4.1.left.4.weight", "layer4.1.left.4.bias",])

layeredParams.append(["fc.weight", "fc.bias",])

layer_count_list = []
layer_count = 1
while layer_count < len(layeredParams):
    layer_count_list.append(layer_count)
    layer_count = layer_count + 1
layer_count_list.append(len(layeredParams))

preparedFrozenLayers = []
for i, item in enumerate(layer_count_list):
    frozenLayer = []
    for j in range(item):
        frozenLayer = frozenLayer + layeredParams[17-j]
    preparedFrozenLayers.append(frozenLayer)
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default=fileRoot+'/model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--dataset_dir', type=str, default=dataRoot+datasetRoot, help='dataset directory')
parser.add_argument('--train_img_list_file', type=str, default=fileRoot+trainRetainFile,
                    help='text file containing image files used for training')
parser.add_argument('--test_img_list_file', type=str, default=fileRoot+testFile,
                    help='text file containing image files used for validation, test or feature extraction')
parser.add_argument('--test_retain_file', type=str, default=fileRoot+testRetainFile,
                    help='text file containing image files used for validation, test or feature extraction')
parser.add_argument('--test_forget_file', type=str, default=fileRoot+testForgetFile,
                    help='text file containing image files used for validation, test or feature extraction')
parser.add_argument('--meta_file', type=str, default=dataRoot+r'/datasets/data/meta/identity_meta2.csv', help='meta file')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
# print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
# 超参数设置
EPOCH = 70   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
# BATCH_SIZE = 128      #批处理尺寸(batch_size)
BATCH_SIZE = 10      #批处理尺寸(batch_size)
LR = 0.1        #学习率
T_threshold = 0.0111

# 0. id label map
meta_file = args.meta_file
id_label_dict = utils.get_id_label_map(meta_file)

# 1. data loader
root = args.dataset_dir
train_img_list_file = args.train_img_list_file
test_img_list_file = args.test_img_list_file
test_retain_file = args.test_retain_file
test_forget_file = args.test_forget_file

# trainset = VGG_Faces2(root, train_img_list_file, id_label_dict, split='train')
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
# print(len(trainset))

testset = VGG_Faces2(root, test_img_list_file, id_label_dict, split='valid')
testForgetSet = VGG_Faces2(root, test_forget_file, id_label_dict, split='valid')
testRetainSet = VGG_Faces2(root, test_retain_file, id_label_dict, split='valid')
# testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(len(testForgetSet))
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = ResNet18().to(device)
# net = ResNet18()
# net = ResNet50().to(device)
# net = nn.DataParallel(net)
# net = net.cuda()

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 使用相同测试集测试各个参数准确率
print("Waiting Test!")

targetFile = 'resnet18_vggface80_retrain_080_epoch.pth'

savedFiles = [
    'resnet18_vggface100_normal_train_080_epoch.pth',
    'resnet18_vgg100_normal_init.pth',
    # 'resnet18_vggface100_reset_1_after_training.pth_030_epoch.pth',
    # 'resnet18_vggface100_reset_2_after_training.pth_040_epoch.pth',
    # 'resnet18_vggface100_reset_3_after_training.pth_039_epoch.pth',
    # 'resnet18_vggface100_reset_4_after_training.pth_070_epoch.pth',
    # 'resnet18_vggface100_reset_5_after_training.pth_068_epoch.pth',
    # 'resnet18_vggface100_reset_6_after_training.pth_053_epoch.pth',
    # 'resnet18_vggface100_reset_7_after_training.pth_054_epoch.pth',
    # 'resnet18_vggface100_reset_8_after_training.pth_058_epoch.pth',
    # 'resnet18_vggface100_reset_9_after_training.pth_054_epoch.pth',
    # 'resnet18_vggface100_reset_10_after_training.pth_059_epoch.pth',
    # 'resnet18_vggface100_reset_11_after_training.pth_053_epoch.pth',
    # 'resnet18_vggface100_reset_12_after_training.pth_060_epoch.pth',
    # 'resnet18_vggface100_reset_14_after_training.pth_068_epoch.pth',
    # 'resnet18_vggface100_reset_15_after_training.pth_059_epoch.pth',
    # 'resnet18_vggface100_reset_16_after_training.pth_057_epoch.pth',
    # 'resnet18_vggface100_reset_17_after_training.pth_058_epoch.pth',
    # 'resnet18_vggface100_reset_18_after_training.pth_070_epoch.pth',
    #
    'resnet18_vggface100_reverse_reset_17_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_16_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_15_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_14_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_13_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_12_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_11_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_10_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_9_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_8_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_7_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_6_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_5_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_4_before_training.pth_best_acc_model.pth',
    'resnet18_vggface100_reverse_reset_3_after_training.pth_066_epoch.pth',
    'resnet18_vggface100_reverse_reset_2_after_training.pth_060_epoch.pth',
    'resnet18_vggface100_reverse_reset_1_after_training.pth_063_epoch.pth',
]

testloader_unforget = torch.utils.data.DataLoader(testRetainSet, batch_size=100, shuffle=False, num_workers=2)
# testloader_forget = torch.utils.data.DataLoader(testForgetSet, batch_size=10, shuffle=False, num_workers=0)
# testloader_all = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 测试准确率
totals = []
corrects = []
for i in range(len(savedFiles)):
    totals.append(0)
    corrects.append(0)

with torch.no_grad():
    for data in testloader_unforget:
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        for i, file in enumerate(savedFiles, 0):
            # net.load_state_dict("./model/" + file, map_location='cpu')
            checkpoint = torch.load("./model/" + file)
            net.load_state_dict(checkpoint)
            outputs = net(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            totals[i] += labels.size(0)
            corrects[i] += (predicted == labels).sum()
    for i, file in enumerate(savedFiles, 0):
        print(file + '测试保留集分类准确率为：%.3f%%' % (100. * corrects[i] / totals[i]))


print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# totals = []
# corrects = []
# for i in range(len(savedFiles)):
#     totals.append(0)
#     corrects.append(0)
# with torch.no_grad():
#     for data in testloader_forget:
#         net.eval()
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         for i, file in enumerate(savedFiles, 0):
#             # net.load_state_dict("./model/" + file, map_location='cpu')
#             checkpoint = torch.load("./model/" + file)
#             net.load_state_dict(checkpoint)
#             outputs = net(images)
#             # 取得分最高的那个类 (outputs.data的索引号)
#             _, predicted = torch.max(outputs.data, 1)
#             totals[i] += labels.size(0)
#             corrects[i] += (predicted == labels).sum()
#     for i, file in enumerate(savedFiles, 0):
#         print(file + '测试遗忘集分类准确率为：%.3f%%' % (100. * corrects[i] / totals[i]))
#
# print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

#测试激活距离
# norm_1s = []
# norm_2s = []
# for i in range(len(savedFiles)):
#     norm_1s.append(0)
#     norm_2s.append(0)
#
# def cal_norm(vec_1, vec_2, ord):
#     diff = vec_1 - vec_2
#     norm_sum = 0
#     for diff_item in diff:
#         norm_sum += np.linalg.norm(diff_item, ord=ord)
#     return norm_sum
#
#
# with torch.no_grad():
#     total_count = 0
#     # for data in testloader_all:
#     for data in testloader_forget:
#     # for data in testloader_unforget:
#         net.eval()
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         total_count += labels.size(0)
#         checkpoint = torch.load("./model/" + targetFile)
#         net.load_state_dict(checkpoint)
#         outputs = net(images)
#         prbblt_target = np.array(torch.nn.functional.softmax(outputs).cpu())
#         for i, file in enumerate(savedFiles, 0):
#             checkpoint = torch.load("./model/" + file)
#             net.load_state_dict(checkpoint)
#             outputs = net(images)
#             prbblt_pred = np.array(torch.nn.functional.softmax(outputs).cpu())
#             norm_1s[i] += cal_norm(prbblt_pred, prbblt_target, 1)
#             norm_2s[i] += cal_norm(prbblt_pred, prbblt_target, 2)
#     for i, file in enumerate(savedFiles, 0):
#         print(file + '测试遗忘集与目标文件的第一范数距离为%.5f，第二范数距离为%.8f' % (1. * norm_1s[i] / total_count, 1. * norm_2s[i] / total_count))
#
# print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# norm_1s = []
# norm_2s = []
# for i in range(len(savedFiles)):
#     norm_1s.append(0)
#     norm_2s.append(0)
# with torch.no_grad():
#     total_count = 0
#     # for data in testloader_all:
#     # for data in testloader_forget:
#     for data in testloader_unforget:
#         net.eval()
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         total_count += labels.size(0)
#         checkpoint = torch.load("./model/" + targetFile)
#         net.load_state_dict(checkpoint)
#         outputs = net(images)
#         prbblt_target = np.array(torch.nn.functional.softmax(outputs).cpu())
#         for i, file in enumerate(savedFiles, 0):
#             checkpoint = torch.load("./model/" + file)
#             net.load_state_dict(checkpoint)
#             outputs = net(images)
#             prbblt_pred = np.array(torch.nn.functional.softmax(outputs).cpu())
#             norm_1s[i] += cal_norm(prbblt_pred, prbblt_target, 1)
#             norm_2s[i] += cal_norm(prbblt_pred, prbblt_target, 2)
#     for i, file in enumerate(savedFiles, 0):
#         print(file + '测试保留集与目标文件的第一范数距离为%.5f，第二范数距离为%.8f' % (1. * norm_1s[i] / total_count, 1. * norm_2s[i] / total_count))
#
#
# print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# norm_1s = []
# norm_2s = []
# for i in range(len(savedFiles)):
#     norm_1s.append(0)
#     norm_2s.append(0)
# with torch.no_grad():
#     total_count = 0
#     for data in testloader_all:
#     # for data in testloader_forget:
#     # for data in testloader_unforget:
#         net.eval()
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         total_count += labels.size(0)
#         checkpoint = torch.load("./model/" + targetFile)
#         net.load_state_dict(checkpoint)
#         outputs = net(images)
#         prbblt_target = np.array(torch.nn.functional.softmax(outputs).cpu())
#         for i, file in enumerate(savedFiles, 0):
#             checkpoint = torch.load("./model/" + file)
#             net.load_state_dict(checkpoint)
#             outputs = net(images)
#             prbblt_pred = np.array(torch.nn.functional.softmax(outputs).cpu())
#             norm_1s[i] += cal_norm(prbblt_pred, prbblt_target, 1)
#             norm_2s[i] += cal_norm(prbblt_pred, prbblt_target, 2)
#     for i, file in enumerate(savedFiles, 0):
#         print(file + '全部测试集与目标文件的第一范数距离为%.5f，第二范数距离为%.8f' % (1. * norm_1s[i] / total_count, 1. * norm_2s[i] / total_count))
#
#
# print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
