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


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainForgetFile = r"/train-20kinds-all.txt"
trainRetainFile = r"/train-80kinds-all.txt"
testForgetFile = r"/test-20kinds-all.txt"
testRetainFile = r"/test-80kinds-all.txt"
trainFile = r"/train-100kinds-200counts.txt"
testFile = r"/test-100kinds-all.txt"

# 2080机器
# fileRoot = r'/home/ubuntu/ml/resnet18-vggface100-2'
# dataRoot = r'/home/ubuntu/ml/resnet18_vggface2'
# datasetRoot = r'/datasets/train'

# 1080机器
fileRoot = r'/media/public/ml/resnet18-vggface100-2'
dataRoot = r'/media/public/ml/resnet18_vggface2'
datasetRoot = r'/datasets/data/root'

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
    for j in range(18 - item):
        frozenLayer = frozenLayer + layeredParams[j]
    preparedFrozenLayers.append(frozenLayer)

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default=fileRoot+'/model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--dataset_dir', type=str, default=dataRoot+datasetRoot, help='dataset directory')
parser.add_argument('--train_img_list_file', type=str, default=fileRoot+trainRetainFile,
                    help='text file containing image files used for training')
parser.add_argument('--test_img_list_file', type=str, default=fileRoot+testRetainFile,
                    help='text file containing image files used for validation, test or feature extraction')
parser.add_argument('--meta_file', type=str, default=dataRoot+r'/datasets/data/meta/identity_meta2.csv', help='meta file')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
# 超参数设置
EPOCH = 70   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
# BATCH_SIZE = 128      #批处理尺寸(batch_size)
BATCH_SIZE = 20      #批处理尺寸(batch_size)
LR = 0.1        #学习率
T_threshold = 0.0111
LR_threshold = 0.003124

args.lr_threshold = LR_threshold
# 0. id label map
meta_file = args.meta_file
id_label_dict = utils.get_id_label_map(meta_file)

# 1. data loader
root = args.dataset_dir
train_img_list_file = args.train_img_list_file
test_img_list_file = args.test_img_list_file

trainset = VGG_Faces2(root, train_img_list_file, id_label_dict, split='train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
print(len(trainset))

testset = VGG_Faces2(root, test_img_list_file, id_label_dict, split='valid')
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(len(testset))

# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
filePath = fileRoot + "/model/"
initModel = "resnet18_vgg100_normal_init.pth"
finishedModel = "resnet18_vggface100_normal_train_080_epoch.pth"
paramList, freezeParamList = generateParamsResnet18(finishedModel, initModel, layeredParams, False, filePath)
for paramIndex, param in enumerate(paramList):
    print(param)

    if paramIndex != 12:
        continue

    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                  threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                  eps=1e-08)
    fileName = filePath + param
    checkpoint = torch.load(fileName)
    net.load_state_dict(checkpoint)
    fileAccName = fileName + "_after_acc.txt"
    fileLogName = fileName + "_after_log.txt"
    fileModelName = fileName + "_after_training"
    # 冻结相关层
    frozenIndex = []
    paramCount = 0
    for name, paramItem in net.named_parameters():
        if name in freezeParamList[paramIndex]:
            frozenIndex.append(paramCount)
        paramCount = paramCount + 1
    fIndex = 0
    for paramItem in net.parameters():
        paramItem.requires_grad = True
        if fIndex in frozenIndex:
            paramItem.requires_grad = False  # 冻结网络
        fIndex = fIndex + 1
    trainFunc(net,device,trainloader,testloader,optimizer,criterion,scheduler,fileAccName,fileLogName,EPOCH,BATCH_SIZE,
              T_threshold,pre_epoch,param,args)


