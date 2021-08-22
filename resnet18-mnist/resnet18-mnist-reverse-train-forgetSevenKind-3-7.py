import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
from common.utils import generateReverseParamsResnet18
from common.resnet_for_mnist import ResNet18
import os
import time
from common.lr_scheduler_temp import ReduceLROnPlateau
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainForgetFile = r"/train-20kinds-all.txt"
trainRetainFile = r"/train-80kinds-all.txt"
testForgetFile = r"/test-20kinds-all.txt"
testRetainFile = r"/test-80kinds-all.txt"
trainFile = r"/train_list_100.txt"
testFile = r"/test_list_100.txt"

# 2080机器
# fileRoot = r'/home/ubuntu/ml/resnet18-vggface100-2'
# dataRoot = r'/home/ubuntu/ml/resnet18_vggface2'
# datasetRoot = r'/datasets/train'

# 1080机器
# fileRoot = r'/media/public/ml/resnet18-vggface100-2'
# dataRoot = r'/media/public/ml/resnet18_vggface2'
# datasetRoot = r'/datasets/data/root'

# 实验室台式机
fileRoot = r'D:\ww2\graduate_expriment\resnet18-mnist'
# dataRoot = r'D:\ww2\graduate_expriment\resnet18_vggface2'
# datasetRoot = r'\datasets\data\root'
# trainForgetFile = r"\train-20kinds-all.txt"
# trainRetainFile = r"\train-80kinds-all.txt"
# testForgetFile = r"\test-20kinds-all.txt"
# testRetainFile = r"\test-80kinds-all.txt"
# trainFile = r"\train_list_100.txt"
# testFile = r"\test_list_100.txt"

# 自己电脑
# fileRoot = r'D:\www\graduate_expriment\resnet18-mnist'
# dataRoot = r'D:\www\graduate_expriment\resnet18_vggface2'
# datasetRoot = r'\datasets\data\root'

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
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
# print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
# 超参数设置
EPOCH = 10   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
# BATCH_SIZE = 128      #批处理尺寸(batch_size)
BATCH_SIZE = 32      #批处理尺寸(batch_size)
LR = 0.005        #学习率
T_threshold = 0.0111
saveModelSpan = 10
tolerate = 10

# 准备数据集并预处理
forget = [1,2,3,4,5,6,9]

train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
print(len(train_ds))
test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())
print(len(test_ds))

train_set_forget_three = []
test_set_forget_three = []
train_set_forget_set = []
test_set_forget_set = []
for item in train_ds:
    data, label = item
    if label not in forget:
        train_set_forget_three.append(item)
    else:
        train_set_forget_set.append(item)
for item in test_ds:
    data, label = item
    if label not in forget:
        test_set_forget_three.append(item)
    else:
        test_set_forget_set.append(item)
print(len(train_set_forget_three))
print(len(test_set_forget_three))
trainloader = DataLoader(train_set_forget_three, batch_size=64, shuffle=True)
testloader = DataLoader(test_set_forget_three, batch_size=64)
forgetTestLoader = DataLoader(train_set_forget_set + test_set_forget_set, batch_size=64, shuffle=True)

# 模型定义-ResNet
net = ResNet18().to(device)

# checkpoint = torch.load(r"D:\ww2\graduate_expriment\resnet18-vggface100-2\model\resnet18_vggface100_reverse_reset_former_13_before_training.pth_best_acc_model_20210706.pth", map_location='cpu')
# checkpoint = torch.load(r"D:\ww2\graduate_expriment\resnet18-vggface100-2\model\resnet18_vggface100_normal_train_080_epoch.pth", map_location='cpu')
# checkpoint = torch.load(r"D:\ww2\graduate_expriment\resnet18-vggface100-2\model\resnet18_vgg100_normal_init.pth", map_location='cpu')
# net.load_state_dict(checkpoint)
# paramsparams = net.state_dict()
# for k, v in paramsparams.items():
#     if k == 'layer4.0.left.0.weight':
#         print(k)  # 打印网络中的变量名
#         print(v)
#         break
# exit()
# 定义损失函数和优化方式

criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
filePath = fileRoot + "/model/"
initModel = "resnet18_mnist_noraml_train_init.pth"
finishedModel = "resnet18_mnist_normal_train_20.pth"
# paramList, freezeParamList = generateParamsResnet18(initModel,finishedModel, layeredParams, True, filePath)
strucName = 'resnet18_'
datasetName = 'mnist_forget_seven_kind_'
paramList, freezeParamList = generateReverseParamsResnet18(net, initModel,finishedModel, layeredParams, filePath,
                                                           strucName, datasetName, range(1, 18))
# paramList.reverse()
# freezeParamList.reverse()
# print(paramList)
# print(freezeParamList)
# exit()
print("begin cycle")
for paramIndex, param in enumerate(paramList):
    print(param)
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
    #                       weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    optimizer = torch.optim.RMSprop(net.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                  threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                  eps=1e-08)
    fileName = filePath + param
    checkpoint = torch.load(fileName)
    net.load_state_dict(checkpoint)
    fileAccName = fileName + "_forget_seven_kind_after_acc.txt"
    fileLogName = fileName + "_forget_seven_kind_after_log.txt"
    fileModelName = fileName + "_forget_seven_kind_after_training"
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
    # 训练
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    best_acc = 0
    tolerate = 10
    with open(fileAccName, "a+") as f:
        with open(fileLogName, "a+")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                lastLoss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    lastTrainLoss = sum_loss / (i + 1)
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Time: %s | File: %s | LR: %.6f'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), param,
                             optimizer.state_dict()['param_groups'][0]['lr']))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | Time: %s | LR: %.6f'
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                optimizer.state_dict()['param_groups'][0]['lr']))
                    f2.write('\n')
                    f2.flush()
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0.0
                    total = 0.0
                    sum_loss = 0
                    for iTest, data in enumerate(testloader):
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        loss = criterion(outputs, labels)
                        sum_loss += loss.item()
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                        lastLoss = sum_loss / (iTest + 1)
                    print('测试分类准确率为：%.3f%%, 当前学习率： %.3f, last loss: %.3f' % (
                        100. * correct / total, optimizer.state_dict()['param_groups'][0]['lr'], lastLoss))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    if acc > best_acc:
                        best_acc = acc
                        print('Saving best acc model......')
                        torch.save(net.state_dict(), '%s/%s_best_acc_model.pth' % (
                            args.outf, param))
                        f.write("save best model\n")
                        f.flush()
                    if (epoch + 1) % saveModelSpan < 1:
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/%s_%03d_epoch.pth' % (
                            args.outf, param.replace("before", "after"), epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,Time=%s,LR=%.6f,BATCH_SIZE:%d,lastLoss:%.3f" % (
                        epoch + 1, acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        optimizer.state_dict()['param_groups'][0]['lr'], BATCH_SIZE, lastLoss))
                    f.write('\n')
                    f.flush()
                    # 遗忘集测试准确率
                    correct = 0.0
                    total = 0.0
                    for iTestForget, data in enumerate(forgetTestLoader):
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    acc = 100. * correct / total
                    print('遗忘集测试分类准确率为：%.3f%%' % acc)
                    f.write('遗忘集测试分类准确率为：%.3f%%' % acc)
                    f.write('\n')
                    f.flush()

                scheduler.step(lastLoss, epoch=epoch)
                if lastTrainLoss < T_threshold and epoch > tolerate:
                    print('train loss达到限值%s，提前退出' % lastTrainLoss)
                    print('Saving model......')
                    torch.save(net.state_dict(),
                               '%s/%s_%03d_epoch.pth' % (args.outf, param.replace("before", "after"), epoch + 1))
                    f.write("train loss达到限值%s，提前退出" % lastTrainLoss)
                    f.write('\n')
                    f.flush()
                    break
                if optimizer.state_dict()['param_groups'][0]['lr'] < 0.003:
                    print("学习率过小，退出")
                    f.write("学习率过小，退出")
                    f.write('\n')
                    f.flush()
                    break
            print('Saving model......')
            torch.save(net.state_dict(),
                       '%s/%s_%03d_epoch.pth' % (args.outf, param.replace("before", "after"), epoch + 1))
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

