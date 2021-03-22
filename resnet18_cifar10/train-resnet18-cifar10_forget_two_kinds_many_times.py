import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
from common.resnet_1 import ResNet18
from common.resnet_1 import ResNet50
import os
import time
import collections
from common.lr_scheduler_temp import ReduceLROnPlateau

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--start', type=int, default=0, help='batch size')
parser.add_argument('--end', type=int, default=30, help='batch size')

args = parser.parse_args()
print(args)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
cuda = torch.cuda.is_available()

if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

# 超参数设置
EPOCH = 30   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
# BATCH_SIZE = 128      #批处理尺寸(batch_size)
# BATCH_SIZE = 40      #批处理尺寸(batch_size)
BATCH_SIZE = args.batch_size      #批处理尺寸(batch_size)
LR = 0.1        #学习率
REPEAT_TIMES_START = 0
REPEAT_TIMES_END = 30

# 模型定义-ResNet
net = ResNet18().to(device)
# net = ResNet50().to(device)
# net = nn.DataParallel(net)
# net = net.cuda()
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
print('Saving model......')
torch.save(net.state_dict(), '%s/resnet18_cifar10_forget_two_kinds_init.pth' % (args.outf))

# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    # 输入要保留的类别,删除了第8和第9个类别
    selectedClasses = [0, 1, 2, 3, 4, 5, 6, 7]
    selectedClassesStr = []
    for i in range(len(selectedClasses)):
        selectedClassesStr.append(str(selectedClasses[i]))
    # 选择删除类别的训练数据集
    selectedTrainSet = []
    selectedTestSet = []
    for i in range(len(trainset)):
        if trainset[i][1] in selectedClasses:
            selectedTrainSet.append(trainset[i])

    for i in range(len(testset)):
        if testset[i][1] in selectedClasses:
            selectedTestSet.append(testset[i])

    trainloader = torch.utils.data.DataLoader(selectedTrainSet, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
    testloader = torch.utils.data.DataLoader(selectedTestSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(len(selectedTrainSet))
    print(len(selectedTestSet))
    # optimizer初始化
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # scheduler初始化
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True,
                                  threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                  eps=1e-08)

    for this_time in range(REPEAT_TIMES_START, REPEAT_TIMES_END):
        with open("resnet18_cifar10_forget_two_kinds_acc_" + str(this_time) + "_time.txt", "a+") as f:
            with open("resnet18_cifar10_forget_two_kinds_log_" + str(this_time) + "_time.txt", "a+")as f2:
                for epoch in range(pre_epoch+1, EPOCH+1):
                    # scheduler.step()
                    print('\nEpoch: %d' % epoch)
                    net.train()
                    sum_loss = 0.0
                    correct = 0.0
                    total = 0.0
                    for k, data in enumerate(trainloader, 0):
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
                        print('[time:%d, epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Time: %s | LR: %.06f'
                              % (this_time, epoch, (k + 1 + (epoch-1) * length), sum_loss / (k + 1), 100. * correct / total, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), optimizer.state_dict()['param_groups'][0]['lr']))
                        f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | Time: %s | LR: %.06f'
                              % (epoch, (k + 1 + (epoch-1) * length), sum_loss / (k + 1), 100. * correct / total, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), optimizer.state_dict()['param_groups'][0]['lr']))
                        f2.write('\n')
                        f2.flush()

                    # 每训练完一个epoch测试一下准确率
                    print("Waiting Test!")
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        loss_val_sum = 0.
                        for data in testloader:
                            net.eval()
                            images, labels = data
                            images, labels = images.to(device), labels.to(device)
                            outputs = net(images)
                            loss_val = criterion(outputs, labels)
                            # 取得分最高的那个类 (outputs.data的索引号)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                            loss_val_sum += loss_val
                        print('测试分类准确率为：%.3f%%, 当前学习率： %.6f' % (100. * correct / total, optimizer.state_dict()['param_groups'][0]['lr']))
                        acc = 100. * correct / total
                        # 将每次测试结果实时写入acc.txt文件中
                        if epoch % 10 < 1 and pre_epoch != epoch:
                            print('Saving model......')
                            torch.save(net.state_dict(), args.outf + '/resnet18_cifar10_forget_two_kinds_'+str(epoch)+'_'+ str(this_time) +'_time_.pth')
                        f.write("EPOCH=%03d,Accuracy= %.3f%%,Time=%s,LR=%.6f" % (epoch, acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), optimizer.state_dict()['param_groups'][0]['lr']))
                        f.write('\n')
                        f.flush()
                    scheduler.step(1. * loss_val_sum / total, epoch=epoch)
                print('Saving model......')
                torch.save(net.state_dict(), args.outf + '/resnet18_cifar10_forget_two_kinds_finished_saving_'+str(epoch)+'_'+ str(this_time) +'_time_.pth')
                print("Training Finished, TotalEPOCH=%d" % EPOCH)

