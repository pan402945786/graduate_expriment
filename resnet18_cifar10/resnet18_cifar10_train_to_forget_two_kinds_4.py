import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
from common.resnet_1 import ResNet18
from common.lr_scheduler_temp import ReduceLROnPlateau
import time


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--start_num', type=int, default=1, help='start number')
parser.add_argument('--end_num', type=int, default=18, help='end number to forget')
args = parser.parse_args()

# 超参数设置
EPOCH = 30   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
# BATCH_SIZE = 128      #批处理尺寸(batch_size)
# BATCH_SIZE = 128      #批处理尺寸(batch_size)
BATCH_SIZE = args.batch_size      #批处理尺寸(batch_size)
LR = 0.1        #学习率

reset_layer_start = args.start_num
reset_layer_end = args.end_num

layer_count_list = []
layer_count = reset_layer_start
while layer_count < reset_layer_end:
    layer_count_list.append(layer_count)
    layer_count = layer_count + 1
layer_count_list.append(reset_layer_end)

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train) #训练数据集
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)

# 输入要删除的类别
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

trainloader = torch.utils.data.DataLoader(selectedTrainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
testloader = torch.utils.data.DataLoader(selectedTestSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 模型定义-ResNet
net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题

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
layeredParams.append([ "layer3.1.left.0.weight", "layer3.1.left.1.weight", "layer3.1.left.1.bias",])
layeredParams.append(["layer3.1.left.3.weight", "layer3.1.left.4.weight", "layer3.1.left.4.bias"])

layeredParams.append(["layer4.0.left.0.weight", "layer4.0.left.1.weight", "layer4.0.left.1.bias",])
layeredParams.append(["layer4.0.left.3.weight", "layer4.0.left.4.weight", "layer4.0.left.4.bias", "layer4.0.shortcut.0.weight", "layer4.0.shortcut.1.weight", "layer4.0.shortcut.1.bias",])
layeredParams.append(["layer4.1.left.0.weight", "layer4.1.left.1.weight", "layer4.1.left.1.bias",])
layeredParams.append(["layer4.1.left.3.weight", "layer4.1.left.4.weight", "layer4.1.left.4.bias",])

layeredParams.append(["fc.weight", "fc.bias",])

preparedFrozenLayers = []
for i, item in enumerate(layer_count_list):
    frozenLayer = []
    for j in range(18-item):
        frozenLayer = frozenLayer + layeredParams[j]
    preparedFrozenLayers.append(frozenLayer)

savedFiles = [
    'resnet18_cifar10_fc_before_training.pth',
    'resnet18_cifar10_fc_conv1_before_training.pth',
    'resnet18_cifar10_fc_conv2_before_training.pth',
    'resnet18_cifar10_fc_conv3_before_training.pth',
    'resnet18_cifar10_fc_conv4_before_training.pth',
    'resnet18_cifar10_fc_conv5_before_training.pth',
    'resnet18_cifar10_fc_conv6_before_training.pth',
    'resnet18_cifar10_fc_conv7_before_training.pth',
    'resnet18_cifar10_fc_conv8_before_training.pth',
    'resnet18_cifar10_fc_conv9_before_training.pth',
    'resnet18_cifar10_fc_conv10_before_training.pth',
    'resnet18_cifar10_fc_conv11_before_training.pth',
    'resnet18_cifar10_fc_conv12_before_training.pth',
    'resnet18_cifar10_fc_conv13_before_training.pth',
    'resnet18_cifar10_fc_conv14_before_training.pth',
    'resnet18_cifar10_fc_conv15_before_training.pth',
    'resnet18_cifar10_fc_conv16_before_training.pth',
    'resnet18_cifar10_fc_conv17_before_training.pth',
]

# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    for i, item in enumerate(layer_count_list):
        # optimizer初始化
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                              weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
        # scheduler初始化
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True,
                                      threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                      eps=1e-08)
        # 网络参数初始化
        checkpoint = torch.load(args.outf + savedFiles[item-1], map_location='cpu')
        net.load_state_dict(checkpoint)
        print('load files:')
        print(savedFiles[item-1])

        # 冻结相关层
        frozenIndex = []
        paramCount = 0
        for name, param in net.named_parameters():
            if name in preparedFrozenLayers[i]:
                frozenIndex.append(paramCount)
            paramCount = paramCount + 1
        j = 0
        for param in net.parameters():
            param.requires_grad = True
            if j in frozenIndex:
                param.requires_grad = False  # 冻结网络
            j = j + 1
        with open(savedFiles[item-1] + "_acc_second_time.txt", "a+") as f:
            with open(savedFiles[item-1] + "_log_second_time.txt", "a+")as f2:
                for epoch in range(pre_epoch+1, EPOCH+1):

                    # if optimizer.state_dict()['param_groups'][0]['lr'] < 0.006260:
                    #     break

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
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Time: %s | File: %s | LR: %.6f'
                              % (epoch, (k + 1 + (epoch-1) * length), sum_loss / (k + 1), 100. * correct / total, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), savedFiles[item-1], optimizer.state_dict()['param_groups'][0]['lr']))
                        f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | Time: %s | LR: %.6f'
                              % (epoch, (k + 1 + (epoch-1) * length), sum_loss / (k + 1), 100. * correct / total, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), optimizer.state_dict()['param_groups'][0]['lr']))
                        f2.write('\n')
                        f2.flush()

                    # 每训练完一个epoch测试一下准确率
                    print("Waiting Test!")
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        loss_val_sum = 0.0
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
                        print('测试分类准确率为：%.3f%%, 当前学习率： %.6f' % (
                        100. * correct / total, optimizer.state_dict()['param_groups'][0]['lr']))
                        acc = 100. * correct / total
                        # 将每次测试结果实时写入acc.txt文件中
                        if epoch % 10 < 1 and pre_epoch != epoch:
                            print('Saving model......')
                            torch.save(net.state_dict(), args.outf + '/' + savedFiles[item - 1] + '_finetuning_' + str(epoch) + '_second_time.pth')
                        f.write("EPOCH=%03d,Accuracy= %.3f%%,Time=%s,LR=%.6f,BATCH_SIZE:%d" % (epoch, acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), optimizer.state_dict()['param_groups'][0]['lr'], BATCH_SIZE))
                        f.write('\n')
                        f.flush()
                    scheduler.step(1.0 * loss_val_sum / total, epoch=epoch)
                print('Saving model......')
                torch.save(net.state_dict(),
                           args.outf + '/' + savedFiles[item - 1] + '_forget_two_kinds_after_finetuning_' + str(epoch) + '_second_time.pth')
                print("Training Finished, TotalEPOCH=%d" % EPOCH)

