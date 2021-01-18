import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 20   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.1        #学习率

SAMPLE_SIZE = 200

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) #训练数据集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

# 输入要删除的类别
selectedClasses = [0, 1, 2, 3, 4, 6, 7, 8]
# selectedClasses = [l]
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
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = ResNet18().to(device)

files = [
    'net_reverse_freeze_1_layer_before_training.pth',
    'net_reverse_freeze_2_layers_before_training.pth',
    'net_reverse_freeze_3_layers_before_training.pth',
    'net_reverse_freeze_5_layers_before_training.pth',
    'net_reverse_freeze_7_layers_before_training.pth',
    'net_reverse_freeze_9_layers_before_training.pth',
    'net_reverse_freeze_13_layers_before_training.pth',
    'net_reverse_freeze_17_layers_before_training.pth',
    'net_reverse_freeze_all_layers_before_training.pth',
]

savedFiles = [
    'net_reverse_freeze_1_layer_after_training.pth',
    'net_reverse_freeze_2_layers_after_training.pth',
    'net_reverse_freeze_3_layers_after_training.pth',
    'net_reverse_freeze_5_layers_after_training.pth',
    'net_reverse_freeze_7_layers_after_training.pth',
    'net_reverse_freeze_9_layers_after_training.pth',
    'net_reverse_freeze_13_layers_after_training.pth',
    'net_reverse_freeze_17_layers_after_training.pth',
    'net_reverse_freeze_all_layers_after_training.pth',
]

layeredParams = []
layeredParams.append(["conv1.0.weight", "conv1.1.weight", "conv1.1.bias"])
layeredParams.append(["layer1.0.left.0.weight", "layer1.0.left.1.weight", "layer1.0.left.1.bias", "layer1.0.left.3.weight",
                   "layer1.0.left.4.weight", "layer1.0.left.4.bias", "layer1.1.left.0.weight", "layer1.1.left.1.weight",
                   "layer1.1.left.1.bias", "layer1.1.left.3.weight", "layer1.1.left.4.weight", "layer1.1.left.4.bias",
                   ])
layeredParams.append(["layer2.0.left.0.weight", "layer2.0.left.1.weight", "layer2.0.left.1.bias", "layer2.0.left.3.weight", "layer2.0.left.4.weight", "layer2.0.left.4.bias", "layer2.0.shortcut.0.weight", "layer2.0.shortcut.1.weight", "layer2.0.shortcut.1.bias", "layer2.1.left.0.weight", "layer2.1.left.1.weight", "layer2.1.left.1.bias", "layer2.1.left.3.weight", "layer2.1.left.4.weight", "layer2.1.left.4.bias",])
layeredParams.append(["layer3.0.left.3.weight", "layer3.0.left.4.weight", "layer3.0.left.4.bias", "layer3.0.left.0.weight", "layer3.0.left.1.weight", "layer3.0.left.1.bias",])
layeredParams.append(["layer3.1.left.3.weight", "layer3.1.left.4.weight", "layer3.1.left.4.bias", "layer3.1.left.0.weight", "layer3.1.left.1.weight", "layer3.1.left.1.bias", "layer3.0.shortcut.0.weight", "layer3.0.shortcut.1.weight", "layer3.0.shortcut.1.bias",])
layeredParams.append(["layer4.0.left.3.weight", "layer4.0.left.4.weight", "layer4.0.left.4.bias", "layer4.0.left.0.weight", "layer4.0.left.1.weight", "layer4.0.left.1.bias",])
layeredParams.append(["layer4.1.left.0.weight", "layer4.1.left.1.weight", "layer4.1.left.1.bias", "layer4.0.shortcut.0.weight", "layer4.0.shortcut.1.weight", "layer4.0.shortcut.1.bias",])
layeredParams.append(["layer4.1.left.3.weight", "layer4.1.left.4.weight", "layer4.1.left.4.bias",])
layeredParams.append(["fc.weight", "fc.bias",])

# for k, file in enumerate(files, 0):
#
#     # 加载参数
#     checkpoint = torch.load("./model/" + file, map_location='cpu')
#     net.load_state_dict(checkpoint)
#
#     # 定义损失函数和优化方式
#     criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
#     optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
#                           weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
#
#     # 冻结相关层
#     frozenLayers = []
#     for j in range(k+1, 9):
#         frozenLayers = frozenLayers + layeredParams[j]
#     print(frozenLayers)
#     print("*************")
# exit()
with open("acc.txt", "w") as f:
    with open("log.txt", "w")as f2:
        for k, file in enumerate(files, 0):

            # 加载参数
            checkpoint = torch.load("./model/" + file, map_location='cpu')
            net.load_state_dict(checkpoint)

            # 定义损失函数和优化方式
            criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

            # 冻结相关层
            frozenLayers=[]
            for j in range(k+1, 9):
                frozenLayers = frozenLayers + layeredParams[j]

            frozenIndex = []
            i = 0
            for name, param in net.named_parameters():
                if name in frozenLayers:
                    frozenIndex.append(i)
                i = i + 1
            j = 0
            for param in net.parameters():
                param.requires_grad = True
                if j in frozenIndex:
                    param.requires_grad = False  # 冻结网络
                j = j + 1

            # 训练
            if __name__ == "__main__":
                best_acc = 50  # 2 初始化best test accuracy
                print("Start Training, Resnet-18! count:" + str(k))  # 定义遍历数据集的次数
                quitFlag = False
                for epoch in range(pre_epoch, EPOCH):
                    if quitFlag:
                        break
                    quitCount = 0
                    print('\nEpoch: %d' % (epoch + 1))
                    net.train()
                    sum_loss = 0.0
                    correct = 0.0
                    total = 0.0
                    for i, data in enumerate(trainloader, 0):
                        # 准备数据
                        length = len(trainloader)
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
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
                        print('[count=%d, epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                              % (k, epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                        f2.write('count=%d, %03d  %05d |Loss: %.03f | Acc: %.3f%% '
                                 % (k, epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                        f2.write('\n')
                        f2.flush()

                    # 每训练完一个epoch测试一下准确率
                    print("Waiting Test!")
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for data in testloader:
                            net.eval()
                            images, labels = data
                            images, labels = images.to(device), labels.to(device)
                            outputs = net(images)
                            # 取得分最高的那个类 (outputs.data的索引号)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                        print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                        acc = 100. * correct / total
                        # 将每次测试结果实时写入acc.txt文件中
                        f.write("count=%d,EPOCH=%03d,Accuracy= %.3f%%" % (k, epoch + 1, acc))
                        f.write('\n')
                        f.flush()
                        # 记录最佳测试分类准确率并写入best_acc.txt文件中
                        if acc > best_acc:
                            f3 = open("best_acc.txt", "a")
                            f3.write("count=%d,EPOCH=%d,best_acc= %.3f%%" % (k, epoch + 1, acc))
                            f3.write('\n')
                            f3.close()
                            best_acc = acc
                            quitCount = 0
                        else:
                            quitCount += 1
                            if quitCount > 10:
                                quitFlag = True
                                break
                print("Training Finished, TotalEPOCH=%d" % EPOCH)
                print('Saving model......')
                torch.save(net.state_dict(), '%s/%s' % (args.outf, savedFiles[k]))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
