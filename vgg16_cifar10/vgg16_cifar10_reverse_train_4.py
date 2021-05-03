import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import sys
sys.path.append("..")
from common.lr_scheduler_temp import ReduceLROnPlateau
from common.vgg import VGG

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 60   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 100      #批处理尺寸(batch_size)
LR = 0.1        #学习率

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

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train) #训练数据集
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

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

trainloader = torch.utils.data.DataLoader(selectedTrainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
testloader = torch.utils.data.DataLoader(selectedTestSet, batch_size=100, shuffle=False, num_workers=2)

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = VGG('VGG16').to(device)

files = [
    # 'vgg16_cifar10_reverse_reset_conv1_before_training.pth',  # 需要1个卷积层
    # 'vgg16_cifar10_reverse_reset_conv2_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv3_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv4_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv5_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv6_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv7_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv8_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv9_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv10_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv11_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv12_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv13_before_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv14_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv1_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv2_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv3_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv4_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv5_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv6_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv7_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv8_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv9_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv10_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv11_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv12_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv13_with_running_mean_var_before_training.pth',
    'vgg16_cifar10_reverse_reset_conv14_with_running_mean_var_before_training.pth',

]

savedFiles = [
    # 'vgg16_cifar10_reverse_reset_conv1_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv2_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv3_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv4_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv5_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv6_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv7_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv8_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv9_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv10_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv11_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv12_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv13_after_training.pth',
    # 'vgg16_cifar10_reverse_reset_conv14_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv1_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv2_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv3_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv4_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv5_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv6_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv7_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv8_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv9_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv10_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv11_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv12_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv13_with_running_mean_var_after_training.pth',
    'vgg16_cifar10_reverse_reset_conv14_with_running_mean_var_after_training.pth',
]

layeredParams = []
layeredParams.append(["classifier.weight", "classifier.bias", ])
layeredParams.append(["features.40.weight", "features.40.bias", "features.41.weight", "features.41.bias", ])
layeredParams.append(["features.37.weight", "features.37.bias", "features.38.weight", "features.38.bias", ])
layeredParams.append(["features.34.weight", "features.34.bias", "features.35.weight", "features.35.bias", ])
layeredParams.append(["features.30.weight", "features.30.bias", "features.31.weight", "features.31.bias", ])
layeredParams.append(["features.27.weight", "features.27.bias", "features.28.weight", "features.28.bias", ])
layeredParams.append(["features.24.weight", "features.24.bias", "features.25.weight", "features.25.bias", ])
layeredParams.append(["features.20.weight", "features.20.bias", "features.21.weight", "features.21.bias", ])
layeredParams.append(["features.17.weight", "features.17.bias", "features.18.weight", "features.18.bias", ])
layeredParams.append(["features.14.weight", "features.14.bias", "features.15.weight", "features.15.bias", ])
layeredParams.append(["features.10.weight", "features.10.bias", "features.11.weight", "features.11.bias", ])
layeredParams.append(["features.7.weight", "features.7.bias", "features.8.weight", "features.8.bias", ])
layeredParams.append(["features.3.weight", "features.3.bias", "features.4.weight", "features.4.bias", ])
layeredParams.append(["features.0.weight", "features.0.bias", "features.1.weight", "features.1.bias", ])


for fileIndex, file in enumerate(files, 0):
    # if k < 9:
    #     print("continue k:" + str(k))
    #     continue
    with open(file+"_acc.txt", "a") as f:
        with open(file+"_log.txt", "a")as f2:
            # 加载参数
            checkpoint = torch.load("./model/" + file, map_location='cpu')
            net.load_state_dict(checkpoint)

            # 定义损失函数和优化方式
            criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True,
                                          threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                          eps=1e-08)
            # 冻结相关层
            frozenLayers = []
            for j in range(0, len(layeredParams)-fileIndex-1):
                frozenLayers = frozenLayers + layeredParams[j]
            # print('frozen layers:')
            # print(frozenLayers)
            frozenIndex = []
            i = 0
            for name, param in net.named_parameters():
                # print(name)
                if name in frozenLayers:
                    frozenIndex.append(i)
                i = i + 1
            j = 0

            for param in net.parameters():
                param.requires_grad = True
                if j in frozenIndex:
                    param.requires_grad = False  # 冻结网络
                j = j + 1

            # for param in net.parameters():
            #     print(param.requires_grad)
            # exit()
            # 训练
            for epoch in range(pre_epoch + 1, EPOCH + 1):
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
                    print('[layers:%d, epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Time: %s | LR: %.06f'
                          % (fileIndex, epoch, (k + 1 + (epoch - 1) * length), sum_loss / (k + 1), 100. * correct / total,
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                             optimizer.state_dict()['param_groups'][0]['lr']))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | Time: %s | LR: %.06f'
                             % (epoch, (k + 1 + (epoch - 1) * length), sum_loss / (k + 1), 100. * correct / total,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                optimizer.state_dict()['param_groups'][0]['lr']))
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
                    print('测试分类准确率为：%.3f%%, 当前学习率： %.6f' % (
                    100. * correct / total, optimizer.state_dict()['param_groups'][0]['lr']))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    # if epoch % 10 < 1 and pre_epoch != epoch:
                    #     print('Saving model......')
                    #     torch.save(net.state_dict(), args.outf + '/' + savedFiles[fileIndex] + '_' + str(epoch) + '_epochs.pth')
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,Time=%s,LR=%.6f,BATCH_SIZE:%d" % (
                        epoch, acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        optimizer.state_dict()['param_groups'][0]['lr'], BATCH_SIZE))
                    f.write('\n')
                    f.flush()
                scheduler.step(1. * loss_val_sum / total, epoch=epoch)
            print('Saving model......')
            torch.save(net.state_dict(), args.outf + '/' + savedFiles[fileIndex])
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

print("+++++++++++++++++++++++++++++END+++++++++++++++++++++++++++++++++++")
