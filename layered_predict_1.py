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
EPOCH = 70   #遍历数据集次数
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
selectedClassesStr = []
for i in range(len(selectedClasses)):
    selectedClassesStr.append(str(selectedClasses[i]))
# 选择删除类别的训练数据集
selectedTrainSet = []
selectedTestSet = []
for i in range(len(trainset)):
    if trainset[i][1] in selectedClasses:
        selectedTrainSet.append(trainset[i])

trainloader = torch.utils.data.DataLoader(selectedTrainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


forgottenExamples = []
unforgottenExamples = []
for i in range(len(testset)):
    if testset[i][1] in selectedClasses:
        forgottenExamples.append(testset[i])
    else:
        unforgottenExamples.append(testset[i])

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 使用相同测试集测试各个参数准确率
print("Waiting Test!")

# savedFiles = [
#     'net_change_1_layer_after_training.pth',
#     'net_change_2_layers_after_training.pth',
#     'net_change_3_layers_after_training.pth',
#     'net_change_5_layers_after_training.pth',
#     'net_change_7_layers_after_training.pth',
#     'net_change_9_layers_after_training.pth',
#     'net_change_13_layers_after_training.pth',
#     'net_change_17_layers_after_training.pth',
#     'net_change_all_layers_after_training.pth',
# ]

savedFiles = [
    'net_frozen_20_1_layer_after_training.pth',
    'net_frozen_20_2_layers_after_training.pth',
    'net_frozen_20_3_layers_after_training.pth',
    'net_frozen_20_5_layers_after_training.pth',
    'net_frozen_20_7_layers_after_training.pth',
    'net_frozen_20_9_layers_after_training.pth',
    'net_frozen_20_13_layers_after_training.pth',
    'net_frozen_20_17_layers_after_training.pth',
    'net_frozen_20_all_layers_after_training.pth',
]

testloader = torch.utils.data.DataLoader(unforgottenExamples, batch_size=100, shuffle=False, num_workers=2)

totals = [0,0,0,0,0,0,0,0,0]
corrects = [0,0,0,0,0,0,0,0,0]
with torch.no_grad():
    for data in testloader:
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
        print(file + '测试分类准确率为：%.3f%%' % (100 * corrects[i] / totals[i]))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

testloader = torch.utils.data.DataLoader(forgottenExamples, batch_size=100, shuffle=False, num_workers=2)

totals = [0,0,0,0,0,0,0,0,0]
corrects = [0,0,0,0,0,0,0,0,0]
with torch.no_grad():
    for data in testloader:
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
        print(file + '测试分类准确率为：%.3f%%' % (100 * corrects[i] / totals[i]))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
