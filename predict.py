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
EPOCH = 135   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 加载原有网络参数
checkpoint = torch.load("./model/net_135.pth", map_location='cpu')
net.load_state_dict(checkpoint)
print("original:", net.state_dict())
# 预测
normal_result = []
normal_labels = []
limit = 20
with torch.no_grad():
    correct = 0
    total = 0
    for i,data in enumerate(testloader,0):
        if i > limit:
            break
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        print(predicted, labels)
        normal_result.append(predicted)
        normal_labels.append(labels[0])
    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
    acc = 100. * correct / total

# 加载删除网络参数
forget_result = []
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
checkpoint = torch.load("./model/net_forget_0-1_020.pth", map_location='cpu')
net.load_state_dict(checkpoint)
print("forget:", net.state_dict())
# 预测
with torch.no_grad():
    correct = 0
    total = 0
    for i,data in enumerate(testloader,0):
        if i > limit:
            break
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        print(predicted, labels)
        forget_result.append(predicted)
    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
    acc = 100. * correct / total
# 综合预测结果，输出最终预测类别
# final_result = []
# for i in range(len(normal_result)):
#     if normal_result[i] != forget_result[i]:
#         final_result.append(normal_result[i][0])
#     else:
#         final_result.append(None)
# print("final result:", final_result)
# print("labels:      ", normal_labels)



