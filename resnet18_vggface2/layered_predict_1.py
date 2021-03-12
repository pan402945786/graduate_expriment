import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import utils
import datasets
from resnet_1 import ResNet50
import os
from vgg_face2 import VGG_Faces2

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
# parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
# parser.add_argument('--dataset_dir', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\root', help='dataset directory')
# parser.add_argument('--train_img_list_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\train_list_100.txt',
#                     help='text file containing image files used for training')
# parser.add_argument('--test_img_list_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\test_list_100.txt',
#                     help='text file containing image files used for validation, test or feature extraction')
# parser.add_argument('--meta_file', type=str, default=r'D:\ww2\graduate_expriment/resnet18_vggface2/datasets/data/meta/identity_meta2.csv', help='meta file')
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')


parser.add_argument('--dataset_dir', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/train', help='dataset directory')
parser.add_argument('--log_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/logs/logs.log', help='log file')
parser.add_argument('--train_img_list_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/train_list2-forget2.txt',
                    help='text file containing image files used for training')
parser.add_argument('--test_img_list_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/test_list_100.txt',
                    help='text file containing image files used for validation, test or feature extraction')
parser.add_argument('--meta_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/meta/identity_meta2.csv', help='meta file')
parser.add_argument('--checkpoint_dir', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/weight/checkpoints',
                    help='checkpoints directory')
parser.add_argument('--feature_dir', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/features/test',
                    help='directory where extracted features are saved')

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--resume', type=str, default='', help='checkpoint file')
# parser.add_argument('--weight_file', type=str, default=r'D:\ww2/graduate_experiment/weight/resnet50_ft_weight.pkl', help='weight file')
parser.add_argument('--weight_file', type=str, default='', help='weight file')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--horizontal_flip', action='store_true',
                    help='horizontally flip images specified in test_img_list_file')


args = parser.parse_args()

# 超参数设置
EPOCH = 70   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.1        #学习率

# print(args)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()

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


# 0. id label map
meta_file = args.meta_file
id_label_dict = utils.get_id_label_map(meta_file)

# 1. data loader
root = args.dataset_dir
train_img_list_file = args.train_img_list_file
test_img_list_file = args.test_img_list_file

# kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
kwargs = {'num_workers': args.workers, 'pin_memory': False} if cuda else {}

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) #训练数据集
# trainset = datasets.VGG_Faces2(root, train_img_list_file, id_label_dict, split='train')
# trainset = VGG_Faces2(root, train_img_list_file, id_label_dict, split='train')
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
# print(len(trainset))

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
# testset = datasets.VGG_Faces2(root, test_img_list_file, id_label_dict, split='valid')
testset = VGG_Faces2(root, test_img_list_file, id_label_dict, split='valid')

# 输入要删除的类别
# selectedClasses = [0, 1, 2, 3, 4, 6, 7, 8]
forgetClasses = [1]
# selectedClassesStr = []
# for i in range(len(selectedClasses)):
#     selectedClassesStr.append(str(selectedClasses[i]))
# # 选择删除类别的训练数据集
# selectedTrainSet = []
# selectedTestSet = []
# for i in range(len(trainset)):
#     if trainset[i][1] in selectedClasses:
#         selectedTrainSet.append(trainset[i])
#
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

forgottenExamples = []
unforgottenExamples = []
for i, item in enumerate(testset):
    if item[1] in forgetClasses:
        forgottenExamples.append(item)
    else:
        unforgottenExamples.append(item)

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = ResNet50().to(device)
net = nn.DataParallel(net)
net = net.cuda()

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 使用相同测试集测试各个参数准确率
print("Waiting Test!")

savedFiles = [
    'resnet50_retrain_50epoch_050.pth',
    'resnet50_normal_80epoch_080.pth',
    'resnet50_normal_50epoch_050.pth'
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
        print(file + '测试分类准确率为：%.3f%%' % (100. * corrects[i] / totals[i]))

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
        print(file + '测试分类准确率为：%.3f%%' % (100. * corrects[i] / totals[i]))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
