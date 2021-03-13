import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
from resnet_1 import ResNet18
import datasets
import os
import time
import utils
from vgg_face2 import VGG_Faces2
from lr_scheduler_temp import ReduceLROnPlateau

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='../model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='../model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径

# parser.add_argument('cmd', type=str,  choices=['train', 'test', 'extract'], help='train, test or extract')
parser.add_argument('--arch_type', type=str, default='resnet50_ft', help='model type',
                    choices=['resnet50_ft', 'senet50_ft', 'resnet50_scratch', 'senet50_scratch'])

# parser.add_argument('--dataset_dir', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\root', help='dataset directory')
# parser.add_argument('--log_file', type=str, default=r'D:\ww2/graduate_experiment/logs/logs.log', help='log file')
# parser.add_argument('--train_img_list_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\train_list_100.txt',
#                     help='text file containing image files used for training')
# parser.add_argument('--test_img_list_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\test_list_100.txt',
#                     help='text file containing image files used for validation, test or feature extraction')
#
# parser.add_argument('--meta_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\meta/identity_meta2.csv', help='meta file')
# parser.add_argument('--checkpoint_dir', type=str, default=r'D:\ww2/graduate_experiment/resnet18_vggface2\weight/checkpoints',
#                     help='checkpoints directory')
# parser.add_argument('--feature_dir', type=str, default=r'D:\ww2/graduate_experiment/resnet18_vggface2\features/test',
#                     help='directory where extracted features are saved')

parser.add_argument('--dataset_dir', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/train', help='dataset directory')
parser.add_argument('--log_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/logs/logs.log', help='log file')
parser.add_argument('--train_img_list_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/train_list_100.txt',
                    help='text file containing image files used for training')
# parser.add_argument('--train_img_list_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/train_list_100_less_for_test.txt',
#                     help='text file containing image files used for training')
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
# print(args)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
# 超参数设置
EPOCH = 70   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
# BATCH_SIZE = 128      #批处理尺寸(batch_size)
BATCH_SIZE = 30      #批处理尺寸(batch_size)
LR = 0.1        #学习率

# 0. id label map
meta_file = args.meta_file
id_label_dict = utils.get_id_label_map(meta_file)

# 1. data loader
root = args.dataset_dir
train_img_list_file = args.train_img_list_file
test_img_list_file = args.test_img_list_file

# kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
kwargs = {'num_workers': args.workers, 'pin_memory': False} if cuda else {}


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

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) #训练数据集
# trainset = datasets.VGG_Faces2(root, train_img_list_file, id_label_dict, split='train')
trainset = VGG_Faces2(root, train_img_list_file, id_label_dict, split='train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
print(len(trainset))

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
# testset = datasets.VGG_Faces2(root, test_img_list_file, id_label_dict, split='valid')
testset = VGG_Faces2(root, test_img_list_file, id_label_dict, split='valid')
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = ResNet18().to(device)
# net = ResNet50().to(device)
# net = nn.DataParallel(net)
net = net.cuda().float()

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
# scheduler = torch.optim.llr_scheduler.MultiStepLR(optimizer,milestones=[20, 40, 60],gamma = 0.2)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-08)
print('Saving model......')
torch.save(net.state_dict(), '%s/resnet18_vgg100_normal_init.pth' % (args.outf))

# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("resnet18_vgg100_normal_train_acc.txt", "a+") as f:
        with open("resnet18_vgg100_normal_train_log.txt", "a+")as f2:
            for epoch in range(pre_epoch, EPOCH):
                # scheduler.step()
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
                    # inputs = inputs.float().cuda()
                    # labels = labels.cuda()
                    optimizer.zero_grad()

                    # forward + backward
                    net = net.cuda().float()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.cuda.synchronize()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Time: %s'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) )
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | Time: %s'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
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
                        loss_val = criterion(outputs, labels)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%, 当前学习率： %.3f' % (100. * correct / total, optimizer.state_dict()['param_groups'][0]['lr']))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    if epoch % 5 < 1:
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/resnet18_vggface100_normal_%03d_epoch.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,Time=%s,LR=%.3f" % (epoch + 1, acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), optimizer.state_dict()['param_groups'][0]['lr']))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
                scheduler.step(loss_val, epoch=epoch)
            print('Saving model......')
            torch.save(net.state_dict(), '%s/resnet18_vggface100_normal_70epoch_%03d.pth' % (args.outf, epoch + 1))
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

