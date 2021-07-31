import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
import sys
sys.path.append("..")
from common.resnet_100kinds_vggface2 import ResNet18
from common.lr_scheduler_temp import ReduceLROnPlateau
from common.vgg_face2 import VGG_Faces2
from common import utils
import os
import time


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trainForgetFile = r"/train-20kinds-200counts.txt"
# trainRetainFile = r"/train-80kinds-200counts.txt"
# testForgetFile = r"/test-20kinds-all.txt"
# testRetainFile = r"/test-80kinds-all.txt"

#2080
trainFile = r"/train-80kinds-all.txt"
testFile = r"/test-80kinds-all.txt"
fileRoot = r'/home/ubuntu/ml/resnet18-vggface100-2'
dataRoot = r'/home/ubuntu/ml/resnet18_vggface2'
datasetRoot = r'/datasets/train'
BATCH_SIZE = 32      #批处理尺寸(batch_size)

#1080
# trainFile = r"/train-100kinds-200counts.txt"
# testFile = r"/test_list_100_count_all.txt"
# fileRoot = r'/media/public/ml/resnet18-vggface100-2'
# dataRoot = r'/media/public/ml/resnet18_vggface2'
# datasetRoot = r'/datasets/data/root'
# BATCH_SIZE = 20      #批处理尺寸(batch_size)

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--dataset_dir', type=str, default=dataRoot+datasetRoot, help='dataset directory')
parser.add_argument('--train_img_list_file', type=str, default=fileRoot+trainFile,
                    help='text file containing image files used for training')
parser.add_argument('--test_img_list_file', type=str, default=fileRoot+testFile,
                    help='text file containing image files used for validation, test or feature extraction')
parser.add_argument('--meta_file', type=str, default=dataRoot+r'/datasets/data/meta/identity_meta2.csv', help='meta file')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
# print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) # 这么定义不管用，必须要显示定义 CUDA_VISIBLE_DEVICES=0 python xxx.py
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
# 超参数设置
EPOCH = 80   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
# BATCH_SIZE = 128      #批处理尺寸(batch_size)

LR = 0.1        #学习率
T_threshold = 0.01

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
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
# net = ResNet18().to(device)
net = ResNet18()

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-08)
# print('Saving model......')
# torch.save(net.state_dict(), '%s/resnet18_vgg100_normal_init.pth' % (args.outf))

# 训练
if __name__ == "__main__":
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("resnet18_vgg80_retrain_allcounts_acc.txt", "a+") as f:
        with open("resnet18_vgg80_retrain_allcounts_log.txt", "a+")as f2:
            for epoch in range(pre_epoch, EPOCH):
                # scheduler.step()
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
                    # inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    optimizer.zero_grad()

                    # forward + backward
                    net = net.cuda()
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
                          % (epoch+1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "null",
                             optimizer.state_dict()['param_groups'][0]['lr']))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | Time: %s | LR: %.6f'
                             % (epoch+1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                optimizer.state_dict()['param_groups'][0]['lr']))
                    f2.write('\n')
                    f2.flush()
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    sum_loss = 0
                    for i, data in enumerate(testloader):
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
                        lastLoss = sum_loss / (i + 1)
                    print('测试分类准确率为：%.3f%%, 当前学习率： %.3f, last test loss: %.3f' % (100. * correct / total, optimizer.state_dict()['param_groups'][0]['lr'], lastLoss))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    if (epoch+1) % 10 < 1:
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/resnet18_vggface80_retrain_%03d_epoch.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,Time=%s,LR=%.6f,BATCH_SIZE:%d,lastTestLoss:%.3f" % (
                    epoch+1, acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    optimizer.state_dict()['param_groups'][0]['lr'], BATCH_SIZE, lastLoss))
                    f.write('\n')
                    f.flush()
                scheduler.step(lastLoss, epoch=epoch)
                if lastLoss < T_threshold:
                    print('loss达到限值%s，提前退出' % lastLoss)
                    print('Saving model......')
                    torch.save(net.state_dict(),
                               '%s/resnet18_vggface80_retrain_%03d_epoch.pth' % (args.outf, epoch + 1))
                    break
            print('Saving model......')
            torch.save(net.state_dict(), '%s/resnet18_vggface80_retrain_%03d_epoch.pth' % (args.outf, epoch + 1))
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

