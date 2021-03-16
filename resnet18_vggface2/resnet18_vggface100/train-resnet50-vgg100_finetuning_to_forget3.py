import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
from resnet_1 import ResNet18
from resnet_1 import ResNet50
import datasets
import os
import time
import utils
import collections
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

# parser.add_argument('--dataset_dir', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/train', help='dataset directory')
# parser.add_argument('--log_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/logs/logs.log', help='log file')
# parser.add_argument('--train_img_list_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/train_list_100.txt',
#                     help='text file containing image files used for training')
# # parser.add_argument('--train_img_list_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/train_list_100_less_for_test.txt',
# #                     help='text file containing image files used for training')
# parser.add_argument('--test_img_list_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/test_list_100.txt',
#                     help='text file containing image files used for validation, test or feature extraction')
# parser.add_argument('--meta_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/meta/identity_meta2.csv', help='meta file')
# parser.add_argument('--checkpoint_dir', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/weight/checkpoints',
#                     help='checkpoints directory')
# parser.add_argument('--feature_dir', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/features/test',
#                     help='directory where extracted features are saved')

parser.add_argument('--dataset_dir', type=str, default=r'/media/public/ml/resnet18_vggface2/datasets/data/root', help='dataset directory')
parser.add_argument('--log_file', type=str, default=r'/media/public/ml/resnet18_vggface2/logs/logs.log', help='log file')
parser.add_argument('--train_img_list_file', type=str, default=r'/media/public/ml/resnet18_vggface2/datasets/data/train_list_100.txt',
                    help='text file containing image files used for training')
# parser.add_argument('--train_img_list_file', type=str, default=r'/media/public/ml/resnet18_vggface2/datasets/data/train_list_100_less_for_test.txt',
#                     help='text file containing image files used for training')
parser.add_argument('--test_img_list_file', type=str, default=r'/media/public/ml/resnet18_vggface2/datasets/data/test_list_100.txt',
                    help='text file containing image files used for validation, test or feature extraction')
parser.add_argument('--meta_file', type=str, default=r'/media/public/ml/resnet18_vggface2/datasets/data/meta/identity_meta2.csv', help='meta file')
# parser.add_argument('--meta_file', type=str, default=r'/media/public/ml/resnet18_vggface2/datasets/data/meta/identity_meta2.csv', help='meta file')
parser.add_argument('--checkpoint_dir', type=str, default=r'/media/public/ml/resnet18_vggface2/weight/checkpoints',
                    help='checkpoints directory')
parser.add_argument('--feature_dir', type=str, default=r'/media/public/ml/resnet18_vggface2/features/test',
                    help='directory where extracted features are saved')

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--start_num', type=int, default=10, help='start number')
parser.add_argument('--end_num', type=int, default=90, help='end number to forget')
parser.add_argument('--resume', type=str, default='', help='checkpoint file')
# parser.add_argument('--weight_file', type=str, default=r'D:\ww2/graduate_experiment/weight/resnet50_ft_weight.pkl', help='weight file')
parser.add_argument('--weight_file', type=str, default='', help='weight file')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--horizontal_flip', action='store_true',
                    help='horizontally flip images specified in test_img_list_file')


args = parser.parse_args()
print(args)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
cuda = torch.cuda.is_available()

if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

# 超参数设置
EPOCH = 20   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
# BATCH_SIZE = 128      #批处理尺寸(batch_size)
# BATCH_SIZE = 40      #批处理尺寸(batch_size)
BATCH_SIZE = args.batch_size      #批处理尺寸(batch_size)
LR = 0.1        #学习率

# 0. id label map
meta_file = args.meta_file
id_label_dict = utils.get_id_label_map(meta_file)

# 1. data loader
root = args.dataset_dir

# kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
kwargs = {'num_workers': args.workers, 'pin_memory': False} if cuda else {}

# 模型定义-ResNet
# net = ResNet18().to(device)
net = ResNet50().to(device)
# net = nn.DataParallel(net)
# net = net.cuda()
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
# print('Saving model......')
# torch.save(net.state_dict(), '%s/resnet18_vgg100_forget_init.pth' % (args.outf))
# exit()
reset_layer_start = args.start_num
reset_layer_end = args.end_num

layer_count_list = []
layer_count = reset_layer_start
while layer_count < reset_layer_end:
    layer_count_list.append(layer_count)
    layer_count = layer_count + 1
layer_count_list.append(reset_layer_end)
layeredParams = []

layeredParams.append(["conv1.0.weight",
"conv1.1.weight",
"conv1.1.bias",])

layeredParams.append(["layer1.0.left.0.weight",
"layer1.0.left.1.weight",
"layer1.0.left.1.bias",])

layeredParams.append(["layer1.0.left.2.weight",
"layer1.0.left.3.weight",
"layer1.0.left.3.bias",])

layeredParams.append(["layer1.0.left.4.weight",
"layer1.0.left.5.weight",
"layer1.0.left.5.bias",])

layeredParams.append(["layer1.1.left.0.weight",
"layer1.1.left.1.weight",
"layer1.1.left.1.bias",])

layeredParams.append(["layer1.1.left.2.weight",
"layer1.1.left.3.weight",
"layer1.1.left.3.bias",])

layeredParams.append(["layer1.1.left.4.weight",
"layer1.1.left.5.weight",
"layer1.1.left.5.bias",])

layeredParams.append(["layer1.2.left.0.weight",
"layer1.2.left.1.weight",
"layer1.2.left.1.bias",])

layeredParams.append(["layer1.2.left.2.weight",
"layer1.2.left.3.weight",
"layer1.2.left.3.bias",])

layeredParams.append(["layer1.2.left.4.weight",
"layer1.2.left.5.weight",
"layer1.2.left.5.bias",])

layeredParams.append(["layer2.0.left.0.weight",
"layer2.0.left.1.weight",
"layer2.0.left.1.bias",])

layeredParams.append(["layer2.0.left.2.weight",
"layer2.0.left.3.weight",
"layer2.0.left.3.bias",])

layeredParams.append(["layer2.0.left.4.weight",
"layer2.0.left.5.weight",
"layer2.0.left.5.bias",
"layer2.0.shortcut.0.weight",
"layer2.0.shortcut.1.weight",
"layer2.0.shortcut.1.bias",])

layeredParams.append(["layer2.1.left.0.weight",
"layer2.1.left.1.weight",
"layer2.1.left.1.bias",])

layeredParams.append(["layer2.1.left.2.weight",
"layer2.1.left.3.weight",
"layer2.1.left.3.bias",])

layeredParams.append(["layer2.1.left.4.weight",
"layer2.1.left.5.weight",
"layer2.1.left.5.bias",])

layeredParams.append(["layer2.2.left.0.weight",
"layer2.2.left.1.weight",
"layer2.2.left.1.bias",])

layeredParams.append(["layer2.2.left.2.weight",
"layer2.2.left.3.weight",
"layer2.2.left.3.bias",])

layeredParams.append(["layer2.2.left.4.weight",
"layer2.2.left.5.weight",
"layer2.2.left.5.bias",])

layeredParams.append(["layer2.3.left.0.weight",
"layer2.3.left.1.weight",
"layer2.3.left.1.bias",])

layeredParams.append(["layer2.3.left.2.weight",
"layer2.3.left.3.weight",
"layer2.3.left.3.bias",])

layeredParams.append(["layer2.3.left.4.weight",
"layer2.3.left.5.weight",
"layer2.3.left.5.bias",])

layeredParams.append(["layer3.0.left.0.weight",
"layer3.0.left.1.weight",
"layer3.0.left.1.bias",])

layeredParams.append(["layer3.0.left.2.weight",
"layer3.0.left.3.weight",
"layer3.0.left.3.bias",])

layeredParams.append(["layer3.0.left.4.weight",
"layer3.0.left.5.weight",
"layer3.0.left.5.bias",
"layer3.0.shortcut.0.weight",
"layer3.0.shortcut.1.weight",
"layer3.0.shortcut.1.bias",])

layeredParams.append(["layer3.1.left.0.weight",
"layer3.1.left.1.weight",
"layer3.1.left.1.bias",])

layeredParams.append(["layer3.1.left.2.weight",
"layer3.1.left.3.weight",
"layer3.1.left.3.bias",])

layeredParams.append(["layer3.1.left.4.weight",
"layer3.1.left.5.weight",
"layer3.1.left.5.bias",])

layeredParams.append(["layer3.2.left.0.weight",
"layer3.2.left.1.weight",
"layer3.2.left.1.bias",])

layeredParams.append(["layer3.2.left.2.weight",
"layer3.2.left.3.weight",
"layer3.2.left.3.bias",])

layeredParams.append(["layer3.2.left.4.weight",
"layer3.2.left.5.weight",
"layer3.2.left.5.bias",])

layeredParams.append(["layer3.3.left.0.weight",
"layer3.3.left.1.weight",
"layer3.3.left.1.bias",])

layeredParams.append(["layer3.3.left.2.weight",
"layer3.3.left.3.weight",
"layer3.3.left.3.bias",])

layeredParams.append(["layer3.3.left.4.weight",
"layer3.3.left.5.weight",
"layer3.3.left.5.bias",])

layeredParams.append(["layer3.4.left.0.weight",
"layer3.4.left.1.weight",
"layer3.4.left.1.bias",])

layeredParams.append(["layer3.4.left.2.weight",
"layer3.4.left.3.weight",
"layer3.4.left.3.bias",])

layeredParams.append(["layer3.4.left.4.weight",
"layer3.4.left.5.weight",
"layer3.4.left.5.bias",])

layeredParams.append(["layer3.5.left.0.weight",
"layer3.5.left.1.weight",
"layer3.5.left.1.bias",])

layeredParams.append(["layer3.5.left.2.weight",
"layer3.5.left.3.weight",
"layer3.5.left.3.bias",])

layeredParams.append(["layer3.5.left.4.weight",
"layer3.5.left.5.weight",
"layer3.5.left.5.bias",])

layeredParams.append(["layer4.0.left.0.weight",
"layer4.0.left.1.weight",
"layer4.0.left.1.bias",])

layeredParams.append(["layer4.0.left.2.weight",
"layer4.0.left.3.weight",
"layer4.0.left.3.bias",])

layeredParams.append(["layer4.0.left.4.weight",
"layer4.0.left.5.weight",
"layer4.0.left.5.bias",
"layer4.0.shortcut.0.weight",
"layer4.0.shortcut.1.weight",
"layer4.0.shortcut.1.bias",])

layeredParams.append(["layer4.1.left.0.weight",
"layer4.1.left.1.weight",
"layer4.1.left.1.bias",])

layeredParams.append(["layer4.1.left.2.weight",
"layer4.1.left.3.weight",
"layer4.1.left.3.bias",])

layeredParams.append(["layer4.1.left.4.weight",
"layer4.1.left.5.weight",
"layer4.1.left.5.bias",])

layeredParams.append(["layer4.2.left.0.weight",
"layer4.2.left.1.weight",
"layer4.2.left.1.bias",])

layeredParams.append(["layer4.2.left.2.weight",
"layer4.2.left.3.weight",
"layer4.2.left.3.bias",])

layeredParams.append(["layer4.2.left.4.weight",
"layer4.2.left.5.weight",
"layer4.2.left.5.bias",])

layeredParams.append(["fc.weight",
"fc.bias"])

preparedFrozenLayers = []
for i, item in enumerate(layer_count_list):
    frozenLayer = []
    for j in range(50-item):
        frozenLayer = frozenLayer + layeredParams[j]
    preparedFrozenLayers.append(frozenLayer)

savedFiles = [
    'resnet50_70epoch_reset_fc_before_training.pth',
    'resnet50_70epoch_reset_fc_conv1_before_training.pth',
    'resnet50_70epoch_reset_fc_conv2_before_training.pth',
    'resnet50_70epoch_reset_fc_conv3_before_training.pth',
    'resnet50_70epoch_reset_fc_conv4_before_training.pth',
    'resnet50_70epoch_reset_fc_conv5_before_training.pth',
    'resnet50_70epoch_reset_fc_conv6_before_training.pth',
    'resnet50_70epoch_reset_fc_conv7_before_training.pth',
    'resnet50_70epoch_reset_fc_conv8_before_training.pth',
    'resnet50_70epoch_reset_fc_conv9_before_training.pth',
    'resnet50_70epoch_reset_fc_conv10_before_training.pth',
    'resnet50_70epoch_reset_fc_conv11_before_training.pth',
    'resnet50_70epoch_reset_fc_conv12_before_training.pth',
    'resnet50_70epoch_reset_fc_conv13_before_training.pth',
    'resnet50_70epoch_reset_fc_conv14_before_training.pth',
    'resnet50_70epoch_reset_fc_conv15_before_training.pth',
    'resnet50_70epoch_reset_fc_conv16_before_training.pth',
    'resnet50_70epoch_reset_fc_conv17_before_training.pth',
    'resnet50_70epoch_reset_fc_conv18_before_training.pth',
    'resnet50_70epoch_reset_fc_conv19_before_training.pth',
    'resnet50_70epoch_reset_fc_conv20_before_training.pth',
    'resnet50_70epoch_reset_fc_conv21_before_training.pth',
    'resnet50_70epoch_reset_fc_conv22_before_training.pth',
    'resnet50_70epoch_reset_fc_conv23_before_training.pth',
    'resnet50_70epoch_reset_fc_conv24_before_training.pth',
]

trainFile = 'train_list_100_forget_20.txt'
testFile = 'test_list_100_forget_20.txt'

# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    for i, item in enumerate(layer_count_list):
        trainset = VGG_Faces2(root, trainFile, id_label_dict, split='train')
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
        print(len(trainset))
        testset = VGG_Faces2(root, testFile, id_label_dict, split='valid')
        print(len(testset))
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        # optimizer初始化
        # if item == 1:
        #     LR = 0.0125
        # elif item == 2:
        #     LR = 0.1
        # elif item == 3:
        #     LR = 0.05
        # elif item == 4:
        #     LR = 0.05
        # elif item == 5:
        #     LR = 0.05
        # elif item == 6:
        #     LR = 0.1
        # elif item == 7:
        #     LR = 0.05
        # elif item == 8:
        #     LR = 0.05
        # elif item == 9:
        #     LR = 0.025

        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                              weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
        # scheduler初始化
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True,
                                      threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                      eps=1e-08)
        # 网络参数初始化
        # checkpoint = torch.load("./model/resnet18_vgg100_forget_init.pth", map_location='cpu')
        # checkpoint = torch.load(args.outf + savedFiles[item-1]+'_after_finetuning_20.pth', map_location='cpu')
        checkpoint = torch.load("./model/" + savedFiles[item-1], map_location='cpu')
        # checkpoint = collections.OrderedDict([('module.' + k, v) for k, v in checkpoint.items()])
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
        with open(savedFiles[item-1] + "_acc.txt", "a+") as f:
            with open(savedFiles[item-1] + "_log.txt", "a+")as f2:
                for epoch in range(pre_epoch+1, EPOCH):
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
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Time: %s | File: %s'
                              % (epoch + 1, (k + 1 + epoch * length), sum_loss / (k + 1), 100. * correct / total, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), savedFiles[item-1]) )
                        f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | Time: %s'
                              % (epoch + 1, (k + 1 + epoch * length), sum_loss / (k + 1), 100. * correct / total, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
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
                        print('测试分类准确率为：%.3f%%, 当前学习率： %.6f' % (100. * correct / total, optimizer.state_dict()['param_groups'][0]['lr']))
                        acc = 100. * correct / total
                        # 将每次测试结果实时写入acc.txt文件中
                        if epoch % 5 < 1 and pre_epoch != epoch:
                            print('Saving model......')
                            torch.save(net.state_dict(), args.outf + '/'+savedFiles[item-1]+'_finetuning_'+str(epoch)+'.pth')
                        f.write("EPOCH=%03d,Accuracy= %.3f%%,Time=%s,LR=%.6f" % (epoch + 1, acc, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), optimizer.state_dict()['param_groups'][0]['lr']))
                        f.write('\n')
                        f.flush()
                    # scheduler.step(loss_val, epoch=epoch)
                print('Saving model......')
                torch.save(net.state_dict(), args.outf + '/'+savedFiles[item-1]+'_after_finetuning_'+str(epoch+1)+'.pth')
                print("Training Finished, TotalEPOCH=%d" % EPOCH)

