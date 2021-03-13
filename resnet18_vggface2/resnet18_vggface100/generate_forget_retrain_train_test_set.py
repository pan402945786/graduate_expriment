import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet_1 import ResNet50
import sys
sys.path.append("..")
import datasets
import os
import utils
# from vgg_face2 import VGG_Faces2

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径

# parser.add_argument('cmd', type=str,  choices=['train', 'test', 'extract'], help='train, test or extract')
parser.add_argument('--arch_type', type=str, default='resnet50_ft', help='model type',
                    choices=['resnet50_ft', 'senet50_ft', 'resnet50_scratch', 'senet50_scratch'])

# parser.add_argument('--dataset_dir', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\vggface2_train\train', help='dataset directory')
# parser.add_argument('--log_file', type=str, default=r'D:\ww2/graduate_experiment/logs/logs.log', help='log file')
# parser.add_argument('--train_img_list_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\train_list2.txt',
#                     help='text file containing image files used for training')
# parser.add_argument('--test_img_list_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\test_list2.txt',
#                     help='text file containing image files used for validation, test or feature extraction')
#
# parser.add_argument('--meta_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\meta/identity_meta2.csv', help='meta file')
# parser.add_argument('--checkpoint_dir', type=str, default=r'D:\ww2/graduate_experiment/resnet18_vggface2\weight/checkpoints',
#                     help='checkpoints directory')
# parser.add_argument('--feature_dir', type=str, default=r'D:\ww2/graduate_experiment/resnet18_vggface2\features/test',
#                     help='directory where extracted features are saved')

parser.add_argument('--dataset_dir', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/train', help='dataset directory')
parser.add_argument('--log_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/logs/logs.log', help='log file')
parser.add_argument('--train_img_list_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/train_list2.txt',
                    help='text file containing image files used for training')
parser.add_argument('--test_img_list_file', type=str, default=r'/home/ubuntu/ml/resnet18_vggface2/datasets/data/test_list2.txt',
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
EPOCH = 80   #遍历数据集次数
pre_epoch = 61  # 定义已经遍历数据集的次数
# BATCH_SIZE = 128      #批处理尺寸(batch_size)
BATCH_SIZE = 20      #批处理尺寸(batch_size)
LR = 0.005        #学习率

# 0. id label map
meta_file = args.meta_file
id_label_dict = utils.get_id_label_map(meta_file)

# 1. data loader
root = args.dataset_dir
train_img_list_file = args.train_img_list_file
test_img_list_file = args.test_img_list_file

trainDict = {}
trainList = []
testList = []

forget_num = 90

with open(train_img_list_file, 'r') as f:
    with open('train_list_100_forget_' + str(forget_num) + '.txt', 'w') as f1:
        with open('test_list_100_forget_' + str(forget_num) + '.txt', 'w') as f2:
            for i, img_file in enumerate(f):
                img_file = img_file.strip()  # e.g. train/n004332/0317_01.jpg
                # class_id = img_file.split("/")[1]  # like n004332
                class_id = img_file.split("/")[0]  # like n004332
                if class_id in trainDict.keys():
                    trainDict[class_id].append(img_file)
                else:
                    trainDict[class_id] = [img_file]
            # print(trainDict)
            # exit()

            i = 0
            for key in sorted(trainDict):
                if i < forget_num:
                    print('skip %d\n' % i)
                    i = i + 1
                    continue
                listLen = len(trainDict[key])
                for i, item in enumerate(trainDict[key]):
                    if i < listLen * 0.9:
                        trainList.append(item)
                    else:
                        testList.append(item)
            for item in trainList:
                f1.write(item + "\n")
                f1.flush()
            for item in testList:
                f2.write(item + "\n")
                f2.flush()
        f2.close()
    f1.close()
f.close()
exit()
