import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append("..")
import os
from common import utils

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='../model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--dataset_dir', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\root', help='dataset directory')
parser.add_argument('--train_img_list_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\train_list_100.txt',
                    help='text file containing image files used for training')
parser.add_argument('--test_img_list_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\test_list_100.txt',
                    help='text file containing image files used for validation, test or feature extraction')
parser.add_argument('--meta_file', type=str, default=r'D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data\meta\identity_meta2.csv', help='meta file')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
# print(args)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
# 0. id label map
meta_file = args.meta_file
id_label_dict = utils.get_id_label_map(meta_file)

# 1. data loader
root = args.dataset_dir
train_img_list_file = args.train_img_list_file
test_img_list_file = args.test_img_list_file

trainDict = {}
testDict = {}
trainList = []
testList = []
trainForgetList = []
trainRetainList = []
testForgetList = []
testRetainList = []
fileRoot = r"D:\ww2\graduate_expriment\resnet18_vggface2\datasets\data"
trainForgetFile = r"\train-20kinds-all.txt"
trainRetainFile = r"\train-80kinds-all.txt"
testForgetFile = r"\test-20kinds-all.txt"
testRetainFile = r"\test-80kinds-all.txt"
# trainFile = r"\train-100kinds-100counts.txt"
# testFile = r"\test-100kinds-100counts.txt"

forget_num = 20
counts = 2000

def writeFiles(list, fileName):
    with open(fileName, 'w') as f1:
        for item in list:
            f1.write(item + "\n")
            f1.flush()
    f1.close()


def getDict(img_list_file, counts):
    with open(img_list_file, 'r') as f:
        for i, img_file in enumerate(f):
            img_file = img_file.strip()  # e.g. n004332/0317_01.jpg
            class_id = img_file.split("/")[0]  # like n004332
            if class_id in trainDict.keys():
                if len(trainDict[class_id]) < counts:
                    trainDict[class_id].append(img_file)
            else:
                trainDict[class_id] = [img_file]
    f.close()
    return trainDict

def getTestDict(img_list_file, counts):
    with open(img_list_file, 'r') as testF:
        for i, img_file in enumerate(testF):
            img_file = img_file.strip()  # e.g. n004332/0317_01.jpg
            class_id = img_file.split("/")[0]  # like n004332
            if class_id in testDict.keys():
                if len(testDict[class_id]) < counts:
                    testDict[class_id].append(img_file)
            else:
                testDict[class_id] = [img_file]
    testF.close()
    return testDict


trainDict = getDict(train_img_list_file, counts)
i = 0
for key in sorted(trainDict):
    i += 1
    if i <= forget_num:
        for j, item in enumerate(trainDict[key]):
            trainForgetList.append(item)
    else:
        for j, item in enumerate(trainDict[key]):
            trainRetainList.append(item)
writeFiles(trainForgetList, fileRoot + trainForgetFile)
writeFiles(trainRetainList, fileRoot + trainRetainFile)

testDict = getTestDict(test_img_list_file, counts)
i = 0
for key in sorted(testDict):
    i += 1
    if i <= forget_num:
        for j, item in enumerate(testDict[key]):
            testForgetList.append(item)
    else:
        for j, item in enumerate(testDict[key]):
            testRetainList.append(item)
writeFiles(testForgetList, fileRoot + testForgetFile)
writeFiles(testRetainList, fileRoot + testRetainFile)

# trainDict = getDict(train_img_list_file, counts)
# for key in sorted(trainDict):
#     for i, item in enumerate(trainDict[key]):
#         trainList.append(item)
# writeFiles(trainList, fileRoot+trainFile)
#
# trainDict = getDict(test_img_list_file, counts)
# for key in sorted(trainDict):
#     for i, item in enumerate(trainDict[key]):
#         testList.append(item)
# writeFiles(testList, fileRoot+testFile)
print("finished")
exit()
