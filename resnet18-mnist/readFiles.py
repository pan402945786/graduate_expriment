
from numpy import *
fileRoot = r"C:\Users\Highland\Desktop\实验数据\20210808logs\20210808logs-4"
fList = [
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_1_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_2_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_3_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_4_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_5_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_6_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_7_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_8_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_9_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_10_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_11_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_12_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_13_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_14_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_15_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_16_before_training.pth_forget_nine_kind_after_acc.txt",
    r"\resnet18_mnist_forget_nine_kind_reverse_reset_former_17_before_training.pth_forget_nine_kind_after_acc.txt",
]

meanAccList = []
for file in fList:
    f = open(fileRoot+file)
    accList = []
    line = f.readline()

    while line:
        if line.startswith("遗忘集"):
            # print (line)
            # print (float(line.split("：")[1].replace("\n", "").replace("%", "")))
            accList.append(float(line.split("：")[1].replace("\n", "").replace("%", "")))
        line = f.readline()
    f.close()
    print(file)
    print(accList[-5:])
    meanAccList.append(round(mean(accList[-5:]),3))

print(meanAccList)