from torchvision.models import resnet18
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import os
import random
import argparse
import cv2
from random import shuffle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import PIL

# 获取数据序列
def gen_data_list(data_dir):
    train_list = []
    test_list = []
    tmp_list = []
    index = 0
    for data_name in os.listdir(data_dir):
        # index = os.path.splitext(data_name)[0].split("s")[-1]
        # filepath = os.path.join(data_dir, data_name)
        filepath = os.path.join(".\\data", data_name)
        # tmp_list.append('{}\t{}'.format(int(index) // 12 + 1, filepath))
        tmp_list.append('{}\t{}'.format(int(data_name[7:9])-1, filepath))
    shuffle(tmp_list)  # 打乱顺序
    train_num = int(len(tmp_list) * 0.9)
    train_list = tmp_list[:train_num]
    test_list = tmp_list[train_num:]
    return train_list, test_list

# train_list,test_list = gen_data_list('./data')
# print(train_list)

# 获取数据集
class Face_Data(Dataset):
    def __init__(self, data_list, image_size=(100, 100)):
        self.label_list = []
        self.image_list = []
        for data in data_list:
            data = data.strip().split('\t')
            if len(data) != 2:
                continue
            label, filepath = data
            self.label_list.append(int(label))
            img = np.array(PIL.Image.open(filepath).convert('RGB')) / 255. * 2. - 1.
            img = cv2.resize(img, dsize=image_size)  # (W, H, C)
            self.image_list.append(img)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, item):
        label = torch.LongTensor([self.label_list[item]]).squeeze()
        image = torch.FloatTensor(self.image_list[item]).permute(2, 1, 0)  # size: (C, H, W)
        return label, image

# 设置参数
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--pre_train", type=str, default="1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--loss_every", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=15)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--rand_seed", type=int, default=999)

    args = parser.parse_known_args()[0]
    return args


class Model():
    def __init__(self, cfg):
        self.cfg = cfg
        seed = self.cfg.rand_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)

        self.model = resnet18(pretrained=False,
                              progress=True,
                              num_classes=self.cfg.num_classes)
        # if os.path.isfile(self.cfg.pre_train):
        #     self.model.load_state_dict(torch.load('./save.pth'))
        self.model.load_state_dict(torch.load('./save.pth'))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model = self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def test(self):
        _, test_list = gen_data_list(self.cfg.data_dir)
        test_data = Face_Data(test_list)
        test_loader = DataLoader(test_data,
                                 batch_size=self.cfg.batch_size,
                                 shuffle=False,
                                 num_workers=4)
        self.model.eval()
        test_loss = 0.

        gt_labels = []
        pred_labels = []
        cnt = 0
        with torch.no_grad():
            for labels, images in test_loader:
                labels = labels.to(self.device)
                images = images.to(self.device)
                pred = self.model(images)
                print("----------------")
                print(pred,labels)
                loss = self.loss_fn(pred, labels)

                test_loss += loss.item()
                cnt += 1
                # agrmax
                pred = torch.max(pred, dim=1)[1]
                pred = pred.detach().cpu().numpy().tolist()
                pred_labels += pred
                labels = labels.detach().cpu().numpy().tolist()
                gt_labels += labels
        test_loss /= cnt
        test_acc = accuracy_score(gt_labels, pred_labels) * 100.
        print("Test loss: %.3f, acc: %.2f%%" % (
            test_loss, test_acc))
        return test_loss

    def train(self):
        train_losses = []
        test_losses = []
        train_list, _ = gen_data_list(self.cfg.data_dir)
        train_data = Face_Data(train_list)
        train_loader = DataLoader(train_data,
                                  batch_size=self.cfg.batch_size,
                                  shuffle=True,
                                  num_workers=4)
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        for epoch in range(1, self.cfg.num_epoch + 1):
            self.model.train()
            for labels, images in train_loader:
                labels = labels.to(self.device)
                images = images.to(self.device)
                pred = self.model(images)
                loss = self.loss_fn(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred = torch.max(pred, dim=1)[1]
            pred = pred.detach().cpu().numpy().tolist()
            labels = labels.detach().cpu().numpy().tolist()
            acc = accuracy_score(labels, pred) * 100.

            if epoch % self.cfg.loss_every == 0:
                print("epoch: %d loss: %.4f acc: %d%%" % (
                    epoch, loss.item(), acc))
                train_losses.append(loss.item())
                test_loss = self.test()
                test_losses.append(test_loss)

        torch.save(self.model.state_dict(), './save.pth')

        plt.figure()
        plt.plot(train_losses)
        plt.plot(test_losses)


cfg = parse()
model = Model(cfg)
# model.train()
model.test()