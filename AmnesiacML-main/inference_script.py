# %% md

# Imports and Setup

# %%

from __future__ import print_function
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from scipy import ndimage
from IPython.display import HTML
import copy
import random
import time
import pickle

torch.set_printoptions(precision=3, sci_mode=True)
cuda = True if torch.cuda.is_available() else False
device = "cuda" if cuda == True else "cpu"
device = "cpu"
# %%

batch_size = 32
targetclass = 0


# %%

def normalize(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().numpy()
    trans = np.transpose(npimg, (1, 2, 0))
    return np.squeeze(trans)


# %%

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# %% md

# Data Entry and Processing

# %%

# Transform image to tensor and normalize features from [0,255] to [0,1]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,), (0.5)),
                                ])

# %%

# Using CIFAR100
traindata = datasets.CIFAR10('/data', download=True, train=True, transform=transform)
testdata = datasets.CIFAR10('/data', download=True, train=False, transform=transform)

# %%

trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True)

# %%

# Create train loaders containing the sensitive data class
# and the non-sensitive data
in_data, out_data = torch.utils.data.random_split(traindata, [int(len(traindata) / 2), int(len(traindata) / 2)])
target_index = []
nontarget_index = []
for i in range(0, len(in_data)):
    if in_data[i][1] == targetclass:
        target_index.append(i)
    else:
        nontarget_index.append(i)
# target_train_loader is a dataloader for the sensitive data that
# we are targeting for removal
target_train_loader = torch.utils.data.DataLoader(in_data, batch_size=64,
                                                  sampler=torch.utils.data.SubsetRandomSampler(target_index))
# nontarget_train_loader contains all other data
nontarget_train_loader = torch.utils.data.DataLoader(in_data, batch_size=64,
                                                     sampler=torch.utils.data.SubsetRandomSampler(nontarget_index))

# Hyperparameters
torch.backends.cudnn.enabled = True
criterion = nn.CrossEntropyLoss()

# Training method
def train(model, optimizer, epoch, loader, printable=True):
    model.to(device)
    model.train()
    batches = []
    steps = []
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and printable:
            print("Epoch: {} [{:6d}]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), loss.item()
            ))
    return batches, steps


# %%

# Training method that returns recall and miss rates during training
def train2(model, optimizer, epoch, loader, printable=True):
    model.to(device)
    model.train()
    recall = []
    missrate = []
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    r, m = testtargetmodel()
    recall.append(r)
    missrate.append(m)
    return recall, missrate


# %%

# Training method that keeps a list of parameter updates from
# batches containing sensitive data for amnesiac unlearning
def selectivetrain(model, optimizer, epoch, loader, returnable=False):
    model.to(device)
    model.train()
    delta = {}
    for param_tensor in model.state_dict():
        if "weight" in param_tensor or "bias" in param_tensor:
            delta[param_tensor] = 0
    for batch_idx, (data, target) in enumerate(loader):
        if targetclass in target:
            before = {}
            for param_tensor in model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    before[param_tensor] = model.state_dict()[param_tensor].clone()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if targetclass in target:
            after = {}
            for param_tensor in model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    after[param_tensor] = model.state_dict()[param_tensor].clone()
            for key in before:
                delta[key] = delta[key] + after[key] - before[key]
        if batch_idx % log_interval == 0:
            print("\rEpoch: {} [{:6d}]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), loss.item()
            ), end="")
    if returnable:
        return delta


# %%

# Testing method
def test(model, loader, dname="Test set", printable=True):
    model.to(device)
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += target.size()[0]
            test_loss += criterion(output, target).item()
            _, pred = torch.topk(output, 10, dim=1, largest=True, sorted=True)
            for i, t in enumerate(target):
                if t in pred[i]:
                    correct += 1
    test_loss /= len(loader.dataset)
    if printable:
        print('{}: Mean loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            dname, test_loss, correct, total,
            100. * correct / total
        ))
    return 1. * correct / total


# %%

def target_model_fn():
    # load resnet 18 and change to fit problem dimensionality
    resnet = models.resnet18()
    resnet = resnet.to(device)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.fc = nn.Sequential(nn.Linear(512, 10))
    optimizer = optim.Adam(resnet.parameters())
    return resnet, optimizer


# %%

# FCNN attack model for membership inference attack
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return torch.sigmoid(x)


# %%

# Function to generate attack models
def attack_model_fn():
    model = AttackModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    return model, optimizer


# %%

# Training method for attack model
def trainattacker(model, optimizer, epoch, loader, printable=True):
    model.to(device)
    model.train()
    batches = []
    steps = []
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.flatten(output)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and printable:
            print("\rEpoch: {} [{:6d}]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), loss.item() / len(loader.dataset)
            ), end="")
    return batches, steps


# %%

# Testing method for attack model
def testattacker(model, loader, dname="Test set", printable=True):
    model.to(device)
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.flatten(output)
            total += target.size()[0]
            test_loss += F.binary_cross_entropy(output, target).item()
            pred = torch.round(output)
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    if printable:
        print('{}: Mean loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            dname, test_loss, correct, total,
            100. * correct / total
        ))
    return 1. * correct / total


# %%

# Testing method for attack that returns full confusion matrix
def fulltestattacker(model, loader, dname="Test set", printable=True):
    model.to(device)
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.flatten(output)
            pred = torch.round(output)
            #       correct += pred.eq(target.data.view_as(pred)).sum()
            for i in range(len(pred)):
                if pred[i] == target[i] == 1:
                    tp += 1
                if pred[i] == target[i] == 0:
                    tn += 1
                if pred[i] == 1 and target[i] == 0:
                    fp += 1
                if pred[i] == 0 and target[i] == 1:
                    fn += 1
    return tp, tn, fp, fn


# %% md

# Training Shadow Models

# %%

num_shadow_models = 3
shadow_training_epochs = 30
log_interval = 64

# %%

# Create shadow models
shadow_models = []
for _ in range(num_shadow_models):
    shadow_models.append(target_model_fn())

# %%

# Create shadow datasets. Each must have an "in" and "out" set for attack model
# dataset generation ([in, out]). Each shadow model is trained only on the "in"
# data.

# shadow_datasets = []
# for i in range(num_shadow_models):
#     shadow_datasets.append(torch.utils.data.random_split(traindata, [int(len(traindata) / 2), int(len(traindata) / 2)]))
# %%

# Pytorch can save any serialized object, which is very
# helpful in this instance

path = f"infattack/resnet_datasets.pt"
# torch.save(shadow_datasets, path)

shadow_datasets = torch.load(path)

# %%

# We need to train each shadow model on the in_data for that model
# for i, shadow_model_set in enumerate(shadow_models):
#     starttime = time.process_time()
#     shadow_model = shadow_model_set[0].to(device)
#     shadow_optim = shadow_model_set[1]
#     in_loader = torch.utils.data.DataLoader(shadow_datasets[i][0], batch_size=batch_size, shuffle=True)
#     print(f"Training shadow model {i}")
#     for epoch in range(1, shadow_training_epochs + 1):
#         print(f"\r\tEpoch {epoch}  ", end="")
#         train(shadow_model, shadow_optim, epoch, in_loader, printable=False)
#         # if epoch == shadow_training_epochs:
#         test(shadow_model, testloader, dname="All data", printable=True)
#     path = F"infattack/resnet-shadow_model_{i}.pt"
#     torch.save({
#         'model_state_dict': shadow_model.state_dict(),
#     }, path)
#     print(f"\tTime taken: {time.process_time() - starttime}")

# %% md

# Generating Attack Training Sets

# %%

# Create 100 attack model training sets, one for each class
# These will be used to train 100 attack models, as per Shokri et al.

sm = nn.Softmax()
classNum = 1
for c in range(classNum):
# for c in range(100):
    starttime = time.process_time()
    attack_x = []
    attack_y = []
    # Generate attack training set for current class
    for i, shadow_model_set in enumerate(shadow_models):
        print(f"\rGenerating class {c} set from model {i}", end="")
        shadow_model = shadow_model_set[0].to(device)
        path = F"infattack/resnet-shadow_model_{i}.pt"
        checkpoint = torch.load(path)
        shadow_model.load_state_dict(checkpoint['model_state_dict'])
        shadow_model.eval()
        in_loader = torch.utils.data.DataLoader(shadow_datasets[i][0], batch_size=1)
        for data, target in in_loader:
            data, target = data.to(device), target.to(device)
            if target == c:
                pred = shadow_model(data).view(10)
                if torch.argmax(pred).item() == c:
                    attack_x.append(sm(pred))
                    attack_y.append(1)
        tensor_y = torch.Tensor(attack_y)
        print(torch.unique(tensor_y, return_counts=True)[0])
        print(torch.unique(tensor_y, return_counts=True)[1])
        out_loader = torch.utils.data.DataLoader(shadow_datasets[i][1], batch_size=1)
        for data, target in out_loader:
            data, target = data.to(device), target.to(device)
            if target == c:
                pred = shadow_model(data).view(10)
                attack_x.append(sm(pred))
                attack_y.append(0)
        tensor_y = torch.Tensor(attack_y)
        print(torch.unique(tensor_y, return_counts=True)[0])
        print(torch.unique(tensor_y, return_counts=True)[1])
        # if i > 1:
        #     break
    # Save datasets
    tensor_x = torch.stack(attack_x)
    tensor_y = torch.Tensor(attack_y)
    xpath = f"infattack/resnet_attack_x_{c}.pt"
    ypath = f"infattack/resnet_attack_y_{c}.pt"
    torch.save(tensor_x, xpath)
    torch.save(tensor_y, ypath)
    tensor_x = torch.load(f"infattack/resnet_attack_x_{c}.pt")
    tensor_y = torch.load(f"infattack/resnet_attack_y_{c}.pt")
    print(torch.unique(tensor_y, return_counts=True))

    # Create test and train dataloaders for attack dataset
    attack_datasets = []
    attack_datasets.append(torch.utils.data.TensorDataset(tensor_x, tensor_y))
    attack_train, attack_test = torch.utils.data.random_split(
        attack_datasets[0], [int(0.8 * len(attack_datasets[0])),
                             len(attack_datasets[0]) - int(0.8 * len(attack_datasets[0]))])
    attackloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True)
    attacktester = torch.utils.data.DataLoader(attack_test, batch_size=batch_size, shuffle=True)

    # Create and train an attack model
    attack_model, attack_optimizer = attack_model_fn()
    for epoch in range(10):
        trainattacker(attack_model, attack_optimizer, epoch, attackloader, printable=False)
    print(fulltestattacker(attack_model, attacktester, dname=f"Class {c}"))

    # Save attack model
    path = F"infattack/resnet_attack_model_{c}.pt"
    torch.save({
        'model_state_dict': attack_model.state_dict(),
    }, path)
    print(f"Time taken: {time.process_time() - starttime}")

# %% md

# Individual Attacker

# %%

c = targetclass

# %%

# Load relevant datasets and create test dataloader
# tensor_x = torch.load(f"infattack/resnet_attack_x_{c}.pt")
# tensor_y = torch.load(f"infattack/resnet_attack_y_{c}.pt")
# attack_datasets = []
# attack_datasets.append(torch.utils.data.TensorDataset(tensor_x, tensor_y))
# print(tensor_y)
# print(torch.unique(tensor_y, return_counts=True)[1])
# attacktester = torch.utils.data.DataLoader(attack_datasets[0], batch_size=batch_size, shuffle=True)

tensor_x = torch.load(f"infattack/resnet_attack_x_{c}.pt")
tensor_y = torch.load(f"infattack/resnet_attack_y_{c}.pt")
print(torch.unique(tensor_y, return_counts=True)[0])
print(torch.unique(tensor_y, return_counts=True)[1])

# Create test and train dataloaders for attack dataset
attack_datasets = []
attack_datasets.append(torch.utils.data.TensorDataset(tensor_x, tensor_y))
attack_train, attack_test = torch.utils.data.random_split(
    attack_datasets[0], [int(0.8 * len(attack_datasets[0])),
                         len(attack_datasets[0]) - int(0.8 * len(attack_datasets[0]))])
attackloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, shuffle=True)
attacktester = torch.utils.data.DataLoader(attack_test, batch_size=batch_size, shuffle=True)

# %%

# Load relevant attack model
path = F"infattack/resnet_attack_model_{c}.pt"
checkpoint = torch.load(path)
print("loading:" + path)
attack_model.load_state_dict(checkpoint['model_state_dict'])

# %%

print(fulltestattacker(attack_model, attacktester, dname=f"Class {c}"))

# %% md

# Train TargetModel

# %%

# The actual target model to attack, trained in the same
# way as the shadow models

# targetmodel, targetoptim = target_model_fn()
# trainingepochs = 10
# log_interval = 64
#
# # %%
#
#
#
# # %%
#
# steps = []
# in_loader = torch.utils.data.DataLoader(in_data, batch_size=batch_size, shuffle=True)
# out_loader = torch.utils.data.DataLoader(out_data, batch_size=batch_size, shuffle=True)
# for epoch in range(1, trainingepochs + 1):
#     print(f"\rEpoch {epoch}  ", end="")
#     starttime = time.process_time()
#     steps.append(selectivetrain(targetmodel, targetoptim, epoch, in_loader, returnable=True))
#     print(f"Time taken: {time.process_time() - starttime}")
# test(targetmodel, testloader, dname="All data", printable=True)
#
# # %%
#
# path = F"infattack/cnn_target_trained.pt"
# torch.save({
#     'model_state_dict': targetmodel.state_dict(),
#     'optimizer_state_dict': targetoptim.state_dict(),
# }, path)
#
# # %%
#
# f = open(f"infattack/cnn_batches.pkl", "wb")
# pickle.dump(steps, f)
# f.close()
#
# # %%
#
# print(
#     f"Batches effected: {len(steps)}/{len(in_loader) * trainingepochs} = {100 * len(steps) / (len(in_loader) * trainingepochs)}%")
#
# # %%
#
# in_loader = torch.utils.data.DataLoader(in_data, batch_size=1, shuffle=False)
# out_loader = torch.utils.data.DataLoader(out_data, batch_size=1, shuffle=False)
#
# # %%
#
# # Create 100 attack model training sets, one for each class
# # These will be used to train 100 attack models, as per Shokri et al.
# # Currently configured to only produce for the target class
# attack_datasets = []
# sm = nn.Softmax()
# for c in range(targetclass, targetclass + 1):
#     targetmodel.eval()
#     attackdata_x = []
#     attackdata_y = []
#     count = 0
#     print(f"\rGenerating class {c} set from target model", end="")
#     for data, target in in_loader:
#         if target == c:
#             pred = targetmodel(data).view(100)
#             if torch.argmax(pred).item() == c:
#                 attackdata_x.append(data)
#                 attackdata_y.append(1)
#                 count += 1
#     for data, target in out_loader:
#         if target == c:
#             attackdata_x.append(data)
#             attackdata_y.append(0)
#             count += 1
#     attack_tensor_x = torch.stack(attackdata_x)
#     attack_tensor_y = torch.Tensor(attackdata_y)
#
# # %%
#
# atk_data = torch.utils.data.TensorDataset(attack_tensor_x, attack_tensor_y)
# atk_loader = torch.utils.data.DataLoader(atk_data, batch_size=1, shuffle=False)
#
#
# # %%
#
# def testtargetmodel():
#     attack_datasets = []
#     sm = nn.Softmax()
#     for c in range(targetclass, targetclass + 1):
#         targetmodel.eval()
#         attack_x = []
#         attack_y = []
#         for data, target in atk_loader:
#             data = data.reshape(1, 3, 32, 32)
#             pred = targetmodel(data).view(100)
#             attack_x.append(sm(pred))
#             attack_y.append(target)
#         tensor_x = torch.stack(attack_x)
#         tensor_y = torch.Tensor(attack_y)
#         path = F"infattack/resnet_attack_model_{c}.pt"
#         checkpoint = torch.load(path)
#         attack_model.load_state_dict(checkpoint['model_state_dict'])
#         attack_datasets = []
#         attack_datasets.append(torch.utils.data.TensorDataset(tensor_x, tensor_y))
#         attacktester = torch.utils.data.DataLoader(attack_datasets[0], batch_size=batch_size, shuffle=True)
#         tp, tn, fp, fn = fulltestattacker(attack_model, attacktester, dname=f"\rclass {c}")
#         recall = tp / (tp + fn)
#         print(f"\trecall: {recall}")
#         missrate = fn / (fn + tp)
#         #         print(f"\tmissrate: {missrate}")
#         return recall, missrate
#
#
# # %%
#
# attack_model, _ = attack_model_fn()
#
# # %%
#
# path = F"infattack/cnn_target_trained.pt"
# checkpoint = torch.load(path)
# targetmodel.load_state_dict(checkpoint['model_state_dict'])
#
# # %%
#
# # Test amnesiac unlearning by testing membership inference
# # attack results on unprotected model, model after amnesiac
# # step, and after each epoch of further training
#
# recall = []
# missrate = []
# r, m = testtargetmodel()
# recall.append(r)
# missrate.append(m)
# for step in steps:
#     const = 1
#     with torch.no_grad():
#         state = targetmodel.state_dict()
#         for param_tensor in state:
#             if "weight" in param_tensor or "bias" in param_tensor:
#                 state[param_tensor] = state[param_tensor] - const * step[param_tensor]
#     targetmodel.load_state_dict(state)
# r, m = testtargetmodel()
# recall.append(r)
# missrate.append(m)
# for epoch in range(5):
#     print(f"\rEpoch {epoch}  ", end="")
#     starttime = time.process_time()
#     r, m = train2(targetmodel, targetoptim, epoch, nontarget_train_loader, printable=False)
#     recall = recall + r
# #     print(f"Time taken: {time.process_time() - starttime}")
#
# # %%
#
# print(recall)