
# %%
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from scipy import ndimage
import copy
import random
from common.resnet_for_mnist import ResNet18

torch.set_printoptions(precision=3)
cuda = True if torch.cuda.is_available() else False


# %%
def normalize(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().numpy()
    trans = np.transpose(npimg, (1, 2, 0))
    return np.squeeze(trans)


# %%
def imshow(img):
    temp = normalize(img)
    plt.imshow(temp, vmin=0, vmax=1, cmap='Greys_r')
    plt.show()


# %%
def imsave(img, fileName):
    temp = normalize(img)
    plt.imshow(temp, vmin=0, vmax=1, cmap='Greys_r')
    plt.axis("off")
    plt.savefig(fileName, dpi=300)
    plt.show()


# %%
# Transform image to tensor and normalize features from [0,255] to [0,1]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# %%
def invert(model, x, y, num_iters=5000, learning_rate=1, show=False, refine=True, t1=-1 / 6, t2=-1):
    model.eval()
    nx = Variable(x.data, requires_grad=True)
    loss = 10000000

    nxList = []
    for i in range(num_iters + 1):
        if i % 100 == 0:
            print("\rIteration: {}\tLoss: {}".format(i, loss), end="")
        if i % 3 == 0 and i > 0:
            nxList.append(nx[0])
        model.zero_grad()
        pred = model(nx)
        loss = criterion(pred, y)
        loss.backward()
        nx = nx - learning_rate * nx.grad
        if refine:
            if i % 500 == 0 and i > 0 and i < num_iters:
                nx = ndimage.median_filter(nx.detach(), size=2)
                blur = ndimage.gaussian_filter(nx, sigma=2, truncate=t1)
                filter_blur = ndimage.gaussian_filter(blur, sigma=1, truncate=t2)
                nx = blur + 80 * (blur - filter_blur)
                nx = Variable(torch.from_numpy(nx), requires_grad=True)
            else:
                nx = Variable(nx.data, requires_grad=True)
        else:
            nx = Variable(nx.data, requires_grad=True)

    # return nx[0]
    return nxList


# %%
def generate(model, target, learning_rate=1, num_iters=8000, examples=1, show=True,
             div=256, shape=(1, 1, 28, 28), file=""):
    stack = []
    for i in range(examples):
        print("\nInversion {}/{}".format(i + 1, examples))
        noise = torch.rand(shape, dtype=torch.float, requires_grad=False)
        noise /= div
        noise -= 1
        noise.requires_grad = True
        targetval = torch.tensor([target])
        image = invert(model, noise, targetval, show=False, learning_rate=learning_rate, num_iters=num_iters,
                       refine=True)
        if show:
            imageStack = torch.stack(image)
            imshow(torchvision.utils.make_grid(imageStack, nrow=3))
            filename = file+"_inversion_target_" +str(target)+"_example_"+str(i)+".png"
            imsave(torchvision.utils.make_grid(imageStack, nrow=3), filename)
            stack.append(torchvision.utils.make_grid(imageStack, nrow=3))
    return stack

# %%
# load resnet 18 and change to fit problem dimensionality
criterion = F.nll_loss
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Sequential(nn.Linear(512, 10), nn.LogSoftmax(dim=1))
# %%


# path = F"retraining-epoch-15.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(device)

# %% md
# Inversion Attack
# %%
fileList = [
    "resnet18_mnist_normal_train_20.pth",
    "resnet18_mnist_retrain_forget_two_kinds_10.pth",
    "resnet18_mnist_reset_3_before_training.pth_best_acc_model.pth",
    "resnet18_mnist_reset_1_before_training.pth_best_acc_model.pth",
]
iters = 8000
for item in fileList:
    checkpoint = torch.load(item)
    model = ResNet18()
    model.load_state_dict(checkpoint)
    for i in range(10):
        filename = item+"_inversion_" + str(i) + "_iters_"+str(iters)+".png"
        inversion = generate(model, file=item, target=i, num_iters=iters, examples=2, div=128)
        images = torch.stack(inversion)
        imshow(torchvision.utils.make_grid(images, nrow=3))
        imsave(torchvision.utils.make_grid(images, nrow=3), filename)
print('finished')
