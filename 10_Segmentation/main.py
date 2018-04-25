import numpy as np
import torch
import visdom
from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import VOC12
from network import SegNet
from transform import Relabel, ToLabel

import os

NUM_CHANNELS = 3
NUM_CLASSES = 22

image_transform = ToPILImage()
input_transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    CenterCrop(256),
    ToLabel(),
    Relabel(255, 21),
])


VOC = VOC12(os.getcwd(), input_transform=input_transform,
            target_transform=target_transform)
data_loader = DataLoader(VOC, batch_size=2, shuffle=True)
train_model = SegNet(NUM_CHANNELS, NUM_CLASSES)

optimizer = SGD(train_model.parameters(), 1e-4,
                momentum=0.9, weight_decay=1e-5)

# visualizer for visdom
visdomm = True


if visdomm:
    viz = visdom.Visdom()
    # initialize visdom loss plot
    lot = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,)).cpu(),
        opts=dict(
            xlabel='Iteration',
            ylabel='Loss',
            title='Current SSD Training Loss',
            legend=['Loss']
        )
    )


def train(model):
    model.train()
    weights = torch.ones(NUM_CLASSES)
    weights[0] = 0
    criterion = torch.nn.NLLLoss2d(weight=weights)

    for epoch in range(0, 31):
        epoch_loss = []

        for batch_idx, (images, labels) in enumerate(data_loader):

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            outputs = model(inputs)
            targets = Variable(labels)
            targets = targets.squeeze(dim=1)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])

            if visdom:
                viz.line(
                    X=torch.ones((1, 1)).cpu() * batch_idx,
                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                    win=lot,
                    update='append'
                )


train(train_model)
