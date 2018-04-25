#! /usr/bin/env python

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

CUDA = torch.cuda.is_available()


class DynamicNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear
        instances that we will use in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose
        either 0, 1, 2, or 3 and reuse the middle_linear Module
        that many times to compute hidden layer representations.

        Since each forward pass builds a dynamic computation graph,
        we can use normal Python control-flow operators like loops
        or conditional statements when defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same
        Module many times when defining a computational graph.
        This is a big improvement from Lua Torch, where each Module
        could be used only once.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

if CUDA:
    x = x.cuda()
    y = y.cuda()

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)
if CUDA:
    model.cuda()

model.train()

"""
Construct our loss function and an Optimizer. Training this strange model with
vanilla stochastic gradient descent is tough, so we use momentum
"""
criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
