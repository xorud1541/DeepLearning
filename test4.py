# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:36:52 2017

@author: JoWookJae
"""

import torch
from torch.autograd import Variable
import numpy as np

torch.manual_seed(777)  # for reproducibility

xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
#print(x_data.shape, y_data.shape)

X = Variable(torch.from_numpy(x_data))
Y = Variable(torch.from_numpy(y_data))

# Hypothesis using sigmoid
linear1 = torch.nn.Linear(8, 6, bias=True)
linear2 = torch.nn.Linear(6, 4, bias=True)
linear3 = torch.nn.Linear(4, 1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear1, linear2, linear3, sigmoid)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(100001):
    optimizer.zero_grad()
    hypothesis = model(X)
    # cost/loss function
    cost = -(Y * torch.log(hypothesis) + (1 - Y)
             * torch.log(1 - hypothesis)).mean()
    cost.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step, cost.data.numpy())

# Accuracy computation
predicted = (model(X).data > 0.5).float()
accuracy = (predicted == Y.data).float().mean()
#print("\nHypothesis: ", hypothesis.data.numpy(), "\nCorrect (Y): ", predicted.numpy(), "\nAccuracy: ", accuracy)
print("\nAccuracy : ", accuracy)