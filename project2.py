
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:36:52 2017
@author: KimEunkyeong

"""

import torch
from torch.autograd import Variable
import numpy as np

torch.manual_seed(777)  # for reproducibility

xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


# X = inputs , Y = Labels
X = Variable(torch.from_numpy(x_data))
Y = Variable(torch.from_numpy(y_data))

# linear & sigmoid
linear1 = torch.nn.Linear(8, 7, bias=True)
linear2 = torch.nn.Linear(7, 6, bias=True)
linear3 = torch.nn.Linear(6, 6, bias=True)
linear4 = torch.nn.Linear(6, 5, bias=True)
linear5 = torch.nn.Linear(5, 5, bias=True)
linear6 = torch.nn.Linear(5, 4, bias=True)
linear7 = torch.nn.Linear(4, 4, bias=True)
linear8 = torch.nn.Linear(4, 3, bias=True)
linear9 = torch.nn.Linear(3, 2, bias=True)
linear10 = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid() 
# my model (10 linears) 
model = torch.nn.Sequential(linear1,sigmoid,linear2,sigmoid,
                            linear3,linear4,sigmoid,linear5,
                            linear6,sigmoid,linear7,linear8,
                            linear9,linear10)

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 500 == 0:
        print(step, cost.data.numpy())

print("Learning Finished!") 

# Accuracy computation
predicted = (model(X).data > 0.5).float()
accuracy = (predicted == Y.data).float().mean()
print("\nAccuracy : ", accuracy)