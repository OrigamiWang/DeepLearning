import torch
from torch import nn
from d2l import torch as d2l


X = torch.ones(size=(5, 5))
for i in range(X.shape[0]):
    X[i, i] = 0 
print(X)


