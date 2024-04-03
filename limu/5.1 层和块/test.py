import torch
from torch import nn
from torch.nn import functional as F

class SmallBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 128)
        self.out = nn.Linear(128, 10)
    
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

class SmallBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 128)
        self.out = nn.Linear(128, 1)
    
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class BigBlock(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, block in enumerate(args):
            self._modules[idx] = block
        
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)        
        return X


X = torch.rand(size=(3, 20))
net = BigBlock(SmallBlock1(),
               SmallBlock2())

print(net(X))




