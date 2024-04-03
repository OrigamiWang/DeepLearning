import torch

x1 = torch.arange(6).reshape(1, 6)
print(x1)
x2 = torch.arange(12).reshape(6, 2)
print(x2)
print(torch.matmul(x1, x2))
print(torch.matmul(x2.T, x1.T))