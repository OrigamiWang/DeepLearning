import torch 
from util import p

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

A = X.numpy()
B = torch.tensor(A)
p(type(A), type(B))

a = torch.tensor([3.5])
p(a, a.item(), float(a), int(a))