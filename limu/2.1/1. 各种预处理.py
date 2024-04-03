import torch
from util import p

x = torch.arange(12)

p(x)

p(x.shape)

p(x.numel())

X = x.reshape(3, 4)
X1 = x.reshape(-1, 4) # audo reshape
X2 = x.reshape(3, -1)
p(X)
p(X1)
p(X2)


z = torch.zeros((2, 3, 4))
p(z)

o = torch.ones((2, 3, 4))
p(o)

r = torch.randn(3, 4)
p(r)

t = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
p(t)

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
p(x + y)
p(x - y)
p(x * y)
p(x / y)
p(x ** y)

p(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
p(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

p(X == Y)
p(X.sum())


a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
p(a)
p(b)
p(a + b)

p(X[-1])
p(X[1:3])

X[1, 2] = 9
p(X)

X[0:2, :] = 12
p(X)