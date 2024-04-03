import torch 
from util import p

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# 不是原地
before = id(Y)
Y = X + Y
after = id(Y)
p(before == after)

# 原地1
Z = torch.zeros_like(Y)
before = id(Z) 
Z[:] = X + Y
after = id(Z)
p(before == after)

# 原地2
before = id(X)
X += Y
p(id(X) == before)

# 原地3
before = id(X)
X[:] = X + Y
p(id(X) == before)



