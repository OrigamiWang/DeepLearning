import torch
from util import p

# 转置
A = torch.arange(20).reshape(5, 4)
p(A)
p(A.T)

# 对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
p(B == B.T)


# 降维
A_sum_axios0 = A.sum(axis=0)
A_sum_axios1 = A.sum(axis=1)
A_sum_axios01 = A.sum(axis=[0, 1])
A_sum = A.sum()
p(A)
p(A_sum_axios0)
p(A_sum_axios1)
p(A_sum_axios01)
p(A_sum)

# dot product 点积
x = torch.tensor([0., 1., 2., 3.])
y = torch.ones(4, dtype = torch.float32)
p(x)
p(y)
p(torch.dot(x, y))
p(torch.sum(x * y))

# 矩阵的乘法
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.ones(4, 3)
p(torch.mm(A, B))


# L2范数
u = torch.tensor([3.0, -4.0])
p(torch.norm(u))

# L1范数
p(torch.abs(u).sum())


# 练习
# 1.
A = torch.randn([3, 4])
p(A == A.T.T)

# 2.
A = torch.randn([3, 4])
B = torch.randn([3, 4])
p(A.T + B.T == (A + B).T)

# 3. 
A = torch.randn([3, 3])
p(A)
p(A + A.T)

# 4.
A = torch.randn([2, 3, 4])
p(len(A))

# 5. 
A = torch.randn([1,2,3,4,5])
p(len(A))

# 6. 
A = torch.arange(12).reshape(3, 4)
p(A)
p(A / A.sum(axis=1))