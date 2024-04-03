import torch
from d2l import torch as d2l
import numpy as np
from util import Timer, p, normal


n = 10000
a = torch.ones([n])
b = torch.ones([n])

# 自己用for来实现 --- 慢
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]

p(f'{timer.stop():.5f} sec')

# 矢量化，重载的+运算符 --- 快
timer.start()
d = a + b
p(f'{timer.stop():.5f} sec')

# 再次使用numpy进行可视化
x = np.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
d2l.plt.show()