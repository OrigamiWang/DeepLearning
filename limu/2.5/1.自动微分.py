import torch
from util import p

# 1.
x = torch.arange(4.0)
p(x)

x.requires_grad_(True)
p(x.grad)

y = 2 * torch.dot(x, x)
p(y)

y.backward()
p(x.grad)

p(x.grad == 4 * x)

# 2.
x.grad.zero_()
y = x.sum()
y.backward()
p(x.grad)

# 3.
x.grad.zero_()
y = x * x
y.sum().backward()
p(x.grad)

# 4.
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
p(x.grad == u)

x.grad.zero_()
y.sum().backward()
p(x.grad)
p(x.grad == 2 * x)

# 5.
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
p(a)
d = f(a)
p(d)
d.backward()
p(a.grad == d / a)

# 
x.grad.zero_()
y = x ** 3
y.sum().backward()
p(x.grad == 3 * (x ** 2))
p(x.grad == 6 * x)


