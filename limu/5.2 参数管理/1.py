import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

# 参数访问
# print(net[0].state_dict()) # hidden: weight[8 * 4], bias[8 * 1] 
# print(net[1].state_dict()) # ReLU
# print(net[2].state_dict()) # out: weight[1 * 8], bias[1 * 1]

# 目标参数
# print(type(net[2].bias))
# print(net[2].bias)
# print(net[2].bias.data)

# print(net[2].weight.grad == None) # 还没有backward

# 一次性访问所有参数
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block: {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# print(rgnet(X))

# print(rgnet)
# print(rgnet[0][2][0].bias)


# 参数初始化

# - 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        # weight初始化为正态分布
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # weight 初始化为常量
        # nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

# net.apply(init_normal)
# print(net[0].weight.data[0], net[0].bias.data[0])

# Xavier初始化：根据输入神经元数量n_in和输出神经元数量n_out决定weight的规模
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
# net[0].apply(init_xavier)
# net[2].apply(init_42)
# print(net[0].weight.data[0])
# print(net[2].weight.data)


# 自定义初始化
#        U(5, 10), p = 1/4
# ω ~ {  0       , p = 1/2  
#        U(-10,-5),p = 1/4

def my_init(m):
    if type(m) == nn.Linear:
        print("init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

# net.apply(my_init)
# print(net[0].weight[:2])

# 参数绑定
# 如果希望在多个层间共享参数，我们可以设置一个稠密层，使用它的参数来设置另一个层的参数

shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
print(net(X))
# 参数相等
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 同一对象
print(net[2].weight.data[0] == net[4].weight.data[0])






