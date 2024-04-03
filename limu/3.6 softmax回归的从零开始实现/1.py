import torch
from IPython import display
from d2l import torch as d2l
from util import p, Accumulator, Animator

# load data
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# init model params
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# define softmax
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# keepdim: keep the dimension
p(X.sum(0, keepdim=True))
p(X.sum(1, keepdim=True))

def softmax(X):
    X_exp = torch.exp(X)
    return X_exp / X_exp.sum(1, keepdim=True)

# mu=0, sigma^2=1, shape=2*5 
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_sum = X_prob.sum(1)
p(X_prob)
p(X_sum)

# define model
# reshape后，保证X能与进行矩阵乘法
# +b是广播操作
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)



# defibe loss function
# 交叉熵损失函数
# 交叉熵损失函数（Cross-Entropy Loss Function），也称为对数损失（Log Loss）
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
p(y_hat[[0, 1], y])

# 交叉熵采用真实标签的预测概率的负对数似然
# 这个函数的作用是计算每个样本的真实类别对应的预测概率的负对数，
# 然后返回这些值的平均值。这是交叉熵损失的核心计算。
def cross_entropy(y_hat, y):
    # range(len(y_hat): y_hat的索引0 ~ i
    # y_hat[range(len(y_hat)), y]: 高级索引，选取y_hat[i][y[i]]
    # -log: 最小化损失
    return - torch.log(y_hat[range(len(y_hat)), y])

p(cross_entropy(y_hat, y))


# 分类精度
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # argmax获得每行中最大元素的索引来获得预测类别
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

p(accuracy(y_hat, y) / len(y))

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        # 因为updater使用的是d2l的手搓sgd，因此进入false
        # 想要进入true，得这样：torch.optim.SGD([W, b], lr, momentum=0.9)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        d2l.plt.pause(0.1) # 这句代码能够让图像动态显示
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

if __name__ == '__main__': 
    # 训练前，一共10个输出，因此accuracy为0.1左右
    # p(evaluate_accuracy(net, test_iter))

    # begin to train
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()

    # predict
    predict_ch3(net, test_iter)
    d2l.plt.show()
