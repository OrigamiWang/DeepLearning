# 如果没有安装pandas，请取消下一行的注释
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from util import download


# 1. download dataset
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# print(train_data.shape)
# print(test_data.shape)

# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 2. data_process

# 删掉train_data的id列和label列，test_data的id列
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 2.1 process numeric data
# 筛选出所有数值型feature的列名
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# x <- (x - μ) / σ
# 标准化数据，使得每个数值型feature的均值为0，标准差为1
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# 设置缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 2.2 process disperce data(离散)
# get_dummies: 将字符串（离散数据）转换为独热编码（One-Hot Encoding）
# dummy_na=True: 缺失值也会被编码
all_features = pd.get_dummies(all_features, dummy_na=True)

# print(all_features.shape)

# 2.3 pandas -> tensor
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# train_labels = torch.tensor(train_data)
train_labels = torch.tensor(train_data['SalePrice'].values.reshape(-1, 1), dtype=torch.float32)

# 3. train
# MSE = Σ((Y_i - Y_hat_i)^2)  / n
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(in_features, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 1))
    return net

# RMSE: Root Mean Squared Error 对数均方根误差
# 用于评价提交质量的误差指标
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, 
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 使用Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), 
                                 lr = learning_rate,
                                 weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step() # 计算完梯度后，调用step来更新模型参数
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证
# 训练数据被分成K个不重叠的子集。 然后执行K次模型训练和验证，每次在K - 1个子集上进行训练
# 并在剩余的一个子集（在该轮中没有用于训练的子集）上进行验证
def get_k_fold_data(k, i, X, y):
    # i 代表k折中的第i个折
    assert k > 1
    # // 为整数除法，向下取整
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
            
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)      
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            d2l.plt.show()
            pass
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 2, 100, 0.08, 35, 128

train_avg_res, valid_avg_res = [], []
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                        weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
f'平均验证log rmse: {float(valid_l):f}')


# 提交kaggle预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    d2l.plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
    
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)