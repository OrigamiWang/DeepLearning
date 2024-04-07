# 目的：预测x_t
# x_t = P(x_t | x_t-1, ..., x1)

# 自回归模型(autoregressive modules)：
    # 使用预测序列: x_t-1, ..., x_t-τ

# 隐变量自回归模型(latent autoregressive models)
    # x_t_hat = P(x_t | h_t)
    # h_t = g(h_t-1, x_t-1)
    # 其中, h_t 是对过去预测的总结
    # 图示：

    # x_t-1_hat               x_t_hat
    #     ^                       ^
    #     |                       |
    #    h_t-1       -->         h_t
    #     ^                       ^
    #     |                       |
    #    x_t-2                   x_t-1

# 整个序列的估计值： P(x_1, ..., x_T) = ΠP(x_t | x_t-1, ..., x_1)

# 马尔可夫模型(Markov model)
# 如果x_t-1, ..., x_t-τ近似精确，我们说序列满足马尔科夫条件(Markov condition)
# 一阶马尔可夫模型(first-order Markov model): τ=1
# P(x1, ..., x_T) = ΠP(x_t | x_t-1)当P(x_1 | x_0) = P(x_1)

# 因果关系
# x_t+1 = f(x_t) + ε，反之不行，因为未来不能改变过去

