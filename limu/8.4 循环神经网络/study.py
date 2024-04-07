# P(x_t | h_t-1)
# h_t = f(x_t, h_t-1)

# 循环神经网络（recurrent neural networks, RNNs）是具有隐状态的神经网络

# 无隐状态的RNN
# H = WX + b
# O = HW + b

# 有隐状态的RNN
# H_t = Φ(XW + H_t-1W + b)
# O = HW + b

# RNN中执行计算的层叫：recurrent layer循环层


# 困惑度（Perplexity）
# 公式：exp(-1/n * ΣlogP(x_t | x_t-1, ..., x1))

