# LSTM 长短期记忆网络 long short-term memory

# 1. input gate, output gate, forget gate

# It = σ(Xt @ Wxi + Ht-1 @ Whi + bi)
# Ft = σ(Xt @ Wxf + Ht-1 @ Whf + bf)
# Ot = σ(Xt @ Wxo + Ht-1 @ Who + bo)

# 2. 候选记忆元 candidate memory cell
# Ct_tilda = tanh(Xt @ Wxc + Ht-1 @ Whc + bc) 

# 3. 记忆元
# Ct = Ft ⊙ Ct-1 + It ⊙ Ct_tilda

# 4. 隐状态
# Ht = Ot ⊙ tanh(Ct)


