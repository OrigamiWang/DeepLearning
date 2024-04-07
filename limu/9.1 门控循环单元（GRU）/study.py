# 门控循环单元 gated recurrent unit

# 1.
# 重置门 reset gate 
# Rt = σ(Xt*Wxr + Ht-1 * Whr + br)

# 更新门 update gate
# Zt = σ(Xt * Wxz + Ht-1 * Whz + bz)

# σ就是sigmoid函数 1 / (1 + exp(-x))

# 2. 候选隐状态
# Ht_~ = tanh(Xt * Wxh + (Rt ⊙ Ht-1) * Whh + bh)
# ⊙ 是Hadamard积，按元素乘积

# 3. 隐状态
# GRU的最终更新公式
# Ht = Zt ⊙ Ht-1 + (1 - Zt) ⊙ Ht_~

# 重置门作用是：决定在计算当前隐状态的时候，应该保留多少之前的状态信息
# 当Rt=0时，H_tilda = tanh(Xt * Wxh + bh) ，将不会考虑Ht-1的任何信息

# 当更新门Zt接近1时，模型倾向于保留旧状态Ht-1
# 当Zt接近0时，Ht会接近于候选隐状态Ht_~ 




