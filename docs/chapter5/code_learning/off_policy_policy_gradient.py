"""
异策略策略梯度（Off-Policy Policy Gradient）教学代码。

为了只突出重要性权重，本例使用一个单状态、双动作的老虎机:
    动作 0 的平均奖励约为 0.2
    动作 1 的平均奖励约为 1.0

行为策略 pi_old 固定产生一批数据，目标策略 pi_theta 多次复用该批数据:

    J(theta) = E_{a~pi_old}[pi_theta(a)/pi_old(a) * A(a)]

运行:
    python off_policy_policy_gradient.py
"""
