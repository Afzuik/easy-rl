"""
第 4 章 REINFORCE —— 蒙特卡洛策略梯度（最简版本）

核心思想:
    跑完一回合 → 从后往前算 G_t → 用 G_t 加权 log π(a_t|s_t) → 梯度上升。
    这是策略梯度最原始、最经典的实现，对应 chapter4_order.md §4.6。

公式对照 (chapter4_order.md):
    梯度:  ∇R̄_θ ≈ (1/N) Σ_n Σ_t  G_t^n · ∇log π_θ(a_t^n | s_t^n)    —— 式 (4.3) / (4.4)
    回报:  G_t = Σ_{k=t+1}^T γ^{k-t-1} r_k = r_{t+1} + γ G_{t+1}     —— 式 (4.8)
    损失:  L = - Σ_t G_t · log π_θ(a_t | s_t)                          —— 式 (4.5) 取负

关键局限（正文 §4.5 讨论）:
    1. 没有基线 → 奖励全是正的时候，未采样动作概率被错误压低
    2. 用整场总分 G_t 给每一步加权 → 信用分配不公
    后续 reinforce_baseline.py 和 reinforce_advantage.py 分别解决这两个问题。

运行方式:
    python reinforce.py
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免在没有 GUI 的环境中报错
import matplotlib.pyplot as plt
from collections import deque
import sys
import os

# 确保中文能在 Windows 控制台正常输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ──────────────────────────────────────────────────────────────────
# 策略网络：输入状态 → 输出动作概率分布
# ──────────────────────────────────────────────────────────────────
class PolicyNet(nn.Module):
    """
    简单的两层 MLP 策略网络。

    输入: 状态 s (CartPole: 4 维连续观测)
    输出: 各动作的 log 概率 (未归一化，配合 CrossEntropyLoss 使用)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """返回 logits（未归一化概率），供 CrossEntropyLoss 使用。"""
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits

    def get_action(self, state):
        """
        给定状态，按策略采样动作。

        返回:
            action:     采样的动作 (标量)
            log_prob:   该动作的对数概率 (标量，用于梯度计算)
        """
        state = torch.FloatTensor(state).unsqueeze(0)  # (1, state_dim)
        logits = self.forward(state)                    # (1, action_dim)
        probs = F.softmax(logits, dim=-1)               # 动作概率分布
        dist = torch.distributions.Categorical(probs)   # 分类分布
        action = dist.sample()                          # 按概率采样
        log_prob = dist.log_prob(action)                # log π(a|s)
        return action.item(), log_prob


# ──────────────────────────────────────────────────────────────────
# REINFORCE 训练
# ──────────────────────────────────────────────────────────────────
def train_reinforce(env, policy_net, optimizer, n_episodes=1000,
                    gamma=0.99, print_every=100):
    """
    原始 REINFORCE 算法（无基线、用整场折扣回报 G_t 加权）。

    每回合:
        1. 用当前策略采样一条完整轨迹
        2. 从后往前计算每个时刻的折扣回报 G_t
        3. 损失 = - Σ_t G_t · log π(a_t|s_t)
        4. 梯度上升（loss 取负 → 梯度下降）

    参数:
        env:         Gymnasium 环境
        policy_net:  策略网络
        optimizer:   PyTorch 优化器
        n_episodes:  训练回合数
        gamma:       折扣因子
        print_every: 每隔多少回合打印一次
    """
    reward_history = []       # 每回合总奖励
    avg_reward_history = []   # 滑动平均奖励

    for ep in range(n_episodes):
        # ── 1. 采样一条轨迹 ──
        state, _ = env.reset()
        log_probs = []   # 每步的 log π(a_t|s_t)
        rewards = []     # 每步的即时奖励 r_t
        done = False
        total_reward = 0

        while not done:
            action, log_prob = policy_net.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            state = next_state

        reward_history.append(total_reward)

        # ── 2. 从后往前计算折扣回报 G_t ──
        #  G_T = r_T,  G_{t} = r_t + γ · G_{t+1}
        T = len(rewards)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        # ── 3. 构造损失: L = - Σ_t G_t · log π(a_t|s_t) ──
        #  式 (4.5): 最大化 Σ G_t log π → 最小化 -Σ G_t log π
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss.append(-log_prob * G_t)  # 负号 → 梯度上升变下降
        loss = torch.stack(policy_loss).sum()

        # ── 4. 反向传播 ──
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── 日志 ──
        avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
        avg_reward_history.append(avg_reward)

        if (ep + 1) % print_every == 0:
            print(f"  回合 {ep+1:>4d}/{n_episodes} | "
                  f"奖励: {total_reward:>4.0f} | "
                  f"近100回合平均: {avg_reward:.1f}")

    return reward_history, avg_reward_history


# ──────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  第 4 章 REINFORCE —— 蒙特卡洛策略梯度")
    print("  ∇R̄ ≈ (1/N) Σ Σ G_t · ∇log π(a_t|s_t)")
    print("=" * 60)

    # ── 环境 ──
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n              # 2
    print(f"\n环境: CartPole-v1")
    print(f"  状态维度: {state_dim}  (连续)")
    print(f"  动作数:   {action_dim}    (离散: 左/右)")
    print(f"  目标:     保持杆子直立，每步 +1 分，最多 500 步\n")

    # ── 网络 & 优化器 ──
    policy_net = PolicyNet(state_dim, action_dim, hidden_dim=128)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    # ── 训练 ──
    print("开始训练...\n")
    rewards, avg_rewards = train_reinforce(
        env, policy_net, optimizer,
        n_episodes=1000, gamma=0.99, print_every=100
    )

    env.close()

    # ── 结果 ──
    print(f"\n[训练完成]")
    print(f"  前100回合平均奖励: {np.mean(rewards[:100]):.1f}")
    print(f"  后100回合平均奖励: {np.mean(rewards[-100:]):.1f}")
    print(f"  最高单回合奖励:    {np.max(rewards):.0f}")

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    axes[0].plot(avg_rewards, color='steelblue', linewidth=2, label='100-ep Avg')
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Max (500)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('REINFORCE (vanilla)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rewards, alpha=0.3, color='steelblue')
    axes[1].plot(avg_rewards, color='steelblue', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward')
    axes[1].set_title('REINFORCE (first 200 episodes)')
    axes[1].set_xlim(0, 200)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reinforce.png", dpi=150)
    plt.close()
    print(f"\n图表已保存为 reinforce.png")

    print("\n" + "=" * 60)
    print("核心要点:")
    print("  1. 策略梯度 = 加权的最大对数似然（权重 = G_t）")
    print("  2. 同策略 (on-policy): 数据只用一次，用完即弃")
    print("  3. 无基线 → 奖励全是正时，未采样动作概率被压低")
    print("  4. 整场总分加权 → 信用分配不公（好/坏动作同权重）")
    print("=" * 60)


if __name__ == "__main__":
    main()
