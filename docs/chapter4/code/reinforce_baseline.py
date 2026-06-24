"""
第 4 章 REINFORCE + 基线（Baseline）—— 解决"奖励总是正的"问题

核心改进（对应 chapter4_order.md §4.5.1）:
    原始 REINFORCE 中所有奖励都是正的 → 未采样到的动作概率被错误压低。
    减一个基线 b 让奖励有正有负 → 低于平均的动作被惩罚，高于平均的被奖励。

公式对照:
    梯度:  ∇R̄_θ ≈ (1/N) Σ_n Σ_t  (G_t - b) · ∇log π_θ(a_t | s_t)    —— §4.5.1
    基线:  b = 历史回报的滑动平均（最简单的做法）
    损失:  L = - Σ_t (G_t - b) · log π_θ(a_t | s_t)

    其中 G_t = Σ_{k=t+1}^T γ^{k-t-1} r_k（折扣回报，§4.5.2 的信用分配）

与 reinforce.py 的区别:
    1. 权重从 G_t 变为 G_t - b（基线修正）
    2. 使用折扣回报 G_t（而非整场总分 R(τ)）—— 已包含信用分配
    3. 训练更稳定，方差更小

运行方式:
    python reinforce_baseline.py
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ──────────────────────────────────────────────────────────────────
# 策略网络
# ──────────────────────────────────────────────────────────────────
class PolicyNet(nn.Module):
    """两层 MLP，输入状态 → 输出动作 logits。"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_action(self, state):
        """按策略采样动作，返回 (action, log_prob)。"""
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


# ──────────────────────────────────────────────────────────────────
# REINFORCE + Baseline 训练
# ──────────────────────────────────────────────────────────────────
def train_reinforce_baseline(env, policy_net, optimizer, n_episodes=1000,
                             gamma=0.99, print_every=100):
    """
    REINFORCE + 基线（滑动平均）。

    每回合:
        1. 采样一条完整轨迹
        2. 从后往前计算折扣回报 G_t
        3. 用历史 G_t 的滑动平均作为基线 b
        4. 损失 = - Σ_t (G_t - b) · log π(a_t|s_t)

    关键: 当 G_t > b → 正权重 → 提升该动作概率
          当 G_t < b → 负权重 → 降低该动作概率
    """
    reward_history = []
    avg_reward_history = []
    baseline = 0.0           # 基线 b，初始为 0
    baseline_alpha = 0.05    # 滑动平均系数（类似学习率）

    for ep in range(n_episodes):
        # ── 1. 采样轨迹 ──
        state, _ = env.reset()
        log_probs = []
        rewards = []
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
        T = len(rewards)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        # ── 3. 更新基线（滑动平均） ──
        #  b ← b + α · (mean(G_t) - b)
        baseline = baseline + baseline_alpha * (returns.mean().item() - baseline)

        # ── 4. 构造损失: L = - Σ_t (G_t - b) · log π(a_t|s_t) ──
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns):
            advantage = G_t - baseline          # 优势估计（简化版）
            policy_loss.append(-log_prob * advantage)
        loss = torch.stack(policy_loss).sum()

        # ── 5. 反向传播 ──
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── 日志 ──
        avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
        avg_reward_history.append(avg_reward)

        if (ep + 1) % print_every == 0:
            print(f"  回合 {ep+1:>4d}/{n_episodes} | "
                  f"奖励: {total_reward:>4.0f} | "
                  f"基线 b: {baseline:>6.1f} | "
                  f"近100回合平均: {avg_reward:.1f}")

    return reward_history, avg_reward_history


# ──────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  第 4 章 REINFORCE + Baseline")
    print("  ∇R̄ ≈ (1/N) Σ Σ (G_t - b) · ∇log π(a_t|s_t)")
    print("=" * 60)

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"\n环境: CartPole-v1")
    print(f"  状态维度: {state_dim}, 动作数: {action_dim}")
    print(f"  改进: 添加基线 b (滑动平均)，让奖励有正有负\n")

    policy_net = PolicyNet(state_dim, action_dim, hidden_dim=128)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    print("开始训练...\n")
    rewards, avg_rewards = train_reinforce_baseline(
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

    axes[0].plot(rewards, alpha=0.3, color='coral', label='Episode Reward')
    axes[0].plot(avg_rewards, color='coral', linewidth=2, label='100-ep Avg')
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Max (500)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('REINFORCE + Baseline')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rewards, alpha=0.3, color='coral')
    axes[1].plot(avg_rewards, color='coral', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward')
    axes[1].set_title('REINFORCE + Baseline (first 200 episodes)')
    axes[1].set_xlim(0, 200)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reinforce_baseline.png", dpi=150)
    plt.close()
    print(f"\n图表已保存为 reinforce_baseline.png")

    print("\n" + "=" * 60)
    print("核心要点:")
    print("  1. 基线 b = 历史 G_t 的滑动平均")
    print("  2. G_t > b → 正权重（好动作，提升概率）")
    print("  3. G_t < b → 负权重（差动作，降低概率）")
    print("  4. 解决未采样动作概率被错误压低的问题")
    print("  5. b 也可用 Critic 网络估计 → Actor-Critic 的起点")
    print("=" * 60)


if __name__ == "__main__":
    main()
