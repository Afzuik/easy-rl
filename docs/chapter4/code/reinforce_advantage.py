"""
第 4 章 REINFORCE + 优势函数（Advantage Function）—— 完整版策略梯度

核心改进（对应 chapter4_order.md §4.5.2）:
    在基线的基础上，用"该动作之后的折扣回报"替代整场总分，实现正确的信用分配。
    权重 = G_t - b = A(s_t, a_t)，即优势函数。

公式对照:
    折扣回报:  G_t = Σ_{k=t+1}^T γ^{k-t-1} r_k                     —— §4.5.2 解法2
    优势函数:  A(s_t, a_t) = G_t - b                                —— §4.5.2 末尾
    梯度:      ∇R̄_θ ≈ (1/N) Σ_n Σ_t  A(s_t, a_t) · ∇log π_θ(a_t|s_t)
    损失:      L = - Σ_t A(s_t, a_t) · log π_θ(a_t | s_t)

三个版本的演进:
    reinforce.py              → 权重 = R(τ)       （整场总分，无基线，无信用分配）
    reinforce_baseline.py     → 权重 = G_t - b     （折扣回报 + 基线）
    reinforce_advantage.py    → 权重 = A(s_t, a_t) （优势函数 = 本章最完整版本）

优势函数的含义:
    "在状态 s_t 采取动作 a_t，相对于平均水平有多好"
    A > 0 → 这个动作比平均好 → 提升概率
    A < 0 → 这个动作比平均差 → 降低概率
    这正是后续 Actor-Critic / A2C / PPO 的核心组件。

运行方式:
    python reinforce_advantage.py
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
# 策略网络（Actor）
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
# REINFORCE + Advantage 训练
# ──────────────────────────────────────────────────────────────────
def train_reinforce_advantage(env, policy_net, optimizer, n_episodes=1000,
                              gamma=0.99, print_every=100):
    """
    REINFORCE + 优势函数（完整版策略梯度）。

    每回合:
        1. 采样一条完整轨迹
        2. 从后往前计算折扣回报 G_t（信用分配）
        3. 用历史 G_t 的滑动平均作为基线 b
        4. 优势 A_t = G_t - b
        5. 损失 = - Σ_t A_t · log π(a_t|s_t)

    这是第 4 章策略梯度的最终形态，也是 Actor-Critic 的基础。
    """
    reward_history = []
    avg_reward_history = []
    advantage_history = []   # 记录每回合平均 |A_t|，观察优势的尺度变化

    baseline = 0.0
    baseline_alpha = 0.05

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

        # ── 2. 从后往前计算折扣回报 G_t（信用分配的核心） ──
        #  G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...
        #  递推: G_T = r_T,  G_t = r_t + γ·G_{t+1}
        T = len(rewards)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        # ── 3. 更新基线 ──
        baseline = baseline + baseline_alpha * (returns.mean().item() - baseline)

        # ── 4. 计算优势函数 A_t = G_t - b ──
        advantages = returns - baseline

        # ── 5. 构造损失: L = - Σ_t A_t · log π(a_t|s_t) ──
        policy_loss = []
        for log_prob, A_t in zip(log_probs, advantages):
            policy_loss.append(-log_prob * A_t)
        loss = torch.stack(policy_loss).sum()

        # ── 6. 反向传播 ──
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── 日志 ──
        avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
        avg_reward_history.append(avg_reward)
        advantage_history.append(advantages.abs().mean().item())

        if (ep + 1) % print_every == 0:
            print(f"  回合 {ep+1:>4d}/{n_episodes} | "
                  f"奖励: {total_reward:>4.0f} | "
                  f"基线: {baseline:>6.1f} | "
                  f"平均|A|: {advantages.abs().mean().item():>5.1f} | "
                  f"近100回合平均: {avg_reward:.1f}")

    return reward_history, avg_reward_history, advantage_history


# ──────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  第 4 章 REINFORCE + Advantage（完整版策略梯度）")
    print("  A(s_t, a_t) = G_t - b")
    print("  ∇R̄ ≈ (1/N) Σ Σ A(s_t, a_t) · ∇log π(a_t|s_t)")
    print("=" * 60)

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"\n环境: CartPole-v1")
    print(f"  状态维度: {state_dim}, 动作数: {action_dim}")
    print(f"  改进: 折扣回报 G_t（信用分配）+ 基线 b → 优势函数 A(s,a)\n")

    policy_net = PolicyNet(state_dim, action_dim, hidden_dim=128)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    print("开始训练...\n")
    rewards, avg_rewards, advantages = train_reinforce_advantage(
        env, policy_net, optimizer,
        n_episodes=1000, gamma=0.99, print_every=100
    )
    env.close()

    # ── 结果 ──
    print(f"\n[训练完成]")
    print(f"  前100回合平均奖励: {np.mean(rewards[:100]):.1f}")
    print(f"  后100回合平均奖励: {np.mean(rewards[-100:]):.1f}")
    print(f"  最高单回合奖励:    {np.max(rewards):.0f}")

    # ── 绘图：三列对比 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 左：奖励曲线
    axes[0].plot(rewards, alpha=0.3, color='seagreen', label='Episode Reward')
    axes[0].plot(avg_rewards, color='seagreen', linewidth=2, label='100-ep Avg')
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Max (500)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('REINFORCE + Advantage')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 中：前200回合放大
    axes[1].plot(rewards, alpha=0.3, color='seagreen')
    axes[1].plot(avg_rewards, color='seagreen', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward')
    axes[1].set_title('First 200 Episodes')
    axes[1].set_xlim(0, 200)
    axes[1].grid(True, alpha=0.3)

    # 右：优势函数尺度变化
    axes[2].plot(advantages, alpha=0.5, color='purple', linewidth=0.8)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Mean |A(s,a)|')
    axes[2].set_title('Advantage Magnitude over Training')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reinforce_advantage.png", dpi=150)
    plt.close()
    print(f"\n图表已保存为 reinforce_advantage.png")

    # ── 三版本对比总结 ──
    print("\n" + "=" * 60)
    print("第 4 章三个版本的对比:")
    print("  reinforce.py             → 权重 = R(τ)      (整场总分)")
    print("  reinforce_baseline.py    → 权重 = G_t - b   (折扣回报 + 基线)")
    print("  reinforce_advantage.py   → 权重 = A(s,a)    (优势函数 = 最终版)")
    print()
    print("核心要点:")
    print("  1. 信用分配: 只算动作之后的奖励（G_t），不回溯过去")
    print("  2. 折扣因子 γ: 越远的奖励权重越小（γ^{t'-t}）")
    print("  3. 基线 b: 让优势有正有负，减少方差")
    print("  4. 优势函数 A(s,a) = G_t - b → Actor-Critic 的核心")
    print("  5. 后续章节: Critic 网络直接估计 A(s,a) → A2C/PPO")
    print("=" * 60)


if __name__ == "__main__":
    main()
