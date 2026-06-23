"""
第 4 章 REINFORCE —— 蒙特卡洛策略梯度（Monte Carlo Policy Gradient）

核心思想（书中 4.6 节）:
    直接对策略 π_θ 建模（一个神经网络），用蒙特卡洛方法估计策略梯度。
    跑完一回合 → 倒推 G_t → 加权交叉熵 → 反向传播。

算法流程:
    1. 用当前策略 π_θ 跑完一回合，得到 (s_1,a_1,r_1), ..., (s_T,a_T,r_T)
    2. 从后向前计算每个 G_t = r_t + γ * G_{t+1}
    3. 对每个 t，计算梯度 G_t * ∇log π_θ(a_t | s_t)
    4. 累加梯度，更新参数

关键公式（式 4.4）:
    ∇R̄_θ ≈ (1/N) Σ_n Σ_t R(τ^n) ∇log p_θ(a_t^n | s_t^n)

    → 即：最大化 G_t * log π(a_t | s_t, θ)
    → 等同于分类问题的交叉熵损失，只是每一项乘了权重 G_t

要点:
    - 同策略（on-policy）：采样数据只用一次，用完必须重新采样
    - 免模型（model-free）：不需要知道环境的状态转移概率
    - 策略网络输出的是动作概率分布，按分布采样（保证探索性）

运行方式:
    python reinforce_basic.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from cliff_walking_env_learning import CliffWalkingEnv

# ── 检查是否有 PyTorch ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ══════════════════════════════════════════════════════════════════════════════
#  策略网络：输入状态 → 输出动作概率分布
# ══════════════════════════════════════════════════════════════════════════════

class PolicyNet(nn.Module):
    """
    策略网络 π_θ(a|s)

    对应书中图 4.2：
        输入: 状态 s（独热编码）
        输出: 每个动作的概率（经过 softmax）

    结构: 输入层(48) → 隐藏层(128) → 隐藏层(64) → 输出层(4)
    """

    def __init__(self, n_states, n_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, n_actions)

    def forward(self, x):
        """
        前向传播。

        参数:
            x: 状态张量，shape 为 (batch, n_states) 或 (n_states,)
        返回:
            probs: 动作概率分布，shape 为 (batch, n_actions) 或 (n_actions,)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  REINFORCE 算法核心
# ══════════════════════════════════════════════════════════════════════════════

class REINFORCE:
    """
    REINFORCE（蒙特卡洛策略梯度）

    书中图 4.20 的两个核心函数:
        - sample():  用当前策略跑一回合，收集数据
        - learn():   用收集的数据计算 G_t，构造损失，更新网络

    参数:
        n_states:  状态空间大小
        n_actions: 动作空间大小
        gamma:     折扣因子（默认 0.99）
        lr:        学习率（默认 1e-3）
        hidden_dim: 隐藏层维度
    """

    def __init__(self, n_states, n_actions, gamma=0.99, lr=1e-3, hidden_dim=128):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

        self.policy_net = PolicyNet(n_states, n_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # 记录训练数据
        self.episode_rewards = []       # 每个回合的总奖励

    def _state_to_tensor(self, state):
        """将状态索引转为独热向量张量。"""
        x = np.zeros(self.n_states, dtype=np.float32)
        x[state] = 1.0
        return torch.tensor(x, dtype=torch.float32)

    def choose_action(self, state):
        """
        根据当前策略选择动作。

        返回:
            action:  选择的动作编号
            log_prob: 该动作的对数概率（用于后续梯度计算）
        """
        state_tensor = self._state_to_tensor(state).unsqueeze(0)  # (1, n_states)
        probs = self.policy_net(state_tensor).squeeze(0)          # (n_actions,)

        # 按概率分布采样 —— 书中强调这是保证探索性的关键
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob

    def sample_episode(self, env, max_steps=500):
        """
        对应书中图 4.20 的 sample() 函数。

        用当前策略跑完一个完整回合，收集 (s_t, log_prob, r_t) 序列。

        返回:
            episode: [(log_prob_1, r_1), (log_prob_2, r_2), ..., (log_prob_T, r_T)]
            total_reward: 本回合总奖励
        """
        episode = []
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < max_steps:
            action, log_prob = self.choose_action(state)
            next_state, reward, done = env.step(state, action)

            episode.append((log_prob, reward))

            total_reward += reward
            state = next_state
            step += 1

        return episode, total_reward

    def learn(self, episode):
        """
        对应书中图 4.20 的 learn() 函数。

        用一回合数据：
        1. 从后向前计算 G_t（式 4.8: G_t = r_t + γ * G_{t+1}）
        2. 构造策略梯度损失: loss = -Σ G_t * log_prob_t（式 4.5）
        3. 反向传播更新参数

        参数:
            episode: [(log_prob_1, r_1), ..., (log_prob_T, r_T)]

        返回:
            loss_val: 本回合的损失值（用于监控）
        """
        # ── 第 1 步：从后向前计算 G_t ──
        # 书中 4.6.3 节的实现技巧：从后往前一次遍历
        returns = []
        G = 0.0
        for _, reward in reversed(episode):
            G = reward + self.gamma * G       # 式 4.8: G_t = r_{t+1} + γ G_{t+1}
            returns.append(G)
        returns.reverse()                      # 恢复为从前往后的顺序

        # 转为张量
        returns = torch.tensor(returns, dtype=torch.float32)

        # ── 第 2 步：构造策略梯度损失 ──
        # 式 4.5: loss = - (1/T) Σ_t G_t * log π(a_t | s_t)
        # 负号是因为我们要最大化期望奖励，但优化器做的是最小化
        log_probs = torch.stack([item[0] for item in episode])

        # 标准化 G_t（可选但有助于稳定训练）
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = -torch.sum(log_probs * returns) / len(episode)

        # ── 第 3 步：反向传播 ──
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()


# ══════════════════════════════════════════════════════════════════════════════
#  训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def train(env, n_episodes=1000, gamma=0.99, lr=1e-3, hidden_dim=128):
    """
    训练 REINFORCE 算法。

    参数:
        env:         悬崖行走环境
        n_episodes:  训练回合数
        gamma:       折扣因子
        lr:          学习率
        hidden_dim:  隐藏层维度

    返回:
        agent:   训练好的 REINFORCE agent
        rewards: 每个回合的总奖励列表
    """
    agent = REINFORCE(
        n_states=env.n_states,
        n_actions=env.n_actions,
        gamma=gamma,
        lr=lr,
        hidden_dim=hidden_dim
    )

    reward_history = []

    for ep in range(n_episodes):
        # ── 第 1 步：采样一个完整回合 ──
        episode, total_reward = agent.sample_episode(env)

        # ── 第 2 步：用这个回合的数据更新策略 ──
        loss = agent.learn(episode)

        reward_history.append(total_reward)

        # ── 打印进度 ──
        if (ep + 1) % 100 == 0:
            avg_r = np.mean(reward_history[-100:])
            print(f"  回合 {ep+1:>4d}/{n_episodes} | "
                  f"近100回合平均奖励: {avg_r:>6.1f} | "
                  f"loss: {loss:.3f}")

    return agent, reward_history


# ══════════════════════════════════════════════════════════════════════════════
#  评估 & 可视化
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(agent, env, n_episodes=10):
    """
    用贪心策略（不采样，直接选概率最大的动作）评估当前策略。

    返回:
        avg_reward: 平均总奖励
        success_rate: 成功到达终点的比例
    """
    rewards = []
    successes = 0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < 500:
            state_tensor = agent._state_to_tensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = agent.policy_net(state_tensor).squeeze(0)
            action = torch.argmax(probs).item()  # 贪心：选概率最大的动作

            state, reward, done = env.step(state, action)
            total_reward += reward
            step += 1

        rewards.append(total_reward)
        if done and total_reward > -500:  # 到达终点（而非超时/掉悬崖）
            successes += 1

    return np.mean(rewards), successes / n_episodes


def extract_policy(agent, env):
    """
    从策略网络中提取贪心策略（用于可视化）。

    返回:
        policy: 每个状态的最优动作数组 (n_states,)
    """
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        state_tensor = agent._state_to_tensor(s).unsqueeze(0)
        with torch.no_grad():
            probs = agent.policy_net(state_tensor).squeeze(0)
        policy[s] = torch.argmax(probs).item()
    return policy


# ══════════════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not HAS_TORCH:
        print("❌ 未检测到 PyTorch，请先安装：pip install torch")
        return

    print("=" * 60)
    print("  第 4 章 REINFORCE —— 蒙特卡洛策略梯度")
    print("  ∇R̄_θ ≈ (1/N) Σ_n Σ_t G_t ∇log π_θ(a_t | s_t)")
    print("=" * 60)

    env = CliffWalkingEnv()

    # ── 训练 ──
    print("\n[开始训练]")
    agent, rewards = train(
        env,
        n_episodes=1000,
        gamma=0.99,
        lr=1e-3,
        hidden_dim=128
    )

    # ── 学习曲线 ──
    print(f"\n[学习曲线]")
    print(f"  前100回合平均奖励: {np.mean(rewards[:100]):.1f}")
    print(f"  后100回合平均奖励: {np.mean(rewards[-100:]):.1f}")

    # ── 评估 ──
    print(f"\n[贪心策略评估 (ε=0)]:")
    avg_r, succ_rate = evaluate(agent, env, n_episodes=10)
    print(f"  10次测试平均奖励: {avg_r:.1f}")
    print(f"  成功率: {succ_rate*100:.0f}%")
    print(f"  (最优路径：贴悬崖走，步数=13，总奖励≈-13)")

    # ── 策略可视化 ──
    policy = extract_policy(agent, env)
    env.print_policy(policy, "REINFORCE 学到的策略")

    # ── 各状态动作概率分布（抽样展示） ──
    print(f"\n[策略网络在各状态的输出概率（部分展示）]")
    print(f"  {'状态':>6s} | {'↑(上)':>8s} {'→(右)':>8s} {'↓(下)':>8s} {'←(左)':>8s} | {'贪心动作':>8s}")
    print(f"  {'-'*6}-+-{'-'*8} {'-'*8} {'-'*8} {'-'*8}-+-{'-'*8}")

    # 展示几个关键位置
    show_states = [
        (3, 0, "起点S"),    # 起点
        (3, 1, "悬崖旁"),   # 起点右边，紧挨悬崖
        (2, 5, "安全区"),   # 安全区域
        (2, 10, "终点前"),  # 终点上方
    ]
    for r, c, desc in show_states:
        idx = env._coord_to_idx((r, c))
        state_tensor = agent._state_to_tensor(idx).unsqueeze(0)
        with torch.no_grad():
            probs = agent.policy_net(state_tensor).squeeze(0).numpy()
        best = np.argmax(probs)
        print(f"  ({r},{c}) {desc:>4s} | "
              f"{probs[0]:>8.3f} {probs[1]:>8.3f} {probs[2]:>8.3f} {probs[3]:>8.3f} | "
              f"{env.action_symbols[best]:>8s}")

    print("\n" + "=" * 60)
    print("核心要点:")
    print("  1. REINFORCE 是蒙特卡洛策略梯度：回合结束后才更新")
    print("  2. 同策略 (on-policy)：数据只能用一次，用完即弃")
    print("  3. 策略网络输出动作概率，按概率采样保证探索性")
    print("  4. loss = -Σ G_t * log π(a_t|s_t)，和分类交叉熵只差权重 G_t")
    print("  5. G_t 从后往前递推计算：G_t = r_t + γ * G_{t+1}")
    print("  6. 不需要知道环境模型 → 免模型 (model-free)")
    print("=" * 60)


if __name__ == "__main__":
    main()
