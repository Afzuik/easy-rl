"""
第 4 章 REINFORCE with Baseline —— 带基线的蒙特卡洛策略梯度

核心思想（书中 4.5.1 节 & 4.5.2 节）:
    在 REINFORCE 的基础上加入两个关键技巧:

    技巧 1 —— 基线 (Baseline):  解决"奖励总是正的"问题
        ∇R̄_θ ≈ (1/N) Σ_n Σ_t (G_t - b) ∇log π_θ(a_t | s_t)
        减去基线 b 让奖励有正有负，避免未采样到的动作概率被错误压低。
        最简单的基线: b = 过去回报的滑动平均。

    技巧 2 —— 信用分配 (Credit Assignment):  只算该动作之后的奖励
        用 G_t（从 t 时刻开始的折扣回报）代替整场总分 R(τ):
        G_t = Σ_{k=t+1}^T γ^{k-t-1} r_k

    两个技巧结合 → 优势函数 (Advantage Function):
        A(s_t, a_t) = G_t - b(s_t)
        含义: 在状态 s_t 采取动作 a_t，相对于平均水平有多好。

    本章用两种方式实现基线:
        1. 简单基线: b = 过去回报的滑动平均
        2. 学习基线: b(s) = 一个价值网络（评论员 critic），估计 V(s)

对比 REINFORCE 基础版:
    - 基础版:   loss = -Σ G_t * log π(a_t|s_t)
    - 基线版:   loss = -Σ (G_t - b) * log π(a_t|s_t)
    - 优势版:   loss = -Σ (G_t - V(s_t)) * log π(a_t|s_t)

运行方式:
    python reinforce_baseline.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from cliff_walking_env_learning import CliffWalkingEnv

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
#  策略网络（演员 Actor）
# ══════════════════════════════════════════════════════════════════════════════

class PolicyNet(nn.Module):
    """
    策略网络 π_θ(a|s) —— 演员（Actor）

    输入: 状态 s（独热编码）
    输出: 每个动作的概率分布（softmax）
    """

    def __init__(self, n_states, n_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  价值网络（评论员 Critic）
# ══════════════════════════════════════════════════════════════════════════════

class ValueNet(nn.Module):
    """
    价值网络 V_φ(s) —— 评论员（Critic）

    输入: 状态 s（独热编码）
    输出: 该状态的标量价值 V(s)

    作用:
        - 作为基线 b(s) = V(s)
        - 优势函数 A(s,a) = G_t - V(s_t)
        - 这是 Actor-Critic 方法的雏形
    """

    def __init__(self, n_states, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)   # 输出一个标量 V(s)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)  # 去掉最后一维，得到 (batch,) 或标量


# ══════════════════════════════════════════════════════════════════════════════
#  REINFORCE with Baseline 算法核心
# ══════════════════════════════════════════════════════════════════════════════

class REINFORCEBaseline:
    """
    REINFORCE with Baseline

    书中 4.5 节的两个技巧:
        - 基线 b: 减去后让优势有正有负
        - 信用分配: 用 G_t（从 t 开始的折扣回报）而不是整场总分

    三种基线模式:
        'none':     不使用基线（等同于基础 REINFORCE）
        'simple':   使用回报的滑动平均作为基线
        'learned':  使用价值网络 V(s) 作为基线（Actor-Critic 雏形）

    参数:
        n_states:    状态空间大小
        n_actions:   动作空间大小
        gamma:       折扣因子（默认 0.99）
        lr_actor:    演员学习率（默认 1e-3）
        lr_critic:   评论员学习率（默认 1e-3）
        hidden_dim:  隐藏层维度
        baseline_type: 基线类型
    """

    def __init__(self, n_states, n_actions, gamma=0.99,
                 lr_actor=1e-3, lr_critic=1e-3, hidden_dim=128,
                 baseline_type='learned'):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.baseline_type = baseline_type

        # 演员: 策略网络
        self.policy_net = PolicyNet(n_states, n_actions, hidden_dim)
        self.optimizer_actor = torch.optim.Adam(self.policy_net.parameters(), lr=lr_actor)

        # 评论员: 价值网络（仅 learned 模式使用）
        if baseline_type == 'learned':
            self.value_net = ValueNet(n_states, hidden_dim)
            self.optimizer_critic = torch.optim.Adam(self.value_net.parameters(), lr=lr_critic)
        else:
            self.value_net = None
            self.optimizer_critic = None

        # 简单基线: 维护回报的滑动平均
        self.running_baseline = 0.0
        self.baseline_alpha = 0.05   # 滑动平均系数

    def _state_to_tensor(self, state):
        """将状态索引转为独热向量张量。"""
        x = np.zeros(self.n_states, dtype=np.float32)
        x[state] = 1.0
        return torch.tensor(x, dtype=torch.float32)

    def choose_action(self, state):
        """
        根据当前策略采样动作。

        返回:
            action:   动作编号
            log_prob: 对数概率（用于梯度计算）
        """
        state_tensor = self._state_to_tensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor).squeeze(0)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def sample_episode(self, env, max_steps=500):
        """
        用当前策略跑完一个完整回合。

        返回:
            episode: [(state_1, log_prob_1, reward_1), ..., (state_T, log_prob_T, reward_T)]
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

            # 记录 state 是为了计算 V(s_t) 作为基线
            episode.append((state, log_prob, reward))

            total_reward += reward
            state = next_state
            step += 1

        return episode, total_reward

    def learn(self, episode):
        """
        用一回合数据更新策略网络（和价值网络）。

        流程:
        1. 从后向前计算每个时间步的 G_t（折扣回报）
        2. 计算基线 b_t:
           - 'none':    b_t = 0
           - 'simple':  b_t = running_baseline
           - 'learned': b_t = V(s_t)（价值网络预测）
        3. 计算优势 A_t = G_t - b_t
        4. Actor 损失:  loss_a = -Σ A_t * log π(a_t|s_t)
        5. Critic 损失: loss_c = Σ (V(s_t) - G_t)²  （仅 learned 模式）
        6. 反向传播

        参数:
            episode: [(state_1, log_prob_1, reward_1), ..., (state_T, ...)]

        返回:
            loss_actor:  演员损失
            loss_critic: 评论员损失（无评论员时返回 0）
        """
        T = len(episode)

        # ── 第 1 步：从后向前计算 G_t ──
        # 书中 4.6.3 节 & 式 4.8: G_t = r_t + γ * G_{t+1}
        G = np.zeros(T, dtype=np.float32)
        running_G = 0.0
        for t in reversed(range(T)):
            _, _, reward = episode[t]
            running_G = reward + self.gamma * running_G
            G[t] = running_G

        # ── 第 2 步：计算基线 ──
        if self.baseline_type == 'none':
            baselines = np.zeros(T, dtype=np.float32)

        elif self.baseline_type == 'simple':
            # 书中 4.5.1 节: b ≈ E[R(τ)]，用滑动平均维护
            self.running_baseline = (
                self.running_baseline * (1 - self.baseline_alpha)
                + np.mean(G) * self.baseline_alpha
            )
            baselines = np.full(T, self.running_baseline, dtype=np.float32)

        elif self.baseline_type == 'learned':
            # 用价值网络预测 V(s_t) 作为基线
            states_tensor = torch.stack([
                self._state_to_tensor(episode[t][0]) for t in range(T)
            ])  # (T, n_states)
            with torch.no_grad():
                baselines = self.value_net(states_tensor).numpy()

        # ── 第 3 步：计算优势 A_t = G_t - b_t ──
        advantages = G - baselines       # (T,)

        # 标准化优势（可选但有助于稳定训练）
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── 第 4 步：Actor 损失 ──
        # 式 4.7: loss = -Σ (G_t - b) * log π(a_t|s_t)
        #          = -Σ A_t * log π(a_t|s_t)
        log_probs = torch.stack([episode[t][1] for t in range(T)])
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

        actor_loss = -torch.sum(log_probs * advantages_tensor) / T

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # ── 第 5 步：Critic 损失（仅 learned 模式） ──
        critic_loss_val = 0.0
        if self.baseline_type == 'learned':
            # 价值网络的目标是逼近真实的 G_t
            states_tensor = torch.stack([
                self._state_to_tensor(episode[t][0]) for t in range(T)
            ])
            predicted_values = self.value_net(states_tensor)     # V(s_t)
            target_values = torch.tensor(G, dtype=torch.float32) # G_t（真实回报）

            # MSE 损失: loss_c = (1/T) Σ (V(s_t) - G_t)²
            critic_loss = F.mse_loss(predicted_values, target_values)

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            critic_loss_val = critic_loss.item()

        return actor_loss.item(), critic_loss_val


# ══════════════════════════════════════════════════════════════════════════════
#  训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def train(env, n_episodes=1000, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3,
          hidden_dim=128, baseline_type='learned'):
    """
    训练 REINFORCE with Baseline。

    参数:
        env:            悬崖行走环境
        n_episodes:     训练回合数
        gamma:          折扣因子
        lr_actor:       演员学习率
        lr_critic:      评论员学习率
        hidden_dim:     隐藏层维度
        baseline_type:  基线类型 ('none', 'simple', 'learned')

    返回:
        agent:   训练好的 agent
        rewards: 每个回合的总奖励列表
    """
    agent = REINFORCEBaseline(
        n_states=env.n_states,
        n_actions=env.n_actions,
        gamma=gamma,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        hidden_dim=hidden_dim,
        baseline_type=baseline_type
    )

    reward_history = []
    loss_a_history = []
    loss_c_history = []

    for ep in range(n_episodes):
        # ── 采样 ──
        episode, total_reward = agent.sample_episode(env)

        # ── 学习 ──
        loss_a, loss_c = agent.learn(episode)

        reward_history.append(total_reward)
        loss_a_history.append(loss_a)
        loss_c_history.append(loss_c)

        # ── 打印进度 ──
        if (ep + 1) % 100 == 0:
            avg_r = np.mean(reward_history[-100:])
            avg_loss_a = np.mean(loss_a_history[-100:])
            bl_info = ""
            if baseline_type == 'simple':
                bl_info = f" | baseline: {agent.running_baseline:.1f}"
            elif baseline_type == 'learned':
                avg_loss_c = np.mean(loss_c_history[-100:])
                bl_info = f" | critic_loss: {avg_loss_c:.3f}"

            print(f"  回合 {ep+1:>4d}/{n_episodes} | "
                  f"近100回合平均奖励: {avg_r:>6.1f} | "
                  f"actor_loss: {avg_loss_a:.3f}{bl_info}")

    return agent, reward_history


# ══════════════════════════════════════════════════════════════════════════════
#  评估 & 可视化
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(agent, env, n_episodes=10):
    """用贪心策略评估。"""
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
            action = torch.argmax(probs).item()

            state, reward, done = env.step(state, action)
            total_reward += reward
            step += 1

        rewards.append(total_reward)
        if done and total_reward > -500:
            successes += 1

    return np.mean(rewards), successes / n_episodes


def extract_policy(agent, env):
    """从策略网络提取贪心策略。"""
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        state_tensor = agent._state_to_tensor(s).unsqueeze(0)
        with torch.no_grad():
            probs = agent.policy_net(state_tensor).squeeze(0)
        policy[s] = torch.argmax(probs).item()
    return policy


def show_value_function(agent, env):
    """展示价值网络估计的 V(s)（仅 learned 基线模式）。"""
    if agent.value_net is None:
        return

    print(f"\n[价值网络 V(s) —— 评论员对每个状态的价值估计]")
    print("  (值越高 = 从该状态出发期望能拿到的折扣回报越高)")
    print()

    for r in range(env.rows):
        row_str = ""
        for c in range(env.cols):
            idx = env._coord_to_idx((r, c))
            # 跳过悬崖和特殊位置
            if env._is_cliff(r, c):
                row_str += f"{'C':>8s}"
            elif (r, c) == env.goal:
                row_str += f"{'G':>8s}"
            elif (r, c) == env.start:
                row_str += f"{'S':>8s}"
            else:
                state_tensor = agent._state_to_tensor(idx).unsqueeze(0)
                with torch.no_grad():
                    v = agent.value_net(state_tensor).item()
                row_str += f"{v:>8.2f}"
        print(f"  {row_str}")


# ══════════════════════════════════════════════════════════════════════════════
#  对比实验: 三种基线模式
# ══════════════════════════════════════════════════════════════════════════════

def compare_baselines(env, n_episodes=800):
    """
    对比三种基线模式的训练效果:
        - 无基线 (none)
        - 简单滑动平均基线 (simple)
        - 学习基线 / 价值网络 (learned)
    """
    print("\n" + "=" * 60)
    print("  对比实验: 不同基线对 REINFORCE 的影响")
    print("=" * 60)

    results = {}

    for bl_type in ['none', 'simple', 'learned']:
        bl_name = {'none': '无基线', 'simple': '滑动平均基线', 'learned': '价值网络基线'}
        print(f"\n--- 训练: {bl_name[bl_type]} ---")

        agent, rewards = train(
            env, n_episodes=n_episodes, baseline_type=bl_type
        )

        avg_r, succ = evaluate(agent, env, n_episodes=10)
        results[bl_type] = {
            'early_avg': np.mean(rewards[:100]),
            'late_avg': np.mean(rewards[-100:]),
            'eval_avg': avg_r,
            'success_rate': succ,
        }

    # ── 汇总对比 ──
    print("\n" + "=" * 60)
    print("  对比结果汇总")
    print("=" * 60)
    print(f"  {'基线类型':<16s} | {'前期奖励':>10s} | {'后期奖励':>10s} | {'评估奖励':>10s} | {'成功率':>8s}")
    print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for bl_type, name in [('none', '无基线'), ('simple', '滑动平均'), ('learned', '价值网络')]:
        r = results[bl_type]
        print(f"  {name:<16s} | {r['early_avg']:>10.1f} | {r['late_avg']:>10.1f} | "
              f"{r['eval_avg']:>10.1f} | {r['success_rate']:>7.0%}")

    print(f"\n  结论:")
    print(f"    1. 无基线: 方差大，收敛慢（奖励全是正的导致未采样动作被惩罚）")
    print(f"    2. 滑动平均基线: 比无基线好，但基线不随状态变化")
    print(f"    3. 价值网络基线: 最好，每个状态有专属基线 → Actor-Critic 雏形")
    print("=" * 60)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not HAS_TORCH:
        print("❌ 未检测到 PyTorch，请先安装：pip install torch")
        return

    print("=" * 60)
    print("  第 4 章 REINFORCE with Baseline")
    print("  ∇R̄_θ ≈ (1/N) Σ_n Σ_t (G_t - b) ∇log π_θ(a_t | s_t)")
    print("=" * 60)

    env = CliffWalkingEnv()

    # ── 默认使用学习的基线（价值网络）训练 ──
    print("\n[开始训练 —— 价值网络基线 (Actor-Critic 雏形)]")
    agent, rewards = train(
        env,
        n_episodes=1000,
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3,
        baseline_type='learned'
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

    # ── 策略可视化 ──
    policy = extract_policy(agent, env)
    env.print_policy(policy, "REINFORCE+Baseline 学到的策略")

    # ── 价值函数可视化 ──
    show_value_function(agent, env)

    # ── 各状态优势函数展示 ──
    print(f"\n[各状态的 V(s) 和最优动作 —— 抽样展示]")
    print(f"  {'位置':>8s} | {'V(s)':>8s} | {'↑(上)':>8s} {'→(右)':>8s} {'↓(下)':>8s} {'←(左)':>8s} | {'最优动作':>8s}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8} {'-'*8} {'-'*8} {'-'*8}-+-{'-'*8}")

    show_states = [
        (3, 0, "起点S"),
        (3, 1, "悬崖旁"),
        (2, 0, "起点上"),
        (2, 5, "安全区"),
        (2, 10, "终点前"),
        (1, 5, "上层中"),
    ]
    for r, c, desc in show_states:
        idx = env._coord_to_idx((r, c))
        state_tensor = agent._state_to_tensor(idx).unsqueeze(0)
        with torch.no_grad():
            probs = agent.policy_net(state_tensor).squeeze(0).numpy()
            if agent.value_net is not None:
                v = agent.value_net(state_tensor).item()
            else:
                v = 0.0
        best = np.argmax(probs)
        print(f"  ({r},{c}) {desc:>4s} | {v:>8.2f} | "
              f"{probs[0]:>8.3f} {probs[1]:>8.3f} {probs[2]:>8.3f} {probs[3]:>8.3f} | "
              f"{env.action_symbols[best]:>8s}")

    print("\n" + "=" * 60)
    print("核心要点:")
    print("  1. 基线 b 让优势 A = G_t - b 有正有负，解决奖励全正问题")
    print("  2. 简单基线 = 回报滑动平均; 学习基线 = 价值网络 V(s)")
    print("  3. 信用分配: 用 G_t（未来回报）替代整场总分 R(τ)")
    print("  4. 优势函数 A(s,a) = G_t - V(s_t) 是 Actor-Critic 的核心")
    print("  5. Actor 按优势方向更新策略; Critic 用 MSE 逼近真实 G_t")
    print("  6. 学习基线更好: 每个状态有自己的参考值，而不只是全局平均")
    print("=" * 60)

    # ── 可选: 运行对比实验 ──
    print("\n💡 提示: 取消下方注释可运行三种基线对比实验:")
    print("  # compare_baselines(env, n_episodes=800)")
    print("  对比实验需要训练 3×800=2400 回合，耗时较长。")


if __name__ == "__main__":
    main()
