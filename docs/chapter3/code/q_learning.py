"""
第 3 章 Q-learning —— 异策略时序差分控制（Off-Policy TD Control）

核心思想:
    每走一步用 (s,a,r,s') 四元组更新 Q 表。行为策略（ε-贪心）和目标策略（纯贪心）
    分离：用 ε-贪心探索，用 max Q 学习最优。

更新公式:
    Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_{t+1} + γ max_a Q(s_{t+1}, a) - Q(s_t, a_t) ]
                                                   ↑
                                    取 s_{t+1} 处所有动作的 Q 值最大值
                                    （不关心实际执行了哪个动作）

对比 Sarsa:
    Sarsa:      目标 = r + γ Q(s', a')     ← a' 是实际采样的下一步动作
    Q-learning: 目标 = r + γ max_a Q(s',a)  ← 假设下一步选最优动作

行为特征:
    Q-learning 更新时不考虑探索噪声，因此更大胆，在悬崖行走中会**贴着悬崖走最短路径**。
    激进、高效。

运行方式:
    python q_learning.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from cliff_walking_env import CliffWalkingEnv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def epsilon_greedy_action(Q, state, epsilon, n_actions):
    """ε-贪心选择动作（行为策略用）。"""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])


def q_learning(env, n_episodes=500, alpha=0.5, gamma=1.0,
               epsilon_start=0.5, epsilon_end=0.01, decay_episodes=300):
    """
    Q-learning 异策略 TD 控制。

    每走一步:
        1. 用 ε-贪心（行为策略）选动作 a_t
        2. 执行 a_t，得到 r_{t+1}, s_{t+1}
        3. Q(s_t,a_t) ← Q(s_t,a_t) + α [ r_{t+1} + γ max_a Q(s_{t+1},a) - Q(s_t,a_t) ]
           ↑ 注意: 这里用 max_a，而不是实际采样的下一步动作

    返回:
        Q:         Q 表 (n_states × n_actions)
        policy:    最终贪心策略 (纯贪心提取)
        rewards:   每个回合的总奖励
    """
    n_states = env.n_states
    n_actions = env.n_actions
    Q = np.zeros((n_states, n_actions))
    rewards_history = []

    for ep in range(n_episodes):
        # ── ε 线性衰减 ──
        if ep < decay_episodes:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / decay_episodes)
        else:
            epsilon = epsilon_end

        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < 500:
            # ── 行为策略: ε-贪心选动作（探索用） ──
            action = epsilon_greedy_action(Q, state, epsilon, n_actions)

            # ── 执行动作 ──
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # ── Q-learning 更新（关键区别） ──
            # 目标策略: 纯贪心 —— 取 max_a Q(s', a)
            best_next = np.max(Q[next_state]) if not done else 0.0
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            step += 1

        rewards_history.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            print(f"  回合 {ep+1:>4d}/{n_episodes} | ε={epsilon:.3f} | "
                  f"平均奖励: {avg_r:.1f}")

    policy = np.argmax(Q, axis=1)
    return Q, policy, rewards_history


def main():
    print("=" * 60)
    print("  第 3 章 Q-learning —— 异策略 TD 控制")
    print("  Q(s,a) ← Q(s,a) + α [ r + γ max_a Q(s',a) - Q(s,a) ]")
    print("=" * 60)

    env = CliffWalkingEnv()

    Q, policy, rewards = q_learning(
        env, n_episodes=500, alpha=0.5, gamma=1.0,
        epsilon_start=0.5, epsilon_end=0.01, decay_episodes=300
    )

    # ── 结果展示 ──
    env.print_policy(policy, "Q-learning 学到的策略 (激进，贴悬崖)")

    print(f"\n[学习曲线]")
    print(f"  前100回合平均奖励: {np.mean(rewards[:100]):.1f}")
    print(f"  后100回合平均奖励: {np.mean(rewards[-100:]):.1f}")

    # ── 测试 ──
    print(f"\n[贪心策略测试 (ε=0)]:")
    test_rewards = []
    for _ in range(10):
        state = env.reset()
        done = False
        ep_r = 0
        while not done:
            action = policy[state]
            state, reward, done = env.step(state, action)
            ep_r += reward
        test_rewards.append(ep_r)
    print(f"  10次测试平均奖励: {np.mean(test_rewards):.1f}")
    print(f"  (Q-learning 的路径是贴着悬崖的最短路，约 -13)")
    print(f"  (但训练中 ε-贪心偶尔掉悬崖导致低奖励)")

    print("\n" + "=" * 60)
    print("核心要点:")
    print("  1. Q-learning 是异策略 (off-policy): 探索学最优")
    print("  2. 更新使用 max_a Q(s',a) → 不考虑探索影响")
    print("  3. 行为激进: 在悬崖行走中贴着悬崖走最短路径")
    print("  4. 只需要 (s,a,r,s') 四元组, 不需要 a'")
    print("  5. 可以重用旧经验 → DQN 中经验回放的基础")
    print("=" * 60)


if __name__ == "__main__":
    main()
