"""
第 3 章 Sarsa —— 同策略时序差分控制（On-Policy TD Control）

核心思想:
    每走一步就用 (s,a,r,s',a') 五元组更新一次 Q 表。行为策略和学习策略是同一个。

更新公式:
    Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_{t+1} + γ Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) ]
                                                   ↑
                                    a_{t+1} 是实际执行的下一步动作 (ε-贪心采样)

行为特征:
    Sarsa 知道自己下一步可能随机探索，因此在悬崖行走中会**主动远离悬崖**，
    走一条绕远但安全的路径。保守稳健。

名称来源:
    State → Action → Reward → State' → Action'

运行方式:
    python sarsa.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from cliff_walking_env import CliffWalkingEnv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def epsilon_greedy_action(Q, state, epsilon, n_actions):
    """ε-贪心选择动作。"""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])


def sarsa(env, n_episodes=500, alpha=0.5, gamma=1.0,
          epsilon_start=0.5, epsilon_end=0.01, decay_episodes=300):
    """
    Sarsa 同策略 TD 控制。

    每走一步:
        1. 用 ε-贪心在当前状态选动作 a_t
        2. 执行 a_t，得到 r_{t+1}, s_{t+1}
        3. 用 ε-贪心在 s_{t+1} 选下一步动作 a_{t+1}
        4. Q(s_t,a_t) ← Q(s_t,a_t) + α [ r_{t+1} + γ Q(s_{t+1},a_{t+1}) - Q(s_t,a_t) ]
        5. s_t ← s_{t+1}, a_t ← a_{t+1}

    返回:
        Q:         Q 表 (n_states × n_actions)
        policy:    最终贪心策略
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
        action = epsilon_greedy_action(Q, state, epsilon, n_actions)
        done = False
        total_reward = 0
        step = 0

        while not done and step < 500:
            # ── 执行动作 ──
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # ── 选下一步动作（关键：Sarsa 用实际执行的动作） ──
            next_action = epsilon_greedy_action(Q, next_state, epsilon, n_actions)

            # ── Sarsa 更新 ──
            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            action = next_action
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
    print("  第 3 章 Sarsa —— 同策略 TD 控制")
    print("  Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]")
    print("=" * 60)

    env = CliffWalkingEnv()

    Q, policy, rewards = sarsa(
        env, n_episodes=500, alpha=0.5, gamma=1.0,
        epsilon_start=0.5, epsilon_end=0.01, decay_episodes=300
    )

    # ── 结果展示 ──
    env.print_policy(policy, "Sarsa 学到的策略 (保守，远离悬崖)")

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
    print(f"  (Sarsa 的路径通常绕行安全区，步数 > 13)")

    print("\n" + "=" * 60)
    print("核心要点:")
    print("  1. Sarsa 是同策略 (on-policy): 学习什么就做什么")
    print("  2. 更新使用实际执行的 a_{t+1} → 考虑探索的影响")
    print("  3. 行为保守: 在悬崖行走中远离悬崖")
    print("  4. n 步 Sarsa / Sarsa(λ): 介于单步和 MC 之间")
    print("=" * 60)


if __name__ == "__main__":
    main()
