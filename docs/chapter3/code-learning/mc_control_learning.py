"""
第 3 章 蒙特卡洛控制（MC Control with ε-greedy）

核心思想:
    在 GPI 框架下，用蒙特卡洛方法估计 Q(s,a)，再用 ε-贪心改进策略。

步骤:
    1. 初始化 Q(s,a) = 0, N(s,a) = 0
    2. 对每个回合:
       a) 用 ε-贪心策略采样一条完整轨迹
       b) 从后往前计算每个 (s,a) 的 G_t
       c) 增量更新: Q(s,a) ← Q(s,a) + (1/N)(G_t - Q(s,a))
       d) ε 逐渐减小（探索→利用）

局限: 必须等到回合结束才能更新（同 MC prediction）。

运行方式:
    python mc_control.py
"""
import numpy as np
import sys
import os
sys.path.insert(0,os.path.dirname(__file__))
from cliff_walking_env_learning import CliffWalkingEnv

if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8")

def epsilon_greedy_action(Q,state,epsilon,n_actions):
    """_summary_
    ε-贪心选择动作
    Args:
        Q (_type_): _description_
        state (_type_): _description_
        epsilon (_type_): _description_
        n_actions (_type_): _description_
    """
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        np.argmax(Q[state])

def mc_control(env,n_episodes=500,gamma=1.0,epsilon_start=0.5,epsilon_end=0.01,decay_episodes=300):
    """_summary_
    蒙特卡洛控制（ε-贪心）
    Args:
        env (_type_): _description_
        n_episode (int, optional): _description_. Defaults to 500.
        gamma (float, optional): _description_. Defaults to 1.0.
        epsilon_start (float, optional): _description_. Defaults to 0.5.
        epsilon_end (float, optional): _description_. Defaults to 0.01.
        decay_episodes (int, optional): _description_. Defaults to 300.
    返回：
        Q：Q表（n_states x n_actions）
        policy：最终贪婪策略
        reward_history：每个回合的总奖励
    """
    n_states = env.n_states
    n_actions = env.n_actions
    Q = np.zeros((n_states,n_actions))
    N = np.zeros((n_states,n_actions))
    rewards_history = []

    for ep in range(n_episodes):
        # -- ε 线性衰减 --
        if ep < decay_episodes:
            epsilon = epsilon_start+(epsilon_end-epsilon_start) * (ep/decay_episodes)
        else:
            epsilon = epsilon_end
    
        # -- 采样一条轨迹 --
        state = env.reset()
        episode = []
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_action(Q,state,epsilon,n_actions)
            next_state,reward,done = env.step(state,action)
            episode.append((state,action,reward))
            total_reward += reward
            state = next_state
        
        rewards_history.append(total_reward)

        # -- MC 增量更新：从后往前 --
        G = 0.0
        visited = set()
        for state,action,reward in reversed(episode):
            G = reward + gamma * G
            if (state,action) not in visited:
                visited.add((state,action))
                N[state,action] += 1
                Q[state,action] += (1.0/N[state,action]) * (G-Q[state,action])
        
        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"  回合 {ep+1:>4d}/{n_episodes} | ε={epsilon:.3f} | "
                  f"最近100回合平均奖励: {avg_reward:.1f}")

    # ── 提取最终贪心策略 ──
    policy = np.argmax(Q, axis=1)
    return Q, policy, rewards_history