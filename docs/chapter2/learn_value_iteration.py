import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 第一部分：网格世界环境（与 policy_evaluation.py 共用同一环境）
# ============================================================
#
# 4×4 网格世界，状态编号如下：
#
#   T   1   2   3       T = 终止状态 (terminal)
#   4   5   6   7       非终止状态: 1~14
#   8   9  10  11
#  12  13  14   T
#
# 动作空间: 0=上, 1=右, 2=下, 3=左
# 奖励函数: 每走一步 -1（催促智能体尽快到达终止态）
# 状态转移: 确定性 p(s'|s,a) = 1；出界则留在原地

class SmallGridWorld:
    def __init__(self):
        self.rows = 4
        self.cols = 4

        self.actions = {
            0:(-1,0),
            1:(0,1),
            2:(1,0),
            3:(0,-1)
        }
        self.action_names = {
            0:"up",
            1:"right",
            2:"down",
            3:"left"
        }
        self.action_symbols = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        self.n_actions = 4
        self.terminal_states = {
            (0,0),
            (3,3)
        }
        self.state_coords = []
        self.coord_to_idx = {}
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in self.terminal_states:
                    self.state_coords.append((r,c))
                    self.coord_to_idx[(r,c)] = idx
                    idx += 1

        self.n_states = len(self.state_coords)
        self.step_reward = -1.0

    def get_next_state(self,state_idx,action):
        r,c = self.state_coords[state_idx]
        dr,dc = self.actions[action]
        nr,nc = r+dr,c+dc

        if nr<0 or nr>=self.rows or nc<0 or nc>=self.cols:
            return state_idx
        if (nr,nc) in self.terminal_states:
            return -1
        return self.coord_to_idx[(nr,nc)]
    
    def get_reward(self,state_idx,action):
        return self.step_reward
    
    def print_grid(self,values=None,title="",fmt=".2f"):
        print(f"\n{title}")
        print("="*40)
        # 构建网格
        grid = np.full((self.rows,self.cols),np.nan)
        # 区分两种显示模式
        if values is not None:
            for i,(r,c) in enumerate(self.state_coords):
                grid[r,c] = values[i]
        else:
            for i,(r,c) in enumerate(self.state_coords):
                grid[r,c] = i + 1
        # 格式化输出
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r,c) in self.terminal_states:
                    row_str += f"{'T':>8s}"
                elif values is not None:
                    row_str += f"{grid[r,c]:>8{fmt}}"
                else:
                    row_str += f"{int(grid[r,c]):>8d}"
            print(row_str)
        print("="*40)
    
    def print_policy(self,policy,title=""):
        print(f"\n{title}")
        print("="*40)
        grid = np.full((self.rows,self.cols),"",dtype=object)
        for i,(r,c) in enumerate(self.state_coords):
            grid[r,c] = self.action_symbols[policy[i]]

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r,c) in self.terminal_states:
                    row_str += f"{'T':>8s}"
                else:
                    row_str += f"{grid[r,c]:>8s}"
            print(row_str)
        print("="*40)

# ============================================================
# 第二部分：价值迭代算法 —— 迭代贝尔曼最优方程
# ============================================================
#
# 核心公式 (Eq. 2.22):
#
#   V^{k+1}(s) = max_a [ R(s,a) + γ Σ_{s'} p(s'|s,a) V^{k}(s') ]
#
# 与策略评估的对比:
#   - 策略评估 (2.18): 使用 Σ_a π(a|s) […]  —— 对动作取加权平均
#   - 价值迭代 (2.22): 使用 max_a […]        —— 直接取最优动作
#
# 迭代停止后，提取最优策略:
#   π*(s) = argmax_a [ R(s,a) + γ Σ_{s'} p(s'|s,a) V*(s') ]
#
# 直观理解（最优性原理）:
#   如果所有后继状态 s' 都已经达到最优价值 V*(s')，
#   那么对当前状态 s 只需选一个动作使 "即时奖励 + 折扣 V*(s')" 最大即可。

def value_iteration(env,gamma=1.0,theta=1e-6,max_iter=1000,verbose=True):
        
    """
    价值迭代算法 —— 迭代贝尔曼最优方程。

    参数:
        env:      SmallGridWorld 环境
        gamma:    折扣因子 γ ∈ [0, 1]
        theta:    收敛阈值（最大价值变化 < θ 时停止）
        max_iter: 最大迭代次数
        verbose:  是否打印迭代信息

    返回:
        V:              收敛后的最优状态价值 V*
        policy:         提取的最优策略 π* (shape=(n_states,)，每个值为动作编号)
        Q:              收敛后的最优 Q*(s,a) (shape=(n_states, n_actions))
        history:        每轮迭代的 V 历史
        converged_iter: 收敛轮次
    """
    n_states = env.n_states
    n_actions = env.n_actions

    V = np.zeros(n_states)
    history = [V.copy()]

    if verbose:
        print("\n" + "=" * 60)
        print(f"价值迭代开始")
        print(f"折扣因子 γ = {gamma}")
        print(f"收敛阈值 θ = {theta}")
        print("=" * 60)
        print(f"\n{'迭代次数':>8s} | {'最大变化量 Δ':>14s}")
        print("-" * 30)
    
    for k in range(1,max_iter+1):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                reward = env.get_reward(s,a)
                next_s = env.get_next_state(s,a)
                if next_s == -1:
                    v_next = 0.0
                else:
                    v_next = V[next_s]
                q_values[a] = reward + gamma * v_next
            V_new[s] = np.max(q_values)
        delta = np.max(np.abs(V_new-V))

        V = V_new
        history.append(V.copy())
        if verbose and (k<=10 or k % 50 == 0 or delta<theta):
            print(f"{k:>8d} | {delta:>14.6f}")
        
        if delta < theta:
            if verbose:
                print("-" * 30)
                print(f"在第 {k} 轮迭代后收敛！ (Δ = {delta:.2e} < θ = {theta})")
            break
    
    # ── 提取最优策略 ──
    # π*(s) = argmax_a [ R(s,a) + γ Σ V*(s') ]
    policy = np.zeros(n_states, dtype=int)
    Q = np.zeros((n_states, n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            reward = env.get_reward(s, a)
            next_s = env.get_next_state(s, a)
            if next_s == -1:
                v_next = 0.0
            else:
                v_next = V[next_s]
            Q[s, a] = reward + gamma * v_next
        policy[s] = np.argmax(Q[s])

    return V, policy, Q, history, k


def main():
    print("=" * 60)
    print("  第 2 章 马尔可夫决策过程 -- 价值迭代演示程序")
    print("  Value Iteration via Bellman Optimality Equation")
    print("=" * 60)

    env = SmallGridWorld()
    print("\n[环境] 4×4 小网格世界 (图 2.18)")
    env.print_grid(title="状态编号 (T = 终止状态)")

    # ══════════════════════════════════════════════════════════
    # 实验 1: 价值迭代, γ = 1.0
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("实验 1: 价值迭代, gamma = 1.0")
    print("贝尔曼最优方程: V^{k+1}(s) = max_a [ R(s,a) + γ Σ P V^{k}(s') ]")
    print("-" * 60)

    V_star, policy, Q, history, iters = value_iteration(
        env, gamma=1.0, theta=1e-6, verbose=True
    )

    env.print_grid(V_star, title=f"最优价值 V*(s) (γ=1.0, {iters} 轮收敛)")
    env.print_policy(policy, title=f"最优策略 π*(s) (γ=1.0)")

    # 打印 Q 表格
    print(f"\n[最优 Q*(s,a) 表格]:")
    header = f"{'状态':>6s}" + "".join(
        f"{env.action_names[a]:>10s}" for a in range(env.n_actions))
    print(header)
    print("-" * (6 + 10 * env.n_actions))
    for s in range(env.n_states):
        row = f"{s+1:>6d}"
        for a in range(env.n_actions):
            row += f"{Q[s, a]:>10.3f}"
        print(row)

    # 验证最优策略
    print("\n[验证] 最优策略下各状态采取的动作:")
    for s in range(env.n_states):
        r, c = env.state_coords[s]
        a = policy[s]
        print(f"  状态 {s+1} ({r},{c}): {env.action_symbols[a]} → {env.action_names[a]}")

if __name__ == "__main__":
    main()