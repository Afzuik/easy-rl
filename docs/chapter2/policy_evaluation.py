"""
第 2 章 马尔可夫决策过程 —— 策略评估（Policy Evaluation）演示代码

本代码实现了书中图 2.18 所示的小网格世界（small gridworld）环境，
并通过迭代贝尔曼期望方程（Eq. 2.18）进行策略评估，帮助读者直观理解：

    V^{t+1}(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_{s'} p(s'|s,a) V^t(s') ]   (2.18)

核心概念：
  - 马尔可夫决策过程 (MDP): <S, A, P, R, γ>
  - 策略 π(a|s): 给定状态下采取各动作的概率
  - 贝尔曼期望方程: 当前状态价值 = 即时奖励 + 折扣的未来状态价值期望
  - 自举 (bootstrapping): 用后继状态的价值估计来更新当前状态的价值估计

运行方式：
  pip install numpy matplotlib
  python policy_evaluation.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免弹窗问题
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import os
import sys

# 设置控制台输出编码为 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 第一部分：定义网格世界环境（对应书中图 2.18）
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
# 奖励函数: 每走一步得到 -1 的奖励（促使智能体尽快到达终止状态）
# 状态转移: 确定性的，p(s'|s,a) = 1（给定动作后下一个状态是确定的）
#           出边界的动作不改变状态（如从状态 4 往左走仍在状态 4）


class SmallGridWorld:
    """
    4×4 小网格世界环境（对应书中图 2.18）。

    状态布局:
        T   1   2   3
        4   5   6   7
        8   9  10  11
       12  13  14   T

    T 表示终止状态，1~14 为非终止状态。
    """

    def __init__(self):
        # 网格尺寸
        self.rows = 4
        self.cols = 4

        # 动作定义: 0=上, 1=右, 2=下, 3=左
        # 对应 (行偏移, 列偏移)
        self.actions = {
            0: (-1, 0),  # 上
            1: (0, 1),   # 右
            2: (1, 0),   # 下
            3: (0, -1),  # 左
        }
        self.action_names = {0: "up", 1: "right", 2: "down", 3: "left"}
        self.n_actions = 4

        # 终止状态的网格坐标
        self.terminal_states = {(0, 0), (3, 3)}

        # 构建非终止状态列表和状态索引映射
        # state_index: (row, col) -> 状态编号 (0~13，共14个非终止状态)
        self.state_coords = []   # 第 i 个元素是第 i 个非终止状态的 (row, col)
        self.coord_to_idx = {}   # (row, col) -> 状态编号
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.terminal_states:
                    self.state_coords.append((r, c))
                    self.coord_to_idx[(r, c)] = idx
                    idx += 1

        self.n_states = len(self.state_coords)  # 14 个非终止状态

        # 奖励: 每走一步 -1（非终止状态之间转移的奖励）
        self.step_reward = -1.0

    def get_next_state(self, state_idx, action):
        """
        给定当前状态编号和动作，返回下一个状态编号。
        如果下一个位置是终止状态，返回 -1。
        如果动作导致出界，留在当前状态。

        对应书中的确定性转移: p(s'|s,a) = 1
        例如 p(2|6, 上) = 1，即从状态 6 往上走一定到状态 2。
        """
        r, c = self.state_coords[state_idx]
        dr, dc = self.actions[action]
        nr, nc = r + dr, c + dc

        # 出界 → 留在当前状态
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            return state_idx

        # 到达终止状态
        if (nr, nc) in self.terminal_states:
            return -1  # -1 表示终止状态

        # 正常转移
        return self.coord_to_idx[(nr, nc)]

    def get_reward(self, state_idx, action):
        """
        奖励函数 R(s, a)。
        每走一步得到 -1 的奖励，终止状态的奖励为 0。
        """
        return self.step_reward

    def print_grid(self, values=None, title=""):
        """打印网格世界的状态编号或价值。"""
        print(f"\n{title}")
        print("=" * 40)
        grid = np.zeros((self.rows, self.cols))
        if values is not None:
            for i, (r, c) in enumerate(self.state_coords):
                grid[r, c] = values[i]
        else:
            for i, (r, c) in enumerate(self.state_coords):
                grid[r, c] = i + 1  # 状态编号从 1 开始

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r, c) in self.terminal_states:
                    row_str += f"{'T':>8s}"
                elif values is not None:
                    row_str += f"{grid[r, c]:>8.2f}"
                else:
                    row_str += f"{int(grid[r, c]):>8d}"
            print(row_str)
        print("=" * 40)


# ============================================================
# 第二部分：策略定义
# ============================================================
#
# 策略 π(a|s) = p(a_t = a | s_t = s)
# 定义在某个状态下采取各动作的概率。
#
# 书中例子:
#   - 均匀随机策略: π(↑|s) = π(→|s) = π(↓|s) = π(←|s) = 0.25
#   - 确定性策略:   π(s) = 左（总是往左走）


def uniform_random_policy(n_actions):
    """
    均匀随机策略（uniform random policy）。
    不管在哪个状态，上、下、左、右的概率均为 0.25。

    对应书中: π(↑|.) = π(→|.) = π(↓|.) = π(←|.) = 0.25

    返回: shape=(n_actions,) 的概率向量
    """
    return np.ones(n_actions) / n_actions


def left_only_policy(n_actions):
    """
    确定性策略: 总是往左走。
    对应书中: π(s) = 左

    返回: shape=(n_actions,) 的概率向量
    """
    pi = np.zeros(n_actions)
    pi[3] = 1.0  # 3 = 左
    return pi


# ============================================================
# 第三部分：策略评估 —— 迭代贝尔曼期望方程
# ============================================================
#
# 核心公式 (Eq. 2.18):
#
#   V^{t+1}(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_{s'} p(s'|s,a) V^t(s') ]
#
# 迭代过程:
#   1. 初始化 V_0(s) = 0, 对所有 s
#   2. 对每一轮迭代 t = 0, 1, 2, ...
#      对每个状态 s，用上一轮的 V^t(s') 计算新的 V^{t+1}(s)
#   3. 当 max_s |V^{t+1}(s) - V^t(s)| < θ 时停止（收敛）
#
# 这就是书中图 2.7 所示的动态规划算法:
#   - 用 V'（上一轮的价值）来计算 V（本轮的价值）
#   - 基于后继状态价值的估计来更新当前状态价值的估计 → 自举 (bootstrapping)


def policy_evaluation(env, policy_func, gamma=1.0, theta=1e-6, max_iter=1000, verbose=True):
    """
    策略评估算法 —— 迭代贝尔曼期望方程。

    参数:
        env:         SmallGridWorld 环境
        policy_func: 策略函数，policy_func(n_actions) 返回概率向量 π(a|s)
        gamma:       折扣因子 γ ∈ [0, 1]
        theta:       收敛阈值 θ，当最大价值变化量 < θ 时停止
        max_iter:    最大迭代次数
        verbose:     是否打印迭代信息

    返回:
        V:              收敛后的状态价值函数，shape=(n_states,)
        history:        每轮迭代的价值函数历史记录，用于可视化
        converged_iter: 收敛时的迭代次数
    """
    n_states = env.n_states
    n_actions = env.n_actions

    # 初始化: V_0(s) = 0，对所有非终止状态
    V = np.zeros(n_states)
    history = [V.copy()]

    if verbose:
        print("\n" + "=" * 60)
        print(f"策略评估开始")
        print(f"折扣因子 γ = {gamma}")
        print(f"收敛阈值 θ = {theta}")
        print("=" * 60)
        print(f"\n{'迭代次数':>8s} | {'最大变化量 Δ':>14s}")
        print("-" * 30)

    for k in range(1, max_iter + 1):
        V_new = np.zeros(n_states)
        pi = policy_func(n_actions)  # 当前策略 π(a|s)

        # ──────────────────────────────────────────────────
        # 对每个状态 s，应用贝尔曼期望方程 (Eq. 2.18):
        #
        #   V^{t+1}(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_{s'} p(s'|s,a) V^t(s') ]
        #                  ↑            ↑              ↑
        #               策略概率     即时奖励      折扣的未来状态价值
        #
        # 由于本环境中 p(s'|s,a) = 1（确定性转移），内层求和退化为单一项。
        # ──────────────────────────────────────────────────
        for s in range(n_states):
            expected_value = 0.0

            for a in range(n_actions):
                # R(s, a): 即时奖励
                reward = env.get_reward(s, a)

                # s' = 下一个状态
                next_s = env.get_next_state(s, a)

                # V^t(s'): 后继状态的价值估计
                if next_s == -1:
                    # 到达终止状态，V(终止状态) = 0
                    v_next = 0.0
                else:
                    v_next = V[next_s]  # 使用上一轮的 V^t

                # π(a|s) × [R(s,a) + γ × V^t(s')]
                expected_value += pi[a] * (reward + gamma * v_next)

            V_new[s] = expected_value

        # 计算最大变化量: max_s |V^{t+1}(s) - V^t(s)|
        delta = np.max(np.abs(V_new - V))

        # 更新价值函数
        V = V_new
        history.append(V.copy())

        if verbose and (k <= 10 or k % 50 == 0 or delta < theta):
            print(f"{k:>8d} | {delta:>14.6f}")

        # 收敛判断
        if delta < theta:
            if verbose:
                print("-" * 30)
                print(f"在第 {k} 轮迭代后收敛！ (Δ = {delta:.2e} < θ = {theta})")
            break

    return V, history, k


# ============================================================
# 第四部分：解析解求解（作为对比参考）
# ============================================================
#
# 对于 MRP，贝尔曼方程可以写成矩阵形式:
#   V = R + γ P V
#   (I - γP) V = R
#   V = (I - γP)^{-1} R
#
# 对于 MDP 给定策略 π 后，可以化归为 MRP:
#   P_π(s'|s) = Σ_a π(a|s) p(s'|s,a)
#   r_π(s)    = Σ_a π(a|s) R(s,a)
#
# 然后直接用解析解求解（复杂度 O(N^3)，仅适用于小规模问题）。


def solve_analytic(env, policy_func, gamma):
    """
    用解析解 V = (I - γP_π)^{-1} r_π 直接求解。

    参数:
        env:         SmallGridWorld 环境
        policy_func: 策略函数
        gamma:       折扣因子 γ

    返回:
        V: 状态价值函数
    """
    n_states = env.n_states
    n_actions = env.n_actions
    pi = policy_func(n_actions)

    # 构建 P_π 和 r_π
    # P_π[i, j] = Σ_a π(a|s_i) p(s_j | s_i, a)
    # r_π[i]    = Σ_a π(a|s_i) R(s_i, a)
    P_pi = np.zeros((n_states, n_states))
    r_pi = np.zeros(n_states)

    for s in range(n_states):
        for a in range(n_actions):
            reward = env.get_reward(s, a)
            r_pi[s] += pi[a] * reward

            next_s = env.get_next_state(s, a)
            if next_s != -1:  # 非终止状态
                P_pi[s, next_s] += pi[a]
            # 终止状态对应 V=0，不需要加入 P_pi

    # 解析解: V = (I - γP_π)^{-1} r_π
    I = np.eye(n_states)
    V = np.linalg.solve(I - gamma * P_pi, r_pi)

    return V


# ============================================================
# 第五部分：可视化
# ============================================================


def plot_evaluation_history(env, history, gamma, title_suffix=""):
    """
    绘制策略评估过程中价值函数的变化（对应书中图 2.19 ~ 图 2.20）。

    展示从初始化为 0 到逐步收敛的过程，可视化价值的"扩散"效应：
    奖励信号从有奖励的状态逐步传播到远处。
    """
    # 选择要展示的迭代轮次
    total_iters = len(history) - 1
    if total_iters <= 6:
        show_iters = list(range(len(history)))
    else:
        # 选取代表性的迭代轮次
        indices = [0, 1, 2, min(5, total_iters), min(10, total_iters),
                   min(50, total_iters), min(total_iters // 2, total_iters),
                   total_iters]
        show_iters = sorted(set(indices))

    n_plots = len(show_iters)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # 找到所有价值的范围，用于统一色彩映射
    all_values = np.concatenate([h for h in history])
    vmin, vmax = np.min(all_values), np.max(all_values)
    norm = Normalize(vmin=vmin, vmax=vmax) if vmin != vmax else Normalize()
    cmap = plt.get_cmap('RdYlGn')

    for plot_idx, iter_idx in enumerate(show_iters):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]

        V = history[iter_idx]

        # 构建 4×4 网格
        grid = np.full((env.rows, env.cols), np.nan)
        for i, (r, c) in enumerate(env.state_coords):
            grid[r, c] = V[i]
        # 终止状态设为 0
        for tr, tc in env.terminal_states:
            grid[tr, tc] = 0.0

        im = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f"Iter {iter_idx}", fontsize=11)

        # 在每个格子中显示数值
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) in env.terminal_states:
                    ax.text(c, r, "T", ha='center', va='center',
                            fontsize=12, fontweight='bold', color='gray')
                else:
                    val = grid[r, c]
                    color = 'white' if abs(val) > (vmax - vmin) * 0.6 + vmin else 'black'
                    ax.text(c, r, f"{val:.1f}", ha='center', va='center',
                            fontsize=9, color=color)

        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏多余的子图
    for plot_idx in range(n_plots, n_rows * n_cols):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        axes[row, col].set_visible(False)

    plt.colorbar(im, ax=axes, shrink=0.8, label="State Value V(s)")
    fig.suptitle(f"Policy Evaluation: Iterative Bellman Expectation (γ={gamma}){title_suffix}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_convergence_curve(history, theta, gamma):
    """绘制收敛曲线：每轮迭代的最大价值变化量。"""
    deltas = []
    for t in range(1, len(history)):
        delta = np.max(np.abs(history[t] - history[t - 1]))
        deltas.append(delta)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(deltas) + 1), deltas, 'b-o', markersize=3, linewidth=1.5)
    ax.axhline(y=theta, color='r', linestyle='--', label=f'θ = {theta}')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Max |ΔV(s)|", fontsize=12)
    ax.set_title(f"Convergence of Policy Evaluation (γ={gamma})", fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def compare_gammas(env, policy_func, gammas, theta=1e-6):
    """
    比较不同折扣因子 γ 下的策略评估结果。

    书中提到：
    - γ = 0: 只关注即时奖励
    - γ = 1: 未来奖励与即时奖励等价
    - 不同的 γ 会产生不同的价值函数
    """
    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 4))
    if len(gammas) == 1:
        axes = [axes]

    all_values = []
    for gamma in gammas:
        V, _, _ = policy_evaluation(env, policy_func, gamma=gamma, theta=theta, verbose=False)
        all_values.append(V)

    # 统一色彩映射范围
    all_v = np.concatenate(all_values)
    vmin, vmax = np.min(all_v), np.max(all_v)
    norm = Normalize(vmin=vmin, vmax=vmax) if vmin != vmax else Normalize()
    cmap = plt.get_cmap('RdYlGn')

    for idx, (gamma, V) in enumerate(zip(gammas, all_values)):
        ax = axes[idx]
        grid = np.full((env.rows, env.cols), np.nan)
        for i, (r, c) in enumerate(env.state_coords):
            grid[r, c] = V[i]
        for tr, tc in env.terminal_states:
            grid[tr, tc] = 0.0

        im = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f"γ = {gamma}", fontsize=13, fontweight='bold')

        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) in env.terminal_states:
                    ax.text(c, r, "T", ha='center', va='center',
                            fontsize=12, fontweight='bold', color='gray')
                else:
                    val = grid[r, c]
                    color = 'white' if abs(val) > (vmax - vmin) * 0.6 + vmin else 'black'
                    ax.text(c, r, f"{val:.1f}", ha='center', va='center',
                            fontsize=10, color=color)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.colorbar(im, ax=axes, shrink=0.8, label="V(s)")
    fig.suptitle("Policy Evaluation under Different Discount Factors γ",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# 第六部分：主程序 —— 运行实验
# ============================================================


def main():
    print("=" * 60)
    print("  第 2 章 马尔可夫决策过程 -- 策略评估演示程序")
    print("  Policy Evaluation via Iterative Bellman Equation")
    print("=" * 60)

    # ── 创建环境 ──
    env = SmallGridWorld()

    print("\n[环境] 4×4 小网格世界 (图 2.18)")
    env.print_grid(title="状态编号 (T = 终止状态)")

    # ══════════════════════════════════════════════════════════
    # 实验 1: 均匀随机策略 + γ=1.0（对应书中图 2.18 的例子）
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("实验 1: 均匀随机策略, gamma = 1.0")
    print("策略: pi(up|s) = pi(right|s) = pi(down|s) = pi(left|s) = 0.25")
    print("-" * 60)

    V1, history1, iter1 = policy_evaluation(
        env, uniform_random_policy, gamma=1.0, theta=1e-6, verbose=True
    )

    env.print_grid(V1, title=f"收敛后的状态价值 V(s) (迭代 {iter1} 轮)")

    # 解析解对比
    V_analytic = solve_analytic(env, uniform_random_policy, gamma=0.99)
    print(f"\n[对比] 解析解 V(s) (γ=0.99):")
    env.print_grid(V_analytic, title="解析解 V = (I - gamma*P)^(-1) * r")

    # ══════════════════════════════════════════════════════════
    # 实验 2: 确定性策略（总是往左走）+ γ=0（对应书中图 2.15 的例子）
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("实验 2: 确定性策略 (总是往左走), gamma = 0")
    print("策略: pi(s) = left")
    print("-" * 60)

    V2, history2, iter2 = policy_evaluation(
        env, left_only_policy, gamma=0.0, theta=1e-6, verbose=True
    )

    env.print_grid(V2, title=f"收敛后的状态价值 V(s) (迭代 {iter2} 轮)")
    print("\n说明: γ=0 时只关注即时奖励，每步奖励为 -1，所以 V(s) = -1")

    # ══════════════════════════════════════════════════════════
    # 实验 3: 不同折扣因子 γ 的比较
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("实验 3: 不同折扣因子 gamma 的比较 (均匀随机策略)")
    print("-" * 60)

    gammas = [0.0, 0.5, 0.9, 1.0]
    for g in gammas:
        V_g, _, iters_g = policy_evaluation(
            env, uniform_random_policy, gamma=g, theta=1e-6, verbose=False
        )
        print(f"\nγ = {g} (迭代 {iters_g} 轮收敛):")
        env.print_grid(V_g, title=f"V(s) with γ={g}")

    # ══════════════════════════════════════════════════════════
    # 可视化
    # ══════════════════════════════════════════════════════════

    # 确保输出目录存在
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 图1: 策略评估迭代过程可视化
    fig1 = plot_evaluation_history(env, history1, gamma=1.0)
    path1 = os.path.join(output_dir, "policy_evaluation_history.png")
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"\n[已保存] 策略评估迭代过程图 → {path1}")

    # 图2: 收敛曲线
    fig2 = plot_convergence_curve(history1, theta=1e-6, gamma=1.0)
    path2 = os.path.join(output_dir, "convergence_curve.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"[已保存] 收敛曲线图 → {path2}")

    # 图3: 不同 γ 的比较
    fig3 = compare_gammas(env, uniform_random_policy, gammas=[0.0, 0.5, 0.9, 1.0])
    path3 = os.path.join(output_dir, "gamma_comparison.png")
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"[已保存] 不同γ比较图 → {path3}")

    print("\n" + "=" * 60)
    print("所有实验完成! 图片已保存到 output/ 目录。")
    print("=" * 60)

    plt.close('all')


if __name__ == "__main__":
    main()
