"""
第 2 章 马尔可夫决策过程 —— 价值迭代（Value Iteration）演示代码

本代码实现了书中图 2.24~2.28 所示的 4×4 小网格世界环境，
并通过迭代贝尔曼最优方程（Eq. 2.22）进行价值迭代，帮助读者直观理解：

    V^{k+1}(s) = max_a [ R(s,a) + γ Σ_{s'} p(s'|s,a) V^{k}(s') ]   (2.22)

核心概念：
  - 贝尔曼最优方程: 最优状态价值 = max_a ( 即时奖励 + 折扣的最优后继价值 )
  - 价值迭代: 从任意初始值出发，反复应用贝尔曼最优备份，逐步收敛到 V*
  - 最优性原理: 当前状态最优 ⟺ 所有后继状态都已经最优
  - 策略提取: 收敛后 π*(s) = argmax_a [ R(s,a) + γ Σ P V*(s') ]
  - 对比: 策略迭代（评估 + 改进交替）vs 价值迭代（直接迭代最优方程）

运行方式：
  pip install numpy matplotlib
  python value_iteration.py
"""

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
        self.rows = 4
        self.cols = 4

        # 动作: 0=上, 1=右, 2=下, 3=左
        self.actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.action_names = {0: "up", 1: "right", 2: "down", 3: "left"}
        self.action_symbols = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        self.n_actions = 4

        # 终止状态
        self.terminal_states = {(0, 0), (3, 3)}

        # 构建 14 个非终止状态
        self.state_coords = []
        self.coord_to_idx = {}
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.terminal_states:
                    self.state_coords.append((r, c))
                    self.coord_to_idx[(r, c)] = idx
                    idx += 1
        self.n_states = len(self.state_coords)  # 14

        self.step_reward = -1.0  # 每走一步扣 1 分

    def get_next_state(self, state_idx, action):
        """确定性转移: p(s'|s,a) = 1。出界留原地，终止返回 -1。"""
        r, c = self.state_coords[state_idx]
        dr, dc = self.actions[action]
        nr, nc = r + dr, c + dc

        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            return state_idx                      # 出界 → 留原地
        if (nr, nc) in self.terminal_states:
            return -1                              # 到达终止状态
        return self.coord_to_idx[(nr, nc)]

    def get_reward(self, state_idx, action):
        """R(s,a) = -1，每走一步的惩罚。"""
        return self.step_reward

    def print_grid(self, values=None, title="", fmt=".2f"):
        """打印网格：状态编号或价值。"""
        print(f"\n{title}")
        print("=" * 40)
        grid = np.full((self.rows, self.cols), np.nan)
        if values is not None:
            for i, (r, c) in enumerate(self.state_coords):
                grid[r, c] = values[i]
        else:
            for i, (r, c) in enumerate(self.state_coords):
                grid[r, c] = i + 1

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r, c) in self.terminal_states:
                    row_str += f"{'T':>8s}"
                elif values is not None:
                    row_str += f"{grid[r, c]:>8{fmt}}"
                else:
                    row_str += f"{int(grid[r, c]):>8d}"
            print(row_str)
        print("=" * 40)

    def print_policy(self, policy, title=""):
        """打印策略：用箭头显示每个状态的动作。"""
        print(f"\n{title}")
        print("=" * 40)
        grid = np.full((self.rows, self.cols), "", dtype=object)
        for i, (r, c) in enumerate(self.state_coords):
            grid[r, c] = self.action_symbols[policy[i]]

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r, c) in self.terminal_states:
                    row_str += f"{'T':>8s}"
                else:
                    row_str += f"{grid[r, c]:>8s}"
            print(row_str)
        print("=" * 40)


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


def value_iteration(env, gamma=1.0, theta=1e-6, max_iter=1000, verbose=True):
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

    # 初始化: V_0(s) = 0，对所有非终止状态
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

    for k in range(1, max_iter + 1):
        V_new = np.zeros(n_states)

        # ──────────────────────────────────────────────────
        # 对每个状态 s，应用贝尔曼最优方程:
        #
        #   V^{k+1}(s) = max_a [ R(s,a) + γ V^{k}(s') ]
        #
        # 注意: 价值迭代中不维护策略 —— 只迭代 V，
        #       策略在收敛后再通过 argmax 提取。
        # ──────────────────────────────────────────────────
        for s in range(n_states):
            q_values = np.zeros(n_actions)  # 临时存储当前状态各动作的 Q 值

            for a in range(n_actions):
                # 即时奖励 R(s,a)
                reward = env.get_reward(s, a)

                # 后继状态 s'
                next_s = env.get_next_state(s, a)

                # V^{k}(s'): 上一轮的后继状态价值
                if next_s == -1:
                    v_next = 0.0       # 终止状态 V = 0
                else:
                    v_next = V[next_s]  # 使用上一轮的 V^k

                # Q(s,a) = R(s,a) + γ V^k(s')
                q_values[a] = reward + gamma * v_next

            # V^{k+1}(s) = max_a Q(s,a)   ← 贝尔曼最优备份
            V_new[s] = np.max(q_values)

        # 计算最大变化量
        delta = np.max(np.abs(V_new - V))

        V = V_new
        history.append(V.copy())

        if verbose and (k <= 10 or k % 50 == 0 or delta < theta):
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


# ============================================================
# 第三部分：可视化
# ============================================================


def plot_value_iteration_history(env, history, gamma):
    """
    绘制价值迭代过程中 V(s) 的逐步收敛过程。
    可视化"反向传播"效应：最优价值从终止状态向外扩散。
    """
    total_iters = len(history) - 1
    if total_iters <= 6:
        show_iters = list(range(len(history)))
    else:
        indices = [0, 1, 2, min(5, total_iters), min(10, total_iters),
                   min(20, total_iters), min(total_iters // 2, total_iters),
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

    all_values = np.concatenate([h for h in history])
    vmin, vmax = np.min(all_values), np.max(all_values)
    norm = Normalize(vmin=vmin, vmax=vmax) if vmin != vmax else Normalize()
    cmap = plt.get_cmap('RdYlGn')

    for plot_idx, iter_idx in enumerate(show_iters):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]

        V = history[iter_idx]
        grid = np.full((env.rows, env.cols), np.nan)
        for i, (r, c) in enumerate(env.state_coords):
            grid[r, c] = V[i]
        for tr, tc in env.terminal_states:
            grid[tr, tc] = 0.0

        im = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f"Iter {iter_idx}", fontsize=11)

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

    for plot_idx in range(n_plots, n_rows * n_cols):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        axes[row, col].set_visible(False)

    plt.colorbar(im, ax=axes, shrink=0.8, label="State Value V*(s)")
    fig.suptitle(f"Value Iteration: Bellman Optimality Backup (γ={gamma})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_optimal_policy(env, V, policy, gamma):
    """绘制最优价值 V* 网格和最优策略箭头。"""
    fig, ax = plt.subplots(figsize=(6, 5))

    grid = np.full((env.rows, env.cols), np.nan)
    for i, (r, c) in enumerate(env.state_coords):
        grid[r, c] = V[i]
    for tr, tc in env.terminal_states:
        grid[tr, tc] = 0.0

    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    norm = Normalize(vmin=vmin, vmax=vmax) if vmin != vmax else Normalize()
    cmap = plt.get_cmap('RdYlGn')

    im = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')

    # 在每格中写入 V 值和策略箭头
    arrow_deltas = {0: (0, 0.35), 1: (0.35, 0), 2: (0, -0.35), 3: (-0.35, 0)}
    for i, (r, c) in enumerate(env.state_coords):
        val = grid[r, c]
        color = 'white' if abs(val) > (vmax - vmin) * 0.6 + vmin else 'black'
        ax.text(c, r, f"{val:.1f}", ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)
        # 画箭头
        a = policy[i]
        dc, dr = arrow_deltas[a]
        ax.arrow(c, r, dc, dr, head_width=0.15, head_length=0.1,
                 fc='blue', ec='blue', alpha=0.8, linewidth=2)

    for tr, tc in env.terminal_states:
        ax.text(tc, tr, "T", ha='center', va='center',
                fontsize=14, fontweight='bold', color='gray')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, label="V*(s)")
    ax.set_title(f"Optimal Value V*(s) & Policy π*(s)  (γ={gamma})",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_convergence_curve(history, theta, gamma):
    """价值迭代的收敛曲线（对数坐标）。"""
    deltas = []
    for t in range(1, len(history)):
        delta = np.max(np.abs(history[t] - history[t - 1]))
        deltas.append(delta)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(deltas) + 1), deltas, 'b-o', markersize=3, linewidth=1.5)
    ax.axhline(y=theta, color='r', linestyle='--', label=f'θ = {theta}')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Max |ΔV(s)|", fontsize=12)
    ax.set_title(f"Convergence of Value Iteration (γ={gamma})", fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def compare_gammas(env, gammas, theta=1e-6):
    """
    对比不同折扣因子 γ 下的最优价值 V*。
    γ=1 时最优策略只关心最短路径到终点；
    γ<1 时还要考虑尽早拿到奖励。
    """
    results = {}
    for gamma in gammas:
        V, _, _, _, iters = value_iteration(
            env, gamma=gamma, theta=theta, verbose=False)
        results[gamma] = (V, iters)

    fig, axes = plt.subplots(1, len(gammas), figsize=(5 * len(gammas), 4))
    if len(gammas) == 1:
        axes = [axes]

    all_v = np.concatenate([r[0] for r in results.values()])
    vmin, vmax = np.min(all_v), np.max(all_v)
    norm = Normalize(vmin=vmin, vmax=vmax) if vmin != vmax else Normalize()

    for idx, gamma in enumerate(gammas):
        V, iters = results[gamma]
        ax = axes[idx]
        grid = np.full((env.rows, env.cols), np.nan)
        for i, (r, c) in enumerate(env.state_coords):
            grid[r, c] = V[i]
        for tr, tc in env.terminal_states:
            grid[tr, tc] = 0.0

        im = ax.imshow(grid, cmap=plt.get_cmap('RdYlGn'), norm=norm,
                       interpolation='nearest')
        ax.set_title(f"γ = {gamma}  ({iters} iters)", fontsize=12, fontweight='bold')

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

    plt.colorbar(im, ax=axes, shrink=0.8, label="V*(s)")
    fig.suptitle("Optimal Value V* under Different γ (Value Iteration)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# 第四部分：主程序
# ============================================================


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

    # ══════════════════════════════════════════════════════════
    # 实验 2: 不同折扣因子对比
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("实验 2: 不同折扣因子 γ 的对比")
    print("-" * 60)

    for g in [0.0, 0.5, 0.9, 1.0]:
        V_g, pi_g, _, _, iters_g = value_iteration(
            env, gamma=g, theta=1e-6, verbose=False)
        print(f"\nγ = {g} ({iters_g} 轮收敛):")
        env.print_grid(V_g, title=f"V*(s)  γ={g}")

    # ══════════════════════════════════════════════════════════
    # 可视化
    # ══════════════════════════════════════════════════════════

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 图1: 价值迭代收敛过程
    fig1 = plot_value_iteration_history(env, history, gamma=1.0)
    path1 = os.path.join(output_dir, "value_iteration_history.png")
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"\n[已保存] 价值迭代收敛过程 → {path1}")

    # 图2: 收敛曲线
    fig2 = plot_convergence_curve(history, theta=1e-6, gamma=1.0)
    path2 = os.path.join(output_dir, "value_iteration_convergence.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"[已保存] 收敛曲线 → {path2}")

    # 图3: 最优策略
    fig3 = plot_optimal_policy(env, V_star, policy, gamma=1.0)
    path3 = os.path.join(output_dir, "optimal_policy.png")
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"[已保存] 最优策略图 → {path3}")

    # 图4: 不同 γ 对比
    fig4 = compare_gammas(env, gammas=[0.0, 0.5, 0.9, 1.0])
    path4 = os.path.join(output_dir, "value_iteration_gamma_comparison.png")
    fig4.savefig(path4, dpi=150, bbox_inches='tight')
    print(f"[已保存] 不同γ对比 → {path4}")

    print("\n" + "=" * 60)
    print("所有实验完成! 图片已保存到 output/ 目录。")
    print("=" * 60)

    plt.close('all')


if __name__ == "__main__":
    main()
