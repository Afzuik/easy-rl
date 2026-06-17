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

class SmallGridWorld:
    def __init__(self):
        self.rows=4
        self.cols=4
        self.actions = {
            0: (-1, 0),  # 上
            1: (0, 1),   # 右
            2: (1, 0),   # 下
            3: (0, -1),  # 左
        }
        self.action_names = {0: "up", 1: "right", 2: "down", 3: "left"}
        self.n_actions = 4
        self.terminal_states = {(0, 0), (3, 3)}
        self.state_coords = []
        self.coords_to_idx = {}
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.terminal_states:
                    self.state_coords.append((r, c))
                    self.coords_to_idx[(r, c)] = idx
                    idx += 1
        self.n_states = len(self.state_coords)
        self.step_reward = -1.0
    
    def get_next_state(self,state_idx,action):
        # 对应书中的确定性转移: p(s'|s,a) = 1
        # 例如 p(2|6, 上) = 1，即从状态 6 往上走一定到状态 2。

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
        return self.coords_to_idx[(nr, nc)]
    
    def get_reward(self,state_idx,action):
        return self.step_reward

    def print_grid(self,values=None,title=""):
        print(f"\n{title}")
        print("="*40)
        grid = np.zeros((self.rows,self.cols))
        if values is not None:
            for i,(r,c) in enumerate(self.state_coords):
                grid[r,c]=values[i]
        else:
            for i,(r,c) in enumerate(self.state_coords):
                grid[r,c] = i+1
        
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r,c) in self.terminal_states:
                    row_str += f"{'T':>8s}"
                elif values is not None:
                    row_str += f"{grid[r,c]:>8.2f}"
                else:
                    row_str += f"{int(grid[r,c]):>8d}"
            print(row_str)
        print("="*40)

def uniform_random_policy(n_actions):
    return np.ones(n_actions)/n_actions

def left_only_policy(n_actions):
    pi = np.zeros(n_actions)
    pi[3] = 1.0
    return pi

def policy_evaluation(env,policy_func,gamma=1.0,theta=1e-6,max_iter=1000,verbose=True):
    n_states = env.n_states
    n_actions = env.n_actions

    V = np.zeros(n_states)
    history = [V.copy()]
    if verbose:
        print("\n" + "=" * 60)
        print(f"策略评估开始")
        print(f"折扣因子 gamma = {gamma}")
        print(f"收敛阈值 theta = {theta}")
        print("=" * 60)
        print(f"\n{'迭代次数':>8s} | {'最大变化量Δ':>14s}")
        print("-"*30)

    for k in range(1,max_iter+1):
        V_new = np.zeros(n_states)
        pi = policy_func(n_actions)

        for s in range(n_states): # 对每一个环境
            expected_value = 0.0 # 期望值
            for a in range(n_actions): # 对每一个动作
                reward = env.get_reward(s,a) # (s,a)->R(s,a)
                next_s = env.get_next_state(s,a) # 下一个状态(确定性转移)
                if next_s == -1:
                    v_next = 0.0
                else:
                    v_next = V[next_s]

                expected_value += pi[a] * (reward+gamma*v_next)

            V_new[s] = expected_value

        delta = np.max(np.abs(V_new-V))

        V = V_new
        history.append(V.copy())

        if verbose and (k<=10 or k%50 == 0 or delta<theta):
            print(f"{k:>8d} | {delta:>14.6f}")
        if delta < theta:
            if verbose:
                print("-" * 30)
                print(f"在第 {k} 轮迭代后收敛！ (Δ = {delta:.2e} < θ = {theta})")
            break

    return V, history, k

def solve_analytic(env,policy_func,gamma):
    n_states = env.n_states
    n_actions = env.n_actions
    pi = policy_func(n_actions)

    # 构建P_🥧和R_🥧
    P_pi = np.zeros((n_states,n_states))
    r_pi = np.zeros(n_states)

    for s in range(n_states):
        for a in range(n_actions):
            reward = env.get_reward(s,a)
            r_pi[s] += pi[a] * reward
            next_s = env.get_next_state(s,a)
            if next_s != -1:
                P_pi[s,next_s] += pi[a]
    
    I = np.eye(n_states)
    V = np.linalg.solve(I-gamma*P_pi,r_pi)

    return V


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


def main():
    print("=" * 60)
    print("  第 2 章 马尔可夫决策过程 -- 策略评估演示程序")
    print("  Policy Evaluation via Iterative Bellman Equation")
    print("=" * 60)

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









