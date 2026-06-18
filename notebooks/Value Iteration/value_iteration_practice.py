"""
================================================================================
 价值迭代 (Value Iteration) 实践代码
 基于《Easy-RL》第2章 第 2.19 节的价值迭代算法

 价值迭代的核心思想：
   利用贝尔曼最优方程 (Bellman Optimality Equation)，反复对所有状态执行
   "最优备份"操作，将最优价值从终止状态逐步反向传播到所有状态。

 算法步骤（严格对应 chapter2_order.md 第2.19节）：
   初始化: V(s) = 0  对所有状态 s
   循环直到收敛:
     对每个状态 s:
       对每个动作 a:
         Q(s,a) = R(s,a) + γ * Σ P(s'|s,a) * V_k(s')     -- 式(2.23)
       V_{k+1}(s) = max_a Q(s,a)                           -- 式(2.24)
     如果 max_s |V_{k+1}(s) - V_k(s)| < θ: 退出

 与策略迭代的区别（第2.20节）：
   - 策略迭代：贝尔曼期望方程 + 贪心改进，每轮需完整评估策略
   - 价值迭代：贝尔曼最优方程，每轮仅一次全状态扫描
   - 中间结果：价值迭代的中间V值无实际意义，只有收敛后的 V* 才有意义
================================================================================
"""

import sys
import io

# 修复 Windows 终端中文编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：搭建经典的 4×4 小网格世界环境
#
# 对应图 2.18 的 small gridworld:
#   - 4×4 网格，状态编号 0~15
#   - 左上角(0) 和右下角(15) 是终止状态（阴影方块）
#   - 每走一步获得 -1 的奖励
#   - 动作结果确定性的：执行哪个方向就走向哪个方向
#   - 出边界时状态不变
#   - 智能体希望尽快到达终止状态（最大化累积奖励）
# ============================================================================

class GridWorld:
    """
    经典 4×4 网格世界（图 2.18）

    状态空间:
       0   1   2   3
       4   5   6   7
       8   9  10  11
      12  13  14  15

    其中 0 和 15 是终止状态。
    """

    def __init__(self, rows=4, cols=4, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.nS = rows * cols          # 状态总数: 16
        self.nA = 4                     # 动作数: 上/下/左/右

        # 动作定义: 0=上, 1=下, 2=左, 3=右
        self.action_names = ['↑上', '↓下', '←左', '→右']
        # 每个动作对应的 (行偏移, 列偏移)
        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 终止状态: 左上角 (0,0) 和 右下角 (3,3)
        self.terminal_states = {0, rows * cols - 1}

        # 构建状态转移矩阵 P
        # P[s][a] = [(probability, next_state, reward, done), ...]
        self.P = self._build_transitions()

    def _state_to_rc(self, s):
        """状态编号 → (行, 列)"""
        return s // self.cols, s % self.cols

    def _rc_to_state(self, r, c):
        """(行, 列) → 状态编号"""
        return r * self.cols + c

    def _build_transitions(self):
        """
        构建确定性的状态转移矩阵

        转移规则:
        - 非终止状态: 按动作方向移动一格，reward = -1
        - 出界: 留在原地
        - 到达终止状态: done = True
        - 已在终止状态: 停留原地，reward = 0 (终止后不再有奖励)
        """
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for s in range(self.nS):
            # 终止状态的处理很重要！
            # 终止状态不再产生奖励，所有动作都停留在原地
            if s in self.terminal_states:
                for a in range(self.nA):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue

            r, c = self._state_to_rc(s)

            for a in range(self.nA):
                dr, dc = self.dirs[a]
                nr, nc = r + dr, c + dc

                # 边界检查
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    next_s = self._rc_to_state(nr, nc)
                else:
                    next_s = s  # 出界，留在原地

                done = (next_s in self.terminal_states)
                reward = -1.0  # 每走一步 -1

                # 确定性转移：概率 = 1.0
                P[s][a] = [(1.0, next_s, reward, done)]

        return P


# ============================================================================
# 第二部分：价值迭代算法
#
# 严格按照式(2.22)-(2.24)实现：
#   V_{k+1}(s) = max_a [ R(s,a) + γ * Σ P(s'|s,a) * V_k(s') ]
# ============================================================================

def value_iteration(env, theta=1e-6, max_iter=10000, verbose=True):
    """
    价值迭代算法

    核心更新公式 (式2.22):
        V_new(s) ← max_a [ R(s,a) + γ * Σ_{s'} P(s'|s,a) * V_old(s') ]

    参数:
        env:      GridWorld 环境
        theta:    收敛阈值 (当 max|V_new - V_old| < theta 时停止)
        max_iter: 最大迭代次数
        verbose:  是否打印迭代过程

    返回:
        V:        最优状态价值函数 V*
        policy:   最优确定性策略, policy[s] = 最优动作
        history:  每轮的 delta 值列表
    """
    nS, nA = env.nS, env.nA
    gamma = env.gamma

    # 步骤1: 初始化 V(s) = 0
    V = np.zeros(nS)

    # 记录每轮的 delta
    history = []

    print(f"\n{'='*55}")
    print(f"  开始价值迭代 (γ={gamma}, θ={theta})")
    print(f"{'='*55}")

    for k in range(1, max_iter + 1):
        delta = 0.0  # 本轮最大变化量

        # 步骤2: 对每个状态执行贝尔曼最优备份
        for s in range(nS):
            # 终止状态的价值始终为 0（吸收状态）
            if s in env.terminal_states:
                continue

            v_old = V[s]

            # 计算所有动作的 Q(s,a) 并取最大值
            # Q(s,a) = R(s,a) + γ * Σ P(s'|s,a) * V(s')   — 式(2.23)
            action_values = []
            for a in range(nA):
                q = 0.0
                for prob, next_s, reward, done in env.P[s][a]:
                    # 关键: 即使 done=True，V[next_s] 也是 0（终止状态价值为0）
                    # 所以不需要特殊处理
                    q += prob * (reward + gamma * V[next_s])
                action_values.append(q)

            # V_{k+1}(s) = max_a Q(s,a)   — 式(2.24)
            V[s] = max(action_values)

            # 更新最大变化量
            delta = max(delta, abs(V[s] - v_old))

        history.append(delta)

        # 打印前几轮和关键节点的迭代信息
        if verbose and (k <= 5 or k % 100 == 0 or delta < theta):
            print(f"  第 {k:4d} 轮: Δ = {delta:.8f}")

        # 收敛判断
        if delta < theta:
            if verbose:
                print(f"\n  [√]收敛! 共迭代 {k} 轮")
            break

    if k >= max_iter and delta >= theta:
        print(f"\n  [!]达到最大迭代次数 {max_iter}，未完全收敛 (Δ={delta:.6f})")

    # 步骤3: 从 V* 提取最优策略
    # π*(s) = argmax_a [ R(s,a) + γ * Σ P(s'|s,a) * V*(s') ]
    policy = extract_optimal_policy(env, V)

    return V, policy, history


def extract_optimal_policy(env, V):
    """
    从最优价值函数 V* 提取最优策略

    对应第2.19节公式:
    π*(s) = argmax_a [ R(s,a) + γ * Σ_{s'} P(s'|s,a) * V*(s') ]

    即: 对每个状态，选择使 Q(s,a) 最大的动作
    """
    nS, nA = env.nS, env.nA
    gamma = env.gamma
    policy = np.zeros(nS, dtype=int)

    for s in range(nS):
        if s in env.terminal_states:
            policy[s] = 0  # 终止状态，动作无意义
            continue

        best_action = 0
        best_value = float('-inf')

        for a in range(nA):
            q = 0.0
            for prob, next_s, reward, done in env.P[s][a]:
                q += prob * (reward + gamma * V[next_s])

            if q > best_value:
                best_value = q
                best_action = a

        policy[s] = best_action

    return policy


# ============================================================================
# 第三部分：可视化
# ============================================================================

def print_value_table(V, env, title="状态价值函数"):
    """以网格形式打印价值函数"""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    for r in range(env.rows):
        row_str = "  |"
        for c in range(env.cols):
            s = r * env.cols + c
            if s in env.terminal_states:
                row_str += f"  终点  |"
            else:
                row_str += f" {V[s]:7.3f}|"
        print(row_str)
        if r < env.rows - 1:
            print("  |--------------|----------------|----------------|----------------|--")
    print()


def print_policy_table(policy, env):
    """以网格形式打印策略"""
    arrows = ['↑', '↓', '←', '→']

    print(f"{'='*55}")
    print(f"  最优策略 π*(s)")
    print(f"{'='*55}")
    for r in range(env.rows):
        row_str = "  |"
        for c in range(env.cols):
            s = r * env.cols + c
            if s in env.terminal_states:
                row_str += f"  终点  |"
            else:
                row_str += f"   {arrows[policy[s]]}   |"
        print(row_str)
        if r < env.rows - 1:
            print("  |--------------|----------------|----------------|----------------|--")
    print()


def plot_convergence(history):
    """绘制价值迭代的收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history, 'b-', linewidth=1)
    ax.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='收敛阈值 θ=1e-6')
    ax.set_yscale('log')
    ax.set_xlabel('迭代轮数')
    ax.set_ylabel('Δ (对数尺度)')
    ax.set_title('价值迭代收敛曲线 —— Δ 随迭代轮数指数下降')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('value_iteration_convergence.png', dpi=150)
    plt.show()
    print("\n  [收敛曲线已保存至 value_iteration_convergence.png]")


def visualize_value_grid(V, env, title="最优状态价值 V*(s)"):
    """用热力图可视化状态价值"""
    grid = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            s = r * env.cols + c
            grid[r, c] = V[s]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap='RdYlGn', aspect='equal')

    # 标注每个格子的值
    for r in range(env.rows):
        for c in range(env.cols):
            s = r * env.cols + c
            text = '终点' if s in env.terminal_states else f'{V[s]:.2f}'
            color = 'white' if abs(V[s]) > 2 else 'black'
            ax.text(c, r, text, ha='center', va='center', fontsize=11,
                   color=color, fontweight='bold')

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticklabels([f'列{i}' for i in range(env.cols)])
    ax.set_yticklabels([f'行{i}' for i in range(env.rows)])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig('value_iteration_heatmap.png', dpi=150)
    plt.show()
    print("  [价值热力图已保存至 value_iteration_heatmap.png]")


# ============================================================================
# 第四部分：演示价值的逐轮传播（对应图 2.23）
# ============================================================================

def demo_value_propagation(env):
    """
    演示价值迭代过程中价值的逐步反向传播

    对应图 2.23：价值从终点向外扩散
    - 第1轮: 紧邻终点的状态获得值
    - 第2轮: 值扩散到距离2的状态
    - ...
    - 就像水波从终点向外传播
    """
    nS, nA = env.nS, env.nA
    gamma = env.gamma
    V = np.zeros(nS)

    # 要展示的轮次
    show_steps = [1, 2, 3, 5, 10, 50]
    max_step = max(show_steps)

    print(f"\n{'='*55}")
    print(f"  价值传播演示 —— 观察 V 如何从终点向外扩散")
    print(f"  (对应图 2.23: 价值迭代的反向传播过程)")
    print(f"{'='*55}")

    for k in range(1, max_step + 1):
        V_old = V.copy()
        for s in range(nS):
            if s in env.terminal_states:
                continue
            action_vals = []
            for a in range(nA):
                q = 0.0
                for prob, ns, r, done in env.P[s][a]:
                    q += prob * (r + gamma * V_old[ns])
                action_vals.append(q)
            V[s] = max(action_vals)

        if k in show_steps:
            print(f"\n  —— 第 k={k} 轮迭代 ——")
            for r in range(env.rows):
                row_str = "  "
                for c in range(env.cols):
                    s = r * env.cols + c
                    if s in env.terminal_states:
                        row_str += " [终点] "
                    else:
                        row_str += f" {V[s]:7.3f}"
                print(row_str)


# ============================================================================
# 第五部分：对比不同 γ 的影响
# ============================================================================

def compare_gamma():
    """
    比较不同折扣因子对结果的影响

    对应第2.5节: γ 的作用
    - γ=0: 只关注即时奖励（非常"短视"）
    - γ 接近 1: 重视长期奖励（"远视"）
    """
    print(f"\n{'='*55}")
    print(f"  对比不同折扣因子 γ 对 V* 的影响")
    print(f"  (γ=0 只看眼前, γ→1 看重未来)")
    print(f"{'='*55}")

    for gamma in [0.1, 0.5, 0.9, 0.99]:
        env = GridWorld(gamma=gamma)
        V, policy, hist = value_iteration(env, theta=1e-6, verbose=False)

        print(f"\n  γ = {gamma} (迭代 {len(hist)} 轮)")
        for r in range(env.rows):
            row_str = "  "
            for c in range(env.cols):
                s = r * env.cols + c
                if s in env.terminal_states:
                    row_str += " [终点] "
                else:
                    row_str += f" {V[s]:7.3f}"
            print(row_str)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  价值迭代 (Value Iteration) 学习实践")
    print("  基于《Easy-RL》第2章 第2.19节")
    print("=" * 55)

    # ---- 1. 创建环境 ----
    env = GridWorld(rows=4, cols=4, gamma=0.9)
    print(f"\n[Env]环境: {env.rows}×{env.cols} 网格世界")
    print(f"   状态数: {env.nS}  |  动作数: {env.nA} (上/下/左/右)")
    print(f"   终止状态: 左上角(状态0) 和 右下角(状态{env.nS-1})")
    print(f"   奖励: 每走一步 -1 (促使智能体尽快到达终点)")

    # ---- 2. 演示价值传播过程 ----
    demo_value_propagation(env)

    # ---- 3. 正式执行价值迭代 ----
    V, policy, history = value_iteration(env, theta=1e-6)

    # ---- 4. 展示结果 ----
    print_value_table(V, env, "最优状态价值函数 V*(s)")
    print_policy_table(policy, env)

    # ---- 5. 打印各状态的 Q 值（帮助理解决策） ----
    print(f"\n{'='*55}")
    print(f"  每个状态下各动作的 Q(s,a) 值")
    print(f"  (*标记的为最优动作)")
    print(f"{'='*55}")
    for s in range(env.nS):
        if s in env.terminal_states:
            continue
        r, c = env._state_to_rc(s)
        qs = []
        for a in range(env.nA):
            q = 0.0
            for prob, ns, rew, done in env.P[s][a]:
                q += prob * (rew + env.gamma * V[ns])
            qs.append(q)
        best = np.argmax(qs)
        print(f"\n  状态 s={s:2d} ({r},{c}):")
        for a in range(env.nA):
            mark = " ★" if a == best else "  "
            print(f"    {env.action_names[a]}: Q={qs[a]:8.4f}{mark}")

    # ---- 6. 策略走一步验证 ----
    print(f"\n{'='*55}")
    print(f"  策略验证：从每个非终止状态走一步")
    print(f"{'='*55}")
    for s in range(env.nS):
        if s in env.terminal_states:
            continue
        r, c = env._state_to_rc(s)
        a = policy[s]
        dr, dc = env.dirs[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.rows and 0 <= nc < env.cols:
            ns = env._rc_to_state(nr, nc)
        else:
            ns = s
        print(f"  状态{s:2d}({r},{c}) --{env.action_names[a]}--> 状态{ns:2d}({nr},{nc})")

    # ---- 7. 对比不同折扣因子 ----
    compare_gamma()

    # ---- 8. 绘制收敛曲线和热力图 ----
    try:
        plot_convergence(history)
    except Exception as e:
        print(f"\n  [收敛曲线绘制跳过: {e}]")

    try:
        visualize_value_grid(V, env)
    except Exception as e:
        print(f"\n  [热力图绘制跳过: {e}]")

    # ---- 9. 总结 ----
    print(f"\n{'='*55}")
    print(f"  学习要点总结")
    print(f"{'='*55}")
    print(f"""
  1. 价值迭代的核心公式（贝尔曼最优方程）:
     V*(s) = max_a [ R(s,a) + γ * Σ P(s'|s,a) * V*(s') ]

  2. 与策略迭代的区别:
     - 策略迭代: 贝尔曼期望方程 + 贪心改进 (评估⇄改进交替)
     - 价值迭代: 贝尔曼最优方程 (直接迭代到 V*)

  3. 收敛保证:
     贝尔曼最优算子是 γ-压缩映射，从任意初始值都能收敛到 V*

  4. 重要细节:
     - 中间迭代的 V 值没有实际意义（不等于任何策略的价值）
     - 只有收敛后的 V* 才能提取出正确的 π*
     - 终止状态的价值必须保持为 0（吸收状态）

  5. 局限性:
     - 需要完整的环境模型 P(s'|s,a) 和 R(s,a)
     - 状态数多时计算量巨大（需扫描所有状态）
     - 第3章的 Q-learning 等方法将解决免模型问题
    """)
