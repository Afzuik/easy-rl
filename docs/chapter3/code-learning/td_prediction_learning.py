"""
第 3 章 时序差分预测 TD(0)（TD Prediction）

核心思想:
    每走一步就更新一次，不等回合结束。用 "即时奖励 + 折扣的下一状态估计值"
    作为目标，实现自举（bootstrap）。

更新公式:
    V(s_t) ← V(s_t) + α [ r_{t+1} + γ V(s_{t+1}) - V(s_t) ]
               ↑              ↑             ↑
           当前估计         TD 目标       TD 误差 δ_t

对比 MC:
    MC:  V(s_t) ← V(s_t) + α [ G_t     - V(s_t) ]  目标 = 完整真实回报
    TD:  V(s_t) ← V(s_t) + α [ r + γV' - V(s_t) ]  目标 = 采样 + 自举

TD 优势: 在线学习、不需完整回合、可用于连续任务
TD 代价: 有偏（因为 V(s_{t+1}) 初始不准）

运行方式:
    python td_prediction.py
"""

import numpy as np
import sys

if hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class SimpleGridWorld:
    def __init__(self):
        self.rows,self.cols = 4,4
        self.terminal = {(0,0),(3,3)}
        self.actions = {
            0:(-1,0),
            1:(0,1),
            2:(1,0),
            3:(0,-1)
        }
        self.n_actions = 4
        self.states = []
        self.coord_to_idx = {}
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in self.terminal:
                    self.states.append((r,c))
                    self.coord_to_idx[(r,c)] = idx
                    idx += 1
        self.n_states = len(self.states)

    def reset(self):
        idx = np.random.randint(self.n_states)
        return idx
    
    def step(self,state_idx,action):
        r,c = self.states[state_idx]
        dr,dc = self.actions[action]
        nr,nc = r+dr,c+dc
        if nr<0 or nr>=self.rows or nc<0 or nc<=self.cols:
            return state_idx,-1.0,False
        if (nr,nc) in self.terminal:
            return -1,-1.0,True
        return self.coord_to_idx[(nr,nc)],-1,False
    
# ============================================================
# TD(0) 预测
# ============================================================
def td_prediction(env,policy_probs,alpha=0.1,gamma=1.0,n_episodes=1000,verbose=True):
    """
    TD(0) 策略评估。

    每走一步执行:
        V(s) ← V(s) + α [ r + γ V(s') - V(s) ]

    参数:
        env:           环境
        policy_probs:  策略 π(a|s) 的概率向量 (shape=(4,))
        alpha:         学习率 (0 < α ≤ 1)
        gamma:         折扣因子
        n_episodes:    回合数
        verbose:       是否打印进度

    返回:
        V:           收敛后的 V(s)
        errors:      每个 TD 误差，用于收敛分析
    """
    V = np.zeros(env.n_states)
    errors = []
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        step = 0

        while not done and step < 200:
            action = np.random.choice(env.n_actions,p=policy_probs)

            next_state,reward,done = env.step(state,action)

            if done or next_state == -1:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]
            
            td_error = td_target - V[state]
            V[state] += alpha * td_error
            errors.append(td_error)

            state = next_state
            step += 1
        
        if verbose and (ep+1)%200 == 0:
            mean_error = np.mean(np.abs(errors[-200:]))
            print(f"    回合{ep+1:>4d}/{n_episodes} | 最近200步平均TD误差 : {mean_error:.4f}")
    
    return V,errors

def td_prediction_with_decay(env,policy_probs,gamma=1.0,n_episodes=1000,verbose=True):
    """_summary_
    带学习率衰减的TD(0)
    a_t = 1/(1+t/N(s))，访问越多的状态更新幅度越小
    Args:
        env (_type_): _description_
        policy_probs (_type_): _description_
        gamma (float, optional): _description_. Defaults to 1.0.
        n_epsilon (int, optional): _description_. Defaults to 1000.
        verbose (bool, optional): _description_. Defaults to True.
    """
    V = np.zeros(env.n_states)
    N = np.zeros(env.n_states)

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        step = 0

        while not done and step<200:
            action = np.random.choice(env.n_actions,p=policy_probs)

            next_state,reward,done = env.step(state,action)
            N[state] += 1
            alpha = 1.0/N[state]

            if done or next_state == -1:
                td_target = reward
            else:
                td_target = reward+gamma * V[next_state]
            
            V[state] += alpha * (td_target-V[state])
            state = next_state
            step += 1
        
        if verbose and (ep+1) % 200 == 0:
            print(f"    回合{ep+1:>4d}/{n_episodes}衰减学习率")
    
    return V

def print_values(env,V,title=""):
    """_summary_
    4x4网格打印
    Args:
        env (_type_): _description_
        V (_type_): _description_
        title (str, optional): _description_. Defaults to "".
    """
    print(f"\n{title}")
    print("*"*40)
    grid = np.full((env.rows,env.cols),np.nan)
    for i,(r,c) in enumerate(env.states):
        grid[r,c] = V[i]
    for r in range(env.rows):
        row_str = ""
        for c in range(env.cols):
            if (r,c) in env.terminal:
                row_str += f"{'T':>8s}"
            else:
                row_str += f"{grid[r,c]:>8.3f}"
        print(row_str)
    print("="*40)

# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("  第 3 章 TD(0) 时序差分预测 (TD Prediction)")
    print("  V(s_t) ← V(s_t) + α [ r + γ V(s_{t+1}) - V(s_t) ]")
    print("=" * 60)

    env = SimpleGridWorld()
    pi = np.ones(env.n_actions) / env.n_actions  # 均匀随机策略

    print(f"\n策略: 均匀随机 π(a|s) = {pi}")

    # ── 实验 1: 固定学习率 TD(0) ──
    print("\n[实验 1] 固定学习率 α = 0.1")
    np.random.seed(42)
    V_td, errors = td_prediction(env, pi, alpha=0.1, gamma=1.0,
                                  n_episodes=1000, verbose=True)

    print_values(env, V_td, "TD(0) α=0.1 后的 V(s)")

    # ── 实验 2: 衰减学习率 ──
    print("\n[实验 2] 衰减学习率 α_t = 1/N(s)")
    np.random.seed(42)
    V_decay = td_prediction_with_decay(env, pi, gamma=1.0,
                                        n_episodes=1000, verbose=True)
    print_values(env, V_decay, "TD(0) 衰减学习率后的 V(s)")

    # ── 实验 3: 不同 α 对比 ──
    print("\n[实验 3] 不同学习率 α 对比 (各 500 回合):")
    for alpha in [0.01, 0.05, 0.1, 0.5]:
        np.random.seed(42)
        V_a, _ = td_prediction(env, pi, alpha=alpha, gamma=1.0,
                                n_episodes=500, verbose=False)
        # 展示最远状态 (index 13, 即右下角倒数第二个)
        print(f"  α={alpha:.2f} → V(状态14) = {V_a[13]:.3f}")

    print("\n" + "=" * 60)
    print("核心要点:")
    print("  1. TD(0) 每步更新: 不等回合结束")
    print("  2. 目标 = r_{t+1} + γ V(s_{t+1}) —— 采样 + 自举")
    print("  3. α 固定 vs 衰减: 固定适应非平稳, 衰减渐近收敛")
    print("  4. MC 无偏差高方差, TD 有偏差低方差")
    print("=" * 60)

if __name__ == "__main__":
    main()