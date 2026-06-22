import numpy as np
import sys
import os

if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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
        self.start_states = self.states
    
    def reset(self,start_idx = None):
        if start_idx is None:
            start_idx = np.random.randint(self.n_states)
        return start_idx
    
    def step(self,state_idx,action):
        # 执行动作，返回(next_idx,reward,done)
        r,c = self.states[state_idx]
        dr,dc = self.actions[action]
        nr,nc = r+dr,c+dc

        if nr<0 or nr>=self.rows or nc<0 or nc>=self.cols:
            return state_idx,-1.0,False
        if (nr,nc) in self.terminal:
            return -1,-1.0,True
        return self.coord_to_idx[(nr,nc)],-1.0,False
    

# ============================================================
# 蒙特卡洛策略评估
# ============================================================
def generate_episode(env,policy_probs):
    '''
    采样一条轨迹，返回[(state_idx,action,reward),...]
    策略:均匀随机 Π(a|s)=policy_probs
    '''
    episode = []
    state = env.reset()
    done = False

    while not done:
        action = np.random.choice(env.n_actions,p=policy_probs)
        next_state,reward,done = env.step(state,action)
        episode.append((state,action,reward))
        state = next_state
    
    return episode

def mc_prediction_batch(env,policy_probs,n_episodes = 5000,gamma=1.0):
    """_summary_
    批量蒙特卡洛：采样所有轨迹后，用经验平均估计V(s)
    步骤：
        1.采样n_episode轨迹
        2.对每条轨迹，从后往前计算 G_t = r_{t+1}+γG_{t+1}
        3.V(s) = Σ G_t/N(s)
    Args:
        env (_type_): _description_
        policy_probs (_type_): _description_
        n_episodes (int, optional): _description_. Defaults to 5000.
        gamma (float, optional): _description_. Defaults to 1.0.
    """
    returns_sum = np.zeros(env.n_states)
    returns_count = np.zeros(env.n_states)

    for ep in range(n_episodes):
        episode = generate_episode(env,policy_probs)
        G = 0.0
        for state,action,reward in reversed(episode):
            G = reward + gamma * G
            returns_sum[state] += G
            returns_count[state] += 1
    
    V = np.divide(returns_sum,
                  returns_count,
                  out=np.zeros_like(returns_sum),
                  where=returns_count>0)
    return V,returns_count

def mc_prediction_incremental(env,policy_probs,n_episodes=5000,gamma=1.0):
    """_summary_
    增量蒙特卡洛：每来一条轨迹就更新一次
    更新公式:
        V(s_t) <- V(s_t)+(1/N(s_t)) x (G_t-V(s_t))
    Args:
        env (_type_): _description_
        policy_probs (_type_): _description_
        n_episode (int, optional): _description_. Defaults to 5000.
        gamma (float, optional): _description_. Defaults to 1.0.
    """
    V = np.zeros(env.n_states)
    N = np.zeros(env.n_states)
    for ep in range(n_episodes):
        episode = generate_episode(env,policy_probs)
        G = 0.0
        for state,action,reward in reversed(episode):
            G = reward + gamma * G
            N[state] += 1
            V[state] += (1.0/N[state])*(G-V[state])
    return V,N

def print_values(env,V,title=""):
    print(f"\n{title}")
    print("="*40)
    grid = np.full((env.rows,env.cols),np.nan)
    for i,(r,c) in enumerate(env.states):
        grid[r,c] = V[i]
    for r in range(env.rows):
        row = ""
        for c in range(env.cols):
            if (r,c) in env.terminal:
                row += f"{'T':>8s}"
            else:
                row += f"{grid[r,c]:>8.3f}"
        print(row)
    print("="*40)

def main():
    print("=" * 60)
    print("  第 3 章 蒙特卡洛策略评估 (MC Prediction)")
    print("  V(s) = 完整轨迹回报的经验平均")
    print("=" * 60)

    env = SimpleGridWorld()
    pi = np.ones(env.n_actions)/env.n_actions

    print(f"\n环境: 4×4 小网格世界")
    print(f"策略: 均匀随机 π(a|s) = {pi}")
    print(f"动作: 0=上, 1=右, 2=下, 3=左")
    print(f"奖励: 每步 -1, 到达 T 结束\n")

    # -- 批量mc --
    np.random.seed(42)
    V_batch,counts = mc_prediction_batch(env,pi,n_episodes=5000,gamma=1.0)
    print_values(env, V_batch, "批量 MC: 5000 条轨迹后 V(s)")
    print(f"各状态访问次数: {counts.astype(int)}")

    # ── 增量 MC ──
    np.random.seed(42)
    V_inc, N_inc = mc_prediction_incremental(env, pi, n_episodes=5000, gamma=1.0)
    print_values(env, V_inc, "增量 MC: 5000 条轨迹后 V(s) (结果应与批量一致)")

    # ── 验证一致性 ──
    diff = np.max(np.abs(V_batch - V_inc))
    print(f"\n批量 vs 增量 最大差异: {diff:.6f}")
    if diff < 1e-6:
        print("✓ 两者结果完全一致")

    print("\n" + "=" * 60)
    print("核心要点:")
    print("  1. MC 走完整个回合才更新 → 必须有终止状态")
    print("  2. 增量形式: V ← V + (1/N)(G - V)")
    print("  3. 不依赖模型 (无需知道 P 和 R)")
    print("  4. 无自举 (bootstrap): 使用真实 G_t，不是估计值")
    print("=" * 60)

if __name__ == "__main__":
    main()