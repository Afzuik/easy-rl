import sys
import numpy as np

if hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class CliffWalking:
    def __init__(self):
        self.rows,self.cols = 4,12
        self.start = (3,0)
        self.goal = (3,11)
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        
    def reset(self):
        self.state = self.start
        return self._idx(self.state)
    
    def _idx(self,pos):
        return pos[0]*self.cols+pos[1]
    
    def step(self,a):
        r,c = self.state
        if a == 0:
            r = max(r-1,0)
        elif a == 1:
            c = min(c+1,self.cols-1)
        elif a == 2:
            r = min(r+1,self.rows-1)
        elif a == 3:
            c = max(c-1,0)
        if r == 3 and 1<=c<=10:
            self.state = self.start
            return self._idx(self.state),-100,False
        self.state = (r,c)
        done = (self.state == self.goal)
        return self._idx(self.state),-1,done
    
def mc_prediction(env,policy,num_episodes=5000,gamma=1.0,alpha=None,max_steps=200):
    """_summary_

    Args:
        env (_type_): _description_
        policy (_type_): shape=(n_states,n_actions)
        num_episodes (int, optional): _description_. Defaults to 5000.
        gamma (float, optional): _description_. Defaults to 1.0.
        alpha (_type_, optional): None 表示用 1/N(s)原始增量MC，否则用固定学习率. Defaults to None.
        max_steps (int, optional): _description_. Defaults to 200.
    """
    V = np.zeros(env.n_states)
    N = np.zeros(env.n_states) # 访问计数

    for ep in range(num_episodes):
        traj = [] # list of (s,r)
        s = env.reset()
        for _ in range(max_steps):
            a = np.random.choice(env.n_actions,p=policy[s])
            s_next,r,done = env.step(a)
            traj.append((s,r))
            s = s_next
            if done:
                break
        
        G = 0.0
        for s_t,r_t in reversed(traj):
            G = r_t + gamma * G
            N[s_t] += 1
            lr = (1.0 / N[s_t]) if alpha is None else alpha
            V[s_t] += lr * (G-V[s_t])

        if (ep+1)%1000 == 0:
            print(f"    episode{ep+1:5d}/{num_episodes},V[start]={V[env._idx(env.start)]:.2f}")
        
    return V

if __name__ == "__main__":
    np.random.seed(42)
    env = CliffWalking()
    policy = np.ones((env.n_states, env.n_actions)) / env.n_actions

    print("[MC 预测] 在悬崖行走环境中估计均匀随机策略的 V 函数...")
    # 注意：使用 γ=0.9。若 γ=1，随机策略会在悬崖反弹中无限累积负值。
    V = mc_prediction(env, policy, num_episodes=5000, gamma=0.9, alpha=None)

    print("\n学到的 V 表（越接近 0 表示越接近终点 G）：")
    print(np.round(V.reshape(env.rows, env.cols), 1))
    print(f"\n起点 S 的价值估计 = {V[env._idx(env.start)]:.2f}")
    print("（随机策略下大概率会反复掉悬崖，因此价值很低，符合直觉。）")
