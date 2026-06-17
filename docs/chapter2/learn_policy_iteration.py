from __future__ import annotations
import sys
from dataclasses import dataclass
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 为接下来的 SmallGridWorld 环境服务，用于封装环境 step() 方法的返回值。
@dataclass(frozen=True)
class StepResult:
    next_state: int | None
    reward:float

class SmallGridWorld:
    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.terminal_coords = {(0,0),(3,3)}
        self.step_reward = -1.0

        self.actions = {
            0: ("↑", -1, 0),
            1: ("→", 0, 1),
            2: ("↓", 1, 0),
            3: ("←", 0, -1),
        }
        
        self.state_coords: list[tuple[int,int]] = []
        self.coord_to_state: dict[tuple[int,int]] = {}

        for row in range(self.rows):
            for col in range(self.cols):
                if (row,col) in self.terminal_coords:
                    continue
                state = len(self.state_coords)
                self.state_coords.append((row,col))
                self.coord_to_state[(row,col)] = state
        
        self.n_states = len(self.state_coords)
        self.n_actions = len(self.actions)

    def step(self,state:int,action:int) -> StepResult:
        row,col = self.state_coords[state]
        _,d_row,d_col = self.actions[action]
        next_row = row + d_row
        next_col = col + d_col

        if not (0<=next_row<self.rows and 0<=next_col<self.cols):
            return StepResult(next_state=state,reward=self.step_reward)
        
        if (next_row,next_col) in self.terminal_coords:
            return StepResult(next_state=None,reward=self.step_reward)
        
        return StepResult(
            next_state=self.coord_to_state[(next_row,next_col)],
            reward=self.step_reward
        )
    
    def action_symbol(self,action:int) -> str:
        return self.actions[action][0]
    
    def print_values(self,values:list[float],title:str) -> None:
        print(f"\n{title}")
        print("="*40)
        for row in range(self.rows):
            cells = []
            for col in range(self.cols):
                if (row,col) in self.terminal_coords:
                    cells.append(f"{'T':>8}")
                else:
                    state = self.coord_to_state[(row,col)]
                    cells.append(f"{values[state]:>8.2f}")
            print("".join(cells))
            print("="*40)
    
    def print_policy(self,policy:list[list[float]],title:str)->None:
        print(f"\n{title}")
        print("="*24)
        for row in range(self.rows):
            cells = []
            for col in range(self.cols):
                if (row,col) in self.terminal_coords:
                    cells.append(f"{'T':>5}")
                else:
                    state = self.coord_to_state[(row,col)]
                    best_action = max(
                        range(self.n_actions),
                        key = lambda action:policy[state][action],
                    )
                    cells.append(f"{self.action_symbol(best_action):>5}")
            print("".join(cells))
        print("="*24)


def q_value(env:SmallGridWorld,values:list[list[float]],state:int,action:int,gamma:float)->float:
    result = env.step(state,action)
    next_value = 0.0 if result.next_state is None else values[result.next_state]
    return result.reward + gamma * next_value


def policy_evaluation(
        env:SmallGridWorld,
        policy:list[list[float]],
        gamma:float=1.0,
        theta:float=1e-6,
        max_iterations:int=1000,
)->tuple[list[float],int]:
    '''
    确定性策略
    '''
    values = [0.0 for _ in range(env.n_states)]
    
    for iteration in range(1,max_iterations+1):
        delta = 0.0
        new_values = values.copy()

        for state in range(env.n_states):
            old_value = values[state]
            new_values[state] = sum(
               action_prob*q_value(env,values,state,action,gamma) for action,action_prob in enumerate(policy[state])
            )
            delta = max(delta,abs(old_value-new_values[state]))

        values = new_values
        if delta<theta:
            return values,iteration
    
    return values,max_iterations

def policy_improvement(
        env:SmallGridWorld,
        values:list[float],
        old_policy:list[list[float]],
        gamma:float=1.0,
)->tuple[list[list[float]],bool]:
    new_policy = [action_probs.copy() for action_probs in old_policy]
    stable = True
    for state in range(env.n_states):
        old_best_action = max(
            range(env.n_actions),
            key = lambda action:old_policy[state][action],
        )
        action_values = [
            q_value(env, values, state, action, gamma)
            for action in range(env.n_actions)
        ]
        best_action = max(
            range(env.n_actions),
            key = lambda action:action_values[action]
        )
        new_policy[state] = [
            1.0 if action == best_action else 0.0
            for action in range(env.n_actions)
        ]
        if best_action != old_best_action or old_policy[state][best_action] != 1.0:
            state = False
    return new_policy,stable

def policy_iteration(
        env:SmallGridWorld,
        gamma:float = 1.0,
        theta:float = 1e-6,
        max_policy_iterations:int = 100,
):
    policy = [
        [1.0 / env.n_actions for _ in range(env.n_actions)]
        for _ in range(env.n_states)
    ]

    for policy_iteration_idx in range(1,max_policy_iterations+1):
        values,eval_iterations = policy_evaluation(
            env=env,
            policy=policy,
            gamma=gamma,
            theta=theta,
        )
        new_policy,stable = policy_improvement(
            env=env,
            values=values,
            old_policy=policy,
            gamma=gamma
        )

        changed_states = sum(
            old_action_probs != new_action_probs
            for old_action_probs,new_action_probs in zip(policy,new_policy)
        )
        print(
            f"第 {policy_iteration_idx:>2} 轮："
            f"策略评估 {eval_iterations:>3} 次，"
            f"策略改变 {changed_states:>2} 个状态"
        )
        policy = new_policy
        if stable:
            return values, policy, policy_iteration_idx

    return values, policy, max_policy_iterations

def main() -> None:
    env = SmallGridWorld()
    gamma = 0.6
    theta = 1e-6

    print("=" * 60)
    print("第 2 章 MDP：策略迭代算法（Policy Iteration）")
    print("=" * 60)
    print(f"环境：4x4 small gridworld，非终止状态数 = {env.n_states}")
    print(f"动作：↑=上，→=右，↓=下，←=左；每走一步奖励 = {env.step_reward}")
    print(f"参数：gamma = {gamma}, theta = {theta}")

    values, policy, rounds = policy_iteration(env, gamma=gamma, theta=theta)

    print(f"\n策略迭代在第 {rounds} 轮收敛。")
    env.print_values(values, "最优状态价值 V*(s)")
    env.print_policy(policy, "最优策略 pi*(s)")


if __name__ == "__main__":
    main()