"""
第 2 章 马尔可夫决策过程 —— 策略迭代（Policy Iteration）演示代码

本代码对应 docs/chapter2/chapter2_order.md 中 2.18「策略迭代」：

1. 策略评估：给定当前策略 pi，反复执行贝尔曼期望备份，求 V_pi。
2. 策略改进：根据 V_pi 计算 Q_pi(s, a)，对每个状态选择 Q 最大的动作。

运行方式：
  python docs/chapter2/policy_iteration.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class StepResult:
    next_state: int | None
    reward: float


class SmallGridWorld:
    """
    4x4 小网格世界，对应第 2 章中的 small gridworld。

    状态布局：
        T   1   2   3
        4   5   6   7
        8   9  10  11
       12  13  14   T

    T 是终止状态，不参与策略迭代；1~14 是非终止状态。
    每移动一步奖励为 -1，越快到达终止状态越好。
    """

    def __init__(self) -> None:
        self.rows = 4
        self.cols = 4
        self.terminal_coords = {(0, 0), (3, 3)}
        self.step_reward = -1.0

        # 动作顺序固定，便于策略数组中的动作编号保持一致。
        self.actions = {
            0: ("↑", -1, 0),
            1: ("→", 0, 1),
            2: ("↓", 1, 0),
            3: ("←", 0, -1),
        }

        self.state_coords: list[tuple[int, int]] = []
        self.coord_to_state: dict[tuple[int, int], int] = {}

        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) in self.terminal_coords:
                    continue
                state = len(self.state_coords)
                self.state_coords.append((row, col))
                self.coord_to_state[(row, col)] = state

        self.n_states = len(self.state_coords)
        self.n_actions = len(self.actions)

    def step(self, state: int, action: int) -> StepResult:
        """执行动作，返回下一状态和即时奖励。next_state 为 None 表示到达终止状态。"""
        row, col = self.state_coords[state]
        _, d_row, d_col = self.actions[action]
        next_row = row + d_row
        next_col = col + d_col

        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            return StepResult(next_state=state, reward=self.step_reward)

        if (next_row, next_col) in self.terminal_coords:
            return StepResult(next_state=None, reward=self.step_reward)

        return StepResult(
            next_state=self.coord_to_state[(next_row, next_col)],
            reward=self.step_reward,
        )

    def action_symbol(self, action: int) -> str:
        return self.actions[action][0]

    def print_values(self, values: list[float], title: str) -> None:
        print(f"\n{title}")
        print("=" * 40)
        for row in range(self.rows):
            cells = []
            for col in range(self.cols):
                if (row, col) in self.terminal_coords:
                    cells.append(f"{'T':>8}")
                else:
                    state = self.coord_to_state[(row, col)]
                    cells.append(f"{values[state]:>8.2f}")
            print("".join(cells))
        print("=" * 40)

    def print_policy(self, policy: list[list[float]], title: str) -> None:
        print(f"\n{title}")
        print("=" * 24)
        for row in range(self.rows):
            cells = []
            for col in range(self.cols):
                if (row, col) in self.terminal_coords:
                    cells.append(f"{'T':>5}")
                else:
                    state = self.coord_to_state[(row, col)]
                    best_action = max(
                        range(self.n_actions),
                        key=lambda action: policy[state][action],
                    )
                    cells.append(f"{self.action_symbol(best_action):>5}")
            print("".join(cells))
        print("=" * 24)


def q_value(env: SmallGridWorld, values: list[float], state: int, action: int, gamma: float) -> float:
    """
    Q_pi(s, a) = R(s, a) + gamma * sum_s' P(s'|s,a) V_pi(s')

    当前环境是确定性转移，所以 sum_s' 只剩下一个下一状态。
    """
    result = env.step(state, action)
    next_value = 0.0 if result.next_state is None else values[result.next_state]
    return result.reward + gamma * next_value


def policy_evaluation(
    env: SmallGridWorld,
    policy: list[list[float]],
    gamma: float = 1.0,
    theta: float = 1e-6,
    max_iterations: int = 1000,
) -> tuple[list[float], int]:
    """
    策略评估：固定当前策略 pi，计算它对应的状态价值函数 V_pi。

    贝尔曼期望备份：
        V(s) <- sum_a pi(a|s) [R(s,a) + gamma * V(s')]
    """
    values = [0.0 for _ in range(env.n_states)]

    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        new_values = values.copy()

        for state in range(env.n_states):
            old_value = values[state]
            new_values[state] = sum(
                action_prob * q_value(env, values, state, action, gamma)
                for action, action_prob in enumerate(policy[state])
            )
            delta = max(delta, abs(old_value - new_values[state]))

        values = new_values
        if delta < theta:
            return values, iteration

    return values, max_iterations


def policy_improvement(
    env: SmallGridWorld,
    values: list[float],
    old_policy: list[list[float]],
    gamma: float = 1.0,
) -> tuple[list[list[float]], bool]:
    """
    策略改进：对每个状态执行 argmax_a Q_pi(s, a)。

    返回：
      new_policy: 贪心改进后的策略
      stable:     如果策略没有变化，说明已收敛到最优策略
    """
    new_policy = [action_probs.copy() for action_probs in old_policy]
    stable = True

    for state in range(env.n_states):
        old_best_action = max(
            range(env.n_actions),
            key=lambda action: old_policy[state][action],
        )
        action_values = [
            q_value(env, values, state, action, gamma)
            for action in range(env.n_actions)
        ]
        best_action = max(range(env.n_actions), key=lambda action: action_values[action])

        new_policy[state] = [
            1.0 if action == best_action else 0.0
            for action in range(env.n_actions)
        ]
        if best_action != old_best_action or old_policy[state][best_action] != 1.0:
            stable = False

    return new_policy, stable


def policy_iteration(
    env: SmallGridWorld,
    gamma: float = 1.0,
    theta: float = 1e-6,
    max_policy_iterations: int = 100,
) -> tuple[list[float], list[list[float]], int]:
    """
    策略迭代完整流程：

    1. 初始化一个任意策略。
    2. 重复执行：
       - 策略评估，求当前策略的 V_pi。
       - 策略改进，用 argmax_a Q_pi(s,a) 得到新策略。
    3. 当策略不再变化时停止。
    """
    policy = [
        [1.0 / env.n_actions for _ in range(env.n_actions)]
        for _ in range(env.n_states)
    ]

    for policy_iteration_idx in range(1, max_policy_iterations + 1):
        values, eval_iterations = policy_evaluation(
            env=env,
            policy=policy,
            gamma=gamma,
            theta=theta,
        )
        new_policy, stable = policy_improvement(
            env=env,
            values=values,
            old_policy=policy,
            gamma=gamma,
        )

        changed_states = sum(
            old_action_probs != new_action_probs
            for old_action_probs, new_action_probs in zip(policy, new_policy)
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
    gamma = 1.0
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
