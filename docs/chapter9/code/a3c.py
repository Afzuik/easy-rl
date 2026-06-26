"""
第 9 章：异步优势演员-评论员算法 A3C 的教学版。

这个文件是自包含脚本，内置一个很小的 LineWorld 环境。
A3C 的重点不在环境复杂度，而在“多个 worker 并行采样、共同更新全局网络”。

每个 worker 的流程：
    1. 从 global_model 同步一份参数到 local_model；
    2. 用 local_model 和自己的环境交互，收集 n-step rollout；
    3. 在 local_model 上计算 actor loss、critic loss 和 entropy bonus；
    4. 把 local_model 的梯度拷贝给 global_model；
    5. 在锁保护下更新 global_model；
    6. 再次同步最新的 global_model 参数。

这个实现为了教学清晰，使用 lock 保护全局参数更新。
经典 A3C 常强调异步更新带来的 decorrelation，这里保留核心思想，
但避免把代码写得过于工程化。

运行：
    conda run -n base python docs/chapter9/code/a3c.py

快速检查：
    conda run -n base python docs/chapter9/code/a3c.py --workers 2 --episodes-per-worker 10
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def set_seed(seed: int) -> None:
    """固定每个进程的随机种子。

    worker 会使用 seed + worker_id，避免所有 worker 产生完全相同的采样轨迹。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LineWorld:
    """一个一维离散动作环境，用来观察 A3C 的训练流程。

    智能体从线段中点出发：
        动作 0 向左；
        动作 1 向右。

    右端是正目标，奖励 +1；左端是失败终点，奖励 -1。
    每一步有 -0.01 的时间成本，让策略倾向于尽快到达右端。
    """

    def __init__(self, length: int = 9, max_steps: int = 20):
        """创建线性世界。

        length 使用奇数，是为了让起点可以严格位于线段正中间。
        """

        if length < 3 or length % 2 == 0:
            raise ValueError("length must be an odd integer >= 3")
        self.length = length
        self.max_steps = max_steps
        self.position = length // 2
        self.steps = 0

    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态。"""

        self.position = self.length // 2
        self.steps = 0
        return self._state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """执行一个离散动作，返回 next_state、reward、done。"""

        self.steps += 1
        move = -1 if action == 0 else 1
        self.position = int(np.clip(self.position + move, 0, self.length - 1))
        done = (
            self.position == 0
            or self.position == self.length - 1
            or self.steps >= self.max_steps
        )
        reward = -0.01
        if self.position == self.length - 1:
            reward = 1.0
        elif self.position == 0:
            reward = -1.0
        return self._state(), reward, done

    def _state(self) -> np.ndarray:
        """把当前位置映射到 [-1, 1] 附近的单维连续状态。"""

        center = (self.length - 1) / 2.0
        return np.array([(self.position - center) / center], dtype=np.float32)


class ActorCritic(nn.Module):
    """A3C 使用的共享 trunk Actor-Critic 网络。

    trunk 负责提取状态特征；
    policy_head 输出动作 logits，用来构造 Categorical 策略；
    value_head 输出 V(s)，用来计算 n-step return 和 advantage。

    A3C 中每个 worker 都有自己的 local_model，
    但所有 worker 最终都把梯度应用到同一个 global_model。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        # 共享特征层：actor 和 critic 都基于这一份状态表示做预测。
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # 策略头输出 logits，不直接输出动作。
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        # 价值头输出一个标量 V(s)。
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回策略 logits 和状态价值。

        logits 用于构造动作分布 pi(a|s)；
        values 用于计算 return - V(s) 形式的 advantage。
        """

        features = self.trunk(states)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


@dataclass
class Transition:
    """worker 在 n-step rollout 中保存的一步交互数据。"""

    state: np.ndarray
    action: int
    reward: float
    done: bool


def choose_action(model: ActorCritic, state: np.ndarray) -> int:
    """用当前 local_model 采样动作。

    这里只负责采样，不保存 log_prob。
    后面 compute_loss 会把一段 transitions 重新送入网络，
    一次性计算这批动作的 log_prob、entropy 和 value。
    """

    state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state_t)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
    return int(action.item())


def compute_loss(
    model: ActorCritic,
    transitions: list[Transition],
    bootstrap_value: torch.Tensor,
    gamma: float,
    value_coef: float,
    entropy_coef: float,
) -> torch.Tensor:
    """根据一段 n-step rollout 计算 A3C 的总损失。

    A3C 不等到完整 episode 结束才更新，而是每收集 n 步就更新一次。
    如果这 n 步后 episode 尚未结束，就用 bootstrap_value 近似尾部价值：

        R_t = r_t + gamma * r_{t+1} + ... + gamma^n * V(s_{t+n})

    如果 episode 已经结束，bootstrap_value 为 0。
    """

    returns: list[torch.Tensor] = []
    running_return = bootstrap_value
    # 从后往前递推 n-step return。
    # 反向计算可以复用 running_return，避免显式写多重折扣求和。
    for transition in reversed(transitions):
        if transition.done:
            # episode 终止后没有未来价值，return 从终止点重新开始。
            running_return = torch.tensor(0.0)
        running_return = torch.tensor(transition.reward, dtype=torch.float32) + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    # returns_t 是 critic 的监督目标，不需要对它反向传播。
    returns_t = torch.stack(returns).detach()

    # 把 rollout 列表整理成批量张量，方便一次前向计算。
    states = torch.as_tensor(
        np.array([t.state for t in transitions], dtype=np.float32),
        dtype=torch.float32,
    )
    actions = torch.as_tensor([t.action for t in transitions], dtype=torch.long)

    logits, values = model(states)
    distribution = torch.distributions.Categorical(logits=logits)
    log_probs = distribution.log_prob(actions)
    entropies = distribution.entropy()
    # advantage 告诉 actor 当前动作比 critic 预期好多少。
    advantages = returns_t - values

    # actor loss 使用 detach 后的 advantage：
    # actor 只根据 advantage 调整动作概率，不通过这条路径更新 critic。
    actor_loss = -(log_probs * advantages.detach()).mean()
    # critic 通过均方误差拟合 n-step return。
    critic_loss = F.mse_loss(values, returns_t)
    # 熵正则鼓励探索，防止策略过早塌缩到确定动作。
    entropy = entropies.mean()
    return actor_loss + value_coef * critic_loss - entropy_coef * entropy


def worker_process(
    worker_id: int,
    global_model: ActorCritic,
    lock,
    result_queue,
    args: argparse.Namespace,
) -> None:
    """单个 A3C worker 的主循环。

    每个 worker 有独立环境和 local_model。
    local_model 用来采样和算梯度；
    global_model 保存所有 worker 共享的最新参数。

    result_queue 只负责把训练日志传回主进程打印。
    """

    set_seed(args.seed + worker_id)
    env = LineWorld(length=args.length, max_steps=args.max_steps)
    local_model = ActorCritic(1, 2, args.hidden_dim)
    # optimizer 绑定 global_model，因为最终被更新的是全局参数。
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr)
    rewards: list[float] = []

    for episode in range(1, args.episodes_per_worker + 1):
        # 每个 episode 开始时先拉取最新的全局参数。
        local_model.load_state_dict(global_model.state_dict())
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            transitions: list[Transition] = []
            # 收集最多 n_steps 步。如果 episode 提前结束，就立即停止收集。
            for _ in range(args.n_steps):
                action = choose_action(local_model, state)
                next_state, reward, done = env.step(action)
                transitions.append(
                    Transition(
                        state=state,
                        action=action,
                        reward=reward,
                        done=done,
                    )
                )
                total_reward += reward
                state = next_state
                if done:
                    break

            with torch.no_grad():
                # n-step rollout 截断时，需要用 V(s_{t+n}) 估计剩余回报；
                # 如果已经 done，则未来价值为 0。
                if done:
                    bootstrap_value = torch.tensor(0.0)
                else:
                    next_state_t = torch.as_tensor(
                        state, dtype=torch.float32
                    ).unsqueeze(0)
                    _, next_value = local_model(next_state_t)
                    bootstrap_value = next_value.squeeze(0)

            # 在 local_model 上计算损失和梯度。
            loss = compute_loss(
                local_model,
                transitions,
                bootstrap_value,
                gamma=args.gamma,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
            )
            optimizer.zero_grad()
            local_model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), args.max_grad_norm)

            with lock:
                # 把 local_model 的梯度拷贝到 global_model。
                # 注意这里不是拷贝参数，而是拷贝梯度，然后 optimizer.step()
                # 作用在 global_model.parameters() 上。
                optimizer.zero_grad()
                for global_param, local_param in zip(
                    global_model.parameters(), local_model.parameters()
                ):
                    if local_param.grad is not None:
                        global_param.grad = local_param.grad.detach().clone()
                optimizer.step()

            # 全局参数更新后，worker 重新同步，避免本地参数落后太多。
            local_model.load_state_dict(global_model.state_dict())

        rewards.append(total_reward)
        if episode == 1 or episode % args.print_every == 0:
            result_queue.put(
                (
                    worker_id,
                    episode,
                    float(total_reward),
                    float(np.mean(rewards[-10:])),
                )
            )

    result_queue.put((worker_id, "done", float(np.mean(rewards[-10:])), len(rewards)))


def train(args: argparse.Namespace) -> ActorCritic:
    """启动多个 worker，并返回训练后的全局 Actor-Critic 网络。"""

    set_seed(args.seed)
    # Windows 下多进程通常使用 spawn；显式指定能减少跨平台差异。
    ctx = get_context("spawn")
    global_model = ActorCritic(1, 2, args.hidden_dim)
    # share_memory 让多个子进程可以访问同一个 global_model 参数存储。
    global_model.share_memory()
    lock = ctx.Lock()
    result_queue = ctx.Queue()

    print("=" * 78)
    print("A3C: multiple workers asynchronously update one global actor-critic")
    print(
        f"workers={args.workers}, episodes_per_worker={args.episodes_per_worker}, "
        f"n_steps={args.n_steps}"
    )
    print("=" * 78)

    processes = []
    for worker_id in range(args.workers):
        # 每个 worker 是一个独立进程，有自己的环境和 local_model。
        process = ctx.Process(
            target=worker_process,
            args=(worker_id, global_model, lock, result_queue, args),
        )
        process.start()
        processes.append(process)

    finished = 0
    while finished < args.workers:
        # 主进程不直接参与训练，只从队列接收 worker 的进度消息。
        message = result_queue.get()
        worker_id, episode, reward, average = message
        if episode == "done":
            finished += 1
            print(
                f"worker={worker_id} done | "
                f"last10_avg={reward:+.2f} | episodes={average}"
            )
        else:
            print(
                f"worker={worker_id} | episode={episode:>4d} | "
                f"reward={reward:+.2f} | local_avg10={average:+.2f}"
            )

    for process in processes:
        # 等待所有子进程干净退出，避免后台进程残留。
        process.join()

    print("\nTraining finished.")
    return global_model


def parse_args() -> argparse.Namespace:
    """解析 A3C 教学脚本的命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes-per-worker", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--length", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
