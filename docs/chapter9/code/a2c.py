"""
第 9 章：优势演员-评论员算法 A2C 的单进程教学版。

这个文件刻意写成自包含脚本：
    - 不依赖 gymnasium，方便直接阅读和运行；
    - 内置一个很小的离散动作 LineWorld 环境；
    - actor 和 critic 共用前面的特征提取网络 trunk，再分成两个输出头。

核心概念对应关系：
    V(s) 由 critic 估计，表示“从状态 s 出发的未来期望回报”。
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    这里的 delta_t 同时扮演 TD 误差和一步 advantage 的角色。

    actor_loss  = -log pi(a_t|s_t) * delta_t
    critic_loss = delta_t^2

直观理解：
    - 如果 delta_t > 0，说明这个动作比 critic 预期更好，
      actor 会提高该动作的概率；
    - 如果 delta_t < 0，说明这个动作比预期更差，
      actor 会降低该动作的概率；
    - critic 用 delta_t^2 学会更准确地预测 V(s)。

运行：
    conda run -n base python docs/chapter9/code/a2c.py

快速检查：
    conda run -n base python docs/chapter9/code/a2c.py --episodes 20 --print-every 5
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def set_seed(seed: int) -> None:
    """固定随机种子，让同一组参数下的实验曲线尽量可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LineWorld:
    """一个用于讲解 actor-critic 的一维离散控制环境。

    环境是一条长度为奇数的线段，智能体从中间位置出发。
    动作空间只有两个动作：
        0 表示向左移动一格；
        1 表示向右移动一格。

    状态是当前位置经过归一化后的单个浮点数：
        最左端接近 -1，中点是 0，最右端接近 +1。

    奖励设计：
        到达右端得到 +1；
        到达左端得到 -1；
        每走一步都有 -0.01 的小惩罚，鼓励更短路径。
    """

    def __init__(self, length: int = 9, max_steps: int = 20):
        """初始化线性世界。

        length 必须是大于等于 3 的奇数，这样中间位置是唯一的，
        智能体每个 episode 都能从正中间开始。
        """

        if length < 3 or length % 2 == 0:
            raise ValueError("length must be an odd integer >= 3")
        self.length = length
        self.max_steps = max_steps
        self.position = length // 2
        self.steps = 0

    def reset(self) -> np.ndarray:
        """开始一个新 episode，把智能体放回线段中点。"""

        self.position = self.length // 2
        self.steps = 0
        return self._state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """执行一步环境转移。

        这里的 action 已经是离散整数：
            0 -> 左移；
            1 -> 右移。

        返回值是 next_state、reward、done。
        done 为 True 表示 episode 结束，原因可能是到达任意端点，
        也可能是超过最大步数。
        """

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
        """把离散位置归一化成神经网络更容易处理的连续状态。"""

        center = (self.length - 1) / 2.0
        return np.array([(self.position - center) / center], dtype=np.float32)


class ActorCritic(nn.Module):
    """共享 trunk 的 Actor-Critic 网络。

    这个网络先用 trunk 把状态编码成隐藏特征，然后分成两个 head：
        policy_head 输出每个离散动作的 logits，用于构造策略分布 pi(a|s)；
        value_head 输出一个标量 V(s)，用于估计状态价值。

    共享 trunk 的好处是 actor 和 critic 可以复用同一份状态表征；
    缺点是两个目标的梯度会共同影响 trunk，所以实际训练中需要用
    value_coef、entropy_coef、梯度裁剪等手段保持更新稳定。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        # trunk 只负责提取状态特征，不直接决定动作或价值。
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # policy_head 的输出是 logits，不需要手动做 softmax；
        # Categorical(logits=...) 内部会处理成合法概率分布。
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        # value_head 只输出一个数，对应当前状态的 V(s)。
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """同时计算策略 logits 和状态价值。

        states 的形状通常是 [batch, state_dim]。
        返回：
            logits: [batch, action_dim]，每个动作一个未归一化分数；
            values: [batch]，每个状态一个价值估计。
        """

        features = self.trunk(states)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据当前策略采样一个动作，并保留训练 actor 所需的信息。

        返回的 log_prob 用于策略梯度：
            -log pi(a_t|s_t) * advantage

        返回的 value 用于构造 TD 误差：
            r_t + gamma * V(s_{t+1}) - V(s_t)

        entropy 用于熵正则，鼓励策略不要过早变得确定。
        """

        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0), entropy.squeeze(0)


@dataclass
class EpisodeStats:
    """保存单个 episode 的训练统计，便于统一打印日志。"""

    episode: int
    reward: float
    actor_loss: float
    critic_loss: float
    average_last: float


def train(args: argparse.Namespace) -> ActorCritic:
    """训练 A2C 模型。

    这个版本是最直接的一步 TD A2C：
        1. 用 actor 根据当前状态采样动作；
        2. 环境返回 reward 和 next_state；
        3. critic 估计 V(s) 和 V(s_{t+1})；
        4. 用 TD 目标计算 advantage；
        5. actor 根据 advantage 调整动作概率；
        6. critic 根据平方误差学习状态价值。
    """

    set_seed(args.seed)
    env = LineWorld(length=args.length, max_steps=args.max_steps)
    model = ActorCritic(state_dim=1, action_dim=2, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reward_history: list[float] = []

    print("=" * 78)
    print("A2C: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)")
    print("Environment: LineWorld, actions: 0=left, 1=right")
    print("=" * 78)

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        actor_losses: list[torch.Tensor] = []
        critic_losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        while not done:
            # 采样动作时同时拿到 log_prob、V(s) 和 entropy。
            # log_prob 服务 actor_loss，value 服务 critic_loss，entropy 服务探索。
            action, log_prob, value, entropy = model.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            with torch.no_grad():
                # 如果 episode 已经结束，就没有后续状态价值；
                # 否则用当前 critic 估计 V(s_{t+1}) 作为 bootstrap 项。
                if done:
                    next_value = torch.tensor(0.0)
                else:
                    next_state_t = torch.as_tensor(
                        next_state, dtype=torch.float32
                    ).unsqueeze(0)
                    _, next_value_batch = model(next_state_t)
                    next_value = next_value_batch.squeeze(0)

            # 一步 TD 目标：
            #   target = r_t + gamma * V(s_{t+1})
            # advantage = target - V(s_t)
            # 在 A2C 中，这个 advantage 用来告诉 actor：
            # “刚才采样的动作比预期好还是差”。
            td_target = torch.tensor(reward, dtype=torch.float32) + args.gamma * next_value
            advantage = td_target - value

            # actor 更新策略时只把 advantage 当作权重，不反向更新 critic，
            # 所以这里对 advantage 做 detach。
            actor_losses.append(-log_prob * advantage.detach())
            # critic 直接最小化 TD 误差平方。
            critic_losses.append(advantage.pow(2))
            # 熵越大，策略越随机；在总 loss 中减去 entropy_bonus，
            # 等价于鼓励策略保持一定探索。
            entropies.append(entropy)

            state = next_state

        # 这个教学实现按 episode 聚合 loss，然后统一反向传播一次。
        actor_loss = torch.stack(actor_losses).sum()
        critic_loss = torch.stack(critic_losses).sum()
        entropy_bonus = torch.stack(entropies).sum()
        loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪可以避免某些 episode 中 TD 误差过大导致参数更新失控。
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        reward_history.append(total_reward)
        average_last = float(np.mean(reward_history[-20:]))
        stats = EpisodeStats(
            episode=episode,
            reward=total_reward,
            actor_loss=float(actor_loss.item()),
            critic_loss=float(critic_loss.item()),
            average_last=average_last,
        )

        if episode == 1 or episode % args.print_every == 0:
            print(
                f"episode={stats.episode:>4d} | "
                f"reward={stats.reward:+.2f} | "
                f"avg20={stats.average_last:+.2f} | "
                f"actor_loss={stats.actor_loss:+.4f} | "
                f"critic_loss={stats.critic_loss:.4f}"
            )

    print("\nTraining finished.")
    print(f"First 20 avg reward: {np.mean(reward_history[:20]):+.3f}")
    print(f"Last 20 avg reward:  {np.mean(reward_history[-20:]):+.3f}")
    return model


def parse_args() -> argparse.Namespace:
    """解析命令行参数，默认值偏向教学演示而不是最佳性能。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--length", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
