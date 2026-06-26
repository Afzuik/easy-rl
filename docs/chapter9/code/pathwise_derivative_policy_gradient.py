"""
第 9 章：路径导数策略梯度的 DDPG 风格教学版。

这个文件是自包含脚本：
    - 不依赖 gymnasium；
    - 内置一个连续动作的 PointWorld 环境；
    - actor 是确定性策略 mu_theta(s)，直接输出连续动作；
    - critic 是 Q_w(s, a)，用于评价“在状态 s 执行动作 a 好不好”；
    - actor 通过 critic 对动作的梯度间接改进策略。

核心概念对应关系：
    mu_theta(s) ~= argmax_a Q_w(s, a)
    J(theta) = E_s[Q_w(s, mu_theta(s))]
    grad_theta J = E_s[dQ/da * dmu_theta/dtheta]

直观理解：
    离散动作策略梯度通常通过 log pi(a|s) 调整“动作概率”；
    连续动作里，动作本身是可微的网络输出，因此可以沿着
    Q(s, a) 对 a 上升最快的方向，反向更新 actor 参数。

运行：
    conda run -n base python docs/chapter9/code/pathwise_derivative_policy_gradient.py

快速检查：
    conda run -n base python docs/chapter9/code/pathwise_derivative_policy_gradient.py --episodes 20
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PointWorld:
    """一个一维连续动作控制环境。

    状态：
        当前点的位置，形状为 [1]。

    动作：
        一个连续位移，表示这一步向左或向右移动多少。
        环境会把动作裁剪到 [-max_action, max_action]。

    目标：
        尽量把点移动到 target_position 附近。

    奖励：
        reward = -abs(target_position - position)
        距离目标越近，惩罚越小；进入目标附近额外得到 +1。
    """

    def __init__(
        self,
        target_position: float = 0.8,
        max_action: float = 0.1,
        max_steps: int = 40,
        seed: int = 0,
    ):
        """初始化连续控制环境。"""

        self.target_position = target_position
        self.max_action = max_action
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.position = 0.0
        self.steps = 0

    def reset(self) -> np.ndarray:
        """开始新 episode。

        初始位置随机放在目标左侧，确保策略需要学习“向右移动”。
        """

        self.position = float(self.rng.uniform(-0.8, -0.2))
        self.steps = 0
        return self._state()

    def step(self, action: float) -> tuple[np.ndarray, float, bool]:
        """执行连续动作并返回 next_state、reward、done。

        即使 actor 输出很大的动作，环境也会做裁剪；
        这对应很多连续控制任务中的动作上下界。
        """

        self.steps += 1
        clipped_action = float(np.clip(action, -self.max_action, self.max_action))
        self.position = float(np.clip(self.position + clipped_action, -1.0, 1.0))
        distance = abs(self.target_position - self.position)
        reward = -distance
        done = self.steps >= self.max_steps or distance < 0.03
        if distance < 0.03:
            reward += 1.0
        return self._state(), reward, done

    def _state(self) -> np.ndarray:
        """把当前位置包装成神经网络输入需要的一维数组。"""

        return np.array([self.position], dtype=np.float32)


@dataclass
class ReplayItem:
    """经验回放池里保存的一条转移样本。"""

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    """固定容量的经验回放池。

    DDPG 风格算法通常是 off-policy 的：
    采样数据先放进 replay buffer，再从 buffer 中随机抽小批量训练。
    这样可以打乱时间相关性，并且让一条经验被多次利用。
    """

    def __init__(self, capacity: int):
        """创建最多保存 capacity 条转移的队列。"""

        self.data: deque[ReplayItem] = deque(maxlen=capacity)

    def append(self, item: ReplayItem) -> None:
        """加入一条交互样本；超过容量时最旧样本会自动被丢弃。"""

        self.data.append(item)

    def __len__(self) -> int:
        """返回当前已经存入的样本数量。"""

        return len(self.data)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从回放池随机采样一个 batch，并转换成 PyTorch 张量。

        返回的 rewards 和 dones 使用 [batch, 1] 形状，
        是为了和 critic 输出的 Q(s,a) 形状保持一致。
        """

        batch = random.sample(self.data, batch_size)
        states = torch.as_tensor(
            np.array([item.state for item in batch], dtype=np.float32)
        )
        actions = torch.as_tensor(
            np.array([item.action for item in batch], dtype=np.float32)
        )
        rewards = torch.as_tensor(
            np.array([item.reward for item in batch], dtype=np.float32)
        ).unsqueeze(-1)
        next_states = torch.as_tensor(
            np.array([item.next_state for item in batch], dtype=np.float32)
        )
        dones = torch.as_tensor(
            np.array([item.done for item in batch], dtype=np.float32)
        ).unsqueeze(-1)
        return states, actions, rewards, next_states, dones


class Actor(nn.Module):
    """确定性 actor：把状态直接映射成连续动作 mu_theta(s)。

    最后一层使用 Tanh，把网络原始输出限制在 [-1, 1]，
    再乘以 max_action，得到环境允许范围内的动作。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """输出连续动作。

        这里没有 Categorical 分布，也没有 log_prob；
        actor 的更新来自 critic 对动作的可微评价 Q(s, mu(s))。
        """

        return self.max_action * self.net(states)


class Critic(nn.Module):
    """动作价值函数 Q_w(s, a)。

    critic 同时接收状态和动作，输出这个状态-动作对的价值。
    在更新 actor 时，critic 会提供关于动作方向的梯度：
        如果某个动作方向能提高 Q，actor 就会被推向那个方向。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """计算 Q(s, a)。

        状态和动作沿最后一维拼接，因为 Q 函数需要同时知道
        “在哪个状态”和“做了哪个动作”。
        """

        return self.net(torch.cat([states, actions], dim=-1))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """软更新目标网络。

    target <- (1 - tau) * target + tau * source

    目标网络变化更慢，可以让 TD 目标更稳定。
    tau 越小，target 网络跟随 source 网络越慢。
    """

    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def train(args: argparse.Namespace) -> tuple[Actor, Critic]:
    """训练路径导数策略梯度模型。

    训练结构接近 DDPG：
        1. actor 根据状态输出确定性动作；
        2. 给动作加探索噪声后与环境交互；
        3. 把转移样本存入 replay buffer；
        4. critic 用 TD 目标学习 Q(s,a)；
        5. actor 最大化 critic 给出的 Q(s, mu(s))；
        6. 对 target_actor 和 target_critic 做软更新。
    """

    set_seed(args.seed)
    env = PointWorld(
        target_position=args.target_position,
        max_action=args.max_action,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    actor = Actor(1, 1, args.hidden_dim, args.max_action)
    critic = Critic(1, 1, args.hidden_dim)
    # target 网络用于构造更稳定的 TD 目标。
    # 初始时 target 和在线网络参数完全相同。
    target_actor = Actor(1, 1, args.hidden_dim, args.max_action)
    target_critic = Critic(1, 1, args.hidden_dim)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(args.buffer_size)
    reward_history: list[float] = []

    print("=" * 78)
    print("Pathwise derivative policy gradient: maximize Q(s, mu_theta(s))")
    print("Environment: continuous PointWorld")
    print("=" * 78)

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        last_actor_loss = 0.0
        last_critic_loss = 0.0

        for _ in range(args.max_steps):
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # actor 输出确定性动作 mu(s)。
                action = actor(state_t).squeeze(0).numpy()
            # 确定性策略本身不会随机探索，所以训练时人为加入高斯噪声。
            noisy_action = action + np.random.normal(0.0, args.exploration_noise, size=1)
            noisy_action = np.clip(noisy_action, -args.max_action, args.max_action).astype(
                np.float32
            )

            next_state, reward, done = env.step(float(noisy_action[0]))
            # 保存实际执行的 noisy_action，而不是原始 actor action。
            # critic 学习的是环境真实看到的状态-动作-奖励关系。
            replay.append(
                ReplayItem(
                    state=state,
                    action=noisy_action,
                    reward=reward,
                    next_state=next_state,
                    done=float(done),
                )
            )
            total_reward += reward
            state = next_state

            if len(replay) >= args.batch_size:
                # 从 replay buffer 随机取样，打破连续轨迹之间的强相关性。
                states, actions, rewards, next_states, dones = replay.sample(
                    args.batch_size
                )

                with torch.no_grad():
                    # TD 目标：
                    #   y = r + gamma * (1 - done) * Q_target(s', mu_target(s'))
                    # done=1 时没有后续价值。
                    next_actions = target_actor(next_states)
                    target_q = rewards + args.gamma * (1.0 - dones) * target_critic(
                        next_states, next_actions
                    )

                # critic 更新：让当前 Q(s,a) 逼近 TD 目标 y。
                q_values = critic(states, actions)
                critic_loss = F.mse_loss(q_values, target_q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # actor 更新：选择能让 critic 评价更高的动作。
                # 因为优化器默认最小化 loss，所以这里取负号：
                #   minimize -Q(s, mu(s))  等价于  maximize Q(s, mu(s))
                actor_actions = actor(states)
                actor_loss = -critic(states, actor_actions).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 在线网络更新后，缓慢推进目标网络。
                soft_update(target_actor, actor, args.tau)
                soft_update(target_critic, critic, args.tau)

                last_actor_loss = float(actor_loss.item())
                last_critic_loss = float(critic_loss.item())

            if done:
                break

        reward_history.append(total_reward)
        if episode == 1 or episode % args.print_every == 0:
            average_last = float(np.mean(reward_history[-20:]))
            with torch.no_grad():
                # 观察几个固定位置的动作输出，帮助判断 actor 是否学会向目标移动。
                probe_states = torch.tensor([[-0.6], [0.0], [0.6]], dtype=torch.float32)
                probe_actions = actor(probe_states).squeeze(-1).numpy()
            print(
                f"episode={episode:>4d} | "
                f"reward={total_reward:+.3f} | avg20={average_last:+.3f} | "
                f"actor_loss={last_actor_loss:+.4f} | critic_loss={last_critic_loss:.4f} | "
                f"mu(-0.6,0,0.6)={probe_actions.round(3).tolist()}"
            )

    print("\nTraining finished.")
    print(f"First 20 avg reward: {np.mean(reward_history[:20]):+.3f}")
    print(f"Last 20 avg reward:  {np.mean(reward_history[-20:]):+.3f}")
    return actor, critic


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--target-position", type=float, default=0.8)
    parser.add_argument("--max-action", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--exploration-noise", type=float, default=0.08)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
