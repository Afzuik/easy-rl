"""第 5 章策略优化算法共用的环境、网络和采样工具。"""

from __future__ import annotations

import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class RolloutBatch:
    """一批由旧策略采集并冻结的数据。

    PPO/TRPO 的共同特点是：先用当前策略采样一批数据，把它视为
    ``old_policy`` 产生的数据；随后在这批固定数据上做多轮优化。
    因此这里保存的不只是 state/action/reward，还要保存采样当时
    的 log_prob 和动作概率分布，用于后续计算:

    - probability ratio: pi_new(a|s) / pi_old(a|s)
    - KL(old_policy || new_policy)
    """

    states: torch.Tensor
    actions: torch.Tensor
    # 采样时旧策略对实际动作 a_t 的 log pi_old(a_t|s_t)。
    old_log_probs: torch.Tensor
    # 采样时旧策略在每个状态下的完整动作分布，用于 KL 计算。
    old_action_probs: torch.Tensor
    # GAE 标准化后的优势函数 A_t。
    advantages: torch.Tensor
    # Critic 的监督目标，等于 V(s_t) + A_t（未标准化前的优势）。
    returns: torch.Tensor


class ActorCritic(nn.Module):
    """离散动作空间的 Actor-Critic 网络。

    这个网络没有共享 trunk，而是分别用 actor/critic 两个 MLP。
    - actor 输出每个离散动作的 logits，再构造成 Categorical 分布；
    - critic 输出状态价值 V(s)，用于 GAE 和 value loss。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def distribution(self, states: torch.Tensor) -> Categorical:
        """返回策略分布 pi(.|s)，后续可采样动作或计算 log_prob。"""
        return Categorical(logits=self.actor(states))

    def value(self, states: torch.Tensor) -> torch.Tensor:
        """返回 V(s)。squeeze(-1) 把形状从 [batch, 1] 变成 [batch]。"""
        return self.critic(states).squeeze(-1)

    def act(self, state: np.ndarray) -> tuple[int, float, np.ndarray, float]:
        """用当前策略采样一步动作，并冻结采样时的旧策略信息。

        返回值中 action/log_prob/action_probs/value 都来自同一时刻的
        当前策略。采样完成后，这些量会被放进 RolloutBatch；等后面
        更新参数时，它们就成为 old_policy 的固定参考值。
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            distribution = self.distribution(state_tensor)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            probs = distribution.probs
            value = self.value(state_tensor)
        return (
            int(action.item()),
            float(log_prob.item()),
            probs.squeeze(0).cpu().numpy(),
            float(value.item()),
        )


def set_seed(seed: int) -> None:
    """固定 Python、NumPy、Torch 的随机种子，便于复现实验曲线。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int) -> gym.Env:
    """创建 CartPole 环境并固定动作空间随机种子。"""
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    return env


def collect_rollout(
    env: gym.Env,
    model: ActorCritic,
    state: np.ndarray,
    episode_return: float,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[RolloutBatch, np.ndarray, float, list[float]]:
    """
    使用当前策略收集固定步数的数据，并用 GAE 计算优势。

    采样完成后，old_log_probs 和 old_action_probs 都会被冻结，
    供后续多个 epoch 计算新旧策略的概率比值或 KL 散度。
    """
    # 这些列表逐步收集 rollout 中每一个时间步的数据。注意 PPO/TRPO
    # 的更新单位不是单个 episode，而是一段固定长度 rollout；这段
    # rollout 里可能跨越多个 episode。
    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[float] = []
    values: list[float] = []
    next_values: list[float] = []
    old_log_probs: list[float] = []
    old_action_probs: list[np.ndarray] = []
    completed_returns: list[float] = []

    for _ in range(rollout_steps):
        # 用当前策略采样动作。此时的策略稍后会成为 pi_old，所以
        # 这里必须立刻记录 log_prob 和完整 action_probs。
        action, log_prob, action_probs, value = model.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 如果 episode 结束，下一状态没有未来价值；否则用当前 critic
        # 估计 V(s_{t+1})，给 GAE 的 TD error 使用。
        if done:
            next_value = 0.0
        else:
            next_state_tensor = torch.as_tensor(
                next_state, dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                next_value = float(model.value(next_state_tensor).item())

        states.append(np.asarray(state, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        dones.append(float(done))
        values.append(value)
        next_values.append(next_value)
        old_log_probs.append(log_prob)
        old_action_probs.append(action_probs.astype(np.float32))

        # 维护 episode_return 的原因是 rollout 可能在 episode 中间截断。
        # 未结束的 episode 回报要跨 rollout 延续，不能在每批数据结束时清零。
        episode_return += reward
        if done:
            completed_returns.append(episode_return)
            episode_return = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    # GAE(Generalized Advantage Estimation):
    #   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    #   A_t = delta_t + gamma*lambda*delta_{t+1} + ...
    # 从后往前递推可以一次算完所有时间步的优势。
    advantages = np.zeros(rollout_steps, dtype=np.float32)
    gae = 0.0
    for index in reversed(range(rollout_steps)):
        not_done = 1.0 - dones[index]
        delta = (
            rewards[index]
            + gamma * next_values[index] * not_done
            - values[index]
        )
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[index] = gae

    values_array = np.asarray(values, dtype=np.float32)
    # return_t = A_t + V(s_t)，作为 Critic 的回归目标。
    returns = advantages + values_array
    # 优势标准化能让策略更新的尺度更稳定，尤其是同一批数据多轮训练时。
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = RolloutBatch(
        states=torch.as_tensor(np.asarray(states), dtype=torch.float32),
        actions=torch.as_tensor(actions, dtype=torch.long),
        old_log_probs=torch.as_tensor(old_log_probs, dtype=torch.float32),
        old_action_probs=torch.as_tensor(
            np.asarray(old_action_probs), dtype=torch.float32
        ),
        advantages=torch.as_tensor(advantages, dtype=torch.float32),
        returns=torch.as_tensor(returns, dtype=torch.float32),
    )
    return batch, state, episode_return, completed_returns


def iterate_minibatches(
    batch_size: int,
    minibatch_size: int,
) -> list[torch.Tensor]:
    """随机打乱 rollout 索引，并切成多个 minibatch。"""
    indices = torch.randperm(batch_size)
    return [
        indices[start : start + minibatch_size]
        for start in range(0, batch_size, minibatch_size)
    ]


def categorical_kl(
    old_probs: torch.Tensor,
    new_distribution: Categorical,
) -> torch.Tensor:
    """计算 D_KL(old_policy || new_policy)。

    这里使用完整动作分布 old_probs，而不是只用被采样动作的 log_prob。
    KL 衡量的是两个策略在同一状态下的整个动作分布差异。
    """
    old_log_probs = torch.log(old_probs.clamp_min(1e-8))
    new_log_probs = torch.log_softmax(new_distribution.logits, dim=-1)
    return torch.sum(old_probs * (old_log_probs - new_log_probs), dim=-1)


def evaluate_policy(
    model: ActorCritic,
    episodes: int = 5,
    seed: int = 10_000,
) -> float:
    """使用贪心动作评估当前策略。

    训练时需要采样动作来保持探索；评估时直接取概率最大的动作，
    这样能更稳定地观察当前策略的实际控制能力。
    """
    env = gym.make("CartPole-v1")
    episode_returns = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(model.actor(state_tensor), dim=-1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_returns.append(total_reward)

    env.close()
    return float(np.mean(episode_returns))
