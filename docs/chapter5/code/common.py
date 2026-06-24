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
    """一批由旧策略采集并冻结的数据。"""

    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_action_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ActorCritic(nn.Module):
    """离散动作空间的 Actor-Critic 网络。"""

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
        return Categorical(logits=self.actor(states))

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states).squeeze(-1)

    def act(self, state: np.ndarray) -> tuple[int, float, np.ndarray, float]:
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int) -> gym.Env:
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
        action, log_prob, action_probs, value = model.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

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

        episode_return += reward
        if done:
            completed_returns.append(episode_return)
            episode_return = 0.0
            state, _ = env.reset()
        else:
            state = next_state

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
    returns = advantages + values_array
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
    indices = torch.randperm(batch_size)
    return [
        indices[start : start + minibatch_size]
        for start in range(0, batch_size, minibatch_size)
    ]


def categorical_kl(
    old_probs: torch.Tensor,
    new_distribution: Categorical,
) -> torch.Tensor:
    """计算 D_KL(old_policy || new_policy)。"""
    old_log_probs = torch.log(old_probs.clamp_min(1e-8))
    new_log_probs = torch.log_softmax(new_distribution.logits, dim=-1)
    return torch.sum(old_probs * (old_log_probs - new_log_probs), dim=-1)


def evaluate_policy(
    model: ActorCritic,
    episodes: int = 5,
    seed: int = 10_000,
) -> float:
    """使用贪心动作评估当前策略。"""
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
