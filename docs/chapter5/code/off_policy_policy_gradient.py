"""
异策略策略梯度（Off-Policy Policy Gradient）教学代码。

为了只突出重要性权重，本例使用一个单状态、双动作的老虎机:
    动作 0 的平均奖励约为 0.2
    动作 1 的平均奖励约为 1.0

行为策略 pi_old 固定产生一批数据，目标策略 pi_theta 多次复用该批数据:

    J(theta) = E_{a~pi_old}[pi_theta(a)/pi_old(a) * A(a)]

运行:
    python off_policy_policy_gradient.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch.distributions import Categorical


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def collect_bandit_batch(
    rng: np.random.Generator,
    behavior_probs: np.ndarray,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """用固定行为策略 pi_old 采集一批 bandit 数据。

    这个例子没有状态，只有两个动作。为了突出异策略策略梯度，
    我们把行为策略固定为 pi_old，并记录每个采样动作在 pi_old 下
    的概率 pi_old(a)，后续用它计算重要性比值:

        ratio = pi_theta(a) / pi_old(a)
    """
    # 所有动作都来自行为策略，而不是后面要优化的目标策略。
    actions = rng.choice(2, size=sample_count, p=behavior_probs)
    reward_means = np.asarray([0.2, 1.0])
    # 动作 1 的平均奖励更高，因此理想目标策略应逐渐偏向动作 1。
    rewards = reward_means[actions] + rng.normal(0.0, 0.2, size=sample_count)
    # 用批次平均奖励作 baseline，advantages 有正有负，降低梯度方差。
    advantages = rewards - rewards.mean()
    return (
        torch.as_tensor(actions, dtype=torch.long),
        torch.as_tensor(advantages, dtype=torch.float32),
        torch.as_tensor(behavior_probs[actions], dtype=torch.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    behavior_probs = np.asarray([0.7, 0.3], dtype=np.float32)
    # 采样只做一次。后面多个 epoch 都复用这批旧策略数据，
    # 这正是异策略学习相对同策略学习的数据复用优势。
    actions, advantages, behavior_action_probs = collect_bandit_batch(
        rng, behavior_probs, args.samples
    )

    # logits 是目标策略 theta 的参数。初始时令目标策略等于行为策略，
    # 这样一开始 ratio 接近 1；随着训练进行，pi_theta 会偏离 pi_old。
    logits = torch.nn.Parameter(torch.log(torch.as_tensor(behavior_probs)))
    optimizer = torch.optim.Adam([logits], lr=args.lr)

    print("=" * 68)
    print("异策略策略梯度: 固定旧策略数据，多次更新目标策略")
    print(f"行为策略 pi_old = {behavior_probs.tolist()}")
    print("=" * 68)

    for epoch in range(1, args.epochs + 1):
        distribution = Categorical(logits=logits)
        # 只取“旧数据中实际出现过的动作”的新策略概率 pi_theta(a_i)。
        new_action_probs = distribution.probs[actions]
        # 重要性采样比值。它把 pi_old 采样得到的数据，修正为可用于
        # 估计 pi_theta 目标的加权样本。
        ratios = new_action_probs / behavior_action_probs

        # 最大化 E[ratio * advantage]，PyTorch 用梯度下降所以取负号。
        objective = torch.mean(ratios * advantages)
        loss = -objective

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                print(
                    f"epoch={epoch:>3d} | "
                    f"J={objective.item():>7.4f} | "
                    f"pi_theta={probs.tolist()} | "
                    f"ratio范围=[{ratios.min().item():.3f}, "
                    f"{ratios.max().item():.3f}]"
                )

    print("\n最终策略应明显偏向奖励更高的动作 1。")
    print("注意: 此示例没有 PPO 的 KL/Clip 约束，更新过多时比值会不断漂移。")


if __name__ == "__main__":
    main()
