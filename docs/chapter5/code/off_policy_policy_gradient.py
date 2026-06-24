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
    actions = rng.choice(2, size=sample_count, p=behavior_probs)
    reward_means = np.asarray([0.2, 1.0])
    rewards = reward_means[actions] + rng.normal(0.0, 0.2, size=sample_count)
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
    actions, advantages, behavior_action_probs = collect_bandit_batch(
        rng, behavior_probs, args.samples
    )

    # logits 是目标策略 theta 的参数。初始时令目标策略等于行为策略。
    logits = torch.nn.Parameter(torch.log(torch.as_tensor(behavior_probs)))
    optimizer = torch.optim.Adam([logits], lr=args.lr)

    print("=" * 68)
    print("异策略策略梯度: 固定旧策略数据，多次更新目标策略")
    print(f"行为策略 pi_old = {behavior_probs.tolist()}")
    print("=" * 68)

    for epoch in range(1, args.epochs + 1):
        distribution = Categorical(logits=logits)
        new_action_probs = distribution.probs[actions]
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

