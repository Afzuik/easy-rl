"""
PPO-Penalty（自适应 KL 惩罚）教学实现。

策略目标:
    maximize E[r_t(theta) * A_t] - beta * KL(pi_old || pi_theta)

算法流程:
    1. 冻结当前策略为 pi_old，并收集一批 rollout
    2. 用 GAE 计算优势 A_t
    3. 在同一批数据上训练多个 epoch
    4. 用 KL 惩罚限制新旧策略差异
    5. 根据实际 KL 自适应调整 beta
    6. 将新策略作为下一轮旧策略，重新采样

运行:
    python ppo_penalty.py

快速验证:
    python ppo_penalty.py --updates 2 --rollout-steps 256 --epochs 2
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    ActorCritic,
    categorical_kl,
    collect_rollout,
    evaluate_policy,
    iterate_minibatches,
    make_env,
    set_seed,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def train(args: argparse.Namespace) -> ActorCritic:
    set_seed(args.seed)
    env = make_env(args.seed)
    state, _ = env.reset(seed=args.seed)
    ongoing_episode_return = 0.0

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim, args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    beta = args.beta
    recent_episode_returns: list[float] = []

    print("=" * 76)
    print("PPO-Penalty: surrogate objective - beta * KL(old || new)")
    print(
        f"target_kl={args.target_kl}, beta初值={beta}, "
        f"rollout={args.rollout_steps}, epochs={args.epochs}"
    )
    print("=" * 76)

    for update in range(1, args.updates + 1):
        batch, state, ongoing_episode_return, completed_returns = collect_rollout(
            env,
            model,
            state,
            ongoing_episode_return,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        recent_episode_returns.extend(completed_returns)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        stopped_early = False

        for _ in range(args.epochs):
            for indices in iterate_minibatches(
                len(batch.states), args.minibatch_size
            ):
                states = batch.states[indices]
                actions = batch.actions[indices]
                old_log_probs = batch.old_log_probs[indices]
                old_action_probs = batch.old_action_probs[indices]
                advantages = batch.advantages[indices]
                returns = batch.returns[indices]

                distribution = model.distribution(states)
                new_log_probs = distribution.log_prob(actions)
                ratios = torch.exp(new_log_probs - old_log_probs)
                kl = categorical_kl(old_action_probs, distribution).mean()

                surrogate = torch.mean(ratios * advantages)
                policy_loss = -surrogate + beta * kl
                value_loss = F.mse_loss(model.value(states), returns)
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())

            with torch.no_grad():
                full_distribution = model.distribution(batch.states)
                full_kl = float(
                    categorical_kl(
                        batch.old_action_probs, full_distribution
                    ).mean().item()
                )
            if full_kl > args.early_stop_kl_multiplier * args.target_kl:
                stopped_early = True
                break

        # 自适应 KL 惩罚系数。KL 太大就加强惩罚，太小就减弱惩罚。
        with torch.no_grad():
            final_distribution = model.distribution(batch.states)
            final_kl = float(
                categorical_kl(
                    batch.old_action_probs, final_distribution
                ).mean().item()
            )

        if final_kl > 1.5 * args.target_kl:
            beta = min(beta * 2.0, 1_000.0)
        elif final_kl < args.target_kl / 1.5:
            beta = max(beta / 2.0, 1e-4)

        average_return = (
            float(np.mean(recent_episode_returns[-20:]))
            if recent_episode_returns
            else 0.0
        )

        if update == 1 or update % args.print_every == 0:
            evaluation = evaluate_policy(
                model, episodes=args.eval_episodes, seed=args.seed + update
            )
            print(
                f"update={update:>3d} | "
                f"rollout回报={average_return:>6.1f} | "
                f"评估回报={evaluation:>6.1f} | "
                f"KL={final_kl:.5f} | beta={beta:.4f} | "
                f"policy_loss={last_policy_loss:+.4f} | "
                f"value_loss={last_value_loss:.3f} | "
                f"early_stop={stopped_early}"
            )

    env.close()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--early-stop-kl-multiplier", type=float, default=4.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
