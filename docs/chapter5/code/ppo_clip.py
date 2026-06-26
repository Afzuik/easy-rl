"""
PPO-Clip（裁剪概率比值）教学实现。

策略目标:
    maximize E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]

算法流程:
    1. 冻结当前策略为 pi_old，并收集一批 rollout
    2. 用 GAE 计算优势 A_t
    3. 在同一批数据上训练多个 epoch
    4. 使用 min + clip 截断越界后的额外优化收益
    5. KL 过大时提前停止当前批次更新
    6. 重新采样并进入下一轮

运行:
    python ppo_clip.py

快速验证:
    python ppo_clip.py --updates 2 --rollout-steps 256 --epochs 2
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
    recent_episode_returns: list[float] = []

    print("=" * 76)
    print("PPO-Clip: min(ratio*A, clip(ratio)*A)")
    print(
        f"clip_epsilon={args.clip_epsilon}, "
        f"rollout={args.rollout_steps}, epochs={args.epochs}"
    )
    print("=" * 76)

    for update in range(1, args.updates + 1):
        # 每个 update 先用当前策略采样一批数据。采样时的策略随后
        # 被视为 pi_old；同一批数据会在下面被重复训练多个 epoch。
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
            # PPO 不是对整批 rollout 一次更新，而是打乱后切成小批量。
            # 同一批 old_policy 数据可被复用多轮，但 KL 不能漂太远。
            for indices in iterate_minibatches(
                len(batch.states), args.minibatch_size
            ):
                states = batch.states[indices]
                actions = batch.actions[indices]
                old_log_probs = batch.old_log_probs[indices]
                advantages = batch.advantages[indices]
                returns = batch.returns[indices]

                distribution = model.distribution(states)
                new_log_probs = distribution.log_prob(actions)
                # ratio = pi_new(a|s) / pi_old(a|s)。
                # old_log_probs 已在采样时冻结，所以这里能衡量新旧策略
                # 对同一个动作的概率变化。
                ratios = torch.exp(new_log_probs - old_log_probs)

                # 未裁剪的替代目标：ratio * A。
                # 若 A>0，希望 ratio 增大；若 A<0，希望 ratio 减小。
                original_objective = ratios * advantages
                clipped_ratios = torch.clamp(
                    ratios,
                    1.0 - args.clip_epsilon,
                    1.0 + args.clip_epsilon,
                )
                # 裁剪后的目标限制 ratio 不要离 1 太远，相当于用简单
                # 一阶优化近似“信任区域”。
                clipped_objective = clipped_ratios * advantages
                # 取 min 是 PPO-Clip 的关键：
                # - A>0 时，ratio 超过上界不会继续增加收益；
                # - A<0 时，ratio 低于下界不会继续增加收益。
                # PyTorch 最小化 loss，所以最大化目标要取负号。
                policy_loss = -torch.min(
                    original_objective, clipped_objective
                ).mean()

                # Critic 拟合 GAE 得到的 returns；entropy 鼓励探索。
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
                # 每个 epoch 后用整批 rollout 监控 KL。如果新策略已经
                # 偏离 pi_old 太远，就停止当前批次的后续 epoch。
                full_distribution = model.distribution(batch.states)
                full_kl = float(
                    categorical_kl(
                        batch.old_action_probs, full_distribution
                    ).mean().item()
                )
            if full_kl > args.early_stop_kl_multiplier * args.target_kl:
                stopped_early = True
                break

        with torch.no_grad():
            # 这些统计量只用于日志，帮助观察 PPO 的约束是否生效。
            final_distribution = model.distribution(batch.states)
            new_log_probs = final_distribution.log_prob(batch.actions)
            final_ratios = torch.exp(new_log_probs - batch.old_log_probs)
            final_kl = float(
                categorical_kl(
                    batch.old_action_probs, final_distribution
                ).mean().item()
            )
            clip_fraction = float(
                (
                    (final_ratios < 1.0 - args.clip_epsilon)
                    | (final_ratios > 1.0 + args.clip_epsilon)
                )
                .float()
                .mean()
                .item()
            )

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
                f"KL={final_kl:.5f} | clip_frac={clip_fraction:.3f} | "
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
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
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
