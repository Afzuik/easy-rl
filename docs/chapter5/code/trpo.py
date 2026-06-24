"""
TRPO（Trust Region Policy Optimization）教学实现。

TRPO 最大化替代目标，同时满足平均 KL 硬约束:

    maximize E[r_t(theta) * A_t]
    subject to KL(pi_old || pi_theta) <= max_kl

核心步骤:
    1. 计算替代目标的策略梯度 g
    2. 用 KL 的 Hessian-vector product 表示 Fisher 矩阵乘法
    3. 用共轭梯度求解 F x = g
    4. 按 max_kl 缩放自然梯度步长
    5. 用回溯线搜索确保目标改善且 KL 不越界

运行:
    python trpo.py

快速验证:
    python trpo.py --updates 2 --rollout-steps 256 --value-epochs 2
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    ActorCritic,
    categorical_kl,
    collect_rollout,
    evaluate_policy,
    make_env,
    set_seed,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def flat_parameters(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def set_flat_parameters(module: torch.nn.Module, flat_values: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in module.parameters():
            element_count = parameter.numel()
            parameter.copy_(
                flat_values[offset : offset + element_count].view_as(parameter)
            )
            offset += element_count


def flat_gradient(
    output: torch.Tensor,
    module: torch.nn.Module,
    create_graph: bool = False,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        output,
        tuple(module.parameters()),
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def conjugate_gradient(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    vector: torch.Tensor,
    iterations: int,
    residual_tolerance: float = 1e-10,
) -> torch.Tensor:
    """不显式构造 Fisher 矩阵，迭代求解 F x = vector。"""
    solution = torch.zeros_like(vector)
    residual = vector.clone()
    direction = residual.clone()
    residual_dot = torch.dot(residual, residual)

    for _ in range(iterations):
        matrix_direction = matrix_vector_product(direction)
        alpha = residual_dot / (torch.dot(direction, matrix_direction) + 1e-8)
        solution += alpha * direction
        residual -= alpha * matrix_direction
        new_residual_dot = torch.dot(residual, residual)

        if new_residual_dot < residual_tolerance:
            break

        beta = new_residual_dot / (residual_dot + 1e-8)
        direction = residual + beta * direction
        residual_dot = new_residual_dot

    return solution


def train(args: argparse.Namespace) -> ActorCritic:
    set_seed(args.seed)
    env = make_env(args.seed)
    state, _ = env.reset(seed=args.seed)
    ongoing_episode_return = 0.0

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim, args.hidden_dim)
    value_optimizer = torch.optim.Adam(model.critic.parameters(), lr=args.value_lr)
    recent_episode_returns: list[float] = []

    print("=" * 76)
    print("TRPO: KL 硬约束 + 自然梯度 + 回溯线搜索")
    print(
        f"max_kl={args.max_kl}, rollout={args.rollout_steps}, "
        f"CG迭代={args.cg_iterations}"
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

        # 先训练 Critic。TRPO 的硬约束只作用于 Actor。
        for _ in range(args.value_epochs):
            predicted_values = model.value(batch.states)
            value_loss = F.mse_loss(predicted_values, batch.returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

        def surrogate_objective() -> torch.Tensor:
            distribution = model.distribution(batch.states)
            new_log_probs = distribution.log_prob(batch.actions)
            ratios = torch.exp(new_log_probs - batch.old_log_probs)
            return torch.mean(ratios * batch.advantages)

        def mean_kl() -> torch.Tensor:
            distribution = model.distribution(batch.states)
            return categorical_kl(
                batch.old_action_probs, distribution
            ).mean()

        objective_before = surrogate_objective()
        policy_gradient = flat_gradient(objective_before, model.actor)

        def fisher_vector_product(vector: torch.Tensor) -> torch.Tensor:
            kl = mean_kl()
            kl_gradient = flat_gradient(kl, model.actor, create_graph=True)
            kl_vector_product = torch.dot(kl_gradient, vector)
            hessian_vector = flat_gradient(
                kl_vector_product, model.actor, create_graph=False
            )
            return hessian_vector.detach() + args.damping * vector

        natural_direction = conjugate_gradient(
            fisher_vector_product,
            policy_gradient.detach(),
            iterations=args.cg_iterations,
        )
        fisher_direction = fisher_vector_product(natural_direction)
        quadratic_term = torch.dot(natural_direction, fisher_direction)
        step_scale = math.sqrt(
            2.0 * args.max_kl / (float(quadratic_term.item()) + 1e-8)
        )
        full_step = natural_direction * step_scale
        expected_improvement = float(torch.dot(policy_gradient, full_step).item())

        old_parameters = flat_parameters(model.actor)
        old_objective = float(objective_before.item())
        accepted = False
        accepted_fraction = 0.0

        for line_search_step in range(args.line_search_steps):
            fraction = args.backtrack_coefficient**line_search_step
            candidate_parameters = old_parameters + fraction * full_step
            set_flat_parameters(model.actor, candidate_parameters)

            with torch.no_grad():
                candidate_objective = float(surrogate_objective().item())
                candidate_kl = float(mean_kl().item())
            actual_improvement = candidate_objective - old_objective
            required_improvement = (
                args.accept_ratio * fraction * expected_improvement
            )

            if (
                candidate_kl <= args.max_kl
                and actual_improvement > required_improvement
            ):
                accepted = True
                accepted_fraction = fraction
                break

        if not accepted:
            set_flat_parameters(model.actor, old_parameters)

        with torch.no_grad():
            final_objective = float(surrogate_objective().item())
            final_kl = float(mean_kl().item())

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
                f"KL={final_kl:.5f}/{args.max_kl:.5f} | "
                f"目标改善={final_objective - old_objective:+.5f} | "
                f"step_fraction={accepted_fraction:.3f} | "
                f"accepted={accepted}"
            )

    env.close()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--cg-iterations", type=int, default=10)
    parser.add_argument("--line-search-steps", type=int, default=10)
    parser.add_argument("--backtrack-coefficient", type=float, default=0.5)
    parser.add_argument("--accept-ratio", type=float, default=0.1)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--value-epochs", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
