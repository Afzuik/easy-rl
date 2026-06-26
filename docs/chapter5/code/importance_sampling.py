"""
重要性采样（Importance Sampling）教学代码。

目标:
    不能直接从目标分布 p 采样时，使用提议分布 q 的样本估计

        E_p[f(x)] = E_q[f(x) * p(x) / q(x)]

本例计算:
    p = N(-2, 1)
    f(x) = x^2
    E_p[x^2] = Var(X) + E[X]^2 = 1 + 4 = 5

运行:
    python importance_sampling.py
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """正态分布概率密度函数。

    重要性采样只需要知道 p(x) 和 q(x) 在样本点上的密度值，
    不要求真的能从目标分布 p 采样。
    """
    coefficient = 1.0 / (std * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coefficient * np.exp(exponent)


def importance_sampling(
    rng: np.random.Generator,
    sample_count: int,
    proposal_mean: float,
    proposal_std: float,
) -> tuple[float, float, float, float]:
    """从 q 采样并返回估计值、标准误、有效样本量和最大权重。

    数学对应:
        E_p[f(x)] = E_q[f(x) * p(x) / q(x)]

    这里 f(x)=x^2。权重 p(x)/q(x) 越极端，有限样本估计越不稳定。
    """
    # 真实采样来自 proposal distribution q，而不是目标分布 p。
    samples = rng.normal(proposal_mean, proposal_std, size=sample_count)
    # 分别计算每个样本在 p 和 q 下的密度，用二者比值修正采样偏差。
    target_density = normal_pdf(samples, mean=-2.0, std=1.0)
    proposal_density = normal_pdf(samples, proposal_mean, proposal_std)
    weights = target_density / proposal_density
    # 对 f(x)=x^2 乘以重要性权重，得到 q 分布下的无偏估计项。
    weighted_values = samples**2 * weights

    estimate = float(weighted_values.mean())
    standard_error = float(weighted_values.std(ddof=1) / math.sqrt(sample_count))
    # ESS 越接近 sample_count，说明权重越均匀；越小表示只有少数样本
    # 承担了大部分权重，估计方差会很高。
    effective_sample_size = float(
        weights.sum() ** 2 / np.sum(weights**2)
    )
    return estimate, standard_error, effective_sample_size, float(weights.max())


def repeated_experiment(
    rng: np.random.Generator,
    repeats: int,
    sample_count: int,
    proposal_mean: float,
    proposal_std: float,
) -> tuple[float, float]:
    """重复多次实验，观察同一 q 下估计值的跨实验波动。"""
    estimates = [
        importance_sampling(
            rng,
            sample_count=sample_count,
            proposal_mean=proposal_mean,
            proposal_std=proposal_std,
        )[0]
        for _ in range(repeats)
    ]
    return float(np.mean(estimates)), float(np.std(estimates, ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    true_value = 5.0

    # 对 X~N(-2,1)，E[X^2]=Var(X)+E[X]^2=1+4=5。
    # 这个真值用于直观看出重要性采样估计是否靠谱。
    print("=" * 68)
    print("重要性采样: 使用 q 的样本估计 E_p[x^2]")
    print("目标分布 p=N(-2,1), f(x)=x^2, 真实值=5")
    print("=" * 68)

    proposals = [
        # q 与 p 接近时，权重不会过于极端，ESS 较高。
        ("接近目标的 q", -1.5, 1.2),
        # q 与 p 距离较远时，理论仍可无偏，但有限样本方差会变大。
        ("远离目标的 q", 2.0, 2.5),
    ]

    for name, mean, std in proposals:
        estimate, standard_error, ess, max_weight = importance_sampling(
            rng,
            sample_count=args.samples,
            proposal_mean=mean,
            proposal_std=std,
        )
        repeated_mean, repeated_std = repeated_experiment(
            rng,
            repeats=args.repeats,
            sample_count=args.samples,
            proposal_mean=mean,
            proposal_std=std,
        )

        print(f"\n{name}: N({mean}, {std}^2)")
        print(f"  单次估计:       {estimate:.4f}")
        print(f"  与真实值误差:   {estimate - true_value:+.4f}")
        print(f"  单次标准误:     {standard_error:.4f}")
        print(f"  有效样本量 ESS: {ess:.0f}/{args.samples}")
        print(f"  最大权重:       {max_weight:.2f}")
        print(f"  重复实验均值:   {repeated_mean:.4f}")
        print(f"  重复估计标准差: {repeated_std:.4f}")

    print("\n结论:")
    print("  1. 只要 q 覆盖 p 的支撑集，重要性采样在理论上无偏。")
    print("  2. q 与 p 差距越大，权重越极端，有效样本量越低。")
    print("  3. 期望可以相同，但有限样本估计的方差可能差很多。")


if __name__ == "__main__":
    main()
