# 第 5 章算法代码

这些脚本与 `chapter5_order.md` 的学习顺序对应。建议依次运行：

| 文件 | 对应内容 | 重点观察 |
|---|---|---|
| `importance_sampling.py` | 重要性采样 | 权重、有效样本量和方差 |
| `off_policy_policy_gradient.py` | 异策略策略梯度 | 固定旧策略数据、多次更新目标策略 |
| `trpo.py` | TRPO | KL 硬约束、共轭梯度、线搜索 |
| `ppo_penalty.py` | PPO-Penalty | KL 软惩罚和自适应 `beta` |
| `ppo_clip.py` | PPO-Clip | 概率比值、裁剪比例和提前停止 |

## 环境依赖

```powershell
pip install numpy torch gymnasium
```

## 运行方式

在仓库根目录执行：

```powershell
python docs/chapter5/code/importance_sampling.py
python docs/chapter5/code/off_policy_policy_gradient.py
python docs/chapter5/code/trpo.py
python docs/chapter5/code/ppo_penalty.py
python docs/chapter5/code/ppo_clip.py
```

PPO 和 TRPO 的默认配置会完整训练 CartPole。只验证代码流程时，可使用：

```powershell
python docs/chapter5/code/trpo.py --updates 2 --rollout-steps 256 --value-epochs 2
python docs/chapter5/code/ppo_penalty.py --updates 2 --rollout-steps 256 --epochs 2
python docs/chapter5/code/ppo_clip.py --updates 2 --rollout-steps 256 --epochs 2
```

`common.py` 不是独立算法，只提供三个 CartPole 算法共用的网络、GAE、采样和评估函数。

