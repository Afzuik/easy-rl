# 第 9 章算法代码

这些脚本与 `chapter9_order.md` 的学习顺序对应。每个算法都是独立文件，脚本内置了一个小型教学环境，不依赖 `gymnasium`。

| 文件 | 对应内容 | 重点观察 |
|---|---|---|
| `a2c.py` | 优势演员-评论员算法 | TD 误差如何同时训练 actor 和 critic |
| `a3c.py` | 异步优势演员-评论员算法 | 多 worker 如何异步更新全局网络 |
| `pathwise_derivative_policy_gradient.py` | 路径衍生策略梯度 | 连续动作中如何用 actor 近似 `argmax_a Q(s,a)` |

## 环境依赖

在 conda 的 base 环境中需要：

```powershell
conda run -n base python -c "import torch, numpy"
```

本目录脚本只依赖 `torch` 和 `numpy`。

## 运行方式

在仓库根目录执行：

```powershell
conda run -n base python docs/chapter9/code/a2c.py
conda run -n base python docs/chapter9/code/a3c.py
conda run -n base python docs/chapter9/code/pathwise_derivative_policy_gradient.py
```

快速验证：

```powershell
conda run -n base python docs/chapter9/code/a2c.py --episodes 20 --print-every 5
conda run -n base python docs/chapter9/code/a3c.py --workers 2 --episodes-per-worker 10 --print-every 5
conda run -n base python docs/chapter9/code/pathwise_derivative_policy_gradient.py --episodes 20 --print-every 5
```
