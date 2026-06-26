# 第 9 章 演员-评论员算法 —— 核心概念与公式总结

> **说明**：本文根据 `chapter9_order.md` 整理，重点归纳第 9 章的核心概念、公式含义、公式之间的依赖关系和常见易错点。它适合作为学习完章节后的复习提纲，也适合在写代码前快速确认每个量的定义。

---

## 一、本章概念总览

演员-评论员算法的核心思想是：

> **演员负责行动，评论员负责评价，演员根据评论员的评价改进策略。**

| 概念 | 英文 | 作用 | 关键公式 |
|---|---|---|---|
| 演员 | actor | 输出策略或动作 | $\pi_\theta(a\mid s)$ 或 $\mu_\theta(s)$ |
| 评论员 | critic | 估计状态或动作价值 | $V_\pi(s)$、$Q_\pi(s,a)$ |
| 回报 | return | 从当前时刻开始的未来累计奖励 | $G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}$ |
| 状态价值 | state value | 当前状态的平均未来价值 | $V_\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]$ |
| 动作价值 | action value | 当前状态采取指定动作后的平均未来价值 | $Q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]$ |
| 优势函数 | advantage | 动作相对当前状态平均水平的好坏 | $A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s)$ |
| TD 误差 | TD error | 用一步奖励和价值估计近似优势 | $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ |
| A2C | advantage actor-critic | 用优势函数稳定策略梯度 | $\nabla\log\pi(a\mid s)A(s,a)$ |
| A3C | asynchronous advantage actor-critic | 多 worker 异步并行采样和更新 | 异步累积并提交 actor/critic 梯度 |
| 路径衍生策略梯度 | pathwise derivative policy gradient | 用评论员的动作梯度指导连续动作演员 | $\nabla_a Q(s,a)\nabla_\theta\mu_\theta(s)$ |

---

## 二、策略梯度：演员如何更新

### 2.1 策略函数

演员通常表示为一个策略函数：

$$
\pi_\theta(a\mid s)
$$

含义：在状态 $s$ 下，参数为 $\theta$ 的策略选择动作 $a$ 的概率。

如果动作空间是离散的，$\pi_\theta(\cdot\mid s)$ 输出动作概率分布；如果动作空间是连续的，策略可以输出连续分布的参数，或者直接输出确定性动作。

---

### 2.2 折扣回报

从时刻 $t$ 开始的折扣回报为：

$$
\boxed{
G_t
=
r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots
=
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
}
$$

其中 $\gamma\in[0,1]$ 是折扣因子。

| $\gamma$ | 含义 |
|---|---|
| $\gamma=0$ | 只关心即时奖励 |
| $\gamma$ 接近 $1$ | 更重视长期奖励 |
| $\gamma=1$ | 所有未来奖励权重相同，通常只适合有限回合任务 |

> **注意**：原章节中也会使用有限回合写法 $\sum_{t'=t}^{T_n}\gamma^{t'-t}r_{t'}^n$。它与上面的无限和写法本质相同，只是回合在 $T_n$ 时结束。

---

### 2.3 带基线的策略梯度

第 9 章回顾的策略梯度公式为：

$$
\boxed{
\nabla \bar{R}_{\theta}
\approx
\frac{1}{N}
\sum_{n=1}^{N}
\sum_{t=1}^{T_n}
\left(
\sum_{t'=t}^{T_n}\gamma^{t'-t}r_{t'}^n-b
\right)
\nabla\log\pi_\theta(a_t^n\mid s_t^n)
}
\tag{9.1}
$$

其中括号内是“当前动作之后的回报减去基线”：

$$
G_t^n-b
$$

该公式可以理解为：

| 部分 | 作用 |
|---|---|
| $\nabla\log\pi_\theta(a_t^n\mid s_t^n)$ | 告诉参数怎样改变才能提高当前动作概率 |
| $G_t^n-b$ | 告诉算法当前动作应该被鼓励还是惩罚 |
| $G_t^n-b>0$ | 增大该动作概率 |
| $G_t^n-b<0$ | 减小该动作概率 |

> **重点**：策略梯度的方向来自 $\nabla\log\pi_\theta$，更新力度和正负号来自 $G_t-b$。

---

### 2.4 策略梯度的主要问题：回报方差大

$G_t$ 是随机变量，随机性来自：

1. 策略本身按概率采样动作；
2. 环境转移可能随机；
3. 长期回报累积了很多未来不确定性。

因此，REINFORCE 使用完整回报 $G_t$ 更新策略时，梯度估计方差通常较大。演员-评论员算法就是为了解决这个问题：用价值函数估计期望回报，替代单次采样得到的高方差回报。

---

## 三、价值函数：评论员评价什么

### 3.1 状态价值函数 $V_\pi(s)$

状态价值函数定义为：

$$
\boxed{
V_\pi(s)
=
\mathbb{E}_\pi[G_t\mid s_t=s]
}
$$

含义：如果当前处于状态 $s$，以后一直按照策略 $\pi$ 行动，预计能得到多少未来回报。

特点：

- 输入只有状态 $s$；
- 输出一个标量；
- 表示该状态在当前策略下的平均价值。

---

### 3.2 动作价值函数 $Q_\pi(s,a)$

动作价值函数定义为：

$$
\boxed{
Q_\pi(s,a)
=
\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
}
$$

含义：如果当前在状态 $s$ 先采取动作 $a$，之后再按照策略 $\pi$ 行动，预计能得到多少未来回报。

特点：

- 输入状态 $s$ 和动作 $a$；
- 输出一个标量；
- 比 $V_\pi(s)$ 更具体，因为它指定了当前动作。

---

### 3.3 $V_\pi(s)$ 与 $Q_\pi(s,a)$ 的关系

状态价值是动作价值在当前策略动作分布下的平均：

$$
\boxed{
V_\pi(s)
=
\mathbb{E}_{a\sim\pi(\cdot\mid s)}
\left[
Q_\pi(s,a)
\right]
}
$$

离散动作空间中：

$$
\boxed{
V_\pi(s)
=
\sum_a \pi(a\mid s)Q_\pi(s,a)
}
$$

> **直观理解**：$Q_\pi(s,a)$ 是“在状态 $s$ 下选某个动作有多好”，$V_\pi(s)$ 是“在状态 $s$ 下按当前策略平均有多好”。

---

## 四、优势函数：动作相对平均水平好多少

### 4.1 优势函数定义

优势函数定义为：

$$
\boxed{
A_\pi(s,a)
=
Q_\pi(s,a)-V_\pi(s)
}
$$

它回答的问题是：

> 在状态 $s$ 下，动作 $a$ 是否比当前策略的平均动作更好？

判断规则：

| 条件 | 含义 | 演员应该怎么做 |
|---|---|---|
| $A_\pi(s,a)>0$ | 动作 $a$ 比平均水平好 | 增大 $\pi(a\mid s)$ |
| $A_\pi(s,a)<0$ | 动作 $a$ 比平均水平差 | 减小 $\pi(a\mid s)$ |
| $A_\pi(s,a)=0$ | 动作 $a$ 与平均水平相当 | 基本不改变概率 |

---

### 4.2 为什么优势函数比直接用 $Q$ 更合理

如果直接用 $Q_\pi(s,a)$ 更新策略，算法只知道动作的绝对价值。但在强化学习中，一个动作是否值得鼓励，取决于它相对当前状态下其他动作是否更好。

例如，某状态下所有动作都能得到较高回报，其中一个动作 $Q=90$，另一个动作 $Q=100$。虽然 $90$ 不低，但它低于该状态平均水平时，就不应该被继续鼓励。

优势函数通过减去 $V_\pi(s)$，把绝对价值变成相对价值。

> **重难点**：优势函数不是“动作好不好”的绝对判断，而是“动作相对同状态平均动作好不好”的相对判断。

---

## 五、TD 误差：只用一个 $V$ 网络近似优势

### 5.1 为什么不直接训练两个网络

根据定义：

$$
A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s)
$$

似乎需要同时训练：

- $Q_\pi(s,a)$ 网络；
- $V_\pi(s)$ 网络。

但两个网络都会有估计误差，相减后误差可能放大。因此 A2C 通常只训练 $V_\pi(s)$，然后用一步 TD 目标近似 $Q_\pi(s,a)$。

---

### 5.2 Bellman 关系

动作价值函数可以写成：

$$
\boxed{
Q_\pi(s_t,a_t)
=
\mathbb{E}
\left[
r_t+\gamma V_\pi(s_{t+1})
\mid
s_t,a_t
\right]
}
$$

含义：先得到一步奖励 $r_t$，再进入下一个状态 $s_{t+1}$，未来价值由 $V_\pi(s_{t+1})$ 估计。

实际采样时，用一次采样近似期望：

$$
\boxed{
Q_\pi(s_t,a_t)
\approx
r_t+\gamma V_\pi(s_{t+1})
}
$$

---

### 5.3 TD 目标

TD 目标定义为：

$$
\boxed{
y_t
=
r_t+\gamma V_\pi(s_{t+1})
}
$$

它是评论员希望 $V_\pi(s_t)$ 靠近的目标值。

如果 $s_{t+1}$ 是终止状态，通常令：

$$
V_\pi(s_{t+1})=0
$$

因此终止状态下：

$$
y_t=r_t
$$

---

### 5.4 TD 误差

TD 误差定义为：

$$
\boxed{
\delta_t
=
y_t-V_\pi(s_t)
=
r_t+\gamma V_\pi(s_{t+1})-V_\pi(s_t)
}
$$

将 TD 误差与优势函数对比：

$$
\boxed{
A_\pi(s_t,a_t)
\approx
\delta_t
}
$$

也就是：

$$
\boxed{
A_\pi(s_t,a_t)
\approx
r_t+\gamma V_\pi(s_{t+1})-V_\pi(s_t)
}
$$

> **重点**：TD 误差既是评论员的预测误差，也是演员更新时常用的优势估计。

---

### 5.5 Monte Carlo 回报与 TD 误差对比

| 对比项 | Monte Carlo 回报 $G_t$ | TD 误差 $\delta_t$ |
|---|---|---|
| 使用信息 | 完整未来轨迹 | 一步奖励 + 下一状态价值估计 |
| 更新时机 | 通常等回合结束 | 每一步都可更新 |
| 方差 | 高 | 较低 |
| 偏差 | 低 | 可能有偏 |
| 代表算法 | REINFORCE | A2C、A3C |

> **理解方式**：TD 误差用“价值函数估计”替代“实际走完整个未来”，所以效率更高，但依赖评论员估计质量。

---

## 六、A2C：优势演员-评论员公式总结

### 6.1 A2C 策略梯度

用优势函数更新演员：

$$
\boxed{
\nabla \bar{R}_{\theta}
\approx
\frac{1}{N}
\sum_{n=1}^{N}
\sum_{t=1}^{T_n}
A_\pi(s_t^n,a_t^n)
\nabla\log\pi_\theta(a_t^n\mid s_t^n)
}
$$

用 TD 误差近似优势函数后：

$$
\boxed{
\nabla \bar{R}_{\theta}
\approx
\frac{1}{N}
\sum_{n=1}^{N}
\sum_{t=1}^{T_n}
\left(
r_t^n+\gamma V_\pi(s_{t+1}^n)-V_\pi(s_t^n)
\right)
\nabla\log\pi_\theta(a_t^n\mid s_t^n)
}
\tag{9.2}
$$

---

### 6.2 Actor Loss

深度学习框架通常最小化损失函数，因此 actor loss 常写成：

$$
\boxed{
L_{\text{actor}}(\theta)
=
-
\log\pi_\theta(a_t\mid s_t)\,
\delta_t
}
$$

其中：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

解释：

| 情况 | loss 的效果 |
|---|---|
| $\delta_t>0$ | 当前动作比预期好，最小化 loss 会提高该动作概率 |
| $\delta_t<0$ | 当前动作比预期差，最小化 loss 会降低该动作概率 |

---

### 6.3 Critic Loss

评论员要让自己的预测 $V(s_t)$ 接近 TD 目标：

$$
y_t=r_t+\gamma V(s_{t+1})
$$

因此 critic loss 为：

$$
\boxed{
L_{\text{critic}}(w)
=
\left(
y_t-V_w(s_t)
\right)^2
}
$$

展开为：

$$
\boxed{
L_{\text{critic}}(w)
=
\left(
r_t+\gamma V_w(s_{t+1})-V_w(s_t)
\right)^2
}
$$

> **注意**：实现时，$y_t$ 中的 $V_w(s_{t+1})$ 通常需要停止梯度传播，避免目标值本身也被当前 loss 牵着走。

---

### 6.4 熵正则

为鼓励探索，可以加入策略熵：

$$
\boxed{
H(\pi(\cdot\mid s))
=
-
\sum_a \pi(a\mid s)\log\pi(a\mid s)
}
$$

熵越大，动作分布越均匀，探索越充分。

常见总损失：

$$
\boxed{
L
=
L_{\text{actor}}
c_v L_{\text{critic}}
-c_H H(\pi)
}
$$

其中：

| 系数 | 作用 |
|---|---|
| $c_v$ | 控制评论员损失权重 |
| $c_H$ | 控制熵奖励权重 |

> **直观理解**：总损失一边让演员选择更高优势动作，一边让评论员预测更准，同时用熵项避免策略过早收缩到单一动作。

---

## 七、A3C：异步优势演员-评论员

### 7.1 A3C 的核心概念

A3C 是 **asynchronous advantage actor-critic**，即异步优势演员-评论员。

它在 A2C 基础上加入多个并行 worker：

- 每个 worker 拥有本地网络；
- 每个 worker 与自己的环境副本交互；
- 每个 worker 计算梯度后异步更新全局网络；
- 全局网络再把新参数同步给 worker。

---

### 7.2 A3C 的参数记号

常见记号如下：

| 符号 | 含义 |
|---|---|
| $\theta$ | 全局演员参数 |
| $w$ | 全局评论员参数 |
| $\theta'$ | 某个 worker 的本地演员参数 |
| $w'$ | 某个 worker 的本地评论员参数 |

同步参数时：

$$
\theta' \leftarrow \theta,\qquad w'\leftarrow w
$$

worker 用本地参数采样并计算梯度，随后把梯度应用到全局参数。

---

### 7.3 A3C 中的 n-step 回报

A3C 常使用多步回报估计。若 worker 从 $t$ 采样到 $t+k$，则：

$$
\boxed{
R_t
=
r_t+\gamma r_{t+1}+\cdots+\gamma^{k-1}r_{t+k-1}
+\gamma^k V_{w'}(s_{t+k})
}
$$

如果 $s_{t+k}$ 是终止状态，则：

$$
V_{w'}(s_{t+k})=0
$$

于是 $R_t$ 退化为实际采样到的多步折扣回报。

---

### 7.4 A3C 的 actor 梯度

对本地策略参数 $\theta'$，累积 actor 梯度：

$$
\boxed{
\mathrm{d}\theta
\leftarrow
\mathrm{d}\theta
+
\nabla_{\theta'}
\log\pi_{\theta'}(a_i\mid s_i)
\left(
R_i-V_{w'}(s_i)
\right)
}
$$

其中

$$
R_i-V_{w'}(s_i)
$$

是多步优势估计。

---

### 7.5 A3C 的 critic 梯度

评论员最小化价值预测误差：

$$
\boxed{
L_{\text{critic}}(w')
=
\left(
R_i-V_{w'}(s_i)
\right)^2
}
$$

对应梯度累积可写成：

$$
\boxed{
\mathrm{d}w
\leftarrow
\mathrm{d}w
+
\nabla_{w'}
\left(
R_i-V_{w'}(s_i)
\right)^2
}
$$

最后 worker 将 $\mathrm{d}\theta$ 和 $\mathrm{d}w$ 异步提交给全局参数 $\theta,w$。

---

### 7.6 A2C 与 A3C 的概念区别

| 对比项 | A2C | A3C |
|---|---|---|
| 全称 | advantage actor-critic | asynchronous advantage actor-critic |
| 更新方式 | 通常同步更新 | 异步更新 |
| worker | 可并行采样后统一更新 | 各 worker 独立计算并提交梯度 |
| 优点 | 实现更简单，更新更整齐 | 采样更快，探索更多样 |
| 常见归类 | on-policy | 通常仍视为 on-policy |

> **易错点**：A3C 中 worker 梯度可能稍微过期，但它并不等同于经验回放式 off-policy 学习。

---

## 八、路径衍生策略梯度

### 8.1 DQN 中的动作选择

DQN 使用 Q 函数选择动作：

$$
\boxed{
a^*
=
\arg\max_a Q(s,a)
}
$$

离散动作空间中，可以枚举所有动作并比较 Q 值。

连续动作空间中，$a$ 是连续向量，精确求解 $\arg\max_a Q(s,a)$ 通常很难。

---

### 8.2 用演员近似 $\arg\max$

路径衍生策略梯度引入确定性演员：

$$
\boxed{
a
=
\mu_\theta(s)
}
$$

目标是让演员输出近似最优动作：

$$
\boxed{
\mu_\theta(s)
\approx
\arg\max_a Q_w(s,a)
}
$$

也就是说，演员网络 $\mu_\theta$ 是一个“求解器”：输入状态 $s$，直接输出使 $Q_w(s,a)$ 尽可能大的动作。

---

### 8.3 演员目标函数

固定评论员 $Q_w$ 时，演员希望最大化：

$$
\boxed{
J(\theta)
=
\mathbb{E}_{s}
\left[
Q_w(s,\mu_\theta(s))
\right]
}
$$

含义：演员输出的动作代入评论员后，得到的 Q 值越大越好。

---

### 8.4 路径衍生梯度公式

由链式法则：

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb{E}_{s}
\left[
\nabla_a Q_w(s,a)\big|_{a=\mu_\theta(s)}
\nabla_\theta \mu_\theta(s)
\right]
}
$$

公式中的两部分：

| 项 | 含义 |
|---|---|
| $\nabla_a Q_w(s,a)\big|_{a=\mu_\theta(s)}$ | 评论员告诉演员：动作往哪个方向改可以提高 Q 值 |
| $\nabla_\theta \mu_\theta(s)$ | 演员告诉参数：怎样改变参数才能改变输出动作 |

> **关键直觉**：普通 A2C 告诉演员“这个动作好还是不好”；路径衍生策略梯度告诉演员“动作应该朝哪个方向变得更好”。

---

### 8.5 从 DQN 到路径衍生策略梯度的目标值

DQN 的目标值：

$$
\boxed{
y_i
=
r_i+\gamma\max_a \hat{Q}(s_{i+1},a)
}
$$

路径衍生策略梯度用目标演员 $\hat{\mu}$ 替代 $\max_a$：

$$
\boxed{
y_i
=
r_i+\gamma \hat{Q}(s_{i+1},\hat{\mu}(s_{i+1}))
}
$$

这样就避免了在连续动作空间中显式求解 $\arg\max$。

---

### 8.6 评论员损失与演员目标

评论员学习 TD 目标：

$$
\boxed{
L_{\text{critic}}(w)
=
\left(
Q_w(s_i,a_i)-y_i
\right)^2
}
$$

演员最大化评论员输出：

$$
\boxed{
J(\theta)
=
Q_w(s,\mu_\theta(s))
}
$$

对应实现中通常会最小化：

$$
\boxed{
L_{\text{actor}}(\theta)
=
-
Q_w(s,\mu_\theta(s))
}
$$

---

### 8.7 当前网络与目标网络

路径衍生策略梯度通常维护四个网络：

| 网络 | 作用 |
|---|---|
| 当前演员 $\mu_\theta$ | 输出当前动作 |
| 目标演员 $\hat{\mu}$ | 计算 TD 目标中的下一动作 |
| 当前评论员 $Q_w$ | 评价当前状态-动作价值 |
| 目标评论员 $\hat{Q}$ | 计算稳定的 TD 目标 |

目标网络的作用与 DQN 中相同：让训练目标不要随着每一步参数更新剧烈变化。

---

## 九、GAN 与演员-评论员的概念对应

演员-评论员和 GAN 的结构相似，但目标不同。

| GAN | 演员-评论员 | 对应关系 |
|---|---|---|
| 生成器 generator | 演员 actor | 产生样本或动作 |
| 判别器 discriminator | 评论员 critic | 评价样本或动作 |
| 生成样本 | 选择动作 | 输出需要被评价的对象 |
| 判别真假 | 估计价值 | 给出反馈信号 |
| 生成器根据判别器更新 | 演员根据评论员更新 | 评价者影响生成者 |

> **注意**：这个类比只用于理解结构。GAN 的目标是生成逼真样本，演员-评论员的目标是最大化长期回报，不能把二者公式直接等同。

---

## 十、公式依赖关系图

本章公式可以按以下依赖链理解：

```text
策略 πθ(a|s)
  ↓ 采样动作并获得奖励
回报 Gt
  ↓ 方差过大，需要期望估计
价值函数 Vπ(s), Qπ(s,a)
  ↓ 与状态平均水平比较
优势函数 Aπ(s,a)=Qπ(s,a)-Vπ(s)
  ↓ 用 Bellman 关系只保留 V 网络
TD 误差 δt=r_t+γV(s_{t+1})-V(s_t)
  ↓
A2C: ∇logπ(a|s)δ_t
  ↓ 多 worker 并行
A3C
```

路径衍生策略梯度的依赖链是另一条线：

```text
DQN: a*=argmax_a Q(s,a)
  ↓ 连续动作空间中 argmax 难求
确定性演员 μθ(s)
  ↓
J(θ)=E_s[Q_w(s,μθ(s))]
  ↓ 链式法则
∇θJ=E_s[∇aQ_w(s,a)|_{a=μθ(s)} ∇θμθ(s)]
```

---

## 十一、易混公式对比

### 11.1 $G_t$、$V$、$Q$、$A$、$\delta$ 的区别

| 符号 | 名称 | 是否采样值 | 是否依赖动作 | 主要用途 |
|---|---|---|---|---|
| $G_t$ | 回报 | 是 | 间接依赖 | REINFORCE 的更新权重 |
| $V_\pi(s)$ | 状态价值 | 否，是期望估计 | 否 | 评论员估计、基线 |
| $Q_\pi(s,a)$ | 动作价值 | 否，是期望估计 | 是 | 评价指定动作 |
| $A_\pi(s,a)$ | 优势函数 | 通常由估计得到 | 是 | 判断动作相对好坏 |
| $\delta_t$ | TD 误差 | 含一步采样 | 是，因为由实际动作产生 | 近似优势、训练评论员 |

---

### 11.2 A2C 与路径衍生策略梯度的演员更新

| 方法 | 演员更新核心 | 评论员给演员的信息 |
|---|---|---|
| A2C | $\nabla\log\pi(a\mid s)A(s,a)$ | 当前动作比平均水平好还是差 |
| 路径衍生策略梯度 | $\nabla_a Q(s,a)\nabla_\theta\mu_\theta(s)$ | 动作应该朝哪个方向改变 |

---

## 十二、常见易错点

### 易错点 1：把 $V(s)$ 和 $Q(s,a)$ 混为一谈

$V(s)$ 不指定动作，是状态平均价值；$Q(s,a)$ 指定动作，是状态-动作价值。

正确关系是：

$$
V_\pi(s)
=
\mathbb{E}_{a\sim\pi(\cdot\mid s)}
\left[
Q_\pi(s,a)
\right]
$$

---

### 易错点 2：认为优势函数就是动作价值

优势函数不是 $Q(s,a)$，而是：

$$
A(s,a)=Q(s,a)-V(s)
$$

它表示的是相对价值，不是绝对价值。

---

### 易错点 3：忽略 TD 误差中的折扣因子

一般公式应写为：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

如果看到：

$$
\delta_t=r_t+V(s_{t+1})-V(s_t)
$$

通常表示省略了 $\gamma$，或默认 $\gamma=1$。

---

### 易错点 4：认为 A3C 是 off-policy

A3C 有多个 worker，且梯度可能稍微过期，但它通常仍被视为 on-policy 算法。判断 on-policy/off-policy 的核心不是是否并行，而是采样数据的行为策略是否基本对应当前学习策略。

---

### 易错点 5：把路径衍生策略梯度等同于普通 A2C

普通 A2C 使用：

$$
\nabla\log\pi(a\mid s)A(s,a)
$$

路径衍生策略梯度使用：

$$
\nabla_a Q(s,a)\nabla_\theta\mu_\theta(s)
$$

前者通过“提高或降低已采样动作概率”更新策略；后者通过“动作方向梯度”直接推动连续动作变好。

---

## 十三、一页速记

### 13.1 必背概念

- 演员：策略网络，决定动作。
- 评论员：价值网络，评价状态或动作。
- $V(s)$：状态平均价值。
- $Q(s,a)$：指定动作后的价值。
- $A(s,a)$：动作相对平均水平的好坏。
- $\delta_t$：一步 TD 误差，常用于近似优势。
- A2C：用优势函数改进策略梯度。
- A3C：多个 worker 异步并行的 A2C。
- 路径衍生策略梯度：用演员网络解决连续动作中的 $\arg\max_a Q(s,a)$。

### 13.2 必背公式

策略梯度：

$$
\nabla \bar{R}_{\theta}
\approx
\frac{1}{N}
\sum_{n,t}
(G_t^n-b)
\nabla\log\pi_\theta(a_t^n\mid s_t^n)
$$

价值函数：

$$
V_\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]
$$

$$
Q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
$$

优势函数：

$$
A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s)
$$

TD 误差：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

A2C 更新：

$$
\nabla \bar{R}_\theta
\approx
\sum_t
\delta_t
\nabla\log\pi_\theta(a_t\mid s_t)
$$

路径衍生策略梯度：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{s}
\left[
\nabla_a Q_w(s,a)\big|_{a=\mu_\theta(s)}
\nabla_\theta\mu_\theta(s)
\right]
$$

### 13.3 一句话总结

演员-评论员算法用评论员估计价值、降低策略梯度的方差；A2C 用 TD 误差近似优势函数，A3C 用异步并行提高采样效率，路径衍生策略梯度则利用评论员对动作的梯度来处理连续动作控制问题。
