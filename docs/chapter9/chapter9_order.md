# 第 9 章 演员-评论员算法：学习顺序整理版

本文是对 `chapter9.md` 的重新整理。原文按照主题逐段展开，这里改成更适合学习的路径：先明确要解决的问题，再建立概念依赖，最后比较 A2C、A3C、路径衍生策略梯度与 GAN 的关系。

## 0. 本章先抓住一个主线

第 9 章的核心问题是：

> REINFORCE 直接用完整回报 $G_t$ 更新策略，但 $G_t$ 方差大、更新慢。能不能用一个价值函数来评价动作，从而让策略更新更稳定、更高效？

演员-评论员算法的答案是：

- **演员（Actor）**：策略网络 $\pi_\theta(a\mid s)$，负责决定动作。
- **评论员（Critic）**：价值网络，负责评价当前策略下的状态或动作有多好。
- **优势函数（Advantage）**：告诉演员“这个动作比当前状态下的平均水平好多少”。
- **TD 误差**：用一步奖励和下一个状态价值近似优势，从而不必等完整轨迹结束。

本章可以按下面的路线学习：

```text
REINFORCE 的高方差问题
        ↓
用 Q(s,a) 估计回报期望
        ↓
用 V(s) 当基线，得到优势函数 A(s,a)
        ↓
用 TD 误差近似优势，只训练 V 网络
        ↓
A2C：同步的优势演员-评论员
        ↓
A3C：多个 worker 异步并行采样与更新
        ↓
路径衍生策略梯度：连续动作中让 actor 近似 argmax_a Q(s,a)
        ↓
与 GAN 的类比：actor 像生成器，critic 像判别器
```

## 1. 前置回顾：为什么 REINFORCE 不够稳定

策略梯度方法直接优化策略参数 $\theta$。在更新策略时，原文给出的梯度估计为：

$$
\nabla \bar{R}_{\theta}
\approx
\frac{1}{N}
\sum_{n=1}^{N}
\sum_{t=1}^{T_n}
\left(
\sum_{t'=t}^{T_n}
\gamma^{t'-t} r_{t'}^n - b
\right)
\nabla \log p_\theta(a_t^n\mid s_t^n)
\tag{9.1}
$$

这里可以把括号中的内容理解成：

$$
G_t - b
$$

其中：

- $G_t$ 是从时间 $t$ 开始直到回合结束的折扣累积回报；
- $b$ 是基线，用来降低方差；
- $\nabla \log p_\theta(a_t\mid s_t)$ 决定怎样调整当前动作概率。

学习这条公式时，先不要陷入符号细节，先理解一句话：

> 如果某个动作带来的回报高于基线，就提高它以后被选中的概率；如果低于基线，就降低它以后被选中的概率。

问题在于，$G_t$ 是采样得到的随机变量。即使在同一个状态采取同一个动作，后续环境变化、策略采样和长时间累积都会让 $G_t$ 产生很大波动。因此 REINFORCE 的典型问题是：

- 必须等完整轨迹结束后才能计算回报；
- 单条或少量轨迹的估计噪声很大；
- 更新方向可能因为偶然采样而偏差很大；
- 学习效率低，训练过程容易不稳定。

所以本章后续所有方法，本质上都在解决同一个问题：

> 怎样不用高方差的完整采样回报 $G_t$，而用更稳定的价值估计来指导策略更新？

图 9.1 是原文对策略梯度回顾的示意：

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/5706e9cbb7554d04a70614ea0e79372727abad72c9ae40dba1190b958fb8900e"/>
</div>
<div align=center>图 9.1 策略梯度回顾</div>

## 2. 从回报采样到价值估计：引入评论员

### 2.1 $G_t$ 的期望就是价值

如果 $G_t$ 太不稳定，一个自然想法是：不要直接用一次采样到的 $G_t$，而是估计它的期望。

对状态-动作对来说：

$$
\mathbb{E}[G_t \mid s_t=s, a_t=a] = Q_\pi(s,a)
$$

这就是动作价值函数 $Q_\pi(s,a)$ 的含义：

> 在状态 $s$ 采取动作 $a$，之后按照策略 $\pi$ 行动，最终能得到的累积回报期望。

对状态本身来说：

$$
V_\pi(s) = \mathbb{E}_{a\sim\pi(\cdot\mid s)}[Q_\pi(s,a)]
$$

也就是：

> 在状态 $s$ 按照当前策略行动，平均能得到多少回报。

### 2.2 $V$ 和 $Q$ 的区别

| 函数 | 输入 | 输出 | 含义 |
|---|---|---|---|
| $V_\pi(s)$ | 状态 $s$ | 一个标量 | 当前状态在策略 $\pi$ 下的平均价值 |
| $Q_\pi(s,a)$ | 状态 $s$ 和动作 $a$ | 一个标量 | 当前状态下指定动作 $a$ 的价值 |

可以这样记：

- $V(s)$ 问的是：“这个状态总体上好不好？”
- $Q(s,a)$ 问的是：“在这个状态下做这个动作好不好？”

图 9.2 是原文对深度 Q 网络和两类价值函数的回顾：

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/393de13e995546ab9d00c0247eff6e78e9dcd58bb97d4430a8baed2d9081fecf"/>
</div>
<div align=center>图 9.2 深度 Q 网络</div>

## 3. 优势函数：评论员到底要告诉演员什么

### 3.1 从基线到优势

在策略梯度公式里，原本用的是：

$$
G_t - b
$$

如果我们用价值函数作为基线，并把 $G_t$ 的期望替换为 $Q_\pi(s_t,a_t)$，就得到：

$$
A_\pi(s_t,a_t)
=
Q_\pi(s_t,a_t) - V_\pi(s_t)
$$

这就是优势函数。

它的意义非常关键：

> $A_\pi(s,a)$ 衡量动作 $a$ 相对于状态 $s$ 下平均动作水平的好坏。

因此：

- $A_\pi(s,a) > 0$：这个动作比平均水平好，应提高选择概率；
- $A_\pi(s,a) < 0$：这个动作比平均水平差，应降低选择概率；
- $A_\pi(s,a) \approx 0$：这个动作差不多是平均水平，策略不需要强烈改变。

这比直接问“动作好不好”更精确，因为它问的是：

> 在当前状态下，这个动作相对于其他可能动作是否更好？

图 9.3 是原文中的优势演员-评论员示意：

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/d02364e1abb64a3db51e2e6cbef2322f4497a4eda56d47a3911a0ba517461800"/>
</div>
<div align=center>图 9.3 优势演员-评论员算法</div>

### 3.2 为什么不直接同时估计 $Q$ 和 $V$

如果直接按定义使用优势函数，需要两个网络：

- 一个 $Q_\pi(s,a)$ 网络；
- 一个 $V_\pi(s)$ 网络。

这会带来两个问题：

- 训练成本更高；
- 两个估计都可能不准，误差会一起影响策略更新。

所以 A2C/A3C 常用一个更简单的做法：

> 只训练 $V(s)$，用一步 TD 误差近似优势。

## 4. TD 误差：把优势函数变成可训练的单步信号

根据 Bellman 思想，动作价值可以写成：

$$
Q_\pi(s_t,a_t)
=
\mathbb{E}
\left[
r_t + \gamma V_\pi(s_{t+1})
\right]
$$

原文为了直观说明，重点强调用 $r_t + V_\pi(s_{t+1})$ 近似 $Q$。更严格地写，通常会带上折扣因子 $\gamma$：

$$
Q_\pi(s_t,a_t)
\approx
r_t + \gamma V_\pi(s_{t+1})
$$

代回优势函数：

$$
A_\pi(s_t,a_t)
\approx
r_t + \gamma V_\pi(s_{t+1}) - V_\pi(s_t)
$$

右边就是一步 TD 误差：

$$
\delta_t
=
r_t + \gamma V_\pi(s_{t+1}) - V_\pi(s_t)
$$

学习时可以这样理解 TD 误差：

> 评论员原来以为状态 $s_t$ 的价值是 $V(s_t)$；但实际走了一步后发现，当前奖励加上下一个状态价值是 $r_t+\gamma V(s_{t+1})$。两者的差，就是这一步比预期好还是差。

因此 $\delta_t$ 同时有两个用途：

- 更新演员：告诉策略当前动作应该被鼓励还是抑制；
- 更新评论员：让 $V(s_t)$ 更接近 TD 目标 $r_t+\gamma V(s_{t+1})$。

## 5. A2C：优势演员-评论员算法

### 5.1 A2C 的核心结构

A2C 是 Advantage Actor-Critic，即优势演员-评论员。它包含两个部分：

| 组件 | 网络 | 作用 |
|---|---|---|
| Actor | $\pi_\theta(a\mid s)$ | 根据状态输出动作分布或连续动作 |
| Critic | $V_w(s)$ | 估计当前状态价值 |

A2C 的更新可以理解为：

```text
1. actor 根据当前策略与环境交互，得到 (s_t, a_t, r_t, s_{t+1})
2. critic 计算 TD 误差 delta_t
3. actor 用 delta_t 更新策略
4. critic 用 TD 目标更新价值函数
5. 重复采样和更新
```

### 5.2 Actor 怎样更新

用 TD 误差近似优势后，策略梯度可以写成：

$$
\nabla_\theta J(\theta)
\approx
\delta_t
\nabla_\theta \log \pi_\theta(a_t\mid s_t)
$$

其中：

$$
\delta_t
=
r_t + \gamma V_w(s_{t+1}) - V_w(s_t)
$$

直观解释：

- 如果 $\delta_t>0$，说明这个动作比评论员预期的好，提高 $\pi_\theta(a_t\mid s_t)$；
- 如果 $\delta_t<0$，说明这个动作比预期差，降低 $\pi_\theta(a_t\mid s_t)$。

### 5.3 Critic 怎样更新

评论员要让自己的估计 $V_w(s_t)$ 接近 TD 目标：

$$
y_t = r_t + \gamma V_w(s_{t+1})
$$

所以可以最小化：

$$
L_{\text{critic}}
=
\left(
y_t - V_w(s_t)
\right)^2
$$

实际训练中通常会对 $y_t$ 停止梯度，让它作为目标值，而不是让目标本身也被当前更新牵着走。

### 5.4 共享网络层

原文提到一个实现技巧：actor 和 critic 的输入都是状态 $s$，因此前几层可以共享。

特别是 Atari 这类图像输入任务中：

- 前面的卷积层负责从像素中提取高级特征；
- actor head 输出动作分布；
- critic head 输出状态价值。

图 9.5 展示了离散动作场景下的 actor-critic 网络：

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/d39a829f7e844f03aceb4243c50cb5822811cd85180c4662a77b715fde2f620f"/>
</div>
<div align=center>图 9.5 离散动作的例子</div>

### 5.5 熵正则：让策略保持探索

如果 actor 很快把概率集中到少数动作上，就会过早失去探索能力。A2C 常加入熵正则，让动作分布不要太尖锐。

对离散策略来说，熵为：

$$
H(\pi(\cdot\mid s))
=
-
\sum_a \pi(a\mid s)\log \pi(a\mid s)
$$

训练时常把 actor loss 写成：

$$
L_{\text{actor}}
=
-
\log\pi_\theta(a_t\mid s_t)\delta_t
-
\beta H(\pi_\theta(\cdot\mid s_t))
$$

其中 $\beta$ 控制探索强度。

### 5.6 A2C 要掌握的三个句子

- A2C 用 critic 的价值估计降低策略梯度的方差。
- A2C 用 TD 误差近似优势函数，不必等完整回合结束。
- A2C 的 actor 负责改策略，critic 负责改价值估计。

图 9.4 是原文中的 A2C 流程图：

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/a6096d7cce50414ebc574bccf88630900f7afcf0212e4f96bf675b817e7ec252"/>
</div>
<div align=center>图 9.4 优势演员-评论员算法流程</div>

## 6. A3C：把 A2C 并行化

### 6.1 A3C 解决什么问题

强化学习训练慢，一个原因是采样慢，而且单个智能体连续采样的数据相关性很强。

A3C 的想法是：

> 同时开多个 worker，让它们在不同环境副本中并行探索，然后异步更新一个全局网络。

原文用“影分身”类比 A3C：多个 worker 同时探索，相当于同时积累经验。

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/496461ab4b19443197b24664ce9382aebc0f59a7f14442d4b4e841aaa4cf9531"/>
</div>
<div align=center>图 9.6 影分身例子</div>

### 6.2 A3C 的工作流程

A3C 中有一个全局 actor-critic 网络，每个 worker 有自己的本地副本。

```text
1. 全局网络保存当前参数
2. worker 复制全局网络参数到本地网络
3. worker 用本地网络与自己的环境交互
4. worker 收集一小段轨迹并计算梯度
5. worker 把梯度异步提交给全局网络
6. 全局网络被更新
7. worker 再同步新的全局参数，继续采样
```

它和 A2C 的主要区别不是公式本身，而是采样和更新方式：

| 算法 | worker | 更新方式 | 特点 |
|---|---:|---|---|
| A2C | 多个或一个 | 通常同步更新 | 实现简单，更新稳定 |
| A3C | 多个 | 异步更新全局网络 | 探索更并行，样本相关性更弱 |

### 6.3 A3C 中的“异步”是什么意思

异步意味着：

- worker 之间不等待彼此；
- 每个 worker 可能基于稍旧的参数采样；
- 全局网络可能已经被其他 worker 更新过；
- 当前 worker 仍然把自己的梯度提交给全局网络。

这会带来一定的参数滞后，但也能提高采样吞吐量，并让不同 worker 的探索轨迹更分散。

图 9.7 是原文中的 A3C 运作流程：

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/c36501bcb3ad49c1be3c0486665f6d096e67c00b6d4142a9a00231ae8f299c56"/>
</div>
<div align=center>图 9.7 异步优势演员-评论员算法的运作流程</div>

## 7. 路径衍生策略梯度：连续动作下让 actor 解决 argmax

### 7.1 为什么需要路径衍生策略梯度

DQN 在离散动作空间中可以这样选动作：

$$
a^* = \arg\max_{a} Q(s,a)
$$

如果动作是离散的，可以把所有动作枚举一遍，选 Q 值最大的动作。

但如果动作是连续向量，比如机械臂关节角度、自动驾驶方向盘角度，就很难直接枚举所有动作，也很难每一步都精确求解：

$$
\arg\max_{a} Q(s,a)
$$

路径衍生策略梯度的想法是：

> 另外训练一个 actor 网络 $\mu_\theta(s)$，让它直接输出接近 $\arg\max_{a} Q(s,a)$ 的动作。

### 7.2 从 DQN 到 actor-critic 的视角转换

DQN 的动作选择是：

$$
a_t = \arg\max_{a} Q(s_t,a)
$$

路径衍生策略梯度把这个 argmax 替换成 actor：

$$
a_t = \mu_\theta(s_t)
$$

然后训练 actor，让它输出的动作能让 critic 打出更高分：

$$
J(\theta)
=
\mathbb{E}_{s}
\left[
Q_\phi(s,\mu_\theta(s))
\right]
$$

actor 的目标是最大化 $J(\theta)$，也就是最大化 critic 对它输出动作的评价。

图 9.8 是原文中的路径衍生策略梯度示意：

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/a782c1f5f1204beaa8f1083d25e8fc097d79a15a179d486994827d9b994fa546"/>
</div>
<div align=center>图 9.8 路径衍生策略梯度</div>

### 7.3 为什么叫“路径衍生”

因为梯度会沿着下面这条计算路径反向传播：

```text
s
↓
actor: a = mu_theta(s)
↓
critic: Q_phi(s, a)
↓
最大化 Q_phi(s, mu_theta(s))
```

令

$$
a_\theta = \mu_\theta(s)
$$

也就是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{s}
\left[
\nabla_a Q_\phi(s,a_\theta)
\cdot
\nabla_\theta \mu_\theta(s)
\right]
$$

这条公式的含义是：

- critic 告诉 actor：“如果动作往哪个方向变，Q 值会上升”；
- actor 根据这个方向调整自己的参数；
- 这比普通 actor-critic 只告诉“当前动作好不好”提供了更具体的信息。

### 7.4 路径衍生策略梯度的训练结构

原文把它和 DQN 进行对照。可以整理成四个替换：

| DQN 中的做法 | 路径衍生策略梯度中的做法 |
|---|---|
| 用 $\arg\max_{a} Q(s,a)$ 选动作 | 用 actor $\mu_\theta(s)$ 输出动作 |
| 目标值里需要 $\max_{a} \hat{Q}(s_{i+1},a)$ | 用目标 actor $\hat{\mu}(s_{i+1})$ 代替 argmax |
| 只训练 Q 网络 | 同时训练 critic 和 actor |
| 有目标 Q 网络 | 同时有目标 critic 和目标 actor |

对应的 critic 目标通常写作：

$$
y_i
=
r_i
+
\gamma
\hat{Q}
\left(
s_{i+1},
\hat{\mu}(s_{i+1})
\right)
$$

critic 最小化：

$$
L_{\text{critic}}
=
\left(
Q_\phi(s_i,a_i)-y_i
\right)^2
$$

actor 最大化：

$$
J(\theta)
=
Q_\phi(s_i,\mu_\theta(s_i))
$$

图 9.9 到图 9.11 展示了路径衍生策略梯度从 DQN 改造而来的过程：

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/529577a994a6499698a0e51c2ee470836b5d4d2ffccf4e40b4d760babe331944"/>
</div>
<div align=center>图 9.9 路径衍生策略梯度算法</div>

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/37f81d6827aa4996884c822f236643255da2f11fa724491bb1cd272a80e7c318"/>
</div>
<div align=center>图 9.10 深度 Q 网络算法</div>

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/f3bbec118f5f4e0bb2774cff3111a6c44aae3e670f51406296e6884d92913bcf"/>
</div>
<div align=center>图 9.11 从深度 Q 网络到路径衍生策略梯度</div>

## 8. 与 GAN 的类比

原文最后把 actor-critic 和 GAN 联系起来。这个类比可以帮助理解，但不要把两者完全等同。

| GAN | Actor-Critic |
|---|---|
| Generator 生成样本 | Actor 生成动作 |
| Discriminator 评价样本真假 | Critic 评价动作或状态价值 |
| Generator 想让 Discriminator 给高分 | Actor 想让 Critic 给高价值 |
| 两个网络互相影响，训练不稳定 | Actor 和 Critic 互相影响，训练也不稳定 |

在路径衍生策略梯度里，这个类比尤其明显：

```text
actor 输出动作 a
critic 评价 Q(s,a)
actor 调整自己，让 critic 给更高 Q 值
```

这类似：

```text
generator 输出样本
discriminator 评价样本
generator 调整自己，让 discriminator 给更高分
```

表 9.1 是原文中的 GAN 与演员-评论员联系：

<div align=center>表 9.1 与生成对抗网络的联系</div>

<div align=center>
<img width="550" src="https://ai-studio-static-online.cdn.bcebos.com/818c1c2e603341f881dd57fb59e109c81702722156be4e6481b276b894c51290"/>
</div>

## 9. 三类方法的对比

### 9.1 REINFORCE、A2C、A3C、路径衍生策略梯度

| 方法 | 主要信号 | 是否有 critic | 适合动作空间 | 关键问题 |
|---|---|---|---|---|
| REINFORCE | 完整回报 $G_t$ | 否 | 离散/连续都可 | 方差大，更新慢 |
| A2C | TD 误差 / 优势 | 是，$V(s)$ | 离散/连续都可 | critic 估计质量影响 actor |
| A3C | n-step 回报 / 优势 | 是，$V(s)$ | 离散/连续都可 | 异步实现更复杂 |
| 路径衍生策略梯度 | $\nabla_a Q(s,a)$ | 是，$Q(s,a)$ | 主要用于连续动作 | critic 误差会误导 actor |

### 9.2 $G$、$V$、$Q$、$A$、$\delta$ 的区别

| 符号 | 名称 | 直观含义 |
|---|---|---|
| $G_t$ | 采样回报 | 这一次真实采样到的未来总奖励 |
| $V_\pi(s)$ | 状态价值 | 这个状态平均有多好 |
| $Q_\pi(s,a)$ | 动作价值 | 这个状态下指定动作有多好 |
| $A_\pi(s,a)$ | 优势函数 | 这个动作比平均动作好多少 |
| $\delta_t$ | TD 误差 | 实际一步结果比 critic 预期好多少 |

### 9.3 A2C 和路径衍生策略梯度的关键区别

| 对比点 | A2C | 路径衍生策略梯度 |
|---|---|---|
| critic 输出 | $V(s)$ | $Q(s,a)$ |
| actor 更新依据 | 当前动作的 log probability 和优势 | critic 对动作的梯度 $\nabla_a Q(s,a)$ |
| 动作选择 | 通常是随机策略 $\pi(a\mid s)$ | 常见为确定性策略 $\mu(s)$ |
| 适用重点 | 通用 actor-critic 框架 | 连续动作控制 |
| critic 给 actor 的信息 | 当前动作好还是差 | 动作应该往哪个方向变好 |

## 10. 学习时最容易混淆的点

### 10.1 把 $V(s)$ 和 $Q(s,a)$ 混在一起

$V(s)$ 不关心当前具体采取哪个动作，它是当前策略下的平均价值。

$Q(s,a)$ 关心当前指定动作，它比 $V(s)$ 更具体。

如果题目问“在状态 $s$ 采取动作 $a$ 后的价值”，应该想到 $Q(s,a)$；如果题目只问“状态 $s$ 的价值”，应该想到 $V(s)$。

### 10.2 以为优势函数就是 Q 值

优势函数不是动作价值本身，而是动作价值减去状态平均价值：

$$
A(s,a)=Q(s,a)-V(s)
$$

它衡量的是相对好坏，而不是绝对价值。

### 10.3 忘记 TD 误差里的折扣因子

标准一步 TD 误差通常写作：

$$
\delta_t
=
r_t + \gamma V(s_{t+1}) - V(s_t)
$$

如果终止状态没有下一个价值，通常令：

$$
V(s_{t+1})=0
$$

### 10.4 以为 A3C 是另一个完全不同的算法

A3C 的核心公式仍然是 actor-critic，只是训练方式变成多 worker 异步并行。

可以这样记：

> A2C 主要是算法结构；A3C 主要是在 A2C 思路上加入异步并行采样和更新。

### 10.5 把路径衍生策略梯度等同于普通 A2C

普通 A2C 中，critic 通常用优势值告诉 actor 当前动作好不好。

路径衍生策略梯度中，critic 的 $Q(s,a)$ 对动作可导，因此能告诉 actor 动作应该朝哪个方向调整。

## 11. 建议学习顺序

如果第一次学本章，建议按下面顺序读：

1. 先理解 REINFORCE 的高方差问题。
2. 再理解 $V(s)$ 和 $Q(s,a)$ 的区别。
3. 然后掌握优势函数 $A(s,a)=Q(s,a)-V(s)$。
4. 接着理解 TD 误差怎样近似优势。
5. 再学 A2C 的 actor loss、critic loss 和熵正则。
6. 然后把 A3C 看成 A2C 的异步并行版本。
7. 最后学习路径衍生策略梯度，把它看成连续动作下用 actor 近似 $\arg\max_{a} Q(s,a)$。
8. GAN 类比放在最后看，用来加深理解，不要作为主线。

## 12. 对应代码阅读路线

本章代码在 `docs/chapter9/code/` 下，可以配合本文按顺序阅读：

| 文件 | 对应部分 | 重点观察 |
|---|---|---|
| `a2c.py` | 第 5 节 A2C | TD 误差如何同时训练 actor 和 critic |
| `a3c.py` | 第 6 节 A3C | 多 worker 如何异步更新全局网络 |
| `pathwise_derivative_policy_gradient.py` | 第 7 节路径衍生策略梯度 | 连续动作中 actor 如何近似 $\arg\max_{a} Q(s,a)$ |

推荐运行命令：

```powershell
conda run -n base python docs/chapter9/code/a2c.py --episodes 20 --print-every 5
conda run -n base python docs/chapter9/code/a3c.py --workers 2 --episodes-per-worker 10 --print-every 5
conda run -n base python docs/chapter9/code/pathwise_derivative_policy_gradient.py --episodes 20 --print-every 5
```

运行时重点看输出里的这些量：

- episode reward 是否逐渐提高；
- actor loss 和 critic loss 是否在合理范围变化；
- A2C 中的 TD 误差如何影响动作概率；
- A3C 中不同 worker 是否都在提交更新；
- 路径衍生策略梯度中 actor 是否逐渐输出更接近目标的连续动作。

## 13. 一页速记

本章最重要的公式：

$$
G_t
=
\sum_{k=t}^{T}
\gamma^{k-t}r_k
$$

$$
V_\pi(s)
=
\mathbb{E}_\pi[G_t \mid s_t=s]
$$

$$
Q_\pi(s,a)
=
\mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]
$$

$$
A_\pi(s,a)
=
Q_\pi(s,a)-V_\pi(s)
$$

$$
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
\nabla_\theta J(\theta)
\approx
\delta_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_s
\left[
\nabla_a Q_\phi(s,a_\theta)
\cdot
\nabla_\theta \mu_\theta(s)
\right]
,\quad
a_\theta=\mu_\theta(s)
$$

本章最重要的几句话：

- REINFORCE 用采样回报 $G_t$，方差大。
- Actor-Critic 用 critic 的价值估计帮助 actor 降低方差。
- Advantage 关心动作相对平均水平的好坏。
- TD 误差可以近似优势，让算法单步更新。
- A2C 是优势演员-评论员的同步版本。
- A3C 是多个 worker 异步并行更新的版本。
- 路径衍生策略梯度让 actor 解决连续动作中的 $\arg\max_{a} Q(s,a)$ 问题。
- GAN 类比可以帮助理解双网络互相训练，但不能替代强化学习中的回报、状态转移和策略更新概念。
