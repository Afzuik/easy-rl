# 第 5 章 PPO 算法

> **说明**：本文是 `chapter5.md` 的整理版本。相比原文，本章对知识点的组织顺序进行了优化——遵循"动机（同策略→异策略）→ 工具（重要性采样）→ 问题（方差危机）→ 解决方案（PPO 的 KL 约束与裁剪机制）→ 算法变种（PPO-Penalty vs PPO-Clip）"的认知路径，并补充了重要性采样方差的数学推导与直观解释、PPO 目标函数从梯度反推的详细过程、PPO-Penalty 自适应 β 的动机说明，以及 PPO-Clip 在不同优势符号下的行为可视化分析，便于初学者循序渐进地理解近端策略优化方法。

---

## 学习路线图

本章建议按以下顺序学习：

1. **先理解动机**：为什么策略梯度算法"采样一次、更新一次"效率低？同策略和异策略的根本区别是什么？（5.1 前半部分）
2. **再掌握工具**：重要性采样的数学原理是什么？为什么它能实现异策略学习？（5.1 后半部分）
3. **认清问题**：重要性采样有什么致命缺陷？为什么 $p$ 和 $q$ 差距大时方差会爆炸？（5.1 末尾 → 5.2 开头）
4. **学习解决方案**：PPO 如何用 KL 散度约束解决分布漂移？PPO-Penalty 和 PPO-Clip 各有什么巧妙设计？（5.2 全部）

### 阅读前先统一符号

本章会反复出现“旧策略”“新策略”“目标分布”“采样分布”。如果不先区分这些角色，后面的概率比值很容易看反。

| 符号 | 角色 | 含义 |
|---|---|---|
| $p(x)$ | 目标分布 | 我们真正想计算其期望的分布 |
| $q(x)$ | 提议分布/采样分布 | 实际负责产生样本的分布 |
| $\pi_{\theta'}$ 或 $\pi_{\theta^k}$ | 旧策略/行为策略 | 本轮与环境交互并收集数据的策略，优化期间保持固定 |
| $\pi_\theta$ | 新策略/待优化策略 | 使用旧策略的数据进行更新的策略 |
| $r_t(\theta)$ | 概率比值 | $\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta'}(a_t\mid s_t)}$，衡量采样动作在新旧策略下的概率变化 |
| $A_t$ | 优势估计 | 动作 $a_t$ 相对状态 $s_t$ 下平均行为好多少 |

> **记忆方法**：重要性权重永远是“**目标分布除以采样分布**”。PPO 中想评价的是新策略，数据来自旧策略，所以分子是新策略，分母是旧策略。

---

## 第一部分：从同策略到异策略 —— 为什么需要重要性采样

> 第 4 章介绍的基本策略梯度算法有一个根本性的效率瓶颈：每次参数更新后，旧数据就"过期"了，必须重新采样。本节分析这个问题的根源，并引入重要性采样作为解决思路。

### 5.1.1 策略梯度的效率瓶颈

#### 同策略：采样即学，学完即弃

回顾第 4 章策略梯度的核心公式：

$$
\nabla \bar{R}_{\theta}=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right] \tag{5.1}
$$

这里有一个容易被忽视但极其关键的细节：期望的下标是 $\tau \sim p_{\theta}(\tau)$，即**对当前策略 $\pi_\theta$ 采样出的轨迹求期望**。这意味着：

1. 智能体用策略 $\pi_\theta$ 与环境交互，采集一批轨迹 $\tau$；
2. 用这批数据按式(5.1)计算梯度，更新参数 $\theta \to \theta'$；
3. **关键问题来了**：参数变成 $\theta'$ 后，轨迹分布也随之改变。旧轨迹是在 $p_\theta$ 下采样的，但当前策略已经是 $p_{\theta'}$——直接把旧数据当成新策略数据会产生分布不匹配；
4. 必须用新策略 $\pi_{\theta'}$ 重新与环境交互，采集新数据。

> **一句话总结**：严格的同策略估计要求数据分布与当前策略匹配。参数更新后继续使用旧数据并非绝对不可能，但更新越多，分布偏差越明显，原策略梯度公式也越不再精确。强化学习中环境交互往往比梯度计算昂贵，因此需要一种受控的数据复用方式。

#### 同策略 vs 异策略的直观对比

| | 同策略（on-policy） | 异策略（off-policy） |
|---|---|---|
| **交互智能体** | $\pi_\theta$（正在学习的策略） | $\pi_{\theta'}$（固定的"示范"策略） |
| **学习智能体** | $\pi_\theta$ | $\pi_\theta$（学习目标不变） |
| **数据复用** | 只能短期复用当前策略刚采集的数据 | 可以长期复用其他策略产生的数据 |
| **效率** | 低——大量时间花在采样上 | 高——一次采样，多次学习 |
| **代表算法** | 基本策略梯度、REINFORCE、A2C、PPO | Q-learning、DQN、SAC |

> **核心动机**：如果我们能让一个固定的行为策略 $\pi_{\theta'}$ 去与环境交互，采集一批数据，然后用这批数据多次训练 $\pi_\theta$，就能提升数据利用率。重要性采样提供了这种分布修正工具。

这里还要区分“**短期复用**”和“**长期复用**”：

- PPO 会将当前旧策略采集的一批数据切成小批量，训练多个 epoch，因此同一批数据会被短期使用多次；
- 但完成这轮更新后，PPO 通常丢弃这批数据，用最新策略重新采样；
- 它不像 DQN、SAC 那样使用经验回放池，长期混合复用许多历史策略产生的数据。

因此，PPO 使用了重要性比值，却仍然通常被归类为**同策略算法**。

---

## 第二部分：重要性采样 —— 异策略学习的数学桥梁

> 怎么实现"用策略 B 采样的数据来训练策略 A"？答案是重要性采样（importance sampling）——一种通过加权修正不同分布间差异的通用统计技术。

### 5.2.1 重要性采样的基本概念

#### 从采样到期望：基础设定

假设我们想计算函数 $f(x)$ 在分布 $p(x)$ 下的期望值：

$$
\mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) \mathrm{d}x
$$

但问题是我们**不能从 $p(x)$ 直接采样**（或者采样代价太高），只能从另一个分布 $q(x)$ 采样。怎么办？

如果能够直接从 $p$ 采样，取 $N$ 个独立样本 $x^{(1)},\ldots,x^{(N)}\sim p$，就可以使用蒙特卡洛平均：

$$
\mathbb{E}_{x\sim p}[f(x)]
\approx
\frac{1}{N}\sum_{i=1}^{N}f\left(x^{(i)}\right)
$$

但如果样本来自 $q$，直接平均 $f(x^{(i)})$ 得到的是 $\mathbb E_q[f]$，而不是 $\mathbb E_p[f]$。重要性权重的作用，就是把每个来自 $q$ 的样本重新加权，使其能够代表 $p$。

#### 重要性采样的核心推导

对期望公式做一个巧妙的恒等变形——分子分母同乘 $q(x)$：

$$
\int f(x) p(x) \mathrm{d}x
= \int f(x) \frac{p(x)}{q(x)} q(x) \mathrm{d}x
= \mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right] \tag{5.3}
$$

这就得到了重要性采样的核心结论：

$$
\boxed{\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[f(x) \cdot \frac{p(x)}{q(x)}\right]}
$$

其中 $\frac{p(x)}{q(x)}$ 称为**重要性权重（importance weight）**，它修正了从 $q$ 采样替代从 $p$ 采样带来的偏差。

> **直观理解**：如果某个 $x$ 在 $p$ 下出现的概率比在 $q$ 下大（$p(x) > q(x)$），那么我们采样到这种 $x$ 的次数就"偏少了"，需要给它的 $f(x)$ 乘上一个大于 1 的权重来补偿；反之，如果 $p(x) < q(x)$，则乘上一个小于 1 的权重来抑制。

#### 什么是支撑集，为什么必须覆盖

分布的**支撑集（support）**可以直观理解为“这个分布可能产生哪些值”。离散情况下：

$$
\operatorname{supp}(p)=\{x\mid p(x)>0\}
$$

连续分布中，更严格的定义会涉及闭包；初学时可以把它理解为概率密度不为零的区域。

重要性采样要求：

$$
\boxed{\operatorname{supp}(p)\subseteq\operatorname{supp}(q)}
$$

也就是：

$$
p(x)>0\Longrightarrow q(x)>0
$$

原因并不只是“除数不能为零”。更根本的原因是：如果某个区域在 $p$ 下可能出现，但在 $q$ 下概率为零，那么无论从 $q$ 采样多少次都不可能看到这个区域。权重只能调整**已经采到的样本**，不能创造从未出现过的样本信息。

例如：

| $x$ | $p(x)$ | $q(x)$ |
|---|---:|---:|
| A | 0.5 | 1.0 |
| B | 0.5 | 0 |

真实期望包含 $0.5f(B)$，但从 $q$ 永远采不到 B，而且权重 $\frac{p(B)}{q(B)}=\frac{0.5}{0}$ 无定义。因此 $q$ 必须覆盖 $p$ 的支撑集。

#### 有限样本下真正计算的估计量

实际计算时，我们从 $q$ 采样 $x^{(i)}$，使用：

$$
\boxed{
\hat{\mu}_{\mathrm{IS}}
=
\frac{1}{N}\sum_{i=1}^{N}
f\left(x^{(i)}\right)
\frac{p\left(x^{(i)}\right)}{q\left(x^{(i)}\right)},
\qquad x^{(i)}\sim q
}
$$

式(5.3)说明这个估计量在条件满足时是无偏的；但无偏只表示重复无数次实验后的平均结果正确，不表示某一次有限样本估计一定准确。

### 5.2.2 重要性采样的方差陷阱 ⚠️ 重点难点

#### 期望相同 ≠ 方差相同

式(5.3)保证了期望值的一致性，但**方差不一致**。我们来推导两者的方差差异。

两个随机变量的方差分别为：

$$
\operatorname{Var}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim p}\left[f(x)^{2}\right] - \left(\mathbb{E}_{x \sim p}[f(x)]\right)^{2}
$$

$$
\begin{aligned}
\operatorname{Var}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]
&= \mathbb{E}_{x \sim q}\left[\left(f(x) \frac{p(x)}{q(x)}\right)^{2}\right] - \left(\mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]\right)^{2} \\
&= \mathbb{E}_{x \sim p}\left[f(x)^{2} \frac{p(x)}{q(x)}\right] - \left(\mathbb{E}_{x \sim p}[f(x)]\right)^{2}
\end{aligned}
$$

上面的第二个等号容易让人误以为平方“消失”了。把两项分别写成积分就清楚了。

第一项：

$$
\begin{aligned}
\mathbb E_q\left[\left(f(x)\frac{p(x)}{q(x)}\right)^2\right]
&=\int q(x)f(x)^2\frac{p(x)^2}{q(x)^2}\mathrm dx\\
&=\int f(x)^2\frac{p(x)^2}{q(x)}\mathrm dx\\
&=\int p(x)\left[f(x)^2\frac{p(x)}{q(x)}\right]\mathrm dx\\
&=\mathbb E_p\left[f(x)^2\frac{p(x)}{q(x)}\right].
\end{aligned}
$$

平方确实出现过，只是：

1. 在 $q$ 下求期望会额外乘一个 $q(x)$，约掉分母中的一个 $q(x)$；
2. 再拿出一个 $p(x)$ 作为“在 $p$ 下求期望”的概率密度；
3. 括号内最终剩下一个 $\frac{p(x)}{q(x)}$。

第二项：

$$
\begin{aligned}
\mathbb E_q\left[f(x)\frac{p(x)}{q(x)}\right]
&=\int q(x)f(x)\frac{p(x)}{q(x)}\mathrm dx\\
&=\int f(x)p(x)\mathrm dx\\
&=\mathbb E_p[f(x)].
\end{aligned}
$$

所以第二项中的权重不是被随意删掉，而是通过重要性采样恒等式整体转换成了 $\mathbb E_p[f(x)]$。

两个方差公式的第二项相同（都是 $(\mathbb{E}[f(x)])^2$），**差别在第一项**：

$$
\begin{aligned}
\text{从 } p \text{ 采样：} &\quad \mathbb{E}_{x \sim p}\left[f(x)^{2}\right] \\
\text{从 } q \text{ 采样+重要性权重：} &\quad \mathbb{E}_{x \sim p}\left[f(x)^{2} \cdot \frac{p(x)}{q(x)}\right]
\end{aligned}
$$

后者的第一项多乘了一个 $\frac{p(x)}{q(x)}$。**如果 $\frac{p(x)}{q(x)}$ 在某些区域远大于 1，方差会被剧烈放大。**

#### 图 5.1 的直观解释 ⚠️ 重点

考虑图 5.1 所示的情景：
- 蓝线 $p(x)$：概率集中在左侧；
- 绿线 $q(x)$：概率集中在右侧；
- 红线 $f(x)$：在左侧为负，右侧为正。

<div align=center>
<img width="550" src="../img/ch5/5.1.png"/>
</div>
<div align=center>图 5.1 重要性采样的问题</div>

如果从 $p$ 采样：大部分样本落在左侧（$p$ 概率高的区域），$f(x) < 0$，因此 $\mathbb{E}_{x \sim p}[f(x)]$ 是**负的**。

如果从 $q$ 采样且采样次数不够多：大部分样本落在右侧（$q$ 概率高的区域），$f(x) > 0$，$\frac{p(x)}{q(x)}$ 也比较适中，算出的 $\mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$ 是**正的**——与真实期望符号相反！

只有在**采样足够多次**后，才会偶尔采到左侧的点。在左侧，$p(x)$ 很大而 $q(x)$ 很小，$\frac{p(x)}{q(x)}$ 是一个非常大的权重，乘上负的 $f(x)$ 产生一个巨大的负值，才能"扳回"右侧累积的正值，最终让估计值回归负数。

> **核心教训**：重要性采样在理论上是无偏的（期望正确），但在有限样本下，如果 $p$ 和 $q$ 差距太大，估计的方差会非常大，导致实际结果极不稳定——可能得到符号都反了的错误结论。

---

## 第三部分：从重要性采样到异策略策略梯度

### 5.3.1 将重要性采样应用于策略梯度

将重要性采样的思想应用到策略梯度中：我们不再用 $\theta$ 去采样，而是用一个固定的示范策略 $\theta'$ 去采样轨迹，再通过重要性权重修正：

$$
\nabla \bar{R}_{\theta}
= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_{\theta}(\tau)\right] \tag{5.4}
$$

**各符号含义**：
- $\tau \sim p_{\theta'}(\tau)$：轨迹由示范策略 $\theta'$ 采样；
- $\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)}$：重要性权重，修正两个策略下轨迹概率的差异；
- $R(\tau) \nabla \log p_{\theta}(\tau)$：策略梯度项，对 $\theta$ 求导。

> **关键好处**：$\theta'$ 只需采样一次（可以多采一些数据），然后 $\theta$ 可以基于这批数据反复做多次梯度上升——因为数据是从固定策略 $\theta'$ 采样的，与 $\theta$ 的变化无关。

#### 轨迹概率到底包含什么

一条轨迹可以写成：

$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T)
$$

在策略 $\pi_\theta$ 下，它的概率为：

$$
p_\theta(\tau)
=
p(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t)
$$

如果环境动力学 $P$ 和初始状态分布 $p(s_0)$ 不随策略参数改变，那么轨迹级重要性权重可以化简为：

$$
\begin{aligned}
\frac{p_\theta(\tau)}{p_{\theta'}(\tau)}
&=
\frac{
p(s_0)\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)
}{
p(s_0)\prod_t\pi_{\theta'}(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)
}\\
&=
\prod_{t=0}^{T-1}
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta'}(a_t\mid s_t)}.
\end{aligned}
$$

环境转移概率之所以能够消掉，是因为分子和分母描述的是**同一条已发生轨迹**，且环境本身相同。最终差异只来自新旧策略选择这些动作的概率不同。

但这个乘积会带来严重的方差问题。假设每一步比值都是 $1.1$，长度为 100 的轨迹权重为：

$$
1.1^{100}\approx 13780
$$

如果每一步都是 $0.9$：

$$
0.9^{100}\approx 0.000027
$$

因此实际策略优化通常避免直接使用完整的长轨迹乘积，而采用逐时间步的替代目标。

### 5.3.2 从轨迹级到状态-动作级的细化

实际实现中，策略梯度不是对整个轨迹给一个统一分数，而是**逐对 $(s_t, a_t)$ 计算优势函数并更新**：

$$
\mathbb{E}_{(s_t, a_t) \sim \pi_{\theta}}\left[A^{\theta}(s_t, a_t) \nabla \log p_{\theta}(a_t \mid s_t)\right]
$$

其中 $A^{\theta}(s_t, a_t)$ 是**优势函数（advantage function）**，表示在状态 $s_t$ 采取动作 $a_t$ 比平均水平好多少（累积奖励减去基线）。

更严格地说：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

- $Q^\pi(s,a)$：在状态 $s$ 先采取动作 $a$，之后遵循策略 $\pi$ 的期望回报；
- $V^\pi(s)$：在状态 $s$ 直接遵循策略 $\pi$ 的平均期望回报；
- $A^\pi(s,a)>0$：动作 $a$ 比当前策略在该状态下的平均选择更好；
- $A^\pi(s,a)<0$：动作 $a$ 比平均选择更差。

实际代码中通常没有真实的 $Q^\pi$ 和 $V^\pi$，因此会用回报、TD 误差或 GAE 来估计 $A_t$。PPO 的策略目标一般把优势估计视为固定训练信号，不通过它反向传播到策略网络。

应用重要性采样后：

$$
\mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{p_{\theta}(s_t, a_t)}{p_{\theta'}(s_t, a_t)} A^{\theta'}(s_t, a_t) \nabla \log p_{\theta}(a_t \mid s_t)\right] \tag{5.5}
$$

注意：这里 $A^{\theta}$ 变成了 $A^{\theta'}$——因为数据是由 $\theta'$ 采样得到的，优势函数也应基于 $\theta'$ 的经验来估计。

### 5.3.3 重要性权重的简化推导 ⚠️ 重点

将联合概率 $p(s_t, a_t)$ 分解：

$$
\begin{aligned}
p_{\theta}(s_t, a_t) &= p_{\theta}(a_t \mid s_t) \cdot p_{\theta}(s_t) \\
p_{\theta'}(s_t, a_t) &= p_{\theta'}(a_t \mid s_t) \cdot p_{\theta'}(s_t)
\end{aligned}
$$

代入重要性权重：

$$
\frac{p_{\theta}(s_t, a_t)}{p_{\theta'}(s_t, a_t)}
= \frac{p_{\theta}(a_t \mid s_t)}{p_{\theta'}(a_t \mid s_t)} \cdot \frac{p_{\theta}(s_t)}{p_{\theta'}(s_t)}
$$

这里的 $p_\theta(s_t)$ 叫作策略诱导的**状态访问分布（state visitation distribution）**：按照策略 $\pi_\theta$ 与环境交互时，在时刻 $t$ 到达状态 $s_t$ 的概率。它不是环境单独决定的，因为策略采取的动作会影响后续访问哪些状态。

例如在岔路环境中，一个策略总是向左，另一个策略总是向右，它们访问到的状态分布显然会有很大差异。因此，“状态与动作关系不大”不能作为一般性结论。

在 PPO 的替代目标中忽略状态分布比值，主要基于以下考虑：

1. **状态分布比值难以计算**：它取决于初始状态、环境转移和之前所有动作，需要对大量可能的历史轨迹求和；
2. **动作概率比值容易计算**：给定数据中的 $s_t$，策略网络可以直接输出 $\pi_\theta(a_t\mid s_t)$ 和 $\pi_{\theta'}(a_t\mid s_t)$；
3. **新旧策略被限制得很接近**：如果策略每轮只改变一点，状态访问分布通常也不会立即发生巨变，此时忽略该比值的近似更合理；
4. **这是替代目标，不是完全等价变形**：PPO 用一个可计算、局部可靠的目标近似真实策略性能变化，再用裁剪或 KL 约束控制近似误差。

因此简化为：

$$
\frac{p_{\theta}(s_t, a_t)}{p_{\theta'}(s_t, a_t)}
\approx
\frac{p_{\theta}(a_t \mid s_t)}{p_{\theta'}(a_t \mid s_t)}
$$

而 $\frac{p_{\theta}(a_t \mid s_t)}{p_{\theta'}(a_t \mid s_t)}$ 是非常容易计算的——只需将 $s_t$ 分别输入策略网络 $\pi_\theta$ 和 $\pi_{\theta'}$，取对应动作 $a_t$ 的概率，做比值即可。

#### 概率比值应该怎样读

通常记：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta'}(a_t\mid s_t)}
$$

它比较的是：**同一个已采样动作 $a_t$，在同一个状态 $s_t$ 下，新策略相对于旧策略给了它多大概率。**

| $r_t(\theta)$ | 含义 |
|---:|---|
| $1$ | 新旧策略对该动作的概率相同 |
| $1.2$ | 新策略选择该动作的概率是旧策略的 1.2 倍 |
| $0.8$ | 新策略选择该动作的概率是旧策略的 0.8 倍 |
| 很大 | 旧策略很少选该动作，新策略却很偏爱它，权重和方差可能变大 |
| 接近 0 | 新策略几乎不再选择旧策略采到的这个动作 |

注意，$r_t$ 不是“新策略整体比旧策略好多少”，也不是两个策略参数的比值。它只描述一个具体状态—动作样本的概率变化。

### 5.3.4 从梯度反推目标函数

利用恒等式 $\nabla f(x) = f(x) \nabla \log f(x)$，我们可以从式(5.5)的梯度形式反推出目标函数（注意对 $\theta$ 求梯度时，$p_{\theta'}$ 和 $A^{\theta'}$ 都是常数）：

$$
J^{\theta'}(\theta)
=
\mathbb{E}_{(s_t,a_t)\sim\pi_{\theta'}}
\left[
\frac{p_\theta(a_t\mid s_t)}
{p_{\theta'}(a_t\mid s_t)}
A^{\theta'}(s_t,a_t)
\right]
$$

这就是**异策略策略梯度的目标函数**。我们用 $\theta'$ 采样，优化 $\theta$——$J^{\theta'}(\theta)$ 中括号外的 $\theta'$ 表示数据来源，括号内的 $\theta$ 表示优化变量。

为什么对这个目标求导会得到概率比值乘对数梯度？因为：

$$
\begin{aligned}
\nabla_\theta
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta'}(a_t\mid s_t)}
&=
\frac{1}{\pi_{\theta'}(a_t\mid s_t)}
\nabla_\theta\pi_\theta(a_t\mid s_t)\\
&=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta'}(a_t\mid s_t)}
\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\end{aligned}
$$

旧策略概率位于分母，但在本轮优化期间 $\theta'$ 固定，所以它对 $\theta$ 来说是常数。实现时通常直接保存旧策略对采样动作的 `log_prob_old`，并计算：

$$
r_t(\theta)
=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\theta'}(a_t\mid s_t)
\right)
$$

使用对数概率再取指数通常比直接相除更数值稳定。

---

## 第四部分：近端策略优化（PPO）—— 解决分布漂移

> 重要性采样允许 PPO 在同一批旧策略数据上更新新策略多次，但也带来了方差和近似误差问题：$p_\theta$ 和 $p_{\theta'}$ 差距大时，估计会变得不稳定。PPO 的核心设计就是**约束新旧策略之间的差异**，在提高批次利用率的同时控制更新幅度。

### 5.4.1 问题定位：分布漂移

在 $\theta$ 多轮更新后，$\pi_\theta$ 和 $\pi_{\theta'}$ 的动作分布可能已经相差很远。此时：
- 重要性权重 $\frac{p_{\theta}(a_t\mid s_t)}{p_{\theta'}(a_t\mid s_t)}$ 可能非常大或非常小；
- 按照 5.2.2 节的分析，方差急剧增大；
- 梯度估计变得不可靠，训练可能崩溃。

> **PPO 的核心思路**：在优化 $J^{\theta'}(\theta)$ 的同时，限制 $\theta$ 与 $\theta'$ 的行为差异，让重要性采样尽量在局部可信的范围内运作。

这里的“分布漂移”可以按一个训练周期理解：

1. 先复制当前策略，得到旧策略 $\pi_{\theta'}$；
2. 用旧策略采集一批 $(s_t,a_t,r_t)$；
3. 固定旧策略概率 $\pi_{\theta'}(a_t\mid s_t)$；
4. 对新策略 $\pi_\theta$ 做多个 epoch 的梯度更新；
5. 更新越多，$\pi_\theta$ 越可能远离产生数据的 $\pi_{\theta'}$，概率比值也越不可靠；
6. 完成本轮后令 $\theta'\leftarrow\theta$，重新采样。

旧策略必须在第 3～4 步保持固定。否则分母也随优化一起变化，概率比值就失去了“新策略相对于数据来源策略”的参照意义。

### 5.4.2 PPO 的目标函数

PPO 的共同目标是限制新旧策略差异，但有两种主要实现方式。本节先介绍 **PPO-Penalty**：在原始替代目标上增加 KL 散度惩罚项；后面的 PPO-Clip 则不把 KL 直接写入目标函数。

$$
\boxed{J_{\mathrm{PPO}}^{\theta'}(\theta) = \underbrace{J^{\theta'}(\theta)}_{\text{异策略目标}} - \underbrace{\beta \cdot \mathrm{KL}(\theta, \theta')}_{\text{KL 散度约束}}} \tag{5.6}
$$

其中：
- $J^{\theta'}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{p_{\theta}(a_t\mid s_t)}{p_{\theta'}(a_t\mid s_t)} A^{\theta'}(s_t, a_t)\right]$ 是异策略目标；
- $\mathrm{KL}(\theta, \theta')$ 衡量两个策略在**动作分布**上的差异（不是参数距离）；
- $\beta$ 是惩罚系数，控制约束的强度。

> **注意**：虽然 PPO 的推导涉及重要性采样（异策略技术），但由于 KL 约束强制 $\theta \approx \theta'$，行为策略和目标策略几乎相同，**PPO 在实践中被视为同策略算法**。

更准确地说，PPO 只复用**当前旧策略刚采集的批次**，并限制新策略不要偏离该旧策略太远；它通常不会从长期经验回放池中任意抽取旧数据。因此，重要性比值在 PPO 中主要用于支持一批数据上的多轮小步更新，而不是把 PPO 变成可以无限复用历史经验的异策略算法。

### 5.4.3 TRPO：PPO 的前身

**信任区域策略优化（TRPO）** 是 PPO 的直接前身，其形式为：

$$
\begin{aligned}
J_{\mathrm{TRPO}}^{\theta'}(\theta) &= \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t)\right] \\
\text{s.t. } &\quad \mathrm{KL}(\theta, \theta') < \delta
\end{aligned}
$$

| 对比维度 | TRPO | PPO |
|---------|------|-----|
| **策略差异的处理** | KL 作为硬约束（$\mathrm{KL} < \delta$） | KL 软惩罚，或使用裁剪替代目标 |
| **优化方式** | 需要二阶优化（共轭梯度 + 线搜索） | 一阶梯度上升即可 |
| **实现难度** | 高——约束优化很复杂 | 低——无约束优化 |
| **性能** | 好 | 与 TRPO 相当 |
| **实际使用** | 实现复杂，使用相对较少 | 实现简单，应用更广泛 |

> **为什么 PPO 更好实现？** 带约束的优化（TRPO）通常需要复杂的二阶方法或投影步骤，而 PPO 将 KL 散度作为正则化项直接加入目标函数，一阶梯度方法就能直接优化——实现简单得多，性能却不差。

### 5.4.4 KL 散度的含义：行为距离 vs 参数距离 ⚠️ 重点难点

这里有一个容易混淆的点：PPO 中的 $\mathrm{KL}(\theta, \theta')$ 到底衡量什么？

**❌ 不是参数距离**：不是计算两组神经网络权重 $\theta$ 和 $\theta'$ 之间的 L2 距离或余弦相似度。

**✅ 是行为距离（behavior distance）**：给定同一个状态 $s$，两个策略分别输出动作概率分布 $\pi_\theta(\cdot\mid s)$ 和 $\pi_{\theta'}(\cdot\mid s)$，计算这两个**概率分布之间的 KL 散度**，然后对所有状态取平均：

$$
\mathrm{KL}(\theta, \theta')
=
\mathbb{E}_{s \sim \pi_{\theta'}}
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta'}(\cdot\mid s)
\,\|\,
\pi_\theta(\cdot\mid s)
\right)
\right]
$$

对于离散动作，在某个固定状态 $s$ 上：

$$
D_{\mathrm{KL}}
\left(
\pi_{\theta'}(\cdot\mid s)
\|
\pi_\theta(\cdot\mid s)
\right)
=
\sum_a
\pi_{\theta'}(a\mid s)
\log
\frac{\pi_{\theta'}(a\mid s)}
{\pi_\theta(a\mid s)}
$$

它有三个重要性质：

1. $D_{\mathrm{KL}}\ge 0$；
2. 当两个动作分布完全相同时，$D_{\mathrm{KL}}=0$；
3. KL 散度不对称，一般有 $D_{\mathrm{KL}}(p\|q)\ne D_{\mathrm{KL}}(q\|p)$。

“对所有状态取平均”在实际中并不是遍历整个状态空间，而是对旧策略采样批次中的状态求样本平均：

$$
\widehat{\mathrm{KL}}
\approx
\frac{1}{N}
\sum_{t=1}^{N}
D_{\mathrm{KL}}
\left(
\pi_{\theta'}(\cdot\mid s_t)
\|
\pi_\theta(\cdot\mid s_t)
\right)
$$

> **为什么用行为距离而不是参数距离？** 神经网络参数与输出行为之间是非线性映射——参数变化一点，动作分布可能天翻地覆；参数变化很大，动作分布也可能几乎不变。我们真正关心的是**策略的行为是否一致**，而非参数值是否接近。重要性采样的方差取决于动作概率的比值，而非参数空间的 L2 距离。

---

## 第五部分：PPO 的两种算法变种

> PPO 有两个主要实现版本：PPO-Penalty（PPO1）用自适应 KL 惩罚，PPO-Clip（PPO2）用裁剪技巧。两者都旨在限制新旧策略的差异，但采用了不同的技术路线。

### 5.5.1 PPO-Penalty（PPO1）：自适应 KL 惩罚

#### 算法框架

在每次迭代中：

1. 用当前策略 $\theta^k$ 与环境交互，采集大量 $(s_t, a_t)$ 对；
2. 估计优势函数 $A^{\theta^k}(s_t, a_t)$；
3. **多次更新 $\theta$**（这是异策略的好处！），每次最大化：

$$
J_{\mathrm{PPO}}^{\theta^{k}}(\theta)
=
\sum_{(s_t,a_t)}
\frac{p_{\theta}(a_t\mid s_t)}
{p_{\theta^k}(a_t\mid s_t)}
A^{\theta^k}(s_t,a_t)
-
\beta\,\mathrm{KL}(\theta,\theta^k)
\tag{5.7}
$$

4. 更新多轮后，用新的 $\theta$ 作为下一轮的 $\theta^{k+1}$，重新采样。

#### 自适应 KL 系数 ⚠️ 重点

$\beta$ 的选择至关重要：
- $\beta$ 太小 → KL 惩罚不起作用，$\theta$ 和 $\theta^k$ 差异过大，重要性采样失效；
- $\beta$ 太大 → 优化被 KL 项主导，$\theta$ 几乎不更新，学习停滞。

PPO 论文提出了**自适应 KL 散度（adaptive KL divergence）**策略，动态调整 $\beta$：

| 条件 | 含义 | 操作 |
|------|------|------|
| $\mathrm{KL}(\theta, \theta^k) > \mathrm{KL}_{\max}$ | KL 散度过大，约束太松 | **增大 $\beta$**（乘以 2） |
| $\mathrm{KL}(\theta, \theta^k) < \mathrm{KL}_{\min}$ | KL 散度过小，约束太紧 | **减小 $\beta$**（除以 2） |
| 其他 | KL 散度在可接受范围内 | 保持 $\beta$ 不变 |

> **直观理解**：这就像一个自动调节的"松紧带"——如果发现策略更新幅度太大（KL 超标），就收紧约束；如果发现策略几乎没更新（KL 太小），就放松约束，让学习能继续推进。

**完整算法流程**：

1. 设定 $\mathrm{KL}_{\max}$ 和 $\mathrm{KL}_{\min}$（如 0.01 和 0.005），初始化 $\beta$；
2. 每次用 $\theta^k$ 采样后，多轮优化 $\theta$；
3. 每轮优化后计算 $\mathrm{KL}(\theta, \theta^k)$；
4. 根据上述规则调整 $\beta$；
5. 如果 KL 散度持续超出范围，可提前终止本轮优化。

### 5.5.2 PPO-Clip（PPO2）：裁剪技巧 ⚠️ 重点难点

> 如果觉得计算 KL 散度太麻烦，PPO-Clip 提供了一种更简洁的方案——直接用裁剪（clipping）来限制策略更新的幅度。

#### 目标函数

先定义概率比值：

$$
r_t(\theta)
=
\frac{p_\theta(a_t\mid s_t)}
{p_{\theta^k}(a_t\mid s_t)}
$$

则裁剪目标可写成：

$$
\begin{aligned}
J_{\mathrm{PPO2}}^{\theta^k}(\theta)
\approx
\sum_{(s_t,a_t)}
\min\Bigl(
&r_t(\theta)A^{\theta^k}(s_t,a_t),\\
&\operatorname{clip}
\bigl(r_t(\theta),1-\varepsilon,1+\varepsilon\bigr)
A^{\theta^k}(s_t,a_t)
\Bigr)
\end{aligned}
\tag{5.8}
$$

#### 裁剪函数的定义

$$
\operatorname{clip}(x, 1-\varepsilon, 1+\varepsilon) = \begin{cases}
1 - \varepsilon, & \text{如果 } x < 1 - \varepsilon \\
1 + \varepsilon, & \text{如果 } x > 1 + \varepsilon \\
x, & \text{否则}
\end{cases}
$$

以 $\varepsilon = 0.2$ 为例：如果概率比值 $\frac{p_{\theta}}{p_{\theta^k}}$ 在 $[0.8, 1.2]$ 之外，就被强制截断到边界值。

<div align=center>
<img width="550" src="../img/ch5/5.2.png"/>
</div>
<div align=center>图 5.2 裁剪函数</div>

#### 分情况分析：$\min$ 操作如何约束更新 ⚠️ 核心难点

PPO-Clip 的精妙之处在于 **$\min$ 操作 + 裁剪**的组合效果。因为训练目标要被最大化，而乘数 $A$ 可能为正也可能为负，所以必须分情况讨论。

先记：

$$
L_t^{\mathrm{clip}}(\theta)
=
\min\left(
r_tA_t,\,
\operatorname{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t
\right)
$$

##### 情况一：$A > 0$（这个动作是好的，应增大其概率）

此时增大 $r_t$ 会增大 $r_tA_t$，所以优化会提高该动作在新策略中的概率。

- 当 $r_t\le 1+\varepsilon$：目标为 $r_tA_t$，提高动作概率仍能增大目标；
- 当 $r_t>1+\varepsilon$：由于 $A_t>0$，有 $r_tA_t>(1+\varepsilon)A_t$，所以 $\min$ 选择常数 $(1+\varepsilon)A_t$，该样本不再奖励继续增大动作概率。

> **效果**：当动作概率增幅超过 $1+\varepsilon$ 时，目标函数不再给予额外奖励——防止某个好动作的概率被过度放大。

##### 情况二：$A < 0$（这个动作是不好的，应减小其概率）

此时需要先注意负数乘法会反转大小关系。减小 $r_t$ 会使 $r_tA_t$ **变得不那么负，也就是数值变大**。由于目标要最大化，优化因此会降低坏动作的概率。

- 当 $r_t\ge 1-\varepsilon$：目标为 $r_tA_t$，减小动作概率仍能提高目标；
- 当 $r_t<1-\varepsilon$：由于 $A_t<0$，乘以负数后不等号方向反转，因此 $r_tA_t>(1-\varepsilon)A_t$。此时 $\min$ 选择更小的常数 $(1-\varepsilon)A_t$，该样本不再奖励继续减小动作概率。

用具体数字检查最不容易出错。设 $A_t=-2$、$\varepsilon=0.2$、$r_t=0.5$：

$$
r_tA_t=0.5\times(-2)=-1
$$

$$
\operatorname{clip}(r_t,0.8,1.2)A_t
=0.8\times(-2)=-1.6
$$

因此：

$$
\min(-1,-1.6)=-1.6
$$

最终选择的是裁剪项，而不是原始项。这正是 $A<0$ 时下界能够生效的原因。

<div align=center>
<img width="550" src="../img/ch5/5.3.png"/>
</div>
<div align=center>图 5.3 $A$ 对裁剪函数输出的影响</div>

- **图 5.3a（$A > 0$）**：绿线 = 原始项 $rA$，蓝线 = 裁剪项 $\operatorname{clip}(r)A$。当 $r > 1+\varepsilon$，蓝线被截平，红线（$\min$）= 蓝线——**更新被截断**，不再鼓励 $p_\theta$ 继续增大。
- **图 5.3b（$A < 0$）**：当 $r < 1-\varepsilon$，负数乘法使原始项 $rA$ 反而大于裁剪项 $(1-\varepsilon)A$，红线（$\min$）选择平坦的裁剪项——**更新被截断**，不再鼓励 $p_\theta$ 继续减小。

可以用下面的分段形式总结：

$$
L_t^{\mathrm{clip}}=
\begin{cases}
A_t\min(r_t,1+\varepsilon), & A_t\ge 0\\
A_t\max(r_t,1-\varepsilon), & A_t<0
\end{cases}
$$

| 优势 | 希望的更新方向 | 何时停止提供额外收益 |
|---|---|---|
| $A_t>0$ | 增大该动作概率 | $r_t>1+\varepsilon$ |
| $A_t<0$ | 减小该动作概率 | $r_t<1-\varepsilon$ |

> **重要边界**：PPO-Clip 并不从数学上保证训练后每个 $r_t$ 都严格位于 $[1-\varepsilon,1+\varepsilon]$。它只是让“沿着优势建议的方向走得过远”不再获得额外目标收益。由于神经网络参数由许多样本共享，一个样本的更新仍可能间接改变其他样本的概率比值。因此实践中还会监控近似 KL，并在 KL 过大时提前停止当前 epoch。

### 5.5.3 PPO-Penalty vs PPO-Clip 对比

| 维度 | PPO-Penalty (PPO1) | PPO-Clip (PPO2) |
|------|-------------------|-----------------|
| **约束方式** | KL 散度作为惩罚项加入目标 | 直接裁剪概率比值 |
| **超参数** | $\beta$（可自适应调整）+ $\mathrm{KL}_{\max}$/$\mathrm{KL}_{\min}$ | $\varepsilon$（通常 0.1 或 0.2） |
| **实现复杂度** | 需要计算 KL 散度 + 自适应逻辑 | 非常简单，几行代码 |
| **调参难度** | 中等（需设置 KL 阈值） | 低（$\varepsilon$ 比较鲁棒） |
| **实际流行度** | 较少使用 | **主流选择**，实现更常见 |

### 5.5.4 PPO-Clip 的完整训练流程

只看裁剪公式容易误以为 PPO 只需要一个策略网络。实际 PPO 通常采用演员—评论员结构：

- **演员（actor）**：输出策略 $\pi_\theta(a\mid s)$；
- **评论员（critic）**：估计状态价值 $V_\phi(s)$；
- 评论员用于计算优势估计，演员根据裁剪目标更新策略。

一轮 PPO 训练可分为以下步骤。

#### 步骤 1：冻结旧策略并采样

令：

$$
\theta_{\mathrm{old}}\leftarrow\theta
$$

使用 $\pi_{\theta_{\mathrm{old}}}$ 与环境交互，保存：

$$
(s_t,a_t,r_t,d_t,\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t),V_\phi(s_t))
$$

旧动作对数概率必须保存或能够用冻结的旧策略重算，因为它是后面概率比值的分母。

#### 步骤 2：计算回报和优势

最简单的优势估计可以写成：

$$
\hat A_t=\hat Q_t-V_\phi(s_t)
$$

实践中常使用 GAE。先计算 TD 误差：

$$
\delta_t
=
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

再计算：

$$
\hat A_t
=
\delta_t
+\gamma\lambda\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}
+\cdots
$$

$\lambda$ 控制偏差—方差折中：

- $\lambda$ 较小：更依赖一步 TD，自举更强，方差较小但偏差可能较大；
- $\lambda$ 较大：更接近蒙特卡洛回报，偏差较小但方差可能较大。

通常还会对一个批次中的优势做标准化：

$$
\hat A_t
\leftarrow
\frac{\hat A_t-\operatorname{mean}(\hat A)}
{\operatorname{std}(\hat A)+\epsilon_{\mathrm{num}}}
$$

这不会改变优势的相对正负关系，但常能改善优化尺度。

#### 步骤 3：多 epoch、小批量更新

对同一批数据训练若干 epoch。每个小批量重新计算新策略的动作概率：

$$
r_t(\theta)
=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
\right)
$$

策略损失通常写成最小化形式：

$$
L_{\mathrm{policy}}
=
-
\mathbb E_t
\left[
\min\left(
r_t\hat A_t,\,
\operatorname{clip}(r_t,1-\varepsilon,1+\varepsilon)\hat A_t
\right)
\right]
$$

评论员通过价值损失更新：

$$
L_{\mathrm{value}}
=
\mathbb E_t
\left[
\left(V_\phi(s_t)-\hat V_t^{\mathrm{target}}\right)^2
\right]
$$

为了防止策略过早变成近乎确定性策略，常加入熵奖励：

$$
H\left(\pi_\theta(\cdot\mid s_t)\right)
=
-
\sum_a\pi_\theta(a\mid s_t)\log\pi_\theta(a\mid s_t)
$$

组合损失可写为：

$$
\boxed{
L
=
L_{\mathrm{policy}}
+c_vL_{\mathrm{value}}
-c_e\mathbb E_t[H(\pi_\theta(\cdot\mid s_t))]
}
$$

其中 $c_v$ 和 $c_e$ 分别控制价值损失与熵奖励的权重。

#### 步骤 4：监控更新幅度并重新采样

每个 epoch 可以监控：

- 近似 KL 散度；
- 被裁剪样本的比例（clip fraction）；
- 策略熵；
- 价值损失和解释方差。

如果 KL 明显超过目标值，可提前停止本轮策略更新。完成后丢弃当前 rollout，令更新后的策略成为下一轮旧策略，再与环境交互。

完整循环可以概括为：

```text
重复：
    old_policy <- current_policy
    使用 old_policy 收集一批轨迹
    计算 return 和 advantage

    重复 K 个 epoch：
        将 rollout 打乱并切成 minibatch
        计算 ratio = exp(new_log_prob - old_log_prob)
        更新裁剪策略目标
        更新价值函数
        可选：KL 过大则提前停止
```

> **最容易混淆的一点**：一个 rollout 内做多个 epoch 时，分母始终是采样时保存的旧策略概率，不能在每个 epoch 后把分母更新为最新策略。

### 5.5.5 PPO 的性能表现

图 5.4 展示了 PPO 与其他算法在多个 MuJoCo 任务上的对比。紫色线代表 PPO——在大多数任务中，PPO 要么是最优算法，要么接近最优。

<div align=center>
<img width="550" src="../img/ch5/5.4.png"/>
</div>
<div align=center>图 5.4 PPO 与其他算法的比较</div>

PPO 的成功可以归因于三点：
1. **实现简单**：相比 TRPO，PPO 只需一阶优化；
2. **稳定性好**：KL 约束或裁剪机制防止了灾难性的策略崩溃；
3. **批次利用率较高**：相比最基础的策略梯度，PPO 可以在同一批 rollout 上进行多个 epoch；但与使用经验回放的异策略算法相比，它仍不属于高样本复用率算法。

---

## 第六部分：本章知识体系总结

### 6.1 核心知识点

| 知识点 | 核心内容 | 所在章节 |
|--------|---------|---------|
| **同策略 vs 异策略** | 同策略要求数据接近当前策略分布，只能短期复用；异策略可学习其他策略产生的数据 | 5.1 / 第一部分 |
| **重要性采样** | $\mathbb{E}_p[f] = \mathbb{E}_q[f \cdot p/q]$，用权重修正分布差异 | 5.1 / 第二部分 |
| **支撑集条件** | 必须满足 $\operatorname{supp}(p)\subseteq\operatorname{supp}(q)$，否则 $q$ 无法采到 $p$ 关心的区域 | 5.2.1 |
| **重要性采样的方差问题** | $p$ 和 $q$ 差距大时 $\frac{p}{q}$ 放大方差，有限样本下估计极不稳定 | 5.2.2 |
| **异策略策略梯度** | $\nabla \bar{R}_\theta = \mathbb{E}_{\tau \sim p_{\theta'}}\left[\frac{p_\theta(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_\theta(\tau)\right]$ | 5.3.1 |
| **重要性权重简化** | 忽略难算的状态访问分布比值，使用可计算的动作概率比值作为局部替代目标 | 5.3.3 |
| **PPO-Penalty 目标** | $J_{\mathrm{PPO}} = J^{\theta'} - \beta \cdot \mathrm{KL}(\theta, \theta')$，在替代目标上加入 KL 软惩罚 | 5.4.2 |
| **TRPO vs PPO** | TRPO 用 KL 硬约束 + 二阶优化；PPO 用 KL 软惩罚 + 一阶优化，实现更简单 | 5.4.3 |
| **KL 散度 = 行为距离** | 衡量两个策略在相同状态下输出动作分布的差异，而非参数 L2 距离 | 5.4.4 |
| **PPO-Penalty** | 用自适应 $\beta$ 动态调节 KL 惩罚强度：KL 过大则增大 $\beta$，过小则减小 | 5.5.1 |
| **PPO-Clip** | 用 $\min$ + $\operatorname{clip}$ 取消越过阈值后的额外优化收益，但不保证所有比值严格留在区间内 | 5.5.2 |
| **PPO 的同策略性质** | 只短期复用当前旧策略的 rollout，并通过裁剪、KL 监控等限制更新，通常归类为同策略 | 5.4.2 |

### 6.2 公式速查表

| 公式 | 编号 | 用途 |
|------|------|------|
| $\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$ | (5.3) | 重要性采样基本公式 |
| $\nabla \bar{R}_\theta = \mathbb{E}_{\tau \sim p_{\theta'}}\left[\frac{p_\theta(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_\theta(\tau)\right]$ | (5.4) | 异策略策略梯度 |
| $J^{\theta'}(\theta)=\mathbb E_{\pi_{\theta'}}[r_t(\theta)A^{\theta'}(s_t,a_t)]$ | — | 异策略目标函数 |
| $J_{\mathrm{PPO}}^{\theta'}(\theta) = J^{\theta'}(\theta) - \beta \cdot \mathrm{KL}(\theta, \theta')$ | (5.6) | PPO-Penalty 目标函数 |
| $J_{\mathrm{PPO2}} = \sum \min(rA, \operatorname{clip}(r, 1-\varepsilon, 1+\varepsilon)A)$ | (5.8) | PPO-Clip 目标函数 |

### 6.3 思维导图

```
第5章 PPO 算法
│
├── 动机：基础同策略方法的批次利用率低
│   └── 策略一变化，旧数据与当前策略的分布偏差就会增大
│
├── 解决方案：异策略 + 重要性采样
│   ├── 重要性采样：E_p[f] = E_q[f · p/q]
│   └── 方差陷阱：p 和 q 差距大 → 方差爆炸
│
├── 异策略策略梯度
│   ├── 轨迹级 → 状态-动作级
│   ├── 重要性权重简化：p(s,a)/p'(s,a) ≈ p(a|s)/p'(a|s)
│   └── 目标函数：J^{θ'}(θ)
│
└── PPO：约束策略差异
    ├── KL 散度约束（行为距离 ≠ 参数距离）
    ├── TRPO（硬约束，二阶优化）→ PPO（软惩罚或裁剪，一阶优化）
    ├── PPO-Penalty：自适应 β 调节 KL 惩罚
    └── PPO-Clip：min + clip 截断越界后的额外收益
        ├── A > 0：阻止过度增大好动作概率（上限 1+ε）
        └── A < 0：阻止过度减小坏动作概率（下限 1-ε）
```

---

## 配套可运行代码

第 5 章的教学代码位于 [`docs/chapter5/code`](code/README.md)：

| 文件 | 对应知识 |
|---|---|
| [`importance_sampling.py`](code/importance_sampling.py) | 重要性采样、权重、有效样本量和方差 |
| [`off_policy_policy_gradient.py`](code/off_policy_policy_gradient.py) | 异策略策略梯度目标函数 |
| [`trpo.py`](code/trpo.py) | KL 硬约束、共轭梯度和回溯线搜索 |
| [`ppo_penalty.py`](code/ppo_penalty.py) | KL 软惩罚和自适应 $\beta$ |
| [`ppo_clip.py`](code/ppo_clip.py) | 概率比值裁剪、clip fraction 和提前停止 |

建议按照表格顺序运行。具体命令和快速验证参数见 [`code/README.md`](code/README.md)。

---

## 参考文献

* [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#)
* [百面机器学习](https://book.douban.com/subject/30285146/)
* [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
* [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477)
* [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)
