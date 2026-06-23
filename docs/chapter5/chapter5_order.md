# 第 5 章 PPO 算法

> **说明**：本文是 `chapter5.md` 的整理版本。相比原文，本章对知识点的组织顺序进行了优化——遵循"动机（同策略→异策略）→ 工具（重要性采样）→ 问题（方差危机）→ 解决方案（PPO 的 KL 约束与裁剪机制）→ 算法变种（PPO-Penalty vs PPO-Clip）"的认知路径，并补充了重要性采样方差的数学推导与直观解释、PPO 目标函数从梯度反推的详细过程、PPO-Penalty 自适应 β 的动机说明，以及 PPO-Clip 在不同优势符号下的行为可视化分析，便于初学者循序渐进地理解近端策略优化方法。

---

## 学习路线图

本章建议按以下顺序学习：

1. **先理解动机**：为什么策略梯度算法"采样一次、更新一次"效率低？同策略和异策略的根本区别是什么？（5.1 前半部分）
2. **再掌握工具**：重要性采样的数学原理是什么？为什么它能实现异策略学习？（5.1 后半部分）
3. **认清问题**：重要性采样有什么致命缺陷？为什么 $p$ 和 $q$ 差距大时方差会爆炸？（5.1 末尾 → 5.2 开头）
4. **学习解决方案**：PPO 如何用 KL 散度约束解决分布漂移？PPO-Penalty 和 PPO-Clip 各有什么巧妙设计？（5.2 全部）

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
3. **关键问题来了**：参数变成 $\theta'$ 后，概率分布 $p_{\theta}(\tau)$ 也随之改变了。旧轨迹是在 $p_\theta$ 下采样的，但新策略是 $p_{\theta'}$——**分布不匹配**，旧数据不能再用；
4. 必须用新策略 $\pi_{\theta'}$ 重新与环境交互，采集新数据。

> **一句话总结**：同策略算法中，**数据采样和参数更新是串行绑定的**——更新一次就得重新采样一次。强化学习中，与环境交互采样往往是最耗时的部分（比如在真实机器人上采集数据可能需要数小时），而梯度计算在 GPU 上可能只需要几秒。这种"采样一次→更新一次→丢弃→重新采样"的模式严重浪费了采样效率。

#### 同策略 vs 异策略的直观对比

| | 同策略（on-policy） | 异策略（off-policy） |
|---|---|---|
| **交互智能体** | $\pi_\theta$（正在学习的策略） | $\pi_{\theta'}$（固定的"示范"策略） |
| **学习智能体** | $\pi_\theta$ | $\pi_\theta$（学习目标不变） |
| **数据复用** | 只能更新一次 | 可反复使用，更新多次 |
| **效率** | 低——大量时间花在采样上 | 高——一次采样，多次学习 |
| **代表算法** | 基本策略梯度、REINFORCE、Sarsa | Q-learning、DQN、PPO |

> **核心动机**：如果我们能让一个固定的"示范策略" $\pi_{\theta'}$ 去与环境交互，采集大量数据，然后用这批数据反复训练"学习策略" $\pi_\theta$（更新很多次梯度），就能大幅提升数据利用率。这就是从同策略走向异策略的根本动力。

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

**使用条件**：$q(x) > 0$ 的所有位置，$p(x)$ 也必须 $> 0$。换句话说，$q$ 的支撑集必须覆盖 $p$ 的支撑集——$q$ 能采样到的区域必须包含所有 $p$ 可能出现的区域，否则 $\frac{p(x)}{q(x)}$ 在 $q(x) = 0$ 处无定义。

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

### 5.3.2 从轨迹级到状态-动作级的细化

实际实现中，策略梯度不是对整个轨迹给一个统一分数，而是**逐对 $(s_t, a_t)$ 计算优势函数并更新**：

$$
\mathbb{E}_{(s_t, a_t) \sim \pi_{\theta}}\left[A^{\theta}(s_t, a_t) \nabla \log p_{\theta}(a_t | s_t)\right]
$$

其中 $A^{\theta}(s_t, a_t)$ 是**优势函数（advantage function）**，表示在状态 $s_t$ 采取动作 $a_t$ 比平均水平好多少（累积奖励减去基线）。

应用重要性采样后：

$$
\mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{p_{\theta}(s_t, a_t)}{p_{\theta'}(s_t, a_t)} A^{\theta'}(s_t, a_t) \nabla \log p_{\theta}(a_t | s_t)\right] \tag{5.5}
$$

注意：这里 $A^{\theta}$ 变成了 $A^{\theta'}$——因为数据是由 $\theta'$ 采样得到的，优势函数也应基于 $\theta'$ 的经验来估计。

### 5.3.3 重要性权重的简化推导 ⚠️ 重点

将联合概率 $p(s_t, a_t)$ 分解：

$$
\begin{aligned}
p_{\theta}(s_t, a_t) &= p_{\theta}(a_t | s_t) \cdot p_{\theta}(s_t) \\
p_{\theta'}(s_t, a_t) &= p_{\theta'}(a_t | s_t) \cdot p_{\theta'}(s_t)
\end{aligned}
$$

代入重要性权重：

$$
\frac{p_{\theta}(s_t, a_t)}{p_{\theta'}(s_t, a_t)}
= \frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} \cdot \frac{p_{\theta}(s_t)}{p_{\theta'}(s_t)}
$$

**关键假设**：$p_{\theta}(s_t) \approx p_{\theta'}(s_t)$，即不同策略下遇到同一状态的概率大致相同。理由有二：
1. **实践理由**：状态的出现概率往往与策略的关系不大——比如无论用哪种策略玩 Atari 游戏，看到的游戏画面分布大致相同；
2. **计算理由**：$p_{\theta}(s_t)$ 几乎无法计算（尤其是连续状态空间），强行保留这一项会使算法不可实现。

因此简化为：

$$
\boxed{\frac{p_{\theta}(s_t, a_t)}{p_{\theta'}(s_t, a_t)} \approx \frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)}}
$$

而 $\frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)}$ 是非常容易计算的——只需将 $s_t$ 分别输入策略网络 $\pi_\theta$ 和 $\pi_{\theta'}$，取对应动作 $a_t$ 的概率，做比值即可。

### 5.3.4 从梯度反推目标函数

利用恒等式 $\nabla f(x) = f(x) \nabla \log f(x)$，我们可以从式(5.5)的梯度形式反推出目标函数（注意对 $\theta$ 求梯度时，$p_{\theta'}$ 和 $A^{\theta'}$ 都是常数）：

$$
\boxed{J^{\theta'}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t)\right]}
$$

这就是**异策略策略梯度的目标函数**。我们用 $\theta'$ 采样，优化 $\theta$——$J^{\theta'}(\theta)$ 中括号外的 $\theta'$ 表示数据来源，括号内的 $\theta$ 表示优化变量。

---

## 第四部分：近端策略优化（PPO）—— 解决分布漂移

> 重要性采样虽然打通了异策略学习，但带来了方差问题：$p_\theta$ 和 $p_{\theta'}$ 差距大时，估计极不稳定。PPO 的核心设计就是**约束新旧策略之间的差异**，在享受异策略效率的同时控制方差。

### 5.4.1 问题定位：分布漂移

在 $\theta$ 多轮更新后，$\pi_\theta$ 和 $\pi_{\theta'}$ 的动作分布可能已经相差很远。此时：
- 重要性权重 $\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)}$ 可能非常大或非常小；
- 按照 5.2.2 节的分析，方差急剧增大；
- 梯度估计变得不可靠，训练可能崩溃。

> **PPO 的核心思路**：在优化 $J^{\theta'}(\theta)$ 的同时，**显式约束 $\theta$ 与 $\theta'$ 之间的差异**，确保重要性采样始终在"安全区"内运作。

### 5.4.2 PPO 的目标函数

PPO 在原始异策略目标上增加 KL 散度惩罚项：

$$
\boxed{J_{\mathrm{PPO}}^{\theta'}(\theta) = \underbrace{J^{\theta'}(\theta)}_{\text{异策略目标}} - \underbrace{\beta \cdot \mathrm{KL}(\theta, \theta')}_{\text{KL 散度约束}}} \tag{5.6}
$$

其中：
- $J^{\theta'}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{p_{\theta}(a_t|s_t)}{p_{\theta'}(a_t|s_t)} A^{\theta'}(s_t, a_t)\right]$ 是异策略目标；
- $\mathrm{KL}(\theta, \theta')$ 衡量两个策略在**动作分布**上的差异（不是参数距离）；
- $\beta$ 是惩罚系数，控制约束的强度。

> **注意**：虽然 PPO 的推导涉及重要性采样（异策略技术），但由于 KL 约束强制 $\theta \approx \theta'$，行为策略和目标策略几乎相同，**PPO 在实践中被视为同策略算法**。

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
| **KL 散度的处理** | 作为硬约束（$\mathrm{KL} < \delta$） | 作为软惩罚（加入目标函数） |
| **优化方式** | 需要二阶优化（共轭梯度 + 线搜索） | 一阶梯度上升即可 |
| **实现难度** | 高——约束优化很复杂 | 低——无约束优化 |
| **性能** | 好 | 与 TRPO 相当 |
| **实际使用** | 几乎被 PPO 取代 | OpenAI 默认算法 |

> **为什么 PPO 更好实现？** 带约束的优化（TRPO）通常需要复杂的二阶方法或投影步骤，而 PPO 将 KL 散度作为正则化项直接加入目标函数，一阶梯度方法就能直接优化——实现简单得多，性能却不差。

### 5.4.4 KL 散度的含义：行为距离 vs 参数距离 ⚠️ 重点难点

这里有一个容易混淆的点：PPO 中的 $\mathrm{KL}(\theta, \theta')$ 到底衡量什么？

**❌ 不是参数距离**：不是计算两组神经网络权重 $\theta$ 和 $\theta'$ 之间的 L2 距离或余弦相似度。

**✅ 是行为距离（behavior distance）**：给定同一个状态 $s$，两个策略分别输出动作概率分布 $\pi_\theta(\cdot|s)$ 和 $\pi_{\theta'}(\cdot|s)$，计算这两个**概率分布之间的 KL 散度**，然后对所有状态取平均：

$$
\mathrm{KL}(\theta, \theta') = \mathbb{E}_{s \sim \pi_{\theta'}}\left[ D_{\mathrm{KL}}\left(\pi_{\theta'}(\cdot|s) \,\|\, \pi_\theta(\cdot|s)\right) \right]
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
J_{\mathrm{PPO}}^{\theta^{k}}(\theta) = \underbrace{\sum_{(s_t, a_t)} \frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)} A^{\theta^k}(s_t, a_t)}_{\text{异策略目标}} - \underbrace{\beta \cdot \mathrm{KL}(\theta, \theta^k)}_{\text{KL 惩罚}} \tag{5.7}
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

$$
\boxed{
J_{\mathrm{PPO2}}^{\theta^{k}}(\theta) \approx \sum_{(s_t, a_t)} \min \left(
\underbrace{\frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)} A^{\theta^k}(s_t, a_t)}_{\text{原始项}},
\ \underbrace{\operatorname{clip}\left(\frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}, 1-\varepsilon, 1+\varepsilon\right) A^{\theta^k}(s_t, a_t)}_{\text{裁剪项}}
\right)
} \tag{5.8}
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

PPO-Clip 的精妙之处在于 **$\min$ 操作 + 裁剪**的组合效果，需分 $A > 0$ 和 $A < 0$ 两种情况理解。

##### 情况一：$A > 0$（这个动作是好的，应增大其概率）

此时我们想让 $p_{\theta}(a_t|s_t)$ 变大。随着比值 $r = \frac{p_{\theta}}{p_{\theta^k}}$ 从 1 开始增大：

- 当 $r < 1 + \varepsilon$（比值在安全区）：$\operatorname{clip}(r) = r$，两项相等，目标 = $r \cdot A$——正常增大；
- 当 $r > 1 + \varepsilon$（比值超过阈值）：裁剪项被截断为 $(1+\varepsilon) \cdot A$，而原始项是 $r \cdot A$（更大）。$\min$ 操作选择较小的裁剪项——**梯度被截断**，比值不再增长。

> **效果**：当动作概率增幅超过 $1+\varepsilon$ 时，目标函数不再给予额外奖励——防止某个好动作的概率被过度放大。

##### 情况二：$A < 0$（这个动作是不好的，应减小其概率）

此时我们想让 $p_{\theta}(a_t|s_t)$ 变小。随着比值 $r = \frac{p_{\theta}}{p_{\theta^k}}$ 从 1 开始减小：

- 当 $r > 1 - \varepsilon$（比值在安全区）：$\operatorname{clip}(r) = r$，两项相等，目标 = $r \cdot A$（负值）——正常减小；
- 当 $r < 1 - \varepsilon$（比值低于阈值）：裁剪项被截断为 $(1-\varepsilon) \cdot A$，而原始项是 $r \cdot A$（更负）。**注意这里 $A < 0$**，所以 $r \cdot A$ 更负意味着绝对值更大。$\min$ 操作选择更负的原始项（因为 $\min(-5, -3) = -5$）……等等，我们需要重新审视。

实际上，当 $A < 0$ 时：
- 裁剪项 = $(1-\varepsilon) \cdot A$（一个绝对值受限的负值）；
- 原始项 = $r \cdot A$，其中 $r < 1-\varepsilon$，所以 $|r \cdot A| > |(1-\varepsilon) \cdot A|$，即原始项更负。

$\min$ 在两个负值之间取更小的那个 = 取更负的那个 = **取原始项**。但由于裁剪项的存在，$\min$ 创造了一个"地板"——当 $r < 1-\varepsilon$ 时，目标函数不会因为继续减小 $p_\theta$ 而变得更优（因为 $\min$ 取的是裁剪项？不……）

让我们更仔细地分析图 5.3：

<div align=center>
<img width="550" src="../img/ch5/5.3.png"/>
</div>
<div align=center>图 5.3 $A$ 对裁剪函数输出的影响</div>

- **图 5.3a（$A > 0$）**：绿线 = 原始项 $rA$，蓝线 = 裁剪项 $\operatorname{clip}(r)A$。当 $r > 1+\varepsilon$，蓝线被截平，红线（$\min$）= 蓝线——**更新被截断**，不再鼓励 $p_\theta$ 继续增大。
- **图 5.3b（$A < 0$）**：当 $r < 1-\varepsilon$，裁剪项 = $(1-\varepsilon)A$（一个受限的负值），而原始项 = $rA$（更负）。$\min$ 取更负的原始项——**实际上并不截断减小方向的更新**。等等，那为什么图中红线在 $r < 1-\varepsilon$ 时是平的？

重新理解图 5.3b：当 $A < 0$ 时，我们希望 $p_\theta$ 减小（$r$ 变小）。当 $r$ 从 1 减小到 $1-\varepsilon$ 时，目标函数正常变小（更负），鼓励继续减小 $p_\theta$。当 $r < 1-\varepsilon$ 时，裁剪项的梯度消失（因为 $\operatorname{clip}$ 输出常数），但原始项继续下降。$\min$ 在原始项和裁剪项之间取最小值——此时裁剪项是常数，原始项继续下降……$\min$ 应该取原始项，但梯度依然存在？

实际上，当 $r < 1-\varepsilon$ 且 $A < 0$ 时：
- 裁剪项 = $(1-\varepsilon)A$（常数，不产生梯度）
- 原始项 = $rA$（$r$ 越小，这个值越负）

$\min($原始项, 裁剪项$)$ = 原始项（因为原始项更负）。所以**梯度依然来自原始项**，$p_\theta$ 还会继续减小。

> **修正理解**：PPO-Clip 在 $A > 0$ 时阻止过大的概率增加（截断上限），在 $A < 0$ 时阻止过大的概率减小（截断下限）。但 $\min$ 操作的实际效果需要仔细辨析——关键是要结合图 5.3 的红线来看。

**更准确的描述**：
- **$A > 0$**（好动作）：$p_\theta$ 增大到比值超过 $1+\varepsilon$ 时，梯度被截断——**不要涨太多**；
- **$A < 0$**（坏动作）：$p_\theta$ 减小到比值低于 $1-\varepsilon$ 时，梯度被截断——**不要降太多**。

> **核心直觉**：PPO-Clip 在"好动作更可能"和"坏动作更不可能"两个方向上都设置了更新上限，确保新旧策略的动作概率比值始终在 $[1-\varepsilon, 1+\varepsilon]$ 的"信任区域"内。

### 5.5.3 PPO-Penalty vs PPO-Clip 对比

| 维度 | PPO-Penalty (PPO1) | PPO-Clip (PPO2) |
|------|-------------------|-----------------|
| **约束方式** | KL 散度作为惩罚项加入目标 | 直接裁剪概率比值 |
| **超参数** | $\beta$（可自适应调整）+ $\mathrm{KL}_{\max}$/$\mathrm{KL}_{\min}$ | $\varepsilon$（通常 0.1 或 0.2） |
| **实现复杂度** | 需要计算 KL 散度 + 自适应逻辑 | 非常简单，几行代码 |
| **调参难度** | 中等（需设置 KL 阈值） | 低（$\varepsilon$ 比较鲁棒） |
| **实际流行度** | 较少使用 | **主流选择**，OpenAI 默认实现 |

### 5.5.4 PPO 的性能表现

图 5.4 展示了 PPO 与其他算法在多个 MuJoCo 任务上的对比。紫色线代表 PPO——在大多数任务中，PPO 要么是最优算法，要么接近最优。

<div align=center>
<img width="550" src="../img/ch5/5.4.png"/>
</div>
<div align=center>图 5.4 PPO 与其他算法的比较</div>

PPO 的成功可以归因于三点：
1. **实现简单**：相比 TRPO，PPO 只需一阶优化；
2. **稳定性好**：KL 约束或裁剪机制防止了灾难性的策略崩溃；
3. **数据效率高**：相比基本策略梯度（一次采样一次更新），PPO 可以多次复用数据。

---

## 第六部分：本章知识体系总结

### 6.1 核心知识点

| 知识点 | 核心内容 | 所在章节 |
|--------|---------|---------|
| **同策略 vs 异策略** | 同策略：交互策略 = 学习策略，数据不可复用；异策略：分离两者，数据可复用 | 5.1 / 第一部分 |
| **重要性采样** | $\mathbb{E}_p[f] = \mathbb{E}_q[f \cdot p/q]$，用权重修正分布差异 | 5.1 / 第二部分 |
| **重要性采样的方差问题** | $p$ 和 $q$ 差距大时 $\frac{p}{q}$ 放大方差，有限样本下估计极不稳定 | 5.2.2 |
| **异策略策略梯度** | $\nabla \bar{R}_\theta = \mathbb{E}_{\tau \sim p_{\theta'}}\left[\frac{p_\theta(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_\theta(\tau)\right]$ | 5.3.1 |
| **重要性权重简化** | $\frac{p_\theta(s,a)}{p_{\theta'}(s,a)} \approx \frac{p_\theta(a|s)}{p_{\theta'}(a|s)}$（假设 $p(s)$ 近似不变） | 5.3.3 |
| **PPO 目标函数** | $J_{\mathrm{PPO}} = J^{\theta'} - \beta \cdot \mathrm{KL}(\theta, \theta')$，在异策略目标上加入 KL 约束 | 5.4.2 |
| **TRPO vs PPO** | TRPO 用 KL 硬约束 + 二阶优化；PPO 用 KL 软惩罚 + 一阶优化，实现更简单 | 5.4.3 |
| **KL 散度 = 行为距离** | 衡量两个策略在相同状态下输出动作分布的差异，而非参数 L2 距离 | 5.4.4 |
| **PPO-Penalty** | 用自适应 $\beta$ 动态调节 KL 惩罚强度：KL 过大则增大 $\beta$，过小则减小 | 5.5.1 |
| **PPO-Clip** | 用 $\min$ + $\operatorname{clip}$ 直接限制概率比值 $r \in [1-\varepsilon, 1+\varepsilon]$，无需计算 KL | 5.5.2 |
| **PPO 的同策略性质** | 虽然推导用了异策略技术，但 KL 约束使行为策略 ≈ 目标策略，实践中为同策略 | 5.4.2 |

### 6.2 公式速查表

| 公式 | 编号 | 用途 |
|------|------|------|
| $\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]$ | (5.3) | 重要性采样基本公式 |
| $\nabla \bar{R}_\theta = \mathbb{E}_{\tau \sim p_{\theta'}}\left[\frac{p_\theta(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_\theta(\tau)\right]$ | (5.4) | 异策略策略梯度 |
| $J^{\theta'}(\theta) = \mathbb{E}_{(s_t,a_t)\sim\pi_{\theta'}}\left[\frac{p_\theta(a_t|s_t)}{p_{\theta'}(a_t|s_t)} A^{\theta'}(s_t,a_t)\right]$ | — | 异策略目标函数 |
| $J_{\mathrm{PPO}}^{\theta'}(\theta) = J^{\theta'}(\theta) - \beta \cdot \mathrm{KL}(\theta, \theta')$ | (5.6) | PPO 目标函数 |
| $J_{\mathrm{PPO2}} = \sum \min(rA, \operatorname{clip}(r, 1-\varepsilon, 1+\varepsilon)A)$ | (5.8) | PPO-Clip 目标函数 |

### 6.3 思维导图

```
第5章 PPO 算法
│
├── 动机：同策略效率低
│   └── 采样一次 → 更新一次 → 丢弃数据 → 重新采样
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
    ├── TRPO（硬约束，二阶优化）→ PPO（软惩罚，一阶优化）
    ├── PPO-Penalty：自适应 β 调节 KL 惩罚
    └── PPO-Clip：min + clip 限制概率比值 ∈ [1-ε, 1+ε]
        ├── A > 0：阻止过度增大好动作概率（上限 1+ε）
        └── A < 0：阻止过度减小坏动作概率（下限 1-ε）
```

---

## 参考文献

* [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#)
* [百面机器学习](https://book.douban.com/subject/30285146/)
* [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
* [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477)
* [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)
