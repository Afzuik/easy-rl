# 第 3 章 表格型方法 — 公式手册

> 从 `docs/chapter3/chapter3.md` 提取，按节编排。每条含 **公式** + **描述**。

---

## 3.1 马尔可夫决策过程

**状态转移概率**

$$
p\big[s_{t+1}, r_t \mid s_t, a_t\big]
$$

> 在状态 $s_t$ 选择动作 $a_t$，转移到状态 $s_{t+1}$ 并获得奖励 $r_t$ 的概率。它具有马尔可夫性质：下一时刻仅由当前状态和动作决定。

**概率函数 & 奖励函数**

$$
P\big[s_{t+1}, r_t \mid s_t, a_t\big], \qquad R\big[s_t, a_t\big]
$$

> $P$ 刻画环境随机性（状态转移），$R$ 给出即时奖励。二者已知即为「有模型」，未知即为「免模型」。

**四元组**

$$
(S, A, P, R)
$$

> 状态空间 $S$、动作空间 $A$、转移概率 $P$、奖励函数 $R$，共同定义一个马尔可夫决策过程 (MDP)；再加折扣因子 $\gamma$ 即五元组。

**价值函数**

$$
V(S), \qquad Q(s, a)
$$

> $V(S)$ 评价状态好坏；$Q$ 函数判断在某状态下采取某动作能获得的最大奖励。

---

## 3.2 Q 表格

**折扣因子**

$$
\gamma \in [0,1], \qquad \gamma^n \to 0 \ \text{as}\ n \to \infty
$$

> 控制未来奖励的权重。$\gamma=0$ 目光短浅（只看单步奖励）；$\gamma=1$ 目光长远（等权加总所有未来奖励）；越往后 $\gamma^n$ 越小，远期奖励影响越小。

**递归回报**

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

> 从后往前递推回报的递推公式，用于悬崖行走示例中分步计算每个状态的 Q 值。

---

## 3.3 免模型预测

### 3.3.1 蒙特卡洛策略评估

**回报 (Return)**

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots \tag{3.0}
$$

> 从时间步 $t$ 开始的折扣累积奖励，MC 方法以此作为更新目标。

**状态价值 (策略 $\pi$ 下)**

$$
V_\pi(s) = \mathbb{E}_{\tau \sim \pi}\big[G_t \mid s_t = s\big]
$$

> 所有从状态 $s$ 出发、遵循策略 $\pi$ 的轨迹的回报期望。

**MC 访问计数与总回报**

$$
N(s) \leftarrow N(s) + 1, \qquad S(s) \leftarrow S(s) + G_t
$$

> 每次访问状态 $s$，计数 +1，累加该轨迹的回报。

**MC 价值估计**

$$
V(s) = \frac{S(s)}{N(s)}
$$

> 以经验均值估计状态价值。大数定律：$N(s) \to \infty$ 时 $V(s) \to V_\pi(s)$。

**经验均值 → 增量均值**

$$
\mu_t = \frac{1}{t}\sum_{j=1}^{t} x_j = \mu_{t-1} + \frac{1}{t}\big(x_t - \mu_{t-1}\big)
$$

> 将批量求平均转化为在线增量更新；$\frac{1}{t}$ 类似学习率，$x_t - \mu_{t-1}$ 为残差。

**增量式蒙特卡洛**

$$
N(s_t) \leftarrow N(s_t) + 1
$$

$$
V(s_t) \leftarrow V(s_t) + \frac{1}{N(s_t)}\big(G_t - V(s_t)\big)
$$

> 每采集一条新轨迹，以增量方式更新状态价值。

**带学习率的 MC 更新**

$$
V(s_t) \leftarrow V(s_t) + \alpha\big(G_t - V(s_t)\big)
$$

> 用固定学习率 $\alpha$ 替代 $1/N(s_t)$，可调节更新速率。

**贝尔曼期望备份 (动态规划)**

$$
V_i(s) \leftarrow \sum_{a \in A} \pi(a \mid s)\bigg(R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s, a) V_{i-1}(s')\bigg)
$$

> DP 中的价值迭代：用上一轮 $V_{i-1}$ 更新本轮 $V_i$；两层求和 — 内层对 $s'$ 求期望，外层对 $a$ 求期望。

**蒙特卡洛备份**

$$
V(s_t) \leftarrow V(s_t) + \alpha\big(G_{i,t} - V(s_t)\big)
$$

> MC 沿一条实际轨迹的回报更新该轨迹上所有访问过的状态。

---

### 3.3.2 时序差分 (TD)

**TD(0) 更新**

$$
V(s_t) \leftarrow V(s_t) + \alpha\big(r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\big) \tag{3.1}
$$

> 一步时序差分：走一步即用「实际奖励 + 下一状态估计值」更新当前状态价值。免模型 + 自举。

**时序差分目标 (TD target)**

$$
r_{t+1} + \gamma V(s_{t+1})
$$

> 由两部分组成：(1) 实际奖励 $r_{t+1}$；(2) 自举估计 $\gamma V(s_{t+1})$。是 $V(s_t)$ 逼近的目标值。

**时序差分误差 (TD error)**

$$
\delta = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

> 目标值与当前估计之差；驱动 TD 学习的核心信号。

**n 步时序差分回报** (式 3.2)

$$
\begin{aligned}
n=1\text{ (TD)}\quad & G_t^{(1)} = r_{t+1} + \gamma V(s_{t+1}) \\[4pt]
n=2\quad & G_t^{(2)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2}) \\[4pt]
&\ \vdots \\[4pt]
n=\infty\text{ (MC)}\quad & G_t^{\infty} = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{T-t-1} r_T
\end{aligned}
$$

> 调节 $n$ 在「MC（无偏高方差）」与「TD（有偏低方差）」之间权衡。$n=\infty$ 退化为 MC。

**n 步回报的通式**

$$
G_t^n = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})
$$

> 前 $n$ 步用实际奖励，第 $n$ 步后用自举。

**n 步 TD 更新**

$$
V(s_t) \leftarrow V(s_t) + \alpha\big(G_t^n - V(s_t)\big)
$$

> 得到 $n$ 步目标后以增量方式更新状态价值。

---

### 3.3.3 统一视角：自举与采样

**DP 备份 (全期望)**

$$
V(s_t) \leftarrow \mathbb{E}_\pi\big[r_{t+1} + \gamma V(s_{t+1})\big]
$$

> 直接计算所有相关状态的期望；**有自举、无采样**。

**MC 备份 (纯采样)**

$$
V(s_t) \leftarrow V(s_t) + \alpha\big(G_t - V(s_t)\big)
$$

> 沿一条实际轨迹更新；**无自举、纯采样**。

**TD(0) 备份 (采样 + 自举)**

$$
\text{TD}(0):\quad V(s_t) \leftarrow V(s_t) + \alpha\big(r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\big)
$$

> 走一步即更新，**既有采样、又有自举**。

| 方法 | 自举 | 采样 |
|---|---|---|
| 动态规划 (DP) | ✅ | ❌ |
| 蒙特卡洛 (MC) | ❌ | ✅ |
| 时序差分 (TD) | ✅ | ✅ |

---

## 3.4 免模型控制

### 策略迭代基础

**策略改进 (贪心)**

$$
\pi'(s) = \arg\max_a Q_\pi(s, a) \tag{3.3}
$$

> 根据当前 $Q$ 值贪心选最大动作，得到改进后的策略。

**Q 函数的贝尔曼方程**

$$
Q_{\pi_i}(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a) V_{\pi_i}(s')
$$

> 需知道 $R$ 与 $P$；免模型时无法直接计算，需引入广义策略迭代 (GPI)。

**广义策略迭代中的贪心改进**

$$
\pi(s) = \arg\max_a Q(s, a)
$$

> GPI 中策略改进步骤：Q 表估计好后直接取最大动作。

---

### ε-贪心探索

**ε-贪心策略改进的单调性证明**

$$
\begin{aligned}
Q_\pi(s, \pi'(s))
&= \sum_{a \in A} \pi'(a \mid s) Q_\pi(s, a) \\
&= \frac{\varepsilon}{|A|} \sum_{a \in A} Q_\pi(s, a) + (1-\varepsilon) \max_a Q_\pi(s, a) \\
&\geqslant \frac{\varepsilon}{|A|} \sum_{a \in A} Q_\pi(s, a) + (1-\varepsilon) \sum_{a \in A} \frac{\pi(a \mid s)-\frac{\varepsilon}{|A|}}{1-\varepsilon} Q_\pi(s, a) \\
&= \sum_{a \in A} \pi(a \mid s) Q_\pi(s, a) = V_\pi(s)
\end{aligned}
$$

> 结论：$V_{\pi'}(s) \geqslant V_\pi(s)$，ε-贪心策略保证单调改进。

---

### 3.4.1 Sarsa（同策略 TD 控制）

**Sarsa 更新公式**

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\big[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\big] \tag{3.4}
$$

> 同策略：下一步的动作 $a_{t+1}$ 是实际执行的（由行为策略选出）。名称来自 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$。

**Sarsa 缩写形式**

$$
Q(S, A) \leftarrow Q(S, A) + \alpha\big(R + \gamma Q(S', A') - Q(S, A)\big)
$$

> 与时序差分更新 $V$ 的形式一致，只是换成了 $Q$。

**n 步 Sarsa 回报** (式 3.5)

$$
\begin{aligned}
n=1\text{ (Sarsa)}\quad & Q_t^1 = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) \\[4pt]
n=2\quad & Q_t^2 = r_{t+1} + \gamma r_{t+2} + \gamma^2 Q(s_{t+2}, a_{t+2}) \\[4pt]
&\ \vdots \\[4pt]
n=\infty\text{ (MC)}\quad & Q_t^{\infty} = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{T-t-1} r_T
\end{aligned}
$$

> $n=1$ 即单步 Sarsa；$n=\infty$ 即 MC 控制。单步更新 vs $n$ 步更新 vs 回合更新。

**在 $t$ 时刻的 Sarsa 价值**

$$
Q_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})
$$

> 单步 Sarsa 的时间差分目标。

**n 步 Q 回报**

$$
Q_t^n = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n}, a_{t+n})
$$

> 通用 $n$ 步回报，前 $n$ 步实际奖励 + 第 $n$ 步自举。

**Sarsa($\lambda$) Q 回报**

$$
Q_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} Q_t^n
$$

> 引入资格迹衰减参数 $\lambda$，对所有 $n$ 步回报做指数加权求和，统一 TD 与 MC。

**Sarsa($\lambda$) 更新**

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\big(Q_t^\lambda - Q(s_t, a_t)\big)
$$

> 用加权回报 $Q_t^\lambda$ 作为目标更新 Q 值。

---

### 3.4.2 Q 学习（异策略 TD 控制）

**目标策略**

$$
\pi(s_{t+1}) = \arg\max_{a'} Q(s_{t+1}, a')
$$

> 直接用 Q 表贪心选动作，不与环境交互。目标策略是待学习的最优策略。

**Q 学习目标的推导**

$$
\begin{aligned}
r_{t+1} + \gamma Q(s_{t+1}, A')
&= r_{t+1} + \gamma Q\big(s_{t+1}, \arg\max Q(s_{t+1}, a')\big) \\
&= r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')
\end{aligned}
$$

> $A'$ 不由行为策略决定，而是取 Q 表的最大值；因此不需要知道 $a_{t+1}$。

**Q 学习更新公式**

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\big[r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)\big]
$$

> 异策略：下一步动作取 $\max_a Q$，与实际执行的动作解耦。更激进、更可能探索到最优策略。

---

### 3.4.3 Sarsa vs Q 学习 — 目标对比

| 算法 | 目标 | 类型 |
|---|---|---|
| **Sarsa** | $r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$ | 同策略 (On-policy) |
| **Q 学习** | $r_{t+1} + \gamma \max_a Q(s_{t+1}, a)$ | 异策略 (Off-policy) |

> Sarsa 保守安全，远离悬崖；Q 学习激进大胆，直奔最优。Sarsa 不取 max，Q 学习取 max。

---

## 偏差-方差概念

> **偏差 (Bias)**：预测值的期望与真实值之间的差距。偏差越高，越偏离真实数据。
>
> **方差 (Variance)**：预测值的变化范围/离散程度。方差越高，数据分布越分散。
>
> - **TD**：低方差（用自举，受估计误差影响小）、有偏
> - **MC**：高方差（依赖完整实际回报）、无偏
> - **$n$ 步 TD**：在两者之间权衡
