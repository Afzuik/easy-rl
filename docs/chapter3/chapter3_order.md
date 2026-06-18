# 第 3 章 表格型方法

> **说明**：本文是 `chapter3.md` 的整理版本。相比原文，本章对知识点的组织顺序进行了优化——遵循"动机（为什么需要免模型）→ 工具（Q表格）→ 预测（MC/TD）→ 控制（Sarsa/Q-learning）"的认知路径，并将分散的偏差-方差讨论集中到预测方法对比中，同时补充了增量均值推导的直观解释、Sarsa(λ) 资格迹的动机说明以及悬崖行走问题的行为可视化分析，便于初学者循序渐进地理解表格型强化学习方法。

---

## 学习路线图

本章建议按以下顺序学习：

1. **先理解动机**：为什么第 2 章的动态规划在现实中不够用？有模型 vs 免模型的根本区别是什么？（3.1 节）
2. **再认识工具**：Q 表格是什么？为什么免模型下需要它而不是 V 函数？（3.2 节）
3. **掌握预测方法**：在没有模型的情况下，如何估计给定策略的价值？（3.3~3.5 节）
4. **学会控制方法**：如何找到最优策略？Sarsa 和 Q-learning 的核心区别在哪？（3.6~3.9 节）

---

## 第一部分：从有模型到免模型 —— 为什么需要表格型方法

> 第 2 章的动态规划方法有两个前提：① 环境模型完全已知（知道 $P$ 和 $R$）；② 状态空间足够小。但在真实世界中，这两个前提往往不成立。本章的方法正是为了突破这些限制而设计的。

### 3.1 MDP 回顾与两种求解路径

强化学习是一个与时间相关的序列决策问题。在 $t-1$ 时刻，我看到熊对我招手，下意识的动作就是逃跑；在 $t$ 时刻，我如果选择装死，可能熊觉得无趣就走开了。每一个动作的选择都会影响后续的状态和奖励——这就是马尔可夫决策过程（MDP）的核心：**序列决策**。

<div align=center>
<img width="550" src="../img/ch3/3.1.png"/>
</div>
<div align=center>图 3.1 马尔可夫决策过程四元组</div>

#### 3.1.1 有模型：已知 $P$ 和 $R$ → 动态规划

如果我们完全知道环境的状态转移概率 $P(s'|s,a)$ 和奖励函数 $R(s,a)$，如图 3.2 所示，整个决策树就是透明的——我们可以直接使用第 2 章的策略迭代或价值迭代来求解。

<div align=center>
<img width="550" src="../img/ch3/3.2.png"/>
</div>
<div align=center>图 3.2 有模型：状态转移与序列决策</div>

比如，在熊发怒的情况下选择装死，假设熊 100% 会走开（$P=1$）；选择逃跑，成功概率大概 0.1，失败概率 0.9。既然知道这些概率，直接用动态规划就能算出最优策略——这就是**有模型**方法。

#### 3.1.2 免模型：未知 $P$ 和 $R$ → 从经验中学习

但真实世界中，人类第一次遇到熊时根本不知道逃脱的概率是多少。0.1 和 0.9 都是事后虚构的——**环境在绝大多数情况下是未知的**。这就是免模型（model-free）的核心前提：我们不知道 $P$ 和 $R$，只能通过与环境交互采集经验（轨迹），从经验中学习。

<div align=center>
<img width="550" src="../img/ch3/3.3.png"/>
</div>
<div align=center>图 3.3 免模型试错探索：从经验中学习</div>

#### 3.1.3 有模型与免模型的对比

| 维度 | 有模型（第 2 章） | 免模型（第 3 章） |
|------|-----------------|-----------------|
| **已知什么** | $P(s' \vert s,a)$ 和 $R(s,a)$ | 什么都不知道 |
| **数据来源** | 模型内部计算（规划） | 与环境真实交互（学习） |
| **代表算法** | 策略迭代、价值迭代 | MC、TD、Q-learning、Sarsa |
| **适用场景** | 棋类（规则已知）、小型网格 | 游戏（Atari）、机器人、自动驾驶 |
| **核心挑战** | 大状态空间下计算量 | 如何平衡探索与利用 |

<div align=center>
<img width="550" src="../img/ch3/model_free_1.png"/>
</div>
<div align=center>图 3.4 有模型方法：环境已知，直接规划</div>

<div align=center>
<img width="550" src="../img/ch3/model_free_2.png"/>
</div>
<div align=center>图 3.5 免模型方法：环境未知，交互采集轨迹</div>

---

## 第二部分：Q 表格 —— 免模型下的价值载体

### 3.2 Q 表格：动作价值的查找表

#### 3.2.1 为什么需要 Q 函数而不是 V 函数？

在第 2 章中，控制问题的最终操作是 $\pi^*(s) = \arg\max_a Q^*(s,a)$——选择 Q 值最大的动作。但动态规划下我们能算出 $Q^*$ 是因为我们**知道 $P$ 和 $R$**：

$$
Q_{\pi}(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\pi(s')
$$

在免模型设定下，$P$ 和 $R$ 未知，上式无法计算。因此需要**直接学习（估计）Q 函数**——用一个表格存储每个 $(s,a)$ 对的估计值，通过采样的轨迹逐步更新它。

> **关键洞察**：Q 函数比 V 函数更适合免模型控制，因为有了 $Q(s,a)$ 就可以不依赖模型直接选动作（argmax），而有了 $V(s)$ 还需要知道 $P$ 和 $R$ 才能算出哪个动作好。

#### 3.2.2 Q 表格的结构

如图 3.11 所示，Q 表格的行是所有状态，列是所有动作（上、下、左、右 4 个动作）。初始时全部填 0，之后智能体不断与环境交互，每走一步就更新对应 $(s,a)$ 格子的值。当经验足够多时，Q 表格就收敛到真实的 $Q^*(s,a)$。

<div align=center>
<img width="550" src="../img/ch3/3.9.png"/>
</div>
<div align=center>图 3.11 Q 表格：行=状态，列=动作</div>

用一个比喻：**Q 表格就像一本"生活手册"**。如图 3.6，通过查看手册，我们知道熊发怒时装死的价值更高，熊走开时偷偷逃跑的价值更高。表中 Q 值代表：在当前状态下采取某个动作后，后续能获得的总奖励的估计。

<div align=center>
<img width="550" src="../img/ch3/3.4.png"/>
</div>
<div align=center>图 3.6 Q 表格如一本生活手册</div>

#### 3.2.3 为什么用未来总奖励评价当前动作？

考虑一个场景（图 3.7）：红灯前普通车应该停，但如果是运送病人的救护车，闯红灯可能是更优的选择——因为**远期的救人奖励远超闯红灯的即时惩罚**。这说明 Q 值必须考虑远期奖励的累积，而非仅看即时奖励。

<div align=center>
<img width="550" src="../img/ch3/3.5.png"/>
</div>
<div align=center>图 3.7 未来的总奖励示例</div>

但也需要**折扣因子 $\gamma$** 来防止目光过于长远。如果任务永远不会结束（**持续式任务**），把所有未来奖励求和会发散到无穷大。引入 $\gamma \in [0,1]$ 后，越远的奖励权值越小（$\gamma^n \to 0$），这就等价于"近期的奖励比远期的更值钱"。

<div align=center>
<img width="550" src="../img/ch3/3.6.png"/>
</div>
<div align=center>图 3.8 股票的例子：太远的奖励不应影响当前决策</div>

#### 3.2.4 悬崖行走问题与折扣因子的效果

> 悬崖行走（cliff walking）是本章的经典测试环境（图 3.9）：智能体从 S 出发到 G，每步 $-1$，掉悬崖 $-100$ 并回到 S。本章后续的 Sarsa 和 Q-learning 对比都基于这个环境。

<div align=center>
<img width="550" src="../img/ch3/3.7.png"/>
</div>
<div align=center>图 3.9 悬崖行走问题</div>

以一条固定路径为例，看 $\gamma$ 如何影响价值计算（图 3.10）：

- $\gamma = 0$：目光短浅，只看单步奖励——所有动作价值相同（都是 $-1$）；
- $\gamma = 1$：目光过于长远，把整条路径的奖励加起来；
- $\gamma = 0.6$：兼顾远期与近期，使用递推公式 $G_t = r_{t+1} + \gamma G_{t+1}$ 从终点反向计算：

$$
\begin{array}{l}
G_{13}=0 \\
G_{12}=r_{13}+\gamma G_{13}=-1+0.6 \times 0=-1 \\
G_{11}=r_{12}+\gamma G_{12}=-1+0.6 \times(-1)=-1.6 \\
G_{10}=r_{11}+\gamma G_{11}=-1+0.6 \times(-1.6)=-1.96 \\
G_{9}=r_{10}+\gamma G_{10}=-1+0.6 \times(-1.96)=-2.176 \approx-2.18 \\
G_{8}=r_{9}+\gamma G_{9}=-1+0.6 \times(-2.176)=-2.3056 \approx-2.3 \\
\end{array}
$$

<div align=center>
<img width="550" src="../img/ch3/3.8.png"/>
</div>
<div align=center>图 3.10 折扣因子的三种效果</div>

---

## 第三部分：免模型预测 —— 如何估计给定策略的价值

> **预测问题**：给定策略 $\pi$（例如均匀随机策略），如何在不知道 $P$ 和 $R$ 的情况下估计 $V_\pi(s)$ 或 $Q_\pi(s,a)$？下面介绍两种核心方法：蒙特卡洛（MC）和时序差分（TD）。

### 3.3 蒙特卡洛方法（MC）

#### 3.3.1 核心思想

蒙特卡洛方法的思路非常直接：**完整地走完一个回合，把实际看到的总奖励作为价值的估计，然后对多条轨迹取平均**。

给定策略 $\pi$，智能体与环境交互产生一条完整的轨迹 $\tau$，该轨迹的回报为：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots + \gamma^{T-t-1} r_T
$$

将所有经过状态 $s$ 的轨迹的 $G_t$ 取平均，就是 $V(s)$ 的估计：

$$
V_\pi(s) = \mathbb{E}_{\tau \sim \pi}[G_t \mid s_t = s] \approx \frac{1}{N(s)} \sum_{i: s \in \tau_i} G_{t,i}
$$

> **关键特征**：MC 不使用自举（bootstrapping），它用的是**真实的完整回报**而非估计值。根据大数定律，$N(s) \to \infty$ 时 $V(s) \to V_\pi(s)$。

#### 3.3.2 MC 的基本步骤

（1）在每个回合中，如果在时间步 $t$ 状态 $s$ 被访问了，那么

- $N(s) \leftarrow N(s) + 1$（访问计数 +1）
- $S(s) \leftarrow S(s) + G_t$（累加回报）

（2）状态 $s$ 的价值通过平均值来估计：$V(s) = S(s) / N(s)$。

**局限**：MC 必须等到**回合结束**才能计算 $G_t$ 并更新，因此只适用于**有终止状态**的环境（episodic tasks），不能用于持续式任务。

#### 3.3.3 增量式 MC 的推导

如果每来一条新轨迹就把所有轨迹的回报重新平均，存储和计算开销很大。**增量均值（incremental mean）**技巧解决了这个问题。

假设有 $t$ 个样本 $x_1, x_2, \ldots, x_t$，其经验均值为 $\mu_t$。可以证明：

$$
\begin{aligned}
\mu_t &= \frac{1}{t}\sum_{j=1}^t x_j
= \frac{1}{t}\left(x_t + \sum_{j=1}^{t-1} x_j\right) \\
&= \frac{1}{t}\left(x_t + (t-1)\mu_{t-1}\right)
= \mu_{t-1} + \frac{1}{t}(x_t - \mu_{t-1})
\end{aligned}
$$

> **直观理解**：新均值 = 旧均值 + 学习率 × （新样本 − 旧均值）。其中 $x_t - \mu_{t-1}$ 是"预测误差"（residual），$\frac{1}{t}$ 的作用类似于学习率——随着样本增多，学习率自动变小，估计越来越稳定。

将这一思想应用于 MC，得到**增量式蒙特卡洛更新**：

$$
\begin{aligned}
N(s_t) &\leftarrow N(s_t) + 1 \\
V(s_t) &\leftarrow V(s_t) + \frac{1}{N(s_t)}\left(G_t - V(s_t)\right)
\end{aligned}
$$

进一步简化为固定学习率 $\alpha$ 的形式（适合非平稳环境）：

$$
V(s_t) \leftarrow V(s_t) + \alpha\left(G_t - V(s_t)\right)
$$

其中 $\alpha$ 是手动设定的学习率，控制每次更新的幅度。$\alpha = \frac{1}{N}$ 就是原始增量 MC，$\alpha$ 固定时则对近期数据更敏感。

---

### 3.4 时序差分方法（TD）

#### 3.4.1 核心思想：走一步就更新

MC 需要等到回合结束才能更新，太"慢"了。时序差分（TD）的核心创新是：**每走一步就更新，用下一步的估计值代替未知的完整回报**。

最简单的 TD 变体是**一步时序差分 TD(0)**，更新公式为：

$$
\boxed{
V(s_t) \leftarrow V(s_t) + \alpha\left(r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\right)
} \tag{3.1}
$$

拆解这个公式的各个部分：

| 符号 | 名称 | 含义 |
|------|------|------|
| $r_{t+1} + \gamma V(s_{t+1})$ | **TD 目标（TD target）** | 对 $G_t$ 的估计：即时奖励 + 折扣的自举后继价值 |
| $r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ | **TD 误差（TD error）** $\delta_t$ | 估计值与当前值的差距 |
| $\alpha$ | 学习率 | 控制更新幅度，$\alpha \in (0, 1]$ |
| $V(s_{t+1})$ | 自举（bootstrap） | 用已有的估计来更新另一个估计 |

> **对比 MC 更新**：MC 用的是 $G_t$（真实完整回报），TD 用的是 $r_{t+1} + \gamma V(s_{t+1})$（单步实际奖励 + 估计的后续价值）。两者的形式完全一致，区别仅在"目标值"不同。

#### 3.4.2 巴甫洛夫条件反射：强化概念的直观类比

图 3.14 用巴甫洛夫实验类比了"强化"的本质：

<div align=center>
<img width="550" src="../img/ch3/3.10.png"/>
</div>
<div align=center>图 3.14 巴甫洛夫条件反射实验</div>

- **无条件刺激**（食物）→ 天然分泌唾液
- **中性刺激**（铃声）→ 最初无反应
- **结合反复出现** → 铃声也能引起唾液分泌（条件反射形成）

这正好对应 TD 更新的逻辑：**奖励（食物）先强化最接近的状态（铃声），然后该状态的价值再反向强化更早的状态**——价值信号像水波一样从奖励源一层层向外传播，如图 3.15 所示。

<div align=center>
<img width="550" src="../img/ch3/3.11.png"/>
</div>
<div align=center>图 3.15 多级条件反射：价值层层反向传播</div>

#### 3.4.3 TD 与 MC 的对比

| 维度 | 蒙特卡洛（MC） | 时序差分 TD(0) |
|------|-------------|-------------|
| **何时更新** | 回合结束后 | 每走一步 |
| **目标值** | $G_t$（完整真实回报） | $r_{t+1} + \gamma V(s_{t+1})$（有偏估计） |
| **自举** | 无 | 有 |
| **能否在线学习** | 否 | 是 |
| **能否不完整序列** | 否（必须完整） | 是 |
| **能否连续环境** | 否（必须有终止） | 是 |
| **马尔可夫假设** | 无假设 | 依赖马尔可夫性质 |
| **偏差** | 无偏 | 有偏（初始 $V$ 不准） |
| **方差** | 高（轨迹随机性大） | 低（只依赖单步） |

> **开车上班的比喻**：MC 是到达公司后才更新"路口 A 堵车→预计迟到多久"；TD 是在路口 A 就根据当前路况立刻更新预计到达时间。TD 更快、更灵活。

#### 3.4.4 $n$ 步 TD：MC 和 TD(0) 的桥梁

TD(0) 只往前走一步就用自举；MC 走完整个回合才更新。**$n$ 步 TD** 提供了一个连续的插值：

$$
\begin{array}{lcl}
n=1\ \text{(TD)} & G_t^{(1)} &= r_{t+1} + \gamma V(s_{t+1}) \\
n=2 & G_t^{(2)} &= r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2}) \\
&\vdots& \\
n=\infty\ \text{(MC)} & G_t^{\infty} &= r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{T-t-1} r_T
\end{array}
$$

一般形式：
$$
G_t^{n} = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^{n} V(s_{t+n})
$$

更新公式保持一致：
$$
V(s_t) \leftarrow V(s_t) + \alpha\left(G_t^{n} - V(s_t)\right)
$$

- $n=1$：纯自举（TD）；
- $n=\infty$：纯采样（MC）；
- $n$ 在中间：兼顾两者的优势。

<div align=center>
<img width="550" src="../img/ch3/TD_5.png"/>
</div>
<div align=center>图 3.18 n 步时序差分：n 越大越接近 MC</div>

---

### 3.5 DP、MC、TD 的统一视角

#### 3.5.1 自举 × 采样的二维分类

三种方法可以从两个维度统一理解——**是否自举**和**是否采样**：

| 方法 | 自举（bootstrap） | 采样（sampling） | 更新公式 |
|------|:---:|:---:|------|
| **动态规划（DP）** | ✓ | ✗ | $V(s_t) \leftarrow \mathbb{E}_\pi[r_{t+1} + \gamma V(s_{t+1})]$ —— 对所有后继状态求期望 |
| **蒙特卡洛（MC）** | ✗ | ✓ | $V(s_t) \leftarrow V(s_t) + \alpha(G_t - V(s_t))$ —— 用采样轨迹的完整回报 |
| **时序差分 TD(0)** | ✓ | ✓ | $V(s_t) \leftarrow V(s_t) + \alpha(r_{t+1} + \gamma V(s_{t+1}) - V(s_t))$ —— 两者兼有 |

#### 3.5.2 广度 × 深度的统一图谱

如图 3.22 所示，三种方法在"更新广度"和"更新深度"两个维度上定位：

- **DP**：广度最大（考虑所有后继状态），但只做一步自举，深度 = 1；
- **TD**：广度窄（只采样一条路径），深度也浅（单步）；
- **MC**：广度窄，但深度深（走到回合终点）；
- **穷举搜索**：广度最大，深度最深——只适用于极小规模问题。

<div align=center>
<img width="550" src="../img/ch3/comparison_5.png"/>
</div>
<div align=center>图 3.22 强化学习的统一视角：广度 vs 深度</div>

#### 3.5.3 偏差-方差权衡

| | MC | TD |
|--|----|----|
| **偏差** | **低**（使用真实 $G_t$ 做目标，无偏估计） | **高**（使用 $V(s_{t+1})$ 做目标，而 $V$ 初始不准） |
| **方差** | **高**（轨迹随机性累积，多步奖励叠加） | **低**（只依赖单步实际奖励 $r_{t+1}$） |

<div align=center>
<img width="550" src="../img/ch3/bias_variance.png"/>
</div>
<div align=center>图 3.27 偏差-方差</div>

**偏差**描述估计值的期望与真实值的差距（越小越准）；**方差**描述估计值自身的波动幅度（越小越稳）。TD 因自举引入偏差但方差低，MC 无偏但方差高——$n$ 步 TD 通过调整 $n$ 在两者之间折中。

---

## 第四部分：免模型控制 —— 寻找最优策略

> 预测问题解决了"给定 $\pi$，估计 $V_\pi$ 或 $Q_\pi$"。控制问题是：**没有给定策略，如何找到 $\pi^*$？** 核心思路是将第 2 章的广义策略迭代（GPI）推广到免模型设定。

### 3.6 广义策略迭代（GPI）与 $\varepsilon$-贪心探索

#### 3.6.1 从策略迭代到 GPI

第 2 章的策略迭代 = 策略评估（$V_\pi$ 或 $Q_\pi$）+ 策略改进（$\arg\max_a Q$）。在免模型下，策略评估不能用 DP，而是用 MC 或 TD，改进步骤保持不变：

$$
\pi_{i+1}(s) = \underset{a}{\arg\max}\ Q_{\pi_i}(s, a) \tag{3.3}
$$

<div align=center>
<img width="550" src="../img/ch3/model_free_control_1.png"/>
</div>
<div align=center>图 3.23 策略迭代：评估与改进交替</div>

<div align=center>
<img width="550" src="../img/ch3/model_free_control_3.png"/>
</div>
<div align=center>图 3.24 广义策略迭代（GPI）：用 MC 或 TD 做评估</div>

#### 3.6.2 $\varepsilon$-贪心探索

但**纯贪心改进**（只选当前 Q 值最大的动作）有一个致命问题：如果某个 $(s,a)$ 对从未被尝试过，它的 Q 值一直是 0（或初始值），智能体永远不会选它——可能错过真正的最优动作。

解决方案：**$\varepsilon$-贪心（$\varepsilon$-greedy）**。以概率 $1-\varepsilon$ 选 Q 值最大的动作（利用），以概率 $\varepsilon$ 随机选动作（探索）：

$$
\pi(a \mid s) = \begin{cases}
1 - \varepsilon + \frac{\varepsilon}{|A|}, & a = \arg\max_{a'} Q(s, a') \\
\frac{\varepsilon}{|A|}, & \text{其他}
\end{cases}
$$

- $\varepsilon$ 通常设为小值（如 0.1），并随时间**衰减**：训练初期多探索，后期多利用；
- $\varepsilon$-贪心保证了所有 $(s,a)$ 对都有非零概率被访问，从而**理论上可以发现最优策略**。

> **$\varepsilon$-贪心单调改进定理**：可以证明，对任意 $\varepsilon$-贪心策略 $\pi$，其对应的贪心改进 $\pi'$ 满足 $V_{\pi'}(s) \ge V_\pi(s)$ 对所有 $s$——策略只会变好，不会变差。

#### 3.6.3 探索性开始 vs $\varepsilon$-贪心

MC 和 TD 要学到最优策略，前提是所有的状态-动作对 $(s,a)$ 都被充分访问过。一种理想的做法叫**探索性开始**：让每个回合从随机均匀采样的 $(s,a)$ 出发，这样覆盖性就有了理论保证。但在真实环境中，我们通常无法任意指定起点（比如游戏每局都从固定关卡开始），所以更实用的替代方案是 **$\varepsilon$-贪心**——通过在每一步以小概率随机选动作，来保证长期内所有 $(s,a)$ 都有机会被探索到。

<div align=center>
<img width="550" src="../img/ch3/model_free_control_4.png"/>
</div>
<div align=center>图 3.25 基于探索性开始的 MC 方法（理论模型）</div>

<div align=center>
<img width="550" src="../img/ch3/model_free_control_7.png"/>
</div>
<div align=center>图 3.26 基于 $\varepsilon$-贪心的 MC 方法（实用方法）</div>

---

### 3.7 Sarsa：同策略 TD 控制

#### 3.7.1 从 TD 预测到 TD 控制

TD 预测更新的是 $V(s)$，而控制需要 $Q(s,a)$ 来选动作。**Sarsa** 的做法很简单：把 TD 公式中的 $V$ 直接替换为 $Q$：

$$
\boxed{
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]
} \tag{3.4}
$$

#### 3.7.2 名称由来与更新流程

该算法每次更新需要用到五个值：**S**tate（当前状态 $s_t$）、**A**ction（当前动作 $a_t$）、**R**eward（奖励 $r_{t+1}$）、**S**tate（下一步状态 $s_{t+1}$）、**A**ction（下一步动作 $a_{t+1}$）——取首字母拼起来就是 **Sarsa**。

如图 3.28 所示，Sarsa 走一步就更新一次：把 $r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$ 当作 TD 目标，$Q(s_t, a_t)$ 用学习率 $\alpha$ 做软更新向目标靠近。

<div align=center>
<img width="550" src="../img/ch3/3.14.png"/>
</div>
<div align=center>图 3.28 时序差分单步更新</div>

更简洁的写法：
$$
Q(S, A) \leftarrow Q(S, A) + \alpha\left(R + \gamma Q(S', A') - Q(S, A)\right)
$$

<div align=center>
<img width="550" src="../img/ch3/3.15.png"/>
</div>
<div align=center>图 3.29 Sarsa 算法</div>

#### 3.7.3 同策略意味着什么？——Sarsa 的"保守"行为

##### 什么是"同策略"？

想象一个厨师同时担任两道角色：

- **掌勺**（行为策略）：负责实际做菜，也就是与环境交互、选择动作；
- **品控**（目标策略）：负责判断这道菜好不好，也就是评估和优化策略。

在 **Sarsa（同策略）** 中，**掌勺和品控是同一个人**——它用同一套 $\varepsilon$-贪心策略来选动作，也用同一套 $\varepsilon$-贪心策略来计算更新目标。也就是说，它**边做什么、边学什么**，两者不分离。

体现在更新公式上就是：
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\Big[r_{t+1} + \gamma\,\underbrace{Q(s_{t+1},\,a_{t+1})}_{\text{实际执行的下一动作}}\Big]
$$

注意 TD 目标里用的是 **$a_{t+1}$**——不是"最优可能的下一步动作"，而是 $\varepsilon$-贪心**真正会执行**的那一个动作。这意味着更新时考虑的是"**以我这套策略继续走下去，会得到什么**"，而不是"假如我下一步突然变完美了，会得到什么"。

##### 为什么同策略让 Sarsa 变"保守"？

在悬崖行走问题中（图 3.9），智能体走在悬崖边上时，$\varepsilon$-贪心策略有概率 $\varepsilon$ 随机选动作——可能刚好选到"向下"掉进悬崖，吃到 $-100$ 的惩罚。**Sarsa 在更新时是知道这一点的**，因为它的 TD 目标里使用的 $a_{t+1}$ 就是 $\varepsilon$-贪心实际采样的动作，而这个动作有非零概率是"掉下去"。

于是 Sarsa 的 Q 值更新会自动把这个风险**折现进去**：

> "在悬崖边，即使最优动作是继续直走，但因为有 $\varepsilon$ 概率随机踩到悬崖，所以**期望上**，悬崖边的 Q 值会被拉低。"

这导致 Sarsa 学到的最优路径会**主动远离悬崖**，宁可绕远一点，也要留出安全距离——因为它的优化目标本身就是"在 $\varepsilon$-贪心策略下的最优表现"，而非"理论上的最短路径"。**Sarsa 保守、稳健**。

> **一句话**：同策略 = 行为策略和目标策略是同一套，更新时用的是**实际执行的 $a_{t+1}$**，所以 Sarsa 无法对探索噪声"视而不见"——它必须把探索可能带来的风险也学到 Q 值里，自然走得更谨慎。

<div align=center>
<img width="550" src="../img/ch3/3.16.png"/>
</div>
<div align=center>图 3.30 Sarsa 代码实现示意：智能体与环境交互一次、学习一次</div>

#### 3.7.4 $n$ 步 Sarsa 与 Sarsa($\lambda$)

类似 TD 预测中的 $n$ 步扩展，Sarsa 也可以使用 $n$ 步回报来做更新：

$$
\begin{array}{lrl}
n=1\ \text{(Sarsa)} & Q_t^{1} &= r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) \\
n=2 & Q_t^{2} &= r_{t+1} + \gamma r_{t+2} + \gamma^2 Q(s_{t+2}, a_{t+2}) \\
&\vdots& \\
n=\infty\ \text{(MC)} & Q_t^{\infty} &= r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{T-t-1} r_T
\end{array} \tag{3.5}
$$

$n$ 步 Q 回报的一般形式：
$$
Q_t^{n} = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^{n} Q(s_{t+n}, a_{t+n})
$$

进一步引入**资格迹（eligibility traces）**参数 $\lambda \in [0,1]$，对所有 $n$ 步回报做指数加权平均，得到 **Sarsa($\lambda$)** 的 Q 回报：

$$
Q_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} Q_t^{n}
$$

更新公式：
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left(Q_t^{\lambda} - Q(s_t, a_t)\right)
$$

> **直观理解**：$\lambda = 0$ 退化为单步 Sarsa（只看眼前一步），$\lambda = 1$ 退化为 MC（看到回合结束）。$\lambda$ 在中间时，越近的步权重越大，越远的步影响越小——相当于同时对不同时间尺度进行学习。

---

### 3.8 Q-learning：异策略 TD 控制

#### 3.8.1 核心思想：把 Sarsa 的 $a_{t+1}$ 换成 $\max$

Q-learning 与 Sarsa 的唯一区别在于**TD 目标的计算方式**。Q-learning 不关心实际执行的下一步动作是什么，它直接假设最优策略会选下一步 Q 值最大的动作：

$$
\boxed{
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)\right]
}
$$

对比 Sarsa：
- Sarsa 目标：$r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$ —— 使用**实际执行**的下一动作
- Q-learning 目标：$r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a)$ —— 使用**最优可能**的下一动作

#### 3.8.2 目标策略 vs 行为策略

Q-learning 是**异策略（off-policy）**算法，它分离了两种策略（图 3.31）：

| | 目标策略 $\pi$（target） | 行为策略 $\mu$（behavior） |
|---|---|---|
| **角色** | 军师——学习最优策略 | 战士——探索环境、采集数据 |
| **动作选择** | 纯贪心：$\arg\max_a Q(s,a)$ | $\varepsilon$-贪心：兼顾探索与利用 |
| **是否与环境交互** | 否 | 是 |
| **数据流向** | 接收行为策略采集的数据进行学习 | 把 $(s,a,r,s')$ 喂给目标策略 |

<div align=center>
<img width="550" src="../img/ch3/3.17.png"/>
</div>
<div align=center>图 3.31 异策略：目标策略（军师）+ 行为策略（战士）</div>

另一个比喻：学习策略（胆小）无法直接在波涛汹涌的大海中交互，但可以派遣激进的探索策略（海盗）去探险，把经验写下来供自己学习（图 3.32）。

<div align=center>
<img width="550" src="../img/ch3/off_policy_learning.png"/>
</div>
<div align=center>图 3.32 异策略的"军师-海盗"比喻</div>

#### 3.8.3 异策略意味着什么？——Q-learning 的"激进"行为

##### 什么是"异策略"？

回到厨师的比喻。在 **Q-learning（异策略）** 中，**掌勺和品控是两个人**：

- **掌勺**（行为策略 $\mu$）：用 $\varepsilon$-贪心策略跟环境交互——该探索就探索，该随机就随机，只管"多跑、多试、多收集数据"；
- **品控**（目标策略 $\pi$）：用纯贪心策略 $\arg\max_a Q$ 来评估——只问"如果下一步按最优来走，这个动作值多少钱"，完全不考虑掌勺可能瞎走。

体现在更新公式上就是：
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\Big[r_{t+1} + \gamma\,\underbrace{\max_a Q(s_{t+1},\,a)}_{\text{最优可能的下一动作}}\Big]
$$

TD 目标里用的是 **$\max_a Q(s_{t+1}, a)$**，而不是实际执行的 $a_{t+1}$。也就是说，Q-learning 更新时**对探索噪声"视而不见"**——它假设下一步一定是当前 Q 表里最好的那个动作，无论掌勺实际是怎么走的。

##### 为什么异策略让 Q-learning 变"激进"？

在悬崖行走问题中，即使 $\varepsilon$-贪心策略下一步有概率随机掉进悬崖，Q-learning 在更新时也**不考虑这个风险**——它的 TD 目标只关心"如果下一步走最优动作，能拿多少分"。

因此 Q-learning 的 Q 值不会被探索噪声拉低：

> "在悬崖边，只要最优动作是继续直走，$\max_a Q(s_{t+1}, a)$ 给出的就是直走的价值——探索可能带来的惩罚不会进入更新公式。"

这导致 Q-learning 学到的最优路径会**贴着悬崖走**，走最短的路——因为它的优化目标是"理论最优策略的表现"，而非"带着探索噪声的策略的表现"。只要探索运气好没真掉下去，学到的就是最短路径。**Q-learning 激进、冒险**。

> **一句话**：异策略 = 行为策略和目标策略是两套，更新时用的是**理论上最优的 $\max_a Q$**，所以 Q-learning 可以一边用 $\varepsilon$-贪心随便探索，一边无视探索噪声只学最优部分——自然走得更直、更敢冒险。

<div align=center>
<img width="550" src="../img/ch3/3.18.png"/>
</div>
<div align=center>图 3.33 Sarsa 与 Q-learning 的伪代码对比</div>

如图 3.34a，Sarsa 用 $(S,A,R,S',A')$ 五元组更新，$A'$ 是实际执行的下一步动作；Q-learning 只需 $(S,A,R,S')$ 四元组，$A'$ 是虚拟的最优动作。

<div align=center>
<img width="550" src="../img/ch3/3.19.png"/>
</div>
<div align=center>图 3.34 Sarsa 与 Q-learning 的区别：是否需要实际 $A'$</div>

#### 3.8.4 异策略学习的三大优势

1. **更高效**：行为策略可以大胆探索，不必顾虑策略质量，目标策略只学习最优部分；
2. **可模仿学习**：可以从其他智能体（甚至人类）产生的经验中学习；
3. **可重用经验**：旧的轨迹数据可以反复用来学习，节省探索计算资源。这一点正是后续 **DQN** 等算法中 **经验回放（experience replay）** 的思想基础。

---

### 3.9 同策略与异策略深度对比

#### 3.9.1 公式对比

| | Sarsa（同策略） | Q-learning（异策略） |
|---|---|---|
| **更新公式** | $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)]$ | $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t,a_t)]$ |
| **需要的经验** | $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ 五元组 | $(s_t, a_t, r_{t+1}, s_{t+1})$ 四元组 |
| **TD 目标的关键操作** | $Q(s_{t+1}, a_{t+1})$（采样下一动作） | $\max_a Q(s_{t+1}, a)$（取最大值） |

#### 3.9.2 行为特征对比（悬崖行走可视化）

| 维度 | Sarsa | Q-learning |
|------|-------|-----------|
| **策略类型** | 同策略（on-policy）——学习什么就做什么 | 异策略（off-policy）——做什么和数据学什么分离 |
| **对探索的态度** | 保守——考虑到 $\varepsilon$-贪心的随机探索可能踩坑 | 激进——更新时假设下一步必然最优 |
| **悬崖行走路径** | 远离悬崖，走安全但绕远的路径 | 贴着悬崖，走最短路径 |
| **收敛稳定性** | 策略因 $\varepsilon$ 变化而不稳定，最终收敛到 $\varepsilon$-贪心下的最优 | 行为策略和目标策略分离，学习更稳定 |
| **最终策略** | $\varepsilon$-贪心策略（非纯贪心） | 纯贪心策略（$\arg\max_a Q$） |

#### 3.9.3 一句话区分

- **Sarsa**：我用什么动作更新，就用什么动作走路——所以我小心。$Q(s_t,a_t)$ 向 $r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$ 靠近。
- **Q-learning**：我用最好的动作来更新，至于我实际怎么走那是另一回事——所以我大胆。$Q(s_t,a_t)$ 向 $r_{t+1} + \gamma \max_a Q(s_{t+1}, a)$ 靠近。

---

## 本章知识点总结

| 知识点 | 核心内容 | 所在章节 |
|--------|---------|---------|
| **免模型 vs 有模型** | 未知 $P,R$ 时从经验中学习 vs 已知 $P,R$ 时用 DP 规划 | 3.1 |
| **Q 表格** | 行=状态，列=动作的查找表，免模型下直接学习 $Q(s,a)$ | 3.2 |
| **蒙特卡洛（MC）** | 完整回合的真实回报取平均；无自举，高方差，无偏；需终止状态 | 3.3 |
| **增量均值推导** | $\mu_t = \mu_{t-1} + \frac{1}{t}(x_t - \mu_{t-1})$ → 适用于增量式更新 | 3.3.3 |
| **TD(0)** | 单步更新：$V(s_t) \leftarrow V(s_t) + \alpha(r_{t+1} + \gamma V(s_{t+1}) - V(s_t))$；有自举，低方差，有偏 | 3.4 |
| **$n$ 步 TD** | $n=1$ 是 TD，$n=\infty$ 是 MC，$n$ 可在两者间插值 | 3.4.4 |
| **DP/MC/TD 统一** | 自举（是/否）× 采样（是/否）的二维分类；广度 vs 深度的统一图谱 | 3.5 |
| **偏差-方差** | MC 无偏高方差；TD 有偏低方差；$n$ 步 TD 折中 | 3.5.3 |
| **GPI** | 策略评估（MC/TD）+ 策略改进（贪心）= 免模型控制框架 | 3.6 |
| **$\varepsilon$-贪心** | $1-\varepsilon$ 利用 + $\varepsilon$ 探索，保证所有 $(s,a)$ 被访问 | 3.6.2 |
| **Sarsa** | 同策略 TD 控制：用实际 $a_{t+1}$ 更新，行为保守 | 3.7 |
| **Sarsa($\lambda$)** | 对所有 $n$ 步回报指数加权平均，$\lambda=0$ → 单步，$\lambda=1$ → MC | 3.7.4 |
| **Q-learning** | 异策略 TD 控制：用 $\max_a Q$ 更新，行为激进；分离目标与行为策略 | 3.8 |
| **on/off-policy** | Sarsa 学习策略 = 行为策略；Q-learning 学习策略 ≠ 行为策略 | 3.9 |

图 3.35 对表格型方法做了全面总结。

<div align=center>
<img width="550" src="../img/ch3/3.21.png"/>
</div>
<div align=center>图 3.35 表格型方法总结</div>

---

## 参考文献

* [百度强化学习](https://aistudio.baidu.com/aistudio/education/lessonvideo/460292)
* [强化学习基础 David Silver 笔记](https://zhuanlan.zhihu.com/c_135909947)
* [Intro to Reinforcement Learning (强化学习纲要）](https://github.com/zhoubolei/introRL)
* [Reinforcement Learning: An Introduction (second edition)](https://book.douban.com/subject/30323890/)
* [百面深度学习](https://book.douban.com/subject/35043939/)
* [神经网络与深度学习](https://nndl.github.io/)
* [机器学习](https://book.douban.com/subject/26708119//)
* [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)
