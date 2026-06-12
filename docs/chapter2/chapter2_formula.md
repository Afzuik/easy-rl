# 第 2 章 马尔可夫决策过程 — 概念与公式速查

> 按「**MP → MRP → MDP → 预测 → 控制**」递进顺序编排。
> 公式编号与 `chapter2_order.md` 一致。

---

## 第一部分：马尔可夫过程（MP）

### 2.1 马尔可夫性质

> 给定现在状态，未来状态与过去状态**条件独立**。

$$
p\left(X_{t+1}=x_{t+1} \mid X_{0:t}=x_{0:t}\right)=p\left(X_{t+1}=x_{t+1} \mid X_{t}=x_{t}\right)
$$

| 术语 | 含义 |
|------|------|
| $X_0, X_1, \cdots, X_T$ | 随机过程（离散时间） |
| 状态空间（state space） | 随机变量所有可能取值的集合 |

---

### 2.2 马尔可夫链

> **马尔可夫过程**：一组具有马尔可夫性质的随机变量序列 $s_1, \cdots, s_t$。
> **马尔可夫链**：离散时间、有限状态的马尔可夫过程。

$$
p\left(s_{t+1} \mid s_{t}\right) = p\left(s_{t+1} \mid h_{t}\right)
\qquad\text{其中 } h_t = \{s_1, s_2, \dots, s_t\} \tag{2.1}
$$

#### 状态转移矩阵 $\boldsymbol{P}$

$$
\boldsymbol{P}=
\begin{pmatrix}
p(s_1\mid s_1) & p(s_2\mid s_1) & \dots & p(s_N\mid s_1) \\
p(s_1\mid s_2) & p(s_2\mid s_2) & \dots & p(s_N\mid s_2) \\
\vdots & \vdots & \ddots & \vdots \\
p(s_1\mid s_N) & p(s_2\mid s_N) & \dots & p(s_N\mid s_N)
\end{pmatrix}
$$

> 每一行是一个**条件概率分布**，$\sum_{j} P_{ij} = 1$。

---

## 第二部分：马尔可夫奖励过程（MRP）

### 2.4 MRP 定义

> MRP = MP + **奖励函数 $R(s)$** + **折扣因子 $\gamma$**

| 元素 | 定义 |
|------|------|
| $R(s)$ | 到达状态 $s$ 时的期望即时奖励 |
| $\gamma \in [0,1]$ | 折扣因子，控制未来奖励的重要性 |

---

### 2.5 回报与状态价值函数

#### 回报 $G_t$

$$
G_{t}=r_{t+1}+\gamma r_{t+2}+\gamma^{2} r_{t+3}+\dots+\gamma^{T-t-1} r_{T}
$$

| 术语 | 含义 |
|------|------|
| $T$ | 最终时刻 |
| $r_t$ | 时刻 $t$ 的即时奖励 |
| Horizon（范围） | 一个回合的长度 |

#### 状态价值函数 $V(s)$

$$
V(s)=\mathbb{E}\left[G_{t} \mid s_{t}=s\right]
=\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^{2} r_{t+3}+\dots \mid s_{t}=s\right]
$$

> 从状态 $s$ 出发，未来**折扣回报的期望**。

#### 使用折扣因子 $\gamma$ 的四个理由

1. 避免环状过程产生无穷奖励
2. 表示对模型的不确定性（未来不完美）
3. 即时奖励更有实际价值
4. 作为可调超参数，控制智能体行为

---

### 2.6 贝尔曼方程（MRP）

$$
\boxed{V(s)=\underbrace{R(s)}_{\text{即时奖励}}+\underbrace{\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)}_{\text{未来奖励的折扣总和}}}
$$

> 当前状态价值 = 即时奖励 + 未来所有可能状态的折扣价值期望。

#### 矩阵形式

$$
\boldsymbol{V} = \boldsymbol{R} + \gamma \boldsymbol{P} \boldsymbol{V}
$$

#### 解析解

$$
\boxed{\boldsymbol{V}=(\boldsymbol{I}-\gamma \boldsymbol{P})^{-1} \boldsymbol{R}}
$$

> ⚠ 矩阵求逆复杂度 $O(N^3)$，仅适用于小规模 MRP。

#### 推导依赖：全期望公式

$$
\mathbb{E}[X]=\sum_{i} \mathbb{E}\left[X \mid A_{i}\right] p\left(A_{i}\right)
$$

辅助恒等式：$\mathbb{E}[V(s_{t+1})|s_t]=\mathbb{E}[G_{t+1}|s_t]$

---

### 2.7 计算 MRP 价值的三种迭代方法

| 方法 | 核心思想 |
|------|----------|
| **蒙特卡洛 (MC)** | 采样轨迹 → 计算回报 → 取平均值估计 $V(s)$ |
| **动态规划 (DP)** | 迭代贝尔曼方程直到收敛（自举 bootstrapping） |
| **时序差分 (TD)** | MC 与 DP 的结合 |

---

## 第三部分：马尔可夫决策过程（MDP）

### 2.9 MDP 定义

> MDP = MRP + **动作 $a$** + **策略 $\pi$**

| 元素 | 定义 |
|------|------|
| $p(s' \mid s, a)$ | 状态转移概率（依赖动作 $a$） |
| $R(s, a)$ | 奖励函数（依赖动作 $a$） |

$$
p\left(s_{t+1} \mid s_{t}, a_{t}\right) = p\left(s_{t+1} \mid h_{t}, a_{t}\right)
$$

---

### 2.10 策略 $\pi$

$$
\pi(a \mid s) = p\left(a_{t}=a \mid s_{t}=s\right)
$$

> 给定状态 $s$，选择动作 $a$ 的概率分布（平稳策略）。

#### MDP → MRP 化简（给定策略后去动作）

$$
P_{\pi}\left(s^{\prime} \mid s\right)=\sum_{a \in A} \pi(a \mid s) \, p\left(s^{\prime} \mid s, a\right)
$$

$$
r_{\pi}(s)=\sum_{a \in A} \pi(a \mid s) \, R(s, a)
$$

---

### 2.11 价值函数

#### 状态价值函数 $V_{\pi}$

$$
\boxed{V_{\pi}(s)=\mathbb{E}_{\pi}\left[G_{t} \mid s_{t}=s\right]} \tag{2.3}
$$

#### 动作价值函数（Q 函数）$Q_{\pi}$

$$
\boxed{Q_{\pi}(s, a)=\mathbb{E}_{\pi}\left[G_{t} \mid s_{t}=s, a_{t}=a\right]} \tag{2.4}
$$

#### $V$ 与 $Q$ 的转换

$$
V_{\pi}(s)=\sum_{a \in A} \pi(a \mid s) \, Q_{\pi}(s, a) \tag{2.5}
$$

#### Q 函数的贝尔曼方程推导

$$
Q(s,a)=R(s,a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s,a\right) V\left(s^{\prime}\right)
$$

---

### 2.12 贝尔曼期望方程

#### $V$ 的贝尔曼期望方程

$$
\boxed{V_{\pi}(s)=\mathbb{E}_{\pi}\left[r_{t+1}+\gamma V_{\pi}\left(s_{t+1}\right) \mid s_{t}=s\right]} \tag{2.6}
$$

#### $Q$ 的贝尔曼期望方程

$$
\boxed{Q_{\pi}(s, a)=\mathbb{E}_{\pi}\left[r_{t+1}+\gamma Q_{\pi}\left(s_{t+1}, a_{t+1}\right) \mid s_{t}=s, a_{t}=a\right]} \tag{2.7}
$$

#### 两种等价展开形式

**形式一：$V$ 用未来 $V$ 表示**

$$
V_{\pi}(s)=\sum_{a} \pi(a\mid s)\Big(R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_{\pi}(s')\Big) \tag{2.10}
$$

**形式二：$Q$ 用未来 $Q$ 表示**

$$
Q_{\pi}(s,a)=R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q_{\pi}(s',a') \tag{2.11}
$$

---

### 备份图（Backup Diagram）

| 符号 | 含义 |
|------|------|
| ◯ 空心圆 | 状态 |
| ⬤ 实心圆 | 状态-动作对 |

> 备份图展示了价值信息从后继状态（或状态-动作对）**反向传播**到当前节点。

$V$ 备份（两层加和）：

$$
V_{\pi}(s)=\sum_{a} \pi(a\mid s)\Big(R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_{\pi}(s')\Big) \tag{2.12}
$$

$Q$ 备份（两层加和）：

$$
Q_{\pi}(s,a)=R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q_{\pi}(s',a') \tag{2.15}
$$

---

## 第四部分：预测（Prediction）

### 2.13 预测问题

| 输入 | 输出 |
|------|------|
| MDP $\langle S,A,P,R,\gamma \rangle$ + 策略 $\pi$ | 价值函数 $V_{\pi}$ |

> **策略评估** = 给定策略 $\pi$，计算每个状态的价值 $V_{\pi}(s)$。

---

### 2.14 策略评估的迭代算法

#### 贝尔曼期望备份 → 迭代

$$
\boxed{V^{t+1}(s)=\sum_{a} \pi(a\mid s)\Big(R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V^{t}(s')\Big)} \tag{2.18}
$$

#### 化归为 MRP 形式

$$
\boxed{V_{t+1}(s)=r_{\pi}(s)+\gamma P_{\pi}\left(s^{\prime} \mid s\right) V_{t}\left(s^{\prime}\right)} \tag{2.19}
$$

| 术语 | 含义 |
|------|------|
| 同步备份 | 每次迭代更新所有状态 |
| 异步备份 | 每次迭代只更新部分状态 |

---

## 第五部分：控制（Control）

### 2.15 控制问题

| 输入 | 输出 |
|------|------|
| MDP $\langle S,A,P,R,\gamma \rangle$ | 最优价值函数 $V^*$ + 最优策略 $\pi^*$ |

#### 最优价值函数

$$
\boxed{V^{*}(s)=\max _{\pi} V_{\pi}(s)}
$$

#### 最优策略

$$
\boxed{\pi^{*}(s)=\underset{\pi}{\arg \max } ~ V_{\pi}(s)}
$$

#### 从 $Q^*$ 提取最优策略

$$
\pi^{*}(a \mid s)=
\begin{cases}
1, & a=\underset{a \in A}{\arg \max} ~ Q^{*}(s, a) \\
0, & \text{其他}
\end{cases}
$$

> 穷举复杂度 $|A|^{|S|}$，不现实 → 需要**策略迭代**或**价值迭代**。

---

### 2.16 动态规划（DP）适用条件

| 条件 | 含义 |
|------|------|
| **最优子结构** | 问题可拆成子问题，组合子解得原解 |
| **重叠子问题** | 子问题重复出现，结果可缓存复用 |

> MDP 的贝尔曼方程天然满足这两个条件。DP 用于**规划问题**（环境完全已知）。

---

### 2.17 贝尔曼最优方程

$$
\boxed{V_{\pi}(s)=\max _{a \in A} Q_{\pi}(s, a)}
$$

> 最优策略下，状态价值 = 该状态下最好动作的 Q 值。

#### $V^*$ 形式

$$
\boxed{V^{*}(s)=\max _{a} Q^{*}(s, a)} \tag{2.20}
$$

#### $Q^*$ 形式（贝尔曼最优方程的 Q 版本）

$$
\boxed{Q^{*}(s, a)=R(s, a)+\gamma \sum_{s^{\prime}} p\left(s^{\prime} \mid s, a\right) \max _{a^{\prime}} Q^{*}\left(s^{\prime}, a^{\prime}\right)}
$$

#### $V^*$ 展开式

$$
\boxed{V^{*}(s)=\max_{a}\Big(R(s,a) + \gamma \sum_{s^{\prime}} p(s^{\prime}\mid s,a) V^{*}(s^{\prime})\Big)}
$$

---

### 2.18 策略迭代

#### 两步循环

```
策略评估 (Policy Evaluation)  →  策略改进 (Policy Improvement)
         ↑                              │
         └──────────────────────────────┘
```

#### 策略改进公式

$$
Q_{\pi_{i}}(s, a)=R(s, a)+\gamma \sum_{s^{\prime}} p(s^{\prime}\mid s,a) V_{\pi_{i}}(s^{\prime})
$$

$$
\boxed{\pi_{i+1}(s)=\underset{a}{\arg \max } ~ Q_{\pi_{i}}(s, a)}
$$

> Q 表格：横轴为状态，纵轴为动作，每列取 arg max 得到最优动作。

---

### 2.19 价值迭代

#### 最优性原理

> $\pi(a|s)$ 在 $s$ 达到 $V^{*}(s)$ ⇔ 所有可达后继 $s'$ 都已达到 $V^{*}(s')$

#### 迭代更新规则（贝尔曼最优方程作 backup）

$$
\boxed{V(s) \leftarrow \max _{a}\Big(R(s,a)+\gamma \sum_{s^{\prime}} p(s^{\prime}\mid s,a) V(s^{\prime})\Big)} \tag{2.22}
$$

#### 算法流程

1. **初始化**：$k=0$，$\forall s,\; V_0(s)=0$
2. **迭代**：对于 $k=1:H$

   $$
   Q_{k+1}(s,a)=R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_k(s') \tag{2.23}
   $$

   $$
   V_{k+1}(s)=\max_a Q_{k+1}(s,a) \tag{2.24}
   $$

3. **提取最优策略**：

   $$
   \pi(s)=\underset{a}{\arg \max}\Big[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_{H+1}(s')\Big]
   $$

> 中间过程的策略和价值无意义；收敛后提取策略。

---

### 2.20 策略迭代 vs 价值迭代

| | 策略迭代 | 价值迭代 |
|------|----------|----------|
| **使用的方程** | 贝尔曼期望方程 | 贝尔曼最优方程 |
| **过程** | 策略评估 + 策略改进交替 | 直接迭代 $V$ |
| **中间结果** | 每轮产出完整策略，有意义 | 中间值无意义 |
| **收敛后** | 策略不再变化 | $V$ 不再变化 |
| **提取策略** | 收敛即得 | 从 $V^*$ 用 arg max 提取 |

---

### 2.21 预测与控制总结

| 问题 | 算法 | 使用的贝尔曼方程 |
|------|------|------------------|
| **预测** (给定 $\pi$, 求 $V_\pi$) | 策略评估（迭代） | **贝尔曼期望方程** |
| **控制** (求 $V^*$ 和 $\pi^*$) | 策略迭代 | **贝尔曼期望方程** |
| **控制** (求 $V^*$ 和 $\pi^*$) | 价值迭代 | **贝尔曼最优方程** |

---

## 公式索引

| 编号 | 公式 | 内容 |
|------|------|------|
| (2.1) | $p(s_{t+1}\mid s_t)=p(s_{t+1}\mid h_t)$ | 马尔可夫链条件 |
| (2.3) | $V_\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]$ | 状态价值函数 |
| (2.4) | $Q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]$ | 动作价值函数 |
| (2.5) | $V_\pi(s)=\sum_a \pi(a\mid s)Q_\pi(s,a)$ | V 与 Q 的转换 |
| (2.6) | $V_\pi(s)=\mathbb{E}_\pi[r_{t+1}+\gamma V_\pi(s_{t+1})\mid s_t=s]$ | 贝尔曼期望方程 (V) |
| (2.7) | $Q_\pi(s,a)=\mathbb{E}_\pi[r_{t+1}+\gamma Q_\pi(s_{t+1},a_{t+1})\mid s_t=s,a_t=a]$ | 贝尔曼期望方程 (Q) |
| (2.10) | $V_\pi(s)=\sum_a\pi(a\mid s)[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_\pi(s')]$ | V 用未来 V 表示 |
| (2.11) | $Q_\pi(s,a)=R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q_\pi(s',a')$ | Q 用未来 Q 表示 |
| (2.18) | $V^{t+1}(s)=\sum_a\pi(a\mid s)[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V^t(s')]$ | 策略评估迭代式 |
| (2.19) | $V_{t+1}(s)=r_\pi(s)+\gamma P_\pi(s'\mid s)V_t(s')$ | MRP 化简迭代式 |
| (2.20) | $V^*(s)=\max_a Q^*(s,a)$ | 贝尔曼最优方程 (V) |
| (2.22) | $V(s)\leftarrow\max_a[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V(s')]$ | 价值迭代更新规则 |
| — | $\pi_{i+1}(s)=\arg\max_a Q_{\pi_i}(s,a)$ | 策略改进 |
| — | $V=(I-\gamma P)^{-1}R$ | MRP 贝尔曼方程解析解 |
