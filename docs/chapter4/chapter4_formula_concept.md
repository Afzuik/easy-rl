# 第 4 章 策略梯度 —— 核心概念与公式总结

---

## 一、强化学习三要素

| 角色 | 含义 | 可控？ |
|---|---|---|
| **演员（actor）** | 策略网络 $\pi_\theta$，输入状态 $s$，输出动作概率分布 | ✅ 唯一可控 |
| **环境（environment）** | 游戏/物理规则，根据 $(s_t, a_t)$ 给出 $s_{t+1}$ | ❌ 不可控 |
| **奖励函数（reward function）** | 对 $(s_t, a_t)$ 打分 | ❌ 不可控 |

> 策略 $\pi_\theta$ 是一个神经网络，参数为 $\theta$。输出是各动作的概率，**按概率采样**选择动作（保证探索性，类似 $\varepsilon$-贪心）。

---

## 二、轨迹与期望奖励

### 2.1 轨迹（Trajectory）

一场完整的交互序列：

$$
\tau = \{s_1, a_1, s_2, a_2, \cdots, s_T, a_T\}
$$

### 2.2 回报（Return）

一条轨迹所有奖励之和：

$$
R(\tau) = \sum_{t=1}^{T} r_t
$$

### 2.3 轨迹出现的概率

$$
p_{\theta}(\tau) = p(s_1) \prod_{t=1}^{T} p_{\theta}(a_t \mid s_t) \, p(s_{t+1} \mid s_t, a_t)
$$

| 项 | 含义 | 由谁决定 |
|---|---|---|
| $p(s_1)$ | 初始状态概率 | 环境 |
| $p_{\theta}(a_t \mid s_t)$ | 策略：在 $s_t$ 选 $a_t$ 的概率 | **演员（唯一可优化）** |
| $p(s_{t+1} \mid s_t, a_t)$ | 状态转移概率 | 环境 |

### 2.4 期望奖励（优化目标）

$R(\tau)$ 是随机变量（策略采样 + 环境随机性），我们最大化其期望：

$$
\boxed{\bar{R}_{\theta} = \sum_{\tau} R(\tau) \, p_{\theta}(\tau) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}[R(\tau)]}
$$

---

## 三、策略梯度推导（核心）

### 目标

梯度上升更新参数：

$$
\theta \leftarrow \theta + \eta \, \nabla \bar{R}_{\theta}
$$

### 推导四步走

#### 第一步：对 $\bar{R}_\theta$ 求梯度

$$
\nabla \bar{R}_{\theta} = \sum_{\tau} R(\tau) \, \nabla p_{\theta}(\tau)
$$

> $R(\tau)$ 与 $\theta$ 无关，不需要可微——即便奖励是黑盒规则也成立。

#### 第二步：对数求导技巧（log-derivative trick）

核心恒等式：

$$
\boxed{\nabla \log f(x) = \frac{\nabla f(x)}{f(x)} \quad \Longrightarrow \quad \nabla f(x) = f(x) \, \nabla \log f(x)} \tag{4.1}
$$

代入得：

$$
\nabla p_{\theta}(\tau) = p_{\theta}(\tau) \, \nabla \log p_{\theta}(\tau)
$$

于是：

$$
\boxed{\nabla \bar{R}_{\theta} = \sum_{\tau} R(\tau) \, p_{\theta}(\tau) \, \nabla \log p_{\theta}(\tau) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\big[\, R(\tau) \, \nabla \log p_{\theta}(\tau) \,\big]} \tag{4.2}
$$

> **目的**：把"对乘积求导"变成"对求和求导"，并且写成期望形式以便采样近似。

#### 第三步：采样近似 + 展开 log

用 $N$ 条轨迹近似期望：

$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \, \nabla \log p_{\theta}(\tau^n)
$$

展开 $\log p_\theta(\tau)$：

$$
\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^{T} \log p_\theta(a_t \mid s_t) + \sum_{t=1}^{T} \log p(s_{t+1} \mid s_t, a_t)
$$

对 $\theta$ 求梯度，环境相关项梯度为 0：

$$
\boxed{\nabla \log p_{\theta}(\tau) = \sum_{t=1}^{T} \nabla \log p_{\theta}(a_t \mid s_t)}
$$

> **关键结论**：即便完全不知道环境动力学 $p(s_{t+1}\mid s_t,a_t)$，也能算出策略梯度——天然是**免模型（model-free）**方法。

#### 第四步：最终梯度公式

$$
\boxed{\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \, \nabla \log p_{\theta}(a_t^n \mid s_t^n)} \tag{4.3}
$$

---

## 四、策略梯度与分类问题的联系

| | 普通分类 | 策略梯度 |
|---|---|---|
| 输入 | 图像 | 状态 $s_t$ |
| 输出 | 各类别概率 | 各动作概率 |
| 标签 | 人工标注（正确） | 采样动作 $a_t$（不一定正确） |
| 目标函数 | $\frac{1}{N}\sum\sum \log p_\theta(a_t\mid s_t)$ | $\frac{1}{N}\sum\sum R(\tau^n) \log p_\theta(a_t\mid s_t)$ |

**核心公式（加权最大似然）**：

$$
\boxed{\nabla \bar{R}_{\theta} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \, \nabla \log p_{\theta}(a_t^n \mid s_t^n)} \tag{4.4}
$$

> 策略梯度 = 在交叉熵的每一项前面乘以**奖励权重 $R(\tau)$**。奖励高 → 权重大；奖励低 → 权重小甚至为负。

### 同策略（on-policy）特性

采样数据**只用一次**——更新参数后旧策略已变，旧数据作废，必须重新采样。这是策略梯度采样效率低的根源。

---

## 五、两个关键技巧

### 5.1 技巧一：添加基线（baseline）

#### 问题

奖励全是正数 → 未采样到的动作概率被归一化"挤"下去，但可能它只是没被采到而非真的不好。

#### 解法

减去基线 $b$，让奖励有正有负：

$$
\boxed{\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \big(R(\tau^n) - b\big) \, \nabla \log p_{\theta}(a_t^n \mid s_t^n)}
$$

- $R(\tau) > b$ → 提升概率
- $R(\tau) < b$ → 降低概率

$b$ 通常取过去回报的滑动平均：$b \approx \mathbb{E}[R(\tau)]$。

### 5.2 技巧二：信用分配（credit assignment）

#### 问题

用整场总分 $R(\tau)$ 给每一步加权不公平——$t$ 时刻的动作只能影响**之后**的奖励。

#### 解法一：只算该动作之后的奖励

$$
\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \Bigg(\sum_{t'=t}^{T_n} r_{t'}^n - b\Bigg) \, \nabla \log p_{\theta}(a_t^n \mid s_t^n)
$$

#### 解法二：引入折扣因子 $\gamma$

$$
\boxed{\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \Bigg(\sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n - b\Bigg) \, \nabla \log p_{\theta}(a_t^n \mid s_t^n)}
$$

- $\gamma = 0$：只看眼前（短视）
- $\gamma = 1$：未来同等重要（长远）

#### 优势函数（advantage function）

$$
\boxed{A^{\theta}(s_t, a_t) = \sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n - b}
$$

> **含义**：在状态 $s_t$ 采取 $a_t$，**相对于其他可能动作有多好**（"相对"是因为减了基线）。

---

## 六、REINFORCE：蒙特卡洛策略梯度

### 蒙特卡洛 vs 时序差分

| 方法 | 更新时机 | 用什么估计 |
|---|---|---|
| **蒙特卡洛（MC）** | 回合结束后 | 实际折扣回报 $G_t$ |
| **时序差分（TD）** | 每步更新 | Q 函数自举估计 |

### $G_t$ 的递推计算

$$
\boxed{G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k = r_{t+1} + \gamma G_{t+1}} \tag{4.8}
$$

> **实现技巧**：从后往前递推，一次遍历完成。

### REINFORCE 算法流程

1. 用当前策略 $\pi_\theta$ 跑完一回合，收集 $(s_1, a_1, r_1), \cdots, (s_T, a_T, r_T)$
2. 从后向前计算 $G_t$
3. 对每个 $t$ 计算梯度 $G_t \nabla \log \pi_\theta(a_t \mid s_t)$
4. 累加梯度，更新参数

### 梯度公式

$$
\boxed{\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} G_t^n \, \nabla \log \pi_{\theta}(a_t^n \mid s_t^n)}
$$

### 独热编码技巧

动作 $a_t$ 转为独热向量，与对数概率向量做内积即可取出对应动作的对数概率：

$$
\text{one-hot}(a_t) \cdot \log \pi_\theta(\cdot \mid s_t) = \log \pi_\theta(a_t \mid s_t)
$$

---

## 七、核心概念速查表

| 概念 | 一句话解释 |
|---|---|
| 策略 $\pi_\theta$ | 输入状态、输出动作概率分布的神经网络 |
| 轨迹 $\tau$ | 一回合 $(s_1, a_1, s_2, a_2, \ldots)$ 的完整序列 |
| 期望奖励 $\bar{R}_\theta$ | 所有轨迹回报按出现概率加权平均 |
| 对数求导技巧 | $\nabla f(x) = f(x) \nabla \log f(x)$，把乘积求导变求和求导 |
| 基线 $b$ | 减去后让奖励有正有负，避免未采样动作被误伤 |
| 优势函数 $A^\theta(s_t, a_t)$ | 动作后折扣回报减基线，衡量动作的相对好坏 |
| REINFORCE | 蒙特卡洛策略梯度：跑完回合 → 倒推 $G_t$ → 加权交叉熵 → 反向传播 |

---

## 八、策略梯度的局限

| 缺陷 | 说明 | 后续改进算法 |
|---|---|---|
| 采样效率低 | 同策略，每次更新后必须重新采样 | PPO、TRPO |
| 方差大 | 奖励随机性被乘进梯度 | Actor-Critic、GAE |

---

## 九、关键公式索引

| 编号 | 公式 | 含义 |
|---|---|---|
| (4.1) | $\nabla f(x) = f(x) \nabla \log f(x)$ | 对数求导技巧 |
| (4.2) | $\nabla \bar{R}_{\theta} = \mathbb{E}_{\tau}[R(\tau) \nabla \log p_{\theta}(\tau)]$ | 期望形式的策略梯度 |
| (4.3) | $\nabla \bar{R}_{\theta} \approx \frac{1}{N}\sum_n\sum_t R(\tau^n) \nabla \log p_{\theta}(a_t^n\mid s_t^n)$ | 采样近似的策略梯度 |
| (4.4) | 同上，写成 $\log$ 形式 | 策略梯度最终公式 |
| (4.8) | $G_t = r_{t+1} + \gamma G_{t+1}$ | 折扣回报的递推计算 |

---

> **本章地位**：策略梯度是连接深度学习和强化学习的桥梁，它让我们可以把任意复杂神经网络（CNN、RNN、Transformer）当作策略来训练。掌握策略梯度的思想是理解后续 PPO、TRPO、Actor-Critic 等所有"基于策略"的 DRL 算法的基础。
