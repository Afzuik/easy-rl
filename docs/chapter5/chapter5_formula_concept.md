# 第 5 章 PPO 算法 —— 概念与公式总结

---

## 一、核心概念

### 1.1 同策略（On-Policy）vs 异策略（Off-Policy）

| | 同策略（on-policy） | 异策略（off-policy） |
|---|---|---|
| **交互智能体** | $\pi_\theta$（正在学习的策略） | $\pi_{\theta'}$（固定的"示范"策略） |
| **学习智能体** | $\pi_\theta$ | $\pi_\theta$（学习目标不变） |
| **数据复用** | 只能更新一次 | 可反复使用，更新多次 |
| **效率** | 低——大量时间花在采样上 | 高——一次采样，多次学习 |
| **代表算法** | 基本策略梯度、REINFORCE、Sarsa | Q-learning、DQN、PPO |

> **根本问题**：同策略算法中，数据采样和参数更新是串行绑定的——更新一次就得重新采样一次，严重浪费采样效率。

---

### 1.2 重要性采样（Importance Sampling）

**动机**：用策略 B 采样的数据来训练策略 A。

**核心公式**：

$$
\boxed{\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[f(x) \cdot \frac{p(x)}{q(x)}\right]} \tag{5.3}
$$

其中 $\dfrac{p(x)}{q(x)}$ 称为**重要性权重（importance weight）**，修正了从 $q$ 采样替代从 $p$ 采样带来的偏差。

**直观理解**：
- 若 $p(x) > q(x)$（$x$ 在 $p$ 下更常见）→ 采样次数偏少 → 乘上 $>1$ 的权重补偿
- 若 $p(x) < q(x)$（$x$ 在 $q$ 下更常见）→ 采样次数偏多 → 乘上 $<1$ 的权重抑制

**使用条件**：$q$ 的支撑集必须覆盖 $p$ 的支撑集（$q(x)=0 \Rightarrow p(x)=0$）。

---

### 1.3 重要性采样的方差陷阱

> **期望相同 ≠ 方差相同**——重要性采样在理论上是无偏的，但方差可能被剧烈放大。

两个方差公式的第一项对比：

$$
\begin{aligned}
\text{从 } p \text{ 采样：} &\quad \mathbb{E}_{x \sim p}\left[f(x)^{2}\right] \\[4pt]
\text{从 } q \text{ 采样+重要性权重：} &\quad \mathbb{E}_{x \sim p}\left[f(x)^{2} \cdot \frac{p(x)}{q(x)}\right]
\end{aligned}
$$

后者多乘了 $\frac{p(x)}{q(x)}$：**若 $\frac{p(x)}{q(x)}$ 在某些区域远大于 1，方差会被剧烈放大**，有限样本下估计极不稳定——甚至可能得到符号都反了的错误结论。

---

### 1.4 异策略策略梯度

将重要性采样应用于策略梯度，用固定示范策略 $\theta'$ 采样，反复训练 $\theta$：

$$
\nabla \bar{R}_{\theta}
= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_{\theta}(\tau)\right] \tag{5.4}
$$

细化到状态-动作级：

$$
\mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{p_{\theta}(s_t, a_t)}{p_{\theta'}(s_t, a_t)} A^{\theta'}(s_t, a_t) \nabla \log p_{\theta}(a_t | s_t)\right] \tag{5.5}
$$

---

### 1.5 重要性权重的简化

**关键假设**：$p_{\theta}(s_t) \approx p_{\theta'}(s_t)$（不同策略下遇到同一状态的概率大致相同）。

$$
\boxed{\frac{p_{\theta}(s_t, a_t)}{p_{\theta'}(s_t, a_t)} \approx \frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)}}
$$

理由：
1. 状态出现概率往往与策略关系不大（如 Atari 游戏画面分布大致相同）
2. $p_{\theta}(s_t)$ 几乎无法计算（尤其连续状态空间）

---

### 1.6 从梯度反推目标函数

利用 $\nabla f(x) = f(x) \nabla \log f(x)$，从梯度形式反推：

$$
\boxed{J^{\theta'}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t)\right]}
$$

这就是**异策略策略梯度的目标函数**。

---

### 1.7 PPO 的核心思路

> 在优化 $J^{\theta'}(\theta)$ 的同时，**显式约束 $\theta$ 与 $\theta'$ 之间的差异**，确保重要性采样始终在"安全区"内运作。

**PPO 目标函数**：

$$
\boxed{J_{\mathrm{PPO}}^{\theta'}(\theta) = \underbrace{J^{\theta'}(\theta)}_{\text{异策略目标}} - \underbrace{\beta \cdot \mathrm{KL}(\theta, \theta')}_{\text{KL 散度约束}}} \tag{5.6}
$$

> **注意**：虽然 PPO 推导涉及重要性采样（异策略技术），但由于 KL 约束强制 $\theta \approx \theta'$，**PPO 在实践中被视为同策略算法**。

---

### 1.8 TRPO：PPO 的前身

$$
\begin{aligned}
J_{\mathrm{TRPO}}^{\theta'}(\theta) &= \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}}\left[\frac{p_{\theta}(a_t | s_t)}{p_{\theta'}(a_t | s_t)} A^{\theta'}(s_t, a_t)\right] \\
\text{s.t. } &\quad \mathrm{KL}(\theta, \theta') < \delta
\end{aligned}
$$

| 对比维度 | TRPO | PPO |
|---------|------|-----|
| **KL 散度的处理** | 硬约束（$\mathrm{KL} < \delta$） | 软惩罚（加入目标函数） |
| **优化方式** | 二阶优化（共轭梯度 + 线搜索） | 一阶梯度上升 |
| **实现难度** | 高 | 低 |
| **实际使用** | 几乎被 PPO 取代 | OpenAI 默认算法 |

---

### 1.9 KL 散度 = 行为距离（非参数距离）

- ❌ **不是**：两组神经网络权重 $\theta$ 和 $\theta'$ 之间的 L2 距离
- ✅ **而是**：给定同一状态 $s$，两个策略输出动作概率分布的 KL 散度，对所有状态取平均：

$$
\mathrm{KL}(\theta, \theta') = \mathbb{E}_{s \sim \pi_{\theta'}}\left[ D_{\mathrm{KL}}\left(\pi_{\theta'}(\cdot|s) \,\|\, \pi_\theta(\cdot|s)\right) \right]
$$

> **为什么用行为距离？** 神经网络参数与输出行为之间是非线性映射——参数变化一点，动作分布可能天翻地覆；参数变化很大，动作分布也可能几乎不变。重要性采样的方差取决于动作概率的比值，而非参数空间的 L2 距离。

---

### 1.10 PPO-Penalty（PPO1）：自适应 KL 惩罚

**算法流程**：
1. 用当前策略 $\theta^k$ 采样，采集大量 $(s_t, a_t)$ 对
2. 估计优势函数 $A^{\theta^k}(s_t, a_t)$
3. **多次更新 $\theta$**，每次最大化：

$$
J_{\mathrm{PPO}}^{\theta^{k}}(\theta) = \sum_{(s_t, a_t)} \frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)} A^{\theta^k}(s_t, a_t) - \beta \cdot \mathrm{KL}(\theta, \theta^k) \tag{5.7}
$$

4. 用新的 $\theta$ 作为 $\theta^{k+1}$，重新采样

**自适应 $\beta$ 调节规则**：

| 条件 | 含义 | 操作 |
|------|------|------|
| $\mathrm{KL}(\theta, \theta^k) > \mathrm{KL}_{\max}$ | KL 散度过大，约束太松 | **增大 $\beta$**（乘以 2） |
| $\mathrm{KL}(\theta, \theta^k) < \mathrm{KL}_{\min}$ | KL 散度过小，约束太紧 | **减小 $\beta$**（除以 2） |
| 其他 | KL 散度在可接受范围内 | 保持 $\beta$ 不变 |

---

### 1.11 PPO-Clip（PPO2）：裁剪技巧

**目标函数**：

$$
\boxed{
J_{\mathrm{PPO2}}^{\theta^{k}}(\theta) \approx \sum_{(s_t, a_t)} \min \left(
\underbrace{\frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)} A^{\theta^k}(s_t, a_t)}_{\text{原始项}},
\ \underbrace{\operatorname{clip}\left(\frac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}, 1-\varepsilon, 1+\varepsilon\right) A^{\theta^k}(s_t, a_t)}_{\text{裁剪项}}
\right)
} \tag{5.8}
$$

**裁剪函数**：

$$
\operatorname{clip}(x, 1-\varepsilon, 1+\varepsilon) = \begin{cases}
1 - \varepsilon, & \text{如果 } x < 1 - \varepsilon \\
1 + \varepsilon, & \text{如果 } x > 1 + \varepsilon \\
x, & \text{否则}
\end{cases}
$$

**分情况分析**：

| 情况 | 效果 |
|------|------|
| **$A > 0$**（好动作，应增大概率） | 当比值 $r > 1+\varepsilon$ 时梯度被截断——**不要涨太多** |
| **$A < 0$**（坏动作，应减小概率） | 当比值 $r < 1-\varepsilon$ 时梯度被截断——**不要降太多** |

> **核心直觉**：PPO-Clip 在"好动作更可能"和"坏动作更不可能"两个方向上都设置了更新上限，确保新旧策略的动作概率比值始终在 $[1-\varepsilon, 1+\varepsilon]$ 的"信任区域"内。

---

### 1.12 PPO-Penalty vs PPO-Clip 对比

| 维度 | PPO-Penalty (PPO1) | PPO-Clip (PPO2) |
|------|-------------------|-----------------|
| **约束方式** | KL 散度作为惩罚项加入目标 | 直接裁剪概率比值 |
| **超参数** | $\beta$（可自适应）+ $\mathrm{KL}_{\max}$/$\mathrm{KL}_{\min}$ | $\varepsilon$（通常 0.1 或 0.2） |
| **实现复杂度** | 需要计算 KL 散度 + 自适应逻辑 | 非常简单，几行代码 |
| **调参难度** | 中等 | 低（$\varepsilon$ 比较鲁棒） |
| **实际流行度** | 较少使用 | **主流选择**，OpenAI 默认实现 |

---

### 1.13 PPO 成功的原因

1. **实现简单**：相比 TRPO，PPO 只需一阶优化
2. **稳定性好**：KL 约束或裁剪机制防止了灾难性的策略崩溃
3. **数据效率高**：相比基本策略梯度（一次采样一次更新），PPO 可以多次复用数据

---

## 二、公式速查表

| 公式 | 编号 | 用途 |
|------|------|------|
| $\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[f(x) \dfrac{p(x)}{q(x)}\right]$ | (5.3) | 重要性采样基本公式 |
| $\nabla \bar{R}_\theta = \mathbb{E}_{\tau \sim p_{\theta'}}\left[\dfrac{p_\theta(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_\theta(\tau)\right]$ | (5.4) | 异策略策略梯度（轨迹级） |
| $\mathbb{E}_{(s_t,a_t)\sim\pi_{\theta'}}\left[\dfrac{p_\theta(s_t,a_t)}{p_{\theta'}(s_t,a_t)} A^{\theta'} \nabla \log p_\theta(a_t\|s_t)\right]$ | (5.5) | 异策略策略梯度（状态-动作级） |
| $\dfrac{p_\theta(s_t,a_t)}{p_{\theta'}(s_t,a_t)} \approx \dfrac{p_\theta(a_t\|s_t)}{p_{\theta'}(a_t\|s_t)}$ | — | 重要性权重简化 |
| $J^{\theta'}(\theta) = \mathbb{E}_{(s_t,a_t)\sim\pi_{\theta'}}\left[\dfrac{p_\theta(a_t\|s_t)}{p_{\theta'}(a_t\|s_t)} A^{\theta'}(s_t,a_t)\right]$ | — | 异策略目标函数 |
| $J_{\mathrm{PPO}}^{\theta'}(\theta) = J^{\theta'}(\theta) - \beta \cdot \mathrm{KL}(\theta, \theta')$ | (5.6) | PPO 目标函数 |
| $J_{\mathrm{PPO}}^{\theta^{k}}(\theta) = \sum \dfrac{p_\theta}{p_{\theta^k}} A^{\theta^k} - \beta \cdot \mathrm{KL}(\theta, \theta^k)$ | (5.7) | PPO-Penalty 目标函数 |
| $J_{\mathrm{PPO2}} = \sum \min\left(rA,\ \operatorname{clip}(r, 1-\varepsilon, 1+\varepsilon)A\right)$ | (5.8) | PPO-Clip 目标函数 |
| $\operatorname{Var}_{x\sim q}\left[f\frac{p}{q}\right] = \mathbb{E}_{x\sim p}\left[f^2\frac{p}{q}\right] - (\mathbb{E}[f])^2$ | — | 重要性采样方差公式 |

> 其中 $r = \dfrac{p_{\theta}(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}$ 为概率比值。

---

## 三、认知路径总结

```
同策略效率低（采样→更新→丢弃）
    ↓
异策略 + 重要性采样（用 q 的数据训练 p）
    ↓
重要性采样的方差陷阱（p 和 q 差距大 → 方差爆炸）
    ↓
PPO 的解决方案：约束新旧策略差异
    ├── PPO-Penalty：自适应 β 调节 KL 惩罚
    └── PPO-Clip：min + clip 限制概率比值 ∈ [1-ε, 1+ε]
```

---

## 四、关键记忆点

1. **重要性采样**：$\mathbb{E}_p[f] = \mathbb{E}_q[f \cdot p/q]$，无偏但方差可能很大
2. **方差根源**：$\frac{p}{q}$ 远大于 1 时方差被剧烈放大
3. **权重简化**：$\frac{p(s,a)}{p'(s,a)} \approx \frac{p(a|s)}{p'(a|s)}$（假设状态分布近似不变）
4. **PPO 核心**：$J - \beta \cdot \mathrm{KL}$，在异策略效率与稳定性之间取得平衡
5. **KL 散度**：衡量的是行为距离（动作概率分布的差异），不是参数距离
6. **PPO-Clip**：用 $\min(rA, \operatorname{clip}(r)A)$ 替代 KL 惩罚，实现更简单
7. **PPO 本质**：虽然用了异策略推导，但 KL 约束使行为策略 ≈ 目标策略，实践中为同策略算法
