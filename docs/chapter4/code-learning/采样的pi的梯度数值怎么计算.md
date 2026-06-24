## 实际计算：就是「加权的交叉熵」，PyTorch 自动求导搞定

---

### 第一步：前向传播，拿到概率分布

假设 CartPole 有 2 个动作（左/右）。状态 $s_t$ 输入策略网络后，输出 logits，再 softmax：

```
logits:  [1.2,  0.3]       ← 网络原始输出
softmax: [0.71, 0.29]      ← 动作概率分布 π_θ(·|s_t)
```

实际采样到的动作是 $a_t = 0$（向左），它的概率是 $\pi_\theta(a_t|s_t) = 0.71$。

---

### 第二步：取对数概率

$$
\log \pi_\theta(a_t|s_t) = \log(0.71) \approx -0.342
$$

---

### 第三步：乘权重 $G_t$，构造"伪损失"

$$
\text{loss}_t = -\,G_t \cdot \log \pi_\theta(a_t|s_t)
$$

负号是因为优化器做**梯度下降**，而我们想要**梯度上升**（最大化 $G_t \log \pi$）。

---

### 第四步：调用 `.backward()`，自动求导完成一切

PyTorch 的 autograd 自动计算：

$$
\nabla_\theta (\text{loss}_t) = -G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

优化器执行 `optimizer.step()` 后：

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta (\text{loss}_t) = \theta + \eta \cdot G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

**梯度上升完成。**

---

### 代码里到底怎么写？

三种等价写法，本质一模一样：

#### 写法 1：手动算 log_prob（我们代码里用的）

```python
logits = policy_net(state)                        # (1, 2)  logits
probs = F.softmax(logits, dim=-1)                 # (1, 2)  概率分布
dist = torch.distributions.Categorical(probs)     # 分类分布
log_prob = dist.log_prob(action)                  # log π(a_t|s_t)  标量
loss = - G_t * log_prob                           # 加权负对数似然
loss.backward()
```

#### 写法 2：直接用 `CrossEntropyLoss`（和分类一模一样）

```python
logits = policy_net(state)                        # (1, 2)
loss = F.cross_entropy(logits, action, reduction='none')  # -log π(a_t|s_t)
weighted_loss = (G_t * loss).sum()
weighted_loss.backward()
```

`CrossEntropyLoss` 内部做的事：`log_softmax(logits) → 取 action 对应的那个 → 取负`，恰好就是 $-\log \pi_\theta(a_t|s_t)$。

#### 写法 3：取 log_softmax 后按索引取值

```python
log_probs = F.log_softmax(logits, dim=-1)         # (1, 2)
log_prob = log_probs[0, action]                   # 取采样动作的那个
loss = - G_t * log_prob
loss.backward()
```

---

### 为什么自动求导能算出 $\nabla_\theta \log \pi_\theta(a_t|s_t)$？

以一个极简网络为例。假设只有一个线性层：$\text{logits} = W s_t + b$。

$$
\pi_\theta(a|s) = \frac{e^{\,z_a}}{\sum_j e^{\,z_j}}, \quad z_j = (W s_t + b)_j
$$

取 log 后对 $W$ 求导（链式法则穿过 softmax → log → 线性层），autograd 自动完成。你不需要手写任何导数——这正是 PyTorch 存在的意义。

---

### 一句话总结

> **$\nabla_\theta \log \pi_\theta(a_t|s_t)$ 不需要你手动算。把 $-\log \pi_\theta(a_t|s_t)$ 当作 loss，乘上 $G_t$ 做权重，调用 `.backward()`，autograd 自动算出 $G_t \cdot \nabla_\theta \log \pi_\theta$。这就是为什么第 4 章反复说「策略梯度 = 加权的交叉熵」——代码和图像分类几乎一样，只是每个样本的 loss 多乘了一个 $G_t$。**