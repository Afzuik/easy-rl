# 第 2 章 马尔可夫决策过程

图 2.1 介绍了强化学习里面智能体与环境之间的交互，智能体得到环境的状态后，它会采取动作，并把这个采取的动作返还给环境。环境得到智能体的动作后，它会进入下一个状态，把下一个状态传给智能体。在强化学习中，智能体与环境就是这样进行交互的，这个交互过程可以通过马尔可夫决策过程来表示，所以马尔可夫决策过程是强化学习的基本框架。

<div align=center>
<img width="550" src="../img/ch2/2.1.png"/>
</div>
<div align=center>图 2.1 智能体与环境之间的交互</div>

本章将介绍马尔可夫决策过程。在介绍马尔可夫决策过程之前，我们先介绍它的简化版本：马尔可夫过程（Markov process，MP）以及马尔可夫奖励过程（Markov reward process，MRP）。通过与这两种过程的比较，我们可以更容易理解马尔可夫决策过程。其次，我们会介绍马尔可夫决策过程中的**策略评估（policy evaluation）**，就是当给定决策后，我们怎么去计算它的价值函数。最后，我们会介绍马尔可夫决策过程的控制，具体有**策略迭代（policy iteration）** 和**价值迭代（value iteration）**两种算法。在马尔可夫决策过程中，它的环境是全部可观测的。但是很多时候环境里面有些量是不可观测的，但是这个部分观测的问题也可以转换成马尔可夫决策过程的问题。

---

## 第一部分：马尔可夫过程（MP）—— 只有状态转移

### 2.1 马尔可夫性质

在随机过程中，**马尔可夫性质（Markov property）**是指一个随机过程在给定现在状态及所有过去状态情况下，其未来状态的条件概率分布仅依赖于当前状态。以离散随机过程为例，假设随机变量 $X_0,X_1,\cdots,X_T$构成一个随机过程。这些随机变量的所有可能取值的集合被称为状态空间（state space）。如果 $X_{t+1}$ 对于过去状态的条件概率分布仅是 $X_t$ 的一个函数，则
$$
p\left(X_{t+1}=x_{t+1} \mid X_{0:t}=x_{0: t}\right)=p\left(X_{t+1}=x_{t+1} \mid X_{t}=x_{t}\right)
$$
其中，$X_{0:t}$ 表示变量集合 $X_{0}, X_{1}, \cdots, X_{t}$，$x_{0: t}$ 为在状态空间中的状态序列 $x_{0}, x_{1}, \cdots, x_{t}$。马尔可夫性质也可以描述为给定当前状态时，将来的状态与过去状态是条件独立的。如果某一个过程满足**马尔可夫性质**，那么未来的转移与过去的是独立的，它只取决于现在。马尔可夫性质是所有马尔可夫过程的基础。

### 2.2 马尔可夫链

马尔可夫过程是一组具有马尔可夫性质的随机变量序列 $s_1,\cdots,s_t$，其中下一个时刻的状态$s_{t+1}$只取决于当前状态 $s_t$。我们设状态的历史为 $h_{t}=\left\{s_{1}, s_{2}, s_{3}, \ldots, s_{t}\right\}$（$h_t$ 包含了之前的所有状态），则马尔可夫过程满足条件：
$$
  p\left(s_{t+1} \mid s_{t}\right) =p\left(s_{t+1} \mid h_{t}\right) \tag{2.1}
$$
从当前 $s_t$ 转移到 $s_{t+1}$，它是直接就等于它之前所有的状态转移到 $s_{t+1}$。

离散时间的马尔可夫过程也称为**马尔可夫链（Markov chain）**。马尔可夫链是最简单的马尔可夫过程，其状态是有限的。例如，图 2.2 里面有4个状态，这4个状态在 $s_1,s_2,s_3,s_4$ 之间互相转移。比如从 $s_1$ 开始，$s_1$ 有 0.1 的概率继续存留在 $s_1$ 状态，有 0.2 的概率转移到 $s_2$，有 0.7 的概率转移到 $s_4$ 。如果 $s_4$ 是我们的当前状态，它有 0.3 的概率转移到 $s_2$，有 0.2 的概率转移到 $s_3$，有 0.5 的概率留在当前状态。


<div align=center>
<img width="550" src="../img/ch2/2.2.png"/>
</div>
 <div align=center>图 2.2 马尔可夫链示例</div>

我们可以用**状态转移矩阵（state transition matrix）**$\boldsymbol{P}$ 来描述状态转移 $p\left(s_{t+1}=s^{\prime} \mid s_{t}=s\right)$：
$$
  \boldsymbol{P}=\left(\begin{array}{cccc}
    p\left(s_{1} \mid s_{1}\right) & p\left(s_{2} \mid s_{1}\right) & \ldots & p\left(s_{N} \mid s_{1}\right) \\
    p\left(s_{1} \mid s_{2}\right) & p\left(s_{2} \mid s_{2}\right) & \ldots & p\left(s_{N} \mid s_{2}\right) \\
    \vdots & \vdots & \ddots & \vdots \\
    p\left(s_{1} \mid s_{N}\right) & p\left(s_{2} \mid s_{N}\right) & \ldots & p\left(s_{N} \mid s_{N}\right)
    \end{array}\right)
$$
状态转移矩阵类似于条件概率（conditional probability），它表示当我们知道当前我们在状态 $s_t$ 时，到达下面所有状态的概率。所以它的每一行描述的是从一个节点到达所有其他节点的概率。

### 2.3 马尔可夫过程的例子

图 2.3 所示为一个马尔可夫过程的例子，这里有七个状态。比如从 $s_1$ 开始，它有0.4的概率到 $s_2$ ，有 0.6 的概率留在当前的状态。 $s_2$ 有 0.4 的概率到$s_1$，有 0.4 的概率到 $s_3$ ，另外有 0.2 的概率留在当前状态。所以给定状态转移的马尔可夫链后，我们可以对这个链进行采样，这样就会得到一串轨迹。例如，假设我们从状态 $s_3$ 开始，可以得到3个轨迹：
* $s_3, s_4, s_5, s_6, s_6$；
* $s_3, s_2, s_3, s_2, s_1$；
* $s_3, s_4, s_4, s_5, s_5$。

通过对状态的采样，我们可以生成很多这样的轨迹。



<div align=center>
<img width="550" src="../img/ch2/2.3.png"/>
</div>
<div align=center>图 2.3 马尔可夫过程的例子</div>

---

## 第二部分：马尔可夫奖励过程（MRP）—— 加入奖励

### 2.4 马尔可夫奖励过程的定义

**马尔可夫奖励过程（Markov reward process, MRP）**是马尔可夫链加上奖励函数。在马尔可夫奖励过程中，状态转移矩阵和状态都与马尔可夫链一样，只是多了**奖励函数（reward function）**。奖励函数 $R$ 是一个期望，表示当我们到达某一个状态的时候，可以获得多大的奖励。这里另外定义了折扣因子 $\gamma$ 。如果状态数是有限的，那么 $R$ 可以是一个向量。

### 2.5 回报与价值函数

这里我们进一步定义一些概念。**范围（horizon）** 是指一个回合的长度（每个回合最大的时间步数），它是由有限个步数决定的。
**回报（return）**可以定义为奖励的逐步叠加，假设时刻$t$后的奖励序列为$r_{t+1},r_{t+2},r_{t+3},\cdots$，则回报为
$$
  G_{t}=r_{t+1}+\gamma r_{t+2}+\gamma^{2} r_{t+3}+\gamma^{3} r_{t+4}+\ldots+\gamma^{T-t-1} r_{T}
$$
其中，$T$是最终时刻，$\gamma$ 是折扣因子，越往后得到的奖励，折扣越多。这说明我们更希望得到现有的奖励，对未来的奖励要打折扣。当我们有了回报之后，就可以定义状态的价值了，就是**状态价值函数（state-value function）**。对于马尔可夫奖励过程，状态价值函数被定义成回报的期望，即
$$
\begin{aligned}
    V^{t}(s) &=\mathbb{E}\left[G_{t} \mid s_{t}=s\right] \\
    &=\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^{2} r_{t+3}+\ldots+\gamma^{T-t-1} r_{T} \mid s_{t}=s\right]
\end{aligned}  
$$
其中，$G_t$ 是之前定义的**折扣回报（discounted return）**。我们对$G_t$取了一个期望，期望就是从这个状态开始，我们可能获得多大的价值。所以期望也可以看成未来可能获得奖励的当前价值的表现，就是当我们进入某一个状态后，我们现在有多大的价值。

我们使用折扣因子的原因如下。第一，有些马尔可夫过程是带环的，它并不会终结，我们想避免无穷的奖励。第二，我们并不能建立完美的模拟环境的模型，我们对未来的评估不一定是准确的，我们不一定完全信任模型，因为这种不确定性，所以我们对未来的评估增加一个折扣。我们想把这个不确定性表示出来，希望尽可能快地得到奖励，而不是在未来某一个点得到奖励。第三，如果奖励是有实际价值的，我们可能更希望立刻就得到奖励，而不是后面再得到奖励（现在的钱比以后的钱更有价值）。最后，我们也更想得到即时奖励。有些时候可以把折扣因子设为 0（$\gamma=0$），我们就只关注当前的奖励。我们也可以把折扣因子设为 1（$\gamma=1$），对未来的奖励并没有打折扣，未来获得的奖励与当前获得的奖励是一样的。折扣因子可以作为强化学习智能体的一个超参数（hyperparameter）来进行调整，通过调整折扣因子，我们可以得到不同动作的智能体。

在马尔可夫奖励过程里面，我们如何计算价值呢？如图 2.4 所示，马尔可夫奖励过程依旧是状态转移，其奖励函数可以定义为：智能体进入第一个状态 $s_1$ 的时候会得到 5 的奖励，进入第七个状态 $s_7$ 的时候会得到 10 的奖励，进入其他状态都没有奖励。我们可以用向量来表示奖励函数，即

$$
  \boldsymbol{R}=[5,0,0,0,0,0,10]
$$

 <div align=center>
<img width="550" src="../img/ch2/2.4.png"/>
</div>
 <div align=center>图 2.4 马尔可夫奖励过程的例子</div>



我们对 4 步的回合（$\gamma=0.5$）来采样回报 $G$。


  （1）$s_{4}, s_{5}, s_{6}, s_{7} \text{的回报}: 0+0.5\times 0+0.25 \times 0+ 0.125\times 10=1.25$

  （2）$s_{4}, s_{3}, s_{2}, s_{1} \text{的回报}: 0+0.5 \times 0+0.25\times 0+0.125 \times 5=0.625$

  （3）$s_{4}, s_{5}, s_{6}, s_{6} \text{的回报}: 0+0.5\times 0 +0.25 \times 0+0.125 \times 0=0$


我们现在可以计算每一个轨迹得到的奖励，比如我们对轨迹 $s_4,s_5,s_6,s_7$ 的奖励进行计算，这里折扣因子是 0.5。在 $s_4$ 的时候，奖励为0。下一个状态 $s_5$ 的时候，因为我们已经到了下一步，所以要把 $s_5$ 进行折扣，$s_5$ 的奖励也是0。然后是 $s_6$，奖励也是0，折扣因子应该是0.25。到达 $s_7$ 后，我们获得了一个奖励，但是因为状态 $s_7$ 的奖励是未来才获得的奖励，所以我们要对之进行3次折扣。最终这个轨迹的回报就是 1.25。类似地，我们可以得到其他轨迹的回报。

这里就引出了一个问题，当我们有了一些轨迹的实际回报时，怎么计算它的价值函数呢？比如我们想知道 $s_4$ 的价值，即当我们进入 $s_4$ 后，它的价值到底如何？一个可行的做法就是我们可以生成很多轨迹，然后把轨迹都叠加起来。比如我们可以从 $s_4$ 开始，采样生成很多轨迹，把这些轨迹的回报都计算出来，然后将其取平均值作为我们进入 $s_4$ 的价值。这其实是一种计算价值函数的办法，也就是通过蒙特卡洛（Monte Carlo，MC）采样的方法计算 $s_4$ 的价值。

### 2.6 贝尔曼方程

但是这里我们采取了另外一种计算方法，从价值函数里面推导出**贝尔曼方程（Bellman equation）**：
$$
  V(s)=\underbrace{R(s)}_{\text {即时奖励}}+\underbrace{\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)}_{\text {未来奖励的折扣总和}}
$$
其中，
* $s'$ 可以看成未来的所有状态，
* $p(s'|s)$  是指从当前状态转移到未来状态的概率。
* $V(s')$ 代表的是未来某一个状态的价值。我们从当前状态开始，有一定的概率去到未来的所有状态，所以我们要把 $p\left(s^{\prime} \mid s\right)$ 写上去。我们得到了未来状态后，乘一个 $\gamma$，这样就可以把未来的奖励打折扣。
* $\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)$ 可以看成未来奖励的折扣总和（discounted sum of future reward）。

贝尔曼方程定义了当前状态与未来状态之间的关系。未来奖励的折扣总和加上即时奖励，就组成了贝尔曼方程。

**1.全期望公式**

在推导贝尔曼方程之前，我们先仿照**全期望公式（law of total expectation）**的证明过程来证明：
$$
  \mathbb{E}[V(s_{t+1})|s_t]=\mathbb{E}[\mathbb{E}[G_{t+1}|s_{t+1}]|s_t]=\mathbb{E}[G_{t+1}|s_t]
$$


>全期望公式也被称为叠期望公式（law of iterated expectations，LIE）。
如果 $A_i$ 是样本空间的有限或可数的划分（partition），则全期望公式可定义为
$$
  \mathbb{E}[X]=\sum_{i} \mathbb{E}\left[X \mid A_{i}\right] p\left(A_{i}\right)
$$


证明：
为了符号简洁并且易读，我们去掉下标，令 $s=s_t$，$g'=G_{t+1}$，$s'=s_{t+1}$。我们可以根据条件期望的定义来重写回报的期望为

$$
  \begin{aligned}
    \mathbb{E}\left[G_{t+1} \mid s_{t+1}\right] &=\mathbb{E}\left[g^{\prime} \mid s^{\prime}\right] \\
    &=\sum_{g^{\prime}} g^{\prime}~p\left(g^{\prime} \mid s^{\prime}\right)
    \end{aligned} \tag{2.2}
$$

>如果 $X$ 和 $Y$ 都是离散型随机变量，则条件期望（conditional expectation）$\mathbb{E}[X|Y=y]$ 定义为
$$
  \mathbb{E}[X \mid Y=y]=\sum_{x} x p(X=x \mid Y=y)
$$


令 $s_t=s$，我们对式(2.2)求期望可得
$$
  \begin{aligned}
    \mathbb{E}\left[\mathbb{E}\left[G_{t+1} \mid s_{t+1}\right] \mid s_{t}\right] &=\mathbb{E} \left[\mathbb{E}\left[g^{\prime} \mid s^{\prime}\right] \mid s\right] \\
    &=\mathbb{E} \left[\sum_{g^{\prime}} g^{\prime}~p\left(g^{\prime} \mid s^{\prime}\right)\mid s\right]\\
    &=\sum_{s^{\prime}} \sum_{g^{\prime}} g^{\prime} p\left(g^{\prime} \mid s^{\prime}, s\right) p\left(s^{\prime} \mid s\right) \\
    &=\sum_{s^{\prime}} \sum_{g^{\prime}} \frac{g^{\prime} p\left(g^{\prime} \mid s^{\prime}, s\right) p\left(s^{\prime} \mid s\right) p(s)}{p(s)} \\
    &=\sum_{s^{\prime}} \sum_{g^{\prime}} \frac{g^{\prime} p\left(g^{\prime} \mid s^{\prime}, s\right) p\left(s^{\prime}, s\right)}{p(s)} \\
    &=\sum_{s^{\prime}} \sum_{g^{\prime}} \frac{g^{\prime} p\left(g^{\prime}, s^{\prime}, s\right)}{p(s)} \\
    &=\sum_{s^{\prime}} \sum_{g^{\prime}} g^{\prime} p\left(g^{\prime}, s^{\prime} \mid s\right) \\
    &=\sum_{g^{\prime}} \sum_{s^{\prime}} g^{\prime} p\left(g^{\prime}, s^{\prime} \mid s\right) \\
    &=\sum_{g^{\prime}} g^{\prime} p\left(g^{\prime} \mid s\right) \\
    &=\mathbb{E}\left[g^{\prime} \mid s\right]=\mathbb{E}\left[G_{t+1} \mid s_{t}\right]
    \end{aligned}    
$$

**2.贝尔曼方程推导**

贝尔曼方程的推导过程如下：

$$
  \begin{aligned}
    V(s)&=\mathbb{E}\left[G_{t} \mid s_{t}=s\right]\\
    &=\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^{2} r_{t+3}+\ldots \mid s_{t}=s\right]  \\
    &=\mathbb{E}\left[r_{t+1}|s_t=s\right] +\gamma \mathbb{E}\left[r_{t+2}+\gamma r_{t+3}+\gamma^{2} r_{t+4}+\ldots \mid s_{t}=s\right]\\
    &=R(s)+\gamma \mathbb{E}[G_{t+1}|s_t=s] \\
    &=R(s)+\gamma \mathbb{E}[V(s_{t+1})|s_t=s]\\
    &=R(s)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)
    \end{aligned}  
$$


>贝尔曼方程就是当前状态与未来状态的迭代关系，表示当前状态的价值函数可以通过下个状态的价值函数来计算。贝尔曼方程因其提出者、动态规划创始人理查德 $\cdot$ 贝尔曼（Richard Bellman）而得名 ，也叫作"动态规划方程"。



贝尔曼方程定义了状态之间的迭代关系，即
$$
  V(s)=R(s)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)
$$

假设有一个马尔可夫链如图 2.5a 所示，贝尔曼方程描述的就是当前状态到未来状态的一个转移。如图 2.5b 所示，假设我们当前在 $s_1$， 那么它只可能去到3个未来的状态：有 0.1 的概率留在它当前位置，有 0.2 的概率去到 $s_2$ 状态，有 0.7 的概率去到 $s_4$ 状态。所以我们把状态转移概率乘它未来的状态的价值，再加上它的即时奖励（immediate reward），就会得到它当前状态的价值。贝尔曼方程定义的就是当前状态与未来状态的迭代关系。


<div align=center>
<img width="550" src="../img/ch2/2.5.png"/>
</div>
<div align=center>图 2.5 状态转移</div>

我们可以把贝尔曼方程写成矩阵的形式：
$$
  \left(\begin{array}{c}
    V\left(s_{1}\right) \\
    V\left(s_{2}\right) \\
    \vdots \\
    V\left(s_{N}\right)
    \end{array}\right)=\left(\begin{array}{c}
    R\left(s_{1}\right) \\
    R\left(s_{2}\right) \\
    \vdots \\
    R\left(s_{N}\right)
    \end{array}\right)+\gamma\left(\begin{array}{cccc}
    p\left(s_{1} \mid s_{1}\right) & p\left(s_{2} \mid s_{1}\right) & \ldots & p\left(s_{N} \mid s_{1}\right) \\
    p\left(s_{1} \mid s_{2}\right) & p\left(s_{2} \mid s_{2}\right) & \ldots & p\left(s_{N} \mid s_{2}\right) \\
    \vdots & \vdots & \ddots & \vdots \\
    p\left(s_{1} \mid s_{N}\right) & p\left(s_{2} \mid s_{N}\right) & \ldots & p\left(s_{N} \mid s_{N}\right)
    \end{array}\right)\left(\begin{array}{c}
    V\left(s_{1}\right) \\
    V\left(s_{2}\right) \\
    \vdots \\
    V\left(s_{N}\right)
    \end{array}\right) 
$$

我们当前的状态是向量$[V(s_1),V(s_2),\cdots,V(s_N)]^\mathrm{T}$。每一行来看，向量$\boldsymbol{V}$乘状态转移矩阵里面的某一行，再加上它当前可以得到的奖励，就会得到它当前的价值。

当我们把贝尔曼方程写成矩阵形式后，可以直接求解：
$$
  \begin{aligned}
    \boldsymbol{V} &= \boldsymbol{\boldsymbol{R}}+ \gamma \boldsymbol{P}\boldsymbol{V} \\
    \boldsymbol{I}\boldsymbol{V} &= \boldsymbol{R}+ \gamma \boldsymbol{P}\boldsymbol{V} \\
    (\boldsymbol{I}-\gamma \boldsymbol{P})\boldsymbol{V}&=\boldsymbol{R} \\
    \boldsymbol{V}&=(\boldsymbol{I}-\gamma \boldsymbol{P})^{-1}\boldsymbol{R}
    \end{aligned}
$$

我们可以直接得到**解析解（analytic solution）**：
$$
  \boldsymbol{V}=(\boldsymbol{I}-\gamma \boldsymbol{P})^{-1} \boldsymbol{R}
$$

我们可以通过矩阵求逆把 $\boldsymbol{V}$ 的价值直接求出来。但是一个问题是这个矩阵求逆的过程的复杂度是 $O(N^3)$。所以当状态非常多的时候，比如从10个状态到1000个状态，或者到100万个状态，当我们有100万个状态的时候，状态转移矩阵就会是一个100万乘100万的矩阵，对这样一个大矩阵求逆是非常困难的。所以这种通过解析解去求解的方法只适用于很小量的马尔可夫奖励过程。

### 2.7 计算马尔可夫奖励过程价值的迭代方法

我们可以将迭代的方法应用于状态非常多的马尔可夫奖励过程（large MRP），比如：动态规划的方法，蒙特卡洛的方法（通过采样的办法计算它），**时序差分学习（temporal-difference learning，TD learning）**的方法（时序差分学习是动态规划和蒙特卡洛方法的一个结合）。

首先我们用蒙特卡洛方法来计算价值。如图 2.6  所示，蒙特卡洛方法就是当得到一个马尔可夫奖励过程后，我们可以从某个状态开始，把小船放到状态转移矩阵里面，让它"随波逐流"，这样就会产生一个轨迹。产生一个轨迹之后，就会得到一个奖励，那么直接把折扣的奖励即回报 $g$ 算出来。算出来之后将它积累起来，得到回报$G_t$。 当积累了一定数量的轨迹之后，我们直接用 $G_t$ 除以轨迹数量，就会得到某个状态的价值。

<div align=center>
<img width="550" src="../img/ch2/2.6.png"/>
</div>
 <div align=center>图 2.6 计算马尔可夫奖励过程价值的蒙特卡洛方法</div>



比如我们要计算 $s_4$ 状态的价值，可以从 $s_4$ 状态开始，随机产生很多轨迹。把小船放到状态转移矩阵里面，然后它就会"随波逐流"，产生轨迹。每个轨迹都会得到一个回报，我们得到大量的回报，比如100个、1000个回报，然后直接取平均值，就可以等价于现在 $s_4$ 的价值，因为 $s_4$ 的价值 $V(s_4)$  定义了我们未来可能得到多少的奖励。这就是蒙特卡洛采样的方法。

如图 2.7 所示，我们也可以用动态规划的方法，一直迭代贝尔曼方程，直到价值函数收敛，我们就可以得到某个状态的价值。我们通过**自举（bootstrapping）**的方法不停地迭代贝尔曼方程，当最后更新的状态与我们上一个状态的区别并不大的时候，更新就可以停止，我们就可以输出最新的 $V'(s)$ 作为它当前的状态的价值。这里就是把贝尔曼方程变成一个贝尔曼更新（Bellman update），这样就可以得到状态的价值。

动态规划的方法基于后继状态价值的估计来更新现在状态价值的估计（如图 2.7 所示算法中的第 3 行用 $V'$ 来更新 $V$ ）。根据其他估算值来更新估算值的思想，我们称其为自举。



 <div align=center>
<img width="550" src="../img/ch2/2.7.png"/>
</div>
 <div align=center>图 2.7 计算马尔可夫奖励过程价值的动态规划算法</div>



>bootstrap 的本意是"解靴带"。这里使用了德国文学作品《吹牛大王历险记》中解靴带自助（拔靴自助）的典故，因此将其译为"自举"。

### 2.8 马尔可夫奖励过程的示例

如图 2.8 所示，如果我们在马尔可夫链上加上奖励，那么到达每个状态，我们都会获得一个奖励。我们可以设置对应的奖励，比如智能体到达状态 $s_1$时，可以获得 5 的奖励；到达 $s_7$ 的时候，可以得到 10 的奖励；到达其他状态没有任何奖励。
因为这里的状态是有限的，所以我们可以用向量 $\boldsymbol{R}=[5,0,0,0,0,0,10]$ 来表示奖励函数，$\boldsymbol{R}$表示每个状态的奖励大小。

我们通过一个形象的例子来理解马尔可夫奖励过程。我们把一艘纸船放到河流之中，它就会随着水流而流动，它自身是没有动力的。所以我们可以把马尔可夫奖励过程看成一个随波逐流的例子，当我们从某一个点开始的时候，纸船就会随着事先定义好的状态转移进行流动，它到达每个状态后，我们都有可能获得一些奖励。

 <div align=center>
<img width="550" src="../img/ch2/2.8.png"/>
</div>
 <div align=center>图 2.8 马尔可夫奖励过程的例子</div>

---

## 第三部分：马尔可夫决策过程（MDP）—— 加入动作与决策

### 2.9 马尔可夫决策过程的定义与区别

相对于马尔可夫奖励过程，马尔可夫决策过程多了决策（决策是指动作），其他的定义与马尔可夫奖励过程的是类似的。此外，状态转移也多了一个条件，变成了$p\left(s_{t+1}=s^{\prime} \mid s_{t}=s,a_{t}=a\right)$。未来的状态不仅依赖于当前的状态，也依赖于在当前状态智能体采取的动作。马尔可夫决策过程满足条件：
$$
  p\left(s_{t+1} \mid s_{t}, a_{t}\right) =p\left(s_{t+1} \mid h_{t}, a_{t}\right)   
$$

对于奖励函数，它也多了一个当前的动作，变成了 $R\left(s_{t}=s, a_{t}=a\right)=\mathbb{E}\left[r_{t} \mid s_{t}=s, a_{t}=a\right]$。当前的状态以及采取的动作会决定智能体在当前可能得到的奖励多少。

马尔可夫决策过程里面的状态转移与马尔可夫奖励过程以及马尔可夫过程的状态转移的差异如图 2.9 所示。马尔可夫过程/马尔可夫奖励过程的状态转移是直接决定的。比如当前状态是 $s$，那么直接通过转移概率决定下一个状态是什么。但对于马尔可夫决策过程，它的中间多了一层动作 $a$ ，即智能体在当前状态的时候，首先要决定采取某一种动作，这样我们会到达某一个黑色的节点。到达这个黑色的节点后，因为有一定的不确定性，所以当智能体当前状态以及智能体当前采取的动作决定过后，智能体进入未来的状态其实也是一个概率分布。在当前状态与未来状态转移过程中多了一层决策性，这是马尔可夫决策过程与之前的马尔可夫过程/马尔可夫奖励过程很不同的一点。在马尔可夫决策过程中，动作是由智能体决定的，智能体会采取动作来决定未来的状态转移。


<div align=center>
<img width="550" src="../img/ch2/2.9.png"/>
</div>
 <div align=center>图 2.9 马尔可夫决策过程与马尔可夫过程/马尔可夫奖励过程的状态转移的对比</div>

### 2.10 马尔可夫决策过程中的策略

策略定义了在某一个状态应该采取什么样的动作。知道当前状态后，我们可以把当前状态代入策略函数来得到一个概率，即 
$$
  \pi(a \mid s)=p\left(a_{t}=a \mid s_{t}=s\right)
$$
概率代表在所有可能的动作里面怎样采取行动，比如可能有 0.7 的概率往左走，有 0.3 的概率往右走，这是一个概率的表示。另外策略也可能是确定的，它有可能直接输出一个值，或者直接告诉我们当前应该采取什么样的动作，而不是一个动作的概率。假设概率函数是平稳的（stationary），不同时间点，我们采取的动作其实都是在对策略函数进行采样。

已知马尔可夫决策过程和策略 $\pi$，我们可以把马尔可夫决策过程转换成马尔可夫奖励过程。在马尔可夫决策过程里面，状态转移函数 $P(s'|s,a)$ 基于它当前的状态以及它当前的动作。因为我们现在已知策略函数，也就是已知在每一个状态下，可能采取的动作的概率，所以我们就可以直接把动作进行加和，去掉 $a$，这样我们就可以得到对于马尔可夫奖励过程的转移，这里就没有动作，即
$$
  P_{\pi}\left(s^{\prime} \mid s\right)=\sum_{a \in A} \pi(a \mid s) p\left(s^{\prime} \mid s, a\right)
$$

对于奖励函数，我们也可以把动作去掉，这样就会得到类似于马尔可夫奖励过程的奖励函数，即
$$
  r_{\pi}(s)=\sum_{a \in A} \pi(a \mid s) R(s, a)
$$

### 2.11 马尔可夫决策过程中的价值函数

马尔可夫决策过程中的价值函数可定义为
$$
V_{\pi}(s)=\mathbb{E}_{\pi}\left[G_{t} \mid s_{t}=s\right] \tag{2.3}
$$
其中，期望基于我们采取的策略。当策略决定后，我们通过对策略进行采样来得到一个期望，计算出它的价值函数。

这里我们另外引入了一个 **Q 函数（Q-function）**。Q 函数也被称为**动作价值函数（action-value function）**。Q 函数定义的是在某一个状态采取某一个动作，它有可能得到的回报的一个期望，即
$$
Q_{\pi}(s, a)=\mathbb{E}_{\pi}\left[G_{t} \mid s_{t}=s, a_{t}=a\right] \tag{2.4}
$$
这里的期望其实也是基于策略函数的。所以我们需要对策略函数进行一个加和，然后得到它的价值。
对 Q 函数中的动作进行加和，就可以得到价值函数：
$$
V_{\pi}(s)=\sum_{a \in A} \pi(a \mid s) Q_{\pi}(s, a)
\tag{2.5}
$$

此处我们对 Q 函数的贝尔曼方程进行推导：
$$
  \begin{aligned}
    Q(s,a)&=\mathbb{E}\left[G_{t} \mid s_{t}=s,a_{t}=a\right]\\
    &=\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^{2} r_{t+3}+\ldots \mid s_{t}=s,a_{t}=a\right]  \\
    &=\mathbb{E}\left[r_{t+1}|s_{t}=s,a_{t}=a\right] +\gamma \mathbb{E}\left[r_{t+2}+\gamma r_{t+3}+\gamma^{2} r_{t+4}+\ldots \mid s_{t}=s,a_{t}=a\right]\\
    &=R(s,a)+\gamma \mathbb{E}[G_{t+1}|s_{t}=s,a_{t}=a] \\
    &=R(s,a)+\gamma \mathbb{E}[V(s_{t+1})|s_{t}=s,a_{t}=a]\\
    &=R(s,a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s,a\right) V\left(s^{\prime}\right)
    \end{aligned}
$$

### 2.12 贝尔曼期望方程与备份图

我们可以把状态价值函数和 Q 函数拆解成两个部分：即时奖励和后续状态的折扣价值（discounted value of successor state）。
通过对状态价值函数进行分解，我们就可以得到一个类似于之前马尔可夫奖励过程的贝尔曼方程————**贝尔曼期望方程（Bellman expectation equation）**：
$$
  V_{\pi}(s)=\mathbb{E}_{\pi}\left[r_{t+1}+\gamma V_{\pi}\left(s_{t+1}\right) \mid s_{t}=s\right] \tag{2.6} 
$$

对于 Q 函数，我们也可以做类似的分解，得到 Q 函数的贝尔曼期望方程：
$$
  Q_{\pi}(s, a)=\mathbb{E}_{\pi}\left[r_{t+1}+\gamma Q_{\pi}\left(s_{t+1}, a_{t+1}\right) \mid s_{t}=s, a_{t}=a\right] \tag{2.7}
$$
贝尔曼期望方程定义了当前状态与未来状态之间的关联。

> **推导过程**：这两个公式并非凭空而来，而是从价值函数的定义一步步推出的。其推导遵循一条清晰的逻辑链：
> $$
> \text{定义} \;\rightarrow\; \text{代入回报递推}\; G_t=r_{t+1}+\gamma G_{t+1} \;\rightarrow\; \text{线性拆分期望} \;\rightarrow\; \text{全期望公式替换}
> $$
>
> **（1）$V_\pi$ 的贝尔曼期望方程推导**
>
> 状态价值函数 $V_\pi(s)$ 的定义为：
> $$
> V_\pi(s) \doteq \mathbb{E}_\pi\left[G_t \mid s_t = s\right]
> $$
> 其中回报 $G_t$ 具有递推结构 $G_t = r_{t+1} + \gamma G_{t+1}$。将其代入定义：
> $$
> V_\pi(s) = \mathbb{E}_\pi\left[r_{t+1} + \gamma G_{t+1} \mid s_t = s\right]
> $$
> 利用期望的线性性质，拆成两项：
> $$
> V_\pi(s) = \underbrace{\mathbb{E}_\pi\left[r_{t+1} \mid s_t = s\right]}_{\text{即时奖励期望}} \;+\; \gamma \underbrace{\mathbb{E}_\pi\left[G_{t+1} \mid s_t = s\right]}_{\text{后续回报期望}}
> $$
> 第一项：在状态 $s$ 下，智能体按策略 $\pi(a \mid s)$ 选择动作 $a$，环境以概率 $P(s' \mid s, a)$ 转移到 $s'$ 并给出奖励 $r$，因此：
> $$
> \mathbb{E}_\pi\left[r_{t+1} \mid s_t = s\right] = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) R(s, a)
> $$
> 第二项：对 $G_{t+1}$ 使用全期望公式（law of total expectation），先对下一状态 $s_{t+1}$ 取条件期望，再对 $s_{t+1}$ 求平均：
> $$
> \begin{aligned}
> \mathbb{E}_\pi\left[G_{t+1} \mid s_t = s\right]
> &= \mathbb{E}_\pi\Bigl[\,\mathbb{E}_\pi[G_{t+1} \mid s_{t+1}] \;\Big|\; s_t = s\Bigr] \\
> &= \mathbb{E}_\pi\bigl[V_\pi(s_{t+1}) \mid s_t = s\bigr] \\
> &= \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) V_\pi(s')
> \end{aligned}
> $$
> 将两项合并，所有对动作和状态的加权求和统一压缩在一个期望符号 $\mathbb{E}_\pi$ 中，即得到式(2.6)：
> $$
> V_\pi(s) = \mathbb{E}_\pi\left[r_{t+1} + \gamma V_\pi(s_{t+1}) \mid s_t = s\right]
> $$
> 这里 $\mathbb{E}_\pi$ 隐含了两层平均——策略采样（选动作）和环境转移（跳到下一状态）。若将两层平均显式展开，就是后面的式(2.10)。
>
> **（2）$Q_\pi$ 的贝尔曼期望方程推导**
>
> 推导思路完全对称，只需把"给定状态"换成"给定状态+动作"。
>
> Q 函数定义为：
> $$
> Q_\pi(s, a) \doteq \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]
> $$
> 代入 $G_t = r_{t+1} + \gamma G_{t+1}$ 并拆分期望：
> $$
> Q_\pi(s, a) = \mathbb{E}_\pi\left[r_{t+1} \mid s, a\right] + \gamma\,\mathbb{E}_\pi\left[G_{t+1} \mid s, a\right]
> $$
> 第一项：已知 $(s,a)$ 后，环境转移给出的期望即时奖励即为 $R(s, a)$：
> $$
> \mathbb{E}_\pi\left[r_{t+1} \mid s, a\right] = \sum_{s'} P(s' \mid s, a) R(s, a, s') = R(s, a)
> $$
> 第二项：已知 $(s,a)$ 后，转移到 $s'$，再在 $s'$ 处按策略 $\pi$ 选择下一动作 $a'$，其条件期望正是 $Q_\pi(s_{t+1}, a_{t+1})$：
> $$
> \begin{aligned}
> \mathbb{E}_\pi\left[G_{t+1} \mid s_t = s, a_t = a\right]
> &= \mathbb{E}_\pi\Bigl[\,\mathbb{E}_\pi[G_{t+1} \mid s_{t+1}, a_{t+1}] \;\Big|\; s_t = s, a_t = a\Bigr] \\
> &= \mathbb{E}_\pi\bigl[Q_\pi(s_{t+1}, a_{t+1}) \mid s_t = s, a_t = a\bigr] \\
> &= \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s') Q_\pi(s', a')
> \end{aligned}
> $$
> 合并两项，同样将加权求和压缩进期望符号，即得到式(2.7)：
> $$
> Q_\pi(s, a) = \mathbb{E}_\pi\left[r_{t+1} + \gamma Q_\pi(s_{t+1}, a_{t+1}) \mid s_t = s, a_t = a\right]
> $$
> 显式展开则得到后面的式(2.11)。
>
> **一句话总结推导核心**：把回报的递推 $G_t=r_{t+1}+\gamma G_{t+1}$ 代入价值函数定义，用期望线性性拆开，再用全期望公式将 $\mathbb{E}[G_{t+1}]$ 替换为下一时刻的价值函数。紧凑形式 (2.6)/(2.7) 把策略采样和环境转移的双重平均收进 $\mathbb{E}_\pi$，展开形式 (2.10)/(2.11) 则将它们显式写出。

我们进一步进行简单的分解，先给出式(2.8)：

$$
  V_{\pi}(s)=\sum_{a \in A} \pi(a \mid s) Q_{\pi}(s, a) \tag{2.8} 
$$

接着，我们再给出式(2.9)：
$$
  Q_{\pi}(s, a)=R(s,a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_{\pi}\left(s^{\prime}\right) \tag{2.9} 
$$

式(2.8)和式(2.9)代表状态价值函数与 Q 函数之间的关联。

我们把式(2.9)代入式(2.8)可得
$$
  V_{\pi}(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_{\pi}\left(s^{\prime}\right)\right) \tag{2.10} 
$$

式(2.10)代表当前状态的价值与未来状态价值之间的关联。

我们把式(2.8)代入式(2.9)可得
$$
  Q_{\pi}(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) \sum_{a^{\prime} \in A} \pi\left(a^{\prime} \mid s^{\prime}\right) Q_{\pi}\left(s^{\prime}, a^{\prime}\right) \tag{2.11} 
$$

式(2.11)代表当前时刻的 Q 函数与未来时刻的 Q 函数之间的关联。

式(2.10)和式(2.11)是贝尔曼期望方程的另一种形式。

接下来我们深入介绍**备份（backup）**的概念。这是理解所有强化学习算法更新规则的关键。

> **什么是备份（backup）？**
>
> "备份"这个词来自动态规划和树搜索，可以直观理解为：**把未来状态的价值估计"往回传递"，用来修正当前状态的价值估计**。
>
> 想象一棵倒置的树（如图 2.10）：树根是当前状态 $s$，叶子是它的后继状态 $s'$。我们已知叶子上的值（后继状态的价值），通过加权求和把信息沿着树枝"备份"回根部——这个**从叶子到根部、从未来到现在的信息回传过程**，就是备份。
>
> **为什么叫"备份"？** 这个译名强调数据的流向：后继状态（叶子）的数据 $\rightarrow$ 向上传递一层 $\rightarrow$ 汇聚到当前状态（根）。英文 "backup" 本身即"往回传送"之意，而非"备份文件"的备份。
>
> **备份图（backup diagram）的视觉语言**
>
> 备份图用两种节点直观表达了贝尔曼方程的结构：
> - **空心圆圈** $\bigcirc$：代表**状态**（state）
> - **实心黑点** $\bullet$：代表**状态-动作对**（state-action pair）
> - 从根到叶子 → 沿着动作和状态交替展开 → 形成一棵树
> - 从叶子到根 ← 价值信息层层"备份"回来
>
> 如图 2.10 所示，$V_\pi$ 的备份图有两层：先从后继状态 $s'$（空心叶子）备份到动作节点（黑点），再从动作节点备份到当前状态（根部的空心圆圈）。这就是式(2.10)中的两层加和：
> $$
> \underbrace{V_\pi(s)}_{\text{根}} = \sum_a \pi(a|s) \underbrace{\Bigl[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\pi(s') \Bigr]}_{\text{从叶子备份到动作节点}}
> $$
>
> **备份的三种常见形式**
>
> 强化学习中所有算法本质上都在做某种"备份"，区别只是**备份的内容**和**备份的时机**不同：
>
> | 备份类型 | 使用的方程 | 作用 | 出现位置 |
> |---------|----------|------|---------|
> | **贝尔曼期望备份** | 贝尔曼期望方程 | 评估给定策略：将后继状态的 $V_\pi$ 备份到当前状态 | 策略评估（第 2.14 节） |
> | **贝尔曼最优备份** | 贝尔曼最优方程 | 寻找最优策略：取最大后继状态的价值备份到当前状态 | 价值迭代（第 2.19 节） |
> | **采样备份** | 基于实际经验的采样更新 | 免模型情况下，用实际采样轨迹替代期望 | 第三章（TD learning、Q-learning） |
>
> **同步备份 vs 异步备份**
>
> - **同步备份（synchronous backup）**：每次迭代**同时更新所有状态**，用上一轮全部 $V_k$ 计算本轮全部 $V_{k+1}$。好处是理论简洁（保证 $\gamma$-压缩），代价是每轮必须扫描所有状态；
> - **异步备份（asynchronous backup）**：每次只更新**一个或部分状态**，就地（in-place）使用最新的值。好处是灵活高效（不必等待全状态扫描），代价是收敛分析更复杂。
>
> 第 2 章的动态规划方法都默认使用同步备份；第三章起介绍的免模型方法（如 Q-learning）则天然是异步备份——智能体走到哪个状态就更新哪个状态。

我们将与图 2.10 类似的图称为**备份图（backup diagram）**或回溯图，因为它们所示的关系构成了更新或备份操作的基础，而这些操作是强化学习方法的核心。这些操作将价值信息从一个状态（或状态-动作对）的后继状态（或状态-动作对）转移回它。



 <div align=center>
<img width="550" src="../img/ch2/2.10.png"/>
</div>

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;图 2.10 $V_{\pi}$备份图

如式(2.12)所示，这里有两层加和。第一层加和是对叶子节点进行加和，往上备份一层，我们就可以把未来的价值（$s'$ 的价值）备份到黑色的节点。
第二层加和是对动作进行加和，得到黑色节点的价值后，再往上备份一层，就会得到根节点的价值，即当前状态的价值。
$$
V_{\pi}(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_{\pi}\left(s^{\prime}\right)\right) \tag{2.12}
$$

图 2.11 所示为状态价值函数的计算分解，图 2.11b 的计算公式为
$$
  V_{\pi}(s)=\sum_{a \in A} \pi(a \mid s) Q_{\pi}(s, a) \tag{2.13} 
$$

图 2.11b 给出了状态价值函数与 Q 函数之间的关系。图 2.11c 计算 Q 函数为

$$
  Q_{\pi}(s,a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_{\pi}\left(s^{\prime}\right) \tag{2.14} 
$$


我们将式(2.14)代入式(2.13)可得
$$
  V_{\pi}(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_{\pi}\left(s^{\prime}\right)\right)
$$

所以备份图定义了未来下一时刻的状态价值函数与上一时刻的状态价值函数之间的关联。



<div align=center>
<img width="650" src="../img/ch2/2.11.png"/>
</div>
<div align=center>图 2.11 状态价值函数的计算分解</div>



对于 Q 函数，我们也可以进行这样的一个推导。如图 2.12 所示，现在的根节点是 Q 函数的一个节点。Q 函数对应于黑色的节点。下一时刻的 Q 函数对应于叶子节点，有4个黑色的叶子节点。
$$
Q_{\pi}(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) \sum_{a^{\prime} \in A} \pi\left(a^{\prime} \mid s^{\prime}\right) Q_{\pi}\left(s^{\prime}, a^{\prime}\right) \tag{2.15}
$$

如式(2.15)所示，这里也有两层加和。第一层加和先把叶子节点从黑色节点推到空心圆圈节点，进入到空心圆圈结点的状态。
当我们到达某一个状态后，再对空心圆圈节点进行加和，这样就把空心圆圈节点重新推回到当前时刻的 Q 函数。



 <div align=center>
<img width="550" src="../img/ch2/2.12.png"/>
</div>

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;图 2.12 $Q^{\pi}$的备份图


图 2.13c 中，
$$
V_{\pi}\left(s^{\prime}\right)=\sum_{a^{\prime} \in A} \pi\left(a^{\prime} \mid s^{\prime}\right) Q_{\pi}\left(s^{\prime}, a^{\prime}\right) \tag{2.16}
$$

我们将式(2.16)代入式(2.14)可得未来 Q 函数与当前 Q 函数之间的关联，即
$$
  Q_{\pi}(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) \sum_{a^{\prime} \in A} \pi\left(a^{\prime} \mid s^{\prime}\right) Q_{\pi}\left(s^{\prime}, a^{\prime}\right)
$$



 <div align=center>
<img width="650" src="../img/ch2/q_function_backup.png"/>
</div>
 <div align=center>图 2.13 Q函数的计算分解</div>

---

## 第四部分：预测（Prediction）—— 给定策略，求价值函数

### 2.13 预测问题与策略评估

预测（prediction）和控制（control）是马尔可夫决策过程里面的核心问题。预测（评估一个给定的策略）的输入是马尔可夫决策过程 $<S,A,P,R,\gamma>$ 和策略 $\pi$，输出是价值函数 $V_{\pi}$。预测是指给定一个马尔可夫决策过程以及一个策略 $\pi$ ，计算它的价值函数，也就是计算每个状态的价值。

已知马尔可夫决策过程以及要采取的策略 $\pi$ ，计算价值函数 $V_{\pi}(s)$ 的过程就是**策略评估**。策略评估在有些地方也被称为**（价值）预测[（value）prediction）]**，也就是预测我们当前采取的策略最终会产生多少价值。如图 2.14a 所示，对于马尔可夫决策过程，我们其实可以把它想象成一个摆渡的人在船上，她可以控制船的移动，避免船随波逐流。因为在每一个时刻，摆渡的人采取的动作会决定船的方向。如图 2.14b 所示，对于马尔可夫奖励过程与马尔可夫过程，纸的小船会随波逐流，然后产生轨迹。马尔可夫决策过程的不同之处在于有一个智能体控制船，这样我们就可以尽可能多地获得奖励。


<div align=center>
<img width="550" src="../img/ch2/2.14.png"/>
</div>
 <div align=center>图 2.14 马尔可夫决策过程与马尔可夫过程/马尔可夫奖励过程的区别</div>

我们再看一下策略评估的例子，探究怎么在决策过程中计算每一个状态的价值。如图 2.15 所示，假设环境里面有两种动作：往左走和往右走。现在的奖励函数应该是关于动作和状态两个变量的函数。但这里规定，不管智能体采取什么动作，只要到达状态 $s_1$，就有 5 的奖励；只要到达状态 $s_7$ ，就有 10 的奖励，到达其他状态没有奖励。我们可以将奖励函数表示为 $\boldsymbol{R}=[5,0,0,0,0,0,10]$。假设智能体现在采取一个策略：不管在任何状态，智能体采取的动作都是往左走，即采取的是确定性策略 $\pi(s)=\text{左}$。假设价值折扣因子$\gamma=0$，那么对于确定性策略，最后估算出的价值函数是一致的，即 $\boldsymbol{V}_{\pi}=[5,0,0,0,0,0,10]$。


<div align=center>
<img width="550" src="../img/ch2/2.29.png"/>
</div>
<div align=center>图 2.15 策略评估示例</div>


我们可以直接通过贝尔曼方程来得到价值函数：
$$
  V^{k}_{\pi}(s)=r(s, \pi(s))+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, \pi(s)\right) V^{k-1}_{\pi}\left(s^{\prime}\right)
$$
其中，$k$ 是迭代次数。我们可以不停地迭代，最后价值函数会收敛。收敛之后，价值函数的值就是每一个状态的价值。

再来看一个例子，如果折扣因子 $\gamma=0.5$，我们可以通过式(2.17)进行迭代：
$$
  V^{t}_{\pi}(s)=\sum_{a} p(\pi(s)=a)\left(r(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V^{t-1}_{\pi}\left(s^{\prime}\right)\right) \tag{2.17}
$$
其中，$t$是迭代次数。然后就可以得到它的状态价值。

最后，例如，我们现在采取随机策略，在每个状态下，有 0.5 的概率往左走，有 0.5 的概率往右走，即 $p(\pi(s)= \text{左})=0.5$，$p(\pi(s)= \text{右})=0.5$，如何求出这个策略下的状态价值呢？我们可以这样做：一开始的时候，我们对$V(s')$进行初始化，不同的 $V(s')$ 都会有一个值；接着，我们将$V(s')$代入贝尔曼期望方程里面进行迭代，就可以算出它的状态价值。

### 2.14 策略评估的迭代算法

策略评估就是给定马尔可夫决策过程和策略，评估我们可以获得多少价值，即对于当前策略，我们可以得到多大的价值。我们可以直接把**贝尔曼期望备份（Bellman expectation backup）** ，变成迭代的过程，反复迭代直到收敛。这个迭代过程可以看作**同步备份（synchronous backup）** 的过程。

>同步备份是指每一次的迭代都会完全更新所有的状态，这对于程序资源的需求特别大。异步备份（asynchronous backup）的思想就是通过某种方式，使得每一次迭代不需要更新所有的状态，因为事实上，很多状态也不需要被更新。


 式(2.18)是指我们可以把贝尔曼期望备份转换成动态规划的迭代。 当我们得到上一时刻的 $V_t$ 的时候，就可以通过递推的关系推出下一时刻的值。 反复迭代，最后$V$的值就是从 $V_1$、$V_2$ 到最后收敛之后的值 $V_{\pi}$。$V_{\pi}$ 就是当前给定的策略 $\pi$ 对应的价值函数。

$$
  V^{t+1}(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V^{t}\left(s^{\prime}\right)\right) \tag{2.18} 
$$

> **推导过程：贝尔曼期望方程 → 迭代公式**
>
> 式(2.18)并不是凭空出现的，它来源于贝尔曼期望方程的展开形式 (2.10)，核心思路可概括为三步：
>
> **第 1 步：出发点是贝尔曼期望方程**
>
> 由第 2.12 节可知，$V_\pi$ 满足不动点方程：
> $$
> V_\pi(s) = \sum_{a \in A} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a) \; V_\pi(s') \right] \tag{2.10}
> $$
> 这个方程左右两边都出现了 $V_\pi$——它是一个**不动点方程**。如果已知 $P$、$R$ 和 $\pi$，那么 $V_\pi$ 就是该方程的唯一解。
>
> **第 2 步：把等号变成赋值号，得到迭代更新规则**
>
> 既然直接求解不动点方程困难（需要解线性方程组，状态多时计算量巨大），策略评估采用逐次逼近（successive approximation）的方法：
> - 先任意初始化 $V^0$（例如全 0）；
> - 第 $t+1$ 轮，把上一轮估计值 $V^t$ 代入右端，算出的新值作为 $V^{t+1}$。
>
> 形式上就是把式(2.10)中右端的 $V_\pi$ 换成 $V^t$，左端换成 $V^{t+1}$：
> $$
> V^{t+1}(s) \;\leftarrow\; \sum_{a \in A} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a) \; V^{t}(s') \right]
> $$
> 此即式(2.18)。对比式(2.10)，唯一的区别是：$V_\pi$（理想真值）被替换为 $V^t$（第 $t$ 轮近似），$V_\pi(s')$ 被替换为 $V^{t}(s')$。
>
> **第 3 步：为什么迭代一定会收敛？——压缩映射原理**
>
> 定义**贝尔曼算子（Bellman operator）**$T^\pi$：
> $$
> (T^\pi V)(s) \;\equiv\; \sum_a \pi(a \mid s) \Bigl[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) \, V(s') \Bigr]
> $$
> 则式(2.10)等价于 $V_\pi = T^\pi V_\pi$，而式(2.18)等价于 $V^{t+1} = T^\pi V^t$。
>
> 可以证明 $T^\pi$ 是一个 **$\gamma$-压缩映射**：对任意两个价值函数 $V_a, V_b$，有
> $$
> \| T^\pi V_a - T^\pi V_b \|_\infty \;\le\; \gamma \,\| V_a - V_b \|_\infty
> $$
> 因为 $\gamma \in [0,1)$，每次应用 $T^\pi$ 都会让与真解的误差缩小至少 $\gamma$ 倍。根据**巴拿赫不动点定理（Banach fixed-point theorem）**，从任意初始 $V^0$ 出发反复应用 $T^\pi$：
> $$
> V^{t+1} = T^\pi V^t
> $$
> 必定收敛到唯一的不动点 $V_\pi$。这就保证了同步备份的收敛性。
>
> **直观理解：信息从奖励源逐步扩散**（如图 2.19、2.20 所示）
> - 第 0 轮：所有 $V^0(s) = 0$；
> - 第 1 轮：与奖励非零状态相邻的状态获得值（此时只有即时奖励 $R$ 起作用，因为 $\gamma V^0 = 0$）；
> - 第 2 轮：值向外扩散一层（邻居的 $V^1$ 已非零，$\gamma V^1$ 开始贡献）；
> - 反复迭代：信息像水波一样从奖励源向外传播；
> - 收敛：所有值不再变化，即 $V^t = V^{t+1} = V_\pi$。
>
> **迭代公式 vs 贝尔曼期望方程**
>
> | | 贝尔曼期望方程 (2.10) | 策略评估迭代 (2.18) |
> |---|---|---|
> | **本质** | 不动点方程（理想等式） | 逐次逼近算法（迭代更新） |
> | **形式** | $V_\pi = T^\pi V_\pi$ | $V^{t+1} \leftarrow T^\pi V^t$ |
> | **$V$ 的含义** | 未知待求解的真值 $V_\pi$ | 第 $t$ 轮近似估计 |
> | **用法** | 定义 $V_\pi$ 的性质 | 实际计算出 $V_\pi$ |
>
> 一句话总结：**把贝尔曼期望方程右端的 $V_\pi$ 替换为当前估计 $V^t$，然后反复执行这个替换过程，利用压缩映射性质逐步逼近真解——这就是策略评估迭代公式的全部奥秘。**

策略评估的核心思想就是把如式(2.18)所示的贝尔曼期望备份反复迭代，然后得到一个收敛的价值函数的值。因为已经给定了策略函数，所以我们可以直接把它简化成一个马尔可夫奖励过程的表达形式，相当于把 $a$ 去掉，即
$$
  V_{t+1}(s)=r_{\pi}(s)+\gamma P_{\pi}\left(s^{\prime} \mid s\right) V_{t}\left(s^{\prime}\right) \tag{2.19}
$$
这样迭代的式子中就只有价值函数与状态转移函数了。通过迭代式(2.19)，我们也可以得到每个状态的价值。因为不管是在马尔可夫奖励过程，还是在马尔可夫决策过程中，价值函数$V$包含的变量都是只与状态有关，其表示智能体进入某一个状态，未来可能得到多大的价值。比如现在的环境是一个小网格世界（small gridworld），智能体的目的是从某一个状态开始行走，然后到达终止状态，它的终止状态就是左上角与右下角（如图 2.18（右）所示的阴影方块）。小网格世界总共有 14 个非终止状态：$1,\cdots,14$。我们把它的每个位置用一个状态来表示。如图 2.18（左）所示，在小网格世界中，智能体的策略函数直接给定了，它在每一个状态都是随机行走，即在每一个状态都是上、下、左、右行走，采取均匀的随机策略（uniform random policy），$\pi(\mathrm{l} \mid .)=\pi(\mathrm{r} \mid .)=\pi(\mathrm{u} \mid .)=\pi(\mathrm{d} \mid .)=0.25$。 它在边界状态的时候，比如在第4号状态的时候往左走，依然留在第4号状态。我们对其加了限制，这个限制就是出边界的动作不会改变状态，相应概率设置为1，如 $p(7\mid7,\mathrm{r})=1$。 
我们给出的奖励函数就是智能体每走一步，就会得到 $-$1 的奖励，也就是到达终止状态之前每走一步获得的奖励都是 $-$1，所以智能体需要尽快地到达终止状态。

给定动作之后状态之间的转移（transition）是确定的，例如$p(2 \mid 6$,u$)=1$，即从第6号状态往上走，它就会直接到达第2号状态。很多时候有些环境是概率性的（probabilistic），比如智能体在第6号状态，它选择往上走的时候，地板可能是滑的，然后它可能滑到第3号状态或者第1号状态，这就是有概率的转移。但我们把环境进行了简化，从6号状态往上走，它就到了第2号状态。因为我们已经知道环境中的每一个概率以及概率转移，所以就可以直接使用式(2.19)进行迭代，这样就会算出每一个状态的价值。





<div align=center>
<img width="550" src="../img/ch2/2.18.png"/>
</div>
 <div align=center>图 2.18 小网格世界环境</div>



我们再来看一个动态的例子，推荐[斯坦福大学的一个网页](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)，这个网页模拟了式(2.18)所示的单步更新的过程中，所有格子的状态价值的变化过程。

如图 2.19a 所示，网格世界里面有很多格子，每个格子都代表一个状态。每个格子里面有一个初始值0。每个格子里还有一个箭头，这个箭头是指智能体在当前状态应该采取什么策略。我们这里采取随机的策略，不管智能体在哪一个状态，它往上、下、左、右的概率都是相同的。比如在某个状态，智能体都有上、下、左、右各 0.25 的概率采取某一个动作，所以它的动作是完全随机的。在这样的环境里面，我们想计算每一个状态的价值。我们也定义了奖励函数，我们可以看到有些格子里面有一个 $R$ 的值，比如有些值是负的。我们可以看到有几个格子里面是 $-$1 的奖励，只有一个 +1 奖励的格子。在网格世界的中间位置，我们可以看到有一个 $R$ 的值是 1。所以每个状态对应一个值，有一些状态没有任何值，它的奖励就为0。

如图 2.19b 所示，我们开始策略评估，策略评估是一个不停迭代的过程。当我们初始化的时候，所有的 $V(s)$ 都是 0。我们现在迭代一次，迭代一次之后，有些状态的值已经产生了变化。比如有些状态的 $R$ 值为 $-$1，迭代一次之后，它就会得到 $-$1 的奖励。对于中间绿色的格子，因为它的奖励为正，所以它是值为 +1 的状态。当迭代第1次的时候，某些状态已经有些值的变化。



<div align=center>
<img width="750" src="../img/ch2/2.19.png"/>
</div>
 <div align=center>图 2.19 网格世界：动态规划示例</div>

如图 2.20a 所示，我们再迭代一次，之前有值的状态的周围状态也开始有值。因为周围状态与之前有值的状态是临近的，所以这就相当于把周围的状态转移过来。如图 2.20b 所示，我们逐步迭代，值是一直在变换的。

<div align=center>
<img width="750" src="../img/ch2/2.20.png"/>
</div>
 <div align=center>图 2.20 网格世界：策略评估过程示例</div>

当我们迭代了很多次之后，有些很远的状态的价值函数已经有值了，而且整个过程是一个呈逐渐扩散的过程，这其实也是策略评估的可视化。当我们每一步进行迭代的时候，远的状态就会得到一些值，值从已经有奖励的状态逐渐扩散。当我们执行很多次迭代之后，各个状态的值会逐渐稳定下来，最后值就会确定不变。收敛之后，每个状态的值就是它的状态价值。

---

## 第五部分：控制（Control）—— 寻找最优策略

> 在第四部分中，我们解决了**预测问题**：给定策略 $\pi$，算出它的价值函数 $V_\pi$。但强化学习的最终目标是找到**最优策略**。第五部分将解决**控制问题**：不再给定策略，而是让算法自己去发现能使累积奖励最大化的最优策略 $\pi^*$。

---

### 2.15 控制问题：目标与定义

**控制问题**的输入和输出分别是：

- **输入**：马尔可夫决策过程 $\langle S, A, P, R, \gamma \rangle$（不包含策略）；
- **输出**：**最优价值函数** $V^*$ 和**最优策略** $\pi^*$。

换句话说，预测是"给定策略，算价值"；控制是"没有策略，找最优策略和最优价值"。两者是递进关系——先学会评估策略（预测），再学会改进策略（控制）。

**预测 vs 控制：一个直观例子**

图 2.16 和图 2.17 用同一个网格世界对比了预测和控制。

- **预测**（图 2.16）：规定策略为均匀随机（上/下/左/右各 0.25），计算在此固定策略下每个状态的价值。结果如图 2.16c。
- **控制**（图 2.17）：不再限制策略，算法自己找出每个状态应该采取的最佳动作。结果如图 2.17b（最优价值）和图 2.17c（最优策略）。

<div align=center>
<img width="550" src="../img/ch2/2.16.png"/>
</div>
<div align=center>图 2.16 网格世界例子：预测</div>

<div align=center>
<img width="550" src="../img/ch2/2.17.png"/>
</div>
<div align=center>图 2.17 网格世界例子：控制</div>

**最优价值函数与最优策略的正式定义**

最优价值函数 $V^*$ 定义为：在所有可能的策略中，使每个状态价值达到最大的那个值：
$$
V^{*}(s)=\max _{\pi} V_{\pi}(s)
$$

对应的最优策略 $\pi^*$ 就是取到这个最大值的策略：
$$
\pi^{*}(s)=\underset{\pi}{\arg \max }~ V_{\pi}(s)
$$

> **注意**：最优策略可能不唯一——多个不同的策略可能达到相同的 $V^*$，但只要某个策略使所有状态的价值都达到最大，它就是最优策略。

**如何从最优 Q 函数提取最优策略？**

一旦我们得到了最优 Q 函数 $Q^*(s,a)$，最优策略就非常简单了：在每个状态 $s$，选 Q 值最大的那个动作即可：
$$
\pi^{*}(a \mid s)=\left\{\begin{array}{ll}
1, &  a=\underset{a \in A}{\arg \max}~ Q^{*}(s, a) \\
0, & \text{其他}
\end{array}\right.
$$

这引出控制问题的核心思路：**想办法找到 $Q^*$（或 $V^*$），然后直接用 arg max 提取 $\pi^*$**。

> **Q：为什么不直接穷举所有策略？**
>
> 如果状态和动作都是有限的，总共有 $|A|^{|S|}$ 种可能的确定性策略——这个数字随状态数指数增长，穷举完全不现实。因此我们需要更高效的算法：**策略迭代**和**价值迭代**。

---

### 2.16 贝尔曼最优方程

策略评估时我们使用贝尔曼期望方程（$V_\pi$ 根据 $\pi$ 求期望）。但在控制问题中，没有给定的 $\pi$，我们需要的是"最优"——因此期望被 **max** 替代。

**从期望到最大化**

回顾贝尔曼期望方程（展开形式）：
$$
V_\pi(s) = \sum_{a} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_\pi(s') \right]
$$

如果是最优策略 $\pi^*$，它不会对所有动作取加权平均，而是**直接选最好的那个动作**。因此把 $\sum_a \pi(a|s)[\cdots]$ 替换为 $\max_a [\cdots]$，就得到了**贝尔曼最优方程**：

$$
\boxed{
V^{*}(s)=\max _{a \in A}\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) V^{*}\left(s^{\prime}\right)\right)
} \tag{2.22}
$$

**推导过程**（从 $V^*$ 和 $Q^*$ 的关系出发）：

由最优价值函数的定义：
$$
V^{*}(s) = \max_a Q^{*}(s, a) \tag{2.20}
$$

而 $Q^*(s,a)$ 仍满足贝尔曼期望方程的形式（因为在 $(s,a)$ 之后仍遵循最优策略）：
$$
Q^{*}(s, a) = R(s, a) + \gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) V^{*}\left(s^{\prime}\right) \tag{2.21}
$$

将式(2.21) 代入式(2.20) 即得式(2.22)：
$$
V^{*}(s) = \max_a \left[ R(s, a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s, a) V^{*}(s^{\prime}) \right]
$$

> **关键点**：max 作用在**整个括号**上，即同时对即时奖励 $R$ 和折扣后的未来价值 $\gamma \sum P V^*$ 取最大——这与贝尔曼期望方程中"对策略取加权平均"形成了鲜明对比。

**Q 函数的贝尔曼最优方程**

将式(2.20) 代入式(2.21) 消去 $V^*$：
$$
\begin{aligned}
Q^{*}(s, a) &= R(s, a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s, a) V^{*}(s^{\prime}) \\
&= R(s, a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s, a) \max_{a^{\prime}} Q^{*}\left(s^{\prime}, a^{\prime}\right)
\end{aligned}
$$

即
$$
\boxed{
Q^{*}(s, a) = R(s, a) + \gamma \sum_{s^{\prime} \in S} P(s^{\prime} \mid s, a) \max_{a^{\prime}} Q^{*}(s^{\prime}, a^{\prime})
}
$$

这个方程是 Q-learning 等算法的理论基础（将在第三章详细介绍）。

**贝尔曼最优方程的特性**

| | 贝尔曼期望方程 | 贝尔曼最优方程 |
|---|---|---|
| **用于** | 预测（评估给定策略） | 控制（寻找最优策略） |
| **操作** | 对动作加权平均 $\sum_a \pi(a \mid s)$ | 取最大值 $\max_a$ |
| **线性性** | 关于 $V$ 是线性的 | 关于 $V$ 是**非线性的**（因为有 max） |
| **求解方式** | 策略评估（迭代贝尔曼期望备份） | 价值迭代（迭代贝尔曼最优备份） |

> 贝尔曼最优方程的非线性（max 操作）意味着不能像 MRP 那样直接求解析解（$V = (I - \gamma P)^{-1}R$），必须通过迭代方法求解。

---

### 2.17 动态规划：求解 MDP 的统一框架

在深入策略迭代和价值迭代之前，先了解它们共同的数学基础——**动态规划（dynamic programming，DP）**。

动态规划适用于满足两个性质的优化问题：

1. **最优子结构（optimal substructure）**：问题可以拆分成子问题，组合子问题的最优解可以得到原问题的最优解；
2. **重叠子问题（overlapping subproblems）**：子问题重复出现，其解可以被缓存和复用。

**MDP 天然适合动态规划**：贝尔曼方程本身就是递归结构——状态 $s$ 的价值取决于后继状态 $s'$ 的价值。子问题（后继状态的价值）的最优解可以被存储和重用。

**DP 在 MDP 中的两个应用方向**

| 问题 | 使用的方程 | 算法 |
|------|----------|------|
| **预测**（给定 $\pi$，求 $V_\pi$） | 贝尔曼期望方程 | 策略评估（迭代贝尔曼期望备份） |
| **控制**（求 $\pi^*$ 和 $V^*$） | 贝尔曼期望方程 + 贪心改进 | **策略迭代** |
| | 贝尔曼最优方程 | **价值迭代** |

> **注意**：动态规划方法要求**环境模型完全已知**（即已知 $P$ 和 $R$），因此属于**有模型**方法，解决的是**规划**问题而非学习问题。

---

### 2.18 策略迭代

**核心思想**：策略迭代交替执行两个步骤，形成一个不断改进的循环：

$$
\text{策略评估（policy evaluation）} \;\rightleftharpoons\; \text{策略改进（policy improvement）}
$$

<div align=center>
<img width="550" src="../img/ch2/2.21.png"/>
</div>
<div align=center>图 2.21 策略迭代：评估与改进交替进行</div>

**步骤 1：策略评估**

给定当前策略 $\pi_i$，用贝尔曼期望备份迭代计算其价值函数 $V_{\pi_i}$（详见第四部分的式(2.18)）。

**步骤 2：策略改进**

利用刚算出的 $V_{\pi_i}$ 计算 Q 函数：
$$
Q_{\pi_i}(s, a) = R(s, a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s, a) V_{\pi_i}(s^{\prime})
$$

然后对每个状态 $s$，**贪心地**选择 Q 值最大的动作作为新策略：
$$
\pi_{i+1}(s) = \underset{a}{\arg \max}~ Q_{\pi_i}(s, a)
$$

**为什么贪心改进不会变差？——策略改进定理**

可以证明，这种贪心更新要么改善策略，要么保持不变——绝不会使策略变差。因为新策略 $\pi_{i+1}$ 在每一步都选了当前估计下最好的动作，所以 $V_{\pi_{i+1}}(s) \ge V_{\pi_i}(s)$ 对所有 $s$ 成立。当改进停止（即 $\pi_{i+1} = \pi_i$）时就达到了最优策略 $\pi^*$。

**Q 表格视角**

如图 2.22，Q 函数可以组织成一个表格：行是动作，列是状态。策略改进就是在每一列中找出最大值所在的行，将对应动作设为该状态的新策略。

<div align=center>
<img width="550" src="../img/ch2/2.46.png"/>
</div>
<div align=center>图 2.22 Q表格：策略改进 = 每列取 arg max</div>

**策略迭代的完整流程**

1. 初始化：随机策略 $\pi_0$，$V(s)=0$
2. 重复直到收敛：
   - **策略评估**：用迭代贝尔曼期望备份，算出当前 $\pi_i$ 下的 $V_{\pi_i}$
   - **策略改进**：对每个 $s$，$\pi_{i+1}(s) = \arg\max_a Q_{\pi_i}(s,a)$
3. 输出 $\pi^*$ 和 $V^*$

图 2.25–2.28 展示了网格世界中策略迭代的每一步：先评估→再改进→再评估→再改进……直到策略不再变化。

<div align=center>
<img width="750" src="../img/ch2/2.25.png"/>
</div>
<div align=center>图 2.25 策略迭代示例：策略评估后执行策略更新</div>

<div align=center>
<img width="750" src="../img/ch2/2.26.png"/>
</div>
<div align=center>图 2.26 策略迭代示例：继续迭代</div>

<div align=center>
<img width="750" src="../img/ch2/2.27.png"/>
</div>
<div align=center>图 2.27 策略迭代示例：继续迭代至收敛</div>

---

### 2.19 价值迭代

策略迭代的缺点是：每次策略评估都要把贝尔曼期望备份迭代到收敛，计算量大。**价值迭代**用一个更简洁的想法绕过这个问题。

**最优性原理**

贝尔曼（1957）提出的**最优性原理（principle of optimality）**告诉我们：

> 一个策略在状态 $s$ 达到最优价值（即 $V_\pi(s) = V^*(s)$），当且仅当对于所有从 $s$ 可达的后继状态 $s'$，都已经达到了最优价值（即 $V_\pi(s') = V^*(s')$）。

这意味着：如果我能解决所有后继状态的子问题，那当前状态的最优解也就确定了。价值迭代就利用这一点：**从任意初始值开始，反复应用贝尔曼最优方程作为更新规则，逐步将最优价值从终止状态"反向传播"到所有状态**。

**从贝尔曼最优方程到迭代公式**

贝尔曼最优方程 (2.22) 本来是一个理想条件——只有当 $V = V^*$ 时等号才成立。但我们可以把它变成**更新规则**（与策略评估中将贝尔曼期望方程变为迭代公式的思路完全一致）：

$$
\boxed{
V^{k+1}(s) \leftarrow \max_{a \in A} \left( R(s, a) + \gamma \sum_{s^{\prime} \in S} P(s^{\prime} \mid s, a) V^{k}(s^{\prime}) \right)
} \tag{2.22}
$$

反复执行这个更新，$V^k$ 就会收敛到 $V^*$。

**价值迭代算法步骤**

1. **初始化**：对所有 $s$，$V_0(s)=0$；$k=0$
2. **迭代**：重复直到收敛（$k = 0,1,2,\ldots$）：

   对每个状态 $s$，分两步更新：
   $$
   Q_{k+1}(s, a) = R(s, a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s, a) V_{k}(s^{\prime}) \tag{2.23}
   $$
   $$
   V_{k+1}(s) = \max_{a} Q_{k+1}(s, a) \tag{2.24}
   $$

   将式(2.23) 代入式(2.24) 即得到式(2.22)。

3. **提取策略**：收敛后，最优策略为
   $$
   \pi^{*}(s) = \underset{a}{\arg \max} \left[ R(s, a) + \gamma \sum_{s^{\prime}} P(s^{\prime} \mid s, a) V^{*}(s^{\prime}) \right]
   $$

**直观理解：价值的反向传播**

如图 2.23 所示，价值迭代像一个从终点向外扩散的波纹：
- 第 $k=1$ 轮：只有紧邻奖励源的状态获得非零值；
- 第 $k=2$ 轮：值传播到距离为 2 的状态；
- 每次迭代，信息向外传播一层，直到所有状态都"感受"到来自终点的折扣奖励。

<div align=center>
<img width="550" src="../img/ch2/2.52.png"/>
</div>
<div align=center>图 2.23 价值迭代：最短路径问题中的反向传播过程</div>

> **注意**：在价值迭代中，**中间过程的 V 值和策略没有实际意义**——只有收敛后的 $V^*$ 才有意义，此时才能提取出正确的 $\pi^*$。这与策略迭代不同：策略迭代的每一轮都对应一个完整的、有意义的策略。

---

### 2.20 策略迭代 vs 价值迭代

两种算法的核心区别总结如下：

| 维度 | 策略迭代 | 价值迭代 |
|------|---------|---------|
| **核心方程** | 贝尔曼期望方程 + 贪心改进 | 贝尔曼最优方程 |
| **每轮做什么** | 先完整评估当前策略（内层迭代至收敛），再改进策略 | 对所有状态执行一次贝尔曼最优备份 |
| **中间结果** | 每轮都产生一个有意义的策略 | 中间 V 值无实际意义，只有收敛后才有意义 |
| **计算量** | 每轮评估需多次迭代（内层开销大） | 每轮仅一次全状态扫描（轻量） |
| **收敛速度** | 轮数少，但每轮开销大 | 轮数多，但每轮开销小 |
| **类比** | "先彻底评估，再改进" | "边评估边改进" |

**网格世界中的对比**（图 2.24–2.28）

<div align=center>
<img width="550" src="../img/ch2/2.54.png"/>
</div>
<div align=center>图 2.24 网格世界：初始化界面</div>

如图 2.28b 所示，切换到价值迭代后，同样能收敛到与策略迭代一致的最优策略和最优价值。两种算法殊途同归。

<div align=center>
<img width="750" src="../img/ch2/2.28.png"/>
</div>
<div align=center>图 2.28 策略迭代与价值迭代示例：收敛后结果一致</div>

在实际应用中，可以在两者之间取折中——比如策略评估时不迭代到完全收敛，只迭代固定步数（这种方法称为**广义策略迭代，generalized policy iteration**，是后续章节中 actor-critic 等方法的思想基础）。

---

### 2.21 预测与控制的统一总结

表 2.1 将本章所有算法按"问题类型 × 使用的贝尔曼方程"做了统一分类：

<div align=center>表 2.1 动态规划算法</div>
<div align=center>
<img width="550" src="../img/ch2/table_1.png"/>
</div>

- **预测问题**（求 $V_\pi$）→ 使用**贝尔曼期望方程**，迭代执行贝尔曼期望备份；
- **控制问题**（求 $\pi^*$）→ 有两种路径：
  - **策略迭代**：贝尔曼期望方程（评估）+ 贪心最大化（改进）；
  - **价值迭代**：直接迭代**贝尔曼最优方程**。

至此，第 2 章覆盖了 MDP 的完整知识体系：从马尔可夫性质 → MRP → MDP → 策略评估（预测）→ 策略迭代/价值迭代（控制）。这些概念构成了后续所有免模型强化学习算法的理论地基。

## 参考文献

* [强化学习基础 David Silver 笔记](https://zhuanlan.zhihu.com/c_135909947)
* [Reinforcement Learning: An Introduction (second edition)](https://book.douban.com/subject/30323890/)
* [David Silver 强化学习公开课中文讲解及实践](https://zhuanlan.zhihu.com/reinforce)
* [UCL Course on RL(David Silver)](https://www.davidsilver.uk/teaching/)
* [Derivation of Bellman's Equation](https://jmichaux.github.io/_notebook/2018-10-14-bellman/)
* [深入浅出强化学习：原理入门](https://book.douban.com/subject/27624485//)
