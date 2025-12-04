#import "@preview/orange-book:0.6.1": (
  appendices, book, chapter, corollary, definition, example, exercise, index, make-index, my-bibliography, notation,
  part, problem, proposition, remark, scr, theorem, update-heading-image, vocabulary,
)
#import "@preview/gentle-clues:1.2.0": *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#import "@preview/mannot:0.3.1": *
#import "@preview/cetz:0.4.2"
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "@preview/fletcher:0.5.8" as fletcher
#import "@preview/suiji:0.5.0" as suiji
#show: codly-init.with()

#codly(languages: codly-languages)

#show: book.with(
  title: "强化学习和多模态教程",
  subtitle: "理论与实践",
  date: "Anno scolastico 2023-2024",
  author: "左元",
  main-color: rgb("#F36619"),
  lang: "zh",
  cover: image("./background.svg"),
  image-index: image("./orange1.jpg"),
  list-of-figure-title: "List of Figures",
  list-of-table-title: "List of Tables",
  supplement-chapter: "Chapter",
  supplement-part: "Part",
  part-style: 0,
  lowercase-references: false,
)

#show raw.where(lang: "python"): it => {
  show regex("\$(.*?)\$"): re => {
    eval(re.text, mode: "markup")
  }
  it
}

#set text(font: (
  (name: "JetBrains Mono", covers: "latin-in-cjk"),
  "FZShusong-Z01",
))

#show strong: set text(font: "FZHei-B01")
#show emph: set text(font: "FZKai-Z03")

#show raw: set text(font: (
  (name: "JetBrains Mono", covers: "latin-in-cjk"),
  "FZShusong-Z01",
))

#let colred(x) = text(fill: red, $#x$)

#set list(marker: ([•], [‣]))

#part("强化学习")

#chapter("强化学习简介", image: image("./orange2.jpg"), l: "rl-introduction")

== 基本概念

*强化学习*（reinforcement learning，RL）讨论的问题是智能体（agent）怎么在复杂、不确定的环境（environment）中最大化它能获得的奖励。

#figure(
  fletcher.diagram(
    node-stroke: 1pt,
    fletcher.node((0, 0), [智能体], corner-radius: 2pt, extrude: (0, 3), name: <agent>, fill: gradient.radial(
      blue.lighten(80%),
      blue,
      center: (30%, 20%),
      radius: 80%,
    )),
    fletcher.node((0, 3), [环境], corner-radius: 2pt, extrude: (0, 3), name: <env>, fill: gradient.radial(
      orange.lighten(80%),
      orange,
      center: (30%, 20%),
      radius: 80%,
    )),
    fletcher.edge((0, 0), (2, 0), (2, 3), (0, 3), [动作$A_t$], "-|>"),
    fletcher.edge((0, 3), (-2, 3), [$S_(t+1)$], "-|>"),
    fletcher.edge((-2, 2), (-2, 4), "--"),
    fletcher.edge((-2, 3), (-4, 3), (-4, 0), (0, 0), [状态$S_t$], "-|>"),
    fletcher.edge(<env>, <agent>, [奖励$R_t$], "-|>"),
  ),
  caption: [强化学习交互循环],
) <fig-rl-loop>

#tip(title: [名词])[
  - Agent：智能体，代理，智能代理
  - Environment：环境
  - State：状态
  - Reward：奖励
  - Action：动作，行动
]

强化学习由两部分组成：*智能体*和*环境*。在强化学习过程中，智能体与环境一直在交互。智能体在环境中获取某个状态后，它会利用该状态输出一个动作（action）。然后这个动作会在环境中被执行，环境会根据智能体采取的动作，输出下一个状态以及当前这个动作带来的奖励。智能体的目的就是尽可能多地从环境中获取奖励。

下图就是一个倒立摆环境。

#figure(
  image("rl-figures/倒立摆环境.svg", width: 30%),
  caption: [倒立摆环境],
)

在这个环境中，智能体是推车。

- 动作空间：推车有两个动作，*向左推*和*向右推*。
- 状态：
  - 推车的位置
  - 推车的速度
  - 木杆的角度
  - 木杆的角速度
- 奖励：推车采取向左推或者向右推的动作之后，只要木杆不倒下，奖励就是1。

游戏的结束条件：

- 推车将杆子推出了屏幕（失败）
- 杆子倒下（失败）
- 采取了200次推车的动作，杆子也没倒下（成功）

那么推车应该采取什么样的动作，杆子才能不倒下呢？或者说，推车应该采取什么样的*策略*（policy），杆子才能不倒下呢？因为只有杆子不倒下，我们才能不停的获得奖励。

如果推车每采取一个动作，杆子还能保持平衡，那么我们获得的*即时奖励*是1。

如果到游戏结束，杆子也没有倒下，那么我们将获得最大的奖励200。

而推车每次要采取什么动作，是取决于环境的状态的。也就是推车要根据环境的状态来决定下一步采取什么动作。例如：此时杆子向左偏，那么推车可能需要采取向左推的动作，才能使杆子不倒下。

换句话说，推车要采取的策略是一个*函数*。

#figure(
  cetz.canvas({
    import cetz.draw: *
    content((0, 0), [要采取的动作的概率分布=#highlight(fill: red)[策略函数]\(环境的状态\)])
    line((-3, -5), (3, -5))
    rect((-1.5, -5), (-1, -3.5), fill: blue, stroke: blue)
    rect((0.5, -5), (1, -1.5), fill: green, stroke: green)
    content((-1.5, -5.2), text(size: 5pt)[向左推的概率=0.3])
    content((1, -5.2), text(size: 5pt)[向右推的概率=0.7])
  }),
  caption: [策略是一个函数],
)

而这里*要采取的动作*一般来说是一个概率分布。例如：向左推的概率=0.3，向右推的概率=0.7。

然后从这个分布中进行采样，采样一个动作出来。当然也可以直接选概率最大的动作。

而我们使用什么样的策略呢？当然是能够让我们在游戏结束时，能够获得最大奖励的策略啦。

当然，这里选择策略有很多的讲究。如果我们每次采取的都是概率最大的动作，那么每次可能都会获得最大的即时奖励。如果这样做的话，可能到了一定的时候，不管向左推还是向右推获得的即时奖励可能都是0，也就是不管怎么推，杆子都会倒下（杆子偏离的角度太大，怎么推都没用）。也就是说，我们太看重眼前利益，忽略了长远的利益。这样就是只*利用*而不*探索*。所以有时也需要采取一下概率低的动作，探索一下环境，万一未来获得的奖励更多呢。

所以偶尔也需要采取一下概率小的动作。

#tip[
  所谓*从概率分布中采样*的意思是，例如概率分布是向左推的概率是0.7，向右推的概率是0.3。那么从这个概率分布中采样一个动作出来，有70%的概率采取的动作是向左推。但也有30%的概率向右推。
]

举个例子：如果我们每次都去最熟悉的餐馆吃饭，可能体验都还可以。而如果去不熟悉的餐厅吃饭，可能体验不好，也可能体验超过了之前的餐馆。在舒适区只利用不探索，就会固步自封。冒险可能会受到伤害，但长期来看，可能会得到提升。

策略是智能体的动作模型，它决定了智能体的动作。它其实是一个函数，用于把输入的状态变成动作。

策略的数学符号是 $pi$ ，所以策略也就是 $pi$ 函数。即 $pi(a|s)=p(a_t=a|s_t=s)$ 。输入一个状态 $s$ ，输出一个概率分布。用条件概率来看，就是在条件：状态为 $s$ 的情况下，策略采取动作 $a$ 的概率是多少。这个概率是智能体所有动作的概率，然后对这个概率分布进行采样，可得到智能体将采取的动作。比如可能是有 0.7 的概率往左，0.3 的概率往右，那么通过采样就可以得到智能体将采取的动作。

#figure(
  $
    \
    \
    \
    pi (a|s) = p( markhl(a_t, tag: #<at>)=a|markhl(s_t, tag: #<st>, color: #blue)=s)
    \
    \
    \
    \
    \
    #annot(<at>, pos: top + right, dy: -1.5em, leader-connect: "elbow")[在状态为$s$时，采取动作$a$的概率]
    #annot(<st>, pos: bottom + right, dy: 1.5em, leader-connect: "elbow")[$t$时刻环境的状态为$s$]
  $,
  caption: [策略函数],
)

所以有如下：

$
  pi (a=a_"向左推"|s) = 0.7 \
  pi (a=a_"向右推"|s) = 0.3
$

如果这个 $pi$ 函数是一个神经网络，那么这就是*深度学习 + 强化学习 = 深度强化学习*。当今 AI 界最热门的话题。

所以玩倒立摆游戏的一个*回合*（episode）就是环境的状态为 $S_0$ ，推车采取动作 $A_0$ ，然后获得奖励 $R_0$ ，然后环境的状态转移到了 $S_1$ ，推车接着采取动作 $A_1$ ，然后获得奖励 $R_1$ ，...。下标是时刻，或者时间步。把它们写到一起，就是一个*轨迹*（trajectory）。轨迹用数学符号 $tau$ 表示，读作*掏*。

$
  tau = (S_0,A_0,R_0,S_1,A_1,R_1,S_2,A_2,R_2, dots.c)
$

由于根据环境的状态，采取的动作是从一个概率分布中采样得到的，所以轨迹会有很多很多条。

#figure(
  cetz.canvas({
    import cetz.draw: *
    // let arr = ([向左推], [向右推])
    // let rng = suiji.gen-rng-f(42)
    // let x = suiji.choice(rng, arr)
    for i in (-2, -1, 0, 1, 2) {
      circle((i, 0), radius: (0.1, 0.1), fill: red, stroke: red)
      circle((i, -1), radius: (0.1, 0.1), fill: red, stroke: red)
      circle((i, -2), radius: (0.1, 0.1), fill: red, stroke: red)
      line((i + 0.1, 0), (i + 0.9, 0), mark: (end: "straight"))
      line((i + 0.1, -1), (i + 0.9, -1), mark: (end: "straight"))
      line((i + 0.1, -2), (i + 0.9, -2), mark: (end: "straight"))
    }
    for i in (-2, -1, 0, 1, 2) {
      content((i + 0.4, 0.2), text(size: 4pt)[向左推])
      content((i + 0.4, -0.8), text(size: 4pt)[向右推])
      if (i == -2 or i == 0 or i == 2) {
        content((i + 0.4, -1.8), text(size: 4pt)[向右推])
      } else {
        content((i + 0.4, -1.8), text(size: 4pt)[向左推])
      }
    }
    line((0, -2.5), (0, -3.5), stroke: (dash: "dotted"))
    content((4, 0), text(size: 5pt, fill: red)[游戏失败])
    content((4, -1), text(size: 5pt, fill: red)[游戏失败])
    content((4, -2), text(size: 5pt, fill: green)[游戏成功])
  }),
  caption: [无数条轨迹],
)

== 价值函数（Value Function）

当位于时刻 $t$ 时，环境此时处于状态 $S_t$ ，然后我们根据策略函数开始采取动作，那么未来我们一共能获得多少奖励呢？环境处于状态 $S_t$ ，我们采取的动作是 $A_t$ ，获取的奖励是 $R_t$ ，然后环境的状态从 $S_t$ 转移到了 $S_(t+1)$ ，然后采取动作 $A_(t+1)$ ，然后获得即时奖励 $R_(t+1)$ ，然后环境的状态从 $S_(t+1)$ 转移到了 $S_(t+2)$ ，然后环境会给我们即时奖励 $R_(t+2)$ ，......。

但是未来的奖励不如现在的奖励有吸引力，所以需要*打折*。那么，从 $t$ 时刻起，未来一共获得的奖励叫做*回报*（或者收益，Return）。

$
  G_t = R_t + gamma R_(t+1) + gamma^2 R_(t+2) + dots.c
$

$gamma$ 叫做折扣因子。随着时间的推移，奖励会被 $gamma$ 指数级削弱。这个 $gamma$ 被称为折扣因子（discount rate），其被设定为 $0.0$ 和 $1.0$ 之间的实数。如果折扣因子是 $0.9$，那么有以下式子成立。

$
  G_t = R_t + 0.9 R_(t+1) + 0.81 R_(t+2) + dots.c
$

引入折扣因子主要是为了防止连续性任务的收益变得无穷大。在连续性任务中，如果没有折扣因子（或 $gamma=1$ ），那么收益就会发散到无穷大。因此，设置折扣因子可以防止收益的发散。

折扣因子也使近期的奖励显得更加重要。这解释了人类乃至生物的许多行动原理。例如，你会选择今天拿到 10000 元还是一年后拿到 20000 元？如果折扣因子使未来的回报呈指数级下降，那么眼前的回报就会更有吸引力。

如果我们用倒立摆作为例子，然后我们运行两个时间步。得到下图。

#figure(
  fletcher.diagram(
    node-stroke: 0.1em,
    node-fill: gradient.radial(blue.lighten(80%), blue, center: (30%, 20%), radius: 80%),
    spacing: 2em,
    fletcher.node((-1, 0), text(size: 4pt)[起始状态], radius: 1em),
    fletcher.node((1, 1), text(size: 4pt)[状态$1'$], radius: 1em),
    fletcher.node((1, -1), text(size: 4pt)[状态$1$], radius: 1em),
    fletcher.node((4, -0.5), text(size: 4pt)[状态$2'$], radius: 1em),
    fletcher.node((4, -1.5), text(size: 4pt)[状态$2$], radius: 1em),
    fletcher.node((4, 0.5), text(size: 4pt)[状态$2''$], radius: 1em),
    fletcher.node((4, 1.5), text(size: 4pt)[状态$2'''$], radius: 1em),
    fletcher.edge((-1, 0), (1, -1), "--|>", text(size: 4pt)[向左推0.7], label-side: center),
    fletcher.edge((-1, 0), (1, 1), "--|>", text(size: 4pt)[向右推0.3], label-side: center),

    fletcher.edge((1, -1), (4, -1.5), "--|>", text(size: 4pt)[向左推0.5], label-side: center),
    fletcher.edge((1, -1), (4, -0.5), "--|>", text(size: 4pt)[向右推0.5], label-side: center),

    fletcher.edge((1, 1), (4, 0.5), "--|>", text(size: 4pt)[向左推0.2], label-side: center),
    fletcher.edge((1, 1), (4, 1.5), "--|>", text(size: 4pt)[向右推0.8], label-side: center),
  ),
  caption: [倒立摆环境运行2个时间步],
)

可以看到，一共有 4 条轨迹。每条轨迹都有一个总的回报。而每条轨迹也都有一个产生的概率。那么我们如何评估在环境处于状态 $S_t$ 时，一直采取策略 $pi$ 未来会获得多少回报呢？也就是未来的预期回报（回报的期望值）是多少呢？那就是*状态价值函数*（State Value Function）。

$
  V_pi (s) & = EE_pi [G_t|S_t=s] \
           & = EE_pi [ sum_(k=0)^infinity gamma^k R_(t+k) | S_t=s ], "对所有的" space s in S
$

#tip[
  状态价值函数，衡量的是在环境处于状态 $s$ 时，一直按照策略 $pi$ 来采取动作，最终的预期回报。
]

状态价值函数的另一种重要的表示形式如下：

$
  V_pi (S_t=s) = EE_pi [R_t + gamma V_pi (S_(t+1)=s'|S_t=s)]
$

#tip(title: "贝尔曼期望方程")[
  $
    V_pi (S_t) = EE_pi [R_t + gamma V_pi (S_(t+1)|S_t)]
  $

  直观解释如下：

  - 当前环境对你的价值 = 当前环境现在给你的奖励 $+$ 折扣因子 $times$ 未来环境对你的价值
  - 你对公司的评估 = 现在公司给你的薪资 $+$ 折扣因子 $times$ 你对公司未来的评估
]

== 倒立摆环境编程实践

#codly(
  header: [安装依赖],
  header-cell-args: (align: center),
)
```bash
$ pip install gym==0.25.2
$ pip install numpy==1.26
$ pip install pygame
```

后续基于倒立摆环境的所有依赖如下：

#codly(header: [倒立摆环境需要的所有依赖])
```python
import gym
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
```

然后我们来创建一个倒立摆环境。

#codly(header: [创建倒立摆环境])
```python
# version确认
print(gym.__version__)  # 0.25.2版本
# 创建倒立摆环境
env = gym.make("CartPole-v0")
```

这样就生成了倒立摆的环境。

将推车向右或向左移动，以保持杆子的平衡。倒立摆的结束条件是杆子的平衡被打破（杆子超过一定的角度），或者推车的移动位置超出了某个范围。

继续执行下面的代码

#figure(caption: [打印初始状态和动作空间维度])[
  ```python
  state = env.reset() # 重置环境的状态为$S_0$
  print(state) # 初始状态$S_0$
  action_space = env.action_space # 推车有几个动作？（向左推，向右推）
  print(action_space) # 动作空间的维度=2
  ```] <print-state-action>

@print-state-action 通过```python state = env.reset()```获得了初始状态。观察它的输出，你会发现它是拥有 4 个元素的数组。作为参考，下面依次列出这 4 个元素。

- 推车的位置
- 推车的速度
- 木杆的角度
- 木杆的角速度

另外，我们可以通过 `env.action_space` 获得行动的维度（可采取的行动数）。它的输出是一个名为 `Discrete(2)` 的类实例。这意味着有两个候选行动。具体来说，0 对应的是向左移动推车的行动，1 对应的是向右移动推车的行动。下面实际地采取行动，向前推进一个时间步。

#codly(header: [执行1步动作])
```python
action = 0 # 或者 1
next_state, reward, done, info = env.step(action)
print(next_state)
```

代码通过 `env.step(action)` 采取行动。作为结果 ，我们得到了以下 4 个信息。

执行1步动作后得到的信息：

- 下一个状态`next_state`：$S_1$
- 奖励`reward`：$R_0$
- 是否结束的标志位`done`
- 附加信息`info`

`reward`是标量值（`float`）。这次的任务在保持平衡的时候总是会得到奖励1。`info`包含有助于调试的信息（如环境模型）。但在实现和评估强化学习的算法时，基本不会用到`info`。

我们先来实现一个随机智能体。也就是这个智能体的策略非常简单，无论环境处于什么状态，我们都是从两个动作中随机采样一个动作。代码如下。

```python
state = env.reset() # 重置环境的状态
done = False # 游戏是否结束，初始值为False，也就是未结束
episode_rewards = []  # 每回合奖励$[R_0, R_1, dots.c, R_T]$
total_reward = 0 # 带折扣因子的回报（总奖励）
frames = []         # 保存每一帧
gamma = 0.95 # 折扣因子$gamma$

# done=True时结束
while not done:
    # 渲染画面，并保存帧
    frames.append(env.render(mode="rgb_array"))

    # 随机选择一个动作$A_t$，0：向左推，1：向右推
    action = random.choice([0, 1])

    # 下一个状态$S_(t+1)$，即时奖励$R_t$，是否结束，_
    next_state, reward, done, _ = env.step(action)
    # $[R_0, R_1, dots.c, R_T]$
    episode_rewards.append(reward)

# 逆序计算$G_t = R_t + gamma G_(t+1)$
for r in episode_rewards[::-1]:
    total_reward = r + gamma * total_reward

print("total_reward:", total_reward)
env.close()
```

#tip(title: [动态规划思想])[
  #tip(title: [回报（总奖励的逆序计算）])[
    - 从时间0开始的回报为$G_0 = R_0 + gamma R_1 + gamma^2R_2 + gamma^3R_3 = R_0 + gamma G_1$
    - 从时间1开始的回报为$G_1 = R_1 + gamma R_2 + gamma^2R_3 = R_1 + gamma G_2$
    - 从时间2开始的回报为$G_2 = R_2 + gamma R_3 = R_2 + gamma G_3$
    - 从时间3开始的回报为$G_3 = R_3$

    所以有：$G_t = R_t + gamma G_(t+1)$

    所以我们要进行逆序计算。
  ]
  #tip(title: [霍纳法则（秦九韶算法）])[
    $
        & a_0 + a_1 x + a_2 x^2 + dots.c + a_n x^n \
      = & a_0 + x(a_1 + x(a_2 + x(a_3 + dots.c + x(a_(n-1) + x a_n) dots.c )))
    $
  ]
  #tip(title: [动态规划])[
    - 前向过程：采样一条轨迹，得到每个时刻的即时奖励。
    - 反向过程：逆序计算回报。
  ]
]

然后我们将创建一个函数`show_animation`来将倒立摆环境的每一帧保存成动画，后续就可以播放了。代码如下：

```python
def show_animation(imgs):
    rc("animation", html="jshtml")
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    frames = []

    text = ax.text(10, 20, "", fontsize=12, color="black")

    for i, img in enumerate(imgs):
        frame = [ax.imshow(img, animated=True)]
        frame.append(ax.text(10, 20, f"Step: {i+1}", animated=True))  # Step数表示
        frames.append(frame)

    ax.axis("off")

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

    # 保存动画
    ani.save("cartpole.mp4", writer="ffmpeg")
    ani.save("cartpole.gif", writer="pillow")

    plt.close(fig)
    return ani
```

然后我们播放动画。

```python
show_animation(frames)
```

由于我们使用的是*随机策略*，所以倒立摆游戏很快就结束了。

== 马尔可夫决策过程

=== 基本概念

马尔可夫决策过程（MDP，Markov Decision Process）。决策过程是智能体通过与环境互动决定其行动的过程。

智能体所处的环境根据其行动而发生改变。在强化学习中，这种情况被称为环境的"状态"（state）。在MDP中，状态的变化取决于智能体的行动，智能体在环境的状态迁移后执行新的行动。

在MDP中，我们需要"时间"的概念。在某一时刻，智能体会采取行动并因此迁移到一个新的状态。此时的时间单位叫作"时间步"。由于时间步是智能体做出决定的间隔时间，因此它的实际单位取决于问题。

智能体要考虑的是将来获得的奖励总和，而不是眼前的奖励。换句话说，智能体的目标是实现奖励总和最大化。

在MDP中，智能体与环境之间会进行互动。要点在于当智能体采取行动时，状态会发生迁移，随之获得的奖励也会相应改变。

参见@fig-rl-loop 。

假设在时刻 $t$ 的状态是 $S_t$ 。基于这个状态 $S_t$ ，智能体执行行动 $A_t$ ，获得奖励 $R_t$ 并迁移到下一个状态 $S_(t+1)$ 。智能体与环境之间的这种实际的互动产生了以下迁移。

$
  S_0,A_0,R_0,S_1,A_1,R_1,S_2,A_2,R_2,dots.c
$

这个时间序列数据从第一个状态 $S_0$ 开始。在状态 $S_0$ ，智能体执行行动 $A_0$ 并获得奖励 $R_0$ 。时刻变为 $1$ ，状态变为 $S_1$ 。接下来，基于状态 $S_1$ ，智能体执行行动 $A_1$ 并获得奖励 $R_1$，然后进入下一个状态 $S_2 dots.c dots.c$ 这个流程不断持续下去。

MDP通过数学式来表示智能体、环境以及二者之间的互动。要做到这一点，需要用数学式来表达以下 3 个要素。

- 状态迁移：状态如何迁移。
- 奖励：如何给予奖励。
- 策略：智能体如何决定行动。

=== 状态转移

假设智能体现在处于状态 $s$ 并执行了行动 $a$ ，那么转移到下一个状态 $s'$ 的概率可以用如下方式表示。

$
  p(s'|s,a)
$

#tip(title: [状态和观察])[
  - 状态（state）：如果能观察到整个环境完整的信息，就叫做状态
  - 观察（observation）：只能观察到环境的部分状态（信息）
]

竖杠 $|$ 的右侧是表示"条件"的概率变量。对于当前问题，条件对应于在状态 $s$ 选择了行动 $a$ 。在给定这两个条件的情况下，转移到 $s'$ 的概率可以表示为 $p(s'|s,a)$ 。像 $p(s'|s,a)$ 这样的概率叫作状态转移概率（state transition probability）。

给定当前状态$s$和采取的动作$a$的情况下，*不一定*会确定性的跳转到某个状态$s'$，所以是状态转移概率。

对于倒立摆环境而言，给定当前状态，采取向左推的动作，一定会确定性的跳转到某个状态。

但对于下棋环境来说，棋盘是状态，当Agent下了一个子儿后，棋盘的状态取决于Agent的对手将棋下在哪里，所以状态转移是不确定的。

$p(s'|s,a)$决定了下一个状态$s'$只取决于当前状态$s$和行动$a$。

$
  p(s_t|s_(t-1),a_(t-1)) = p(s_t|s_(t-1),a_(t-1),s_(t-2),a_(t-2),dots.c,s_0,a_0)
$

换句话说，状态转移不需要过去的信息——此前处于什么状态以及执行了哪些行动。这个特性被称为*马尔可夫性质*（Markov property）。

#tip[
  MDP通过假设马尔可夫性质的存在来模拟状态转移和奖励。引入马尔可夫性质主要是为了使问题更容易解决。如果不假定马尔可夫性质，那么就必须考虑之前的所有状态和行动，而且组合的数量会呈指数级增长。
]

=== 奖励函数

当智能体处于状态 $s$ ，执行行动 $a$，下一个状态是 $s'$ 时，得到的奖励由函数 $r(s,a,s')$ 定义。$r(s,a,s')$ 叫做*奖励函数*。

#tip(title: [倒立摆环境中的奖励函数])[
  倒立摆环境中的奖励是确定性的，只要木杆不倒下，就给奖励1。
]

奖励函数的建模是一个非常困难的问题，例如：我们的机器人Agent，它的任务是将插头正确的插入插座，如果只有正确的插入插头才会给奖励1，其它情况都给奖励0，那么机器人可能一辈子也拿不到奖励了。这叫做*奖励稀疏性问题*。

=== 策略

策略表示智能体如何决定其行动。策略的关键在于它使得智能体仅根据当前状态来决定其行动。之所以说只基于当前状态就足够了，是因为环境的迁移是符合马尔可夫性质的。

环境的状态迁移只以当前状态 $s$ 和行动 $a$ 为条件来决定下一个状态 $s'$ ，而不需要先前的信息。同样，奖励也是基于当前状态 $s$ 、行动 $a$ 和迁移后的状态 $s'$ 来决定的。这意味着关于环境的所有必要信息都在当前状态中。因此，智能体只需基于当前状态即可决定其行动。

#tip[
  MDP的马尔可夫性质可以被看作是对环境而不是对智能体的约束。这意味着为了满足马尔可夫性质，环境需要保持某个"状态"。从智能体的角度来看，在当前状态下有足够的信息来做出最佳选择，所以它可以在此基础上采取行动。
]

智能体的行动是由随机性策略决定的，数学式如下所示。

$
  pi(a|s)
$

$pi(a|s)$ 表示在状态 $s$ 下采取行动 $a$ 的概率。

我们已经成功地用数学式来表示状态迁移、奖励函数和策略。接下来让我们使用这三者来定义MDP的目标。

=== MDP的目标

到目前为止，我们已经用数学式描述了环境和智能体的行动。简要回顾一下，智能体根据策略 $pi(a|s)$ 采取行动。首先，它根据它的行动和状态迁移概率 $p(s'|s,a)$ 迁移到下一个状态。然后，它根据奖励函数 $r(s,a,s')$ 获得奖励。在这个框架内，MDP的目标是找到*最优策略*（optimal policy）。最优策略是使收益最大化的策略。

=== 收益

为了定义收益，我们思考这样一个场景：设时刻为 $t$ ，状态为 $S_t$ （其中 $t$ 是任意值）。然后，智能体根据策略 $pi$ 执行行动 $A_t$ ，获得奖励 $R_t$ ，之后迁移到新状态 $S_(t+1)$，这个过程不断重复进行。在这种情况下，收益 $G_t$ 的定义如下所示。

$
  G_t = R_t + gamma R_(t+1) + gamma^2 R_(t+2) + dots.c
$

$gamma$为折扣因子。

=== 状态价值函数

我们已经重新定义了"收益"。智能体的目标是使这种收益最大化。这里有一点需要注意，那就是智能体和环境的行动可能是"随机性"的。智能体可能随机地决定行动，状态也可能随机迁移。在这种情况下，获得的收益将呈现随机的特点。即使从相同的状态开始，不同回合的收益也随机变化。例如，某个回合的收益为$10.4$，另一个回合的收益为$8.7$。

为了处理这种随机行动，需要使用期望值或"收益的期望值"作为衡量标准。收益的期望值的数学式如下所示。

$
  v_pi (s) = EE[G_t|S_t=s,pi]
$ <vpis1>

我们指定的条件是状态 $S_t$ 为 $s$ 智能体的策略为 $pi$ （其中时刻 $t$ 是任意值）。在这些条件下，智能体获得的收益的期望值为上面的公式。在这里，我们用特殊符号 $v_pi (s)$ 来表示收益的期望值。就是*状态价值函数*（state-value function）。

在公式的右侧，智能体的策略$pi$被作为条件给出。这是因为如果策略$pi$发生变化，那么智能体获得的奖励也会发生变化，而这些奖励的总和，即收益也会发生变化。为了明确这一点，状态价值函数通常会写作$v_pi (s)$，将$pi$写在$v$的右下角。此外，上面的公式也可以写成下面的形式。

$
  v_pi (s) = EE_pi [G_t|S_t=s]
$ <vpis2>

在上面的式子中，记载 $pi$ 的位置是 $EE_pi$ 。与@vpis1 一样，这么写的含义是策略 $pi$ 是作为条件给出的。从现在起，本教程将采取式@vpis2 的风格来书写式子。

在强化学习中，我们的目标是获得最优策略。

最优策略的状态价值函数叫作*最优状态价值函数*（optimal state-value function）。可以使用 $v_*$ 来表示最优状态价值函数。

== 贝尔曼方程

贝尔曼方程是在MDP中成立的最重要的方程，为许多强化学习算法提供了重要基础。

这里使用骰子作为例子。我们使用的骰子是理想的六面体，每一面的数字出现的概率都是$1/6$。在数学式中，我们用随机变量$x$表示掷骰子出现的数字，$x$是$1$和$6$之间的整数。那么每个数字出现的概率就是$1/6$。 我们用$p(x)=1/6$表示骰子的数字出现的概率。现在来计算掷骰子的期望值。计算式如下所示。

$
  EE[x] & = 1 times 1/6 + 2 times 1/6 + 3 times 1/6 + 4 times 1/6 + 5 times 1/6 + 6 times 1/6 \
        & = 3.5
$

如上所述，期望值是在所有情况下"出现的数字"和"概率"相乘并加在一起得到的和。顺带一提，如果使用$sum$符号，那么期望值的数学式如下所示。

#definition(name: [期望的定义以及相关性质])[
  $
       EE[x] & = sum_x x p(x) \
    EE[f(x)] & = sum_x f(x)p(x) \
       EE[X] & = EE[EE[X|Y]]    && "（全期望公式）" \
     EE[X|Y] & = EE[EE[X|Z]|Y]  && "（条件期望的迭代公式）"
  $
]

另外，$x$和$y$同时发生的概率（这叫作"联合概率"）如所示。

$
  p(x,y)=p(x)p(y|x)=p(y)p(x|y)
$

假设奖励用 $r(x,y)$ 来表示，那么奖励的期望值如下

$
  EE[r(x,y)] & = sum_x sum_y p(x,y)r(x,y) \
             & = sum_x sum_y p(x)p(y|x)r(x,y)
$

回顾收益（回报）的定义。

$
  G_t = R_t + gamma R_(t+1) + gamma^2 R_(t+2) + dots.c
$

那么有以下公式

$
  G_(t+1) = R_(t+1) + gamma R_(t+2) + gamma^2 R_(t+3) + dots.c
$

所以有以下递推公式：

$
  G_t & = R_t+gamma R_(t+1)+gamma^2R_(t+2)+dots.c \
      & = R_t + gamma(R_(t+1)+gamma R_(t+2)+dots.c) \
      & = R_t+gamma G_(t+1)
$

这种递推关系被用于强化学习的许多理论和算法中。

下面将递推公式代入状态价值函数的定义式中。状态价值函数是收益的期望值，其数学式的定义如下所示。

$
  v_pi(s) = EE_pi [G_t|S_t=s]
$

如上面的公式所示，状态 $s$ 的价值函数被表示为 $v_pi (s)$ 。将递推公式带入上面的式子的 $G_t$ 中，得到下面的式子：

$
  v_pi(s) & = EE_pi [G_t|S_t=s] \
          & = EE_pi [R_t+gamma G_(t+1)|S_t=s] \
          & = EE_pi [R_t|S_t=s] + gamma EE_pi [G_(t+1)|S_t=s]
$

#tip(title: [期望的线性性质])[
  $EE[a X+b Y] = a EE[X] + b EE[Y]$
]

先来推导上面公式中的第一项 $EE_pi [R_t|S_t=s]$ 。

$EE_pi [R_t|S_t=s]$ 的含义是在 $t$ 时刻，环境的状态是 $s$ ，那么在 $t$ 时刻，获得的即时奖励的期望是多少呢？那么需要考虑所有的情况，然后加起来就可以了。

$
  EE_pi [R_t|S_t=s]=sum_a sum_s' pi(a|s)p(s'|s,a)r(s,a,s')
$ <bellman-left>

如上面的数学式所示，将智能体行动的概率$pi(a|s)$、要迁移的状态的概率$p(s'|s,a)$和奖励函数$r(s,a,s')$相乘。对所有候选项都进行上述计算，得到它们的总和。

#tip[
  @bellman-left 中只有$s$是确定的，假设可选择的动作只有两个${a_1,a_2}$，环境状态只有两种${s_1,s_2}$。那么状态转移和概率如下：
  #math.equation(
    $
      s arrow a_1 arrow s_1: pi(a=a_1|s) = 0.2, p(s'=s_1|s,a=a_1) = 0.3, r(s,a=a_1,s'=s_1)=10 \
      s arrow a_1 arrow s_2: pi(a=a_1|s) = 0.2, p(s'=s_2|s,a=a_1) = 0.7, r(s,a=a_1,s'=s_2)=20 \
      s arrow a_2 arrow s_1: pi(a=a_2|s) = 0.8, p(s'=s_1|s,a=a_2) = 0.5, r(s,a=a_2,s'=s_1)=30 \
      s arrow a_2 arrow s_2: pi(a=a_2|s) = 0.8, p(s'=s_2|s,a=a_2) = 0.5, r(s,a=a_2,s'=s_2)=40
    $,
    block: true,
    numbering: none,
  )
  那么即时奖励的期望是：
  #math.equation(
    $
      E & = 0.2 times 0.3 times 10 + 0.2 times 0.7 times 20 + 0.8 times 0.5 times 30 + 0.8 times 0.5 times 40 \
        & = 0.6 + 2.8 + 12.0 + 16.0 \
        & = 31.4
    $,
    block: true,
    numbering: none,
  )
]

第一项推导完毕，接下来推导第二项。

#figure(
  $
    \
    \
    \
    v_pi (s) = markhl(EE_pi [R_t|S_t=s], tag: #<bellleft>) + markhl(gamma EE_pi [G_(t+1)|S_t=s], tag: #<bellright>, color: #blue)
    \
    \
    \
    \
    \
    #annot(<bellleft>, pos: top + right, dy: -1.5em, leader-connect: "elbow")[$sum_(a,s') pi(a|s)p(s'|s,a)r(s,a,s')$]
    #annot(<bellright>, pos: bottom + right, dy: 1.5em, leader-connect: "elbow")[接下来推导这一项]
  $,
  caption: [推导第二项],
)

剩下的项是$gamma EE_pi [G_(t+1)|S_t=s]$。由于$gamma$是常数，因此我们要看的是$EE_pi [G_(t+1)|S_t=s]$。这个式子虽然与状态价值函数的定义式相似，但在$G_(t+1)$的部分有所不同。状态价值函数的式子如下所示，式子中是$G_t$，而不是$G_(t+1)$。

$
  v_pi (s) = EE_pi [G_t|S_t=s]
$

因此，我们首先要将$t+1$代入上面的式子的$t$中。式子变化如下。

$
  v_pi (s) = EE_pi [G_(t+1)|S_(t+1)=s]
$

这就是状态$S_(t+1)=s$时的价值函数。接下来要关注的是$EE_pi [G_(t+1)|S_t=s]$。这是在当前时刻为$t$时，下一个时刻$(t+1)$的收益期望值。解决的关键在于将条件$S_t=s$变为$S_(t+1)=s$的形式。换句话说，就是要进入下一个时刻。

通过观察可以得到

$
  EE_pi [G_(t+1)|S_t=s] & = sum_(a,s') pi(a|s)p(s'|s,a)EE_pi [G_(t+1)|S_(t+1)=s'] \
                        & = sum_(a,s') pi(a|s)p(s'|s,a)v_pi (s')
$

完成第二项的展开以后，汇总一下得到

#theorem(name: [贝尔曼方程])[
  $
    v_pi (s) & = EE_pi [R_t|S_t=s] + gamma EE_pi [G_(t+1)|S_t=s] \
             & = sum_(a,s')pi(a|s)p(s'|s,a)r(s,a,s')+gamma sum_(a,s')pi(a|s)p(s'|s,a)v_pi (s') \
             & = sum_(a,s')pi(a|s)p(s'|s,a){r(s,a,s')+gamma v_pi (s')}
  $
]

上面的式子就是大名鼎鼎的*贝尔曼方程*。贝尔曼方程是表示状态$s$的价值函数和下一个可能的状态$s'$的价值函数之间关系的式子。这个贝尔曼方程对所有状态$s$和所有策略 $pi$ 都成立。

状态价值函数的贝尔曼方程的另一种重要的表示形式：*贝尔曼期望方程*。

#theorem(name: [贝尔曼期望方程])[
  $
    V_pi (S_t) = EE_pi [R_t+gamma V_pi (S_(t+1))|S_t]
  $
]

贝尔曼期望方程的推导过程如下：

$
  V_pi (S_t) & = EE_pi [G_t|S_t] \
             & = EE_pi [sum_(k=0)^infinity gamma^k R_(t+k) | S_t] \
             & = EE_pi [R_t+gamma sum_(k=0)^infinity gamma^k R_(t+k+1) | S_t] \
             & = EE_pi [R_t+gamma G_(t+1)|S_t] \
             & = EE_pi [R_t|S_t]+gamma EE_pi [G_(t+1)|S_t] \
             & = EE_pi [R_t|S_t]+gamma EE_pi [EE_pi [G_(t+1)|S_(t+1)]|S_t] \
             & = EE_pi [R_t|S_t]+gamma EE_pi [V_pi (S_(t+1))|S_t] \
             & = EE_pi [R_t+gamma V_pi (S_(t+1))|S_t] \
$

== 下一步

智能体的目标是使收益最大化。这里有一点需要注意，那就是智能体和环境的行动可能是"随机性"的。智能体可能随机地决定行动，状态也可能随机迁移。在这种情况下，获得的收益将呈现随机的特点。即使从相同的状态开始，不同回合的收益也随机变化。例如，某个回合的收益为$10.4$，另一个回合的收益为$8.7$。

在倒立摆环境中，我们如何选择智能体的"策略"来让木杆多坚持一段时间呢？

接下来我们来学习*策略梯度法*（policy gradient method）。

#part("基于人类反馈的强化学习")

#part("多模态")

#chapter("Vision Transformer", image: image("./orange2.jpg"), l: "multimodal-chap1")

#tip[*ViT*: Vision Transformer]

基于自注意力的Transformer模型由Vaswani等人在2017年的论文《Attention Is All You Need》中首次提出，并已广泛应用于自然语言处理中。Transformer模型是OpenAI用来创建ChatGPT的模型。Transformer不仅适用于文本，还适用于图像，基本上可以处理任何序列数据。2021年，Dosovitsky等人在他们的论文《An Image is Worth 16$times$16 Words: Transformers for Image Recognition at Scale》中引入了将Transformer用于计算机视觉任务（例如图像分类）的想法。在他们的论文中，与卷积网络相比，他们的Vision Transformer模型能够取得出色的结果，并且需要更少的资源来训练。

在本教程中，我们将从头开始构建一个Vision Transformer模型，并在MNIST数据集上进行测试。

#figure(
  image("model_scheme.svg"),
  caption: [Vit架构图],
)

== 导入库和模块

#codly(
  header: [本章所需依赖],
  header-cell-args: (align: center),
)

```python
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import numpy as np
```

== 补丁嵌入（Patch Embedding）

#figure(
  image("补丁示例.png"),
  caption: [补丁的例子],
)

#codly(
  header: [补丁嵌入类],
  header-cell-args: (align: center),
)

```python
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model,    # 模型的维度
        img_size,   # 图片大小
        patch_size, # 补丁大小
        n_channels  # 通道数量
    ):
        super().__init__()

        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.linear_project = nn.Conv2d(
            self.n_channels, # in_channels
            self.d_model, # out_channels
            kernel_size=self.patch_size, # kernel_size
            stride=self.patch_size # stride
        )

    # B: 批次大小
    # C: 通道数量
    # H: 图像高度
    # W: 图像宽度
    # P_col: 补丁的列
    # P_row: 补丁的行
    def forward(self, x):
        x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)
        x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
        return x
```

创建Vision Transformer的第一步是将输入图像拆分为补丁，并创建这些补丁的线性嵌入序列。我们能够通过使用 PyTorch 的 `Conv2d` 方法来实现这一点。

`Conv2d` 方法获取输入图像，将它们拆分为补丁，并提供大小等于 `d_model` 的线性投影。通过将 `kernel_size` 和步幅设置为补丁大小，我们确保补丁大小正确且没有重叠。

```python
self.linear_project = nn.Conv2d(
    self.n_channels,
    self.d_model,
    kernel_size=self.patch_size,
    stride=self.patch_size
)
```

在 `forward` 方法中，我们通过 `linear_project/Conv2d` 方法传递具有形状 `(B, C, H, W)` 的输入，并输出形状 `(B, d_model, P_col, P_row)` 的张量。

```python
def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
```

#figure(
  image("图像拆分为补丁.svg"),
  caption: [将图像转换为补丁：第一步],
)

我们使用展平方法将补丁列和补丁行维度组合成一个补丁维度，从而得到 `(B, d_model, P)` 的形状

```python
x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)
```

#figure(
  image("将补丁转换为一维.svg"),
  caption: [将图像转换为补丁：第二步],
)

最后，我们使用转置方法切换 `d_model` 和补丁维度，得到 `(B, P, d_model)` 的形状。

```python
x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
```

#figure(
  image("一维补丁转置.svg"),
  caption: [将图像转换为补丁：第三步],
)

== 类别对应的 token 和位置编码

#codly(
  header: [位置编码类],
  header-cell-args: (align: center),
)
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        # 类别token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 创建位置编码
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))
        # 将位置编码固定
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # 为批次中的每张图片分配一个类别token
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
        # 将类别token添加到每个图像的补丁嵌入数组的开头
        x = torch.cat((tokens_batch,x), dim=1)
        # 将位置编码添加到嵌入中
        x = x + self.pe
        return x
```

ViT模型使用向补丁嵌入添加可学习的类别token的标准方法来执行分类。

```python
# class token: 类别或者分类对应的token
self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
```

#figure(
  image("位置编码的作用.svg"),
  caption: [位置编码的作用],
)

与LSTM等按顺序接受嵌入的模型不同，Transformer并行的接受嵌入。虽然这提高了速度，但Transformer不知道序列的顺序是什么。这是一个很大的问题，因为改变序列的顺序很可能会改变其含义。上图就是一个例子，它显示更改图像的补丁顺序可以将图像从O更改为更接近X的图像。为了解决这个问题，需要将位置编码添加到嵌入中。每个位置编码对于它所表示的位置都是唯一的，这允许模型识别每个补丁嵌入应该在的位置。为了将位置编码添加到补丁嵌入中，它们必须具有相同的维度，`d_model` 。我们使用下面的公式获得位置编码。

#definition(name: "位置编码")[
  $
      #text[PE] _((#text[pos], 2i)) & = sin (#text[pos] / 10000^((2i) / (d_(#text[model]))) ) \
    #text[PE] _((#text[pos], 2i+1)) & = cos (#text[pos] / 10000^((2i) / (d_(#text[model]))) )
  $
]

```python
# max_seq_length：补丁嵌入的数量，d_model：嵌入维度
pe = torch.zeros(max_seq_length, d_model)

for pos in range(max_seq_length):
    for i in range(d_model):
        if i % 2 == 0:
            pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
        else:
            pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))
# 禁止pe更新，pe保存在了显存中，不会被反向传播更新
self.register_buffer('pe', pe.unsqueeze(0))
```

在 `forward` 方法中，输入是多个图像的一批补丁嵌入。例如，`32x32` 的图像可以分解为 16 个 `8x8` 大小的补丁。在此 `max_seq_length` 中，需要为 `16+1=17` 才能创建足够的位置嵌入，每个补丁一个，分类对应的 token 一个。

因此，我们需要使用 `expand` 函数才能使用 `self.cls_token` 为批处理中的每个图像创建分类对应的 token 。注意力机制会将有关整个序列的信息编码到序列中的每个token中。由于每个token都受到其自身信息的偏见，因此分类对应的token会创建序列中所有token的独立的摘要信息。

```python
def forward(self, x):
    tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
```

然后，使用 `torch.cat` 方法将这些分类token添加到每个补丁嵌入的开头。

```python
x = torch.cat((tokens_batch,x), dim=1)
```

位置编码在输出之前添加。

```python
x = x + self.pe
return x
```

这是添加分类token之前和之后的数据的样子：

#figure(
  image("添加类别token.png"),
  caption: [左图：使用 `Conv2d` 运算分成 `16` 个 `8x8` 块的 `32x32` MNIST 图像。右图：添加位置编码和类别token后的 `16` 个图像补丁，使用随机数据初始化。],
)

请注意，我们已经用 64 个卷积核初始化了 `Conv2d` 运算，每个卷积核中的每个补丁只占用一个像素，以免扭曲图像。

== 注意力头

#codly(
  header: [注意力头],
  header-cell-args: (align: center),
)

```python
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        # 计算Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # $Q K ^ T$
        attention = Q @ K.transpose(-2,-1)

        # $"softmax"( (Q K^T)/sqrt(d_"head") )V$
        attention = attention / (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V

        return attention
```

#figure(
  image("注意力机制.svg"),
  caption: [注意力机制],
)

ViT 使用注意力机制，这是一种通信机制，允许模型专注于图像的重要部分。可以使用下面的公式计算注意力分数。

$
  #text[Attention] (Q,K,V) = #text[softmax] ( (Q K^T) / sqrt(d_k) ) V
$

计算注意力的第一步是获取token的Q、K和V。token的Q是token要查找的内容，K是token包含的内容，V是token之间的互信息。Q、K和V可以通过线性层传递token来计算。

```python
def forward(self, x):
    # 计算Q，K，V
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)
```

我们能够通过计算Q和K的点积来获取序列中token之间的关系。

```python
attention = Q @ K.transpose(-2,-1)
```

我们需要缩放这些值以控制初始化时的方差，以便token能够聚合来自多个其他token的信息。通过将点积除以注意力头大小的平方根来应用缩放。

```python
attention = attention / (self.head_size ** 0.5)
```

然后，我们需要对缩放的点积应用 `softmax` 。

```python
attention = torch.softmax(attention, dim=-1)
```

最后，我们需要得到 `softmax` 和V矩阵之间的点积。这本质上是在相应的token之间传递信息。

```python
attention = attention @ V
return attention
```

== 多头注意力

#codly(
  header: [多头注意力],
)
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads
        self.W_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.head_size) for _ in range(n_heads)
        ])

    def forward(self, x):
        # 拼接多个注意力头
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out
```

多头注意力只是并行运行多个头的自注意力并将它们组合起来。我们可以通过将注意力头添加到模块列表中来做到这一点，

```python
self.heads = nn.ModuleList([
  AttentionHead(d_model, self.head_size)
  for _ in range(n_heads)
])
```

传递输入然后拼接。

```python
def forward(self, x):
    # 拼接多个注意力头
    out = torch.cat([head(x) for head in self.heads], dim=-1)
```

然后，我们需要将输出传递给另一个线性层。

```python
out = self.W_o(out)
return out
```

== Transformer编码器

#codly(
  header: [Transformer编码器],
)
```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 层归一化
        self.ln1 = nn.LayerNorm(d_model)

        # 多头注意力
        self.mha = MultiHeadAttention(d_model, n_heads)

        # 层归一化
        self.ln2 = nn.LayerNorm(d_model)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Linear(d_model*r_mlp, d_model)
        )

    def forward(self, x):
        out = x + self.mha(self.ln1(x)) # 跳跃连接
        out = out + self.mlp(self.ln2(out)) # 跳跃连接
        return out
```

Transformer编码器由两个子层组成：第一个子层执行多头注意力，第二个子层包含MLP。多头注意力子层执行token之间的通信，而MLP子层允许token单独"思考"与它们通信的内容。

层归一化是一种优化技术，可跨其特征独立归一化批处理中的每个输入。对于我们的模型，我们将在每个子层的开头通过层归一化模块传递我们的输入。

```python
self.ln1 = nn.LayerNorm(d_model)
self.ln2 = nn.LayerNorm(d_model)
```

MLP 将由两个线性层组成，中间有一个GELU层。使用GELU代替RELU，因为它没有RELU在零点不可导的限制。

```python
    # 编码器的MLP
    self.mlp = nn.Sequential(
        nn.Linear(width, width*r_mlp),
        nn.GELU(),
        nn.Linear(width*r_mlp, width)
    )
```

在编码器的 `forward` 方法中，输入在执行多头注意力之前通过第一层归一化模块。通过执行多头注意力将原始输入添加到输出中，以创建残差连接。

然后，在输入 MLP 之前，它会通过另一个层归一化模块。通过将 MLP 的输出添加到第一个残差连接的输出，创建另一个残差连接。

残差连接用于通过创建一条路径来帮助防止梯度消失问题，以便梯度不受阻碍地反向传播回原始输入。

```python
def forward(self, x):
    out = x + self.mha(self.ln1(x))
    out = out + self.mlp(self.ln2(out))
    return out
```

== Vision Transformer

#codly(header: [ViT类])
```python
class VisionTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        n_classes,
        img_size,
        patch_size,
        n_channels,
        n_heads,
        n_layers
    ):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0  \
           and img_size[1] % patch_size[1] == 0, \
           "img_size 必须能被 patch_size 整除"
        assert d_model % n_heads == 0, \
           "d_model 必须能被 n_heads 整除"

        self.d_model = d_model # 模型维度，嵌入的维度（宽度）
        self.n_classes = n_classes # 类别的数量
        self.img_size = img_size # 图片大小
        self.patch_size = patch_size # 补丁大小
        self.n_channels = n_channels # 通道数
        self.n_heads = n_heads # 注意力头的数量
        # 补丁的数量 = $(32 times 32)/(4 times 4)$
        self.n_patches = (self.img_size[0] * self.img_size[1]) \
                      // (self.patch_size[0] * self.patch_size[1])
        # 序列的长度 = 1（分类token） + 补丁的数量
        self.max_seq_length = self.n_patches + 1
        # 补丁嵌入
        self.patch_embedding = PatchEmbedding(
            self.d_model,
            self.img_size,
            self.patch_size,
            self.n_channels
        )
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            self.d_model,
            self.max_seq_length
        )
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoder(self.d_model, self.n_heads)
            for _ in range(n_layers)
        ])

        # 用于分类的MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # 将图片转换成补丁的嵌入（embedding）
        x = self.patch_embedding(images)
        # 添加位置编码
        x = self.positional_encoding(x)
        # 编码
        x = self.transformer_encoder(x)
        # 分类的线性层
        x = self.classifier(x[:,0])
        return x
```

在创建我们的 `VisionTransformer` 类时，我们首先需要确保输入图像可以均匀地分成大小为 `patch_size` 的补丁，并且模型的维数可以被注意力头的数量整除。

```python
assert img_size[0] % patch_size[0] == 0  \
   and img_size[1] % patch_size[1] == 0, \
   "img_size 必须能被 patch_size 整除"
assert d_model % n_heads == 0, \
   "d_model 必须能被 n_heads 整除"
```

我们还需要计算位置编码的最大序列长度，该长度等于补丁的数量加1。补丁的数量可以通过将输入图像的高和宽的乘积除以补丁大小的高和宽的乘积来计算。

```python
self.n_patches = (self.img_size[0] * self.img_size[1]) \
              // (self.patch_size[0] * self.patch_size[1])
self.max_seq_length = self.n_patches + 1
```

`VisionTransformer` 还需要能够拥有多个编码器模块。这可以通过将编码器层列表放入顺序包装器中来实现。

```python
self.encoder = nn.Sequential(*[
    TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)
])
```

`VisionTransformer` 模型的最后一部分是 MLP 分类头。它由一个线性层和一个 `softmax` 层组成。

```python
self.classifier = nn.Sequential(
    nn.Linear(self.d_model, self.n_classes),
    nn.Softmax(dim=-1)
)
```

在 `forward` 方法中，输入图像首先经过补丁嵌入层，将图像分割成补丁，并获取这些补丁的线性嵌入序列。然后，它们经过位置编码层，添加分类token和位置编码，最后经过编码器模块。之后，分类token再经过分类MLP，以确定图像的类别。

```python
def forward(self, images):
    x = self.patch_embedding(images)
    x = self.position_embedding(x)
    x = self.encoder(x)
    x = self.classifier(x[:,0])
    return x
```

我们已经完成了模型的构建。现在我们需要对其进行训练和测试。

== 训练参数

```python
d_model = 9 # 嵌入的维度9
n_classes = 10 # 类别数量为10
img_size = (32,32) # 图片大小为32x32
patch_size = (16,16) # 补丁的大小是16x16
n_channels = 1 # 灰度图片通道数量为1
n_heads = 3 # 3个注意力头
n_layers = 3 # 3层编码器
batch_size = 128 # 每个批次128张图片
epochs = 5 # 训练5个epoch
alpha = 0.005 # 学习率5e-3
```

== 加载MNIST数据集

```python
transform = T.Compose([
    T.Resize(img_size), # 28x28 --> 32x32
    T.ToTensor() # 转换成torch.tensor
])

train_set = MNIST(
    root="datasets", train=True, download=True, transform=transform
)
test_set = MNIST(
    root="datasets", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
```

== 训练循环

下面的代码使用 MNIST 训练集训练我们的 VisionTransformer 类，并展示了整个 epoch 的平均损失。

```python
device = torch.device("cuda")

ViT = VisionTransformer(
    d_model,
    n_classes,
    img_size,
    patch_size,
    n_channels,
    n_heads,
    n_layers
).to(device)

optimizer = Adam(ViT.parameters(), lr=alpha)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    training_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # 取出图像和对应的标签
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = ViT(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} loss: \
            {training_loss  / len(train_loader) :.3f}")
```

== 评估模型

```python
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = ViT(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"\n预测准确率: {100 * correct // total} %")
```

使用此模型，我们仅用 5 个 epoch 就能够在 MNIST 数据集上实现约 92% 的准确率。此示例展示了自注意力机制可以替代深度卷积网络。

#chapter("CLIP", image: image("./orange2.jpg"), l: "multimodal-chap2")

计算机视觉系统历来仅限于一组固定的类别，CLIP是一场革命，它允许通过"预测哪些图像和文本配对在一起"来识别开放世界中的对象。CLIP能够通过学习批量训练数据的图像和文本特征之间的余弦相似度来预测这一点。这在下图的对比预训练部分显示，其中图像之间的点积特征 ${I_1, I_2, ..., I_N}$ 和文本特征 ${T_1, T_2, ..., T_N}$ 被占用。

#figure(
  image("main-diagrams.svg"),
  caption: [clip原理，来自原始论文],
)

在本章中，我们将从头开始构建CLIP并在MNIST数据集上对其进行测试。

== 导入库和模块

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
```

== 图像和文本编码器

我们将首先构建图像和文本编码器。两者分别将图像和文本嵌入到单个token中，然后可用于对比损失计算。

=== 位置嵌入


```python
class PositionalEmbedding(nn.Module):
    def __init__(self, width, max_seq_length):
        super().__init__()
        # 创建一个 (token的数量, 嵌入的维度) 形状的全0张量
        # width 就是 d_model
        pe = torch.zeros(max_seq_length, width)
        # 将位置编码信息填充到pe中
        for pos in range(max_seq_length):
            for i in range(width):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/width)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/width)))
        # 位置编码信息进行冻结，不参与反向传播
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 直接将位置编码和token的嵌入进行相加
        x = x + self.pe
        return x
```

=== 注意力头


```python
class AttentionHead(nn.Module):
    def __init__(self, width, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(width, head_size)
        self.key = nn.Linear(width, head_size)
        self.value = nn.Linear(width, head_size)

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = Q @ K.transpose(-2,-1)
        attention = attention / (self.head_size ** 0.5)
        # 掩码
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V
        return attention
```

Transformer编码器和解码器之间的主要区别在于解码器使用注意力掩码，而编码器则不使用。虽然CLIP是仅编码器模型（Encoder-Only），但由于在分词时，对输入文本添加了pad（填充符），所以仍然需要与文本编码器一起使用掩码。请注意，掩码是可选的，因此这个注意力头可用于文本和视觉编码器。

#figure(
  image("注意力分数的掩码.svg"),
  caption: [注意力分数的掩码],
)


```python
# 使用注意力掩码
if mask is not None:
    attention = attention.masked_fill(mask == 0, float("-inf"))
```

=== 多头注意力


```python
class MultiHeadAttention(nn.Module):
    def __init__(self, width, n_heads):
        super().__init__()
        self.head_size = width // n_heads
        self.W_o = nn.Linear(width, width)
        self.heads = nn.ModuleList([
            AttentionHead(width, self.head_size)
            for _ in range(n_heads)
        ])

    def forward(self, x, mask=None):
        # 拼接多个注意力头
        out = torch.cat([head(x, mask=mask) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out
```

=== Transformer编码器


```python
class TransformerEncoder(nn.Module):
    def __init__(self, width, n_heads, r_mlp=4):
        super().__init__()
        self.width = width # 嵌入的维度d_model
        self.n_heads = n_heads

        # 层归一化
        self.ln1 = nn.LayerNorm(width)

        # 多头注意力
        self.mha = MultiHeadAttention(width, n_heads)

        # 层归一化
        self.ln2 = nn.LayerNorm(width)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.width, self.width*r_mlp),
            nn.GELU(),
            nn.Linear(self.width*r_mlp, self.width)
        )

    def forward(self, x, mask=None):
        x = x + self.mha(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x
```

=== 文本的分词器


我们的词汇表是ascii码，所以 `vocab_size` = 256 。

```python
def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        out = chr(2) + text + chr(3) # 添加 SOT token 和 EOT token
        out = out + "".join([
            chr(0) for _ in range(max_seq_length-len(out))
        ]) # 添加Padding
        out = torch.IntTensor(list(out.encode("utf-8"))) # 对文本进行编码
        mask = torch.ones(len(out.nonzero()))
        mask = torch.cat((
            mask,
            torch.zeros(max_seq_length - len(mask))
        )).type(torch.IntTensor)
    else:
        # 将input_ids解码为text文本
        out = [chr(x) for x in text[1:len(mask.nonzero())-1]]
        out = "".join(out)
        mask = None

    return out, mask
```

\

Transformer无法处理原始文本，因此我们需要做的第一件事是在将输入字符串通过文本编码器之前对其进行分词。

在本教程中，我们将进行一个简单版本的分词，其中我们只使用 UTF-8 编码。为什么使用 UTF-8 编码进行分词？因为我们只会在示例中使用简单的ascii码文本。对于更复杂的示例，可能需要使用BPE分词器。这是因为使用UTF-8编码时，词表大小（`vocab_size`）为256，这意味着对于更复杂的示例，可能会有更长的输入序列，由于上下文长度有限，这在计算注意力时效率低下。

#figure(
  image("分词.svg"),
  caption: [分词过程],
)

分词器的第一步是将文本的开头和文本的结尾token添加到输入字符串中。

```python
text = chr(2) + text + chr(3)
```

添加文本的开头和文本的结尾token后，我们需要将序列的长度填充到最大序列长度。

```python
text = text + "".join([chr(0) for _ in range(10-len(text))])
```

我们通过将文本序列编码为UTF-8并将输出转换为 `IntTensor` 来完成分词。

```python
text = torch.IntTensor(list(text.encode("utf-8")))
```

对文本进行分词后，我们需要为文本创建掩码。虽然Transformer中通常使用的掩码用于确保token不与未来的token通信，但我们在这里应用的掩码只是使pad token被忽略。因此，掩码将只是一个大小等于最大序列长度的张量，其中元素在有填充的情况下为0，否则为1。

```python
mask = torch.ones(len(text.nonzero()))
mask = torch.cat((mask,torch.zeros(10-len(mask)))).type(torch.IntTensor)
```

=== 文本编码器

#codly(
  highlighted-lines: (40, 41),
  annotations: (
    (
      start: 40,
      end: 41,
      content: block(
        width: auto,
        align(
          left,
        )[将文本嵌入映射到CLIP嵌入空间#sym.space],
      ),
    ),
  ),
)
```python
class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size, # 词汇表大小=256
        width, # 宽度d_model
        max_seq_length, # 文本最大长度
        n_heads,
        n_layers,
        emb_dim # 嵌入维度
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.encoder_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = PositionalEmbedding(
            width,
            max_seq_length
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(width, n_heads)
            for _ in range(n_layers)
        ])
        # 可学习投影（projection）
        # $W _ ("width" times "emb_dim")$
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, text, mask=None):
        # 文本嵌入
        x = self.encoder_embedding(text)
        # 位置嵌入
        x = self.positional_embedding(x)
        # Transformer编码器
        for encoder_layer in self.encoder:
            x = encoder_layer(x, mask=mask)
        # 从EOT的嵌入抽取特征
        x = x[
            torch.arange(text.shape[0]), # 批次中数据的索引
            # 取出掩码mask矩阵的第0行，加和再减1，就得到了EOT的索引
            torch.sub(torch.sum(mask[:,0],dim=1),1)
        ]
        if self.projection is not None:
            x = x @ self.projection
        # 向量x转换为模长为1的向量
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x
```

对于文本编码器，我们将使用常规Transformer模型。创建文本编码器的第一步是创建大小为 `（vocab_size, width)` 的嵌入表。此嵌入表包含一个向量表示，其大小等于词汇表中每个 token 的 Transformer 模型的 `width` 。

```python
self.encoder_embedding = nn.Embedding(vocab_size, width)
```

在输出 Transformer 的结果之前，我们需要将特征嵌入到联合嵌入空间中。我们将通过获取文本特征的点积以及使用 `nn.Parameter` 创建的可学习的投影来实现这一点。

```python
self.projection = nn.Parameter(torch.randn(width, emb_dim))
```

在 `forward` 方法中，我们要做的第一件事是通过嵌入表传递文本的token。

```python
x = self.encoder_embedding(text)
```

然后，我们需要将位置编码添加到嵌入表的输出中。

```python
x = self.positional_embedding(x)
```

添加位置编码后，我们现在可以将其与掩码一起通过编码器层。

```python
for encoder_layer in self.encoder:
    x = encoder_layer(x, mask=mask)
```

编码器层的输出是文本的特征。我们将使用从 `EOT` 的嵌入中抽取的特征。

```python
# 从EOT的嵌入抽取特征
x = x[torch.arange(
  text.shape[0]),
  torch.sub(torch.sum(mask[:,0],dim=1),1)
]
```

最后，我们通过计算特征和投影之间的点积，将文本嵌入映射到CLIP嵌入空间中，并通过除以归一化的点积对其进行归一化。

#tip(title: "为什么要做映射？")[
  主要是为了在CLIP嵌入空间中，文本嵌入向量的维度和图像嵌入向量的维度一致。
  向量除以向量的模长，就是模长为1的向量。这样文本嵌入向量和图像嵌入向量都变成了模长为1的向量。两个向量的点积，就是两个向量的余弦相似度！
]

```python
if self.projection is not None:
    x = x @ self.projection
x = x / torch.norm(x, dim=-1, keepdim=True)
return x
```

=== 图像编码器

#codly(
  highlighted-lines: (58, 59),
  annotations: (
    (
      start: 58,
      end: 59,
      content: block(
        width: auto,
        align(
          left,
        )[将图像嵌入映射到CLIP嵌入空间#sym.space],
      ),
    ),
  ),
)
```python
class ImageEncoder(nn.Module):
    def __init__(
        self,
        width, # 补丁嵌入的维度d_model
        img_size,
        patch_size,
        n_channels,
        n_layers,
        n_heads,
        emb_dim
    ):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0  \
           and img_size[1] % patch_size[1] == 0, \
           "img_size必须能被patch_size整除"
        assert width % n_heads == 0, \
           "width必须能被n_heads整除"

        self.n_patches = (img_size[0] * img_size[1]) \
                      // (patch_size[0] * patch_size[1])
        self.max_seq_length = self.n_patches + 1
        self.linear_project = nn.Conv2d(
            n_channels,
            width,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.positional_embedding = PositionalEmbedding(
            width,
            self.max_seq_length
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(width,n_heads)
            for _ in range(n_layers)
        ])

        # $W _ ("width" times "emb_dim")$
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self,x):
        # 补丁嵌入
        x = self.linear_project(x)
        x = x.flatten(2).transpose(1, 2)

        # 位置嵌入
        x = torch.cat((self.cls_token.expand(x.size()[0], -1, -1),x), dim=1)
        x = self.positional_embedding(x)

        # Transformer编码器
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # 获取类别token
        x = x[:, 0, :]

        if self.projection is not None:
            x = x @ self.projection
        # 向量x转换为模长为1的向量
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x
```

== CLIP模型


```python
class CLIP(nn.Module):
    def __init__(
        self,
        emb_dim, # 经过可学习投影后的嵌入维度
        vit_width, # 图像编码器的宽度(d_model)
        img_size,
        patch_size,
        n_channels,
        vit_layers,
        vit_heads,
        vocab_size,
        text_width, # 文本编码器的宽度（d_model）
        max_seq_length,
        text_heads,
        text_layers
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            vit_width,
            img_size,
            patch_size,
            n_channels,
            vit_layers,
            vit_heads,
            emb_dim
        )
        self.text_encoder = TextEncoder(
            vocab_size,
            text_width,
            max_seq_length,
            text_heads,
            text_layers,
            emb_dim
        )
        # 可学习温度
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = torch.device("cuda")


    def forward(self, image, text, mask=None):
        # $I_e$是图像嵌入，形状 [B, D=emb_dim]
        I_e = self.image_encoder(image)
        # $T_e$是文本嵌入，形状 [B, D=emb_dim]
        T_e = self.text_encoder(text, mask=mask)

        # 缩放逐点余弦相似度[n, n]
        # 形状 $I_e dot.c T_e$ : [B, D] @ [D, B] --> [B, B]
        logits = (I_e @ T_e.transpose(-2,-1)) * torch.exp(self.temperature)

        # 对称损失函数：labels形状为[B]，值为 [0, 1, 2, ..., B-1]
        labels = torch.arange(logits.shape[0]).to(self.device)
        # 从文本 --> 图像方向，以文本嵌入 $T_3$ 为例子，
        # 交叉熵损失的目标是让 $T_3 dot.c I_3$ 越大越好
        loss_i = nn.functional.cross_entropy(
            logits.transpose(-2,-1),
            labels
        )
        # 从图像 --> 文本方向，以图像嵌入 $I_3$ 为例子，
        # 交叉熵损失的目标是让 $I_3 dot.c T_3$ 越大越好
        loss_t = nn.functional.cross_entropy(
            logits,
            labels
        )
        # 两个方向的损失求平均值
        loss = (loss_i + loss_t) / 2

        return loss
```

当给定一批图像和标题时，CLIP应该告诉我们哪些标题与哪些图像搭配。它通过一起训练文本编码器和图像编码器来最大化应该在一起的对的成对余弦相似度分数，并最小化不应该在一起的对来做到这一点。

为此，我们首先需要从图像和文本编码器中获取特征的嵌入。

```python
def forward(self,image,text, mask=None):
    I_e = self.image_encoder(image)
    T_e = self.text_encoder(text, mask=mask)
```

使用嵌入的特征，我们可以通过使用嵌入的图像特征和嵌入文本特征的转置版本之间的点积来计算缩放的成对余弦相似度。余弦相似度应沿图中正确的图像和文本配对在一起的对角线最大化。

#figure(
  image("clip-1.svg"),
  caption: [余弦相似度的计算],
)


```python
logits = (I_e @ T_e.transpose(-2,-1)) * torch.exp(self.temperature)
```

这在 `I_e` 和 `T_e` 分别包含N个批次时工作，从而产生上图所示的矩阵。为了最大化相关图像之间的余弦相似度，CLIP使用对称/对比损失。我们可以通过首先创建与批次中的每条数据相对应的标签来计算此损失。

```python
# 对称损失函数
labels = torch.arange(logits.shape[0]).to(self.device)
```

然后，我们计算沿logit行的交叉熵损失，以获得图像的损失。

```python
loss_i = nn.functional.cross_entropy(logits.transpose(-2,-1), labels)
```

文本的损失是通过计算沿列的交叉熵损失来计算的。

```python
loss_t = nn.functional.cross_entropy(logits, labels)
```

我们通过计算图像损失和文本损失之间的平均值来获得最终损失。

```python
loss = (loss_i + loss_t) / 2
return loss
```

=== 数据


```python
class MNIST(Dataset):
    def __init__(self, train=True):
        self.dataset = load_dataset("./../datasets/clip-mnist/")
        self.transform = T.ToTensor()
        if train:
            self.split = "train"
        else:
            self.split = "test"

        self.captions = {
            0: "An image of Zero",
            1: "An image of One",
            2: "An image of Two",
            3: "An image of Three",
            4: "An image of Four",
            5: "An image of Five",
            6: "An image of Six",
            7: "An image of Seven",
            8: "An image of Eight",
            9: "An image of Nine"
        }

    def __len__(self):
        return self.dataset.num_rows[self.split]

    def __getitem__(self, i):
        # 取出第i张图片
        img = self.dataset[self.split][i]["image"]
        # 转换成张量
        img = self.transform(img)
        # 图片对应的文本，以及掩码
        cap, mask = tokenizer(
           self.captions[self.dataset[self.split][i]["label"]]
        )
        # 为什么要repeat？
        mask = mask.repeat(len(mask), 1)
        return {"image": img, "caption": cap, "mask": mask}
```


在本教程中，我们将使用MNIST数据集。我们选择这个数据集是因为它相当小并且保持训练时间合理。

```python
self.dataset = load_dataset("clip-mnist")
```

对于数据集中的每个条目，我们将需要3样东西：图像、文本和文本掩码。

对于图像，我们唯一需要进行的更改是将图像转换为张量。

```python
img = self.dataset[self.split][i]["image"]
img = self.transform(img)
```

对于标题，我们需要将其传递给我们创建的分词器，以获取token表示以及token的掩码。

```python
cap, mask = tokenizer(self.captions[self.dataset[self.split][i]["label"]])
```

我们从分词器获得的掩码是大小为 `max_seq_length` 的1维张量。在文本编码器中，掩码将应用于形状为 `(max_seq_length, max_seq_length)` 的注意力分数。因此，我们需要扩展掩码，以便将其应用于注意力分数的每一行。

#figure(
  image("扩展掩码.svg"),
  caption: [复制掩码],
)


```python
mask = mask.repeat(len(mask), 1)
```

图像、标题和掩码作为字典保存在数据集中。

```python
return {"image": img, "caption": cap, "mask": mask}
```

#pagebreak()

#error(title: "不要和因果注意力混淆！")[
  #figure(
    image("因果注意力.svg"),
    caption: [因果注意力分数的掩码],
  )
  创建 `[context_length , context_length]` 的上三角矩阵。

  ```python
  self.register_buffer(
      'mask',
      torch.triu(torch.ones(
          context_length,
          context_length
      ), diagonal = 1))
  ```

  将注意力分数的上三角置为`-torch.inf`，下三角保留。得到注意力分数的下三角矩阵。

  ```python
  attention_score.masked_fill_(
      self.mask[:seq_len, :seq_len].bool(),
      -torch.inf
  )
  ```
]

=== 训练参数


```python
emb_dim = 32 # 文本编码器和图像编码器输出的张量的维度
vit_width = 9 # 图像编码器的嵌入的宽度
img_size = (28,28)
patch_size = (14,14)
n_channels = 1
vit_layers = 3 # 图像编码器中编码器的层的数量
vit_heads = 3 # 图像编码器中注意力头的数量
vocab_size = 256 # 词汇表大小
text_width = 32 # 文本编码器的嵌入的宽度
max_seq_length = 32 # 最大序列长度
text_heads = 8 # 文本编码器中注意力头的数量
text_layers = 4 # 文本编码器中编码器的层数
lr = 1e-3 # 学习率
epochs = 10
batch_size = 128
```

=== 加载数据集


```python
train_set = MNIST(train = True)
test_set = MNIST(train = False)

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
```

=== 训练模型


```python
device = torch.device("cuda")

model = CLIP(
    emb_dim,
    vit_width,
    img_size,
    patch_size,
    n_channels,
    vit_layers,
    vit_heads,
    vocab_size,
    text_width,
    max_seq_length,
    text_heads,
    text_layers
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

best_loss = np.inf
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        img = data["image"].to(device)
        cap = data["caption"].to(device)
        mask = data["mask"].to(device)
        loss = model(img, cap, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Batch Loss: {loss.item():.3f}")

    # 保存模型
    if loss.item() <= best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "./clip.pt")
        print("模型已经保存.")
```

=== 评估模型


```python
# 加载最好的模型
model = CLIP(
    emb_dim,
    vit_width,
    img_size,
    patch_size,
    n_channels,
    vit_layers,
    vit_heads,
    vocab_size,
    text_width,
    max_seq_length,
    text_heads,
    text_layers
).to(device)
model.load_state_dict(torch.load("./clip.pt", map_location=device))

# 获取数据集的标签和图片进行对比
text = torch.stack(
    [tokenizer(x)[0] for x in test_set.captions.values()]
).to(device)
mask = torch.stack(
    [tokenizer(x)[1] for x in test_set.captions.values()]
)
mask = mask.repeat(
    1,
    len(mask[0])
).reshape(
    len(mask),
    len(mask[0]
),len(mask[0])).to(device)

correct, total = 0, 0
with torch.no_grad():
    for data in test_loader:
        # 图像
        images = data["image"].to(device)
        # 文本
        labels = data["caption"].to(device)
        # 使用clip模型中的图像编码器对图像抽取特征
        image_features = model.image_encoder(images)
        # 使用clip模型中的文本编码器对文本抽取特征
        text_features = model.text_encoder(text, mask=mask)
        # 转换为模长为1的向量
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # I_e @ T_e^T: $I_e dot.c T_e$
        similarity = (
            100.0 * image_features @ text_features.T
        ).softmax(dim=-1)
        _, indices = torch.max(similarity, 1)
        # 预测结果
        pred = torch.stack([
            tokenizer(test_set.captions[int(i)])[0]
            for i in indices
        ]).to(device)
        # 预测正确的样本数量
        correct += int(sum(torch.sum((pred==labels),dim=1)//len(pred[0])))
        total += len(labels)

print(f'\n预测准确率: {100 * correct // total} %')
```

我们通过获取模型训练的标题并将其与实际标题进行比较来测试模型。在训练时，我们使用了相同的标题模板（`"A image of a（n） {class}"）`，因此这个测试阶段与任何其他图像分类器几乎相同。我们实现了大约 85% 的模型准确率。

=== 零样本分类

```python
# 加载模型
model = CLIP(
    emb_dim,
    vit_width,
    img_size,
    patch_size,
    n_channels,
    vit_layers,
    vit_heads,
    vocab_size,
    text_width,
    max_seq_length,
    text_heads,
    text_layers
).to(device)
model.load_state_dict(torch.load("./clip.pt", map_location=device))

# 标题
class_names = [
    "a photo of 0",
    "an image of one",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "Trump",
    "Musk"
]

text = torch.stack(
    [tokenizer(x)[0] for x in class_names]
).to(device)
mask = torch.stack(
    [tokenizer(x)[1] for x in class_names]
)
mask = mask.repeat(
    1,
    len(mask[0])
).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)

idx = 1000
# 去测试数据集中的第1000张图片
img = test_set[idx]["image"][None,:]
plt.imshow(img[0].permute(1, 2, 0), cmap="gray")
# 将图片的标题文本展示，例如"An Image Of Nine"
plt.title(tokenizer(
    test_set[idx]["caption"],
    encode=False,
    mask=test_set[idx]["mask"][0]
)[0])
plt.show()
img = img.to(device)
with torch.no_grad():
    # 抽取图片的特征
    image_features = model.image_encoder(img)
    # 抽取文本的特征
    text_features = model.text_encoder(text, mask=mask)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
# 计算第1000张图片和所有文本的相似度
similarity = (
    100.0 * image_features @ text_features.T
).softmax(dim=-1)
# 返回所有文本中和图片特征最相似的5个文本
values, indices = similarity[0].topk(5)

# 打印结果
print("\n预测结果:\n")
for value, index in zip(values, indices):
    print(f"{class_names[int(index)]:>16s}: {100 * value.item():.2f}%")
```

对于zero-shot分类，我们将图像与类别的名称进行比较。我们输入标签以与图像进行比较，它将返回前5个预测以及预测的可能性。这不是CLIP执行zero-shot分类的最佳示例。使用MNIST数据集使模型易于训练，但标题不是很丰富。要真正理解CLIP的zero-shot功能，包含多个名称的训练集会更合适。真正的zero-shot检测将允许检测以前未见过的排列。

#tip(title: "应用举例：文搜图")[
  具体步骤如下：
  + 将数据库中的所有图片使用clip的图像编码器进行抽取特征
  + 将抽取的图片特征存入向量数据库
  + 将输入的搜索文本通过clip的文本编码器抽取文本特征
  + 使用余弦相似度找出向量数据库中和文本特征最接近的几张图片
]

#chapter("ClipCap（图生文）", image: image("./orange2.jpg"), l: "multimodal-clipcap")

#tip(title: "Image Captioning Model")[
  图片标题模型（image captioning model）是一种将图片作为输入并生成图片描述的模型。
]

下面是一个简单的示例，展示了一个图片标题模型：

#figure(
  image("clipcap的简单示例.svg"),
  caption: [图片标题模型的简单示例],
)

== ClipCap的工作原理

ClipCap 是一种结合了 CLIP 和 GPT-2 的图片标题架构。

CLIP 是我们将用来创建输入图像嵌入的模型。

GPT-2 是一种基于解码器的模型，用于生成文本。

ClipCap 的基本工作原理如下：

输入图像首先通过 CLIP 模型转换为嵌入，目的是利用这种嵌入（捕捉图像意义）来引导 GPT-2 生成文本。

但有一个问题：CLIP 和 GPT-2 的嵌入空间不同。所以我们不能直接把这个嵌入输入到 GPT-2 中。

为了解决这个问题，我们使用一个映射网络将 CLIP 嵌入映射到 GPT-2 的嵌入空间。

这些映射的图像嵌入称为前缀（prefixes），因为它们是 GPT-2 生成图片说明所需的上下文。

#figure(
  image("将clip嵌入映射到gpt嵌入.svg"),
  caption: [将图片的CLIP嵌入映射到GPT2的嵌入空间],
)

== 关于训练

CLIP 生成的图像嵌入开箱即用已经足够好——所以我们不训练 CLIP 模型。

根据 GPT-2 是否经过微调，ClipCap 有两种变体 ：

- 如果我们对 GPT-2 进行微调 ，那么我们就用 MLP 作为映射网络。GPT-2 和 MLP 都经过训练。
- 如果我们不对 GPT-2 进行微调，那么我们就用Transformer架构（例如Bert）作为映射网络，只有Transformer本身被训练。

就我而言，我选择对 GPT-2 模型进行微调 ，并使用 MLP 作为映射网络。

== 推理

在我们这里，这意味着为没见过的图片生成说明文字。

让我们用一辆蓝色汽车的图片作为推理过程的例子：

+ 输入图像被转换为CLIP图像嵌入。
+ CLIP图像嵌入通过映射网络传递，生成前缀嵌入 。
+ 在时间步`t=1`时，我们将这些前缀嵌入传递到GPT2。模型预测下一个token——假设是"a"。我们将此附加到序列中。
+ 在`t=2`时，我们将更新后的输入序列（前缀+"a"）传递给GPT2。它预测下一个token——这次可能是"蓝色"。
+ 这一过程持续进行，模型一次预测一个token，直到它：
  - 生成 `<EOS>`（序列结束）token，或
  - 生成的标题长度达到最大。

如下图所示

#figure(
  image("clipcap的推理.svg"),
  caption: [ClipCap的推理过程],
)

== ClipCap的代码实现

=== 项目配置

#codly(
  header: [config.py],
)
```python
import torch

CLIP_MODEL_PATH = "./chinese-clip-vit-base-patch16"
# 一张图片的嵌入经过投影转换成10个token的embedding，每个embedding的dim是768
IMAGE_TOKEN_LENGTH = 10 # 图片的token的数量
MAX_LENGTH = 100 # 最大token数量
# clip对接的大语言模型
LLM_PATH = "./gpt2-chinese-cluecorpussmall"
LLM_WORD_EMBD_DIM = 768 # gpt2的词嵌入维度
IMAGE_EMBD_DIM = 512 # clip输出的图像嵌入的维度
device = torch.device("cuda")
```

=== 处理数据

#codly(
  header: [process_data.py],
)
```python
from PIL import Image
import pickle
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from config import CLIP_MODEL_PATH


def main():
    # 加载clip模型
    # clip模型只用来生成图片的嵌入，不进行微调。
    clip_model = ChineseCLIPModel.from_pretrained(CLIP_MODEL_PATH)
    # 加载clip处理器
    processor = ChineseCLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    # 将2张图片进行处理，处理完之后交给clip抽取特征
    inputs_1 = processor(images=Image.open("1.jpg"), return_tensors="pt")
    inputs_2 = processor(images=Image.open("2.jpg"), return_tensors="pt")
    # 获取第一张图片的嵌入（dim: 512）
    image_1_features = clip_model.get_image_features(**inputs_1)
    image_2_features = clip_model.get_image_features(**inputs_2)
    # 除以模长，归一化
    image_1_features = image_1_features / \
        image_1_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    image_2_features = image_2_features / \
        image_2_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    # key：图片id
    # value: 图片的嵌入
    # 下面的字典也可以放在chroma这样的向量数据库
    image_id2embed = {
        1: image_1_features,
        2: image_2_features,
    }
    # 图片的id和图片标题对
    caption_list = [
        (1, "两只狗在雪地里嬉闹"),
        (2, "一件好看的立领的很特别的紫色风衣"),
    ]

    with open("caption_image.pkl", 'wb') as f:
        pickle.dump([caption_list, image_id2embed], f)

    print(f'图像嵌入的数量:{len(image_id2embed)}')
    print(f'图像文本的数量:{len(caption_list)}')


if __name__ == '__main__':
    main()
```

=== 模型结构

#figure(
  image("clipcap模型的设计要点.svg"),
  caption: [clipcap的模型设计要点],
)

#codly(
  header: [model.py],
)
```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from typing import Sequence
from config import LLM_PATH, IMAGE_TOKEN_LENGTH, IMAGE_EMBD_DIM


class MLP(nn.Module):
    """投影层"""
    def __init__(self, sizes: Sequence[int]):
        super().__init__()

        in_dim, h1, out_dim = sizes
        self.l1 = nn.Linear(in_dim, h1)
        self.act1 = nn.Tanh()
        self.l2 = nn.Linear(h1, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        return x


class ClipCaptionModel(nn.Module):
    def __init__(self):
        super(ClipCaptionModel, self).__init__()
        # 大语言模型：用来生成图片的文本描述
        self.gpt2 = GPT2LMHeadModel.from_pretrained(LLM_PATH)
        # gpt2的词嵌入维度是768
        self.word_embd_dim = self.gpt2.config.n_embd
        # 投影层定义
        self.projection = MLP((
            # 输入维度是512,也就是clip的图像编码器输出的嵌入的维度
            IMAGE_EMBD_DIM,
            # (768 * 10) // 2
            # 10是图片嵌入转换成的token数量
            (self.word_embd_dim * IMAGE_TOKEN_LENGTH) // 2,
            # 768 x 10，图片占10个token，每个token的嵌入和词嵌入相同是768维度
            self.word_embd_dim * IMAGE_TOKEN_LENGTH
        ))

    def forward(self, image_embeds, caption_ids, mask):
        # 张量形状：[B, 文本长度, gpt2的词嵌入的维度]
        # 标题caption的每个token的词嵌入
        caption_embeds = self.gpt2.transformer.wte(caption_ids)
        # 将图片的嵌入转换为像词嵌入那样的维度：
        # [B, 图片token的长度为10, gpt2的词嵌入的维度]
        image_as_word_embeds = self.projection(
            image_embeds
        ).view(-1, IMAGE_TOKEN_LENGTH, self.word_embd_dim)
        # 10个图片的token + 文本的token
        # 张量形状：[B, 10+文本长度, 词嵌入维度]
        embedding_cat = torch.cat((
            image_as_word_embeds, # 图像的token
            caption_embeds # 文本的token
        ), dim=1)
        out = self.gpt2(inputs_embeds=embedding_cat, attention_mask=mask)
        # 张量形状：[B, 10+文本长度，词嵌入维度]
        logits = out.logits
        return logits
```

=== 准备训练数据集

#codly(header: [clipcap_dataset.py])
```python
import torch
from torch.utils.data import Dataset
import pickle
from typing import Tuple
from config import IMAGE_TOKEN_LENGTH, MAX_LENGTH


class ClipCapDataset(Dataset):
    def __init__(self, tokenizer):
        # 填充符
        pad_id = tokenizer.pad_token_id
        # 取出图片的文本和图片的嵌入
        with open("caption_image.pkl", 'rb') as f:
            caption_list, image_id2embed = pickle.load(f)
        print('图片嵌入的总数:{}'.format(len(image_id2embed)))
        print('图片描述的总数:{}'.format(len(caption_list)))

        image_embed_list = []
        caption_ids_list = []
        mask_list = []
        for image_id, caption in caption_list:
            # 使用图像id获取图像的特征（clip.image_encoder输出的）
            image_embed = image_id2embed[image_id]
            # 只对文本进行分词，不添加任何特殊token
            caption_ids = tokenizer.encode(
                caption,
                add_special_tokens=False
            )
            # 在文本的token列表后面添加一个分隔符token
            caption_ids.append(tokenizer.sep_token_id)

            # 截断
            # 只能留下前90个token，因为图像对应的token需要占用10个token的位置
            # 最终的数据是：图像的token列表 + 文本的token列表
            caption_ids = caption_ids[:MAX_LENGTH - IMAGE_TOKEN_LENGTH]
            # 图像部分和文本部分的token都要掩码
            mask = [1] * (IMAGE_TOKEN_LENGTH + len(caption_ids))

            # 填充pad
            padding_len = MAX_LENGTH         \
                        - IMAGE_TOKEN_LENGTH \
                        - len(caption_ids)
            caption_ids += [pad_id] * padding_len
            # 将填充符掩码为0
            mask += [0] * padding_len

            caption_ids = torch.tensor(caption_ids).long()
            mask = torch.tensor(mask).long()

            image_embed_list.append(image_embed)
            caption_ids_list.append(caption_ids)
            mask_list.append(mask)
        # 保存训练数据
        with open("train_data.pkl", 'wb') as f:
            pickle.dump([
                image_embed_list, # clip输出的图片特征的列表
                caption_ids_list, # 图片文本的input_ids的列表
                mask_list # 掩码的列表
            ], f)
        self.image_embed_list = image_embed_list
        self.caption_ids_list = caption_ids_list
        self.mask_list = mask_list
        print(f'训练数据总数：{len(self.image_embed_list)}')

    def __len__(self) -> int:
        return len(self.caption_ids_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        image_embed = self.image_embed_list[index]
        caption_ids = self.caption_ids_list[index]
        mask = self.mask_list[index]
        return image_embed, caption_ids, mask
```

=== 训练

#figure(
  image("clip接生成模型的训练目标.svg"),
  caption: [clipcap的训练目标],
)

#codly(header: [train.py])
```python
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from clipcap_dataset import ClipCapDataset
from model import ClipCaptionModel
import torch.nn.functional as F
from config import LLM_PATH, IMAGE_TOKEN_LENGTH, device


def train(model, train_loader, optimizer):
    model.train()
    for _ in range(20):
        for _, data in enumerate(tqdm(train_loader)):
            image_embed, caption_ids, mask = data
            image_embed = image_embed.to(device)
            caption_ids = caption_ids.to(device)
            mask = mask.to(device)
            # 输出的logits
            logits = model(image_embed, caption_ids, mask)

            # 计算loss
            # [图片的最后一个token]，[两]，[只]，[狗]
            #          ↓            ↓    ↓
            #         [两]         [只]  [狗]
            shift_logits = logits[
                :,
                # 截取范围[图片的最后一个token～倒数第二个token]
                IMAGE_TOKEN_LENGTH - 1: -1, # 去掉最后一个token
                :
            ].contiguous().view(-1, logits.size(-1))
            # 预测目标
            shift_labels = caption_ids.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)
            # logits.size(-1): 取 logits 的最后一维大小。一般最后一维是词表大小 vocab_size。
            # 原 logits 形状通常是 [B, L, V]（批大小、序列长度、词表大小）。
            # 经过切片后，shift_logits 的形状是 [B, L-1, V]。
            # 再 `.contiguous().view(-1, logits.size(-1))`
            # 就变成 [B*(L-1), V]，把前两维展平，便于和标签做交叉熵。
            # caption_ids 形状通常是 [B, L-1]（与 shift_logits 的时间步对齐）。
            # caption_ids.view(-1) 把它展平成 [B*(L-1)]，与 shift_logits 的第一维对齐，
            # 用于计算 CrossEntropyLoss(shift_logits, shift_labels)。

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save(model.state_dict(), f'model.pt')


def main():
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(LLM_PATH)
    # 加载模型
    model = ClipCaptionModel().to(device)

    dataset = ClipCapDataset(tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train(model, train_dataloader, optimizer)


if __name__ == '__main__':
    main()
```

=== 推理

#figure(
  image("clipcap根据图片嵌入预测下一个token.svg"),
  caption: [clipcap只根据图片的嵌入预测文本的token，也就是看图说话，不再需要文本了！],
)

#codly(header: [infer.py])
```python
from PIL import Image
import torch
from transformers import BertTokenizer, ChineseCLIPModel, ChineseCLIPProcessor
from model import ClipCaptionModel
import torch.nn.functional as F
from config import LLM_PATH, CLIP_MODEL_PATH, IMAGE_TOKEN_LENGTH, LLM_WORD_EMBD_DIM, device, MAX_LENGTH


def generate(model, image_embeds, tokenizer):
    """
    :param image_embeds: [B, IMAGE_EMBD_DIM=512]
    """

    b_size = image_embeds.size(0)
    pad_id = tokenizer.pad_token_id
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id
    temperature = 0.7

    cur_len = 0
    caption_ids = []    # 存储生成的caption

    # gpt2模型的输入: inputs_embeds:[B, 图片token的数量为10, gpt2的词嵌入维度768]
    # 先将图片特征投影为10个图片token
    inputs_embeds = model.projection(
        image_embeds
    ).view(-1, IMAGE_TOKEN_LENGTH, LLM_WORD_EMBD_DIM)
    finish_flag = [False] * b_size  # 第i个输入是否完成生成的标志

    while True:
        out = model.gpt2(inputs_embeds=inputs_embeds)
        logits = out.logits  # [B, len, vocab_size]
        # 采样下一个token
        next_token_logits = logits[:, -1, :]    # 取最后一个单词的预测分布
        next_token_logits = next_token_logits / temperature
        next_token_logits[:, unk_id] = -float('Inf')   # 将unk设为无穷小

        # 采样下一个token，多项分布
        next_token_ids = torch.multinomial(
            F.softmax(next_token_logits, dim=-1),
            num_samples=1
        ).squeeze(1).tolist()

        # 分别判断生成图片是否已生成完毕
        # index表示第index张正在生成文本的图片
        for index in range(len(next_token_ids)):
            token_id = next_token_ids[index]
            # 如果第i个句子已经生成结束
            if finish_flag[index]:
                next_token_ids[index] = pad_id
            # 如果第i个句子生成结束，预测到了分隔符
            elif token_id == sep_id:
                finish_flag[index] = True
            # 生成刚开始
            elif cur_len == 0:
                caption_ids.append([token_id])
            else:
                caption_ids[index].append(token_id)
        next_token_ids = torch.tensor(next_token_ids).to(device)
        next_token_embeds = model.gpt2.transformer.wte(
            next_token_ids).to(device).unsqueeze(1)
        # 将生成的next token拼接到上文的后面，继续生成
        inputs_embeds = torch.cat((inputs_embeds, next_token_embeds), dim=1)

        cur_len += 1 # 生成长度+1
        # 如果生成长度大于最大长度，或者所有图片的生成文本都结束了，退出生成过程。
        if cur_len > MAX_LENGTH or False not in finish_flag:
            break

    # 对token_id进行解码
    captions = []
    for caption_id in caption_ids:
        caption = tokenizer.convert_ids_to_tokens(caption_id)
        caption = ''.join(caption)
        captions.append(caption)

    return captions


def main():
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(LLM_PATH)
    # 初始化模型
    model = ClipCaptionModel().to(device)
    # 加载权重
    model.load_state_dict(torch.load(
        "model.pt",
        map_location=device
    ), False)
    model.eval()

    # 加载clip模型
    clip_model = ChineseCLIPModel.from_pretrained(CLIP_MODEL_PATH)
    processor = ChineseCLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    inputs_1 = processor(images=Image.open("1.jpg"), return_tensors="pt")
    inputs_2 = processor(images=Image.open("2.jpg"), return_tensors="pt")
    image_1_features = clip_model.get_image_features(**inputs_1)
    image_2_features = clip_model.get_image_features(**inputs_2)
    image_1_features = image_1_features / \
        image_1_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    image_2_features = image_2_features / \
        image_2_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    # 将两张图片的特征打包成一个批次数据
    data = torch.stack([
        image_1_features,
        image_2_features
    ], dim=0).to(device)
    captions = generate(model, data, tokenizer)
    print(captions)
    captions = generate(model, data, tokenizer)
    print(captions)
    captions = generate(model, data, tokenizer)
    print(captions)
    captions = generate(model, data, tokenizer)
    print(captions)
    captions = generate(model, data, tokenizer)
    print(captions)


if __name__ == '__main__':
    main()
```

== 应用：图生文

当前电商平台上的商品描述主要依赖商家手工编写，质量参差不齐，存在描述不准确、信息不完整、语言不标准等问题。这不仅影响用户的购买决策，还增加了平台内容审核与优化的成本。如何利用智能化手段快速生成高质量的商品描述，成为亟待解决的核心问题。

#figure(
  image("A800训练10个小时的效果.png"),
  caption: [A800训练10个小时的效果],
)

== 拓展：Qwen3-VL

#figure(
  image("qwen2.5vl_arc.jpeg"),
  caption: [Qwen2.5-VL 框架展示了视觉编码器与语言模型解码器的集成，用以处理包括图像和视频在内的多模态输入。视觉编码器被设计为处理原生分辨率的输入并支持动态帧率采样。不同尺寸的图像和具有不同时帧率的视频帧被动态映射为长度各异的token序列。值得注意的是，MRoPE 在时间维度上将时间 ID 与绝对时间对齐，使模型能够更好地理解时间动态，例如事件的节奏和精确的时刻定位。处理后的视觉数据随后被输入到 Qwen2.5 LM 解码器。我们对ViT架构进行了重构，加入了诸如带 SwiGLU 激活的前馈网络（FFN）、用于归一化的 RMSNorm 以及基于窗口的注意力机制等先进组件，以提升性能和效率。],
)

Qwen2.5-VL 的整体模型架构由三部分组成：

+ 大语言模型：Qwen2.5-VL 系列以大语言模型（LLM）作为其基础组件。该模型以 Qwen2.5 LLM 的预训练权重进行初始化。为了更好地满足多模态理解的需求，我们将一维 RoPE（旋转位置嵌入）修改为与绝对时间对齐的多模态旋转位置嵌入（Multimodal Rotary Position Embedding Aligned to Absolute Time）。
+ 视觉编码器：Qwen2.5-VL 的视觉编码器采用了重新设计的ViT架构。在结构上，我们引入了二维 RoPE 和窗口注意力，以支持原生输入分辨率并加速整个视觉编码器的计算。在训练和推理过程中，输入图像的高度和宽度在送入 ViT 之前会被调整为 28 的倍数。视觉编码器通过以 14 的步幅将图像切分为补丁来处理图像，生成一组图像特征。
+ 基于 MLP 的视觉-语言合并器：为了解决图像特征长序列带来的效率问题，我们采用一种简单但有效的方法：在将特征序列输入大型语言模型（LLM）之前对其进行压缩。具体来说，我们不是直接使用由ViT提取的原始的补丁嵌入，我们首先将空间上相邻的四个补丁嵌入分为一组。然后将这些分组嵌入串联起来，并通过一个两层的多层感知器（MLP）投影到与在 LLM 中使用的文本嵌入对齐的维度。该方法不仅降低了计算成本，还为动态压缩不同长度的图像特征序列提供了一种灵活的途径。


#chapter("扩散模型", image: image("./orange2.jpg"), l: "multimodal-diffuser")

扩散模型的兴起可以看作是近年来AI生成艺术作品领域取得突破的主要因素。

在图像创作方面，扩散模型已成为内容生成领域的前沿技术。尽管该模型于2015年首次推出，但已取得显著进展，并已成为DALLE和Midjourney等知名模型的核心机制。

#tip(title: "DDPM算法")[Denoising Diffusion Probabilistic Models，去噪扩散概率模型]

== 通俗讲解

=== 扩散

==== 物理学中的类比

想象一下一杯透明的水。如果我们加入少量其他颜色的液体，比如黄色的液体，会发生什么？黄色液体会逐渐均匀地扩散到整个玻璃杯中，最终的混合物会呈现出略带透明的黄色。

#figure(
  image("水滴的扩散过程.svg"),
  caption: [水滴的扩散过程],
)

上面的过程被称为 *正向扩散* ：我们通过添加少量其他液体来改变环境状态。然而，进行 *反向扩散* ——将混合物恢复到其原始状态——是否同样容易？事实证明并非如此。即使在最好的情况下，实现这一点也需要高度复杂的机制。

==== 将类比应用于机器学习

扩散也可以应用于图像。想象一下一张高质量的狗狗照片。我们可以通过逐渐添加随机噪声来轻松地变换这幅图像。结果，像素值会发生变化，使图像中的狗狗变得不那么明显，甚至无法辨认。这个变换过程称为 *正向扩散* 。

#figure(
  image("小狗图像的正向扩散和反向扩散过程.svg"),
  caption: [高清图像的扩散过程],
)

我们也可以考虑反向操作：给定一张噪声图像，目标是重建原始图像。这项任务更具挑战性，因为与大量可能的噪声变化相比，可高度识别的图像状态要少得多。用前面提到的物理类比，这个过程称为 *反向扩散* 。

在本文中，我将通过示意图来解释它的工作原理。

=== 扩散模型的架构

为了更好地理解扩散模型的结构，让我们分别检查两个扩散过程。

==== 正向扩散

如前所述，前向扩散涉及逐步向图像添加噪声。然而，在实践中，这个过程要更加微妙一些。

最常见的方法是从均值为 0 的 *高斯分布* 中为图片中的每个像素采样一个随机值。然后将这个采样值（可以是正值也可以是负值）添加到像素的原始值中。对所有像素重复此操作会得到原始图像的噪声版本。

#figure(
  image("从高斯分布中采样一个点加到原来的像素点上.svg"),
  caption: [从高斯分布中采样一个随机值],
)

#notify[
  所选的高斯分布通常方差较小，这意味着采样值通常较小。因此，每一步只会对图像产生微小的变化。

  图中有49个像素点，那么需要从一个相同的高斯分布中采样49次噪声，然后将49个噪声分别加到49个像素点上。
]

正向扩散是一个迭代过程，其中噪声被多次应用于图像。随着每次迭代，生成的图像与原始图像的差异越来越大。经过数百次迭代（这在实际扩散模型中很常见）后，图像最终变得无法从纯噪声中识别出来。

==== 反向扩散

现在你可能会问：执行所有这些正向扩散变换的目的是什么？答案是，每次迭代生成的图像都用于训练神经网络。

具体来说，假设我们在正向扩散过程中应用了100次连续噪声变换。然后，我们可以在每一步获取图像，并训练神经网络重建上一步的图像。预测图像与实际图像之间的差异使用损失函数计算——例如均方误差 (MSE) ，它衡量两幅图像之间的平均像素差异。

#figure(
  image("预测图像与实际图像之间的差异使用损失函数计算.svg"),
  caption: [预测图像与实际图像之间的差异使用损失函数计算],
)

#notify[
  该模型的目标是检测添加的噪声并重建先前的图像。然后将预测图像与实际图像进行比较以计算损失。

  这个例子展示了扩散模型重构原始图像的过程。同时，扩散模型可以被训练来预测添加到图像中的噪声。在这种情况下，要重构原始图像，只需要从前一次迭代的图像中减去预测的噪声就足够了。

  虽然这两个任务看起来可能相似，但预测添加的噪声比图像重构要简单。
]

#figure(
  image("预测噪声.svg"),
  caption: [将噪声作为预测目标],
)

=== 模型设计

在对扩散技术有了基本的了解之后，有必要探索一些更高级的概念，以更好地理解扩散模型设计。

==== 迭代次数

迭代次数是扩散模型中的关键参数之一：

#notify[
  一方面，使用更多迭代次数意味着相邻步骤中的图像对差异会更小，从而使模型的学习任务更容易。另一方面，更高的迭代次数会增加计算成本。
]

虽然较少的迭代次数可以加快训练速度，但模型可能无法学习步骤之间的平滑过渡，从而导致性能不佳。

通常，迭代次数选择在 50 到 1000 之间。

==== 神经网络架构

最常见的是，U-Net 架构被用作扩散模型的主干。以下是一些原因：

- U-Net保留了输入和输出图像的尺寸，确保在整个逆扩散过程中图像大小保持一致。
- 其瓶颈架构能够在将整幅图像压缩到潜在空间后将其重建。同时，通过残差连接保留关键图像特征。
- U-Net最初设计用于生物医学图像分割，其中像素级精度至关重要，它的优势可以很好地转化为需要精确预测单个像素值的扩散任务。

#figure(
  image("u-net-illustration-correct-scale2.svg"),
  caption: [U-net架构（以最低分辨率为32$times$32像素为例）。每个蓝色方框对应一个多通道特征图。方框顶部标注了通道数量。方框左下角标注了x-y尺寸。白色方框表示复制的特征图。箭头表示不同的操作。],
)

==== 共享网络

乍一看，似乎有必要为扩散过程的每次迭代训练一个单独的神经网络。虽然这种方法可行，并且可以产生高质量的推理结果，但从计算角度来看，它效率极低。例如，如果扩散过程包含1000个时间步，我们就需要训练1000个U-Net模型——这是一项极其耗时且资源密集的任务。

然而，我们可以观察到，*不同迭代中的任务配置本质上是相同的*：在每种情况下，我们都需要重建一个尺寸相同、且经过相似幅度噪声改变的图像。这一重要洞见促成了*在所有迭代中使用单个共享神经网络*的想法。

实际上，这意味着我们使用一个具有共享权重的 U-Net 模型，该模型基于来自不同扩散步骤的图像对进行训练。在推理过程中，含噪图像会多次通过同一个经过训练的 U-Net 模型，逐步进行优化，直到生成高质量的图像。

#figure(
  image("共享模型.svg"),
  caption: [共享模型],
)

虽然由于仅使用单一模型，生成质量可能会略有下降，但训练速度的提升却非常显著。

== 扩散模型理论简介

=== 概述

#figure(
  image("正向扩散和逆向扩散.png"),
  caption: [正向扩散和逆向扩散],
)

扩散模型的训练可以分为两部分：

+ 正向扩散过程 #sym.arrow 给图像添加噪声。
+ 反向扩散过程 #sym.arrow 从图像中去除噪声。

=== 正向扩散过程

#figure(
  image("正向扩散公式图解.svg"),
  caption: [正向扩散公式图解],
)

#notify(title: "方差计划")[
  假设$beta_#text[start] = 0.0002, beta_#text[end] = 0.04$，则第1步添加的高斯噪声的方差是0.0002, 第2步添加的高斯噪声的方差是$0.0002 + (0.04 - 0.0002) / 1000$。
]

前向扩散过程逐步将高斯噪声添加到输入图像 $x_0$ 中，总共会有 $T$ 步。该过程将产生一系列带噪声的图像样本 $x_1,x_2,...,x_T$ 。

当 $T arrow infinity$ 时，最终结果将变成完全噪声图像，就像从*各向同性*的高斯分布中采样出来的噪声一样。

首先，如果 $z tilde cal(N) (mu, sigma^2)$ 的话，那么正态分布可以写成如下公式：

$
  z=mu+sigma epsilon space "其中" epsilon tilde cal(N) (0,1)
$

利用这个技巧，我们可以将采样图像 $x_t$ 表示如下：

$
  x_t = sqrt(1-beta_t) x_(t-1) + sqrt(beta_t) epsilon_(t-1)
$ <step-by-step-add-noise>

#codly(header: [@step-by-step-add-noise 的代码实现，逐步向图片中添加噪声])
```python
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

image = plt.imread("./flower.png")
print(image.shape)

preprocess = transforms.ToTensor()
x = preprocess(image)
print(image.shape)

def reverse_to_img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)

# 最大时间步
T = 1000
# 方差计划的起始值
beta_start = 0.0001
# 方差计划的结束值
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
print(betas)

imgs = []

for t in range(T):
    if t % 100 == 0:
        img = reverse_to_img(x)
        imgs.append(img)

    beta = betas[t]
    eps = torch.randn_like(x) # 生成和x形状相同的噪声
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

# 2行5列的方式显示10张图片
plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Noise: {i * 100}")
    plt.axis("off")

plt.show()
```

#danger(title: "一步一步的添加噪声太麻烦了！")[
  根据上面的公式，如果想要从原始图片 $x_0$ 得到添加了 500 步噪声的图片 $x_500$ 需要迭代 500 次！
]

但我们不需要设计一种算法来迭代地向图像中添加噪声，而是可以使用闭式公式（解析解）在特定的时间步长 $t$ 直接对噪声图像进行采样。

给定原始图片 $x_0$ 和时间步 $t$ 可以直接得到添加了 $t$ 步噪声的图像 $x_t$ 。公式如下：

#tip(title: "给定原始图片" + $x_0$ + "和时间步" + $t$ + "直接采样出" + $x_t$ + "的公式")[
  $
    x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon
  $ <closed-form-add-noise>
  其中：
  - $overline(alpha)_t=alpha_t alpha_(t-1) dots alpha_1$
  - $alpha_t = 1 - beta_t$
  - $epsilon tilde cal(N) (0,1)$
]

现在我们可以使用此公式在任何时间步直接对 $x_t$ 进行采样，这使得前向扩散过程更快。

#codly(header: [@closed-form-add-noise 的实现，使用闭式解添加噪声])
```python
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

image = plt.imread("./flower.png")
print(image.shape)

preprocess = transforms.ToTensor()
x = preprocess(image)
print(image.shape)


def reverse_to_img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)


# 最大时间步
T = 1000
# 方差计划的起始值
beta_start = 0.0001
# 方差计划的结束值
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
print(betas)

# 一步得到x_t,使用闭式解（closed form）
def add_noise(x_0, t, betas):
    T = len(betas)

    alphas = 1 - betas  # [α_1, α_2, ...]
    # cumprod功能：[1,2,3,4] --> [1,2,6,24]
    alpha_bars = torch.cumprod(alphas, dim=0)
    t_idx = t - 1
    alpha_bar = alpha_bars[t_idx]  # alpha_bar_t

    eps = torch.randn_like(x_0)
    # 闭式解
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar)*eps

    return x_t

t = 100
x_t = add_noise(x, t, betas)

img = reverse_to_img(x_t)
plt.imshow(img)
plt.title(f"Noise: {t}")
plt.axis("off")
plt.show()
```

=== 反向扩散过程

反向扩散过程是从一张完全的高斯噪声图片中，逐步去除噪声，来生成一张图片。这个逆向过程很难，相当于一滴墨水在水里面扩散开来（正向扩散），想要将变黑的水逆向为原来的墨水刚滴入水中时的状态。这几乎不可能做到，所以我们用神经网络来解决这个问题。下面是训练神经网络的原理。

#figure(
  image("预测噪声.svg"),
  caption: [以噪声为预测目标],
)

因此最终的训练目标如下：

#linebreak()
#linebreak()
#linebreak()
$
  markrect(x_t = sqrt(overline(alpha)_t)x_0 + sqrt(1-overline(alpha)_t) epsilon, color: #red, tag: #<p>)
  #linebreak()
  #linebreak()
  #linebreak()
  #linebreak()
  #linebreak()
  #linebreak()
  #linebreak()
  cal(L)_(#text[simple]) = EE_(t,x_0,epsilon) [ ||epsilon - epsilon_theta (markrect(x_t, color: #red, tag: #<xt>),t)||^2 ]
  #annot(<p>, pos: top + left, dy: -1.5em, leader-connect: "elbow")[时间步t的图片]
$

#annot-cetz(
  (<p>, <xt>),
  cetz,
  {
    import cetz.draw: *
    set-style(mark: (end: "straight"))
    bezier-through("p.south", (rel: (x: 1, y: -.5)), "xt.north", stroke: red)
  },
)

=== U-Net模型

==== 数据集

在每个Epoch：

+ 将为每个训练样本（图像）选择一个随机时间步长 $t$ 。
+ 对每幅图像添加高斯噪声（对应于 $t$ ）。
+ 将时间步转换为嵌入（向量）。

#figure(
  image("unet训练步骤.png"),
  caption: [训练unet],
)

==== 训练

#figure(
  algo(
    line-numbers: true,
    strong-keywords: false,
    comment-prefix: [#sym.triangle.stroked.r ],
    comment-styles: (fill: rgb(100%, 0%, 0%)),
  )[
    *Repeat*#i\
    $bold("x")_0 tilde q(bold("x")_0)$ #comment[从数据集中抽取一张图片]\
    $t tilde "Uniform"({1,...,T})$ #comment[从均匀分布中采样一个时间步]\
    $epsilon tilde cal(N)(0, bold("I"))$ #comment[从正态分布中采样一个噪声]\
    使用梯度下降法，梯度为\
    $
      nabla_theta ||epsilon-epsilon_theta (sqrt(overline(alpha)_t)bold("x")_0 + sqrt(1-overline(alpha)_t)epsilon,t)||^2
    $#d\
    *Until*收敛
  ],
  caption: [U-net训练算法],
)

官方的训练算法如上，下图是一个训练步骤的示意图：

#figure(
  image("扩散模型训练步骤示意图.svg"),
  caption: [扩散模型训练步骤],
)

==== 反向扩散

#figure(
  algo(
    line-numbers: true,
    strong-keywords: false,
    comment-prefix: [#sym.triangle.stroked.r ],
    comment-styles: (fill: rgb(100%, 0%, 0%)),
  )[
    $bold("x")_T tilde cal(N)(0, bold("I"))$ #comment[从高斯分布中采样一张纯噪声图片]\
    *for* $t=T,...,1$ *do* #i\
    $bold("z") tilde cal(N) (0,bold("I"))$ *if* $t>1$, *else* $bold("z")=0$\
    $bold("x")_(t-1) = 1/sqrt(alpha_t)(bold("x")_t - (1-alpha_t)/sqrt(1-overline(alpha)_t) epsilon_theta (bold(x), t)) + sigma_t bold("z")$ #comment[$sigma_t$一般取$sqrt(beta_t)$] #d\
    *end for*\
    *return* $bold("x")_0$
  ],
  caption: [采样算法（去噪算法，反向扩散算法）],
)

我们可以使用上述算法从噪声中生成图像。下图是它的说明：

#figure(
  image("扩散模型采样示意图.svg"),
  caption: [扩散模型的采样过程],
)

请注意，在最后一步，我们只是输出学习到的平均值 $mu_theta (x_1,1)$ ，而不向其中添加噪声。

== 代码实现

=== 扩散过程相关代码

我们需要针对扩散的时间步来制定一个方差的调度计划。每个时间步，都在前面一个时间步的图像中添加一个高斯噪声。在开始向图像中添加噪声的时候，需要高斯噪声的方差小一些，不至于一开始就把图像变得很模糊。而到了后面，添加噪声就可以大胆一些了，反正已经模糊了。所以后面添加的高斯噪声需要方差大一些。

#tip(title: "高斯噪声的方差")[
  公式中的 $beta_t$ 就是第 $t$ 个时间步要添加的噪声的方差。
]

在代码中如下

```python
# self.betas: 方差计划调度表
self.betas = torch.linspace(
    beta_start, # beta_start: 起始$beta$值，论文中等于0.0001
    beta_end, # beta_end: 结束$beta$值，论文中等于0.02
    num_timesteps, # num_timesteps: 时间步的数量：1000
    device=device
)
```

而由于 $alpha_t=1-beta_t$ ，所以有如下代码

```python
self.alphas = 1 - self.betas
```

这样就计算出了每个时间步的 $α_t$ 。

而由于 $overline(alpha)_t=product_(s=1)^t alpha_s$ ，所以使用 `torch.cumprod` 来计算

```python
self.alpha_bars = torch.cumprod(self.alphas, dim=0)
```

由于我们已经使用重参数技巧来给图像添加噪声，也就是通过解析解可以直接得到添加了 $t$ 个时间步的噪声的图像。所以前向扩散过程就用了这个公式。

#tip(title: "给定原始图片" + $x_0$ + "和时间步" + $t$ + "直接采样出" + $x_t$ + "的公式")[
  $
    x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon
  $
  其中：
  - $overline(alpha)_t=alpha_t alpha_(t-1) dots alpha_1 = product_(s=1)^t alpha_s$
  - $alpha_t = 1 - beta_t$
  - $epsilon tilde cal(N) (0, bold("I"))$
]

公式和代码基本是对应的。

```python
def add_noise(self, x_0, t):
    # $x_0$是原始图片，$t$是时间步
    # T：时间步的数量
    T = self.num_timesteps
    # 确保 $1 <= t <= T$
    assert (t >= 1).all() and (t <= T).all()

    t_idx = t - 1  # alpha_bars[0] is for t=1
    # $overline(alpha)_t$
    alpha_bar = self.alpha_bars[t_idx]  # (N,)
    N = alpha_bar.size(0)
    alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (N, 1, 1, 1)
    # $epsilon tilde cal(N)(0, bold("I"))$
    # 噪声的形状或者点数等于图片的像素点
    noise = torch.randn_like(x_0, device=self.device)
    # $x_t = sqrt(overline(alpha)_t)x_0+sqrt(1-overline(alpha)_t) epsilon$
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    # 返回第$t$步的图像$x_t$和添加的噪声noise，后面作为训练数据
    return x_t, noise
```

接下来编写反向去噪过程的代码，也就是从 $x_t$ 预测 $x_(t-1)$ 。

#figure(
  image("预测噪声.svg"),
  caption: [以噪声为预测目标],
)

给定了时间步和添加噪声的图像，模型可以预测出这幅图片中的噪声有多少，那么我们从 $x_t$ 中将预测出的噪声减掉，就可以去噪了！

```python
def denoise(self, model, x, t):
    # x是图片，模型会认为x是添加了t步噪声的图片
    # t是时间步
	T = self.num_timesteps
	assert (t >= 1).all() and (t <= T).all()

	t_idx = t - 1  # alphas[0] is for t=1
	alpha = self.alphas[t_idx] # $alpha_t$
	alpha_bar = self.alpha_bars[t_idx] # $overline(alpha)_t$
	alpha_bar_prev = self.alpha_bars[t_idx-1] # $overline(alpha)_(t-1)$

	N = alpha.size(0)
	alpha = alpha.view(N, 1, 1, 1)
	alpha_bar = alpha_bar.view(N, 1, 1, 1)
	alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

	model.eval()
	with torch.no_grad():
		eps = model(x, t) # 根据时间步和图像xₜ预测噪声
	model.train()
	# 公式中的z
	noise = torch.randn_like(x, device=self.device)
	# 如果时间步为1的话，我们就不再添加$σ_z$噪声
	noise[t == 1] = 0  # no noise at t=1
	# 均值$mu_theta (x_t, t)=1/sqrt(alpha_t) (bold("x")_t - (1-alpha_t)/sqrt(1-overline(alpha)_t) epsilon_theta (bold("x")_t,t))$
	mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps)
	   / torch.sqrt(alpha)
	# 标准差$sigma_t = sqrt(((1-alpha_t)(1-overline(alpha)_(t-1)))/(1-overline(alpha)_t))$
	std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) \
      / (1-alpha_bar))
	# 返回$x_(t-1)$
	return mu + noise * std
```

#figure(
  algo(
    line-numbers: true,
    strong-keywords: false,
    comment-prefix: [#sym.triangle.stroked.r ],
    comment-styles: (fill: rgb(100%, 0%, 0%)),
  )[
    $bold("x")_T tilde cal(N)(0, bold("I"))$ #comment[从高斯分布中采样一张纯噪声图片]\
    *for* $t=T,...,1$ *do* #i\
    $bold("z") tilde cal(N) (0,bold("I"))$ *if* $t>1$, *else* $bold("z")=0$\
    $bold("x")_(t-1) = 1/sqrt(alpha_t)(bold("x")_t - (1-alpha_t)/sqrt(1-overline(alpha)_t) epsilon_theta (bold(x), t)) + sigma_t bold("z")$ #comment[在DDPM论文中$sigma_t=sqrt(((1-alpha_t)(1-overline(alpha)_(t-1)))/(1-overline(alpha)_t))$] #d\
    *end for*\
    *return* $bold("x")_0$
  ],
  caption: [采样算法（去噪算法，反向扩散算法）],
)

上面的代码实现了伪代码中的第 4 步。

=== U-Net神经网络

U-Net最初是为医学图像的语义分割而开发的模型。语义分割是为图像中的每个像素分配特定的类别标签的任务，如图所示。

#figure(
  image("unet-tiger.png"),
  caption: [U-Net执行语义分割],
)

图中U-Net的输入是形状为`(C, H, W)`的图像数据，其中，`C`是输入图像的通道数（RGB图像为3），`H`是图像的高度，`W`是图像的宽度。输出是形状为`(D, H, W)`的张量，其中`D`是要分类的类别数。模型对每个像素输出`D`个类别的概率分布。而在扩散模型中，输出被设置为`(C, H, W)`（D=C），即输入和输出的通道数都被设置为`C`。

U-Net的名称源于网络结构的形状，因为它与字母"U"的形状相似。

#figure(
  image("myunet.svg"),
  caption: [本章要实现的U-Net],
)

U-Net的处理过程分为前半部分的"缩小阶段"和后半部分的"扩大阶段"。在前半部分的缩小阶段中，在卷积层进行处理的同时，特征图会逐渐缩小。缩小特征图的层称为"下采样层"。在后半部分的扩大阶段中，在卷积层进行特征抽取的同时，特征图会逐渐扩大，这与前半部分正好相反。扩大特征图的层称为"上采样层"。

U-Net的重要特征是跳跃连接（skip connection）。这是一种在网络的缩小阶段和扩大阶段之间直接传递特征图的机制。这种跳跃连接使得U-Net能够捕捉对象整体的特征，同时使用更精细的空间位置信息进行处理。

如上图所示，U-Net包含两个缩小阶段和两个扩大阶段。每个阶段由两个卷积层进行处理。为了实现这个U-Net，首先要实现一个名为ConvBlock的类。如下图所示，该类分别对卷积层，批量归一化层和ReLU函数的处理执行了两次。

#figure(
  image("ConvBlock类执行的处理.svg"),
  caption: [ConvBlock类执行的处理],
)

然后是ConvBlock类的代码，如下所示。

```python
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
            nn.BatchNorm2d(out_ch)
            nn.ReLU()
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
            nn.BatchNorm2d(out_ch)
            nn.ReLU()
        )

    def forward(self, x):
        return self.convs(x)
```

这段代码使用`nn.Sequential()`串联了多个层。这样做的目的是使数据按顺序逐层通过。使用ConvBlock类，我们可以实现上图所示的UNet类。代码如下所示。

```python
class UNet(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()

        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.bot1 = ConvBlock(128, 256)
        self.up2 = ConvBlock(128 + 256, 128)
        self.up1 = ConvBlock(128 + 64, 64)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)
        x = self.bot1(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        x = self.out(x)
        return x
```

这段代码使用最大池化（`nn.MaxPool2d`）来缩小数据。这会使张量的大小缩小1/2。而在扩大数据的处理中，代码使用了双线性插值的上采样（`nn.Upsample`）。这会使张量的大小扩大２倍。

#tip(title: "双线性插值")[
  双线性插值是一种用于扩大图像大小的技术。双线性插值可以创建新的像素，其值基于原始图像中像素的值计算得出。这样就能平滑地扩大图像。
]

进行U-Net的跳跃连接的代码是`torch.cat([x, x1], dim=1)`。`torch.cat()`函数是连接张量的函数，其中`dim=1`指定了连接的维度。在本例中，
如果`x`的形状是`(N, C, H, W)`，`x1`的形状是`(N, I, H, W)`，那么连接后的张量的形状就是`(N, C + I, H, W)`。

上面我们实现了处理$x_t$的U-Net。剩下的任务是将时刻$t$引入U-Net中。需要注意的是这里的$t$是整数。在神经网络中，将整数变换为向量可以提高训练和预测的效率。扩散模型在对时刻$t$进行编码时，常常使用正弦位置编码。

正弦位置编码因在论文"Attention Is All You Need"的Transformer模型中使用而闻名。正弦位置编码的意思是"使用正弦波（sin波）对位置信息进行编码"。位置信息是指序列数据中每个元素出现的位置。例如，在自然语言处理任务中，单词出现的位置就相当于位置信息。扩散模型的时刻$t$也是位置信息。

正弦位置编码可以将整数$t$变换为向量$bold(v)$。如果变换后的向量$bold(v)$的维度为$D$，则其第$i$个元素的数学式如下所示。

$
  bold("v")_i = cases(
    sin (t/(10000^(i/D))) ",  " i"为偶数时", ,
    cos (t/(10000^(i/D))) ",  " i"为奇数时"
  )
$

上面的正弦位置编码不以绝对值来编码位置信息，而是通过具有循环特性的sin函数和cos函数来编码位置信息。这样一来，位置信息中的相对差异和循环模式就能清晰地显示出来，使模型能够更有效地学习序列数据中的相对位置关系。

下面我们来实现正弦位置编码。首先，我们将实现对单个时刻数据（整数）进行编码的函数`_pos_encoding()`。然后，我们将使用它来实现一个名为`pos_encoding()`的处理批量数据的的数。以下是`_pos_encoding()`的代码。

```python
import torch

def _pos_encoding(t, output_dim, device="cpu"):
    D = output_dim
    v = torch.zero(D, device=device)
    i = torch.arange(0, D, device=device) ①
    div_term = 10000 ** (i / D)

    v[0::2] = torch.sin(t / div_term[0::2]) ②
    v[1::2] = torch.cos(t / div_term[1::2])
    return v
```

这段代码基于输入时刻（`t`）和输出维度（`output_dim`）进行位置编码。代码①处通过`torch.arange(D)`创建张量`[0, 1, ..., D]`。代码②处使用切片语法`v[0::2]`来指定"从`v`的第0个元素开始，每次跳过一个元素后的所有元素"。也就是说，指定向量`v`的偶数索引`(0, 2, 4, ...)`对应的元素，并对这些元素进行正弦编码。

接下来，我们将实现用于处理批量数据的正弦位置编码。为了易于理解，这里使用`for`语句来实现，代码如下所示。

```python
def pos_encoding(ts, output_dim, device="cpu"):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)
    return v
```

参数`ts`是张量。对于该张量的每个元素，调用刚刚实现的`_pos_encoding`函数。这样我们就完成了正弦位置编码的实现。

最后将正弦位置编码嵌入到U-Net中。这里我们将正弦位置编码嵌入到之前实现的ConvBlock类中。ConvBlock是拥有两个卷积层的处理单元。我们使用ConvBlock实现了U-Net，如图所示。

#figure(
  image("我们自己实现的U-Net.svg"),
  caption: [我们实现的U-Net],
)

这里要向5个ConvBlock添加正弦位置编码信息。

#figure(
  image("加入正弦位置编码的U-Net.svg"),
  caption: [加入正弦位置编码的U-Net],
)

如图所示，新的ConvBlock中也将接收正弦位置编码信息v作为输入。然后在这个新的ConvBlock内部执行下图所示的处理。

#figure(
  image("新的ConvBlock类.svg"),
  caption: [新的ConvBlock类],
)

图中的x的形状为`(N, C, H, W)`，v的形状为`(N, D)`。v被MLP（多层感知机，这里指由全连接层组成的神经网络）变换为`(N, C)`的形状，然后再被变换为`(N, C, 1, 1)`的形状。通过这个变形，广播函数得以应用于随后的加法运算中。新的ConvBlock的实现如下所示。

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y
```

参数`time_embed_dim`是通过正弦位置编码变换后的向量的维度。名为`self.mlp`的全连接层可以将维度为`time_embed_dim`的向量变换为维度为`in_ch`的向量。

最后使用这个新的`ConvBlock`类来实现`UNet`类。代码如下所示。

```python
class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x
```

这样我们就完成了扩散模型使用的神经网络的实现。

== 完整代码


```python
import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


img_size = 28
batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-3
device = "cuda"


def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            i += 1
    plt.show()

# 位置编码部分
def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(timesteps, output_dim, device='cpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y

class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x


class Diffuser:
    def __init__(self,
        num_timesteps=1000, # 时间步T=1000
        beta_start=0.0001, # 方差的起始值$beta_0 = 0.0001$
        beta_end=0.02, # 方差的最终值$beta_T = 0.02$
        device="cpu"
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        # 方差调度计划
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            device=device
        )
        # $alpha_t = 1 - beta_t$
        self.alphas = 1 - self.betas
        # $overline(alpha)_t=alpha_t alpha_(t-1) dots.c alpha_1$
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 \
            + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t):
        """去除一步噪声，x为时间步t的带噪声图片$x_t$"""
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) \
           / torch.sqrt(alpha)
        std = torch.sqrt(
            (1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar)
        )
        # $x_(t-1)$
        return mu + noise * std

    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 28, 28)):
        """从纯噪声图片$x_1000$反向扩散出$x_0$"""
        batch_size = x_shape[0]
        # 采样一张白噪声图片$x_1000$出来
        x = torch.randn(x_shape, device=self.device)
        # for t = T, T-1, ..., 0
        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor(
                [i] * batch_size,
                device=self.device,
                dtype=torch.long
            )
            # 一步去噪，$x_t -> x_(t-1)$
            x = self.denoise(model, x, t)

        images = [
            self.reverse_to_img(x[i])
            for i in range(batch_size)
        ]
        return images


preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(
    root="./../datasets",
    download=True,
    transform=preprocess
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

diffuser = Diffuser(num_timesteps, device=device)
model = UNet()
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    # 每个 epoch 都生成采样的图像 =======================
    images = diffuser.sample(model)
    show_images(images)
    # ================================================

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        # 随机取一个时间步
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)
        # x_noisy是$x_t$，noise是添加的真正的噪声
        x_noisy, noise = diffuser.add_noise(x, t)
        # 模型根据$x_t$和时间步$t$，预测给$x_t$添加的噪声
        noise_pred = model(x_noisy, t)
        # 添加的真实噪声和预测噪声之间进行均方误差计算
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch {epoch} | Loss: {loss_avg}")

# 画出损失
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 从完全噪声的图片反向扩散
images = diffuser.sample(model)
show_images(images)
```

#chapter("扩散模型背后的数学理论", image: image("./orange2.jpg"), l: "multimodal-diffuser-math")

由于我们正在创建一个生成模型，我们希望找到真实图像的分布。

给定一个目标和未知的概率分布，可以构造一条马尔可夫链，用来近似该目标概率分布。执行近似时包含的步骤越多，样本的分布就越接近真实分布。

当当前决策仅取决于你所在的位置而非起点时，过程就是马尔可夫过程。存在概率转移矩阵，我们可以构建一个链结构，称为马尔可夫链。

#figure(
  $
             p(x) & prop e^(-f(x)) arrow.stroked log p(x) = -f(x) + "常数" space space space & (1) \
    therefore x_t & = x_(t-1) - eta_t nabla f(x_(t-1))                                       & (2) \
          x_(t+1) & = x_t - epsilon/2 nabla f(x_t) + sqrt(epsilon) cal(N)(0,I)               & (3) \
  $,
  caption: [郎之万动力学],
) <Langevin>

郎之万动力学是目前最流行的MCMC（马尔可夫蒙特卡洛采样）方法之一。如果我们有一个概率分布 $p(x)$ 且是马尔可夫分布，则它属于指数分布类。这在@Langevin(1)中给出，其中$f(x)$是某个使得该方程成立的函数。假设我们没有直接从该分布中采样的方法。假设我们只能知道函数在每一点的梯度$nabla f(x)$。如果是这样，我们可以应用梯度下降来求出分布的模态。这由@Langevin(2)给出，其中$eta$是时间步$t$的学习率。如果我们能找到分布的模态，那么我们就能找到密度中具有峰值的区域。由于在我们的训练数据中，会有很多"好"图像，这意味着去到峰值时，会得到这些好图像。郎之万采样是对该梯度下降算法的有趣修改，由@Langevin(3)给出。这里我们在梯度下降公式的每个步骤中添加一个噪声项。

一个问题是，如果想加快过程，可能会选择$epsilon$的值过高，这可能导致上述方程(3)不稳定并发散。所以，如果我们能提供某种形式的保证，保证更新规则会收敛，那就更好了。



#figure(
  image("正向扩散和逆向扩散.png"),
  caption: [正向扩散和逆向扩散],
)

扩散模型的训练可以分为两部分：

1. 正向扩散过程 #sym.arrow.long 给图像添加噪声。
2. 反向扩散过程 #sym.arrow.long 从图像中去除噪声。

== 正向扩散过程

#figure(
  image("正向扩散公式图解.svg"),
  caption: [正向扩散公式图解],
)

前向扩散过程逐步将高斯噪声添加到输入图像 $x_0$ 中，总共会有 $T$ 步。该过程将产生一系列带噪声的图像样本 $x_1, dots, x_T$ 。

当 $T arrow infinity$ 时，最终结果将变成完全噪声图像，就像从*各向同性*的高斯分布中采样出来的噪声一样。

但我们不需要设计一种算法来迭代地向图像中添加噪声，而是可以使用闭式公式（解析解）在特定的时间步长 $t$ 直接对噪声图像进行采样。

*前向扩散的闭式公式（解析解）*

可以使用*重参数化技巧*推导出解析形式的采样公式。

#tip(title: [重参数技巧])[
  首先，如果 $z tilde cal(N) (mu,sigma^2)$ 的话，那么有下面的结论

  $
    z = mu + sigma epsilon space "其中" epsilon tilde cal(N)(0, 1)
  $

  两个正态分布（独立）随机变量的和也是正态分布，也就是如果$X tilde cal(N)(mu_X, sigma_X^2)$和$Y tilde cal(N)(mu_Y, sigma_Y^2)$，那么对于$Z = X + Y$有$Z tilde cal(N)(mu_X + mu_Y, sigma_X^2+sigma_Y^2)$，也就是

  $
    z = mu_X + mu_Y + sqrt(sigma_X^2+sigma_Y^2)epsilon space "其中" epsilon tilde cal(N)(0, 1)
  $
]

利用这个技巧，我们可以将采样图像 $x_t$ 表示如下：

$
  x_t = sqrt(1-beta_t)x_(t-1) + sqrt(beta_t) epsilon_(t-1)
$

*然后我们可以递归地展开它来得到闭合形式的公式*：

我们先来写一些前置结论

$
  epsilon_0, ..., epsilon_(t-2), epsilon_(t-1) tilde cal(N) (0, I) \
  overline(epsilon)_0, ..., overline(epsilon)_(t-2), overline(epsilon)_(t-1) tilde cal(N) (0, I) \
  epsilon tilde cal(N)(0,I) \
  alpha_t = 1 - beta_t \
  overline(alpha)_t = product_(i=1)^t alpha_i
$

然后我们开始推导
#pagebreak()

$
  x_t &= sqrt(1-beta_t)x_(t-1) + sqrt(beta)_t colred(epsilon_(t-1)) \
  &= sqrt(alpha_t) markrect(x_(t-1), color: #red, tag: #<xtminus1>) + sqrt(1-alpha_t) colred(epsilon_(t-1)) \
  #linebreak()
  #linebreak()
  #linebreak()
  #linebreak()
  &= sqrt(alpha_t)markrect((sqrt(alpha_(t-1))x_(t-2) + sqrt(1-alpha_(t-1)) colred(epsilon_(t-2))), color: #red, tag: #<xtminus2>) + sqrt(1-alpha_t) colred(epsilon_(t-1)) \
  &= sqrt(alpha_t alpha_(t-1))x_(t-2) + markrect(sqrt(alpha_t (1 - alpha_(t-1))) colred(epsilon_(t-2)) + sqrt(1-alpha_t) colred(epsilon_(t-1)), color: #red, tag: #<xtminus3>) \
  &= sqrt(alpha_t alpha_(t-1))x_(t-2) + 0 + sqrt(alpha_t (1 - alpha_(t-1)))epsilon_(t-2) + 0 + sqrt(1-alpha_t) epsilon_(t-1) \
  &= sqrt(alpha_t alpha_(t-1))x_(t-2) + cal(N)(0,alpha_t (1 - alpha_(t-1))bold(I)) + cal(N)(0,(1-alpha_t)bold(I)) \
  &= sqrt(alpha_t alpha_(t-1))x_(t-2) + cal(N)(0,(1 - alpha_t alpha_(t-1))bold(I)) \
  &= sqrt(alpha_t alpha_(t-1))x_(t-2) + markrect(sqrt(1-alpha_t alpha_(t-1)) colred(overline(epsilon)_(t-2)), color: #red, tag: #<xtminus4>) \
  & space dots.v \
  &= sqrt(alpha_t alpha_(t-1) dots.c alpha_1)x_0 + sqrt(1-alpha_t alpha_(t-1) dots.c alpha_1)epsilon \
  &= sqrt(overline(alpha)_t)x_0 + sqrt(1-overline(alpha)_t)epsilon
$
#annot(<xtminus2>, pos: top + right)[一步加噪声公式]
#annot-cetz(
  (<xtminus1>, <xtminus2>, <xtminus3>, <xtminus4>),
  cetz,
  {
    import cetz.draw: *
    set-style(mark: (end: "straight"), stroke: (dash: "dashed"))
    bezier-through("xtminus1.south", (rel: (x: 1, y: -.5)), "xtminus2.north", stroke: red)
    bezier-through("xtminus3.south-east", (rel: (x: 2.5, y: -0.5)), "xtminus4.north-east", stroke: red)
  },
)

#tip[
  所有 $epsilon$ 都是 i.i.d.（独立同分布）标准正态随机变量。

  使用不同的符号和下标来区分它们非常重要，因为它们是独立的，并且在采样后它们的值可能会有所不同。
]

重复这些步骤将为我们提供以下仅取决于输入图像 $x_0$ 的公式：

$
  x_t = sqrt(overline(alpha)_t)x_(0)+sqrt(1-overline(alpha)_t) epsilon
$

现在我们可以使用此公式在任何时间步骤直接对 $x_t$ 进行采样，这使得前向过程更快。

== 逆向扩散过程

前向过程的联合概率分布是

$
  q(x_0 x_1 dots.c x_T) = q(x_0)q(x_1|x_0) dots.c q(x_T|x_(T-1))
$

反向过程的联合概率分布是

$
  q(x_0 x_1 dots.c x_T) = q(x_T)q(x_(T-1)|x_T) dots.c q(x_0|x_1)
$

联合概率的乘法公式是

$
  q(x_t,x_(t-1)) = q(x_(t)|x_(t-1))q(x_(t-1)) = q(x_(t-1)|x_(t))q(x_(t))
$

贝叶斯公式

$
  q(x_(t-1)|x_t) = (q(x_t|x_(t-1)) q(x_(t-1))) / q(x_t)
$

贝叶斯推断

$
  q(x_(t-1)|x_t) prop q(x_t|x_(t-1)) q(x_(t-1))
$

贝叶斯推断为什么要忽略掉分母 $q(x_t)$ 呢？因为我们要计算的是，在 *固定* $x_t$ 的情况下，求 $x_(t-1)$ 的概率，所以 $q(x_t)$ 是个常数。可以忽略掉。

现在的问题是，$q(x_t|x_(t-1))$ 我们已经知道公式了，但 $q(x_(t-1))$ 我们不知道怎么计算。

#linebreak()
#linebreak()
$
  q(x_(t-1)|x_t) prop mark(q(x_t|x_(t-1)), color: #green, tag: #<qcond1>) mark(q(x_(t-1)), color: #red, tag: #<qcond2>)
  #annot(<qcond1>, pos: top + left, dy: -1.5em, leader-connect: "elbow")[已经知道公式了]
  #annot(<qcond2>, pos: top + right, dy: -1.5em, leader-connect: "elbow")[#emoji.zombie 算不出来了]
$

#figure(
  image("去噪的解析解无法计算.svg"),
  caption: [去噪的解析解无法计算],
)

#theorem(name: "Feller")[
  如果前向过程的马尔可夫链的转移概率分布$q(x_t|x_(t-1))$是高斯分布，且$beta_t$充分小，那么反向过程的马尔可夫链的转移概率分布$q(x_(t-1)|x_t)$也是高斯分布。
]

#figure(
  image("马尔可夫逆向1.svg"),
  caption: [通过贝叶斯定理可以对反向分布$q(x_(t-1)|x_t)$进行推断。右图上的红色曲线显示了用3个高斯分布混合表示的边缘分布$q(x_(t-1))$，而左图显示了以$x_(t-1)$为中心的高斯前向噪声过程$q(x_t|x_(t-1))$作为$x_t$的分布。通过将它们相乘并进行归一化，我们得到了蓝色曲线所示的对于特定选择的$x_t$的分布$q(x_(t-1)|x_t)$。因为左图的分布相对较宽，对应着较大的方差$beta_t$，所以分布$q(x_(t-1)|x_t)$具有复杂的多峰结构。],
)

#figure(
  image("马尔可夫逆向2.svg"),
  caption: [左图中的高斯分布$q(x_t|x_(t-1))$的方差$beta_t$要小的多。我们可以看到右图中相应的分布$q(x_t|x_(t-1))$（蓝色）接近高斯分布，具有与$q(x_t|x_(t-1))$类似的方差],
)

我们可以将边缘分布$q(x_(t-1))$写成以下形式

$
  q(x_(t-1)) = integral q(x_(t-1)|bold(upright(x))_0)q(bold(upright(x))_0) upright(d) bold(upright(x))_0
$

其中 $q(x_(t-1)|bold(upright(x))_0)$ 可以由前向扩散的闭式解搞定。然而，上式的分布是难以处理的。因为我们必须对未知的数据密度$q(bold(upright(x))_0)$进行积分。

如果我们使用训练数据集的样本来近似积分，我们得到一个复杂的分布，表示为高斯混合分布。

#tip[无论多么复杂的概率分布，都可以由多个高斯分布加权相加来拟合。概率分布的万能逼近定理。]

#figure(
  image("gaussian_mixture_model.svg"),
  caption: [高斯混合模型（GMM）],
)

相反，我们考虑条件分布的反向分布，条件是数据向量$bold(upright(x))_0$，定义为$q(x_(t-1)|x_t,bold(upright(x))_0)$，我们很快将看到它实际上是是一个简单的高斯分布。 直观上讲，这是合理的，因为对于给定的噪声图像，很难猜测是哪个更低噪声的图像产生了它，而如果我们还知道起始图像，那么问题就变得容易得多。

我们可以使用贝叶斯定理计算这个条件分布：

$
  q(x_(t-1)|x_t,bold(upright(x))_0) = (q(x_t|x_(t-1),bold(upright(x))_0)q(x_(t-1)|bold(upright(x))_0)) / q(x_t|bold(upright(x))_0)
$

现在我们利用前向过程的马尔可夫性质来写

$
  q(x_t|x_(t-1),bold(upright(x))_0) = q(x_t|x_(t-1))
$

#corollary[
  第$t$步到第$t-1$步在第$0$步的条件下的转移概率分布$q(x_(t-1)|x_t,bold(upright(x))_0)$是以下高斯分布：
  $
    q(x_(t-1)|x_t,bold(upright(x))_0) = cal(N)(x_(t-1);tilde(bold(mu))_t (x_t, bold(upright(x))_0),tilde(beta)_t bold(I))
  $
  其均值是
  $
    tilde(bold(mu))_t (x_t, bold(upright(x))_0) = (sqrt(alpha_t)(1-overline(alpha)_(t-1))) / (1 - overline(alpha)_t) x_t + (sqrt(overline(alpha)_(t-1))(1-alpha_t)) / (1 - overline(alpha)_t) bold(upright(x))_0
  $
  方差的系数是
  $
    tilde(beta)_t = (1-overline(alpha)_(t-1)) / (1-overline(alpha)_t) beta_t
  $
]

*证明* 使用贝叶斯定理，利用前向过程的转移概率是高斯分布的性质，可以推导出反向过程的条件转移概率分布也是高斯分布，并得到其概率密度函数。

$
  & q(x_(t-1)|x_t,bold(upright(x))_0) \
  & = (q(x_(t-1)|bold(upright(x))_0)q(x_t|x_(t-1),bold(upright(x))_0)) / q(x_t|bold(upright(x))_0) \
  & = (q(x_(t-1)|bold(upright(x))_0)q(x_t|x_(t-1))) / q(x_t|bold(upright(x))_0) \
  & = (cal(N)(x_(t-1);sqrt(overline(alpha)_(t-1))bold(upright(x))_0,(1-overline(alpha)_(t-1))bold(I))cal(N)(x_t;sqrt(alpha)_t x_(t-1),(1-alpha_t)bold(I))) / (cal(N)(x_t;sqrt(overline(alpha)_t)bold(upright(x))_0, (1-overline(alpha)_t)bold(I))) \
  & prop exp { - 1/2 [ ( (x_t - sqrt(alpha_t)x_(t-1))^2 ) / beta_t + ( (x_(t-1) - sqrt(1-overline(alpha)_(t-1))bold(upright(x))_0)^2 ) / (1-overline(alpha)_(t-1)) - (x_t - sqrt(overline(alpha)_t)bold(upright(x))_0)^2 / (1-overline(alpha)_t) ] } \
  & = exp { - 1/2 [ ( x_t^2 - 2 sqrt(alpha_t) x_(t-1) + alpha_t x_(t-1)^2 ) / beta_t + ( x^2_(t-1) - 2 sqrt(overline(alpha)_(t-1))x_(t-1) bold(upright(x))_0 + overline(alpha)_(t-1) bold(upright(x))_0^2 ) / (1-overline(alpha)_(t-1)) - (x_t - sqrt(overline(alpha)_t)bold(upright(x))_0)^2 / (1-overline(alpha)_t) ] } \
  & = exp { - 1/2 [ (alpha_t / beta_t + 1 / (1-overline(alpha)_(t-1)))x_(t-2)^2 - 2 ( sqrt(alpha_t) / beta_t x_t + sqrt(overline(alpha)_(t-1)) / (1 - overline(alpha)_(t-1)) )x_(t-1) + C(x_t,bold(upright(x))_0) ] }
$

$C(x_t,bold(upright(x))_0)$是对于$x_(t-1)$的常数。高斯分布的方差系数是

$
  overline(beta)_t & = 1 / (alpha_t/beta_t + 1/(1-overline(alpha)_(t-1))) \
                   & = (1-overline(alpha)_(t-1)) / (1-overline(alpha)_t) beta_t
$

均值是

$
  tilde(mu)_t (x_t,bold(upright(x))_0) &= ( sqrt(alpha)_t / beta_t x_t + sqrt(overline(alpha)_(t-1))/(1-overline(alpha)_(t-1))bold(upright(x))_0 ) / ( alpha_t / beta_t + 1 / (1-overline(alpha)_(t-1)) ) \
  & = (sqrt(alpha)_t / beta_t x_t + sqrt(overline(alpha)_(t-1))/(1-overline(alpha)_(t-1))bold(upright(x))_0)(1-overline(alpha)_(t-1)) / (1-overline(alpha)_t) (1-alpha_t) \
  & = (sqrt(alpha_t)(1-overline(alpha)_(t-1))) / (1 - overline(alpha)_t) x_t + (sqrt(overline(alpha)_(t-1))(1-alpha_t)) / (1 - overline(alpha)_t) bold(upright(x))_0
$

*证明完毕*

根据前向扩散的闭式解

$
  x_t = sqrt(overline(alpha)_t)x_0 + sqrt(1-overline(alpha)_t)epsilon
$

可以得到

$
  x_0 = 1/sqrt(overline(alpha)_t) (x_t - sqrt(1-overline(alpha)_t)epsilon)
$

代入 $tilde(mu)_t (x_t,bold(upright(x))_0)$ 可以得到其只依赖 $x_t$ 的形式：

$
  tilde(mu)_t (x_t,bold(upright(x))_0) &= (sqrt(alpha_t)(1-overline(alpha)_(t-1))) / (1 - overline(alpha)_t) x_t + (sqrt(overline(alpha)_(t-1))(1-alpha_t)) / (1 - overline(alpha)_t) 1/sqrt(overline(alpha)_t) (x_t - sqrt(1-overline(alpha)_t)epsilon) \
  & = 1/sqrt(overline(alpha)_t) (x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t)epsilon)
$

假设前向过程和反向过程都是马尔可夫链，前向过程的转移概率分布是高斯分布。在此基础上可以得出，反向过程的转移概率分布也是高斯分布，并且能够求出第$t$步前向的转移概率分布$q(x_t|x_0)$和反向的条件转移概率分布$q(x_(t-1)|x_t,x_0)$。

前向过程表示一步步加噪的随机过程，由超参数$beta_t (1,2,...,T)$控制，是事先确定的。反向过程表示一步步去噪的随机过程，由学习得到的神经网络控制。

DDPM用神经网络表示反向的转移概率分布$p_theta (x_(t-1)|x_t), t=1,2,...,T$。由定理可知，这个转移概率分布是高斯分布

$
  p_theta (x_(t-1)|x_t) = cal(N)(x_(t-1),bold(mu)_theta (x_t,t), bold(Sigma)_theta (x_t,t))
$

其均值和方差由神经网络决定。神经网络的输入是样本$x_t$和步数$t$，输出是均值$bold(mu)_theta$和方差$bold(Sigma)_theta$，参数是$theta$。

反向过程的联合概率分布表示为

$
  p(x_0 x_1 dots.c x_T) = p(x_T)cal(N)(x_(T-1);bold(mu)_theta (x_T,T),bold(Sigma)_theta (x_T, T)) dots.c cal(N)(x_1;bold(mu)_theta (x_1,1),bold(Sigma)_theta (x_1, 1))
$

假设每一步的协方差矩阵是对角阵。

$
  bold(Sigma)_theta (x_t,t) = sigma^2_t bold(I)
$

前向过程从第$t-1$步到第$t$步根据$q(x_t|x_(t-1))$采样，由$x_(t-1)$得到$x_t$。反向过程从第$t$步到第$t-1$步根据$p_theta (x_(t-1)|x_t)$采样，由$x_t$得到$x_(t-1)$。原理上$p_theta (x_(t-1)|x_t)$应该近似转移概率分布$q(x_(t-1)|x_t)$。但$q(x_(t-1)|x_t)$难以计算，DDPM实际上使用$p_theta (x_(t-1)|x_t)$近似条件转移概率分布$q(x_(t-1)|x_t,x_0)$。

我们的目的是希望$p_theta (bold(upright(x))_0)$能够逼近理想中的那个$p(bold(upright(x))_0)$。

#theorem(name: [詹生不等式])[
  $EE_(q(x)) [log f(x)] <= log EE_(q(x)) [f(x)]$
]

推导*证据下界*：ELBO。

#tip(title: [ELBO])[
  ELBO：Evidence Lower Bound，证据下界，变分下界
]

$
  & log p_theta (x_0) \
  & = log integral p_theta (x_0 x_1 dots.c x_T) upright(d) x_1 upright(d) x_2 dots upright(d) x_T \
  & = log integral p_theta (x_0 x_1 dots.c x_T) q(x_1 x_2 dots.c x_T|x_0)/q(x_1 x_2 dots.c x_T|x_0) upright(d) x_1 upright(d) x_2 dots upright(d) x_T \
  & = log EE_q(x_1 x_2 dots.c x_T|x_0) [ (p_theta (x_0 x_1 dots.c x_T)) / q(x_1 x_2 dots.c x_T|x_0) ] && "（显示为期望值）" \
  & >= EE_q(x_1 x_2 dots.c x_T|x_0) [ log (p_theta (x_0 x_1 dots.c x_T)) / q(x_1 x_2 dots.c x_T|x_0) ] && "（詹生不等式）"
$

那么就有

$
  underbrace(EE_(q(x_0)) log p_theta (x_0), "证据") >= underbrace(EE_q(x_0 x_1 x_2 dots.c x_T) [ log (p_theta (x_0 x_1 dots.c x_T)) / q(x_1 x_2 dots.c x_T|x_0) ], "证据下界")
$

直接最大化左边是不可行的，也就是不可求导，无法优化。所以DDPM通过最大化变分下界来优化目标。所以损失函数得到一下结果，其中用$p_theta (x_(t-1)|x_t)$近似$q(x_(t-1)|x_t,x_0)$。

$
  & L(theta) \
  & = EE_q(x_0 x_1 x_2 dots.c x_T) [ log q(x_1 x_2 dots.c x_T|x_0) / (p_theta (x_0 x_1 dots.c x_T)) ] \
  & = EE_q(x_0 x_1 x_2 dots.c x_T) [ log (product_(t=1)^T q(x_t|x_(t-1))) / (p_theta (x_T) product_(t=1)^T p_theta (x_(t-1)|x_t)) ] \
  & = EE_q(x_0 x_1 x_2 dots.c x_T) [ - log p_theta (x_T) + sum_(t=1)^T log q(x_t|x_(t-1)) / (p_theta (x_(t-1)|x_t)) ] \
  & = EE_q(x_0 x_1 x_2 dots.c x_T) [ - log p_theta (x_T) + sum_(t=2)^T log q(x_t|x_(t-1)) / (p_theta (x_(t-1)|x_t)) + log q(x_1|x_0) / (p_theta (x_0|x_1)) ] \
  & = EE_q(x_0 x_1 x_2 dots.c x_T) [ - log p_theta (x_T) + sum_(t=2)^T log q(x_(t-1)|x_t,x_0) / (p_theta (x_(t-1)|x_t)) + sum_(t=2)^T log q(x_t|x_0) / q(x_(t-1)|x_0) + log q(x_1|x_0) / (p_theta (x_0|x_1)) ] \
  & = EE_q(x_0 x_1 x_2 dots.c x_T) [ - log p_theta (x_T) + sum_(t=2)^T log q(x_(t-1)|x_t,x_0) / (p_theta (x_(t-1)|x_t)) + log q(x_T|x_0) / q(x_1|x_0) + log q(x_1|x_0) / (p_theta (x_0|x_1)) ] \
  & = EE_q(x_0 x_1 x_2 dots.c x_T) [ log q(x_T|x_0)/(p_theta (x_T)) + sum_(t=2)^T log q(x_(t-1)|x_t,x_0) / (p_theta (x_(t-1)|x_t)) - log p_theta (x_0|x_1) ] \
  & = underbrace(EE_(q(x_0,x_T)) [ log q(x_T|x_0)/(p_theta (x_T)) ], L_T) + sum_(t=2)^T underbrace(EE_(q(x_0,x_(t-1),x_t))[log q(x_(t-1)|x_t,x_0) / (p_theta (x_(t-1)|x_t))], L_(t-1))- underbrace(EE_(q(x_0,x_1))[log p_theta (x_0|x_1)], L_0)
$

这里的分布 $q(x_(t-1)|x_t,x_0)$ 的定义如下：

$
           q(x_(t-1)|x_t,x_0) & = cal(N)(x_(t-1);tilde(bold(mu))_t (x_t,x_0), tilde(beta)_t bold(I)) \
  tilde(bold(mu))_t (x_t,x_0) & = 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t)epsilon)
$

与之对应，分布$p_theta (x_(t-1)|x_t)$的定义如下：

$
  p_theta (x_(t-1)|x_t) &= cal(N)(x_(t-1);tilde(bold(mu))_theta (x_t,t), sigma^2_t bold(I)) \
  tilde(bold(mu))_theta (x_t,t) &= 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t)bold(epsilon)_theta (x_t, t))
$

假设两个分布$q(x_(t-1)|x_t,x_0)$和$p_theta (x_(t-1)|x_t)$均值具有相同的形式：方差相同，即设$tilde(beta)_t=sigma^2_t$。因此，神经网络简化为$epsilon_theta (x_t,t)$，其输入是样本$x_t$和步数$t$，输出是噪声$epsilon$，$theta$是参数。

DDPM的学习实际通过简化的损失函数的最小化进行，训练一个预测噪声的神经网络。

#tip(title: [KL散度])[
  $
    D_"KL" (P||Q) & = sum_(x in cal(X)) P(x) log P(x)/Q(x) \
                  & = EE_(x tilde P) [log P(x)/Q(x)]
  $
]

将损失函数展开，第一项损失 $L_T$ 如下：

$
  L_T = EE_(q(x_0)) ["KL" (q(x_T|x_0)||p_theta (x_T))]
$

由于 $q$ 没有可学习的参数$theta$，而 $p_theta (x_T)$ 是完全的高斯噪声，因此该项在训练期间将是一个常数，因此可以忽略不计。

这个损失是常数，所以对最小化不起作用。

中间各项的损失 $L_(t-1) (t=T,T-1,dots.c,2)$ 如下。通过计算分布 $q(x_(t-1)|x_t,x_0)$ 和 $p_theta (x_(t-1)|x_t)$ 的 KL 散度的期望得到，期望是针对分布 $q(x_0)$ 和 $p(epsilon)$ 的。

$
  L_(t-1) & = EE_(q(x_0,x_(t-1),x_t))[log q(x_(t-1)|x_t,x_0) / (p_theta (x_(t-1)|x_t))] \
  & = integral q(x_(t-1),x_t|x_0) log q(x_(t-1)|x_t,x_0) / (p_theta (x_(t-1)|x_t)) upright(d) x_(t-1) upright(d) x_t \
  & = integral q(x_t|x_0) q(x_(t-1)|x_t,x_0) log q(x_(t-1)|x_t,x_0) / (p_theta (x_(t-1)|x_t)) upright(d) x_(t-1) upright(d) x_t \
  & = integral q(x_t|x_0) underbrace(integral q(x_(t-1)|x_t,x_0) log q(x_(t-1)|x_t,x_0) / (p_theta (x_(t-1)|x_t)) upright(d) x_(t-1), "KL散度") upright(d) x_t \
  & = EE_(q(x_t|x_0)) [D_"KL" (q(x_(t-1)|x_t,x_0)||p_theta (x_(t-1)|x_t))] \
  & = EE_(q(x_t|x_0)) [ 1/(2 sigma^2 (t)) || bold(mu)(x_t,x_0) - bold(mu)_theta (x_t,t) ||^2 ] \
  & = EE_(q(x_t|x_0)) [ 1/(2 sigma^2 (t)) norm(1/sqrt(alpha_t) [x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t)epsilon] - 1/sqrt(alpha_t) [x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t)epsilon_theta (x_t,t)])^2 ] \
  & = EE_(q(x_t|x_0)) [ 1/(2 sigma^2 (t)) (1-alpha_t)^2 / (alpha_t (1-overline(alpha)_t)) norm(epsilon - epsilon_theta (x_t,t))^2 ] \
  & = EE_(q(x_t|x_0)) [ 1/(2 sigma^2 (t)) (1-alpha_t)^2 / (alpha_t (1-overline(alpha)_t)) norm(epsilon - epsilon_theta (sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon,t))^2 ]
$

最后一项损失 $L_0$ 如下：

$
  L_0 = EE_(q(x_1|x_0)) [ - log p_theta (x_0|x_1) ]
$

我们想计算的是条件概率$p_theta (x_0|x_1)$，其中：

- $x_0$是离散的图像数据（像素值离散，通常是$0 tilde 255$的整数值）。
- $x_1$是扩散过程中的一个中间变量（通常是连续变量，浮点数）。

$p_theta (x_0|x_1)$本身是个连续的高斯分布，我们却想要求解某个离散的整数像素值$x_0^i$的概率（$x_0^i$是原始图片$x_0$的第$i$个像素点），直接用像素点取值，只能取到概率密度值，想要得到某个像素点的概率，需要积分一个特别小的区间。由于像素值$0 tilde 255$被归一化到了$[-1,1]$区间。所以DDPM作者选择的积分区间是$(x_0^i - 1/255, x_0^i + 1/255)$。

那么此时有

$
  p_theta (x_0|x_1) &= product_(i=1)^D integral_(x_0^i - 1/255)^(x_0^i + 1/255) cal(N) (x;mu_theta^i (x_1,1),sigma_1^2) upright(d) x \
  & approx product_(i=1)^D cal(N) (x;mu_theta^i (x_1,1),sigma_1^2) times 2/255 \
  & = product_(i=1)^D 1/sqrt(2 pi sigma_1^2) exp (-(x_0^i - mu_theta^i (x_1,1))^2/(2 sigma_1^2)) times 2/255
$

其中$D$是图片像素点的数量。

两边取对数可以得到

$
  log p(x_0|x_1) = - 1/(2 sigma^2) norm(x_0 - mu_theta (x_1,1))^2 + C
$

其中$C$是常数，对优化（求导）没有影响。

针对损失 $L_(t-1),t=T,T-1,dots.c,2$ ，忽略系数，只对平方损失部分进行优化。针对损失 $L_0$ ，假设进行同样的平方损失优化。忽略损失 $L_T$ ，这样得到以下简化的整体损失函数：

$
  L'(theta) & = sum_(t=1)^T EE_(q(x_t|x_0)) [ norm(epsilon - epsilon_theta (x_t,t))^2 ] \
  & = sum_(t=1)^T EE_(q(x_t|x_0)) [ norm(epsilon - epsilon_theta (sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon,t))^2 ]
$

神经网络$epsilon_theta (x_t,t)$预测的是前向过程第$t$步的高斯噪声，其直观的解释是，这样的神经网络也能对反向过程的第$t$步$(t=1,2,dots.c,T)$进行有效的去噪。

反向过程的第$t$步到第$t-1$步的转移概率分布，可以利用学习得到的神经网络$epsilon_theta (x_t,t)$计算。使用随机变量表示形式：

$
  x_(t-1) = 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t)epsilon_theta (x_t,t)) + sigma_t epsilon
$

上面的式子称为DDPM反向过程的迭代公式，用于数据生成。设每一步的方差系数与对应的前向过程的方差系数相同，$sigma_t = sqrt(beta_t)$。

#tip(title: [两个正态分布之间的KL散度])[
  两个正态分布之间的KL散度可以通过解析的方式求得。如果$q(bold(z))=cal(N)(bold(z);bold(mu)_1,bold(sigma)_1^2 bold(I)), p(bold(z))=cal(N)(bold(z);bold(mu)_2,bold(sigma)_2^2 bold(I))$，那么KL散度可以表示为以下式子（式子中的$H$是$bold(z)$的维度）。

  $
    D_"KL" (q || p) = - 1/2 sum_(h=1)^H ( 1 + log (sigma_(1,h)^2) / (sigma_(2,h)^2) - (mu_(1,h)-mu_(2,h))^2 / sigma_(2,h)^2 - sigma_(1,h)^2 / sigma_(2,h)^2)
  $
]

#tip(title: [期望值的线性性质和相关变量的期望值])[
  ① 期望的线性性质

  期望值的线性性质是指有下面的等式成立。

  $
    EE_(p(x,y)) [x+y] = EE_(p(x))[x] + EE_(p(y))[y]
  $

  上式中的$x$和$y$是随机变量，假定它们相互关联（$p(x,y) != p(x)p(y)$）。上式成立的证明如下所示。

  $
    EE_(p(x,y)) [x+y] &= integral.double (x+y) p(x,y) upright(d) x upright(d) y \
    &= integral.double x p(x,y) upright(d) x upright(d) y + integral.double y p(x,y) upright(d) x upright(d) y \
    &= integral x underbrace((integral p(x,y) upright(d) y), p(x)) upright(d) x + integral y underbrace((integral p(x,y) upright(d) x), p(y)) upright(d) y \
    &= integral x p(x) upright(d) x + integral y p(y) upright(d) y \
    &= EE_(p(x))[x] + EE_(p(y))[y]
  $

  这里展示的等式的意思是"和的期望值"等于"期望值的和"。这一关系也可以扩展到$T$个随机变量，因此有下面的式子成立。

  $
    EE_(p(x_1 x_2 dots.c x_T)) [ sum_(t=1)^T x_t ] = sum_(t=1)^T EE_(p(x_1 x_2 dots.c x_T)) [ x_t ]
  $

  ② 相关变量的期望值

  这里假设$f(x)$是以$x$为参数的任意函数。因此，有下式成立。

  $
    EE_(p(x,y)) [f(x)] = EE_(p(x)) [f(x)]
  $

  证明过程如下所示。

  $
    EE_(p(x,y)) [f(x)] & = integral.double f(x)p(x,y) upright(d) x upright(d) y \
                       & = integral.double f(x) p(x) p(y|x) upright(d) x upright(d) y \
                       & = integral f(x) p(x) underbrace(integral p(y|x) upright(d) y, =1) upright(d) x \
                       & = integral f(x) p(x) upright(d) x \
                       & = EE_(p(x)) [f(x)]
  $

  "在$p(x,y)$分布下$f(x)$的期望值"等同于"在$p(x)$分布下$f(x)$的期望值"。重点在于概率分布中与期望值内部的对象（$f(x)$）内容无关的随机变量可以被"消除"。因此，有下面的等式成立。

  $
    EE_(p(x_1 x_2 dots.c x_T)) [f(x_t)] = EE_(p(x_t)) [f(x_t)]
  $

  另外，$f(x_(t-1),x_t)$的期望值如下所示。

  $
    EE_(p(x_1 x_2 dots.c x_T)) [f(x_(t-1), x_t)] = EE_(p(x_(t-1),x_t)) [f(x_(t-1),x_t)]
  $
]

#chapter("条件扩散模型", image: image("./orange2.jpg"), l: "multimodal-cond-diffuser")

我们之前对数据$x$的概率$p(x)$进行了建模。但在实用层面，我们更希望对条件概率$p(x|y)$进行建模（其中$y$表示条件）。如果能成功建立$p(x|y)$的模型，那么就可以通过条件$y$控制想生成的$x$。

条件$y$可以是文本、图像或标签等。如果$y$是一张低分辨率的图像，那么可以考虑将其变换为高分辨率的图像。这就是被称为超分辨率成像（super-resolution imaging）的技术。使用扩散模型进行超分辨率处理的模型包括级联扩散模型（cascaded diffusion model）等。

本节将以$y$作为标签的例子进行说明。具体来说，我们将创建一个模型，在给定MNIST数据的数字标签后，该模型能够生成与该标签相对应的图像。

#figure(
  image("将要实现的条件扩散模型.svg"),
  caption: [本章将要实现的条件扩散模型],
)

== 向扩散模型添加条件

首先，我们从复习开始。在扩散模型中，我们可以选择在何处使用神经网络。例如，可以考虑使用神经网络对$bold(mu)_theta (x_t,t)$和$bold(epsilon)_theta (x_t,t)$进行建模。

#figure(
  image("预测图像和预测噪声.svg"),
  caption: [预测时刻$t-1$的图像（正态分布的均值向量）的横型（上）和预测添加到$x_t$中的噪声的模型（下）],
)

下面思考使用神经网络对$bold(mu)_theta (x_t,t)$建模的情况。在这种情祝下，$p_theta (x_(t-1)|x_t)$的数学式如下所示。

$
  p_theta (x_(t-1)|x_t) = cal(N) (x_(t-1);bold(mu) (x_t,t),sigma^2_q (t)bold(I))
$

数据$x_0$的概率$p_theta (x_0)$的数学式如下所示：

$
  p_theta (x_0) & = integral p_theta (x_0,x_1,dots.c,x_T)"d" x_1 dots.c "d" x_T & "  （概率的边际化）" \
  & = integral p_theta (x_0|x_1) dots.c p_theta (x_(T-1)|x_T)p(x_T) "d" x_1 dots.c "d" x_T & "（马尔可夫性质）"
$

这里有 $p(x_T)=cal(N) (x_T;bold(0),bold(I))$。

为了得到条件扩散模型，我们要建模的对象是$p_theta (x_0|y)$，而不是$p_theta (x_0)$。实现这个目标的最简单的方法是在每个时刻的$p_theta (x_(t-1)|x_t)$中添加条件$y$。具体的数学式如下所示：


$
  p_theta (x_0|colred(y))=integral p_theta (x_0|x_1,colred(y)) dots.c p_theta (x_(T-1)|x_T,colred(y)) p(x_T) upright(d)x_1 upright(d)x_T
$

此时的 $p_theta (x_(t-1)|x_t,colred(y))$ 的数学式如下所示：

$
  p_theta (x_(t-1)|x_t,colred(y)) = cal(N) (x_(t-1);bold(mu)_theta (x_t,t,colred(y)),sigma^2_q (t)bold(I))
$

在常规的扩散模型中，$bold(mu)_theta (x_t,t)$通常由神经网络进行建模。而在条件扩散模型中，如$bold(mu)_theta (x_t,t,colred(y))$所示，会额外添加参数$y$。换言之，通过向神经网络中添加$y$，可以使模型"进化"为条件扩散模型。同样，对于预测噪声的神经网络，也可以通过将参数$y$添加到$epsilon_theta (x_t,t)$中，使其变为$epsilon_theta (x_t,t,y)$来实现相应的功能（参见 @条件扩散模型的神经网络 ）。

#figure(
  image("条件扩散模型的神经网络.svg"),
  caption: [条件扩散模型的神经网络],
) <条件扩散模型的神经网络>

== 条件扩散模型的实现

我们已经使用神经网络实现了$epsilon_theta (x_t,t)$。$epsilon_theta (x_t,t)$的参数$t$是整数，可以通过正弦位置编码变换为向量。而新添加的条件$y$是标签，也是整数。这个整数$y$可以通过嵌入层（embedding layer）变换为向量。具体来说，$y$被变换成向量，然后与变换后的$t$向量相加。

#figure(
  image("在扩散模型中添加嵌入层.svg"),
  caption: [在扩散模型中添加嵌入层],
)

#tip[
  嵌入层的初始值被设置为随机值，然后通过训练进行忧化。这样就可以训练得到与每个标签相对应的匹配任务的向量。而由于与时刻ｔ相关的特定任务的训练要素很少，因此我们使用了一种称为正弦位置编码的固定向量变换的方法。
]

以下是代码示例。这里我们在上一章中实现的UNet类中添加了名为UNetCond类的代码，类名中的Cond是Conditional（条件）的缩写。

```python
class UNetCond(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        # ❶ 处理标签的嵌入层
        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        t = pos_encoding(timesteps, self.time_embed_dim)

        # ❷ 标签的处理
        if labels is not None:
            t += self.label_emb(labels)

        x1 = self.down1(x, t)
        x = self.maxpool(x1)
        x2 = self.down2(x, t)
        x = self.maxpool(x2)

        x = self.bot1(x, t)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, t)
        x = self.out(x)
        return x
```

代码的主要修改有两处。在代码❶处，我们准备了处理标签的嵌入层（`nn.Embedding`）。具体来说，就是通过`nn.Embedding(num_labels, time_embed_dim)`将共计`num_labels`个不同整数变换为`time_embed_dim`维的向量。在代码❷处，通过`nn.Embedding`层处理标签，然后再加到变换后的时间数据的向量中。

接下来是`Diffuser`类。对于`Diffuser`类，修改生成数据的方法。下面只展示了修改部分的代码。

```python
class Diffuser:
    def denoise(self, model, x, t, labels):
        # ...（省略）
        with torch.no_grad():
            eps = model(x, t, labels) # 同时提供labels
            # ...

    def sample(self, model, x_shape=(20, 1, 28, 28), labels=None):
        #...
        if labels is None:
            labels = torch.randint(0, 10, (len(x),), device=self.device)
        #...
        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device,
                             dtype=torch.long)
            x = self.denoise(model, x, t, labels) # 还提供labels
        #...
        return images, labels
```

修改的部分还向模型提供了标签。具体来说，标签是在生成数据和去噪过程中提供的。最后是训练的代码。

```python
diffuser = Diffuser(num_timesteps, device=device)
model = UNetCond(num_labels=10) ❶
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device) ❷
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t, labels) ❸
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1
```

训练代码的修改内容如下所示。

❶ `UNetCond(num_labels=10)`

创建一个具有10个分类的条件扩散模型。

❷ `labels.to(device)`

在device上准备标签数据。

❸ `model(x_noisy, t, labels)`

向模型提供labels进行训练。

以上就是条件扩散模型的实现。现在运行代码。经过10轮训练后，最终生成了下图所示的图像。

虽然仍有改进的余地，但生成的图像是符合条件的。在下一节中，我们将探讨进一步改进这个条件扩散模型的方法。

完整代码如下：

```python
import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


img_size = 28
batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_images(images, labels=None, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            if labels is not None:
                ax.set_xlabel(labels[i].item())
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            i += 1
    plt.tight_layout()
    plt.show()

def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(timesteps, output_dim, device='cpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y

class UNetCond(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # 如果条件标签不为None
        if num_labels is not None:
            # 将标签转换为嵌入，嵌入和时间步的嵌入形状相同
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        t = pos_encoding(timesteps, self.time_embed_dim)
        # 如果标签不为空，将标签转换为嵌入，然后加到时间步的嵌入上面。
        if labels is not None:
            t += self.label_emb(labels)

        x1 = self.down1(x, t)
        x = self.maxpool(x1)
        x2 = self.down2(x, t)
        x = self.maxpool(x2)

        x = self.bot1(x, t)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, t)
        x = self.out(x)
        return x


class Diffuser:
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device='cpu'
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            device=device
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        alpha_bar = alpha_bar.view(
            alpha_bar.size(0), 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 \
            + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, labels):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t, labels)  # 添加了标签的嵌入
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) \
           / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std

    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 28, 28), labels=None):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        if labels is None:
            labels = torch.randint(0, 10, (len(x),), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor(
                [i] * batch_size,
                device=self.device,
                dtype=torch.long
            )
            # 带着标签去噪
            x = self.denoise(model, x, t, labels)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images, labels


preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(
    root='./../datasets', download=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

diffuser = Diffuser(num_timesteps, device=device)
model = UNetCond(num_labels=10)
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    # generate samples every epoch ===================
    images, labels = diffuser.sample(model)
    show_images(images, labels)
    # ================================================

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        # 使用标签数据
        labels = labels.to(device)
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        # 带着标签预测噪声
        noise_pred = model(x_noisy, t, labels)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')

# plot losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# generate samples
images, labels = diffuser.sample(model)
show_images(images, labels)
```

== 得分函数

在上面的内容里面，我们实现了一个简单的条件扩散模型。这个"简单"的意思是我们只是向模型提供了条件而已。因此，模型有可能会轻视我们提供的条件，甚至在最坏的情况下有可能会忽略我们提供的条件。接下来，我们将介绍一种称为"指引"的方法。指引是一种将给定条件纳入模型并给予更多重视的机制。

要理解指引机制，首先需要了解得分函数。

=== 什么是得分函数

截至目前，本章讨论的去噪扩散模型与另一类相对独立发展的深度生成式模型密切相关，它们都基于得分匹配（score matching）。这些模型利用的得分函数（或斯坦因得分）则定义为对数似然函数关于数据向量$bold(x)$的梯度。

$
  s(x) = nabla_x log p(x)
$ <score-match>

这里需要强调的是，梯度是针对*数据向量*计算的，而不是针对任何参数向量。注意$s(x)$是一个与$x$维度相同的向量值函数，并且其中的每个元素$s_i (x) = (partial log p(x)) / (partial x_i)$都与$x$的相应元素$x_i$相关联。例如，如果$x$是一个图像，那么$s(x)$也可以表示为一个同尺寸的图像，其中对应的像素是图像的得分。下图显示了二维中的概率密度示例以及相应的得分函数。

#figure(
  image("score-matching.svg"),
  caption: [得分函数的示意图，图中显示了两种信息的融合：由高斯混合分布构成的二维分布，表示为热图；以及由 @score-match 定义的相应得分函数，其作为向量绘制在$x$值的规则网络上],
)

为了理解得分函数的用处，考虑两个函数$q(x)$和$p(x)$，它们具有得分相等的属性，所以对于$x$的所有值，$nabla_x log q(x) = nabla_x log p(x)$。如果我们对这个等式的两边关于$x$积分并取指数，则可以得到$q(x)=K p(x)$，其中$K$是一个独立于$x$的常数。所以如果我们能学习到一个得分函数的模型$s(x,w)$，就可以重建原始数据分布，所得结果与之前的相比，最多相差一个常数倍数。

在实现扩散模型时，我们使用神经网络对$epsilon_theta (x_t,t)$进行了建模。$epsilon_theta (x_t,t)$基于$x_t$和$t$来推断噪声$epsilon$。下图是这个过程的示意图。

#figure(
  image("扩散模型中使用的推断噪声的神经网络.svg"),
  caption: [扩散模型中使用的推断噪声$epsilon$的神经网络],
)

此时有以下与$epsilon$相关的数学式成立（稍后将给出证明）。

$
  epsilon approx - sqrt(1-overline(alpha)_t) nabla_(x_t) log p(x_t)
$ <epsilon近似>

$nabla$是表示梯度的符号，读作"那勃乐"（Nabla）。$nabla_(x_t) log p(x_t)$是对数似然$log p (x_t)$对输入数据$x_t$的梯度，称为"得分函数"或"得分"。

#tip[
  $p(x_t)$是表示数据$x_t$为"真"的随机密度函数。而$p_theta (x_t)$表示使用参数对真的概率密度函数进行近似的概率密度函数。由于$nabla_(x_t) log p (x_t)$和$nabla_(x_t) log p_theta (x_t)$表示关于输入$x_t$的梯度，因此称为"得分"。另外，在某些领域，关于参数的梯度（$nabla_theta log p_theta (x_t)$）也称为"得分"。在本书中，我们将对输入的梯度称为"得分"。
]

根据式 @epsilon近似 ，$epsilon$可以近似表示为 $nabla_(x_t) log p(x_t)$。值得注意的是，$epsilon$ 与 $nabla_(x_t) log p(x_t)$之间只相差负常数倍（$-sqrt(1-overline(alpha)_t)$）。这说明以$-sqrt(1-overline(alpha)_t) epsilon$来代替$epsilon$作为训练数据的神经网络也是可行的。

#figure(image("用于推断得分的神经网络.svg"), caption: [用于推断得分的神经网络])

图中的神经网络将得分的近似值 $-sqrt(1-overline(alpha)_t) epsilon$ 推断为 $s_theta (x_t,t)$ 。因此，我们也可以通过推断得分的神经网络来实现扩散模型。

#tip[
  生成模型中包括对得分进行建模的方法。这些模型统称为基于得分的模型。以对数似然最大化的方式可以推导出各种模型（GMM、VAE、扩散模型等）。这些模型可以被称为基于似然的模型。重要的是，扩散模型也可以作为基于得分的模型推导得出。
]

=== @epsilon近似 的证明

现在让我们来证明 @epsilon近似 成立。

$
  epsilon approx - sqrt(1-overline(alpha)_t) nabla_(x_t) log p(x_t)
$

我们先来复习一下。时刻$t$的噪声数据$x_t$可以基于以下正态分布从原始数据$x_0$生成。

$
  q(x_t|x_0) = cal(N) (x_t;sqrt(overline(alpha)_t)x_0,(1-overline(alpha))bold(I))
$ <xtformula>

利用重参数化技巧，可以通过以下式子得到$x_t$的样本。

$
  epsilon tilde cal(N) (0,bold(I)) \
  x_t = sqrt(overline(alpha)_t)x_0 + sqrt(1-overline(alpha)_t)epsilon
$ <重参数技巧>

接下来要用到的是Tweedie公式，这个公式如下所示。

#tip(title: "Tweedie公式")[
  当基于 $x tilde cal(N)(x;bold(mu),bold(Sigma))$ 得到样本 $x$ 时，有以下式子成立。
  $
    EE [bold(mu)|x] = x + Sigma nabla_x log p(x)
  $
]

式子中的$bold(mu)$被视为随机变量。左侧的$EE [bold(mu)|x]$是在$x$作为条件下的$bold(mu)$的期望值。右侧的$nabla_x log p(x)$表示得分。我们将这个Tweedie公式应用于@xtformula。

#tip(title: "Tweedie公式的应用")[
  当基于 $x_t tilde cal(N) (x_t;sqrt(overline(alpha)_t)x_0,(1-overline(alpha)_t)bold(I))$得到样本$x_t$时，有以下式子成立。
  $
    EE [sqrt(overline(alpha)_t)x_0|x_t] & = x_t + (1-overline(alpha)_t)bold(I) nabla_x_t log p(x_t) \
                                        & = x_t + (1-overline(alpha)_t) nabla_x_t log p(x_t)
  $
]

然后，利用 @重参数技巧 中的 $x_t = sqrt(overline(alpha)_t)x_0 + sqrt(1-overline(alpha)_t)epsilon$ 成立这一事实将式子展开，如下所示。

$
  & EE [colred(sqrt(overline(alpha)_t)x_0)|x_t] = x_t + (1-overline(alpha)_t) nabla_x_t log p(x_t) \
  <=> & EE [colred(x_t-sqrt(1-overline(alpha)_t)epsilon)|x_t] = x_t + (1-overline(alpha)_t) nabla_x_t log p(x_t) \
  <=> & EE [x_t|x_t] - EE [sqrt(1-overline(alpha)_t) epsilon|x_t] = x_t + (1-overline(alpha)_t) nabla_x_t log p(x_t) \
  <=> & x_t - sqrt(1-overline(alpha)_t) EE [epsilon|x_t] = x_t + (1-overline(alpha)_t) nabla_x_t log p(x_t) \
  therefore & EE [epsilon|x_t] = (1-overline(alpha)_t) nabla_x_t log p(x_t)
$

期望值 $EE [epsilon|x_t]$ 可以用蒙特卡洛方法近似。这里我们用一个采样数据来近似$x_t$。这样就得到了以下数学式。

$
  epsilon approx - sqrt(1-overline(alpha)_t) nabla_x_t log p(x_t)
$

这样我们就完成了推导。

== 分类器指引

在上面的内容中，我们了解到可以通过预测得分的神经网络来实现扩散模型。在本节中，我们将从得分的角度来研究扩散模型，并推导出一种称为指引的方法。指引有两种主要类型：分类器指引（classifier guidance）和无分类器指引（classifier-free guidance）。我们先来介绍分类器指引。

=== 什么是分类器

分类器指引是一种使用分类器指导数据生成的方法。分类器是一个能够对数据进行分类的预训练神经网络，如图所示。

#figure(
  image("使用预训练神经网络进行分类的示例.svg"),
  caption: [使用预训练神经网络进行分类的示例（参数为$phi.alt$）],
)

图中的神经网络将时刻$t$的噪声图像$x_t$作为输入，并输出类别$y$的概率$p_phi.alt (y|x_t)$。利用这个分类器的神经网络，可以将普通的扩散模型变换为条件扩散模型。在这个过程中，我们需要掌握以下两点。

- 扩散模型可以通过预测得分 $nabla_x_t log p(x_t)$ 的神经网络来实现。
- （基于同样的原理）条件扩散模型可以通过预测条件概率得分 $nabla_x_t log p(x_t|y)$ 的神经网络来实现。

在记住以上两点之后，我们进行式子的变形，推导出分类器指引。

=== 分类器指引的推导

首先，根据贝叶斯定理，将条件概率$p(x_t|y)$表示为下面的数学式。

$
  p(x_t|y) = (p(x_t)p(y|x_t)) / (p(y))
$

然后，计算关于 $x_t$ 的梯度（$nabla_x_t$）。

$
  nabla_x_t log p(x_t|y) & = nabla_x_t log ( (p(x_t)p(y|x_t)) / p(y) ) \
                         & = nabla_x_t log p(x_t) + nabla_x_t log p(y|x_t) - underbrace(nabla_x_t log p(y), 0) \
                         & = nabla_x_t log p(x_t) + nabla_x_t log p(y|x_t)
$

中间的式子中出现 $nabla_x_t log p(y)$ ，但由于 $p(y)$ 中不包含 $x_t$ ，因此 $nabla_x_t log p(y)=0$ 。由此得到的数学式如下所示。

$
  underbrace(nabla_x_t log p(x_t|y), "条件得分") = underbrace(nabla_x_t log p(x_t), "❶得分") + underbrace(nabla_x_t log p(y|x_t), "❷分类器的对数似然的梯度")
$

根据这个式子可以推导出分类器指引。下面是对式子中的要点的说明。

❶ 此项可以使用预测得分的神经网络$s_theta (x_t,t)$来计算。

❷ 此项可以使用作为分类器的神经网络来计算。

由此可见，我们可以利用得分和分类器这两个神经网络来表示条件得分（而一旦有了条件得分，便可以实现条件扩散模型）。另外，❷的$nabla_x_t log p(y|x_t)$可以很容易地通过反向传播求出。

#figure(
  image("通过反向传播可以求出logp.svg"),
  caption: [通过反向传播可以求出$nabla_x_t log p(y|x_t)$],
)

上图中的 $nabla_x_t log p(y|x_t)$ 显示了在当前 $x_t$ 下，类别 $y$ 的对数似然增加最快的方向。因此，如果沿着 $nabla_x_t log p(y|x_t)$ 的方向更新 $x_t$ ，那么更新后的图像被分类为类别 $y$ 的概率会更高。
#linebreak()
#linebreak()
分类器指引的理念是使用得分和分类器来表示条件得分。通常，我们会向分类器指引中引入权重$gamma$，这样就能调整分类器的贡献度。引入后的数学式如下所示。

$
  nabla_x_t log p(x_t|y) = nabla_x_t log p(x_t) + colred(gamma) nabla_x_t log p(y|x_t)
$

权重$gamma$是人为设定的值（超参数），用于调整分类器向类别$y$方向引导的程度。$gamma$值越大，条件$y$的作用就越显著。使用表示推断得分的神经网络$s_theta (x_t,t)$和表示分类器的$p_phi.alt (y|x_t)$，可以将式子改写为如下形式。

$
  nabla_x_t log p(x_t|y) approx s_theta (x_t,t) + gamma nabla_x_t log p(y|x_t)
$

作为参考，上式所执行的处理如下图所示。

#figure(
  image("使用了推断得分的神经网络的分类器指引.svg"),
  caption: [使用了推断得分的神经网络的分类器指引],
)

如图所示，分类器可以与常规的扩散模型（推断得分的模型）相结合，作为条件扩散模型来生成数据。以上就是分类器指引的核心思想。

== 无分类器指引

分类器指引可以生成强调条件的数据。然而，分类器指引在实际应用中存在一个问题，即需要单独准备一个分类器。针对分类器指引的这一缺点进行改进的技术是无分类器指引。

=== 无分类器指引的理论知识

顾名思义，无分类器指引就是不需要分类器的指引。它的机制可以通过以下过程得出。

$
  & nabla_x_t log p(x_t|y) \
  & = nabla_x_t log p(x_t) + gamma nabla_x_t log p(y|x_t) \
  & = nabla_x_t log p(x_t) + gamma nabla_x_t log (p(x_t|y)p(y)) / p(x_t) \
  & = nabla_x_t log p(x_t) + gamma ( nabla_x_t log p(x_t|y) + underbrace(nabla_x_t log p(y), 0) - nabla_x_t log p(x_t) ) \
  & = nabla_x_t log p(x_t) + gamma ( nabla_x_t log p(x_t|y) - nabla_x_t log p(x_t) )
$

上面我们使用贝叶斯定理进行了式子的展开，最后得到了上面的式子。根据上面的式子，可以推导出无分类器指引。上式的意思是从点 $nabla_x_t log p(x_t)$ 出发，沿着 $nabla_x_t log p(x_t|y)$ 的方向前进了 $gamma$ 倍的距离。下图是这个式子的示意图。

#figure(
  image("式子的可视化.svg"),
  caption: [上面式子的可视化],
)

式子中的 $nabla_x_t log p(x_t)$ 和 $nabla_x_t log p(x_t|y)$ 分别表示无条件得分和条件得分。它们可以由以下两个神经网络进行推断。

- 无条件得分推断器：$s_theta_1 (x_t,t)$
- 条件得分的推断器：$s_theta_2 (x_t,t,y)$

不过，准备两个模型很麻烦。更简单的方法是使用一个条件得分推断器 $s_theta_2 (x_t,t,y)$ ，并按如下方式建模。

- 无条件得分的推断器：$s_theta (x_t,t,emptyset)$
- 条件得分的推断器：$s_theta (x_t,t,y)$

这里我们用$emptyset$表示"无条件"对应的类别。条件$y$可以被嵌入层变换为向量，但当类别为$emptyset$时，它会被变换为"零向量"，使其不包含任何信息。根据以上信息，无分类器指引的数学式可以表示成如下形式。

$
  nabla_x_t log p(x_t|y) approx s_theta (x_t,t,emptyset) + gamma ( s_theta (x_t,t,y) - s_theta (x_t,t,emptyset) )
$

下图展示了这一计算过程。

上面介绍的是推断得分函数的神经网络，但由于得分和噪声之间只有常数倍的差别，因此同样的机制也适用于推断噪声的神经网络。

#tip[
  在提供文本的图像生成服务中，我们经常会用到一种名为反向提示词（negative prompt）的技术。反向提示词是一种指定不希望生成的文本的技术。在上面的公式的$emptyset$中插入反向提示词即可实现这一技术。
]

=== 无分类器指引的实现

下面我们通过修改节中的部分代码来实现无分类器指引。先来回忆一下前面已经实现的UNetCond类，然后在这个类中实现`forward()`方法，代码如下所示。

```python
class UNetCond(nn.Module):
    #...（省略）
    def forward(self, x, timesteps, labels=None):
        t = pos_encoding(timesteps, self.time_embed_dim)

        if labels is not None:
            t += self.label_emb(labels)
        #...
```

从上面的代码来看，只有指定了参数`labels`，才会处理类标签。如果没有指定参数（当`labels=None`时），则不做任何处理。这相当于"无条件"的处理。

无分类器指引的训练是在"无条件"和"有条件"两种情况下进行的条件扩散模型的训练。例如，我们可以以一定的比例进行"无条件"的训练，除此之外进行"有条件"的训练。基于这一点，我们可以按如下方式实现。

```python
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        labels = labels.to(device)
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

        # 以10%的概率进行"无条件"的训练
        if np.random.random() < 0.1:
            labels = None

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t, labels)
        loss = F.mse_loss(noise, noise_pred)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1
```

上面的代码以10%的概率进行了"无条件"的训练。对于"无条件"的情况，设置`labels=None`。

最后是`Diffuser`类进行去噪处理的代码。

```python
class Diffuser:
    #...（省略）

    def denoise(self, model, x, t, labels, gamma):
        #...
        with torch.no_grad():
            eps_cond = model(x, t, labels)
            eps_uncond = model(x, t)
            eps = eps_uncond + gamma * (eps_cond - eps_uncond)
        #...
```

参数`gamma`的值越大，就越应该重视条件部分。另外，这里实现的是如下数学式的代码。

$
  epsilon approx epsilon_theta (x_t , t, emptyset) + gamma (epsilon_theta (x_t , t , y ) - epsilon_theta ( x_t, t, emptyset ))
$

经过以上修改之后，我们就完成了无分类器指引的实现。这里以$gamma = 3$（`gamma=3.0`）生成了数据。

无分类器指引可以通过$gamma$来调整条件的重视程度。而且由于无需另外训练分类器，因此在资源和时间方面更为高效。

完整代码如下：

```python
import math
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


img_size = 28
batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_images(images, labels=None, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            if labels is not None:
                ax.set_xlabel(labels[i].item())
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            i += 1
    plt.tight_layout()
    plt.show()

def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(timesteps, output_dim, device='cpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y

class UNetCond(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        t = pos_encoding(timesteps, self.time_embed_dim)

        if labels is not None:
            t += self.label_emb(labels)

        x1 = self.down1(x, t)
        x = self.maxpool(x1)
        x2 = self.down2(x, t)
        x = self.maxpool(x2)

        x = self.bot1(x, t)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, t)
        x = self.out(x)
        return x


class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        alpha_bar = alpha_bar.view(alpha_bar.size(0), 1, 1, 1)  # (N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t, labels, gamma):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t, labels)
            eps_uncond = model(x, t)
            eps = eps_uncond + gamma * (eps - eps_uncond)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std

    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 28, 28), labels=None, gamma=3.0):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        if labels is None:
            labels = torch.randint(0, 10, (len(x),), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels, gamma)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images, labels


preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

diffuser = Diffuser(num_timesteps, device=device)
model = UNetCond(num_labels=10)
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    # generate samples every epoch ===================
    #images, labels = diffuser.sample(model)
    #show_images(images, labels)
    # ================================================

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        labels = labels.to(device)
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

        if np.random.random() < 0.1:
            labels = None

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t, labels)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')

# plot losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# generate samples
images, labels = diffuser.sample(model)
show_images(images, labels)
```

== Stable Diffusion

到目前为止，我们已经实现了处理MNIST的小型扩散模型以及小型条件扩散模型。当然，现代的扩散模型规模相当庞大，而且经过了多项改进。下面是一些进一步改进扩散模型的指引。这里我们介绍一个著名的模型——Stable Diffusion，它能够生成下图所示的高清图像。另外，它的代码和预训练的权重数据是公开的，任何人都可以运行这些代码。

#figure(
  image("sd-image.png"),
  caption: [Stable Diffusion生成的高清图像],
)

#chapter("DALL-E2 （文生图）", image: image("./orange2.jpg"), l: "multimodal-dalle2")

#tip(title: [DALL-E2])[
  - 输入：文本提示词
  - 输出：图像
]

#tip(title: [训练DALL-E2的数据集])[图文对！]

去噪扩散概率模型（DDPM）是一种流行的生成式人工智能模型，由Ho等人于2020年提出，并由Nichol等人于2021年对其进行了改进。这些模型背后的基本思想是，在正向扩散过程中将噪声添加到图像中，以便训练模型预测在反向扩散过程中应在特定时间步去除的噪声。在对图像进行采样（去噪）时，需要从纯噪声的图像开始，并在每个时间步迭代地去除模型预测的噪声，直到获得最终图像。

为了让DDPM生成多种类型的图像，同时仍允许用户选择所需的图像类型，模型需要根据某些输入进行条件调节（条件扩散模型）。Ramesh等人提出了一种名为unCLIP的条件调节方法，该方法已用于OpenAI的DALL-E 2模型。在Ramesh等人描述的方法中，输入的图片标题或者描述首先被传递到一个*先验网络*（prior model），该网络将使用经过训练的CLIP模型来获取CLIP文本嵌入。然后，仅解码器架构的Transformer使用这些文本嵌入来生成可能的CLIP图像嵌入。先验网络生成的CLIP图像嵌入将被解码器网络（由UNet模型组成）用于条件调节所创建的图像。在本章中，我们将使用此过程构建一个简单的扩散模型。

#figure(
  image("DALLE模型整体架构.png"),
  caption: [
    DALL-E2 模型整体架构
  ],
)

#tip(title: [DALL-E2需要训练3个模型])[
  + CLIP模型
  + 先验模型：Decoder-Only Transformer
  + 生成图像的解码器网络：UNet
]

== CLIP

为了从文本创建扩散图像（文生图），我们将使用CLIP模型中的嵌入。从CLIP获得的文本嵌入用于调节先验模型，使其扩散相应的图像嵌入。然后，这些图像嵌入用于调节解码器模型，用来指导解码器生成对应的图像。

#tip[
  CLIP模型训练完毕后，冻结起来供后续使用。
]

== 先验模型（Prior Model）

#danger[
  训练先验模型时，需要使用上一节训练的CLIP模型，CLIP模型必须*冻结*！
]

#figure(
  image("先验模型架构图.png"),
  caption: [先验模型架构图],
)

先验模型的作用：*根据文本标题预测CLIP图像嵌入* 。也可以放弃先验模型，而以CLIP文本嵌入为条件，而不是用先验模型生成的CLIP图像嵌入作为条件，但使用先验模型的效果最佳。

先验模型是一个仅解码器架构的Transformer。

=== 先验模型的训练

要训练一个模型，我们必须搞清楚模型的输入和输出（预测目标）。

训练先验模型时，我们手上有的是：图文对和一个训练好的CLIP模型。

*先验模型的输入有6个：*

+ Text Captions：图文对中的"文本"。
+ CLIP Text Embeddings：图文对中的"文本"经过上一节训练好的*冻结的CLIP模型*生成的CLIP文本嵌入。
+ Timestep Embeddings：随机选取的时间步$t$的嵌入。
+ Noisy Image Embeddings：添加了$t$步噪声的CLIP图像嵌入。这个CLIP图像嵌入是图文对中的"图像"经过上一节训练好的*冻结的CLIP模型*生成的CLIP图像嵌入。
+ Learned Embeddings：一个随机初始化的可学习的嵌入。
+ Causal Attention Mask：因果注意力掩码。因为我们要训练的是一个Decoder-Only Transformer架构的模型，所以需要使用因果注意力掩码。

#danger[
  Noisy Image Embeddings是针对CLIP*图片嵌入*添加噪声，而不是针对图片添加噪声！
]

*先验模型的输出（预测目标）：*

- 图文对中的"图像"经过上一节训练好的*冻结的CLIP模型*生成的CLIP图像嵌入。

*训练细节：*

+ 6个输入经过Decoder-Only Transformer之后会输出5个张量，我们取*最后一个*，也就是图中蓝色的#box(fill: rgb("#D0E4F2"), inset: (x: 3pt, y: 0pt), outset: (y: 3pt), radius: 2pt)[Learned Embeddings]。
+ #box(fill: rgb("#D0E4F2"), inset: (x: 3pt, y: 0pt), outset: (y: 3pt), radius: 2pt)[Learned Embeddings]会送入一个MLP，输出的就是预测的CLIP图片嵌入。
+ 预测的CLIP图片嵌入与真正的CLIP图片嵌入计算损失，并反向传播更新网络。

=== 先验模型的推理

#tip[
  在推理时，我们手上只有：
  + 自己编写的提示词
  + 训练好的CLIP模型
  + 训练好的先验模型
]

*先验模型的输入有6个：*

+ #box(fill: rgb("#FEF2CB"), inset: (x: 3pt, y: 0pt), outset: (y: 3pt), radius: 2pt)[Text Captions]：自己编写的文本提示词。
+ CLIP Text Embeddings："自己编写的文本提示词"经过上一节训练好的*冻结的CLIP模型*生成的CLIP文本嵌入。
+ Timestep Embeddings：最大时间步$T=1000$的时间步嵌入。
+ Noisy Image Embeddings：纯噪声构成的嵌入（和图片嵌入的形状相同，从高斯分布中采样得到）。
+ Learned Embeddings：训练好的参数（先验模型训练完之后就在模型中了）。
+ Causal Attention Mask：因果注意力掩码。

#tip[
  所以只有#box(fill: rgb("#FEF2CB"), inset: (x: 3pt, y: 0pt), outset: (y: 3pt), radius: 2pt)[Text Captions]是我们自己编写的文本提示词，其它输入都是模型或者程序自动生成的。
]

*先验模型的输出：*

+ 6个输入经过Decoder-Only Transformer之后会输出5个张量，我们取*最后一个*，也就是图中蓝色的#box(fill: rgb("#D0E4F2"), inset: (x: 3pt, y: 0pt), outset: (y: 3pt), radius: 2pt)[Learned Embeddings]。
+ #box(fill: rgb("#D0E4F2"), inset: (x: 3pt, y: 0pt), outset: (y: 3pt), radius: 2pt)[Learned Embeddings]会送入两次MLP，生成两个预测的CLIP图片嵌入。
+ 两个预测的CLIP图片嵌入分别和CLIP文本嵌入计算余弦相似度，然后选取最相似的一个作为预测。

=== 关键代码解释

==== 余弦调度的方差计划

#figure(
  image("线性调度方差和余弦调度方差的对比.png"),
  caption: [线性调度方差和余弦调度方差的对比],
)

对于方差调度，在Ho等人的原始DDPM论文中，采用了线性调度。虽然该调度在高分辨率图像上效果良好，但当图像较小（`64x64`或更小）时，前向过程最终会产生过多的噪声。为了解决这个问题，Nichol等人建议使用余弦调度，其公式如下所示。

$
             beta_t & = 1 - overline(alpha)_t / overline(alpha)_(t-1) \
  overline(alpha)_t & = f(t) / f(0) \
               f(t) & = cos ( (t/T + s) / (1+s) dot pi/2 )^2
$

这些值被剪裁为小于或等于0.999，以防止在 t = T 附近出现奇点。因为我们使用包含低分辨率图像的 FashionMNIST 数据集，所以我们将使用余弦方差计划。

综合起来，前向扩散过程的代码看起来是这样的：

```python
# Gets elements from indicies and makes sure output has a certain dimension
def extract_and_expand(x, idx, shape):
    return x[idx].reshape(idx.shape[0], *((1, ) * (len(shape) - 1)))

# 返回方差调度计划
def get_beta_schedule(schedule, max_time, s=0.008):
    if schedule == "linear":
        scale = 1000 / max_time
        betas = torch.linspace(1e-4  * scale, 0.02  * scale, max_time)
    elif schedule == "cosine":
        t = torch.linspace(0, max_time, max_time + 1)
        a_bars = torch.cos(
            (((t / max_time) + s) / (1 + s)) * (np.pi / 2)
        ) ** 2
        a_bars = a_bars / a_bars[0]
        betas = 1 - (a_bars[1:] / a_bars[:-1])
        betas = torch.clamp(betas, min=0, max=0.999)
    else:
        Exception("Beta schedule not implemented.")

    return betas

def get_schedule_values(config):
        schedule_values = {}
        schedule_values["betas"] = get_beta_schedule(
            config.prior.schedule,
            config.prior.max_time
        ).to(config.device)
        schedule_values["alphas"] = 1.0 - schedule_values["betas"]
        schedule_values["alpha_bars"] = torch.cumprod(
            schedule_values["alphas"],
            axis = 0
        )
        schedule_values["sqrt_recip_alphas"] = torch.sqrt(
            1.0 / schedule_values["alphas"])
        schedule_values["sqrt_alpha_bars"] = torch.sqrt(
            schedule_values["alpha_bars"]
        )
        schedule_values["sqrt_one_minus_alpha_bars"] = torch.sqrt(
            1.0 - schedule_values["alpha_bars"]
        )
        schedule_values["alpha_bars_prev"] = torch.cat((
            torch.ones(1, device=config.device),
            schedule_values["alpha_bars"][:-1])
        )
        schedule_values["sigma"] = schedule_values["betas"] \
            * (1.0 - schedule_values["alpha_bars_prev"])    \
            / (1.0 - schedule_values["alpha_bars"])
        return schedule_values

# 获取x_t，前向扩散过程
def forward_diffusion(x_0, schedule_values, t):
    # 噪声和图片的尺寸一样
    noise = torch.randn_like(x_0)
    sqrt_alpha_bars = extract_and_expand(
        schedule_values["sqrt_alpha_bars"], t, x_0.shape)
    sqrt_one_minus_alpha_bars = extract_and_expand(
        schedule_values["sqrt_one_minus_alpha_bars"], t, x_0.shape)
    x_noisy = (sqrt_alpha_bars * x_0) \
            + (sqrt_one_minus_alpha_bars * noise)
    # 返回x_t, 和添加的噪声
    return x_noisy, noise
```

==== 时间步嵌入

时间步嵌入是扩散的重要组成部分。这是因为不同时间步的图像具有不同的噪声量。为了在我们的模型中利用这些信息，我们将使用正弦位置编码。这些位置编码与 Transformer中常用的位置编码相同。主要区别在于，我们的输入时间步很可能不会按顺序排列，并且包含所有可能的时间步，因此我们只需要获取与输入时间步对应的位置编码。

```python
class SinusoidalPositionalEncodings(nn.Module):
    def __init__(
        self,
        max_seq_length,    # 最大序列长度
        width              # 模型的宽度（嵌入的维度d_model）
    ):
        super().__init__()
        # Create positional encodings
        pe = torch.zeros(max_seq_length, width)
        for pos in range(max_seq_length):
            for i in range(width):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000**(i/width)))
                else:
                    pe[pos][i] = np.cos(pos/(10000**((i-1)/width)))

        self.register_buffer('pe', pe)

    def forward(self, x):
        # Get positional encodings corresponding to inputted timesteps
        x = self.pe[x]
        return x
```

然后，这些时间编码通过 MLP 来进一步捕获时间信息。

```python
self.time_mlp = nn.Sequential(
    SinusoidalPositionalEmbedding(
        config.decoder.max_time,
        config.decoder.model_channels
    ),
    nn.Linear(
        config.decoder.model_channels,
        config.decoder.cond_channels
    ),
    nn.SiLU(),
    nn.Linear(
        config.decoder.cond_channels,
        config.decoder.cond_channels
    )
)
```

有多种方法可以调节时间步嵌入的输入，这些方法在本章的UNet：残差块部分中进行了描述。

==== 冻结模型

先验模型首先将图像的文本标题作为输入，然后获取CLIP文本嵌入和图像嵌入。加载CLIP模型时，所有层都应冻结，并将模式设置为eval。

```python
def freeze_model(model, set_eval=True):
    if set_eval:
        model.eval()

    for param in model.parameters():
        param.requires_grad = False

# Constructor
self.clip = CLIP(config).to(config.device)
self.clip.load_state_dict(torch.load(
    config.clip.model_location,
    map_location=config.device
))
# 冻结clip模型
freeze_model(self.clip)

# Forward
image_embeddings = self.clip.image_encoder(
    images
) # (B, C, H, W) -> (B, latent_dim)
text_embeddings = self.clip.text_encoder(
    captions,
    mask=masks
) # (B, text_seq_length) -> (B, latent_dim)
```

==== 加噪声的CLIP图片嵌入

然后，它会获取批次中每个图像的随机时间步，并使用它们从前向扩散中获取噪声CLIP图像嵌入。

#danger[
  不是对图片加噪声，而是对CLIP图片嵌入加噪声！
]


```python
# Forward
# 随机采样一个时间步
timesteps = torch.randint(
    0, self.config.prior.max_time, (images.shape[0],)) # (B, )

noisy_image_embedding, _ = forward_diffusion(
    image_embeddings, # 对图片的嵌入加噪声
    self.schedule_values, # 方差调度计划
    timesteps # 时间步
)
```

然后将时间步长通过 MLP 以获得时间步长嵌入。

```python
# Constructor
self.time_mlp = nn.Sequential(
    SinusoidalPositionalEmbedding(
        config.prior.max_time,
        config.latent_dim
    ),
    nn.Linear(
        config.latent_dim,
        config.latent_dim * config.prior.r_mlp,
        bias=config.prior.bias
    ),
    nn.SiLU(),
    nn.Linear(
        config.latent_dim * config.prior.r_mlp,
        config.latent_dim,
        bias=config.prior.bias
    )
)

# Forward
timestep_embeddings = self.time_mlp(timesteps) # (B, ) -> (B, latent_dim)
```

==== 可学习的嵌入

先验模型的另一个重要部分是可以训练的嵌入层。这些嵌入是 Torch 的一个参数，将用于预测最终输出。从构造函数中学习到的嵌入需要在 Forward 方法中进行扩展，以便批次中的每个项目都有一个嵌入。

```python
# 初始化一个随机的需要学习的嵌入参数
self.learned_embedding = nn.Parameter(torch.randn(config.latent_dim))

# Forward
# 为批次中的每一项复制一份可学习参数
learned_embeddings = self.learned_embedding.repeat(
    images.shape[0],
    1
) # (latent_dim) -> (B, latent_dim)
```

==== 输入的拼接

文本标题、CLIP文本嵌入、时间步嵌入、加噪声的CLIP图像嵌入以及需要学习的嵌入将被连接成一个序列。所有这些项都将具有形状 `(B, latent_dim)`，但我们需要在中间添加一个额外的维度，使它们具有形状 `(B, 1, latent_dim)`。我们将在这个新维度上进行连接，使序列具有形状 `(B, 5, latent_dim)`。如果使用包含更高质量图像的数据集，一种提高模型质量的可能方法是将文本嵌入、图像嵌入和/或时间步嵌入传递到卷积层，以增加序列长度。

```python
tokens = torch.cat((
    captions,               # 图像标题
    text_embeddings,        # CLIP文本嵌入
    timestep_embeddings,    # 时间步嵌入
    noisy_image_embedding,  # 加噪声的CLIP图像嵌入
    learned_embeddings      # 需要学习的嵌入
), dim=1) # (B, 5, latent_dim)
```

然后，该序列会输入到一个带有因果注意力掩码的仅解码器架构的Transformer。因果注意力掩码是一个由 1 组成的下三角矩阵，它使得某个 token 只能关注在它之前出现的token。

```python
# 初始化
self.decoder = nn.ModuleList(
    [TransformerBlock(
        config.latent_dim,
        cond_width=config.latent_dim,
        n_heads=config.prior.n_heads,
        dropout=config.prior.dropout,
        r_mlp=config.prior.r_mlp,
        bias=config.prior.bias
    ) for _ in range(config.prior.n_layers)]
)

self.register_buffer(
    "causal_attention_mask",
    torch.tril(torch.ones(5, 5))[None, :]
)

# Forward
for block in self.decoder:
    tokens = block(tokens, mask=self.causal_attention_mask)
```

==== 预测目标

最后，我们从 Transformer 的输出中获取学习到的嵌入，并将其传递给 LayerNorm 和 Linear 层以获得预测的图像嵌入。

```python
# 初始化
self.output = nn.Sequential(
    nn.LayerNorm(config.latent_dim),
    nn.Linear(
        config.latent_dim,
        config.latent_dim,
        bias=config.decoder.bias
    )
)

# Forward：取最后一个训练出来的可学习嵌入
pred_image_embeddings = self.output(tokens[:, -1, :])
```

虽然通常情况下，扩散模型会预测噪声，并通过迭代去除噪声来生成样本，但unCLIP论文中提到，最好直接预测CLIP图像嵌入。该模型的损失函数应该是预测值与实际CLIP图像嵌入之间的均方误差损失。

```python
loss = nn.functional.mse_loss(pred_image_embeddings, image_embeddings)
```

在采样过程中，为了提高质量，模型应该生成两个CLIP图像嵌入的样本，并选择与CLIP文本嵌入点积较高的样本。

```python
def get_one_sample(self, text_embeddings, captions):
    # 获取一个和CLIP文本嵌入形状相同的白噪声
    # 而clip的图像嵌入和文本嵌入的形状相同
    noisy_image_embeddings = torch.randn(
        text_embeddings.shape,
        device=self.config.device
    )

    # 所有项的时间步都是最大时间步T，因为图像嵌入目前都是纯噪声
    timesteps = torch.full(
        (captions.shape[0],),
        self.config.prior.max_time - 1
    )

    # 获取时间步嵌入
    timestep_embeddings = self.time_mlp(
        timesteps
    ) # (B, ) -> (B, latent_dim)

    # (B, latent_dim) -> (B, 1, latent_dim)
    timestep_embeddings = timestep_embeddings[:, None, :]

    # 为批次中每一项都创建一个需要学习的嵌入
    learned_embeddings = self.learned_embedding.repeat(
        captions.shape[0],
        1
    ) # (latent_dim) -> (B, latent_dim)

    # (B, latent_dim) -> (B, 1, latent_dim)
    learned_embeddings = learned_embeddings[:, None, :]

    tokens = torch.cat((
        captions,               # 图像标题
        text_embeddings,        # CLIP文本嵌入
        timestep_embeddings,    # 最大时间步的嵌入（比如T=1000）
        noisy_image_embeddings,  # 白噪声
        learned_embeddings      # 需要学习的嵌入
    ), dim=1) # (B, 5, latent_dim)

    # 和因果注意力掩码一起输入给transformer块
    for block in self.decoder:
        tokens = block(tokens, mask=self.causal_attention_mask)

    # 预测的图像嵌入（取出学习到的嵌入，输入给投影层）
    pred_image_embeddings = self.output(tokens[:, -1, :])

    return pred_image_embeddings

def sample(self, captions, masks=None):
    # 获取CLIP文本嵌入
    t_emb = self.clip.text_encoder(
        captions,
        mask=masks
    ) # (B, text_seq_length) -> (B, latent_dim)

    # (B, latent_dim) -> (B, 1, latent_dim)
    text_embeddings = t_emb[:, None, :]

    # 使图片文本的长度等于 latent dimension
    if self.config.text_seq_length >= self.config.latent_dim:
        # (B, max_seq_len) -> (B, latent_dim)
        captions = captions[:, :self.config.latent_dim]
    else:
        captions = nn.functional.pad(
            captions,
            (0, self.config.latent_dim - self.config.text_seq_length)
        ) # (B, max_seq_len) -> (B, latent_dim)

    # (B, latent_dim) -> (B, 1, latent_dim)
    captions = captions[:, None, :]

    # 获取两个采样
    sample_1 = self.get_one_sample(text_embeddings, captions)
    sample_2 = self.get_one_sample(text_embeddings, captions)

    gen_image_embeddings = torch.zeros(sample_1.shape)

    # 选择和文本嵌入点积更大的样本
    for i in range(gen_image_embeddings.shape[0]):
        if sample_1[i] @ t_emb[i] >= sample_2[i] @ t_emb[i]:
            gen_image_embeddings[i] = sample_1[i]
        else:
            gen_image_embeddings[i] = sample_2[i]
    # 预测的图片嵌入
    return gen_image_embeddings
```

==== 先验模型完整代码

综合以上所有，扩散先验模型的代码看起来应该是这样的：

```python
class DiffusionPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 加载clip模型
        self.clip = CLIP(config).to(config.device)
        self.clip.load_state_dict(torch.load(
            config.clip.model_location,
            map_location=config.device
        ))
        freeze_model(self.clip)

        self.config = config

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(
                config.prior.max_time,
                config.latent_dim
            ),
            nn.Linear(
                config.latent_dim,
                config.latent_dim * config.prior.r_mlp,
                bias=config.prior.bias
            ),
            nn.SiLU(),
            nn.Linear(
                config.latent_dim * config.prior.r_mlp,
                config.latent_dim,
                bias=config.prior.bias
            )
        )

        self.learned_embedding = nn.Parameter(
            torch.randn(config.latent_dim)
        )

        self.schedule_values = get_schedule_values(config)

        # Transformer blocks
        self.decoder = nn.ModuleList(
            [TransformerBlock(
                config.latent_dim,
                cond_width=config.latent_dim,
                n_heads=config.prior.n_heads,
                dropout=config.prior.dropout,
                r_mlp=config.prior.r_mlp,
                bias=config.prior.bias
            ) for _ in range(config.prior.n_layers)]
        )

        # Output Projection，输出为预测的图片嵌入
        self.output = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(
                config.latent_dim,
                config.latent_dim,
                bias=config.decoder.bias
            )
        )

        self.register_buffer(
            "causal_attention_mask",
            torch.tril(torch.ones(5, 5))[None, :]
        )

    def get_one_sample(self, text_embeddings, captions):
        # Get image embeddings that are pure noise
        noisy_image_embeddings = torch.randn(text_embeddings.shape, device=self.config.device)

        # timestep is max for all items because image embeddings are pure noise
        timesteps = torch.full((captions.shape[0],), self.config.prior.max_time - 1)

        # Get timestep embeddings
        timestep_embeddings = self.time_mlp(timesteps) # (B, ) -> (B, latent_dim)
        timestep_embeddings = timestep_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Expand learned embedding so that there is one for each item in batch
        learned_embeddings = self.learned_embedding.repeat(captions.shape[0], 1) # (latent_dim) -> (B, latent_dim)
        learned_embeddings = learned_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        tokens = torch.cat((
            captions,               # Image Caption
            text_embeddings,        # CLIP Text Embedding
            timestep_embeddings,    # Timestep Embedding
            noisy_image_embeddings,  # Noisy CLIP Image Embedding
            learned_embeddings      # Learned Embedding
        ), dim=1) # (B, 5, latent_dim)

        # Pass through transformer blocks with causal attention mask
        for block in self.decoder:
            tokens = block(tokens, mask=self.causal_attention_mask)

        # Get learned embeddings and pass through output projection to get CLIP image embeddings
        pred_image_embeddings = self.output(tokens[:, -1, :])

        return pred_image_embeddings

    def sample(self, captions, masks=None):
        # Get CLIP text embeddings
        t_emb = self.clip.text_encoder(captions, mask=masks) # (B, text_seq_length) -> (B, latent_dim)
        text_embeddings = t_emb[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Make caption length equal to latent dimension
        if self.config.text_seq_length >= self.config.latent_dim:
            captions = captions[:, :self.config.latent_dim]  # (B, max_seq_len) -> (B, latent_dim)
        else:
            captions = nn.functional.pad(captions, (0, self.config.latent_dim - self.config.text_seq_length)) # (B, max_seq_len) -> (B, latent_dim)

        captions = captions[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Getting two samples
        sample_1 = self.get_one_sample(text_embeddings, captions)
        sample_2 = self.get_one_sample(text_embeddings, captions)

        gen_image_embeddings = torch.zeros(sample_1.shape)

        # Choosing the samples with the higher dot product with text embeddings
        for i in range(gen_image_embeddings.shape[0]):
            if sample_1[i] @ t_emb[i] >= sample_2[i] @ t_emb[i]:
                gen_image_embeddings[i] = sample_1[i]
            else:
                gen_image_embeddings[i] = sample_2[i]

        return gen_image_embeddings

    def forward(self, images, captions, masks=None):
        # Get CLIP image embeddings
        image_embeddings = self.clip.image_encoder(images) # (B, C, H, W) -> (B, latent_dim)

        # Get CLIP text embeddings
        text_embeddings = self.clip.text_encoder(captions, mask=masks) # (B, text_seq_length) -> (B, latent_dim)
        text_embeddings = text_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Make caption length equal to latent dimension
        if self.config.text_seq_length >= self.config.latent_dim:
            captions = captions[:, :self.config.latent_dim]  # (B, max_seq_len) -> (B, latent_dim)
        else:
            captions = nn.functional.pad(captions, (0, self.config.latent_dim - self.config.text_seq_length)) # (B, max_seq_len) -> (B, latent_dim)

        captions = captions[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Get random timesteps for forward diffusion
        timesteps = torch.randint(0, self.config.prior.max_time, (images.shape[0],)) # (B, )

        # Get timestep embeddings
        timestep_embeddings = self.time_mlp(timesteps) # (B, ) -> (B, latent_dim)
        timestep_embeddings = timestep_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Perform forward diffusion to get noisy CLIP image embeddings
        noisy_image_embedding, _ = forward_diffusion(image_embeddings, self.schedule_values, timesteps)
        noisy_image_embedding = noisy_image_embedding[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        # Expand learned embedding so that there is one for each item in batch
        learned_embeddings = self.learned_embedding.repeat(images.shape[0], 1) # (latent_dim) -> (B, latent_dim)
        learned_embeddings = learned_embeddings[:, None, :] # (B, latent_dim) -> (B, 1, latent_dim)

        tokens = torch.cat((
            captions,               # Image Caption
            text_embeddings,        # CLIP Text Embedding
            timestep_embeddings,    # Timestep Embedding
            noisy_image_embedding,  # Noisy CLIP Image Embedding
            learned_embeddings      # Learned Embedding
        ), dim=1) # (B, 5, latent_dim)

        # Pass through transformer blocks with causal attention mask
        for block in self.decoder:
            tokens = block(tokens, mask=self.causal_attention_mask)

        # Get learned embeddings and pass through output projection to get CLIP image embeddings
        pred_image_embeddings = self.output(tokens[:, -1, :])

        loss = nn.functional.mse_loss(
           pred_image_embeddings,
           image_embeddings
        )

        return loss
```


== 解码器

#danger[
  训练用于生成图片的解码器时，CLIP模型和先验模型都必须*冻结*！
]

=== 概述

#figure(
  image("unet-decoder.png"),
  caption: [解码器架构],
)

扩散解码器是模型中生成图像的部分。它通过预测每个时间步应该去除的噪声，并迭代地从噪声图像中去除预测的噪声来实现这一点。

最常用作其主干的两个模型架构是UNet和Transformer。

Transformer架构的主要优势之一是其可扩展性，它可以很好地扩展至大型数据集和更复杂的模型。另一个优势是，虽然UNet主要用于图像，但Transformer更加灵活，只需进行修改即可用于其他数据类型。

UNet是许多图像相关任务的主要选择。这是因为它擅长通过卷积层获取局部信息，同时通过跳跃连接保持高分辨率特征。UNet的另一个优点是输入和输出的形状应该相同，这在图像扩散中非常有用。

UNet的工作原理是，接收输入并将其传入编码器层，编码器层会识别/捕获相关特征，同时降低分辨率。然后，输入会传入解码器层，解码器层会尝试定位特征，同时将分辨率提升至原始形状。由于编码器层包含空间信息，因此会从编码器到解码器添加跳跃连接，以帮助保留这些信息。

=== 残差块

#figure(
  image("残差块.png"),
  caption: [残差块的架构],
)

编码器和解码器层均由残差块构成，这些残差块包含用于识别和定位特征的卷积。这些残差块通常由归一化、激活、卷积、归一化、激活、卷积组成；然而，我们的残差块将是归一化、卷积、归一化、激活、卷积、归一化（如图所示）。Han等人的论文证明了这种架构可以在保持非线性的同时提升性能。

```python
# 初始化
self.layers1 = nn.Sequential(
    nn.GroupNorm(n_groups, d_in),
    nn.Conv2d(d_in, d_out, kernel_size, padding=1)
)

self.layers2 = nn.Sequential(
    nn.GroupNorm(n_groups, d_out),
    nn.SiLU(),
    nn.Dropout(p=dropout),
    nn.Conv2d(d_out, d_out, kernel_size, padding=1),
    nn.GroupNorm(n_groups, d_out)
)

# Forward
x = self.layers1(x_0)
x = self.layers2(x)
```

对于残差块的归一化部分，我们将使用GroupNorm。之所以使用GroupNorm而不是BatchNorm，是因为它的性能与批次大小无关，这使得它在小批次或可变批次大小上表现更佳。使用GroupNorm也有助于提高训练期间的稳定性。

残差块的激活部分用于实现非线性特性。虽然ReLU通常用于残差网络，但我们将使用SiLU作为激活函数。在Ramachandran等人的论文中，尽管模型和超参数是专门为ReLU设置的，但SiLU的表现优于ReLU和其他激活函数。由于SiLU的简单性以及与ReLU的相似性，因此只需用它来代替ReLU即可轻松实现。

对于卷积，我们将使用 `3x3` 核大小和 SAME 填充的 Conv2d 来保留空间维度。SAME 填充的工作原理是在输入的边界添加零，以确保输出形状与未填充时的输入形状相同。第一个卷积会将输入从 `d_in` 通道投影到 `d_out` 通道，而第二个卷积则仅保留 `d_out` 通道。

残差块也将根据输入的嵌入信息进行调节。执行调节的主要方法之一是将嵌入添加到输入中。对于我们的模型，我们将执行线性投影以获得比例和偏差值。然后将输入乘以比例，然后添加偏差。Nichol 和 Dhariwal 表明，与加法相比，使用这种调节方法可以提高 FID 分数。编写此部分代码时需要注意的一点是，调节嵌入的维度可能比输入的维度少。因此，我们需要在嵌入的末尾添加维度。例如，如果输入的形状为 `(B, C, L)`，而嵌入的形状为 `(B, C)`，则需要添加一个维度以使其形状为 `(B, C, 1)`。

```python
# 初始化
self.use_scale_shift = use_scale_shift

self.cond_layers = nn.Sequential(
    nn.SiLU(),
    nn.Linear(model_channels, d_out * 2 if use_scale_shift else d_out)
)

# Forward
emb = self.cond_layers(emb)

while len(emb.shape) < len(x.shape):
    emb = emb[..., None]

if self.use_scale_shift:
    y_s, y_b = emb.chunk(2, dim=1)
    x = y_s * x + y_b
else:
    x += emb
```

调节是在第一次卷积之后进行的，而不是在残差块的开始处进行的，因为它允许模型在使用时间步信息对其进行细化之前获取基本特征，从而获得更好的表示。

最后，通过将 ResNet 模块的原始输入添加到输出来创建跳跃连接。如果需要更改通道数以匹配输出，则对输入执行 `1x1` 卷积。这些跳跃连接用于增强特征传播。

```python
# 初始化
self.residual = nn.Conv2d(d_in, d_out, 1) if d_in != d_out else nn.Identity()

# Forward
x += self.residual(x_0)
```

将所有这些放在一起，残差块的最终代码应该是这样的：

```python
class ResidualBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        cond_channels=128,
        n_groups=8, # 组归一化GroupNorm每一组的数据量
        kernel_size=(3,3),
        dropout=0.0,
        use_scale_shift=True
    ):
        super().__init__()

        self.use_scale_shift = use_scale_shift

        self.layers1 = nn.Sequential(
            nn.GroupNorm(n_groups, d_in),
            nn.SiLU(),
            nn.Conv2d(d_in, d_out, kernel_size, padding=1)
        )

        # Activation & Linear Projection for Embedding
        self.cond_layers = nn.Sequential(
            nn.SiLU(),
            # d_out multiplied by 2 in order to split into scale & shift if necessary
            nn.Linear(cond_channels, d_out * 2 if use_scale_shift else d_out)
        )

        self.layers2 = nn.Sequential(
            nn.GroupNorm(n_groups, d_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(d_out, d_out, kernel_size, padding=1),
            nn.GroupNorm(n_groups, d_out)
        )

        self.residual = nn.Conv2d(d_in, d_out, 1) \
                        if d_in != d_out          \
                        else nn.Identity()

    def forward(self, x_0, emb):
        # x_0是原始图片，emb是图片的clip嵌入
        x = self.layers1(x_0)

        emb = self.cond_layers(emb)

        # Adding dimensions to embedding
        while len(emb.shape) < len(x.shape):
            emb = emb[..., None]

        # Conditioning input with embedding
        if self.use_scale_shift:
            # Getting scale and shift
            y_s, y_b = emb.chunk(2, dim=1)
            # Performing scale and shift
            x = y_s * x + y_b
        else:
            # Adding embedding to input
            x += emb

        x = self.layers2(x)

        # Skip Connection
        x += self.residual(x_0)

        return x
```

=== 注意力块

#figure(
  image("注意力块.png"),
  caption: [注意力块],
)

在我们的模型中，我们将把注意力模块放置在编码器和解码器内层以及瓶颈层（bottle层，连接编码器和解码器的底层）的残差模块之后。例如，如果编码器和解码器有四层，那么第二层和第三层就会有一个注意力模块。

使用注意力模块对我们的模型有诸多好处。其一是它有助于捕捉图像不同部分之间的空间关系信息。其二是它有助于衡量模型特征的重要性。最后，注意力模块还可以用来调节图像生成过程。

有多种方法可以为注意力模块实现注意力机制。其中一些选项包括自注意力和交叉注意力。我们将使用的方法是 GLIDE 模型中用到的两种方法的混合。对于这种方法，我们将从输入和条件信息中获取K和V，并将它们连接在一起以获得最终的K和V。

```python
Q, K, V = self.qkv(x).chunk(3, dim=-1)

Q = Q.view(B, L, self.n_heads, self.head_size).transpose(1, 2)

K = K.view(B, L, self.n_heads, self.head_size).transpose(1, 2)

V = V.view(B, L, self.n_heads, self.head_size).transpose(1, 2)

# Concatenating keys and values of input and condition
if cond is not None:
    k_c, v_c = self.cond_kv(cond).chunk(2, dim=-1)
    k_c = k_c.view(B, cond.shape[1], self.n_heads, self.head_size).transpose(1, 2)
    v_c = v_c.view(B, cond.shape[1], self.n_heads, self.head_size).transpose(1, 2)
    K = torch.cat((K, k_c), dim=-2)
    V = torch.cat((V, v_c), dim=-2)
```

需要注意的是，输入图像的形状为 `(B, C, H, W)`，但对于我们的注意力机制，我们希望其形状为 `(B, L, C)`。因此，我们需要将 `H` 和 `W` 维度合并，并将其与 `C` 维度转置。执行注意力机制后，我们需要将输出转换回其原始形状。

```python
b, c, h, w = x_0.shape

# Changing shape to perform attention
x = x.permute(0, 2, 3, 1).view(b, h * w, c) # (B, C, H, W) -> (B, H * W, C)

# Attention
x = self.attention(x, cond)

# Changing back to original shape
x = x.view(b, h, w, c).permute(0, 3, 1, 2)
```

最终的代码看起来应该是这样的：

```python
class AttentionBlock(nn.Module):
    def __init__(self, n_channels, cond_channels, n_groups=8, n_heads=1, dropout=0.0):
        super().__init__()

        assert n_channels % n_heads == 0, "n_channels must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_size = n_channels // n_heads
        self.scale = self.head_size ** -0.5

        self.group_norm = nn.GroupNorm(n_groups, n_channels)

        self.qkv = nn.Linear(n_channels, n_channels * 3)

        self.cond_kv = nn.Linear(cond_channels, n_channels * 2)

        self.out_proj = nn.Linear(n_channels, n_channels)

        self.dropout = nn.Dropout(dropout)

    def attention(self, x, cond=None):
        B, L, _ = x.shape

        # Getting queries, keys, and values for input
        Q, K, V = self.qkv(x).chunk(3, dim=-1)

        Q = Q.view(B, L, self.n_heads, self.head_size).transpose(1, 2)

        K = K.view(B, L, self.n_heads, self.head_size).transpose(1, 2)

        V = V.view(B, L, self.n_heads, self.head_size).transpose(1, 2)

        # Concatenating keys and values of condition to keys and values of input
        if cond is not None:
            k_c, v_c = self.cond_kv(cond).chunk(2, dim=-1)
            k_c = k_c.view(B, cond.shape[1], self.n_heads, self.head_size).transpose(1, 2)
            v_c = v_c.view(B, cond.shape[1], self.n_heads, self.head_size).transpose(1, 2)
            K = torch.cat((K, k_c), dim=-2)
            V = torch.cat((V, v_c), dim=-2)

        # Get dot product between queries and keys
        attention = torch.matmul(Q, K.transpose(-2, -1))

        # Scale
        attention = attention * self.scale

        # Applying softmax
        attention = torch.softmax(attention, dim=-1)

        # Get dot product with values
        attention = torch.matmul(attention, V)

        # Combine heads
        attention = attention.transpose(1, 2)
        attention = attention.contiguous().view(x.shape)

        # Output projection
        attention = self.out_proj(attention)

        # Dropout
        attention = self.dropout(attention)

        return attention

    def forward(self, x_0, cond=None):
        b, c, h, w = x_0.shape

        # Group normalization
        x = self.group_norm(x_0)

        # Changing shape to perform attention
        x = x.permute(0, 2, 3, 1).view(b, h * w, c) # (B, C, H, W) -> (B, H * W, C)

        # Attention
        x = self.attention(x, cond)

        # Changing back to original shape
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        # Residual connection
        x = x + x_0

        return x
```

=== 下采样

在每个编码器层之间，我们需要降低输入的分辨率。主要的下采样方法有两种：池化和使用卷积层。池化无需参数，这使得它的计算效率更高，并且有助于防止过拟合。卷积层的优势在于它本身带有参数，这使得它能够学习并保留重要的特征。我们将在模型中使用步长2，这将使输入的分辨率降低2倍。

我们将对模型进行编码，以便能够使用任一方法，但是在训练模型时，我们将使用卷积方法。

```python
class Downsample(nn.Module):
    def __init__(self, n_channels, kernel_size=(3,3), stride=2, down_pool=False):
        super().__init__()

        if down_pool:
            self.down = nn.AvgPool2d(stride)
        else:
            self.down = nn.Conv2d(n_channels, n_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        x = self.down(x)
        return x
```

=== 上采样

在每个解码器层之间，我们需要提高输入的分辨率。为此，我们首先进行插值，将输入的高度和宽度乘以2。之后，我们将它传入一个卷积层，该层将学习并保留重要特征，同时确保通道数正确。

```python
class Upsample(nn.Module):
    def __init__(self, d_in, d_out, kernel_size=(3,3)):
        super().__init__()

        self.conv = nn.Conv2d(d_in, d_out, kernel_size, padding=1)

    def forward(self, x):
        # interpolate：插值
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x
```

=== 模型的训练

#figure(
  image("文生图扩散解码器.svg"),
  caption: [解码器训练流程],
)

在训练解码器时，模型需要做的第一件事就是设置条件信息。为此，我们首先需要从先验模型中采样以获取CLIP图像嵌入。与CLIP模型一样，在加载先验模型时，应冻结层并将模式设置为eval。

```python
# 初始化先验网络
self.prior = DiffusionPrior(config).to(config.device)
self.prior.load_state_dict(torch.load(
    config.prior.model_location,
    map_location=config.device
))
# 冻结模型
freeze_model(self.prior)

# Forward
# 得到预测的图片clip嵌入
img_embeddings = self.prior.sample(caption, mask).to(x.device)
```

之后，需要使用输入的时间步来获取时间步嵌入。然后将先验模型生成的CLIP图像嵌入投影并添加到时间步嵌入中。这些嵌入将用于调节模型的残差块。

```python
# 投影层，将clip图片嵌入转换成和时间步嵌入相同的形状，方便相加
self.img_projection = nn.Sequential(
    nn.Linear(config.latent_dim, config.decoder.cond_channels),
    nn.SiLU(),
    nn.Linear(config.decoder.cond_channels, config.decoder.cond_channels)
)

#Forward
# 条件嵌入 = 时间步嵌入 + clip图片嵌入
c_emb = self.time_mlp(time) + self.img_projection(img_embeddings)

for module in self.encoder:
    if isinstance(module, ResidualBlock):
        x = module(x, c_emb)
```

对于注意力条件信息，我们首先需要将文本标题传入文本Transformer编码器。在unCLIP论文中，之所以使用这些文本编码，是因为 Ramesh 等人认为这有助于学习CLIP无法学习的自然语言知识。但在测试过程中，他们发现这种方法效果不佳，因此这部分是可选的。

```python
# Constructor
self.text_embedding = nn.Embedding(config.vocab_size, config.latent_dim)
self.positional_encodings = nn.Parameter(torch.randn(config.text_seq_length,config.latent_dim) * (config.latent_dim ** -0.5))

self.text_encoder = nn.ModuleList(
    [TransformerBlock(
        config.latent_dim,
        cond_width=config.latent_dim,
        n_heads=config.decoder.n_heads,
        dropout=config.decoder.dropout,
        r_mlp=config.decoder.r_mlp,
        bias=config.decoder.bias
     ) for _ in range(config.decoder.text_layers)]
)

self.final_ln = nn.LayerNorm(config.latent_dim)

# Function
def encode_text(self, text, mask=None):
    x = self.text_embedding(text)

    x = x + self.positional_encodings

    for block in self.text_encoder:
        x = block(x, mask=mask)

    x = self.final_ln(x)

    return x

# Forward
text_encodings = self.encode_text(text, mask)
```

对文本进行编码后，CLIP图像嵌入被投影到四个额外的token中，并连接到编码文本的末尾。

```python
# Constructor
self.get_img_tokens = nn.Linear(1, config.decoder.n_img_tokens)

# Forward
img_tokens = self.get_img_tokens(img_embeddings[..., None]).permute(0, 2, 1)

c_attn = torch.cat([text_encodings, img_tokens], dim=1)
```

设置条件信息后，噪声图像将通过初始卷积层，以获得初始模型通道的通道数。

```python
# Constructor
ch = config.decoder.model_channels

self.in_conv = nn.Conv2d(
    config.img_channels,
    ch,
    config.decoder.kernel_size,
    padding=1
)

# Forward
x = self.in_conv(x)
```

现在，含噪图像已达到所需的通道数，它们可以与条件信息一起通过UNet层。UNet将包含四个编码器层和解码器层，中间有一个瓶颈层。对于每层的通道数，我们将使用 `[1, 2, 4, 8]` 作为层通道数与模型通道数的比率。UNet的编码器、解码器和瓶颈层都将包含两个残差块。在瓶颈残差​​块之间以及编码器和解码器内层的每个残差块之后，放置了注意力块。跳跃连接连接位于编码器和解码器的残差块之间。

```python
# Config
model_channels:int = 32
channel_ratios:list[int] = field(default_factory=lambda: [1, 2, 4, 8])
n_layer_blocks:int = 2

# Constructor
# UNet Encoder Layers
self.encoder = nn.ModuleList([])
for r in config.decoder.channel_ratios:
    for _ in range(config.decoder.n_layer_blocks):
        self.encoder.append(ResidualBlock(ch, config.decoder.model_channels * r, config.decoder.cond_channels, config.decoder.n_groups, config.decoder.kernel_size, config.decoder.dropout, config.decoder.use_scale_shift))

        ch = config.decoder.model_channels * r
        if r != config.decoder.channel_ratios[0] and r != config.decoder.channel_ratios[-1]:
            self.encoder.append(AttentionBlock(ch, config.latent_dim, config.decoder.n_groups, config.decoder.n_heads, config.decoder.dropout))

    if r != config.decoder.channel_ratios[-1]:
        self.encoder.append(Downsample(ch, config.decoder.kernel_size, config.decoder.stride, config.decoder.down_pool))

# UNet Bottleneck Layers
self.bottleneck = nn.ModuleList([])
for block in range(config.decoder.n_layer_blocks):
    self.bottleneck.append(ResidualBlock(ch, ch, config.decoder.cond_channels, config.decoder.n_groups, config.decoder.kernel_size, config.decoder.dropout, config.decoder.use_scale_shift))

    if block != config.decoder.n_layer_blocks - 1:
        self.bottleneck.append(AttentionBlock(ch, config.latent_dim, config.decoder.n_groups, config.decoder.n_heads, config.decoder.dropout))

# UNet Decoder Layers
self.decoder = nn.ModuleList([])
for r in range(len(config.decoder.channel_ratios))[::-1]:
    for _ in range(config.decoder.n_layer_blocks):
        self.decoder.append(ResidualBlock(ch * 2, ch, config.decoder.cond_channels, config.decoder.n_groups, config.decoder.kernel_size, config.decoder.dropout, config.decoder.use_scale_shift))

        if r != 0 and r!= len(config.decoder.channel_ratios) - 1:
            self.decoder.append(AttentionBlock(ch, config.latent_dim, config.decoder.n_groups, config.decoder.n_heads, config.decoder.dropout))

    if r != 0:
        ch = config.decoder.model_channels * config.decoder.channel_ratios[r-1]
        self.decoder.append(Upsample(config.decoder.model_channels * config.decoder.channel_ratios[r], ch, config.decoder.kernel_size))

# Forward
for module in self.encoder:
    if isinstance(module, ResidualBlock):
        x = module(x, c_emb)
        self.connections.append(x)
    elif isinstance(module, AttentionBlock):
        x = module(x, cond=c_attn)
    else:
        x = module(x)

for module in self.bottleneck:
    if isinstance(module, ResidualBlock):
        x = module(x, c_emb)
    elif isinstance(module, AttentionBlock):
        x = module(x, cond=c_attn)
    else:
        x = module(x)

for module in self.decoder:
    if isinstance(module, ResidualBlock):
        x = torch.cat([x, self.connections.pop()], dim=1)
        x = module(x, c_emb)
    elif isinstance(module, AttentionBlock):
        x = module(x, cond=c_attn)
    else:
        x = module(x)
```

然后，UNet解码器层的输出会经过GroupNorm和SiLU激活层，之后使用Conv2d让输出恢复到原始通道数。

```python
# Constructor
self.output = nn.Sequential(
    nn.GroupNorm(config.decoder.n_groups, config.decoder.model_channels),
    nn.SiLU(),
    nn.Conv2d(config.decoder.model_channels, config.img_channels, config.decoder.kernel_size, padding=1)
)

# Forward
x = self.output(x)
```

将所有内容放在一起，解码器模型的最终代码将如下所示：

```python
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Loading Prior Model
        self.prior = DiffusionPrior(config).to(config.device)
        self.prior.load_state_dict(torch.load(
            config.prior.model_location,
            map_location=config.device
        ))
        freeze_model(self.prior)

        # MLP to get time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(
                config.decoder.max_time,
                config.decoder.model_channels
            ),
            nn.Linear(
                config.decoder.model_channels,
                config.decoder.cond_channels
            ),
            nn.SiLU(),
            nn.Linear(
                config.decoder.cond_channels,
                config.decoder.cond_channels
            )
        )

        # MLP to project CLIP image embeddings
        self.img_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.decoder.cond_channels),
            nn.SiLU(),
            nn.Linear(config.decoder.cond_channels, config.decoder.cond_channels)
        )

        # Projection to get image tokens
        self.get_img_tokens = nn.Linear(1, config.decoder.n_img_tokens)

        # Embedding layer for text captions
        self.text_embedding = nn.Embedding(config.vocab_size, config.latent_dim)

        # Learned positional encodings for text captions
        self.positional_encodings = nn.Parameter(torch.randn(
            config.text_seq_length,
            config.latent_dim
        ) * (config.latent_dim ** -0.5))

        # Transformer encoder blocks to encoder text captions
        self.text_encoder = nn.ModuleList(
            [TransformerBlock(
                config.latent_dim,
                cond_width=config.latent_dim,
                n_heads=config.decoder.n_heads,
                dropout=config.decoder.dropout,
                r_mlp=config.decoder.r_mlp,
                bias=config.decoder.bias
            ) for _ in range(config.decoder.text_layers)]
        )

        self.final_ln = nn.LayerNorm(config.latent_dim)

        ch = config.decoder.model_channels

        # Initial convolution
        self.in_conv = nn.Conv2d(
            config.img_channels,
            ch,
            config.decoder.kernel_size,
            padding=1
        )

        # UNet Encoder Layers
        self.encoder = nn.ModuleList([])
        for r in config.decoder.channel_ratios:
            for _ in range(config.decoder.n_layer_blocks):
                self.encoder.append(ResidualBlock(
                    ch,
                    config.decoder.model_channels * r,
                    config.decoder.cond_channels,
                    config.decoder.n_groups,
                    config.decoder.kernel_size,
                    config.decoder.dropout,
                    config.decoder.use_scale_shift
                ))

                ch = config.decoder.model_channels * r
                if r != config.decoder.channel_ratios[0] \
                    and r != config.decoder.channel_ratios[-1]:
                    self.encoder.append(AttentionBlock(
                        ch,
                        config.latent_dim,
                        config.decoder.n_groups,
                        config.decoder.n_heads,
                        config.decoder.dropout
                    ))

            if r != config.decoder.channel_ratios[-1]:
                self.encoder.append(Downsample(
                    ch,
                    config.decoder.kernel_size,
                    config.decoder.stride,
                    config.decoder.down_pool
                ))

        # UNet Bottleneck Layers
        self.bottleneck = nn.ModuleList([])
        for block in range(config.decoder.n_layer_blocks):
            self.bottleneck.append(ResidualBlock(
                ch,
                ch,
                config.decoder.cond_channels,
                config.decoder.n_groups,
                config.decoder.kernel_size,
                config.decoder.dropout,
                config.decoder.use_scale_shift
            ))

            if block != config.decoder.n_layer_blocks - 1:
                self.bottleneck.append(AttentionBlock(
                    ch,
                    config.latent_dim,
                    config.decoder.n_groups,
                    config.decoder.n_heads,
                    config.decoder.dropout
                ))

        # UNet Decoder Layers
        self.decoder = nn.ModuleList([])
        for r in range(len(config.decoder.channel_ratios))[::-1]:
            for _ in range(config.decoder.n_layer_blocks):
                self.decoder.append(ResidualBlock(
                    ch * 2,
                    ch,
                    config.decoder.cond_channels,
                    config.decoder.n_groups,
                    config.decoder.kernel_size,
                    config.decoder.dropout,
                    config.decoder.use_scale_shift
                ))

                if r != 0 and r!= len(config.decoder.channel_ratios) - 1:
                    self.decoder.append(AttentionBlock(
                        ch,
                        config.latent_dim,
                        config.decoder.n_groups,
                        config.decoder.n_heads,
                        config.decoder.dropout
                    ))

            if r != 0:
                ch = config.decoder.model_channels \
                   * config.decoder.channel_ratios[r-1]
                self.decoder.append(Upsample(
                    config.decoder.model_channels \
                        * config.decoder.channel_ratios[r],
                    ch,
                    config.decoder.kernel_size
                ))

        # Output projection
        self.output = nn.Sequential(
            nn.GroupNorm(
                config.decoder.n_groups,
                config.decoder.model_channels
            ),
            nn.SiLU(),
            nn.Conv2d(
                config.decoder.model_channels,
                config.img_channels,
                config.decoder.kernel_size,
                padding=1
            )
        )

        # Skip connections
        self.connections = []

    def encode_text(self, text, mask=None):
        x = self.text_embedding(text)

        x = x + self.positional_encodings

        for block in self.text_encoder:
            x = block(x, mask=mask)

        x = self.final_ln(x)

        return x

    def forward(self, x, time, caption=None, mask=None):
        # Sample prior model to get CLIP image embeddings
        # 使用文本标题输入给先验模型prior model，获得预测的clip图片嵌入
        img_embeddings = self.prior.sample(caption, mask).to(x.device)

        # 条件嵌入 = 时间步嵌入 + clip图片嵌入
        c_emb = self.time_mlp(time) + self.img_projection(img_embeddings)

        # Get conditioning information for attention blocks
        c_attn = self.get_img_tokens(img_embeddings[..., None]).permute(0, 2, 1)
        # 图片标题为空，那么unet是一个无条件的扩散模型
        if caption is not None:
            c_attn = torch.cat([self.encode_text(caption, mask), c_attn], dim=1)

        # Initial convolution
        x = self.in_conv(x)

        # UNet encoder layers
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
                x = module(x, c_emb)
                self.connections.append(x)
            elif isinstance(module, AttentionBlock):
                x = module(x, cond=c_attn)
            else:
                x = module(x)

        # UNet bottleneck layers
        for module in self.bottleneck:
            if isinstance(module, ResidualBlock):
                x = module(x, c_emb)
            elif isinstance(module, AttentionBlock):
                x = module(x, cond=c_attn)
            else:
                x = module(x)

        # UNet decoder layers
        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                x = torch.cat([x, self.connections.pop()], dim=1)
                x = module(x, c_emb)
            elif isinstance(module, AttentionBlock):
                x = module(x, cond=c_attn)
            else:
                x = module(x)

        # Output projection
        x = self.output(x)

        return x
```

=== 损失的计算


训练解码器模型的一个重要部分是损失函数。我们的损失函数首先要做的就是对批次中的每个图像进行随机时间步长的采样。

```python
timesteps = torch.randint(
    0,
    config.decoder.max_time,
    (image.shape[0],),
    device=config.device,
    dtype=torch.long
)
```

然后使用这些时间步长和计划值来获取通过前向扩散添加的噪声图像和噪声。

```python
# 方差计划
schedule_values = get_schedule_values(config)
# x_t, 真实的噪声noise
noisy_image, noise = forward_diffusion(
    image,
    schedule_values,
    timesteps
)
```

然后将噪声图像、时间步长、文本标题和文本掩码传递到模型中以获得预测噪声。

```python
# x_t, 时间步，图片标题（最终被处理为条件信息，clip图片嵌入）
pred_noise = decoder(noisy_image, timesteps, caption, mask)
```

最后，我们可以使用模型的预测噪声和前向扩散函数的实际噪声来计算损失。我们将用于解码器的损失是预测噪声和实际噪声之间的均方误差损失。

```python
# 预测的噪声和真实噪声之间计算均方误差
loss = nn.functional.mse_loss(pred_noise, noise)
```

综合起来，我们得到：

```python
# Calculating Loss
get_schedule_values(config)
timesteps = torch.randint(0, config.decoder.max_time, (image.shape[0],), device=config.device, dtype=torch.long)
noisy_image, noise = forward_diffusion(image, schedule_values, timesteps)
pred_noise = decoder(noisy_image, timesteps, caption, mask)
loss = nn.functional.mse_loss(pred_noise, noise)
```

== 训练配置


```python
@dataclass
class CLIPConfig:
    # Vision Transformer
    patch_size:tuple[int,int] = (4,4)
    vit_width:int = 256
    vit_layers:int = 6
    vit_heads:int = 8
    # Text Transformer
    text_width:int = 256
    text_layers:int = 6
    text_heads:int = 8
    # Attention
    dropout:float = 0.2
    r_mlp:int = 4
    bias:bool = False
    # Training
    augment_data:bool = True
    num_workers:int = 0
    batch_size:int = 128
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 200
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    get_val_accuracy:bool = False
    model_location:str = "../clip_fmnist.pt"

@dataclass
class PriorConfig:
    # Diffusion
    max_time:int = 1000
    schedule:str = "cosine"
    schedule_offset:float = 0.008
    # Transformer Decoder
    width:int = 256
    n_layers:int = 6
    n_heads:int = 8
    # Attention
    dropout:float = 0.2
    r_mlp:int = 4
    bias:bool = False
    # Training
    augment_data:bool = False
    num_workers:int = 0
    batch_size:int = 128
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 150
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    model_location:str = "../prior_fmnist.pt"

@dataclass
class DecoderConfig:
    # Diffusion
    max_time:int = 1000
    schedule:str = "cosine"
    # UNet
    n_groups:int = 8
    kernel_size:tuple[int, int] = (3,3)
    model_channels:int = 32
    cond_channels:int = 128
    channel_ratios:list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    n_layer_blocks:int = 2
    dropout:float = 0.1
    use_scale_shift:bool = True
    n_heads:int = 8
    stride:int = 2
    down_pool:bool = False
    r_mlp:int = 4
    bias:bool = False
    text_layers:int = 4
    n_img_tokens:int = 4
    # Training
    augment_data:bool = False
    num_workers:int = 0
    batch_size:int = 32
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 100
    warmup_epochs:int = 5
    grad_max_norm:float = 1.0
    sample_after_epoch:bool = True
    model_location:str = "../decoder_fmnist.pt"

@dataclass
class FMNISTConfig:
    latent_dim:int = 256
    # Dataset Info
    dataset:str = "fashion_mnist"
    data_location:str = "./datasets"
    img_size:tuple[int,int] = (32,32)
    img_channels:int = 1
    vocab_size:int = 256
    text_seq_length:int = 64
    # Data Augmentation / Normalization
    prob_hflip:float = 0.5
    crop_padding:int = 4
    train_mean:list[float] = field(default_factory=lambda: [0.2855552])
    train_std:list[float] = field(default_factory=lambda: [0.33848408])
    # Training
    train_val_split:tuple[int,int] = (50000, 10000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model Configs
    clip = CLIPConfig()
    prior = PriorConfig()
    decoder = DecoderConfig()
```

== 训练

对于这个文本到图像的扩散模型，我们实际上需要训练三个独立的模型：CLIP、扩散先验模型和扩散解码器UNET。包含这三个训练脚本的所有代码会非常多。

对于数据集，所有图像的尺寸都从 `(28,28)` 调整为 `(32,32)`。图像也进行了归一化，平均值和标准差设置为训练样本的平均值和标准差。

```python
transform = T.Compose([
    T.Resize(config.img_size),
    T.ToTensor(),
    T.Normalize(config.train_mean, config.train_std)
])
```

对于CLIP模型的训练集，我还通过随机水平翻转和随机裁剪实现了训练分割的数据增强。

```python
transform = T.Compose([
    T.Resize(config.img_size),
    T.RandomHorizontalFlip(p=config.prob_hflip)
    T.RandomCrop(config.img_size[0], padding=config.crop_padding)
    T.ToTensor(),
    T.Normalize(config.train_mean, config.train_std)
])
```

在训练这三个模型时，当使用权重衰减时，我使用 AdamW 作为优化器，否则使用 Adam。

```python
if config.clip.weight_decay == 0:
    optimizer = Adam(clip.parameters(), lr=config.clip.lr)
else:
    optimizer = AdamW(clip.parameters(), lr=config.clip.lr, weight_decay=config.clip.weight_decay)
```

对于所有模型，我还使用了带有线性预热的余弦退火学习率调度器。该调度器在每个 epoch 结束时更新。

```python
if config.clip.warmup_epochs > 0:
    warmup = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=(1 / config.clip.warmup_epochs), end_factor=1.0, total_iters=(config.clip.warmup_epochs - 1), last_epoch=-1)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.clip.epochs - config.clip.warmup_epochs), eta_min=config.clip.lr_min)
```

所有模型还使用了最大范数为 1.0 的梯度剪裁，以帮助防止梯度爆炸。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.[model].grad_max_norm)
```

在训练 CLIP 模型时，训练数据被随机分成训练集和验证集。当某个 epoch 的验证损失小于或等于前一个最低验证损失时，模型权重会被保存。

采样过程的第一步是随机生成仅包含噪声的图像。

```python
B = prompts.shape[0]
# Get completely noisy image
img = torch.randn((B, config.img_channels, config.img_size[0], config.img_size[1]), device=config.device)
```

得到噪声图像后，我们将计算一些需要计算 $x_(t-1)$ 的时间表值。

```python
def get_schedule_values(config):
        schedule_values = {}
        schedule_values["betas"] = get_beta_schedule(config.decoder.schedule, config.decoder.max_time).to(config.device)
        schedule_values["alphas"] = 1.0 - schedule_values["betas"]
        schedule_values["alpha_bars"] = torch.cumprod(schedule_values["alphas"], axis = 0)
        schedule_values["sqrt_recip_alphas"] = torch.sqrt(1.0 / schedule_values["alphas"])
        schedule_values["sqrt_alpha_bars"] = torch.sqrt(schedule_values["alpha_bars"])
        schedule_values["sqrt_one_minus_alpha_bars"] = torch.sqrt(1.0 - schedule_values["alpha_bars"])
        schedule_values["alpha_bars_prev"] = torch.cat((torch.ones(1, device=config.device), schedule_values["alpha_bars"][:-1]))
        schedule_values["sigma"] = schedule_values["betas"] * (1.0 - schedule_values["alpha_bars_prev"]) / (1.0 - schedule_values["alpha_bars"])
        return schedule_values

schedule_values = get_schedule_values(config)
```

然后，我们需要以相反的顺序迭代遍历所有时间步，从 0 到 `max_time-1`。我们还需要扩展时间步，以便批次中的每条数据都有一个时间步。

```python
for t in range(0, config.max_time)[::-1]:
    timesteps = torch.full((B,), t, device=config.device, dtype=torch.long)
```

对于每个时间步的采样，第一步是获取该时间步的调度值。获取该时间步的调度值后，需要对其进行扩展，使其维度数等于图像的维度数。

```python
# Getting schedule values for timestep
sqrt_recip_alphas_t = extract_and_expand(schedule_values["sqrt_recip_alphas"], timesteps, img.shape)
betas_t = extract_and_expand(schedule_values["betas"], timesteps, img.shape)
sqrt_one_minus_alpha_bars_t = extract_and_expand(schedule_values["sqrt_one_minus_alpha_bars"], timesteps, img.shape)
sigma_t = extract_and_expand(schedule_values["sigma"], timesteps, img.shape)
```

之后，我们使用解码器模型来预测时间步长 t 的噪声。

```python
# Predicting noise at timestep t with decoder
pred_noise = decoder(img, timesteps, caption=prompt, mask=mask)
```

如果不是最终时间步，我们还需要生成随机噪声$z$。我们需要这个值和 $sigma_t$，因为我们的模型直接预测从 $x_t$ 到 $x_0$ 的噪声，而不是从 $x_t$ 到 $x_(t-1)$ 的噪声。因此，模型会预测从 $x_t$ 到 $x_0$ 的噪声，并将从 $x_(t-1)$ 到 $x_0$ 的噪声加回去。这样一来，只有从 $x_t$ 到 $x_(t-1)$ 的噪声被移除。最后一步不需要 $sigma_t$ 和 $z$ 值，因为从 $x_t$ 到 $x_0$ 的预测噪声与从 $x_t$ 到 $x_(t-1)$ 的预测噪声相同。

```python
# Generating random noise
z = torch.randn_like(img) if t > 0 else 0
```

利用生成的噪声、方差计划值和预测噪声，我们可以计算时间步 t-1 的图像。

```python
# Calculating image at timestep t-1
img = sqrt_recip_alphas_t * (img - (betas_t / sqrt_one_minus_alpha_bars_t) * pred_noise) + (sigma_t * z)
img = torch.clamp(img, -1.0, 1.0)
```

综合起来，采样和图像的代码将如下所示：

```python
@torch.no_grad()
def sample_image(config, prompt, mask, schedule_values=None):
    # Load decoder model
    decoder = Decoder(config).to(config.device)
    decoder.load_state_dict(torch.load(config.decoder.model_location, map_location=config.device))
    decoder.eval()

    B = prompt.shape[0]
    # Get completely noisy image
    img = torch.randn((B, config.img_channels, config.img_size[0], config.img_size[1]), device=config.device)

    # Calculate schedule values
    if schedule_values is None:
        schedule_values = get_schedule_values(config)

    for t in range(0, config.decoder.max_time)[::-1]:
        # Setting the timesteps for all the items in the batch
        timesteps = torch.full((B,), t, device=config.device, dtype=torch.long)

        # Getting schedule values for timestep
        sqrt_recip_alphas_t = extract_and_expand(schedule_values["sqrt_recip_alphas"], timesteps, img.shape)
        betas_t = extract_and_expand(schedule_values["betas"], timesteps, img.shape)
        sqrt_one_minus_alpha_bars_t = extract_and_expand(schedule_values["sqrt_one_minus_alpha_bars"], timesteps, img.shape)
        sigma_t = extract_and_expand(schedule_values["sigma"], timesteps, img.shape)

        # Predicting noise at timestep t with decoder
        pred_noise = decoder(img, timesteps, caption=prompt, mask=mask)

        # Generating random noise
        z = torch.randn_like(img) if t > 0 else 0

        # Calculating image at timestep t-1
        img = sqrt_recip_alphas_t * (img - (betas_t / sqrt_one_minus_alpha_bars_t) * pred_noise) + (sigma_t * z)

        img = torch.clamp(img, -1.0, 1.0)

    return img
```

== 训练结果

为了查看模型的结果，我将展示每个标题的反向扩散过程。为了查看这一点，我们创建了一个修改版的 sample_image 函数，用于绘制反向扩散过程中十个时间步的图像。

```python
# Displaying Results
config = FMNISTConfig()
captions = {
    0: "An image of a t-shirt/top",
    1: "An image of trousers",
    2: "An image of a pullover",
    3: "An image of a dress",
    4: "An image of a coat",
    5: "An image of a sandal",
    6: "An image of a shirt",
    7: "An image of a sneaker",
    8: "An image of a bag",
    9: "An image of an ankle boot"
}
sample_captions = torch.stack([tokenizer(x, text_seq_length=config.text_seq_length)[0] for x in captions.values()]).to(config.device)
sample_masks = torch.stack([tokenizer(x, text_seq_length=config.text_seq_length)[1] for x in captions.values()]).to(config.device)
for i in range(len(sample_captions)):
    caption = sample_captions[None, (i % len(sample_captions))]
    mask = sample_masks[None, (i % len(sample_masks))]
    test = sample_plot_image(config, caption, mask)
```

#figure(
  image("dalle训练结果.png"),
  caption: [训练结果],
)

#chapter("总结", image: image("./orange2.jpg"), l: "multimodal-summary")

+ ViT可以对解决手写数字识别问题。ViT可以解决图像分类的问题。
  + 将图片切成patch（补丁），添加一个分类token。
  + 每个patch都会加上位置编码信息。
  + NLP Transformer, Audio Transformer, Vision Transformer

+ Clip的目标是将图片和文本对齐。
  + 训练一个图像编码器(ViT)，训练一个文本编码器(NLP Transformer)
  + 要求图文对输出的两个嵌入的余弦相似度很高，说明图文对很对齐。
  + 可以解决的问题是*文搜图*。

+ 你的文搜图是怎么实现的呢？
  + 我使用了clip-chinese-vit-patch16预训练模型，结合本公司标注的图文对（1万对～10万对）进一步微调。
  + 将（图片id，图片的路径等信息，图片经过模型编码后的嵌入张量）存入chroma之类的向量数据库。
  + 输入文本，文本经过clip模型输出文本嵌入。然后使用文本嵌入在向量数据库中检索相似的向量，并返回图片文件的各类信息。

+ ClipCap：Clip + Caption，目标是给模型输入图片，能够生成图片对应的文字描述。也就是*图生文*。
  + Clip（预训练模型clip-chinese-vit-patch16，冻结） + 生成模型（GPT2，Qwen0.5B）。
  + Clip输出的图像嵌入的维度和gpt2要求的词嵌入维度不同，需要一个MLP转换形状。
  + 将图像嵌入通过一个mlp转换成10个token嵌入。然后和文本的token嵌入列表进行拼接。然后对gpt2和mlp进行微调。
  + 微调的方法：预测下一个token，但*只计算文本部分的损失*。
  + 构建数据集：图文对。一张图片对应多条文本。例如：1张图片有10条文本，那么是10对训练数据。50万对图文对，使用A800，训练20个小时。如果换生成模型，那么需要的GPU资源会多很多。
  + 标准的clipcap是冻结clip模型和冻结生成模型，只微调mlp。但是如果生成模型不够好，那么效果堪忧。

+ 扩散模型的目标是：生成没见过的图片。
  + 正向扩散：逐步给图片添加噪声（解析解，一步采样到$x_{500}$）。需要图片，随机选择的时间步，以及*方差计划*。
  + 反向扩散：
    + 需要训练一个UNET（输入和输出的分辨率一致。先下采样，再上采样，残差连接）。
    + 输入：$(x_500, t=500)$。输出是预测的噪声。训练方式是预测的噪声和真实添加的噪声进行mse loss。
    + 逐步去噪：$("纯噪声图片" x_1000,t=1000)$ 发送给UNET，unet输出预测的噪声，$(x_1000-"noise") arrow x_999$ 。接下来，$(x_999,t=999) arrow "noise"_999$, $x_999 - "noise"_999 arrow x_998$ 。。。。

+ 带条件的扩散模型的目标是给定提示词，生成提示词对应的没见过的图片，*文生图*。
  + 使用图文对训练clip模型，后续使用需要冻结。
  + 训练一个prior model（先验模型）。先验模型要解决的问题是：能够给定提示词，先验模型可以输出预测的clip图像嵌入。先验模型的目的是想要生成一个高质量的条件。
    + （训练阶段）模型的输入：（图片的文本id列表，图片的文本clip嵌入，随机选择的时间步$t=500$，针对图片的图像clip嵌入添加500步噪声后的图片嵌入，需要学习的嵌入）。预测目标是*图像clip嵌入*。
    + （推理阶段）模型的输入：（图片的文本id列表，图片的文本clip嵌入，时间步是1000，和图像嵌入相同形状的白噪声，嵌入）。输出：预测的clip图片嵌入。
    + 训练好先验模型以后，后续使用需要冻结。
  + 训练一个带条件的unet，用来扩散出图像。
    + （训练阶段）模型的输入：$(x_(t=500)$, $t=500$, 条件=先验模型根据提示词预测的图片嵌入)。输出是预测的噪声。
    + unet里面加了*注意力模块*。使用注意力模块处理*提示词信息*，捕获图像和提示词的全局信息。
    + 反向扩散：输入是(白噪声$x_1000,t=1000$, 先验模型根据提示词预测的图片嵌入, 提示词(用来给注意力模块))。然后逐步去噪。
