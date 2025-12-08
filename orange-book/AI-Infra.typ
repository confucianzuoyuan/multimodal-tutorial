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
#import "@preview/neural-netz:0.2.0": draw-network
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

#chapter("FlashAttention", image: image("./orange2.jpg"), l: "flash-att")

我们将从第一性原理出发。首先，我们会理解标准注意力机制是如何实现的，然后逐一解决其中的低效问题——就仿佛我们在独立探索并发现 Flash Attention 一样。

此外，我的一个小目标是为编译器社区的一些行话（lingo）"祛魅"：比如 kernel（算子/核函数）、kernel fusion（算子融合）、materialization（物化）等。

#figure(
  image("ai-infra-figures/FlashAttention.png"),
  caption: [在本章结束时应该能完全理解这张图],
)

废话不多说，让我们先拆解论文标题：

"FlashAttention：快速且内存高效的精确注意力，兼具IO感知"（"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"）

核心要点在于，FlashAttention具有以下特性：

- 快速 (Fast) —— 摘自论文原文："我们在训练 BERT-large（序列长度 512）时，比 MLPerf 1.1 的训练速度纪录快了15%；训练 GPT2（序列长度 1K）时，比 HuggingFace 和 Megatron-LM 的基线实现快了 3 倍；在 Long-Range Arena（序列长度 1K-4K）上，比基线快了 2.4 倍。"
- 内存高效 (Memory-efficient) —— 相比于随序列长度呈二次方增长 $O(N^2)$ 的标准注意力机制，该方法是次二次方甚至线性的 $O(N)$。我们稍后会探讨其原理和实现方式。
- 精确 (Exact) —— 这意味着它不是对注意力机制的近似（不像稀疏注意力或低秩矩阵近似方法那样），它的输出结果与"标准（Vanilla）"注意力机制完全一致。
- IO感知 (IO aware) —— 相比于标准注意力，FlashAttention 就像是有了"自主意识"（sentient）一样。

开个玩笑:) —— 这其实是指它并没有将底层硬件视为一个黑盒。相反，它利用了底层硬件（例如 GPU，当然其他 AI 加速器也适用，但我将以 GPU 为例）*存储层次结构（Memory Hierarchy）* 的相关知识。

让我们再深入展开一下这个 *IO 感知（IO awareness）* 的部分。"IO"（输入/输出）正是为什么更高的 FLOPS（每秒浮点运算次数）并不一定能转化为更短的实际运行时间（Wall-clock time）的原因（这可能有点反直觉，但如果你了解硬件的工作原理，就会觉得显而易见）。

论文中的相关摘录如下：

#tip[
  "尽管这些 [近似] 方法将计算需求降低到了随序列长度呈线性或近线性的水平，但其中许多方法在实际运行时间上并未表现出相对于标准注意力的加速，因此未能获得广泛采用。一个主要原因是它们专注于减少 FLOP（这可能与实际运行时间并不相关），而往往忽略了来自*内存访问（IO）*的开销。"
]

那么，秘诀是什么呢？

答案在于*硬件*：

#figure(
  image("ai-infra-figures/内存墙.png"),
  caption: [内存墙],
)

多年来，GPU *计算能力 (FLOPS)* 的增长速度一直快于*内存吞吐量 (TB/s)*的提升速度。

如果没有数据可供处理，哪怕你拥有百亿亿次 (exaFLOPS) 级别的计算速度也是徒劳。这两者需要紧密协同，既然硬件层面失去了这种平衡，我们就必须通过软件层面来加以弥补。

这就是"IO感知"的由来。

根据*计算量*与*内存访问量*之间的比例，各种操作通常被分为两类：

- *计算受限 (Compute-bound)*（例如：矩阵乘法）
- *内存受限 (Memory-bound)*（例如：逐元素操作（激活函数、Dropout、Masking），以及归约操作（Softmax、Layer Norm、求和等）……）

#notify[
  关于术语的说明：该比率通常用算术强度（arithmetic intensity）来衡量，即每字节内存访问所执行的算术运算次数。
]

事实证明，注意力机制（Attention）在当前的AI加速器上是*内存受限（memory-bound）*的。

为什么？

因为它"主要由逐元素（element-wise）操作组成"，或者更准确地说，注意力机制的*算术密度（arithmetic density）*并不高。

让我们放大看看论文中的这张图：

#figure(
  image("ai-infra-figures/内存受限.png"),
  caption: [Dropout、Softmax 和 Masking —— 这些全都是逐元素操作（elementwise ops），且均属于内存受限型（memory bound）。可以看出，正是它们占据了大部分的运行时间。],
)

== 从第一性原理出发，让深度学习"火力全开" (Go Brrrr)

*所以，你想提升深度学习模型的性能。你会怎么着手这项任务呢？*

通常，大家会依赖一堆以前可能奏效或者在推特上看到的"大杂烩"技巧。"用原位（in-place）操作！把梯度设为 None！装 PyTorch 1.10.0 别装 1.10.1！"

用户在现代系统（尤其是深度学习领域）上采取这种"头痛医头"的方法是可以理解的——性能优化往往感觉既像科学，又像"玄学"（alchemy）。话虽如此，从*第一性原理*出发进行推理，仍然可以排除掉一大堆无效的方法，从而让问题变得更容易上手。

举个例子，在数据集上获得良好的深度学习性能确实涉及很多猜测。但是：

- 如果你的*训练损失远低于测试损失*，那你正处于"过拟合"状态，此时试图增加模型容量（capacity）就是在浪费时间。
- 或者，如果你的*训练损失和验证损失完全一样*，那试图对模型进行正则化（regularize）也是在浪费时间。

同样地，你可以将深度学习的效率理解为由三个不同的部分组成：

+ *计算 (Compute)*：GPU 花在实际浮点运算 (FLOPS) 上的时间。
+ *内存 (Memory)*：GPU 内部传输张量所花的时间。
+ *开销 (Overhead)*：其他所有时间。

就像训练机器学习模型一样，知道自己处于哪种状态（regime），能让你专注于真正重要的优化：

- *内存受限时*：如果你把时间全花在内存传输上（即处于*内存带宽受限*状态），那么增加 GPU 的 FLOPS 根本没用。
- *计算受限时*：另一方面，如果你把时间全花在执行巨大的矩阵乘法（big chonky matmuls）上（即*计算受限*状态），那么把模型逻辑重写成 C++ 以减少开销也无济于事。

所以，如果你想让你的 GPU 继续"火力全开"（Go Brrrr），我们就来讨论一下系统可能耗时的这三个部分——*计算、内存带宽和开销*。

#figure(
  image("ai-infra-figures/苦涩的教训.png"),
  caption: [GPT-3 在"让印钞机火力全开"迷因格式下的含义：图中的大佬是Rich Sutton，他在说"GPUs go bitter"（GPU 带来苦涩教训）。这引用了他的名篇《苦涩的教训》（The Bitter Lesson），其核心观点是：大多数"聪明"的AI创新最终都是徒劳的，因为它们往往会成为AI性能的绊脚石，并最终被那些假设更少、但使用更多算力和数据的方法所超越。而另一边那个流泪抱怨的形象则是AI学术界的拟人化。在学术界，"精妙的设计"备受推崇，而大量使用算力被视为"作弊"和"丑陋/不优雅"。此刻他正痛哭流涕，抱怨像GPT-3这种（简单粗暴的）方法竟然打败了学术界几十年来苦心钻研的精妙系统。],
)

#tip[
  本文主要以GPU和PyTorch为例，但这些原理几乎适用于所有的硬件和框架。
]

=== 计算

优化深度学习系统的一个视角是：我们希望尽可能让系统处于"计算受限"状态（compute-bound regime）。你花大价钱买了那 312 Teraflops（每秒万亿次浮点运算）的算力，理想情况下，你就应该实打实地用到这 312 Teraflops。但是，为了让你昂贵的矩阵乘法运算"值回票价"，你需要减少花在其他环节上的时间。

但为什么要专注于最大化计算，而不是（比如说）内存带宽呢？原因很简单——你可以降低开销或内存成本，但（在大多数情况下）如果不改变实际执行的运算逻辑，你是无法减少所需的计算量的。

让"最大化计算利用率"变得更加困难的是：计算能力的增长速度远超内存带宽的增长速度。看看这张关于 CPU FLOPS 翻倍时间与内存带宽翻倍时间的对比表：

