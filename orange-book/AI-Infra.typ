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
  title: "AI-Infra教程",
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

#part("CUDA编程指南")

== CUDA和CUDA编程指南

CUDA是由NVIDIA开发的一款并行计算平台和编程模型，通过利用GPU的强大性能实现了计算性能的显著提升。它允许开发者加速计算密集型应用，广泛应用于深度学习、科学计算和高性能计算（HPC）等领域。

本CUDA编程指南是关于CUDA编程模型以及如何使用CUDA平台在GPU上编写代码的官方、全面资源。本指南涵盖了从CUDA编程模型和CUDA平台到语言扩展的细节，以及如何利用特定的硬件和软件特性。本指南为新手开发者提供了学习CUDA的途径，同时也为开发者在使用CUDA构建应用时提供了必备资源。

== 本指南的组织

即使是主要使用库、框架或DSL的开发者，理解CUDA编程模型以及GPU如何执行代码，对于了解抽象层背后发生的事情也很有价值。本指南从一章关于CUDA编程模型的章节开始，涵盖了任何特定编程语言之外的内容，适用于任何对CUDA工作原理感兴趣的人，即使是非开发者也适用。

本指南分为五个主要部分：

- 第一部分：简单介绍与编程模型概述
  - CUDA编程模型的语言无关概述以及CUDA平台的简要介绍。
  - 本节适合任何想了解GPU及GPU代码执行概念的人阅读，即使他们不是开发者。
- 第二部分：使用CUDA进行GPU编程
  - 使用 CUDA C++ 编程 GPU 的基础知识。
  - 本节适合任何想入门 GPU 编程的人阅读。
  - 本节旨在教学而非完整讲解，教授CUDA编程中最重要和常见的部分，包括一些常见的性能考量。
- 第三部分：CUDA高级编程
  - 引入了CUDA的一些更高级功能，既实现了细粒度控制，也实现了更多最大化性能的机会，包括在单一应用中使用多块GPU。
  - 本节以第四部分涵盖的功能介绍结束，简要介绍每个功能的目的和功能，并按开发者何时及为何觉得每个功能有用为何排序。
- 第四部分：CUDA特性
  - 本节包含了特定CUDA特性的完整介绍，如CUDA图、动态并行性、与图形API的互作性以及统一内存。
  - 当需要了解特定CUDA功能的全貌时，应查阅该部分。在可能的情况下，我们已在前几节中谨慎介绍并阐述了本节涵盖的内容的动机。
- 第五部分：技术附录
  - 技术附录提供了关于 CUDA 的 C++高级语言支持、硬件特定规范及其他技术规范的一些参考文档。
  - 本节旨在作为对 CUDA 元素语法、语义和技术行为的具体描述的技术参考。

第 1 至 3 部分为新手开发者提供引导式学习体验，同时也为任何经验水平的 CUDA 开发者提供有用见解和最新信息。

第 4 部分和第 5 部分提供了关于具体功能和详细主题的丰富信息，旨在为需要了解更多细节的开发者提供一个精心策划、组织良好的参考资料，帮助他们在编写 CUDA 应用时获得更多信息。

#chapter("CUDA简介", image: image("./orange2.jpg"), l: "cuda-intro")

== CUDA简介

=== 图形处理单元（Graphics Processing Unit）

图形处理单元（GPU）最初作为一种专用的3D图形处理器诞生，最初作为固定功能硬件，用于加速实时3D渲染中的并行运算。随着几代的推移，GPU变得更加可编程。到2003年，图形流水线的某些阶段实现了完全可编程，可以为3D场景或图像的每个组成部分并行运行自定义代码。

2006年，NVIDIA推出了计算统一设备架构（Compute Unified Device Architecture，CUDA），使任何计算工作负载能够独立于图形API使用GPU的吞吐量能力。

此后，CUDA和GPU计算被用于加速几乎所有类型的计算工作负载，从流体力学或能源传输等科学仿真到数据库和分析等商业应用。此外，GPU的能力和可编程性为从图像分类到生成式人工智能（如扩散模型或大语言模型）等新算法和技术的发展奠定了基础。

=== 使用GPU的好处

在类似价格和功耗范围内，GPU提供了远高于CPU的指令吞吐量和内存带宽。许多应用利用这些功能，在GPU上运行速度远快于CPU（参见GPU应用）。其他计算设备，如FPGA，也非常节能，但编程灵活性远不及GPU。

GPU和CPU的设计目标不同。CPU设计上擅长以最快速度执行串行操作序列（称为线程），并可并行执行数十个线程；而GPU则设计为并行执行数千线程，以单线程性能下降为代价以实现更高的总吞吐量。

GPU专注于高度并行计算，并分配更多晶体管用于数据处理单元，而CPU则投入更多晶体管用于数据缓存和流量控制。@gpu-devotes-more-transistors-to-data-processing 展示了CPU与GPU芯片资源分布的示例。

#figure(
  image("ai-infra-figures/gpu-devotes-more-transistors-to-data-processing.png"),
  caption: [GPU将更多晶体管用于数据处理],
) <gpu-devotes-more-transistors-to-data-processing>

=== 快速开始

利用GPU提供的计算能力有很多种方式。本指南涵盖了CUDA GPU平台的编程，使用高级语言如C++。然而，有很多方法在应用中使用GPU，而无需直接编写GPU代码。

通过专门的库，来自多个领域的算法和例程不断增加。当库已经实现——尤其是NVIDIA提供的库——使用它通常比从零重写算法更高效、更高效。像cuBLAS、cuFFT、cuDNN和CUTLASS这样的库只是帮助开发者避免重复实现成熟算法的几个例子。这些库还为每种GPU架构进行了优化，提供了生产力、性能和可移植性的理想平衡。

还有一些框架，尤其是用于人工智能的框架，提供GPU加速的构建模块。许多这些框架通过利用上述GPU加速库实现了加速。

此外，领域特定语言（DSL）如NVIDIA的Warp或OpenAI的Triton可直接编译在CUDA平台上运行。这提供了比本指南中介绍的高级语言更高级的GPU编程方法。

NVIDIA加速计算中心包含教授GPU和CUDA计算的资源、示例和教程。

== 编程模型

本章以高层次介绍CUDA编程模型，区别于任何语言。这里介绍的术语和概念适用于任何支持的编程语言中的CUDA。后续章节将用C++说明这些概念。

=== 异构系统（Heterogeneous Systems）

CUDA编程模型假设一个异构计算系统，即包含GPU和CPU的系统。CPU和直接连接的内存分别称为主机和主机内存。GPU及其直接连接的内存分别称为设备和设备内存。在某些系统单芯片（SoC）系统中，这些元件可能是单一封装的一部分。在较大的系统中，可能存在多个CPU或GPU。

CUDA应用程序在GPU上执行部分代码，但应用程序总是从CPU开始执行。主机代码（即运行在CPU上的代码）可以使用CUDA API在主机内存和设备内存之间复制数据，启动GPU上的代码执行，并等待数据复制或GPU代码完成。CPU和GPU可以同时执行代码，通常通过最大化CPU和GPU的利用率来获得最佳性能。

应用程序在GPU上执行的代码被称为设备代码，而在GPU上调用执行的函数，出于历史原因，被称为*核*。启动运行核的行为称为启动核。核启动可以看作是启动多个线程，在GPU上并行执行核代码。GPU线程的运算方式类似于CPU线程，但存在一些对正确性和性能至关重要的差异，这些将在后续章节中详细介绍。

=== GPU硬件模型

像任何编程模型一样，CUDA依赖于底层硬件的概念模型。在CUDA编程中，GPU可以被视为一组流式多处理器（Streaming Multiprocessors，SM），这些SM被组织成称为图形处理集群（GPC）的组。每个SM包含一个本地寄存器文件、统一的数据缓存以及多个执行计算的功能单元。统一的数据缓存为共享内存和L1缓存提供物理资源。统一数据缓存的分配可通过运行时配置L1和共享内存。不同类型的内存大小以及SM中功能单元数量可能因GPU架构而异。

#tip[
  GPU 的实际硬件布局或其物理执行编程模型的方式可能会有所不同。这些差异不会影响使用 CUDA 编程模型编写的软件的正确性。
]

#figure(
  image("ai-infra-figures/gpu-cpu-system-diagram.png"),
  caption: [GPU包含多个流式多处理器（SM），每个SM包含许多功能单元。图形处理集群（GPC）是SM的集合。GPU是一组连接到GPU内存的GPC。CPU通常有多个核心和一个连接系统内存的内存控制器。CPU和GPU通过PCIe或NVLINK等进行连接。],
)

==== 线程块和网格（Thread Blocks and Grids）

当应用程序启动一个核时，它会使用许多线程，通常是数百万线程。这些线程被组织成块。一组线程组成的块被称为线程块，这或许并不令人意外。线程块被组织成网格。网格中所有线程块的大小和尺寸都相同。 @grid-of-thread-blocks 展示了线程块网格的示意图。

#figure(
  image("ai-infra-figures/grid-of-thread-blocks.png"),
  caption: [线程块网格。每个箭头代表一个线程（箭头数量并不代表实际线程的数量）。],
) <grid-of-thread-blocks>

线程块和网格可以是1维、2维或3维的。这些维度可以简化单个线程与工作单元或数据项的映射。

当核被启动时，会使用特定的执行配置启动，该配置指定了网格和线程块的尺寸。执行配置还可以包含可选参数，如集群大小、流和SM配置，这些将在后续章节中介绍。

通过内置变量，每个执行核的线程都可以确定其在包含这个线程的块中的位置，以及该块在包含这个线程块的网格中的位置。线程还可以利用这些内置变量确定线程块的尺寸以及核启动时的网格。这为每个线程在所有运行核的线程中拥有唯一身份。该身份常用于确定线程负责哪些数据或运算。

线程块的所有线程都在同一个SM中执行。这使得线程块内的线程能够高效地通信和同步。线程块内的线程都可以访问芯片上的共享内存，用于线程块中的不同线程交换信息。

一个网格可能由数百万个线程块组成，而执行网格的GPU可能只有数十甚至数百个SM。线程块的所有线程都由单个SM执行，并且在大多数情况下会在该SM上运行直到完成。线程块之间无法保证调度，因此线程块不能依赖其他线程块的结果，因为在该线程块完成之前，其他线程块可能无法调度。@thread-block-scheduling 展示了网格线程块如何分配给 SM 的示例。

#figure(
  image("ai-infra-figures/thread-block-scheduling.png"),
  caption: [每个SM有一个或多个活跃线程块。在这个例子中，每个SM同时调度三个线程块。对于网格中线程块分配给SM的顺序没有保证。],
) <thread-block-scheduling>

CUDA编程模型使任意大的网格能够在任何大小的GPU上运行，无论它只有一个SM还是数千个SM。为此，CUDA编程模型（除少数例外）要求不同线程块中线程之间不存在数据依赖关系。也就是说，线程不应依赖于同一网格中不同线程块内的线程结果或与线程同步。线程块内的所有线程同时运行在同一个SM上。网格内的不同线程块在可用SM之间被调度，并可按任意顺序执行。简而言之，CUDA编程模型要求可以任意顺序执行线程块，无论是并行还是串行。

===== 线程块组成的簇

除了线程块外，具备9.0及以上计算能力的GPU还有一种称为集群的可选分组层级。集群是一组线程块，像线程块和网格一样，可以布局为一维、二维或三维。 图 5 展示了线程块网格，也被组织成簇。指定簇不会改变网格中的网格尺寸或线程块的索引。



== CUDA平台

#chapter("使用CUDA对GPU进行编程", image: image("./orange2.jpg"), l: "cuda-programming")

== CUDA C++简介

== 编写CUDA SIMT核

== 异步执行

== 统一内存和系统内存

== NVCC：NVIDIA CUDA Compiler

#chapter("CUDA高级用法", image: image("./orange2.jpg"), l: "cuda-advanced")

== 高级CUDA接口和特性

== 高级CUDA核编程

== CUDA驱动接口

== 多GPU编程

== CUDA特性游览

#chapter("CUDA特性", image: image("./orange2.jpg"), l: "cuda-features")

== 统一内存

== CUDA图

== 流序内存分配器

== Cooperative Groups

== 程序依赖启动和同步

== 绿色上下文

== 懒加载

== 错误日志管理

== 异步屏障

== 管道（流水线）

== 异步数据复制

== Work Stealing with Cluster Launch Control

== L2缓存控制

== 内存同步域

== 进程间通信

== 虚拟内存管理

== 扩展GPU内存

== CUDA动态并行

== CUDA和API的交互特性

== 驱动入口访问

#chapter("相关技术附录", image: image("./orange2.jpg"), l: "cuda-appendix")

== 计算能力

== CUDA环境变量

== C++语言支持

== C++语言扩展特性

== 浮点计算

== 设备可调用API与内部原理

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
- *内存受限 (Memory-bound)*（例如：逐元素运算（激活函数、Dropout、Masking），以及归约操作（Softmax、LayerNorm、求和等）……）

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

通常，大家会依赖一堆以前可能奏效或者在推特上看到的"大杂烩"技巧。"用原地（in-place）运算！把梯度设为 None！装 PyTorch 1.10.0 别装 1.10.1！"

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

优化深度学习系统的一个视角是：我们希望尽可能让系统处于"计算受限"状态（compute-bound regime）。你花大价钱买了那 312 Teraflops（每秒万亿次浮点运算）的算力，理想情况下，你就应该实打实地用到这 312 Teraflops。所以，为了让你昂贵的矩阵乘法运算"值回票价"，你需要减少花在其他环节上的时间。

但为什么要专注于最大化计算，而不是（比如说）内存带宽呢？原因很简单——你可以降低开销或内存成本，但（在大多数情况下）如果不改变实际执行的运算逻辑，你是无法减少所需的计算量的。

让"最大化计算利用率"变得更加困难的是：计算能力的增长速度远超内存带宽的增长速度。看看这张关于 CPU FLOPS 翻倍时间与内存带宽翻倍时间的对比表：

#figure(
  image("ai-infra-figures/bw_moores_law.png"),
  caption: [CPU FLOPS翻倍时间与内存带宽翻倍时间的对比表],
)

一种理解计算的方式是把它看作工厂。我们向工厂发送指令（开销，overhead），发送材料（内存带宽，memory-bandwidth），所有这些都是为了保持工厂高效运行（计算，compute）。

#figure(
  image("ai-infra-figures/factory.png"),
  caption: [看作工厂],
)

因此，如果工厂效率提升速度超过材料供应速度，工厂就更难达到峰值效率。

#figure(
  image("ai-infra-figures/factory_doubled.png"),
  caption: [即使我们工厂规模（FLOPS）翻倍——如果带宽跟不上，性能也不会翻倍],
)

除了意味着机器学习系统工程师的永久就业保障外，这种计算运用困难的增加也使得理解瓶颈变得更加重要。

关于 FLOPS 还有一点补充。现代机器学习加速器都配备了专门用于矩阵乘法的硬件，比如英伟达的"张量核心"（Tensor Cores）。

#figure(
  image("ai-infra-figures/a100_specs.png"),
  caption: [A100规范],
)

所以，如果你不做矩阵乘法，你只能达到19.5万亿次浮点（teraflops），而不是标称的312。注意，这并非GPU独有——事实上，TPU的通用性甚至比GPU还低。

GPU在非矩阵乘法上速度明显慢，乍看之下可能有问题——那我们的其他算子，比如LayerNorm或激活函数呢？事实上，这些算子只是用FLOPS来四舍五入的误差。例如，我们来看这张论文中BERT上不同算子类型FLOP计数的表格，其中"Tensor contraction"=matmuls。

#figure(
  image("ai-infra-figures/bert_flops.png"),
  caption: [],
)

你可以看到，我们的非矩阵相乘运算只占flops的0.2%，所以GPU计算非矩阵相乘运算慢15倍也无关紧要。

但在这种情况下，归一化运算和逐点运算实际上比我们的矩阵相乘运算低250倍和700倍的FLOPS。

那么，为什么我们的非矩阵相乘算子花费的时间远远超过应有的？

回到我们的比喻，罪魁祸首往往是将材料运送到工厂和从工厂运回材料所需的时间。换句话说，就是内存带宽。

=== 带宽

带宽成本本质上是将数据从一个地点传输到另一个地方所支付的成本。这可能是将数据从CPU移动到GPU，从一个节点移动到另一个节点，甚至是从CUDA全局内存到CUDA共享内存。尤其是最后这一点，我们这里将重点关注，通常被称为"带宽成本"或"内存带宽成本"。

另外两个（通常分别称为"数据传输成本"和"网络成本"）也很重要，但不是本章重点。

为了理解内存带宽的代价，我们回到工厂的比喻。

虽然我们的工厂是实际工作的场所，但它不适合作为散装储存单元。很大一部分原因是，由于我们在这里做的是实际工作，所有存储都为快速使用（SRAM）而优化，而不是大量存储。

那么，我们把实际结果和材料存放在哪里？典型做法是建仓库，可能选在土地便宜且空间充足的地方（DRAM）。然后，我们可以向工厂（内存带宽）运输物资。

#figure(
  image("ai-infra-figures/factory_bandwidth.png"),
  caption: [],
)

将数据搬运到计算单元和从计算单元搬运回来所产生的成本被称为"内存带宽"。顺便说一句，你GPU的DRAM是`nvidia-smi`显示的，也是导致你那个"CUDA Out of Memory"错误的主要原因。

需要注意的是，每次我们执行GPU核函数时，都需要将数据在GPU的DRAM（也就是仓库）和计算单元之间来回搬运。

现在，想象一下当我们执行像 `torch.cos` 这样的一元运算时会发生什么。我们需要先将数据从存储运送到仓库，然后对每条数据进行少量计算，再将该存储运回去。运输东西成本相当高。因此，我们几乎所有时间都花在传输数据上，而非实际计算本身。

由于我们把所有时间都花在内存带宽上，这样的作称为内存受限运算，这意味着我们不会花太多时间在计算上。

好吧，这并不理想。我们能做些什么？让我们看看一组算子可能是什么样子。

#figure(
  image("ai-infra-figures/multi_operators.png"),
  caption: [一系列逐点运算的表现形式],
)

嘿！这是个非常愚蠢的安排。为什么我们要反复把同样的数据发送到全局内存，然后又返回计算单元？我们应该把数据留在工厂，完成所有计算，然后再送回去！

#figure(
  image("ai-infra-figures/operator_fusion.png"),
  caption: [我们不是把三角形数据送回全局内存再读一次，而是一次性完成所有运算。],
)

这就是算子融合——深度学习编译器中最重要的优化。简单来说，我们不是把数据写入全局内存再读一次，而是通过同时执行多次计算来避免额外的内存访问。

例如，如果我们执行`x.cos().cos()`，通常需要执行 4 次全局读写。

```python
x1 = x.cos() # Read from x in global memory, write to x1
x2 = x1.cos() # Read from x1 in global memory, write to x2
```

但算子融合只需两次全局读写！所以算子融合会让它加速2倍。

```python
x2 = x.cos().cos() # Read from x in global memory, write to x2
```

好多了。

有几个需要注意的地方让这件事变得有些棘手。首先，GPU需要知道当前运算接下来会发生什么。所以，你不能在eager-mode下进行优化，也就是PyTorch一次只运行一个算子。其次，我们实际上需要为此生成CUDA代码，这会带来全新的问题。

并非所有算子融合都像逐点运算一样简单。你可以将逐点运算算子融合到归约运算上，或者将逐点运算算子融合到矩阵乘法上。甚至矩阵乘法本身也可以看作是广播乘法融合，然后是归约。

如果你有兴趣编写自定义CUDA内核，这很可能是你能看到最大收益的地方。任意两个PyTorch算子都可能实现融合，从而节省它们之间向全局内存读写的内存带宽成本。此外，许多现有编译器通常可以执行"简单"的融合——NVFuser和XLA就是两个例子。然而，自动化系统无法与人类的创造力抗衡，所以如果你想自己尝试编写一些自定义CUDA内核，Triton是一个很好的起点。

最后，算子融合带来了一些令人惊讶的结果。首先，融合后的`x.cos().cos()`与单独调用`x.cos()`所花费的时间几乎完全相同。这也是为什么几乎所有激活函数的成本都相同，尽管 gelu 显然包含了远多于 relu 的运算。

==== 关于内存带宽成本的推理

在判断算子是否受内存带宽限制时，计算器能帮上很大忙。

对于简单的算子，直接推理内存带宽是可行的。例如，A100 拥有 1.5 TB/秒的全局内存带宽，计算能力为 19.5 TB/秒。所以，如果你使用 32 位浮点数（即 4 字节），你可以在GPU执行20万亿次运算的这段时间内可以加载4000亿个数值。此外，要执行一个简单的一元算子（比如将张量乘以2），我们实际上需要将张量写回全局内存。

所以...除非你能用一元算子做大约一百个运算，否则你花在内存访问上的时间会超过实际计算。

如果你拿一个 PyTorch 函数，比如

```python
def f(x: Tensor[N]):
    for _ in range(repeat):
        x = x * 2
    return x
```

并用算子融合编译器进行基准测试，我们就能计算不同重复值下的 FLOPS 和内存带宽。增加重复是一种简单增加计算量的方法 ，同时不增加内存访问次数——这也称为增加*计算强度*。

具体来说，假设我们对这段代码进行基准测试，计算每秒执行的迭代次数。然后，作为N（张量大小）的函数，我们执行 `2*N` 次内存访问，`N*repeat` FLOP。因此，实现的内存带宽为 `bytes_per_elem * 2 * N * itrs_per_second` ，FLOPS 为 `N * repeat * itrs_per_second`。

现在，让我们绘制计算强度变化时的运行时间、浮点率和内存带宽。注意所有内容均为对数对数刻度（log-log scale）。

#figure(
  image("ai-infra-figures/microbench.png"),
  caption: [],
)

首先， 注意运行时间直到我们进行 64 乘法时才明显增加。这意味着在那之前，我们主要受内存带宽限制——我们的计算大多处于空闲状态。

因此，我们一开始只能实现可怜的 0.2 teraflops。随着计算强度的加倍，这个数值线性增长，直到接近峰值 9.75 teraflops。 一旦接近峰值兆浮点，我们就被视为"计算受限"状态。

最后，你可以看到我们的内存带宽从接近峰值开始，随着计算强度的增加，带宽开始下降。这正是我们应该预期的情况，因为我们花在实际计算上的时间越来越多，而不是访问内存。

在这种情况下，很容易判断什么时候是计算限制，什么时候是内存限制。对于 `repeat < 32`，我们正在饱和内存带宽，而计算能力却被低估。相反，`repeat > 64` 时，我们会发现计算量已经饱和（即接近峰值 FLOPS），而使用的内存带宽开始下降。

对于大型系统来说，往往很难判断你是计算受限还是内存带宽受限，因为它们包含了计算受限和内存受限的成分混合。

衡量你计算受限程度的一个常见方法是将你实现的 FLOPS 占峰值 FLOPS 的百分比来衡量。比如，如果你已经达到了80%的峰值 FLOPS，那你就知道至少80%的计算能力受限，这已经相当不错了！剩下的时间大概都用来做内存带宽操作。

不过，除了内存带宽消耗之外，还有一个可能导致GPU不"起飞"的因素。

==== 开销（Overhead）

开销是指你的代码花时间做任何不传输张量或计算的事情。比如说，使用 Python 解释器的时间？开销。在 PyTorch 框架中花了多少时间？开销。启动 CUDA 内核（但不执行）花了多少时间？也是...开销。

开销之所以成为一个严重的问题，主要原因是现代 GPU 速度非常快。A100 每秒可执行 312 万亿次浮点运算（312 万亿次浮点运算）。相比之下，Python 真的非常慢。本地基准测试，Python 每秒可完成 3200 万次加法。

这意味着在 Python 能完成一次 FLOP 的时间里，A100 可能已经吞噬了 975 万次 FLOP。

更糟的是，Python解释器甚至不是唯一的开销来源——像PyTorch这样的框架在你到达内核之前，还有许多层调度。如果你用PyTorch做同样的实验，我们每秒只能做28万次作。当然，微小的张量不是PyTorch的用途，但......如果你用的是微小的张量（比如科学计算），你可能会觉得PyTorch比C++慢得多。

例如，看看PyTorch进行单次加法的火焰图配置文件。就是那个盒子？这才是实际计算的对象。其他一切都是纯粹的开销。

#figure(
  image("ai-infra-figures/flamegraph.png"),
  caption: [火焰图],
)

鉴于此，你可能会对有人使用PyTorch感到震惊，但请记住，现代深度学习模型往往执行着大规模运算。此外，像PyTorch这样的框架是异步执行的。也就是说，当PyTorch运行CUDA核时，它可以继续并排队更多CUDA核。所以，只要PyTorch能"领先"CUDA核，大部分框架开销就能完全隐藏！

#figure(
  image("ai-infra-figures/overhead.png"),
  caption: [如果我们的GPU算子足够大，那么CPU可以运行在GPU前面（因此CPU开销无关紧要）。另一方面，如果GPU算子太小，GPU大部分时间就会变成昂贵的镇纸。],
)

那么，你怎么判断自己是否处于这种状态？由于开销通常不会随问题大小增长（而计算和内存则会），最简单的判断方法是直接增加数据大小。如果这不能按比例增加运行时间，那你就只能负担开销了。例如，如果你的批次规模翻倍，但运行时间只增加了10%，那很可能是开销负担。

另一种方法是使用 PyTorch 分析器。这里，粉色线条实际上显示了 CPU 内核与 GPU 内核的匹配情况。

#figure(
  image("ai-infra-figures/overhead_tracer.png"),
  caption: [GPU在等待CPU开销时有很多空隙],
)

#figure(
  image("ai-infra-figures/no_overhead.png"),
  caption: [我们的CPU运行远远领先于显卡],
)

另外补充一下——`nvidia-smi`里的"GPU-Util"（不是"Volatile GPU-Util"）条目基本上是测量底排实际运行GPU内核的百分比。这也是另一种很好的俯视方式。

这种开销主要源于像PyTorch这样的框架拥有的各种灵活性。本质上，需要花大量时间去"弄清楚该怎么做"。

这可能是Python（查找属性或调度到正确函数）或PyTorch的代码（PyTorch的调度器全部）。例如，当你执行`a + b`时，需要执行以下步骤。

+ Python需要查找`__add__`在`a`上调度到什么。
+ PyTorch需要确定张量的许多属性（如dtype、device，以及是否需要autograd），以决定调用哪个内核。
+ PyTorch需要真正启动内核。

从根本上说，这种开销来自于每个环节都能灵活地做不同的事情。如果你不需要这种灵活性，解决这种灵活性的一种方法是用 `jit.trace`、`FX` 或 `jax.jit` 来追踪它。或者，你也可以用更低级别的 CUDA Graphs 来实现。

不幸的是，这也意味着失去灵活性。我很期待的一种方法，可以让我们两全其美，就是通过在虚拟机层面进行内省，写出更接近"真正"JIT的作品。更多信息请参见TorchDynamo。

=== 结论

如果你想加快深度学习系统的速度，最重要的是了解模型中的瓶颈是什么。这个瓶颈决定了加速系统的最佳方式。

我经常看到研究人员和其他想加快PyTorch代码速度的人，在不了解自己所处的环境的情况下盲目尝试。

