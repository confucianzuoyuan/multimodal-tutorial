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
  title: "大语言模型、强化学习和多模态教程",
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

// #set par(leading: 1pt)

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
#let colblue(x) = text(fill: blue, $#x$)

#set list(marker: ([•], [‣]))

#set math.mat(delim: "[")

// Display inline code in a small box
// that retains the correct baseline.
#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)

#part("大语言模型")

#chapter("microgpt", image: image("./orange2.jpg"), l: "microgpt")

"此文件包含了完整算法，其余一切不过是效率优化。"——Andrej Karpathy

这 200 行代码中运行的数学运算，与 ChatGPT、Claude、Gemini 乃至所有基于 Transformer 架构的语言模型所执行的运算完全一致。区别仅在于规模与速度——而非算法本身。

以下是整个程序的功能：

```
┌─────────────────────────────────────────────────────┐
│  1. DATASET: Load 32,000 human names ("emma", ...)  │
│  2. TOKENIZER: Map each character → integer ID      │
│  3. MODEL: Build a tiny GPT (4,192 parameters)      │
│  4. TRAIN: Show it names, adjust parameters         │
│  5. GENERATE: Ask it to invent new names            │
└─────────────────────────────────────────────────────┘
```

训练完成后，模型能生成从未见过的、听起来合理的名字，比如"Aalina"或"Relyn"。它纯粹通过示例学会了英文名字的统计规律——哪些字母会跟在哪些字母后面，名字如何开头和结尾。



== 数据集

大语言模型的燃料是文本数据流，这些数据可选择性地划分为一组文档。在生产级应用中，每个文档可能是一个互联网网页，但对于microgpt，我们使用一个更简单的示例：32,000个名字，每行一个：

```python
# Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # list[str] of documents
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

数据集看起来像这样。每个名字就是一个文档：

```
emma
olivia
ava
isabella
sophia
charlotte
mia
amelia
harper
... (~32,000 names follow)
```

模型的目标是学习数据中的模式，然后生成具有相同统计模式的新文档。作为预览，到脚本结束时，我们的模型将生成（"幻觉"出！）新的、听起来合理的名字。提前看一下，我们会得到：

```
sample  1: kamon
sample  2: ann
sample  3: karai
sample  4: jaire
sample  5: vialan
sample  6: karia
sample  7: yeran
sample  8: anna
sample  9: areli
sample 10: kaina
sample 11: konna
sample 12: keylen
sample 13: liole
sample 14: alerin
sample 15: earan
sample 16: lenne
sample 17: kana
sample 18: lara
sample 19: alela
sample 20: anton
```

看起来并不起眼，但从 ChatGPT 这类模型的视角来看，你与它的对话不过是一份形式奇特的"文档"。当你用提示词初始化这份文档时，模型视角下的回应本质上只是基于统计的文档补全。

== 分词器

神经网络并不理解文本，它们只理解数字。首要任务就是将字符转换为整数。

在底层，神经网络处理的是数字而非字符，因此我们需要一种方法将文本转换为整数标记ID序列，并能反向还原。像 tiktoken（GPT-4 使用的）这样的生产级分词器会按字符块处理以提高效率，但最简单的分词器只需为数据集中的每个唯一字符分配一个整数即可。

```python
# Let there be a Tokenizer to translate strings to discrete symbols and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")
```

在上述代码中，我们收集了数据集中所有不重复的字符（即所有小写字母a-z），将其排序后，每个字母根据其索引获得一个ID。请注意，这些整数值本身没有任何实际意义；每个标记只是一个独立的离散符号。它们完全可以被替换为不同的表情符号，而非0、1、2。此外，我们创建了一个名为`BOS`（序列起始符）的特殊标记，它作为分隔符使用：告诉模型"新文档在此开始/结束"。在后续训练中，每个文档两侧都会包裹`BOS`：`[BOS, e, m, m, a, BOS]`。模型会学习到`BOS`标志着一个新名称的开始，而另一个`BOS`则标志着其结束。因此，我们最终得到包含 27 个标记的词汇表（26 个可能的小写字母 a-z，加上 1 个序列起始符标记）。

数据集包含 32,000 个名字。`uchars`收集所有名字中的每个独特字符并进行排序，从而得到字符到整数的映射：

```
Character:  a  b  c  d  e ... x  y  z
Token ID:   0  1  2  3  4 ... 23 24 25
BOS token:  26
```

`BOS`（序列起始）是一个特殊token，用于标识"名称从此处开始"和"名称在此处结束"。像"emma"这样的名称会转化为token序列`[26, 4, 12, 12, 0, 26]`——先是 BOS，接着是 e-m-m-a，最后再次出现 BOS。第二个 BOS 充当结束token：

```
 BOS   e    m    m    a   BOS
[26]  [4]  [12] [12] [0] [26]
  ↑                        ↑
start                    end
```

这是一个字符级分词器。像 GPT-4 这样的生产模型使用子词分词器（BPE），词汇量约为~200,000 个词元，其中像"the"这样的常见词是一个单独的词元，而罕见词则会被拆分成多个部分。其原理是相同的：将文本映射为整数序列。

== 自动微分（Autograd）

这是代码中最优雅的部分。`Value` 类实现了自动微分——即通过任意数学运算链计算导数的能力。这正是神经网络训练得以实现的关键。

导数为什么这么重要？

训练神经网络意味着寻找能使损失函数最小化的参数值。损失函数衡量的是模型预测的偏差程度。为了降低损失，我们需要了解：对于每个参数，如果对其进行微小调整，损失会增加还是减少？这就是损失函数对每个参数的导数（梯度）。

```
Parameter: 0.5
                        ┌─────────────┐
  nudge right → 0.501 ──┤             ├── loss = 2.38   ← went up
  original  →   0.500 ──┤    model    ├── loss = 2.37
  nudge left →  0.499 ──┤             ├── loss = 2.36   ← went down
                        └─────────────┘

  Gradient is positive → move the parameter left (decrease it)
```

由于有 4,192 个参数，我们需要 4,192 个梯度。如果通过逐个微调每个参数来计算这些梯度，则需要进行 4,192 次正向传播。而反向传播只需一次反向传播即可计算出所有梯度。这就是自动梯度的魔力。

```python
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

每个`Value`存储三项内容：计算结果（`data`）、其梯度（`grad`，在反向传播过程中填充）以及生成方式（`_children`和`_local_grads`）。这些内容共同构成一个计算图——即每项数学运算的记录。

当你写下`c = a + b`时，生成的`Value`会记住它是通过加法从`a`和`b`得出的：

```
  a (data=3.0) ──┐
                 ├──(+)──→ c (data=5.0)
  b (data=2.0) ──┘

  children: (a, b)
  local_grads: (1, 1)     ← derivative of (a+b) w.r.t. a is 1,
                              derivative of (a+b) w.r.t. b is 1
```

对于乘法，局部梯度有所不同：

```python
def __mul__(self, other):
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

```
  a (data=3.0) ──┐
                 ├──(×)──→ c (data=6.0)
  b (data=2.0) ──┘

  children: (a, b)
  local_grads: (2.0, 3.0)  ← d(a×b)/da = b = 2.0
                               d(a×b)/db = a = 3.0
```

这是微积分中的乘法法则：$a times b$对$a$的导数是$b$，反之亦然。`Value`类中的每项运算都包含其自身的导数规则。

`backward()`方法以反向顺序遍历计算图，并利用链式求导法则累计梯度：如果$z$依赖于$y$，而$y$又依赖于$x$，则$(d z)/(d x) = (d z)/(d y) dot.c (d y)/(d x)$。

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

首先， `build_topo`执行拓扑排序——它排列所有节点，使得每个节点都出现在其子节点之后。然后，梯度从损失（其梯度按定义为 1）反向流经每个操作，最终到达每个参数。

#pagebreak()

```
Forward pass (left to right):  compute values
─────────────────────────────────────────────────────────────→

  a=3.0 ──(×)──→ d=6.0 ──(+)──→ f=7.0 ──(-log)──→ loss=−1.95
  b=2.0 ──┘      e=1.0 ──┘

←─────────────────────────────────────────────────────────────
Backward pass (right to left): compute gradients

  loss.grad = 1.0

  f.grad = 1.0 × (−1/f.data) = −0.143        ← chain rule through -log
  d.grad = f.grad × 1 = −0.143                 ← chain rule through +
  e.grad = f.grad × 1 = −0.143                 ← chain rule through +
  a.grad = d.grad × b.data = −0.143 × 2 = −0.286  ← chain rule through ×
  b.grad = d.grad × a.data = −0.143 × 3 = −0.429  ← chain rule through ×
```

`+=`中的`child.grad += local_grad * v.grad`很重要——当一个值在多个操作中被使用时，其梯度会累积来自所有操作的贡献。这处理了一个参数通过多条路径影响损失的情况。

这正是 PyTorch、TensorFlow 和 JAX 所实现的反向传播算法。区别在于这些框架基于张量（多维数值数组）运行，并可在 GPU 上执行。而 microGPT 则对单个标量值进行操作，这使得其速度慢约~1,000,000 倍，但在概念上完全一致。

训练神经网络需要梯度：对于模型中的每个参数，我们需要知道"如果我将这个数值稍微调高一点，损失值会上升还是下降？变化幅度是多少？"计算图有许多输入（模型参数和输入词元），但最终汇聚成一个标量输出：损失值（我们将在下文准确定义损失值）。反向传播从该单一输出开始，沿计算图逆向推进，计算损失值相对于每个输入的梯度。这依赖于微积分中的链式法则。在实际应用中，PyTorch等库会自动处理这一过程。在此，我们通过一个名为`Value`的类从头实现该功能：

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
```

这是数学和算法上最密集的部分。简单来说，一个`Value`包装了一个单一的标量数字（`.data`），并追踪它是如何被计算出来的。可以把每个操作想象成一个小乐高积木：它接收一些输入，产生一个输出（前向传播），并且知道其输出相对于每个输入的变化情况（局部梯度）。这就是自动微分从每个积木中需要的全部信息。其余的一切都只是链式法则，将这些积木串联起来。

每次使用`Value`对象进行数学运算（加法、乘法等）时，结果都会生成一个新的`Value`，它会记住其输入值（`_children`）以及该运算的局部导数（`_local_grads`）。例如，`__mul__`记录了$(partial (a dot.c b))/(partial a) = b$和$(partial (a dot.c b))/(partial b)$。完整的积木块集合如下：

#figure(
  table(
    columns: 3,
    [运算], [前向传播], [梯度],
    [`a + b`], [$a+b$], [$(partial)/(partial a)=1,(partial)/(partial b)=1$],
    [`a * b`], [$a dot.c b$], [$(partial)/(partial a)=b,(partial)/(partial b)=a$],
    [`a ** n`], [$a^n$], [$(partial)/(partial a)=n dot.c a^(n-1)$],
    [`log(a)`], [$ln(a)$], [$(partial)/(partial a)=1/a$],
    [`exp(a)`], [$e^a$], [$(partial)/(partial a)=e^a$],
    [`relu(a)`], [$max(0, a)$], [$(partial)/(partial a)=bold(1)_(a>0)$],
  ),
  caption: [运算，前向传播，梯度],
)

`backward()`方法以逆拓扑顺序遍历该图（从损失开始，到参数结束），在每一步应用链式法则。如果损失为$L$，且节点$v$有一个子节点$c$，其局部梯度为$(partial v)/(partial c)$，则：

$
  (partial L)/(partial c) "+=" (partial v)/(partial c) dot.c (partial L)/(partial v)
$

我们从损失节点开始设置```python self.grad = 1```，因为$(partial L)/(partial L)=1$：损失相对于自身的变化率显然为1。由此，链式法则只需沿着每条返回参数的路径，将局部梯度相乘即可。

注意这里的`+=`（累加，而非赋值）。当某个值在计算图中被多处使用（即图存在分支）时，梯度会沿每个分支独立回流，且必须求和。这是多元链式法则的必然结果：若$c$通过多条路经影响$L$，则总导数等于各路径贡献之和。

在`backward()`完成以后，图中的每个`Value`都包含一个`.grad`，其中含有$(partial L)/(partial v)$，这告诉我们如果调整该值，最终损失将如何变化。

以下是一个具体示例。注意`a`被使用了两次（图出现分支），因此其梯度是两条路径之和：

```python
a = Value(2.0)
b = Value(3.0)
c = a * b       # c = 6.0
L = c + a       # L = 8.0
L.backward()
print(a.grad)   # 4.0 (dL/da = b + 1 = 3 + 1, via both paths)
print(b.grad)   # 2.0 (dL/db = a = 2)
```

这正是PyTorch的`.backward()`所提供的：

```python
import torch
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b
L = c + a
L.backward()
print(a.grad)   # tensor(4.)
print(b.grad)   # tensor(2.)
```

这是PyTorch的`loss.backward()`所运行的相同算法，只不过是在标量而非张量（标量数组）上执行——算法上完全一致，但规模更小、更简单，当然效率也低得多。

让我们解读一下上面`.backward()`给出的结果。自动微分计算得出，如果`L = a*b + a`、`a=2`和`b=3`成立，那么`a.grad = 4.0`反映的是`a`对`L`的局部影响程度。当你微调输入`a`时，`L`会朝哪个方向变化？这里，`L`对`a`的导数为`4.0`，意味着若将`a`增加微小量（如`0.001`），`L`将增加约4倍于此的值（0.004）。同理，`b.grad = 2.0`表示对`b`施加相同幅度的调整，会使 L 增加约 2 倍于此的值（0.002）。换言之，这些梯度既指明了每个输入对最终输出（损失值）的影响方向（正负取决于符号），也揭示了影响陡峭程度（梯度幅值）。这使我们能够通过迭代微调神经网络参数来降低损失值，从而提升其预测性能。

有了自动求导机制，我们就可以定义 Transformer 模型了。microGPT 遵循 GPT-2 架构，并做了少量简化。

== 参数

参数是模型的知识。它们是一大堆浮点数（包裹在`Value`中用于自动求导），初始时是随机的，并在训练过程中通过迭代进行优化。每个参数的具体作用将在下面定义模型架构时更加清晰，但目前我们只需要对它们进行初始化。

```python
n_embd = 16     # embedding dimension
n_head = 4      # number of attention heads
n_layer = 1     # number of layers
block_size = 16 # maximum sequence length
head_dim = n_embd // n_head # dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
```

每个参数都初始化为从高斯分布中抽取的小随机数。 `state_dict` 将它们组织成命名矩阵（借用 PyTorch 的术语）：嵌入表、注意力权重、MLP 权重以及最终输出投影。我们还将所有参数展平为单个列表 params ，以便优化器后续可以遍历它们。在我们这个微型模型中，参数总数为 4,192 个。GPT-2 拥有 16 亿参数，而现代 LLM 则拥有数千亿参数。

模型对token做的第一件事就是查找其嵌入向量——一个在模型内部空间中代表该标记的、经过学习的数字向量：

```python
tok_emb = state_dict['wte'][token_id] # token embedding
pos_emb = state_dict['wpe'][pos_id] # position embedding
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

```
Token "e" (id=4)                Position 1
        │                              │
        ▼                              ▼
 ┌─────────────┐               ┌─────────────┐
 │  wte[4]     │               │  wpe[1]     │
 │  (lookup    │               │  (lookup    │
 │   row 4)    │               │   row 1)    │
 └──────┬──────┘               └──────┬──────┘
        │                              │
        ▼                              ▼
 [0.02, -0.05, 0.11, ...]      [0.08, 0.01, -0.03, ...]
        │                              │
        └──────────┬───────────────────┘
                   ▼
              element-wise add
                   │
                   ▼
         [0.10, -0.04, 0.08, ...]
              16 numbers
```

词元嵌入表（`wte`）是一个每行对应一个词条（27 行×16 列）的表格。每行是一个 16 维向量，模型将学习将该向量与该字符的含义关联起来。初始时这些是随机数；在训练过程中，模型会调整它们，使得具有相似角色（如元音）的字符在这个 16 维空间中彼此靠近。

位置嵌入（`wpe`）是一个独立的表格（16 行×16 列），用于编码序列中token出现的位置。模型需要这个信息，因为 Transformer 是并行处理标记的——如果没有位置信息，它就无法区分"ab"和"ba"。将位置嵌入添加到标记嵌入中，使得每个标记的表示既能编码其内容，也能编码其位置。



== 架构

模型架构是一个无状态函数：它接收一个 token、一个位置、参数以及来自先前位置的缓存键/值，并返回模型认为序列中下一个应出现的 token 的对数概率（分数）。我们沿用 GPT-2 架构并做了少量简化：使用 RMSNorm 替代 LayerNorm，取消偏置项，用 ReLU 替代 GeLU。首先，三个小型辅助函数：

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

`linear`是矩阵向量乘法。它接收向量`x`和权重矩阵`w`，并对`w`的每一行计算一个点积。这是神经网络的基本构建模块：一种学习得到的线性变换。

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

`softmax`将原始分数向量（logits）从$-infinity$到$+infinity$的范围转换为概率分布：所有值最终落在$[0,1]$范围内且总和为 1。我们首先减去最大值以保证数值稳定性（这在数学上不会改变结果，但能防止`exp`中的溢出）。

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

`rmsnorm`（均方根归一化）通过重新缩放向量，使其数值具有单位均方根。这能防止激活值在网络中传播时过度增长或收缩，从而稳定训练过程。它是原始 GPT-2 中使用的层归一化（LayerNorm）的简化变体。

现在来看模型本身：

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits
```

该函数在时间上的特定位置（`pos_id`）处理一个token（ID为`token_id`），并结合前几次迭代中由`keys`和`values`的激活值总结的上下文信息（即KV Cache）。以下是逐步执行过程：

嵌入（Embedding）。神经网络无法直接处理像5这样的原始token ID，它只能处理向量（数字列表）。因此，我们为每个可能的token关联一个学习到的向量，并将其作为神经特征输入。token ID和位置 ID分别从各自的嵌入表中查找对应行（`wte`和`wpe`）。这两个向量相加后，为模型提供同时编码令牌内容及其在序列中位置的表示。现代 LLM 通常跳过位置嵌入，转而采用其他基于相对位置的方案，例如`RoPE`。

注意力模块。当前词元被投影为三个向量：查询（Q）、键（K）和值（V）。直观理解，查询表示"我在寻找什么？"，键表示"我包含什么？"，值表示"如果被选中，我能提供什么？"。例如在名字"emma"中，当模型处理第二个"m"并试图预测下一个字符时，它可能会学习到类似"最近出现了哪些元音？"的查询。较早的"e"会生成与这个查询高度匹配的键，从而获得较高的注意力权重，其值（关于元音的信息）就会流向当前位置。键和值会被追加到 KV Cache中，以便后续位置可以访问。每个注意力头会计算其查询与所有缓存键的点积（经$sqrt(d_"head")$缩放），通过`softmax`函数得到注意力权重，再对缓存值进行加权求和。所有注意力头的输出被拼接后通过`attn_wo`进行投影。值得强调的是，注意力模块是位置`t`的词元能够"查看"过去`0..t-1`词元的唯一精确位置。注意力是一种词元通信机制。

MLP块。MLP是"多层感知机"的缩写，它是一个两层前馈网络：先投影到嵌入维度的 4 倍大小，应用 ReLU 激活函数，再投影回原维度。这是模型在每个位置上进行大部分"思考"的地方。与注意力机制不同，这个计算在时间维度上完全局部化 t 。Transformer 将通信（注意力机制）与计算（MLP）交替进行。

残差连接。注意力模块和 MLP 模块都将输出加回到输入中（`x = [a + b for ...]`）。这使得梯度可以直接在网络中流动，从而让更深的模型变得可训练。

输出。最终的隐藏状态通过 `lm_head` 投影到词汇表大小，为词汇表中的每个词元生成一个分数（logit）。在我们的案例中，这仅仅是 27 个数字。分数越高，表示模型认为对应的词元更有可能出现在下一个位置。

你可能会注意到，我们在训练过程中使用了 KV 缓存，这并不常见。人们通常认为 KV 缓存仅用于推理阶段。但从概念上讲，KV 缓存始终存在，即使在训练期间也是如此。在生产实现中，它只是被隐藏在处理序列所有位置的高度向量化注意力计算内部。由于 microgpt 每次只处理一个 token（没有批次维度，也没有并行时间步长），我们显式地构建了 KV 缓存。与典型的推理场景（KV 缓存保存分离的张量）不同，这里的缓存键和值是计算图中活跃的 Value 节点，因此我们实际上会通过它们进行反向传播。

== 训练循环

现在我们将所有部分串联起来。训练循环会重复执行以下步骤：(1) 选取一个文档，(2) 对文档中的词元进行模型前向传播，(3) 计算损失值，(4) 通过反向传播获取梯度，(5) 更新参数。

```python
# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters.
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients.
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
```

让我们逐一解析每个部分：

分词。每个训练步骤会选取一个文档，并在其两侧添加 BOS ：例如，名称"emma"会变成 `[BOS, e, m, m, a, BOS]` 。模型的任务是根据前面的词元预测下一个词元。

前向传播与损失计算。我们逐个将词元输入模型，在过程中逐步构建 KV 缓存。在每个位置上，模型输出 27 个 logits 值，通过 softmax 函数将其转换为概率。该位置的损失即为正确下一个词元的负对数概率：$-log P("target")$这被称为交叉熵损失。直观而言，损失衡量的是预测偏差程度：模型对实际出现的后续内容感到"惊讶"的程度。若模型对正确词元赋予 1.0 的概率，则完全不会感到惊讶，损失为 0；若赋予接近 0 的概率，则模型会非常惊讶，损失趋近于$+infinity$。我们将文档中每个位置的损失取平均值，得到单一的标量损失值。

反向传播。对`loss.backward()`的一次调用会通过整个计算图运行反向传播，从损失函数一路回溯经过`softmax`、模型，直至每个参数。此后，每个参数的`.grad`会告诉我们如何调整它以降低损失。

Adam 优化器。我们本可以只使用 `p.data -= lr * p.grad` （梯度下降法），但 Adam 更智能。它为每个参数维护两个运行平均值： m 追踪近期梯度的均值（动量，类似滚动的球）， v 则追踪近期梯度平方的均值（为每个参数自适应调整学习率）。 `m_hat` 和 `v_hat` 是偏差修正项，用于补偿 m 和 v 初始化为零而需要的预热过程。学习率在训练过程中线性衰减。更新完成后，我们将 `.grad = 0` 重置为下一步做好准备。

经过 1000 步训练，损失从约 3.3（在 27 个 token 中随机猜测：$-log(1/27) approx 3.3$，降至约`2.37`。数值越低越好，最低可达 0（完美预测），因此仍有改进空间，但模型显然正在学习名字的统计规律。

== 推理

训练完成后，我们可以从模型中采样生成新名称。此时参数已冻结，只需循环执行前向传播，将每次生成的词元作为下一个输入重新传入：

```python
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

我们以 `BOS` 词元作为每个样本的起始标记，告知模型"开始生成新名称"。模型输出 27 个 logits 值，我们将其转换为概率分布，并根据这些概率随机采样一个词元。该词元会作为下一个输入重新传入，重复此过程直到模型输出 `BOS` （表示"生成完毕"）或达到最大序列长度。

temperature 参数控制随机性。在 softmax 之前，我们将 logits 值除以温度系数。温度为 1.0 时直接根据模型学习到的分布进行采样。较低的温度（如这里的 0.5）会使分布更加尖锐，让模型更保守，倾向于选择概率最高的选项。当温度趋近于 0 时，模型将始终选择最可能的单个词元（贪婪解码）。较高的温度则会使分布更平坦，生成更多样化但可能连贯性较差的输出。

== 运行

```bash
python train.py

train.py
num docs: 32033
vocab size: 27
num params: 4192
step    1 / 1000 | loss 3.3660
step    2 / 1000 | loss 3.4243
step    3 / 1000 | loss 3.1778
step    4 / 1000 | loss 3.0664
step    5 / 1000 | loss 3.2209
step    6 / 1000 | loss 2.9452
step    7 / 1000 | loss 3.2894
step    8 / 1000 | loss 3.3245
step    9 / 1000 | loss 2.8990
step   10 / 1000 | loss 3.2229
step   11 / 1000 | loss 2.7964
step   12 / 1000 | loss 2.9345
step   13 / 1000 | loss 3.0544
...
```

观察它从~3.3（随机）下降到~2.37。这个数值越低，说明网络对序列中下一个 token 的预测能力越强。训练结束时，训练 token 序列的统计模式知识被提炼到模型参数中。固定这些参数后，我们现在可以生成全新的、虚构的名称。你将再次看到：

```
sample  1: kamon
sample  2: ann
sample  3: karai
sample  4: jaire
sample  5: vialan
sample  6: karia
sample  7: yeran
sample  8: anna
sample  9: areli
sample 10: kaina
sample 11: konna
sample 12: keylen
sample 13: liole
sample 14: alerin
sample 15: earan
sample 16: lenne
sample 17: kana
sample 18: lara
sample 19: alela
sample 20: anton
```

== 真实世界

microgpt 包含了训练和运行 GPT 的完整算法精髓。但在它和 ChatGPT 这样的生产级 LLM 之间，存在着一长串需要改变的东西。这些改变都不会影响核心算法和整体架构，但正是它们让模型能够真正大规模运行。按顺序逐一说明：

数据。生产级模型不再使用 3.2 万个短名称，而是在数万亿个互联网文本 token 上进行训练：网页、书籍、代码等。这些数据会经过去重、质量过滤，并在不同领域间精心混合。

分词器。生产级模型不再使用单个字符，而是采用 BPE（字节对编码）等子词分词器，这些分词器会学习将频繁共现的字符序列合并为单个 token。像 "the" 这样的常见词会成为一个 token，而罕见词则会被拆分成多个片段。这样能形成约 10 万 token 的词汇表，并且效率更高，因为模型在每个位置上能处理更多内容。

Autograd。microgpt 在纯 Python 中操作标量 Value 对象。生产系统使用张量（大型多维数字数组），并在每秒执行数十亿次浮点运算的 GPU/TPU 上运行。像 PyTorch 这样的库处理张量上的自动微分，而像 FlashAttention 这样的 CUDA 内核则融合多个运算以提升速度。数学原理完全相同，只是相当于并行处理大量标量。

架构。microgpt 拥有 4,192 个参数。GPT-4 类模型则有数千亿个参数。总体而言，它看起来是一个非常相似的 Transformer 神经网络，只是更宽（嵌入维度超过 10,000）且更深（超过 100 层）。现代 LLM 还加入了更多类型的积木块，并调整了它们的顺序：例如使用 RoPE（旋转位置嵌入）替代学习位置嵌入，使用 GQA（分组查询注意力）来减少 KV 缓存大小，使用门控线性激活替代 ReLU，以及混合专家（MoE）层等。但注意力（通信）和 MLP（计算）在残差流上交替的核心结构得到了很好的保留。

训练。生产环境中的训练并非每步处理一个文档，而是采用大批量（每步处理数百万个 token）、梯度累积、混合精度（float16/bfloat16）以及精细的超参数调优。训练一个前沿模型需要数千块 GPU 连续运行数月。

优化。microgpt 使用 Adam 优化器配合简单的线性学习率衰减，仅此而已。在大规模场景下，优化本身已成为一门独立学科。模型采用低精度训练（bfloat16 甚至 fp8），并跨大型 GPU 集群运行以提升效率，这又带来了新的数值计算挑战。优化器参数（学习率、权重衰减、beta 参数、预热策略、衰减策略）必须精确调优，而最佳取值取决于模型规模、批次大小和数据集构成。缩放定律（如 Chinchilla 法则）指导如何在模型参数量和训练 token 数量之间分配固定计算预算。在大规模场景下，任何细节失误都可能导致数百万美元的计算资源浪费，因此团队会在启动完整训练前，通过大量小规模实验来预测最佳参数配置。

后训练。训练完成后得到的基础模型（称为"预训练"模型）是一个文档补全器，而非聊天机器人。将其转化为 ChatGPT 需要两个阶段。首先，SFT（监督微调）：只需将文档替换为精心策划的对话，并继续训练。从算法层面看，没有任何变化。其次，RL（强化学习）：模型生成回复，这些回复会获得评分（由人类、另一个"评判"模型或算法完成），模型则从反馈中学习。本质上，模型仍在训练文档，但这些文档现在由模型自身生成的 token 组成。

推理。为数百万用户提供模型服务需要专门的工程架构：批量处理请求、KV 缓存管理与分页（如 vLLM 等）、通过推测解码提升速度、量化处理（使用 int8/int4 而非 float16 以降低内存占用），以及将模型分布到多个 GPU 上。从根本上说，我们仍在预测序列中的下一个 token，但投入了大量工程优化来提升速度。

这些都是重要的工程和研究贡献，但如果你理解了 microgpt，你就掌握了算法的精髓。

== FAQ

模型是否"理解"任何东西？这是个哲学问题，但从机械层面看：并没有魔法发生。模型本质上是一个大型数学函数，它将输入词元映射到下一个词元的概率分布上。在训练过程中，参数会被调整，使得正确的下一个词元概率更高。这算不算"理解"取决于你的定义，但其机制完全包含在上述 200 行代码中。

为什么它能工作？模型拥有数千个可调参数，优化器在每一步都会对它们进行微调以降低损失值。经过大量步骤后，参数会稳定在能够捕捉数据统计规律的值上。以名字为例，这意味着：名字常以辅音开头，"qu"倾向于同时出现，名字很少连续出现三个辅音等。模型并非学习显式规则，而是学习一个恰好能反映这些规律的的概率分布。

这和 ChatGPT 有什么关系？ChatGPT 本质上就是同一个核心循环（预测下一个词元、采样、重复）的巨量扩展版本，再加上后期训练使其具备对话能力。当你与它对话时，系统提示、你的消息和它的回复都只是序列中的词元。模型每次生成一个词元来完成文档，就像 microgpt 完成一个名字那样。

"幻觉"是怎么回事？模型通过从概率分布中采样来生成词元。它没有真实性的概念，只知道哪些序列在训练数据中具有统计合理性。microgpt"幻觉"出像"karia"这样的名字，与 ChatGPT 自信地陈述虚假事实是同一现象。两者都是听起来合理但实际不存在的补全结果。

为什么这么慢？microgpt 在纯 Python 环境中逐个处理标量。单次训练步骤需要数秒。同样的数学运算在 GPU 上可以并行处理数百万个标量，运行速度快出数个数量级。

我能让它生成更好的名字吗？可以。延长训练时间（增加`num_steps`）、扩大模型规模（`n_embd`、`n_layer`、`n_head`），或使用更大的数据集。这些正是大规模应用中同样重要的调节参数。

如果我更换数据集会怎样？模型会学习数据中的任何模式。只要换成城市名称、宝可梦名字、英语单词或短诗的文件，模型就会学习生成这些内容。代码的其他部分无需改动。



#chapter("LLM简介", image: image("./orange2.jpg"), l: "llm-introduction")

== 大语言模型要解决的问题是什么？

大语言模型要解决的问题是#underline[预测下一个token（next token prediction，NTP）]。

#figure(
  image("ntp.svg"),
  caption: [预测下一个token],
)

== 如何训练一个能预测下一个token的模型？

- 训练数据（输入-目标对）长什么样子？
- 模型结构如何设计？神经网络本质上是一个函数，这个函数长什么样子？参数有多少？
- 神经网络接收输入之后，输出和目标之间的损失（差异）怎么衡量？也就是说损失函数怎么设计？
- 如何让损失函数最小化？梯度下降法以及变种：SGD，Adam，AdamW，Muon等优化器
- 梯度下降法需要求损失函数对于参数的导数（梯度），那么梯度如何得来？反向传播算法求参数的梯度：`loss.backward()`。

== 训练数据（输入-目标对）长什么样子？

文本数据："君不见黄河之水天上来"

#figure(
  table(
    columns: 2,
    [输入], [预测目标],
    [君不见黄河], [之],
    [君不见黄河之], [水],
    [君不见黄河之水], [天],
    [君不见黄河之水天], [上],
    [君不见黄河之水天上], [来],
  ),
  caption: [输入和预测目标],
)

可以看到训练数据可以通过程序处理原始文本数据，自动生成，所以叫做"自监督学习"。

也就是说，预测目标不是人类标注而来，而是程序通过切分文本自动标注的。

为了充分利用GPU的并行计算能力，训练数据一般如下组织。

#figure(
  image("llm-figures/2.svg"),
  caption: [数据组织方法],
)

== 模型结构的设计

Decoder-Only Transformer。仅解码器的Transformer架构。

#figure(
  image("llm-figures/3.svg"),
  caption: [GPT-2架构],
)

== 损失函数的设计

LLM本质上是一个分类模型。

从极大似然估计的角度看到LLM。

假设训练数据为"abcd"，那么我们希望下面的概率越大越好。

$
  P_theta ("d"|"abc") dot.c P_theta ("c"|"ab") dot.c P_theta ("b"|"a")
$

其中$theta$是神经网络的参数。最大化上面的式子等价于最大化下面的式子

$
  log { P_theta ("d"|"abc") dot.c P_theta ("c"|"ab") dot.c P_theta ("b"|"a") } \
  = log P_theta ("d"|"abc") + log P_theta ("c"|"ab") + log P_theta ("b"|"a")
$

而最大化上面的式子，等价于最小化下面的式子

$
  -log P_theta ("d"|"abc") - log P_theta ("c"|"ab") - log P_theta ("b"|"a")
$

而上面的这个式子就是#underline[交叉熵损失函数]！

也就是给定输入"ab"，我们希望预测的下一个token属于分类"c"的概率越大越好。

给定输入"abc"，我们希望预测的下一个token属于分类"d"的概率越大越好。

#figure(
  image("llm-figures/4.svg"),
  caption: [LLM本质是个分类模型],
)

#chapter("GPT-2", image: image("./orange2.jpg"), l: "gpt2")

== GPT-2模型结构的定义以及数据集的准备

创建一个文件`gpt_model.py`。

我们首先导入需要的依赖。

```python
import tiktoken # 分词器库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```

=== 数据集的准备

#figure(
  image("llm-figures/5.svg"),
  caption: [分词],
)

接下来我们构建数据集。

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对整个文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本切割成**重叠的**长度为`max_length`的序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

然后我们封装一个工具函数，用来创建训练数据集。

```python
def create_dataloader_v1(
  txt,
  batch_size=4,
  max_length=256,
  stride=128,
  shuffle=True,
  drop_last=True,
  num_workers=0,
):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # 创建dataloader
    dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      drop_last=drop_last,
      num_workers=num_workers,
    )

    return dataloader
```

#figure(
  image("llm-figures/6.png"),
  caption: [数据集],
)

=== 模型结构

接下来我们自顶向下的来定义模型结构。

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
          *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds # 形状：[batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

神经网络本身是一个函数，而神经网络这个函数的定义关键要看`.forward`方法，也就是`.forward`方法定义了神经网络的计算图长什么样子。

==== 词嵌入

首先GPT-2模型的输入`in_idx`是一批token id列表，形状为`(batch_size, seq_len)`。

所以```python batch_size, seq_len = in_idx.shape```这一行可以提取批次的大小`batch_size`和每条数据序列的长度`seq_len`。

而```python tok_embeds = self.tok_emb(in_idx)```将输入的token id列表转换成了形状为`(batch_size, seq_len, emb_dim)`的张量。

`nn.Embedding`的作用是将token id转换成一个向量，如GPT-2会将一个token id转换成一个768个浮点数元素的向量。

#figure(
  image("llm-figures/7.png"),
  caption: [词嵌入],
)

#figure(
  image("llm-figures/8.png"),
  caption: [一条数据和一批数据的词嵌入示例],
)

在PyTorch中，嵌入层实现的功能与执行矩阵乘法的线性层相同；我们使用嵌入层主要是出于计算效率的考虑。

```python
# 假设我们有以下3个训练样本，
# 它们可能代表LLM上下文中的token ID
idx = torch.tensor([2, 3, 1])

# 嵌入矩阵的行数可以通过
# 获取最大的token ID + 1来确定。
# 如果最大的token ID是3，那么我们需要4行，用于可能的
# token ID：0, 1, 2, 3
num_idx = max(idx)+1

# 期望的嵌入维度是一个超参数
out_dim = 5
```

实现一个简单的嵌入层

```python
torch.manual_seed(123)
embedding = torch.nn.Embedding(num_idx, out_dim)
print(embedding.weight) # 查看嵌入权重
# 获取训练示例ID为1的向量表示
print(embedding(torch.tensor([1])))
# 获取训练示例ID为2的向量表示
print(embedding(torch.tensor([2])))
# 查找一批ID的向量表示
idx = torch.tensor([2, 3, 1])
print(embedding(idx))
```

现在我们用独热编码和`nn.Linear`实现和上面的嵌入层一样的功能。

首先将token ID转换为独热编码

```python
onehot = torch.nn.functional.one_hot(idx)
print(onehot)
```

接下来，我们初始化一个线性变换层，它会执行一个矩阵乘法：$X W^T$

```python
torch.manual_seed(123)
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
print(linear.weight)
```

请注意，PyTorch中的线性层同样是用小的随机权重初始化的；为了与上面的Embedding层直接比较，我们必须使用相同的小随机权重，这就是我们在此处重新赋值它们的原因：

```python
linear.weight = torch.nn.Parameter(embedding.weight.T)
```

现在我们可以将线性层应用于输入数据的独热编码表示

```python
print(linear(onehot.float()))
print(embedding(idx))
```

正如我们所见，这与使用嵌入层时得到的结果完全相同。

底层执行的是对第一个训练示例的token ID进行的如下计算：

由于每行独热编码中除一个索引外全为 0（这是设计的必然结果），该矩阵乘法实质上等同于对独热元素的查表操作。

这种在独热编码上使用矩阵乘法的做法等价于嵌入层查找操作，但当处理大型嵌入矩阵时会效率低下，因为存在大量与零相乘的无效计算。

==== 位置编码

GPT-2使用的是#underline[可学习]位置编码。

```python
self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
```

为token id的位置构建一个嵌入查找表，例如如果我们的`context_length`为256的话。那么查找表如下：

#figure(
  table(
    columns: 2,
    [token id的索引], [位置嵌入向量],
    [0], [768个元素的向量],
    [1], [768个元素的向量],
    [2], [768个元素的向量],
    [...], [...],
    [255], [768个元素的向量],
  ),
  caption: [上下文长度为256的位置嵌入查找表],
)

所以

```python
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
```

为`[0, 1, ..., seq_len - 1]`中的每个位置分配了一个768元素的向量，作为对应的位置编码。也就是将序列的每个位置编号组成的序列变换成形状`(batch_size, seq_len, emb_dim)`。

然后我们就会将token的词嵌入向量和token所在位置的位置嵌入向量直接相加。

```python
x = tok_embeds + pos_embeds
```

初看起来，在token向量上面添加位置信息会破坏输入向量，使得网络的任务变得更加难以完成。不过，从两个随机选择的不相关向量在高维空间中几乎是正交的这一现象中，我们可以了解到这一方法为什么能够很好的发挥作用，这表明网络能够相对独立的处理token的身份信息和位置信息。另外，请注意，由于Transformer中的层与层之间都有残差连接，因此位置信息从一层转到下一层的过程中不会丢失。

这里我们发现了，每个token id对应的768元素向量，在训练结束时，会固定下来。例如：假设"hello"这个token的id是1，那么1对应的768个元素向量就固定了。

而序列中的为个位置（索引）对应的768个元素向量，在训练结束时也会固定下来。

设想有两条文本：

- "我是左元"
- "我是莎士比亚"

两句话中的token"我"的token嵌入和位置嵌入一模一样

- 相同token id对应的token嵌入向量一定相同
- 相同位置（两个"我"都为索引0）对应的位置嵌入向量也一定相同

所以`x = tok_embeds + pos_embeds`是一样的！但很明显，两个"我"指代的人名是不一样的，所以这两个语义在GPT-2中如何进行区分呢？那就是注意力机制，或者说Transformer模块了！

==== 仅解码器的Transformer

在模型的初始化时，我们堆叠了很多层的TransformerBlock。

```python
self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
```

然后在前向传播的过程中，

```python
x = self.trf_blocks(x)
```

堆叠很多层TransformerBlock，然后顺序执行。这里也可以看出TransformerBlock的输入和输出的形状必须一致！否则无法堆叠！

送进transformer模块的是#underline[token嵌入+位置嵌入]，形状为：`(batch_size, seq_len, emb_dim)`。

#figure(
  image("llm-figures/10.png"),
  caption: [进入`self.trf_blocks`的数据],
)

接下来我们编写TransformerBlock的代码

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 连接到注意力模块输出的残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) # 形状：[batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut # 将原始的输入加过来

        # 连接到feed-forward block输出的残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # 将原始的输入加过来

        return x
```

#figure(
  image("llm-figures/9.png"),
  caption: [TransformerBlock的架构图],
)

`self.trf_blocks(x)`的输入的形状也是下面的形状

#figure(
  image("llm-figures/10.png"),
  caption: [`self.trf_blocks`输出的数据],
)

==== 层归一化（LayerNorm）

$
  y = (x - upright(E)[x])/(sqrt("Var"[x] + epsilon)) * gamma + beta
$

#figure(
  image("llm-figures/12.png"),
  caption: [层归一化],
)

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # $epsilon$
        self.scale = nn.Parameter(torch.ones(emb_dim)) # $gamma$
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # $beta$

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # $upright(E)[x]$
        var = x.var(dim=-1, keepdim=True, unbiased=False) # $"Var"[x]$
        # $(x - upright(E)[x])/(sqrt("Var"[x] + epsilon))$
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # $y = (x - upright(E)[x])/(sqrt("Var"[x] + epsilon)) * gamma + beta$
        return self.scale * norm_x + self.shift
```

层归一化模块(LayerNorm)：在`emb_dim`这个维度进行归一化。

这里要注意的是：LayerNorm的输入和输出形状相同！也就是说，输入`x`的形状`(batch_size, seq_len, emb_dim)`，输出的形状也是`(batch_size, seq_len, emb_dim)`。只是在`emb_dim`这一个维度进行了归一化。

#figure(
  image("llm-figures/11.png"),
  caption: [公式注解],
)

LayerNorm中有两个参数$gamma$和$beta$会在训练时更新。

==== 多头注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 n_heads 整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # $K = X dot.c W_K^T$
        keys = self.W_key(x) # 形状：(b, num_tokens, d_out)
        # $Q = X dot.c W_Q^T$
        queries = self.W_query(x)
        # $V = X dot.c W_V^T$
        values = self.W_value(x)

        # 通过添加一个`num_heads`维度，我们隐式的将矩阵进行了split
        # 将最后一个维度展开：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置：(b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（scaled dot-product attention）也就是自注意力，使用因果注意力掩码（causal mask）
        # $"attention_scores"=Q K^T$
        attn_scores = queries @ keys.transpose(2, 3) # 每个头计算点积

        # 原始的掩码方阵需要切割token序列的长度，然后转换成布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 形状：(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将多个头组合起来，这里 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # 可选的线性变换（投影）

        return context_vec
```

==== 前馈层

TransformerBlock里的Feed Forward Network，简称FFN/MLP，可以理解为：在注意力机制完成"不同token之间的信息交换"之后，对每个token各自做一次非线性特征加工和升维变换。Self-Attention负责"看别人、融合上下文"，FFN负责"想一想、重新组织这个token的内部表示"。两者配合起来，Transformer才既能建模上下文关系，又有足够强的表达能力。

也就是说：

- Attention子层：让token之间互相通信
- FFN子层：对每个token的表示单独进行非线性变换

Transformer中常见的FFN是一个两层MLP：

$
  "FFN"(x) = W_2 sigma (W_1 x + b_1) + b_2
$

其中：

- $x$是某个token的隐藏状态（hidden state）。
- $W_1$把维度从$d_"model"$扩展到$d_"ff"$
- $sigma$是激活函数，例如ReLU、GELU、SwiGLU等。
- $W_2$再把维度从$d_"ff"$压回$d_"model"$。

中间维度通常是模型维度的4倍左右。

Self-Attention本质上主要是在不同token的表示之间做加权组合。

如果没有FFN，模型会更像是在反复做"上下文混合"，但缺少足够强的非线性加工能力。

FFN中的激活函数，例如GELU，会引入非线性，这使模型能够表达更复杂的函数关系，而不是简单地线性混合信息。

可以类比一下：

- Attention：把相关信息从上下文中找出来
- FFN：对找到的信息进行加工、判断、抽象

就像开会时，Attention是"听取大家意见"，FFN是"自己消化一下"。不然只听不想，会议纪要会很热闹，但决策不一定靠谱。

FFN对序列中的每个位置使用同一套参数，但每个token独立处理。

它不会直接让token之间交流。

token之间的交流主要发生在Self-Attention中。

FFN通常会先升维，再降维。这个结构很重要。升维到更大的空间后，模型可以形成更多中间特征；再压回原维度，相当于筛选和组合这些特征。

可以理解为：

- 把token表示投影到更丰富的特征空间
- 通过激活函数筛选重要模式
- 再压缩回原来的表示维度

这有点像把一张小照片放大后仔细修图，再缩回合适尺寸。不是为了变大，而是为了在更大的空间里做更细的处理。

在大语言模型里，FFN/MLP层往往被认为承担了大量"知识存储"和"模式记忆"的功能。

有一些研究观察到：

- Attention更偏向于路由信息：决定从哪里取信息
- FFN更偏向于内容变换：决定把信息变成什么
- 某些MLP神经元会对特定概念、语言模式、事实关联有响应

当然，这不是说"知识只存在FFN中"，但FFN确实是参数量很大的部分，往往承载了大量可学习模式。

Attention把信息聚合过来之后，需要有一个模块对聚合后的信息做"解释"。

例如句子：

```
The bank is near the river.
```

Attention可以让`bank`关注`river`，但FFN可以进一步把bank的表示推向"河岸"这个语义区域。

另一个句子：

```
I deposited money in the bank.
```

Attention可以让bank关注money，FFN再把它加工成"银行机构"的语义表示。

当然这是简化描述，但直觉上很有用。

FFN 的参数量很大。

一个标准 FFN 参数量大约是：

$
  d_"model" times d_"ff" + f_"ff" times d_"model"
$

如果

$
  d_"ff" = 4d_"model"
$

那么FFN参数量约为：

$
  8d^2_"model"
$

而 Attention 中 Q、K、V、O 四个矩阵大约是：

$
  4d^2_"model"
$

所以FFN往往比Attention参数还多。

这也是为什么它对模型能力非常关键。

Transformer block 中的 Feed Forward Network 的作用是：

对每个 token 的上下文表示进行独立的非线性变换，提升模型表达能力、重组特征，并承担大量模式/知识存储功能。

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

==== 简单的推理代码

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx的形状为(B, T)
    for _ in range(max_new_tokens):
        # 将当前上下文裁剪到我们支持的上下文长度
        # 例如LLM如果只支持5个token，而上下文长度为10，
        # 那么只保留最后5个token作为上下文
        idx_cond = idx[:, -context_size:]

        # 预测
        with torch.no_grad():
            logits = model(idx_cond)

        # 只取最后一个
        # (batch, n_token, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # 获取logits分数最高的值的索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (batch, 1)

        # 将下一个token在词汇表中的索引追加到序列后面
        idx = torch.cat((idx, idx_next), dim=1) # (batch, n_tokens + 1)

    return idx
```

main函数如下：

```python
if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "emb_dim": 768,          # 词嵌入维度
        "n_heads": 12,           # 注意力头的数量
        "n_layers": 12,          # transformer block的层数
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # 禁用dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
```

== GPT-2模型的训练

创建文件`gpt_train.py`。

首先导入我们要用的库。

```python
import matplotlib.pyplot as plt
import os
import torch
import tiktoken
# 从本地库导入
from gpt_model import GPTModel, create_dataloader_v1, generate_text_simple
```

接下来我们编写`token_ids`和`text`之间的转换代码。

```python
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
```

接下来我们编写计算损失的代码。

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
```

然后编写整个数据集的损失的计算

```python
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```

然后编写评估模型的函数

```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
```

然后编写一下简单的生成补全的代码。

```python
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) # 压缩一下打印的格式
    model.train()
```

然后我们编写训练模型的代码

```python
def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    # 初始化一些列表，用来跟踪损失和其它一些指标
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # 主训练循环
    for epoch in range(num_epochs):
        model.train() # 将模型设置为训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 将上一批数据的损失的梯度清空
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 计算损失函数中参数的梯度
            optimizer.step() # 使用梯度更新模型的参数
            tokens_seen += input_batch.numel()
            global_step += 1

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每一轮训练完毕之后，测试一下补全的文本
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
```

编写一下绘制损失图像的代码

```python
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # 绘制每一个epoch的训练损失和验证损失
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    # plt.show()
```

接下来编写main函数。

```python
def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # 准备训练数据
    ##############################

    file_path = "data.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # 初始化模型
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # 准备训练数据集和验证数据集
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # 训练模型
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model
```

然后编写文件的入口点函数

```python
if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # 词汇表大小
        "context_length": 256,  # 上下文长度（开源的gpt-2模型的上下文长度是1024）
        "emb_dim": 768,         # 词嵌入维度
        "n_heads": 12,          # 注意力头的数量
        "n_layers": 12,         # transformer block的数量
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    ###########################
    # 训练模型
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # 训练完成之后
    ###########################

    # 绘制损失图像
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # 保存和加载模型的代码
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
```

== 加载OpenAI开源的GPT-2预训练权重

```python
BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
```

接下来下载预训练权重。

```python
file_name = "gpt2-small-124M.pth"
# file_name = "gpt2-medium-355M.pth"
# file_name = "gpt2-large-774M.pth"
# file_name = "gpt2-xl-1558M.pth"

import os
import requests

url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

if not os.path.exists(file_name):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(file_name, "wb") as f:
        f.write(response.content)
    print(f"Downloaded to {file_name}")
```

加载权重

```python
import torch
from gpt_model import GPTModel

gpt = GPTModel(BASE_CONFIG)
gpt.load_state_dict(torch.load(file_name, weights_only=True))
gpt.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)
```

生成文本

```python
import tiktoken

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 去掉批次维度
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # 计算下一个token的分数，只关注最后一个token
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # top_k采样策略
        if top_k is not None:
            # 选分数最大的k个token作为候选token
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # 温度缩放
        if temperature > 0.0:
            logits = logits / temperature

            # 数值稳定性技巧：在softmax之前，减去一行中的最大值。
            # 因为softmax是按行进行的
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # 使用softmax获取概率值
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 从分布中进行采样（多项分布）
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 贪婪策略：直接选择分数最高的token
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果设置了eos_id，然后生成了eos_token，那么停止生成
        if idx_next == eos_id:
            break

        # 将采样得到的下一个token的id添加到序列的末尾
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate(
    model=gpt.to(device),
    idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
    max_new_tokens=30,
    context_size=BASE_CONFIG["context_length"],
    top_k=1,
    temperature=1.0
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

#part("强化学习")

#chapter("强化学习简介", image: image("./orange2.jpg"), l: "rl-introduction")

== 基本概念

*强化学习*（reinforcement learning，RL）讨论的问题是智能体（agent）怎么在复杂、不确定的环境（environment）中最大化它能获得的奖励。

#figure(
  image("rl-figures/智能体和环境的交互循环.png"),
  caption: [强化学习交互循环],
) <fig-rl-loop>

#tip(title: [名词])[
  - Agent：智能体
  - Environment：环境
  - State：环境的状态
  - Reward：即时奖励
  - Action：动作
]

强化学习由两部分组成：*智能体*和*环境*。在强化学习过程中，智能体与环境一直在交互。智能体在环境中获取到环境的状态后，会根据环境的状态采取一个动作（action）。然后这个动作会在环境中被执行，环境会根据智能体采取的动作，转移到下一个状态，环境还会反馈给智能体当前这个动作带来的即时奖励。智能体的目的就是尽可能多地从环境中获取奖励。

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
- 即时奖励：推车采取向左推或者向右推的动作之后，只要木杆不倒下，奖励就是1。

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

策略的数学符号是$pi$，所以策略也就是$pi$函数。即$pi(a|s)=p(a_t=a|s_t=s)$。输入一个状态$s$，输出一个概率分布。用条件概率来看，就是在条件：状态为$s$的情况下，策略采取动作$a$的概率是多少。这个概率是智能体所有动作的概率，然后对这个概率分布进行采样，可得到智能体将采取的动作。比如可能是有0.7的概率往左，0.3的概率往右，那么通过采样就可以得到智能体将采取的动作。

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

如果这个$pi$函数是一个神经网络，那么这就是*深度学习 + 强化学习 = 深度强化学习*。当今AI界最热门的话题，被认为是*唯一有可能*实现AGI的路线。

#tip[
  我们在对策略函数$pi(a_t|s_t)$进行建模时，策略函数的含义是智能体只会根据#underline[环境的当前状态]$s_t$来决定采取动作$a_t$的概率分布，也就是说智能体做决策时，不考虑历史状态（$s_(t-1), s_(t-2), dots.c$）。
]

所以玩倒立摆游戏的一个*回合*（episode）就是环境的状态为$S_0$，推车采取动作$A_0$，然后获得即时奖励$R_0$，然后环境的状态转移到了$S_1$，推车接着采取动作$A_1$，然后获得奖励$R_1$，...。下标是时刻，或者时间步。把它们写到一起，就是一条*轨迹*（trajectory）。轨迹用数学符号$tau$表示，读作*掏*。

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

接下来我们对环境的状态转移进行数学建模：

假设智能体现在处于状态$s$并执行了动作$a$，那么转移到下一个状态$s'$的概率可以用如下方式表示。

$
  p(s'|s,a)
$

竖杠$|$的右侧是表示"条件"的概率变量。对于当前问题，条件对应于在状态$s$选择了动作$a$。在给定这两个条件的情况下，转移到$s'$的概率可以表示为$p(s'|s,a)$。像$p(s'|s,a)$这样的概率叫作状态转移概率（state transition probability）。

给定当前状态$s$和采取的动作$a$的情况下，*不一定*会确定性的跳转到某个状态$s'$，所以是状态转移概率。

例如：对于下棋环境来说，棋盘是状态，当AlphaGo下了一个棋子之后，棋盘的状态取决于AlphaGo的对手将棋下在哪里，所以状态转移是不确定的。

$p(s'|s,a)$决定了下一个状态$s'$只取决于当前状态$s$和动作$a$。

$
  p(s_t|s_(t-1),a_(t-1)) = p(s_t|s_(t-1),a_(t-1),s_(t-2),a_(t-2),dots.c,s_0,a_0)
$

换句话说，状态转移不需要过去的信息——此前处于什么状态以及执行了哪些动作。这个特性被称为*马尔可夫性质*（Markov property）。

#tip(title: [马尔可夫性质是我们为了方便而做的假设])[
  强化学习通过#underline[假设]马尔可夫性质的存在来模拟状态转移和奖励。引入马尔可夫性质主要是为了使问题更容易解决。如果不假定马尔可夫性质，那么就必须考虑之前的所有状态和行动，而且组合的数量会呈指数级增长。
]

环境的状态转移概率我们是无法确切知道的。换句话说，环境的状态转移是智能体无法控制的。毕竟，大自然环境变幻莫测。

我们还需要对即时奖励进行数学建模：

当环境处于状态$s$，智能体执行动作$a$，并且环境转移到下一个状态$s'$时，奖励函数建模如下：

$
  r(s,a,s')
$

或者也可以写作

$
  R_t = r(s_t=s, a_t=a, s_(t+1)=s')
$

#tip(title: [奖励函数也是智能体无法控制的])[
  智能体无法控制环境如何给奖励。所以对奖励函数的建模是强化学习中最困难的一个问题。奖励太容易拿会导致#underline[奖励黑客]（reward hacking），奖励太难拿会导致智能体无法学到东西，这叫做#underline[奖励稀疏性问题]。
]

== 价值函数（Value Function）

当位于时刻$t$时，环境此时处于状态$S_t$，然后我们根据策略函数开始采取动作，那么未来我们一共能获得多少奖励呢？环境处于状态$S_t$，我们采取的动作是$A_t$，获取的奖励是$R_t$，然后环境的状态从$S_t$转移到了$S_(t+1)$，然后采取动作$A_(t+1)$，然后获得即时奖励$R_(t+1)$，然后环境的状态从$S_(t+1)$转移到了$S_(t+2)$，然后环境会给我们即时奖励$R_(t+2)$，......。

但是未来的奖励不如现在的奖励有吸引力，所以需要*打折*。那么，从$t$时刻起，未来一共获得的奖励叫做*回报*（或者收益，Return）。

$
  G_t = R_t + gamma R_(t+1) + gamma^2 R_(t+2) + dots.c
$

由上面的式子可以推导出如下递推公式：

$
  G_t = R_t + gamma G_(t+1)
$

$gamma$叫做折扣因子。随着时间的推移，奖励会被$gamma$指数级削弱。这个$gamma$被称为折扣因子（discount rate），其被设定为$0.0$和$1.0$之间的实数。如果折扣因子是$0.9$，那么有以下式子成立。

$
  G_t = R_t + 0.9 R_(t+1) + 0.81 R_(t+2) + dots.c
$

引入折扣因子主要是为了防止连续性任务的收益变得无穷大。在连续性任务中，如果没有折扣因子（或$gamma=1$），那么收益就会发散到无穷大。因此，设置折扣因子可以防止收益的发散。

折扣因子也使近期的奖励显得更加重要。这解释了人类乃至生物的许多行动原理。例如，你会选择今天拿到10000元还是一年后拿到20000元？如果折扣因子使未来的回报呈指数级下降，那么眼前的回报就会更有吸引力。

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

可以看到，一共有4条轨迹。每条轨迹都有一个总的回报。而每条轨迹也都有一个产生的概率。那么我们如何评估在环境处于状态$S_t$时，一直采取策略$pi$未来会获得多少回报呢？也就是未来的预期回报（回报的期望值）是多少呢？那就是*状态价值函数*（State Value Function）。

$
  V_pi (s) & = EE_pi [G_t|S_t=s] \
           & = EE_pi [ sum_(k=0)^infinity gamma^k R_(t+k) | S_t=s ], "对所有的" space s in S
$

#tip[
  状态价值函数，衡量的是在环境处于状态$s$时，一直按照策略$pi$来采取动作，最终的预期回报。
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

#codly(header: [打印初始状态和动作空间维度])
```python
state = env.reset() # 重置环境的状态为$S_0$
print(state) # 初始状态$S_0$
action_space = env.action_space # 推车有几个动作？（向左推，向右推）
print(action_space) # 动作空间的维度=2
```

上面的代码通过```python state = env.reset()```获得了初始状态。观察它的输出，你会发现它是拥有 4 个元素的数组。作为参考，下面依次列出这 4 个元素。

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
    # $G_t = R_t + gamma G_(t+1)$
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
    EE[X|Y=y] & = sum_x x dot.c P(X=x|Y=y) space space colblue("（条件期望）")
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
  v_pi (s) = EE_pi [G_t|S_t=s]
$

如上面的公式所示，状态 $s$ 的价值函数被表示为 $v_pi (s)$ 。将递推公式带入上面的式子的 $G_t$ 中，得到下面的式子：

$
  v_pi (s) & = EE_pi [G_t|S_t=s] \
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

如上面的数学式所示，将智能体行动的概率$pi(a|s)$、要转移的状态的概率$p(s'|s,a)$和奖励函数$r(s,a,s')$相乘。对所有候选项都进行上述计算，得到它们的总和。

#tip(title: [$s arrow s'$的转移概率])[
  由于$s$是确定的，所以$p(s,a)=pi(a|s)$。
  $
    p(s'|s) = sum_a p(s',a|s) = sum_a p(s'|s,a)p(a,s)=sum_a p(s'|s,a)pi(a|s)
  $
]

#tip(title: [证明])[
  $
    EE_pi [R_t|S_t=s] & = EE_pi [R_t=r(s,a,s')|S_t=s] \
  $
  上面的期望，计算的是随机变量$R_t$的期望。而$S_t=s$是确定好的条件。由于$r(s,a,s')$在$s,a,s'$确定的情况下，计算出来的确定的值。所以$R_t=r(s,a,s')$发生的概率是$p(s,a,s')$。所以有如下：
  $
    EE_pi [R_t|S_t=s] & = EE_pi [R_t=r(s,a,s')|S_t=s] \
                      & = sum_a sum_s' r(s,a,s') p(s,a,s') \
                      & = sum_a sum_s' r(s,a,s') p(s'|s,a) p(s,a) \
                      & = sum_a sum_s' r(s,a,s') p(s'|s,a) pi(a|s) \
                      & = sum_(a,s') r(s,a,s') p(s'|s,a) pi(a|s)
  $

  要注意的是：$S_t=s$是确定的，所以随机变量为$a,s'$。所以需要遍历所有可能的$a,s'$事件。
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
  v_pi (s') = EE_pi [G_(t+1)|S_(t+1)=s']
$

这就是状态$S_(t+1)=s'$时的价值函数。接下来要关注的是$EE_pi [G_(t+1)|S_t=s]$。这是在当前时刻为$t$时，下一个时刻$(t+1)$的收益期望值。解决的关键在于将条件$S_t=s$变为$S_(t+1)=s$的形式。换句话说，就是要进入下一个时刻。

通过观察可以得到

$
  EE_pi [G_(t+1)|S_t=s] & = sum_(a,s') pi(a|s)p(s'|s,a)EE_pi [G_(t+1)|S_(t+1)=s'] \
                        & = sum_(a,s') pi(a|s)p(s'|s,a)v_pi (s')
$

#tip(title: [证明])[
  思路就是将$s arrow s'$的所有路径都遍历计算一遍。
  #math.equation(
    $
      EE_pi [G_(t+1)|S_t=s] & = sum_(G_(t+1)) p(G_(t+1)|S_t=s) dot.c G_(t+1) \
      & = sum_(G_(t+1)) { sum_s' p(G_(t+1), S_(t+1) = s' | S_t=s) } dot.c G_(t+1) \
      & =sum_(G_(t+1)) { sum_s' p(G_(t+1)|S_(t+1)=s',S_t=s)p(S_(t+1)=s'|S_t=s) } dot.c G_(t+1) \
      & = sum_(G_(t+1)) { sum_s' p(G_(t+1)|S_(t+1)=s')p(S_(t+1)=s'|S_t=s) } dot.c G_(t+1) space space colblue("（马尔可夫性质）")\
      & = sum_s' p(S_(t+1) = s'|S_t=s) {sum_(G_(t+1)) p(G_(t+1)|S_(t+1)=s') dot.c G_(t+1)} \
      & = sum_s' p(S_(t+1) = s'|S_t=s) EE_pi [G_(t+1)|S_(t+1)=s'] \
      & = sum_(a,s')pi(a|s)p(s'|s,a)v_pi (s') \
      & = EE_pi [v_pi (s')|S_t=s] space space colblue("（期望形式）")
    $,
    block: true,
    number-align: bottom,
  )
  在$S_(t+1)$确定下来之后，$G_(t+1)$不再依赖$S_t$。因为$G_(t+1) = R_(t+1) + gamma G_(t+2) = r(S_(t+1)=s', a, S_(t+2)=s'') + gamma G_(t+2)$。
]


完成第二项的展开以后，汇总一下得到

#theorem(name: [贝尔曼方程])[
  $
    v_pi (s) & = EE_pi [R_t|S_t=s] + gamma EE_pi [G_(t+1)|S_t=s] \
             & = sum_(a,s')pi(a|s)p(s'|s,a)r(s,a,s')+gamma sum_(a,s')pi(a|s)p(s'|s,a)v_pi (s') \
             & = sum_(a,s')pi(a|s)p(s'|s,a){r(s,a,s')+gamma v_pi (s')}
  $
]

上面的式子就是大名鼎鼎的*贝尔曼方程*。贝尔曼方程是表示状态$s$的价值函数和下一个可能的状态$s'$的价值函数之间关系的式子。这个贝尔曼方程对所有状态$s$和所有策略$pi$都成立。

状态价值函数的贝尔曼方程的另一种重要的表示形式：*贝尔曼期望方程*。

#theorem(name: [贝尔曼期望方程])[
  $
    V_pi (S_t) = EE_pi [R_t+gamma V_pi (S_(t+1))|S_t]
  $
]

贝尔曼期望方程的推导过程如下：

$
  V_pi (S_t) & = EE_pi [R_t|S_t=s] + gamma EE_pi [G_(t+1)|S_t=s] \
             & = EE_pi [R_t|S_t=s] + gamma EE_pi [V_pi (s')|S_t=s] \
             & = EE_pi [R_t + gamma V_pi (s')|S_t=s]
$

== 下一步

智能体的目标是使收益最大化。这里有一点需要注意，那就是智能体和环境的行动可能是"随机性"的。智能体可能随机地决定行动，状态也可能随机转移。在这种情况下，获得的收益将呈现随机的特点。即使从相同的状态开始，不同回合的收益也随机变化。例如，某个回合的收益为$10.4$，另一个回合的收益为$8.7$。

在倒立摆环境中，我们如何选择智能体的"策略"来让木杆多坚持一段时间呢？

接下来我们来学习*策略梯度法*（policy gradient method）。

#chapter("策略梯度法", image: image("./orange2.jpg"), l: "rl-pg")

== 原始策略梯度法

=== 理论介绍

在倒立摆游戏的环境中，我们唯一能控制的，就是推车所采用的*策略*。而环境和奖励函数我们是无法控制的。

#figure(
  table(
    columns: 3,
    table.cell(
      fill: green.lighten(60%),
    )[我们可以控制的],
    table.cell(
      fill: red.lighten(60%),
    )[我们无法控制的],
    table.cell(
      fill: red.lighten(60%),
    )[我们无法控制的],
    [策略函数], [环境], [奖励函数],
    [推车的策略], [倒立摆环境], [保持平衡，奖励为1],
  ),
  caption: [只有策略函数是我们可以控制的，环境和奖励我们无法控制],
)

现在通行的做法就是策略函数是一个神经网络。输入是环境的状态，输出是动作的概率分布。

#figure(
  fletcher.diagram(
    let blob(pos, label, tint: white, ..args) = fletcher.node(
      pos,
      align(center, label),
      width: 28mm,
      fill: tint.lighten(60%),
      stroke: 1pt + tint.darken(20%),
      corner-radius: 5pt,
      ..args,
    ),
    blob((-3, 1), [推车的位置], tint: green, shape: fletcher.shapes.rect),
    blob((-3, 1.5), [推车的速度], tint: green, shape: fletcher.shapes.rect),
    blob((-3, 2), [木杆的角度], tint: green, shape: fletcher.shapes.rect),
    blob((-3, 2.5), [木杆的角速度], tint: green, shape: fletcher.shapes.rect),

    blob((-1, 1), [], tint: white, shape: fletcher.shapes.circle, width: 6mm, name: <A>),
    blob((-1, 1.5), [], tint: white, shape: fletcher.shapes.circle, width: 6mm, name: <B>),
    blob((-1, 2), [], tint: white, shape: fletcher.shapes.circle, width: 6mm, name: <C>),
    blob((-1, 2.5), [], tint: white, shape: fletcher.shapes.circle, width: 6mm, name: <D>),

    blob((0, 1.25), [], tint: white, shape: fletcher.shapes.circle, width: 6mm, name: <AA>),
    blob((0, 1.75), [], tint: white, shape: fletcher.shapes.circle, width: 6mm, name: <BB>),
    blob((0, 2.25), [], tint: white, shape: fletcher.shapes.circle, width: 6mm, name: <CC>),

    blob((1, 1.5), [], tint: white, shape: fletcher.shapes.circle, width: 6mm, name: <AAA>),
    blob((1, 2), [], tint: white, shape: fletcher.shapes.circle, width: 6mm, name: <BBB>),

    blob((3, 1.5), [向左推0.7], tint: purple, shape: fletcher.shapes.rect, name: <AAAA>),
    blob((3, 2), [向右推0.3], tint: purple, shape: fletcher.shapes.rect, name: <BBBB>),

    fletcher.edge((-3, 1), (-1, 1), "-|>", stroke: 0.1em),
    fletcher.edge((-3, 1.5), (-1, 1.5), "-|>", stroke: 0.1em),
    fletcher.edge((-3, 2), (-1, 2), "-|>", stroke: 0.1em),
    fletcher.edge((-3, 2.5), (-1, 2.5), "-|>", stroke: 0.1em),

    for i in (<A>, <B>, <C>, <D>) {
      for j in (<AA>, <BB>, <CC>) {
        fletcher.edge(i, j, "->")
        for k in (<AAA>, <BBB>) {
          fletcher.edge(j, k, "->")
        }
      }
    },

    fletcher.edge(<AAA>, <AAAA>, "-|>", stroke: 0.1em),
    fletcher.edge(<BBB>, <BBBB>, "-|>", stroke: 0.1em),
  ),
  caption: [输出的动作是概率],
)

上图就是一个例子。玩倒立摆游戏。策略函数是一个神经网络；输入是倒立摆环境的状态，一个浮点数数组；输出是我们可以执行的动作，有几个动作，输出层就有几个神经元。假设我们现在可以执行的动作有2个，输出层就有2个神经元，每个神经元对应一个可以采取的动作。输入一个东西后，策略神经网络会给每一个可以采取的动作一个分数。我们可以把这个分数当作概率，智能体根据概率的分布来决定它要采取的动作，比如0.7的概率向左推、0.3的概率向右推。概率分布不同，推车采取的动作就会不一样。

如下面的式子所示，首先，环境是一个函数，我们可以把倒立摆环境看成一个函数，虽然它不一定是神经网络，可能是基于规则的（rule-based）模型，但我们可以把它看作一个函数。

$
  "倒立摆环境"(S_t,A_t) arrow (S_(t+1),R_t)
$

倒立摆环境就是一个基于规则的函数，只要推车不倒下，就返回奖励1。

讲了这么多策略有关的东西，大家应该明白策略是什么东西了。

而通过神经网络等方法将策略模型化，并使用梯度来优化策略的方法叫作*策略梯度法*（policy gradient method）。

研究者们提出了各种基于策略梯度法的算法。本章首先介绍最简单的策略梯度法。然后，在改进这个简单的策略梯度法的过程中，我们推导出了被称为*REINFORCE*的算法。接下来，在进一步改进REINFORCE的过程中， 我们又推导出了*带基线*的REINFORCE方法和Actor-Critic（演员-评论家） 方法。

随机性策略用数学式可以表示为$pi(a|s)$。$pi(a|s)$是在状态$s$下采取动作$a$的概率。这里采用神经网络对策略进行建模。此时用符号$theta$来汇总表示神经网络的所有权重参数（$theta$是将所有参数的元素排成一列的向量）。另外，可以将基于神经网络的策略表示为$pi_theta (a|s)$，如下所示。

$
  \
  \
  \
  \
  markhl(pi, tag: #<p1>, color: #blue)_markhl(theta, tag: #<p2>, color: #orange) ( markhl(a, tag: #<p3>, color: #green) | markhl(s, tag: #<p4>, color: #red) )
  \
  \
  \
  \
  #annot(<p1>, pos: top + left, dy: -1.5em, leader-connect: "elbow")[策略神经网络]
  #annot(<p2>, pos: bottom + left, dy: 1.5em, leader-connect: "elbow")[神经网络的参数]
  #annot(<p3>, pos: bottom + right, dy: 1.5em, leader-connect: "elbow")[在状态$s$下，采取的动作$a$]
  #annot(<p4>, pos: top + right, dy: -1.5em, leader-connect: "elbow")[当前状态$s$]
$

还是倒立摆环境，每当倒立摆环境处于某个状态$s$时，我们就会使用神经网络$pi_theta$来决定要采取什么动作。

那么，问题是：策略神经网络怎么训练？

#tip(title: [神经网络的训练])[
  想要训练一个神经网络，需要有*输入-输出*对。例如，MNIST手写数字数据集，要训练一个可以识别手写数字的卷积神经网络，需要构建一个网络结构，然后提供输入（图片）以及输出（分类标签）。

  当然，还得有*损失函数*，例如交叉熵损失函数或者均方误差损失函数，等等。
]

#let layers = (
  (
    type: "input",
    image: "default",
    height: 8,
    depth: 8,
    label: "input",
    channels: (3, 224),
  ),
  (
    type: "conv",
    widths: (0.3, 0.3),
    height: 8,
    depth: 8,
    label: "conv1",
    channels: (64, 64, 224),
    offset: 1.9,
  ),
  (
    type: "pool",
    height: 6,
    depth: 6,
    label: "pool1",
  ),
  (
    type: "convres",
    widths: (0.4, 0.4),
    height: 6,
    depth: 6,
    label: "res2",
    channels: (128, 128, 112),
  ),
  (
    type: "pool",
    height: 4,
    depth: 4,
    label: "pool2",
  ),
  (
    type: "convres",
    widths: (0.5, 0.5, 0.5),
    height: 4,
    depth: 4,
    label: "res3",
    channels: (256, 256, 256, 56),
  ),
  (
    type: "pool",
    height: 2,
    depth: 2,
    label: "pool3",
  ),
  (
    type: "convres",
    widths: (0.6, 0.6, 0.6),
    height: 2,
    depth: 2,
    label: "res4",
    channels: (512, 512, 512, 28),
    offset: 1,
  ),
  (
    type: "pool",
    height: 1,
    depth: 1,
    label: "pool4",
  ),
  (
    type: "convres",
    widths: (0.6, 0.8, 0.8),
    height: 1,
    depth: 1,
    label: "res5",
    channels: (512, 512, 512, 14),
    offset: 0.8,
  ),
  (
    type: "pool",
    height: 0.5,
    depth: 0.5,
    label: "pool4",
  ),
  (
    type: "fc",
    label: "fc",
    channels: (4096,),
    height: 5,
    depth: 0.3,
    offset: 0.8,
  ),
  (
    type: "fc",
    label: "fc",
    channels: (4096,),
    height: 5,
    depth: 0.3,
    offset: 0.5,
  ),
  (
    type: "fc",
    label: "fc",
    channels: (1000,),
    height: 4,
    depth: 0.3,
    offset: 0.5,
  ),
  (
    type: "softmax",
    label: "softmax",
    height: 4,
    depth: 0.3,
    offset: 0.9,
  ),
)

#figure(
  draw-network(
    layers,
    show-relu: true,
  ),
  caption: [ResNet18],
)

但是对于倒立摆环境，想要训练推车的策略神经网络，输入是什么，输出是什么？以及损失函数又是什么？

首先回顾一下第一章的知识。

首先明确问题的设定。这里考虑的是回合制任务，并基于策略$pi_theta$选择动作的情况。在这种情况下，假定得到了以下由"状态、动作、奖励"构成的时间序列数据。

$
  tau = (S_0, A_0, R_0, S_1, A_1, R_1, dots.c, S_(T+1))
$

这个$tau$也叫作轨迹（trajectory）。

而一条轨迹发生的概率是：

$
  "Pr"(tau) & = p(S_0) pi_theta (A_0|S_0) p(S_1|S_0,A_0) pi_theta (A_1|S_1) p(S_2|S_1,A_1) dots.c pi_theta (A_T|S_T) p(S_(T+1)|S_T,A_T) \
  &= p(S_0) product_(t=1)^T pi_theta (A_t|S_t)p(S_(t+1)|S_t,A_t)
$

此时可以使用折扣因子$gamma$对回报（Return，收益）作如下定义。

$
  G(tau) = R_0 + gamma R_1 + gamma^2 R_2 + dots.c + gamma^T R_T
$

为了表明回报可以由$tau$计算出来，上面的式子将其表示为了$G(tau)$。此时，目标函数$J(theta)$可以表示为下面的式子。

#definition(name: [强化学习的目标函数])[
  $
    J(theta) = EE_(tau tilde pi_theta) [G(tau)]
  $
]

回报$G(tau)$是随机变动的，所以它的期望值是目标函数。上式中期望值$EE$的下标为$tau tilde pi_theta$ ，这个下标表示$tau$是基于$pi_theta$ 生成的。

#figure(
  $
    \
    \
    \
    \
    J(markhl(theta, tag: #<pgo1>, color: #blue))=EE_markhl(tau tilde pi_theta, tag: #<pgo2>, color: #orange) [ markhl(G(tau), tag: #<pgo3>, color: #red) ]
    \
    \
    \
    \
    #annot(<pgo1>, pos: bottom + left, [策略神经网络的参数], leader-connect: "elbow", dy: 1.5em)
    #annot(<pgo2>, pos: bottom + right, [轨迹由策略$pi$生成], leader-connect: "elbow", dy: 1.5em)
    #annot(<pgo3>, pos: top + left, [轨迹$tau$的回报], leader-connect: "elbow", dy: -1.5em)
  $,
  caption: [策略梯度法的目标函数],
)

我们的目标是让$J(theta) = EE_(tau tilde pi_theta) [G(tau)]$最大。但是我们发现$G(tau)$这个轨迹的回报无法求导，换句话说就是对目标函数*无法进行优化*。

目标函数可以优化的意思其实就是可以求导。

在神经网络的训练中，我们的目的是让*损失函数*最小。而在策略梯度法中，我们的目的是让*目标函数*最大。所以都可以使用梯度法。让损失函数最小，使用梯度下降法。让目标函数最大，使用梯度上升法。

所以，我们还是得求解$J(theta)$的梯度。需要求导的参数是$theta$。

确定了目标函数后，下一步是计算它的梯度。这里将参数$theta$的梯度表示为$nabla_theta$。我们的目标是求$nabla_theta J(theta)$。$nabla$是梯度符号。

#theorem(name: [策略梯度定理])[
  $
    nabla_theta J(theta) & = nabla_theta EE_(tau tilde pi_theta) [G(tau)] \
                         & = EE_(tau tilde pi_theta) [sum_(t=0)^T G(tau) nabla_theta log pi_theta (A_t|S_t)]
  $ <pg-theorem>
]

上面的式子中值得注意的是，$nabla_theta$在$EE$中（梯度计算的部分是$nabla_theta log pi_theta (A_t|S_t)$）。后面会对此做详细介绍。求出$nabla_theta J(theta)$之后，接下来更新神经网络的参数。最优化方法多种多样，下面的式子表示的是一种简单的方法。

#figure(
  $
    theta arrow.l theta + alpha nabla_theta J(theta)
  $,
  caption: [策略梯度法的梯度上升算法],
)

上面的式子朝着梯度的方向更新参数$theta$。更新的值与$alpha$相关。这里的$alpha$表示学习率。这是属于梯度上升法的算法。

只要沿着梯度上升的方向更新参数$theta$，那么目标函数$J(theta)$就会越来越大，也就是所有轨迹的期望就会越来越大，那么就是我们的策略越来越好了。

#danger[
  策略梯度法求得的梯度是一个期望值，而期望值无法计算。
]

这里有一个问题，那就是*真正的期望*是无法准确求出来的。因为期望是所有的轨迹得到的奖励计算出来的。而轨迹有无数条。如果我们能够走满倒立摆的200步，那么不同轨迹的数量可能是$2^(200)$，如下图所示，这是一个天文数字。所以我们希望能够求出*近似*期望值的数值。比如利用*大数定理*，多采样几条轨迹，那么就会比较接近期望值。

#figure(
  $
    & S_0^tau_0 arrow A_0^tau_0 arrow S_1^tau_0 arrow A_1^tau_0 arrow dots.c \
    & S_0^tau_1 arrow A_0^tau_1 arrow S_1^tau_1 arrow A_1^tau_1 arrow dots.c \
    & S_0^tau_2 arrow A_0^tau_2 arrow S_1^tau_2 arrow A_1^tau_2 arrow dots.c \
    & \
    & dots.c dots.c
  $,
  caption: [无数条轨迹],
)

#tip(title: [大数定律])[
  样本数量越多，则其算术平均值就有越高的概率接近期望。

  例如，抛掷一颗均匀的6面的骰子，1，2，3，4，5，6应等概率出现，所以每次扔出骰子后，出现点数的期望是
  $
    (1+2+3+4+5+6)/6=3.5
  $
  根据大数定理，如果多次抛掷骰子，随着抛掷次数的增加，平均值（样本平均值）应该接近3.5。

  如果随机变量$X_1,X_2,dots$是独立同分布，且期望$EE(X_1)=EE(X_2)=dots.c=mu$。那么
  $
    overline(X)_n=1/n (X_1+X_2+dots.c+X_n)
  $
  当$n arrow infinity$时, 收敛于真值
  $
    overline(X)_n arrow mu
  $
]

#figure(
  $
    & S_0^tau_0 arrow A_0^tau_0 arrow S_1^tau_0 arrow A_1^tau_0 arrow dots.c \
    & colred(S_0^tau_1 arrow A_0^tau_1 arrow S_1^tau_1 arrow A_1^tau_1 arrow dots.c) \
    & S_0^tau_2 arrow A_0^tau_2 arrow S_1^tau_2 arrow A_1^tau_2 arrow dots.c \
    & colred(S_0^tau_3 arrow A_0^tau_3 arrow S_1^tau_3 arrow A_1^tau_3 arrow dots.c) \
    & S_0^tau_4 arrow A_0^tau_4 arrow S_1^tau_4 arrow A_1^tau_4 arrow dots.c \
    & \
    & dots.c dots.c
  $,
  caption: [只采样$n$条轨迹],
)

如@pg-theorem 所示，$nabla_theta J(theta)$表示期望值。接下来我们来计算期望值。这里，我们令策略$pi_theta$的智能体实际采取动作，得到$n$个轨迹$tau$，如上图所示。此时，通过对每个$tau$计算式子的期望值内部的式子（$sum_(t=0)^T G(tau)nabla_theta log pi_theta (A_t|S_t)$），并求出其平均值，从而近似得到$nabla_theta J(theta)$。数学式如下所示。

$
  "采样": tau^((i)) tilde pi_theta space space (i=1,2,dots.c,n) \
  x^((i)) = sum_(t=0)^T G(tau^((i)))nabla_theta log pi_theta (A_t^((i))|S_t^((i))) \
  nabla_theta J(theta) approx (x^((1))+x^((2))+dots.c+x^((n)))/n
$

上面式子中的$tau^((i))$，表示在第$i$回合得到的轨迹，$A_t^((i))$表示在第$i$回合的时刻$t$的动作，$S_t^((i))$表示在第$i$回合的时刻$t$的状态。

另外，再思考一下蒙特卡洛方法的样本数为$1$，即上式中$n=1$的情况，如下图所示。

#figure(
  $
    & S_0^tau_0 arrow A_0^tau_0 arrow S_1^tau_0 arrow A_1^tau_0 arrow dots.c \
    & colred(S_0^tau_1 arrow A_0^tau_1 arrow S_1^tau_1 arrow A_1^tau_1 arrow dots.c) \
    & S_0^tau_2 arrow A_0^tau_2 arrow S_1^tau_2 arrow A_1^tau_2 arrow dots.c \
    & S_0^tau_3 arrow A_0^tau_3 arrow S_1^tau_3 arrow A_1^tau_3 arrow dots.c \
    & S_0^tau_4 arrow A_0^tau_4 arrow S_1^tau_4 arrow A_1^tau_4 arrow dots.c \
    & \
    & dots.c dots.c
  $,
  caption: [只采样$1$条轨迹],
)

在这种情况下，数学式可以简化为如下形式。

$
  "采样": tau tilde pi_theta \
  nabla_theta J(theta) approx sum_(t=0)^T G(tau)nabla_theta log pi_theta (A_t|S_t)
$

为了简单起见，本章将使用以上面的式子为对象的策略梯度法。上面的式子的计算就是对所有时刻（$t=0 tilde T$）求$nabla_theta log pi_theta (A_t|S_t)$，然后将各梯度乘以作为权重的回报$G(tau)$，最后求它们的和。这个计算过程如下图所示。

#figure(
  image("rl-figures/使用一条轨迹计算策略梯度.svg"),
  caption: [使用一条轨迹计算策略梯度],
)

=== 代码实现

接下来，我们用代码来实现一下策略梯度法。

我们讲一些实现细节。我们可以把强化学习想成一个分类问题，这个分类问题就是输入倒立摆的状态，输出某个类。在解决分类问题时，我们要收集一些训练数据，数据中要有输入与输出的对。在实现的时候，我们把倒立摆的状态当作分类器的输入，就像在解决图像分类的问题，只是现在的类不是图像里面的东西，而是看到倒立摆的状态我们要采取什么样的动作，每一个动作就是一个类。比如第一个类是向左，第二个类是向右。

在解决分类问题时，我们要有输入和正确的输出，要有训练数据。但在强化学习中，我们通过采样来获得训练数据。假设在采样的过程中，在某个状态下，我们采样到要采取动作$A$， 那么就把动作$A$当作标准答案（ground truth）。比如，我们在某个状态下，采样到要向左。因为是采样，所以向左这个动作不一定概率最高。假设我们采样到向左，在训练的时候，让智能体调整网络的参数，如果看到某个状态，我们就向左。在一般的分类问题里面，我们在实现分类的时候，目标函数都会写成最小化交叉熵（cross entropy），最小化交叉熵就是最大化对数似然（log likelihood）。

#tip(title: [从极大似然估计的角度看策略梯度])[
  $
    J(theta) & = sum_(t=0)^T G(tau) log pi_theta (A_t|S_t) \
             & = G(tau) log (product_(t=0)^T pi_theta (A_t|S_t))
  $
  最大化目标就会提升某些状态对应的动作的概率。
]

#codly(header: [策略神经网络$pi_theta$])
```python
class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128) # 推车环境中状态是一个4维数组
        self.l2 = nn.Linear(128, action_size) # 输出是一个2维数组

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x # 输出：$[pi_theta (a="向左推"|s),pi_theta (a="向右推"|s)]$
```

上面的代码实现的神经网络模型由两层全连接层构成。最终输出的元素数是动作的数量（`action_size`）。由于这个最终输出是`softmax`函数的输出，因此可以得到每个动作的概率。

这个网络结构就是我们要训练的策略神经网络$pi_theta$。其中$theta$就是神经网络的参数。

我们还记得，策略神经网络的输入是环境的状态，输出是要采取的动作的概率分布。所谓动作的概率分布就是：向左推的概率是多少，向右推的概率是多少。

而我们的网络结构输入的张量的维度是4。为什么呢？因为倒立摆环境的状态有4个维度。

输出是`action_size`个维度。也就是动作的数量个维度。

#tip(title: [softmax])[
  如果将具有$n$个元素的向量输入到softmax函数中，那么输出的同样是具有$n$个元素的向量。此时，第$i$个输出$y_i$的式子如下所示。
  $
    y_i = (upright(e)^(x_i))/(sum_(k=1)^n upright(e)^(x_k))
  $
  这里的$upright(e)$是自然常数(值为$2.718 28...$的无限小数)。softmax函数的输出值全部为$0$以上$1$以下的实数，它们的合计值为$1$（$sum_(i=1)^n y_i = 1$）。因此，softmax函数的输出可以作为概率使用。
]

下面是`Agent`类的代码。首先显示初始化和`get_action`方法。

#figure(caption: [智能体代码])[
  #show raw.where(lang: "python"): it => {
    show regex("\$(.*?)\$"): re => {
      eval(re.text, mode: "markup")
    }
    it
  }
  ```python
  class Agent:
      def __init__(self):
          self.gamma = 0.98 # 折扣因子$gamma$
          self.lr = 0.0002 # 学习率$alpha$
          self.action_size = 2 # 两个动作

          self.pi = PolicyNet(self.action_size) # 策略神经网络$pi_theta$
          self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

      def get_action(self, state):
          probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
          # 根据动作的概率分布创建一个二项分布
          m = Categorical(probs)
          # 使用二项分布采样动作$pi_theta (a_t|s_t)$
          action = m.sample().item()

          # (采样的动作, [向左推的概率, 向右推的概率])
          return action, probs
  ```
] <agent-code-1>

`get_action`方法决定了在`state`状态下采取的动作。为此，可以通过`self.pi(state)`进行神经网络的前向传播，得到概率分布`probs`。然后，基于该概率分布，进行一次动作的采样。该方法还返回了所有动作的概率（上面代码中的`probs`）。

下面来试用一下`get_action`方法。代码如下所示。

#codly(header: [执行1步动作然后反向传播，观察结果])
```python
env = gym.make("CartPole-v0")
state = env.reset() # $S_0$
agent = Agent()

action, probs = agent.get_action(state)
print("动作：", action) # $A_0$
print("动作的概率：", probs[action].item()) # $pi_theta (A_0|S_0)$

G = 100.0 # $G(tau)$
J = -G * probs[action].log() # $-G(tau) log pi_theta (A_0|S_0)$
print("J: ", J)

J.backward() # 求导（梯度）$- nabla_theta G(tau) log pi_theta (A_0|S_0)$
agent.optimizer.step() # 梯度下降：$theta=theta+alpha nabla_theta G(tau) log pi_theta (A_0|S_0)$
# 在相同的状态state下，采取动作的概率变大了
# （测试一下）将G变为负值，也就是负的奖励，会发现采取动作的概率下降了
_, probs = agent.get_action(state)
print("动作：", action)
print("动作的概率：", probs[action].item())
```

上面的代码取出了初始状态下的动作及其概率。另外，它还显示了使用虚拟的权重来计算由下面的式子表示的梯度的代码（这是从式子取出的$t = 0$的相关项的式子）。

$
  G(tau) nabla_theta log pi_theta (A_0|S_0)
$

作为参考，下面对照列出了上面的代码中出现的变量与相应的数学式。

- `probs[action]`：$pi_theta (A_0|S_0)$
- `G`：$G(tau)$
- `J`：$G(tau) log pi_theta (A_0|S_0)$

求出`J`之后，通过`J.backward()`求$G(tau) nabla_theta log pi_theta (A_0|S_0)$。下面是`Agent`类剩下的代码。

我们先来编写采集一条轨迹的详细信息的代码。

#figure(caption: [采集一条轨迹的详细数据，接@agent-code-1])[
  #show raw.where(lang: "python"): it => {
    show regex("\$(.*?)\$"): re => {
      eval(re.text, mode: "markup")
    }
    it
  }
  ```python
  class Agent:
      ...

      def collect_trajectory(self, env):
          state = env.reset()
          states, actions, rewards = [], [], []
          done = False

          while not done:
              action, _ = self.get_action(state)
              next_state, reward, done, _ = env.step(action)
              states.append(state) # $S_t$
              actions.append(action) # $A_t$
              rewards.append(reward) # $R_t$
              state = next_state # $S_t arrow S_(t+1)$
          # $[S_0, S_1, dots.c, S_T]$
          # $[A_0, A_1, dots.c, A_T]$
          # $[R_0, R_1, dots.c, R_T]$
          return states, actions, rewards
  ```
] <agent-code-2>

然后编写更新策略的代码：

#figure(caption: [更新策略的代码，接@agent-code-2])[
  #show raw.where(lang: "python"): it => {
    show regex("\$(.*?)\$"): re => {
      eval(re.text, mode: "markup")
    }
    it
  }
  ```python
  class Agent:
      ...

      def update(self, trajectory):
          states, actions, rewards = trajectory
          # 逆序计算$G(tau)$
          G = 0
          for r in rewards[::-1]:
              G = r + self.gamma * G
          loss = 0
          for s, a in zip(states, actions):
              probs = self.pi(torch.tensor(s).unsqueeze(0)).squeeze(0)
              log_prob = torch.log(probs)[a] # $log pi_theta (A_t|S_t)$
              loss += -log_prob * G # $-sum_(t=0)^T G(tau) log pi_theta (A_t|S_t)$

          self.optimizer.zero_grad()
          loss.backward() # $nabla_theta { -sum_(t=0)^T G(tau) log pi_theta (A_t|S_t) }$
          self.optimizer.step() # $theta=theta-alpha nabla_theta { -sum_(t=0)^T G(tau) log pi_theta (A_t|S_t) }$
  ```
] <agent-code-3>

在训练神经网络时，通常要设置损失函数。对于这个例子，我们可以将目标函数$J(theta)$乘以$-1$所得到的$-J(theta)$作为损失函数，此时可以通过梯度下降法的最优化方法（SGD、Adam等）更新参数。

@agent-code-3 中第10 #sym.tilde 14 行可以改写成并行版本，代码如下：

#codly(header: [并行版本])
```python
states = torch.tensor(states)
actions = torch.tensor(actions).view(-1, 1)
log_probs = torch.log(self.pi(states).gather(1, actions))
loss = -torch.sum(log_probs) * G
```

最后在倒立摆环境中训练智能体。代码如下所示。

```python
env = gym.make("CartPole-v0")
agent = Agent()
return_list = []
episode_list = []

for episode in range(3000):
    trajectory = agent.collect_trajectory(env)
    reward_list = trajectory[2]
    return_list.append(sum(reward_list))
    episode_list.append(episode)

    agent.update(trajectory)

    if episode % 100 == 0:
        print("回合:{}, 总奖励:{:.1f}".format(episode, sum(reward_list)))
```

首先，在`while`语句中，增加智能体获得的奖励（`reward`）和行动的概率（`prob`）。然后在离开`while`语句后（回合结束时），通过`agent.update()`更新策略。

运行此代码，随着回合的推进，获得的奖励也会增加。下图是结果的示意图。

绘制示意图的代码如下：

```python
def plot_loss(episode_list, return_list, filename):
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")

plot_loss(episode_list, return_list, "pg-loss.pdf")
```

通过观察绘制出来的图像，我们发现每回合的奖励随着训练震荡的很厉害，但是随着训练的进行，每回合获得的总奖励确实越来越多了。

#figure(
  image("rl-figures/pg-loss.svg"),
  caption: [策略梯度算法获得的奖励],
)

我们可以测试一下训练的策略神经网络。看看倒立摆能不能坚持很长时间。

```python
def test_agent(agent, env):
    state = env.reset()
    done = False
    frames = []

    while not done:
        frames.append(env.render(mode="rgb_array"))
        action, _ = agent.get_action(state)
        next_state, _, done, _ = env.step(action)
        state = next_state

    env.close()
    show_animation(frames)

test_agent(agent, env)
```

我们大概可以知道，随着回合的推进，奖励的总和会逐渐增加。但即使经历了 3000 个回合，依然没有达到这次任务的上限值 200，所以似乎还有改进的余地。下面让我们来改进一下这里推导的最简单的策略梯度法。这个改进算法就是著名的#underline("REINFORCE")算法。

=== 深入讨论

采样一条轨迹的梯度是$nabla_theta J(theta) approx sum_(t=0)^T G(tau) nabla_theta log pi_theta (A_t|S_t)$。

如果将梯度符号去掉，可以得到如下：

$
  J(theta) & approx sum_(t=0)^T G(tau) log pi_theta (A_t|S_t) \
           & = G(tau) sum_(t=0)^T log pi_(theta) (A_t|S_t) \
           & = G(tau) (log pi_theta (A_0|S_0) + log pi_theta (A_1|S_1) + dots.c)
$

我们的目标是让$J(theta)$越大越好，那么在经过梯度上升算法之后，$J(theta)$会变大，那么肯定有的$log pi_theta (A_t|S_t)$变大了，也可能有的$log pi_theta (A_(t')|S_(t'))$变小了，但总和肯定变大了。也就是说，策略梯度法的本质是，我们使用*当前策略*$pi_theta$采样一条轨迹$tau$，如果采样的轨迹的回报$G(tau)$很高，那么说明我们在某个状态$S_t$采取的动作$A_t$比较好，导致了总的奖励也就是回报比较高。那么通过梯度上升法，更新$theta$之后，新的策略$pi_theta_"new"$会提升在$S_t$时执行$A_t$的概率。例如$A_t$是"向左推车"，那么在$S_t$状态下策略采取"向左推车"的概率就会提升，而同时策略采取"向右推车"的概率就会下降。

举一个最简单的例子，我们采样的轨迹只执行了一步动作，就结束了。那么有如下公式

$
  J(theta) = G(tau) log pi_theta (A_0|S_0)
$

- 如果$G>0$，那么对$J(theta)$优化之后，策略$pi_theta$在状态$S_0$时采取动作$A_0$的概率一定会变大。
- 如果$G<0$，那么对$J(theta)$优化之后，策略$pi_theta$在状态$S_0$时采取动作$A_0$的概率一定会变小。

如果$G$特别大，例如$G=10000$，或者$G$特别小，例如$G=-10000$，那么梯度上升算法的步长$alpha G nabla_theta log pi_theta (A_0|S_0)$的绝对值一定会很大，进而导致策略的参数$theta$更新幅度巨大，进而导致策略$pi_theta$爆炸式的更新。

换句话说，我们只是提高了在状态$S_0=s$是采取动作$A_0$的概率，但是在玩倒立摆游戏时，并不能保证状态$s$经常出现。碰到其它的状态，更新后的策略表现可能非常的差。好比一个人，突然间中了1000万，那么如此大的奖励会直接彻底改变这个人的策略，他的策略彻底变成了靠运气生活。或者一个人犯了一点点小错误，却得到了巨大的惩罚，如此大的惩罚也会让一个人信心崩溃。这就是策略崩溃的例子。

#figure(
  image("rl-figures/梯度爆炸式更新.svg"),
  caption: [梯度爆炸式更新],
)

== REINFORCE

=== REINFORCE算法原理

REINFORCE是对上一节的策略梯度法的改进算法。本节首先会基于数学式推导REINFORCE算法，然后会通过修改之前的部分代码的做法来实现REINFORCE。

#tip[
  REINFORCE这个名字是"REward Increment = Nonnegative Factor $times$ Offset Reinforcement $times$ Characteristic Eligibility"（奖励增量=非负因子$times$偏移强化$times$特征资格）的首字母缩写。
]

先来复习一下第一节。最简单的梯度策略法是基于下面的公式实现的。

$
  nabla_theta J(theta) & = nabla_theta EE_(tau tilde pi_theta) [G(tau)] \
                       & = EE_(tau tilde pi_theta) [sum_(t=0)^T G(tau) nabla_theta log pi_theta (A_t|S_t)]
$

上面的式子中的$G(tau)$是目前为止获得的所有奖励的总和（准确地说是"带折扣因子"的奖励的总和）。这里要思考的问题是，无论在哪个时刻$t$，式子中都是$G(tau) nabla_theta log pi_theta (A_t|S_t)$，我们始终会使用固定不变的权重$G(tau)$来增加（或减少）采取行动$A_t$的概率。

智能体行动的好坏是根据行动之后获得的奖励总和来评估的（回顾一下价值函数的定义）。反过来说，采取某个行动之前获得的奖励与该行动的好坏无关。如果要评估在某个时刻$t$采取的行动$A_t$，那么在此之前做了什么以及获得了多少奖励都无所谓。我们是根据采取行动$A_t$之后的结果（在时刻$t$以后获得的奖励的总和）来判断行动$A_t$的好坏的。

上面的式子中行动$A_t$的权重是$G(tau)$。这个权重$G(tau)$包括在时刻$t$之前的奖励。也就是说，原本不相关的奖励作为噪声数据包含在内了。为了改进这一点（去除噪声数据），可以对权重$G(tau)$作如下修改。

#theorem(name: [REINFORCE版策略梯度定理])[
  $
    nabla_theta J(theta) & = EE_(tau tilde pi_theta) [sum_(t=0)^T G_t nabla_theta log pi_theta (A_t|S_t)] \
                     G_t & = R_t + gamma R_(t+1) + dots.c + gamma^(T-1) R_T
  $
]

如上式所示，权重变成了$G_t$。权重$G_t$是在时刻$t tilde T$获得的奖励的总和。因此，选择行动$A_t$的概率将由不包含时刻$t$之前的奖励的权重$G_t$增强。这就是改进第一节的策略梯度法的思路。基于上式的算法叫作#underline[REINFORCE]。

#tip[
  基于上式的REINFORCE算法优于最简单的策略梯度法（基于第一节的公式的算法）。通过无限增加的样本数，两个公式都会收敛到正确的$nabla_theta J(theta)$（可以说是无偏差的）。但第一个公式的方差更大，因为公式中的权重包含了无关的数据（噪声）。
]

=== REINFORCE的代码实现

由于REINFORCE的方差小，因此即使数据样本少，也能高精度地近似数据。下面我们来实现REINFORCE以验证其精度。REINFORCE的代码与上一节中的代码基本相同，不同之处只有`Agent`类的`update`方法。下面仅列出了不同部分的代码。

```python
class Agent:
    ...

    def update(self, trajectory):
        states, actions, rewards = trajectory
        G, loss = 0, 0
        for r, s, a in zip(rewards[::-1], states[::-1], actions[::-1]):
            G = r + self.gamma * G # $G_t=R_t+gamma G_(t+1)$
            probs = self.pi(torch.tensor(s).unsqueeze(0)).squeeze(0)
            log_prob = torch.log(probs)[a] # $log pi_theta (A_t|S_t)$
            loss += -log_prob * G # $-sum_(t=0)^T G_t log pi_theta (A_t|S_t)$

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

我们使用可视化代码将回合和奖励绘制出来。

```python
plot_loss(episode_list, return_list, "reinforce-pg-loss.pdf")
test_agent(agent, env)
```

从上图中可以看出，随着回合的推进，奖励的总和会逐渐增加。与上一次的结果相比，不但训练稳定了，训练速度也提高了。

#figure(
  image("rl-figures/reinforce-pg-loss.svg"),
  caption: [REINFORCE策略梯度算法获得的奖励],
)



== 带基线（baseline）的策略梯度法

前面介绍的REINFORCE玩倒立摆有一个问题，那就是不管我们怎么推车，获得的奖励都是正奖励。也就是说不管怎么玩，我们都会提升某个状态采取某个动作的概率，即使奖励很少也是如此。其实我们在玩倒立摆时，获得的奖励很少时，应该降低某个状态下采取某个动作的概率。也就是说，我们应该在获得的奖励很多时，给予策略正的奖励，而在获得奖励很少时，给予策略负的奖励。这样会更加的合理。这就是*基线*的技术。

倒立摆环境中只要倒立摆没有倒，给出的奖励总是1。也就是不管策略多么的差劲，采样出的轨迹的回报一定是正的。也就是说，即使轨迹的回报很低，也会提升在$S_t$采取$A_t$的概率，只是可能提升的不大。

如果加入基线，那么如果轨迹的回报很低，权重就会成为负值，从而在策略更新之后，直接降低在$S_t$采取$A_t$的概率。所以训练速度会加快。

下面介绍基线（baseline）技术，该技术可以改进REINFORCE。让我们先通过一个简单的例子来了解一下基线的思路，然后再将基线应用于REINFORCE。

下式是REINFORCE的数学式。

$
  nabla_theta J(theta) = EE_(tau tilde pi_theta) [sum_(t=0)^T G_t nabla_theta log pi_theta (A_t|S_t)]
$

将基线应用于这个REINFORCE的数学式如下式所示。

#theorem(name: [带基线的REINFORCE策略梯度定理])[
  $
    nabla_theta J(theta) = EE_(tau tilde pi_theta) [sum_(t=0)^T (G_t-b(S_t)) nabla_theta log pi_theta (A_t|S_t)]
  $
]

上式用$G_t-b(S_t)$代替了$G_t$。这里$b(S_t)$可以是任何函数。也就是说，只要输入是$S_t$，$b(S_t)$是什么函数都行。这个$b(S_t)$就是基线（baseline）。

上式中的$b(S_t)$可以是任何函数。例如，在状态$S_t$下，可以考虑使用之前获得的奖励的平均值作为$b(S_t)$。实践中经常使用的是价值函数，数学式为$b(S_t)=V_(pi_theta) (S_t)$。如果能够使用基线减小方差，那么就可以进行样本效率更高的训练。另外，将价值函数作为基线使用时，我们是不知道真正的价值函数$v_(pi_theta) (S_t)$的。这种情况下还需要训练价值函数神经网络。

最后，我们再通过直观介绍补充说明一下为什么使用基线更好。这里以倒立摆为例，思考下图的状态。

#figure(
  image("rl-figures/不管怎么推都会倒下.svg"),
  caption: [无论怎么推，倒立摆都会在3步之后倒下],
)

上图表示游戏结束之前的杆子失去平衡的状态。在这种状态下，无论采取什么样的行动，在几个时间步之后游戏都将结束。

设图的状态为$s$，在此状态下采取的行动为$a$。假定从状态$s$开始几个时间步（比如3个时间步）后游戏一定会结束。在这种情况下，状态$s$的回报$G$为$3$（这里设折扣因子$gamma$为$1$。如果使用的是没有基线的REINFORCE，那么状态$s$下的行动$a$就会因为权重$3$而被增强（状态$s$下选择行动$a$的概率会变高）。但无论采取什么样的行动，3个时间步之后游戏一定会结束，所以可以说这种提高行动$a$被选择的概率的工作是无意义的。

此时就要用到基线了。这里使用价值函数作为基线，假设我们已经知道图中的例子中的$V_(pi_theta) (S_t)=3$。此时的权重为$G_t-V_(pi_theta)$所以是0。由于权重是0，因此无论选择什么行动，采取那个行动的概率都不会变大，也不会变小。像这样使用基线，有望减少无谓的训练。

== Actor-Critic（演员-评论家架构）

如果在上节介绍的带基线的REINFORCE中使用价值函数作为基线，那么就可以将其视为基于价值且基于策略的方法。本节将进一步改进带基线的REINFORCE，推导一个叫作Actor-Critic的算法。Actor-Critic也是基于价值且基于策略的方法。

#figure(
  image("rl-figures/演员评论家漫画.jpg", width: 60%),
  caption: [演员评论家架构],
)

=== Actor-Critic的数学推导

首先从复习带基线的REINFORCE开始。带基线的REINFORCE的目标函数的梯度的数学式如下所示。

$
  nabla_theta J(theta) = EE_(tau tilde pi_theta) [sum_(t=0)^T (G_t-b(S_t))nabla_theta log pi_theta (A_t|S_t)]
$

上式中的$G_t$表示回报，$b(S_t)$表示基线。可以使用任何函数作为基线。这里我们使用基于神经网络建模的价值函数作为基线。因此，我们要用到以下这些新的记号。

- $omega$：表示价值函数的神经网络的所有权重参数。
- $V_omega (S_t)$：将价值函数模型化的神经网络。

此时目标函数的梯度的数学式如下式所示

#theorem(name: "Actor-Critic架构的REINFORCE策略梯度定理")[
  $
    nabla_theta J(theta) = EE_(tau tilde pi_theta) [sum_(t=0)^T (G_t - V_omega (S_t))nabla_theta log pi_theta (A_t|S_t)]
  $
]

上式中存在一个问题，即只要没抵达目标，就无法确定回报$G_t$的值。也就是说，在抵达目标之前，无法更新策略和价值函数。如果这是基于蒙特卡洛方法的算法，那么二者无法更新就是它的缺点。

#tip(title: [蒙特卡洛采样])[
  我们之前使用的是*蒙特卡洛采样*，也就是采样出一条完整的轨迹。然后更新策略神经网络$pi_theta$。
]

消除这个缺点的方法是*时序-差分方法*（TD方法，Time Difference Method）。使用TD方法训练价值函数时，使用1个时间步（或$n$个时间步）后的结果就能进行更新，如图所示。

#figure(
  image("rl-figures/蒙特卡洛方法和TD方法对比.svg", width: 50%),
  caption: [蒙特卡洛方法和TD方法对比],
)

如图所示，在训练价值函数$V_omega (S_t)$时，蒙特卡洛方法使用的是回报$G_t$，而TD方法使用的是$R_t+gamma V_omega (S_(t+1))$ 。

#tip(title: [时序差分误差])[
  使用神经网络对价值函数建模时，我们以接近$R_t+gamma V_omega (S_(t+1))$为目标训练$V_omega (S_t)$的值。具体来说就是将$V_omega (S_t)$和$R_t+gamma V_omega (S_(t+1))$的均方差作为损失函数，通过梯度下降法更新神经网络的权重。

  贝尔曼期望方程：$V_omega (S_t = s) = EE_pi [G_t|S_t=s] = EE_pi [ R_t + gamma V_omega (S_(t+1)=s') | S_t = s]$

  如果我们的价值函数是#underline[真实的]价值函数的话，那么用$R_t + gamma V_omega (S_(t+1))$替换掉$G_t$是*无偏的*。两者区别在于：

  - $G_t$是真实的未来回报。
  - $R_t + gamma V_omega (S_(t+1))$是用价值函数来估计未来的回报。

  如果$V$是真实的价值函数，那么在期望意义下，$V_omega (S_t = s) = EE_pi [G_t|S_t=s] = EE_pi [ R_t + gamma V_omega (S_(t+1)=s') | S_t = s]$。

  而我们的$V_omega$是一个神经网络，显然不是真实的价值函数，所以不能保证是#underline[无偏估计]。所以希望随着训练，能够逼近真实的价值函数。但也有可能偏差越来越大！

  可以看到$V_omega (S_t) arrow R_t+gamma V_omega (S_(t+1))$是由贝尔曼期望方程而来的。
]

下面将基于蒙特卡洛方法的公式切换为TD方法，其中，代替$G_t$的是$R_t+gamma V_omega (S_(t+1))$。此时得到的式子如下所示。

#theorem(name: [1步TD误差的Actor-Critic策略梯度定理])[
  $
    nabla_theta J(theta)=EE_(tau tilde pi_theta) [sum_(t=0)^T (R_t+gamma V_omega (S_(t+1))-V_omega (S_t)) nabla_theta log pi_theta (A_t|S_t)]
  $
]

1步TD误差经常表示为$delta = R_t + gamma V_omega (S_(t+1)) - V_omega (S_t)$。

基于上面的公式的算法就是Actor-Critic。策略$pi_theta$和价值函数$V_omega$是神经网络，我们要同时训练这两个神经网络。具体来说，对于策略$pi_theta$要基于上面的公式进行训练；而对于价值函数$V_omega$，则通过TD方法，以接近$R_t+gamma V_omega (S_(t+1))$为目标训练$V_omega (S_t)$这个神经网络。

#tip[
  Actor-Critic中的Actor是*演员*（采取动作的人）的意思，也就是采取动作的人，相当于策略$pi_theta$。而Critic是*评论家*的意思，相当于价值函数$V_omega$。因此，Actor-Critic的意思是"使用$V_omega$，来评论基于策略$pi_theta$采取的动作的好坏"。
]

#tip(title: [为什么可以用$R_t+gamma V_omega (S_(t+1))$替换掉$G_t$？])[
  首先我们知道$G_t=R_t+gamma G_(t+1)$以及$V_omega (S_(t+1))=EE[G_(t+1)]$。这里我们假设训练到理想情况，$V_omega (S_(t+1))$接近于真实的$V(S_(t+1))$。所以我们做了替换。]

=== Actor-Critic的代码实现

下面实现Actor-Critic。策略和价值函数这两个神经网络的代码如下所示。

#codly(header: [策略网络和价值网络])
```python
class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
```

上面的代码将策略的网络实现为了`PolicyNet`类，将价值函数的网络实现为了`ValueNet`类。策略的最终输出是`Softmax`函数的输出，所以输出的是概率。接下来是`Agent`类的代码。

#codly(header: [演员评论家架构中的智能体])
```python
class Agent:
    def __init__(self):
        self.gamma = 0.98 # 折扣因子$gamma$
        self.lr_pi = 0.0002 # 策略网络的学习率$alpha_"actor"$
        self.lr_v = 0.0005 # 价值网络的学习率$alpha_"critic"$
        self.action_size = 2
        # 初始化策略网络（演员网络）$pi_theta$
        self.pi = PolicyNet(self.action_size)
        # 初始化价值函数网络（评论家网络）$V_omega$
        self.v = ValueNet()

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs

    def update(
        self,
        state, # $S_t$
        action_prob, # $pi_theta (A_t|S_t)$
        reward, # $R_t$
        next_state, # $S_(t+1)$
        done
    ):
        state = torch.tensor(state).unsqueeze(0) # $S_t$
        next_state = torch.tensor(next_state).unsqueeze(0) # $S_(t+1)$

        # ① self.v的损失：均方差
        # $"TD目标"=R_t+gamma V_omega (S_(t+1))$
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        v = self.v(state) # $V_omega (S_t)$
        loss_fn = nn.MSELoss() # $(R_t+gamma V_omega (S_(t+1))-V_omega (S_t))^2 arrow 0$
        loss_v = loss_fn(v, target.detach()) # target要从计算图中剥离出去

        # ② self.pi的损失
        delta = target - v # $delta=R_t+gamma V_omega (S_(t+1)) - V_omega (S_t)$
        # $-(R_t+gamma V_omega (S_(t+1)) - V_omega (S_t) ) log pi_theta (A_t|S_t)$
        loss_pi = -torch.log(action_prob) * delta.detach().item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward() # $nabla_omega (R_t+gamma V_omega(S_(t+1))-V_omega (S_t))^2$
        loss_pi.backward() # $-( R_t+gamma V_omega(S_(t+1))-V_omega (S_t))nabla_theta log pi_theta (A_t|S_t)$
        self.optimizer_v.step()
        self.optimizer_pi.step()
```

`get_action`方法可以基于策略取出动作。需要注意的是，由于输入到神经网络中的数据将作为小批量进行处理，因此在处理其中一个数据（状态）时需要小批量的轴。另外，`get_action`方法返回了两个值，即选择的动作及其概率。选择动作的概率将在稍后的损失函数计算中使用。

`update`方法可以训练价值函数和策略。在代码①处为价值函数(`self.v`)计算损失。为此要计算TD目标（target），求出当前状态下其与价值函数（v）的均方差。然后，在代码②处为策略（`self.pi`）计算损失。需要将其乘以$-1$的值作为损失。剩下的就是一直以来的神经网络的训练代码。

#codly(header: [训练循环])
```python
env = gym.make("CartPole-v0")
agent = Agent()
return_list = []
episode_list = []

for episode in range(2000):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, probs = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        # 执行完一个动作，立即更新$pi_theta$和$V_omega$这两个网络
        agent.update(state, probs[action], reward, next_state, done)

        state = next_state
        total_reward += reward

    return_list.append(total_reward)
    episode_list.append(episode)
    if episode % 100 == 0:
        print("回合:{}, 总奖励:{:.1f}".format(episode, total_reward))

plot_loss(episode_list, return_list, "actor-critic-pg-loss.pdf")
test_agent(agent, env)
```

#figure(
  image("rl-figures/actor-critic-pg-loss-1.pdf"),
  caption: [actor-critic每一轮的回报走势图],
)

== Actor-Critic的代码实现（完整采集一条轨迹的版本）

```python
# actor-critic的实现

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


def plot_loss(episode_list, return_list, filename):
    """绘制奖励图像"""
    f = plt.figure()
    plt.plot(episode_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("CartPole-v0")
    plt.show()
    f.savefig(filename, bbox_inches="tight")


class PolicyNet(nn.Module):
    """策略神经网络的结构"""

    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):  # $x$是$S_t$
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):
    """价值函数神经网络V_ω"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.005
        self.action_size = 2
        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs

    def collect_trajectory(self, env):
        """采样一条轨迹"""
        state = env.reset()
        states, next_states, actions, rewards, dones = [], [], [], [], []
        done = False

        while not done:
            action, _ = self.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)  # $S_t$
            next_states.append(next_state)  # $S_(t+1)$
            actions.append(action)  # $A_t$
            rewards.append(reward)  # $R_t$
            dones.append(done)  # $"done"_t$

            state = next_state

        # states: $[S_0, S_1, S_2, ..., S_(T-1)]$
        # next_states: $[S_1, S_2, S_3, ..., S_T]$
        # actions: $[A_0, A_1, A_2, ..., A_(T-1)]$
        # rewards: $[R_0, R_1, R_2, ..., R_(T-1)]$
        # dones: [False, False, False, ..., True]
        return states, next_states, actions, rewards, dones

    def update(self, trajectory):
        """整条轨迹的actor-critic更新"""
        states, next_states, actions, rewards, dones = trajectory

        states = torch.tensor(states)  # $[S_0, S_1, S_2, ..., S_(T-1)]$
        # $[[A_0], [A_1], [A_2], ..., [A_(T-1)]]$
        actions = torch.tensor(actions).view(-1, 1)
        # $[[R_0], [R_1], [R_2], ..., [R_(T-1)]]$
        rewards = torch.tensor(rewards).view(-1, 1)
        next_states = torch.tensor(next_states)  # $[S_1, S_2 ..., S_T]$
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1)

        v = self.v(states)  # $[V(S_0), V(S_1), ..., V(S_(T-1))]$
        # $[R_0+gamma V(S_1), R_1+gamma V(S_2), ..., R_(T-1)]$
        td_target = rewards + self.gamma * self.v(next_states) * (1 - dones)
        # 价值网络的损失，把td目标从计算图中剥离
        loss_v = F.mse_loss(v, td_target.detach())
        # 策略网络的损失，gather的用法要注意
        # self.pi(states): $π_θ (a_t|s_t)$
        # actions是真实标签
        # gather根据真实标签actions取出对应的概率
        # 例如self.pi(states)如下：
        # [[0.3, 0.7],
        #  [0.2, 0.8]]
        # actions如下：
        # [[1],
        #  [0]]
        # gather后的结果
        # [[0.7],
        #  [0.2]]
        action_probs = self.pi(states).gather(1, actions)
        # torch.log(action_probs): $[log π_θ (a_0|s_0),...,log π_θ (a_(T-1)|s_(T-1))]$
        # (td_target-v).detach():
        # $[R_0+gamma V(S_1)-V(S_0),...,R_(T-1)+gamma V(S_T)-V(S_(T-1))]$
        loss_pi = - \
            torch.sum(torch.log(action_probs)
                      * (td_target-v).detach())

        self.optimizer_pi.zero_grad()
        self.optimizer_v.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_pi.step()
        self.optimizer_v.step()


env = gym.make("CartPole-v0")
agent = Agent()
return_list = []
episode_list = []

for episode in range(3000):
    state = env.reset()  # $S_0$

    trajectory = agent.collect_trajectory(env)
    # 采样一条轨迹，更新一次策略网络和价值网络
    agent.update(trajectory)

    return_list.append(sum(trajectory[3]))
    episode_list.append(episode)
    if episode % 100 == 0:
        print(f"回合：{episode}, 总奖励：{sum(trajectory[3])}")

plot_loss(episode_list, return_list, "sample-one-trajectory-actor-critic-pg-loss.pdf")
```

损失图像如下

#figure(
  image("rl-figures/sample-one-trajectory-actor-critic-pg-loss.pdf"),
  caption: [采样一条完整的轨迹来计算单步TD误差],
)

== 多步TD误差

每个时刻的1步TD误差计算如下：

$
      delta_t & = R_t + gamma V(S_(t+1)) - V(S_t) \
  delta_(t+1) & = R_(t+1) + gamma V(S_(t+2)) - V(S_(t+1)) \
  delta_(t+2) & = R_(t+2) + gamma V(S_(t+3)) - V(S_(t+2)) \
  delta_(t+3) & = R_(t+3) + gamma V(S_(t+4)) - V(S_(t+3)) \
       dots.c
$

1步TD误差为：

$
  A_t^((1)) & = R_t + gamma V(S_(t+1)) - V(S_t) \
            & = -V(S_t) + R_t + gamma V(S_(t+1)) \
            & = delta_t
$

#tip(title: [贝尔曼期望方程])[
  $
    V(S_(t+1)) = EE[R_(t+1) + gamma V(S_(t+2))]
  $
]

将式子中的$V(S_(t+1))$由$R_(t+1)+gamma V(S_(t+2))$替换，得到如下式子，就是2步TD误差。

$
  A_t^((2)) & = -V(S_t) + R_t + gamma (R_(t+1) + gamma V(S_(t+2))) \
  & = -V(S_t)+R_t+gamma R_(t+1)+gamma^2 V(S_(t+2)) \
  & = underbrace(R_t + gamma V(S_(t+1)) - V(S_t), delta_t) + gamma (underbrace(R_(t+1) + gamma V(S_(t+2)) - V(S_(t+1)), delta_(t+1))) \
  & = delta_t + gamma delta_(t+1)
$

== 广义优势估计

#tip[
  广义优势估计：Generalized Advantage Estimation，GAE
]

上面的1步TD误差表示为：$delta = R_t + gamma V_omega (S_(t+1)) - V_omega (S_t)$。在很多强化学习文献中，这个叫做优势$A$（Advantage）。

也就是1步TD目标$R_t + gamma V_omega (S_(t+1))$相对于基线$V_omega (S_t)$的优势是多少。

我们在上一节的Actor-Critic中使用1步TD误差来作为优势。而现在常用的一种方法是针对*多步TD误差*计算指数加权平均的方式来估算优势，也就是*广义优势估计*。

首先，$t$时刻的1步TD误差是：

$
  delta_t = R_t + gamma V(s_(t+1)) - V(s_t)
$

其中$V$是价值函数神经网络$V_omega$。那么我们计算一下多步TD误差：

$
  A_t^((1)) &= delta_t &&= -V(s_t)+R_t + gamma V(s_(t+1)) \
  A_t^((2)) &= delta_t+gamma delta_(t+1) &&= -V(s_t)+R_t+gamma R_(t+1)+gamma^2 V(s_(t+2)) \
  A_t^((3)) &= delta_t + gamma delta_(t+1) + gamma^2 delta_(t+2) &&= -V(s_t)+R_t+gamma R_(t+1) + gamma^2 R_(t+2) + gamma^3 V(s_(t+3)) \
  &space dots.v && space space dots.v \
  A_t^((k)) &= sum_(l=0)^(k-1) gamma^l delta_(t+l) &&= -V(s_t)+R_t+gamma R_(t+1)+dots.c+gamma^(k-1) R_(t+k-1)+gamma^k V(s_(t+k))
$

然后，GAE将这些不同步数的优势估计进行指数加权平均：

$
  A_t^"GAE" &= (1-lambda)(A_t^((1))+lambda A_t^((2))+lambda^2 A_t^((3))+ dots.c) \
  &= (1-lambda)( delta_t+lambda(delta_t+gamma delta_(t+1)) + lambda^2 (delta_t+gamma delta_(t+1) + gamma^2 delta_(t+2))+ dots.c) \
  &= (1-lambda)(delta_t (1+lambda+lambda^2+dots.c)+gamma delta_(t+1) (lambda+lambda^2+lambda^3+dots.c)+gamma^2 delta_(t+2) (lambda^2+lambda^3+lambda^4+dots.c)+ dots.c) \
  &= (1-lambda) ( delta_t 1/(1-lambda)+gamma delta_(t+1) lambda/(1-lambda)+gamma^2 delta_(t+2) lambda^2 / (1-lambda) + dots.c ) \
  &= sum_(l=0)^infinity (gamma lambda)^l delta_(t+l)
$

其中，$lambda in [0,1]$是在GAE中额外引入的一个超参数。当$lambda=0$时，$A_t^"GAE"= delta_t=R_t+gamma V(s_(t+1))-V(s_t)$ ，也就是仅仅只看一步差分得到的优势；当$lambda=1$时，$A_t^"GAE"=sum_(l=0)^infinity gamma^l delta_(t+l)=sum_(l=0)^infinity gamma^l R_(t+l)-V(s_t)$，则是看每一步差分得到的优势的完全平均值。

有上面的式子，我们还可以推导出一个递推公式

$
  A_t^"GAE" = delta_t + gamma lambda A_(t+1)^"GAE"
$

下面是计算GAE的过程，给定 $gamma$ 和 $lambda$ 以及每个时间步的 $delta_t$ 之后，我们可以根据公式直接进行优势估计。

先前向（forward）计算每一个时刻的*1步TD误差*，然后再逆向（backward）计算每一个时刻的GAE。

首先，我们有 $n$ 个1步TD误差。

$
  "TD"_delta = [ delta_t, delta_(t+1), dots.c, delta_(t+n) ]
$

先把数组逆序

$
  "TD"_delta = [ delta_(t+n), delta_(t+n-1), dots.c, delta_t]
$

然后遍历逆序数组，有如下结果：

$
    A_(t+n)^"GAE" & = delta_(t+n) \
  A_(t+n-1)^"GAE" & = delta_(t+n-1) + gamma lambda A_(t+n)^"GAE" \
  A_(t+n-2)^"GAE" & = delta_(t+n-2) + gamma lambda A_(t+n-1)^"GAE" \
           dots.v \
        A_t^"GAE" & = delta_t + gamma lambda A_(t+1)^"GAE"
$

这样每一个时间步的广义优势估计就计算出来了。

#tip(title: [GAE目标])[
  $"GAE Target" = A_t^"GAE" + V(S_t)$
]

#theorem(name: [GAE的Actor-Critic策略梯度定理])[
  $
    nabla_theta J(theta)=EE_(tau tilde pi_theta) [sum_(t=0)^T A^"GAE"_t nabla_theta log pi_theta (a_t|s_t)]
  $
]

#tip[
  为了使符号更清晰，将动作$A_t$改为$a_t$，将状态$S_t$改为$s_t$。$A^"GAE"_t$中的$A$表示"Advantage"（优势）。
]

== 相关数学证明

=== 策略梯度法的证明

当$J(theta)=EE_(tau tilde pi_theta) [G(tau)]$时，其梯度如下面的式子所示。

$
  nabla_theta J(theta)=EE_(tau tilde pi_theta) [sum_(t=0)^T G(tau)nabla_theta log pi_theta (A_t|S_t)]
$

下面对上面的式子进行证明。

$
  nabla_theta J(theta) &= nabla_theta EE_(tau tilde pi_theta) [G(tau)] \
  &= nabla_theta sum_tau "Pr"(tau|theta) G(tau) colblue("（展开期望值）") \
  &= sum_tau nabla_theta ("Pr"(tau|theta)G(tau)) colblue("（将"nabla_theta"移动到"sum"中）") \
  &= sum_tau { G(tau)nabla_theta"Pr"(tau|theta) + "Pr"(tau|theta)nabla_theta G(tau) } colblue("（积的微分）") \
  &= sum_tau G(tau)nabla_theta"Pr"(tau|theta) colblue("（"nabla_theta G(tau)"永远为0）") \
  &= sum_tau G(tau)"Pr"(tau|theta) colred((nabla_theta"Pr"(tau|theta))/("Pr"(tau|theta))) colblue("（乘以""Pr"(tau|theta)/"Pr"(tau|theta)"）") \
  &= sum_tau G(tau)"Pr"(tau|theta) colred(nabla_theta log"Pr"(tau|theta)) colblue("（"log"梯度技巧）") \
  &= EE_(tau tilde pi_theta) [G(tau)nabla_theta log"Pr"(tau|theta)]
$ <pgproof>

这里对"log梯度的技巧"进行说明。这个技巧利用了以下等式。

$
  nabla_theta log"Pr"(tau|theta)=(nabla_theta"Pr"(tau|theta))/("Pr"(tau|theta))
$

#tip(title: [log梯度技巧])[
  $
    (log f(x))' = (f'(x))/f(x)
  $ <logtrick>
]

根据上面的式子，我们就知道

$
  nabla_theta"Pr"(tau|theta)="Pr"(tau|theta)nabla_theta log"Pr"(tau|theta)
$

这就是著名的*log梯度的技巧*。是机器学习领域常用的数学式的变形形式。

接下来，我们将利用以下等式进一步展开@pgproof。

$
  "Pr"(tau|theta) & = p(S_0)pi_theta (A_0|S_0)p(S_1|S_0,A_0)dots.c pi_theta (A_T|S_T)p(S_(T+1)|S_T,A_T) \
                  & = p(S_0)product_(t=0)^T pi_theta (A_t|S_t)p(S_(t+1)|S_t,A_t)
$

#tip(title: [轨迹的概率])[
  轨迹是：$tau=(S_0,A_0,S_1,A_1,S_2,A_2,S_3)$，那么轨迹的概率如下计算：
  $
    "Pr"(tau) & = p(S_0,A_0,S_1,A_1,S_2,A_2,S_3) \
              & = p(S_3|S_0,A_0,S_1,A_1,S_2,A_2)p(S_0,A_0,S_1,A_1,S_2,A_2) \
              & = p(S_3|S_2,A_2)p(S_0,A_0,S_1,A_1,S_2,A_2) \
              & = p(S_3|S_2,A_2)p(A_2|S_0,A_0,S_1,A_1,S_2)p(S_0,A_0,S_1,A_1,S_2) \
              & = p(S_3|S_2,A_2)p(A_2|S_2)p(S_0,A_0,S_1,A_1,S_2) \
              & = p(S_3|S_2,A_2)pi_theta (A_2|S_2)p(S_0,A_0,S_1,A_1,S_2) \
              & = p(S_3|S_2,A_2)pi_theta (A_2|S_2)p(S_2|S_0,A_0,S_1,A_1)p(S_0,A_0,S_1,A_1) \
              & = p(S_3|S_2,A_2)pi_theta (A_2|S_2)p(S_2|S_1,A_1)p(S_0,A_0,S_1,A_1) \
              & = p(S_3|S_2,A_2)pi_theta (A_2|S_2)p(S_2|S_1,A_1)p(A_1|S_0,A_0,S_1)p(S_0,A_0,S_1) \
              & = p(S_3|S_2,A_2)pi_theta (A_2|S_2)p(S_2|S_1,A_1)p(A_1|S_1)p(S_0,A_0,S_1) \
              & = p(S_3|S_2,A_2)pi_theta (A_2|S_2)p(S_2|S_1,A_1)pi_theta (A_1|S_1)p(S_0,A_0,S_1) \
              & = p(S_3|S_2,A_2)pi_theta (A_2|S_2)p(S_2|S_1,A_1)pi_theta (A_1|S_1)p(S_1|S_0,A_0)p(S_0,A_0) \
              & = p(S_3|S_2,A_2)pi_theta (A_2|S_2)p(S_2|S_1,A_1)pi_theta (A_1|S_1)p(S_1|S_0,A_0)p(A_0|S_0)p(S_0) \
              & = p(S_3|S_2,A_2)pi_theta (A_2|S_2)p(S_2|S_1,A_1)pi_theta (A_1|S_1)p(S_1|S_0,A_0)pi_theta (A_0|S_0)p(S_0)
  $
]

这里，$p(S_0)$表示初始状态$S_0$的概率。上面的式子表明，得到轨迹$tau$的概率可以用初始状态的概率、策略以及下一个状态的转移概率的乘积来表示。另外，我们可以用下面的式子来表示$log"Pr"(tau|theta)$。

$
  log"Pr"(tau|theta)=log p(S_0) + sum_(t=0)^T log p(S_(t+1)|S_t,A_t) + sum_(t=0)^T log pi_theta (A_t|S_t)
$

由于$log x y = log x + log y$，所以可以像上面的式子那样表示为和的形式。基于上面的式子，可以将$nabla_theta log"Pr"(tau|theta)$展开为如下形式。

$
  nabla_theta log"Pr"(tau|theta) &= nabla_theta { log p(S_0) + sum_(t=0)^T log p(S_(t+1)|S_t,A_t) + sum_(t=0)^T log pi_theta (A_t|S_t) } \
  &= nabla_theta sum_(t=0)^T log pi_theta (A_t|S_t)
$

$nabla_theta$是对$theta$的梯度。与$theta$无关的元素的梯度$nabla_theta log p(S_0)$和$nabla_theta sum_(t=0)^T log p(S_(t+1)|S_t,A_t)$为0。因此，从上面的式子可以得到下列式子。

$
  nabla_theta J(theta) & =EE_(tau tilde pi_theta) [G(tau) nabla_theta log"Pr"(tau|theta)] \
                       & = EE_(tau tilde pi_theta) [sum_(t=0)^T G(tau)nabla_theta log pi_theta (A_t|S_t)]
$

这样我们就完成了$nabla_theta J(theta)$的推导。

#tip(title: [从极大似然估计的角度看策略梯度])[
  $
    J(theta) & = sum_(t=0)^T G(tau) log pi_theta (A_t|S_t) \
             & = G(tau)log(product_(t=0)^T pi_theta (A_t|S_t))
  $
  最大化目标就会提升某些状态对应的动作的概率。
]

=== 基线（Baseline）的推导

$
  nabla_theta J(theta) & = EE_(tau tilde pi_theta) [sum_(t=0)^T G_t nabla_theta log pi_theta (A_t|S_t)] \
  & = EE_(tau tilde pi_theta) [sum_(t=0)^T (G_t-b(S_t)) nabla_theta log pi_theta (A_t|S_t)] \
  &= EE_(tau tilde pi_theta) [sum_(t=0)^T G_t nabla_theta log pi_theta (A_t|S_t)] - underbrace(EE_(tau tilde pi_theta) [sum_(t=0)^T b(S_t) nabla_theta log pi_theta (A_t|S_t)], "证明这一项等于0即可") \
  &= EE_(tau tilde pi_theta) [sum_(t=0)^T G_t nabla_theta log pi_theta (A_t|S_t)]
$

如上面的式子所示，我们可以使用$G_t-b(S_t)$代替$G_t$。$b(S_t)$是*任何函数*，我们称之为"基线"。下面进行上式的推导。

首先，证明以下式子成立。

$
  EE_(x tilde P_theta) [nabla_theta log P_theta (x)] = 0
$ <baselineproof-1>

这里假设随机变量$x$是基于概率分布$P_theta (x)$生成的。$P_theta (x)$会根据参数$theta$改变概率分布的形状。此时有以下式子成立。

$
  sum_x P_theta (x) = 1
$

由于$P_theta (x)$是概率分布，因此所有$x$的值的和为1。然后，求这个式子的梯度。

$
  nabla_theta sum_x P_theta (x) = nabla_theta 1 = 0
$

接下来，使用log梯度的技巧将式子展开，过程如下所示。

$
  0 & = nabla_theta sum_x P_theta (x) \
    & = sum_x nabla_theta P_theta (x) \
    & = sum_x P_theta (x) nabla_theta log P_theta (x) colblue("（log梯度技巧）") \
    & = EE_(x tilde P_theta) [nabla_theta log P_theta (x)]
$

@baselineproof-1 证明完毕。接下来将证明的式子用于我们的问题。具体来说，用$A_t$代替@baselineproof-1 中的$x$，然后使用$pi_theta (dot.c|S_t)$代替$P_theta (dot.c)$。这样就可以得到以下式子。

$
  EE_(A_t tilde pi_theta) [nabla_theta log pi_theta (A_t|S_t)] = 0
$ <baselineproof-2>

上面的式子是对$A_t$的期望值。因此，我们可以像下面的式子那样，将任何函数$b(S_t)$放入期望值中。$E[x]=0 arrow E[c x]=c dot.c 0 = 0$。

$
  EE_(A_t tilde pi_theta) [b(S_t) nabla_theta log pi_theta (A_t|S_t)] = 0
$ <baselineproof-3>

$b(S_t)$是以$S_t$为参数的函数，即使$A_t$发生变化，它的值也不会改变。由于@baselineproof-3 是对$A_t$的期望值，因此即使在期望值中加入函数$b(S_t)$，等式也成立。

#danger[
  动作$A_t$的变化会导致收益$G_t$的变化，因此以下式子不成立。
  $
    EE_(A_t tilde pi_theta) [G_t nabla_theta log pi_theta (A_t|S_t)] = 0
  $
]

@baselineproof-3 在整个$t=0 tilde T$的范围都成立，所以可以得到以下式子。

$
  EE_(A_t tilde pi_theta) [sum_(t=0)^T b(S_t) nabla_theta log pi_theta (A_t|S_t)] = 0
$

所以基线证明完毕。


=== REINFORCE证明

可以证明：把整条轨迹回报$G(tau)$换成从时刻$t$开始的回报$G(t)$，本质上是利用了*因果性*：时刻$t$的动作不会影响时刻$t$之前已经发生的奖励。因此，过去奖励那部分在期望下对梯度贡献为$0$。

将$G(tau)$拆成过去和未来两部分

对某个时刻$t$，可以把整条轨迹回报拆成：

$
  G(tau)
  =
  sum_(k=0)^(t-1) gamma^k R_k
  +
  sum_(k=t)^(T-1) gamma^k R_k
$

记过去回报为

$
  G_(<t)
  =
  sum_(k=0)^(t-1) gamma^k R_k
$

未来回报为

$
  G_(>= t)
  =
  sum_(k=t)^(T-1) gamma^k R_k
$

于是有

$
  G(tau)=G_(<t)+G_(>= t)
$

代入梯度中的第$t$项：

$
  EE[nabla_theta log pi_theta (A_t|S_t) G(tau)]
$

得到

$
  EE[nabla_theta log pi_theta (A_t|S_t) G_(<t)] +
  EE[nabla_theta log pi_theta (A_t|S_t) G_(>=t)]
$

关键是证明第一项为0。

在上一节中，我们证明过：由于$G_(<t)$不依赖于$A_t$变化，所以相当于是常数。

所以第一项为0。

如果定义从时刻$t$开始的"reward-to-go"为

$
  G_t = sum_(k=t)^(T-1) gamma^(k-t) R_k
$

那么上式也常写为

$
  nabla_theta J(theta) = EE_(tau tilde pi_theta) [sum_(t=0)^(T-1) gamma^t nabla_theta log pi_theta (A_t|S_t) G_t]
$

很多教材会把折扣因子的位置简化或吸收到$G_t$的定义中，因此也常见写法：

$
  nabla_theta J(theta) = EE_(tau tilde pi_theta) [sum_(t=0)^(T-1) nabla_theta log pi_theta (A_t|S_t) G_t]
$

直观解释

原始形式中，第$t$个策略梯度项是：

$
  nabla_theta log pi_theta (A_t|S_t)G(tau)
$

但$A_t$只会影响：

- 当前奖励；
- 未来状态；
- 未来奖励。

它不可能影响已经发生的过去奖励。因此把过去奖励乘到

$
  nabla_theta log pi_theta (A_t|S_t)
$

上，只会增加噪声，不会改变期望值。

所以用$G_t$代替$G(tau)$不会改变梯度的无偏性，但可以降低方差。

结论

REINFORCE中将整条轨迹回报$G(tau)$替换为从当前时刻开始的回报$G(t)$是成立的，因为对于任意时刻$t$，过去奖励部分满足

$
  EE[nabla_theta log pi_theta (A_t|S_t) sum_(k=0)^(t-1) gamma^k R_k] = 0
$

因此删掉过去奖励不会引入偏差，只是减少了方差（噪声）。这就是所谓的"reward-to-go trick"或"因果性改进"。


== 总结

在本章中，我们学习了基于策略的方法——策略梯度法。具体来说，我们学习了几种策略梯度法的算法。它们的统一的数学式如下所示。

$
  nabla_theta J(theta) = EE_(tau tilde pi_theta) [sum_(t=0)^T Phi_t nabla_theta log pi_theta (A_t|S_t)]
$

$
  "1." space space space & Phi_t=G(tau)                          & "（最简单的策略梯度法）" \
  "2." space space space & Phi_t=G_t                             &          "（REINFORCE）" \
  "3." space space space & Phi_t=G_t-b(S_t)                      &  "（带基线的REINFORCE）" \
  "4." space space space & Phi_t = R_t + gamma V(S_(t+1))-V(S_t) &       "（Actor-Critic）" \
  "5." space space space & Phi_t = A_t^"GAE"                     &       "（广义优势估计）"
$

上面几种方法的权重$Phi_t$各不相同。最简单的策略梯度法在所有时刻的权重都是$G(tau)$。对它进行改进，将时刻$t$的收益$G_t$作为权重进行评估的是REINFORCE。另外，通过加入"基线"这一方法，减小了方差的方法是带基线的REINFORCE。Actor-Critic是一种除了策略之外，对价值函数也用神经网络建模的方法。期待$(1) arrow (2) arrow (3) arrow (4) arrow (5)$这样的编号一直延续下去，从而有更高级的方法出现，产生更好的结果。

#tip(title: [优势函数])[
  $Phi_t$也叫做优势函数（Advantage Function），优势函数经常在文献中表示为$A(s,a)$，也就是在状态$s$采取动作$a$时，策略的优势是多少。

  强化学习中的很多花活儿都是针对如何改进$Phi_t$而来的。比如GRPO算法使用的*组相对优势*。
]

#chapter("近端策略优化（PPO）", image: image("./orange2.jpg"), l: "rl-ppo")

#tip[
  PPO: Proximal Policy Optimization，近端策略优化
]

== 策略梯度法存在的问题

回顾一下策略梯度法的公式。

$
  nabla_theta J(theta) & = nabla_theta EE_(tau tilde pi_theta) [G(tau)] \
                       & = EE_(tau tilde pi_theta) [sum_(t=0)^T G(tau) nabla_theta log pi_theta (A_t|S_t)]
$

还有梯度上升算法

$
  theta arrow.l theta + alpha nabla_theta J(theta)
$

*步子迈得太大会导致灾难*。但如果步子太小，模型学习速度就会太慢。把奖励函数想象成高耸的山峰。如果新策略走得太远，它采取的行动可能会差之毫厘，最终从悬崖上掉下来。当我们重新开始探索时，我们从一个表现不佳的状态开始，采用局部糟糕的策略。性能会崩溃，而且需要很长时间才能恢复。

#figure(
  image("rl-figures/策略梯度法步子太大.png"),
  caption: [梯度上升算法中步幅太大，策略直接崩溃了],
)

#figure(
  image("rl-figures/策略梯度法的问题.png"),
  caption: [左图是步幅太大，中间的图是理想的步幅，右边的图是方向不对],
)

#tip(title: [为什么在深度学习中这个问题没有那么严重呢？])[
  深度学习中，使用的训练数据集是固定不变的。换句话说，不管神经网络的参数怎么变，训练数据集是不会变的。也就是说，神经网络的参数和训练数据集没关系。

  而强化学习中，训练数据是每一轮根据策略来采样的新的轨迹。而采样的轨迹的质量好坏无法控制。也就是说训练数据集和策略神经网络的参数是有关系的。这就麻烦了。

  我们真正应该优化的目标是$J(theta)=EE_(tau tilde pi_theta) [G(tau)]$。但在蒙特卡洛采样法只采样一条轨迹的情况下，我们优化的目标成了：
  $
    J(theta) approx sum_(t=0)^T G(tau) log pi_theta (a_t|s_t)
  $
  所以每次优化的目标函数其实是不一样的！也就是说在山里走，每次走一步，山的形状就变了。

  而在深度学习中，在山里走，山的形状是不变的。
]

在强化学习中很难找到合适的学习率。假设学习率是专门针对上图黄点调整的。该区域相对平坦，因此为了获得良好的学习速度，学习率应该高于平均值。但是，一步走错，我们就会从悬崖上掉到红点。红点处的梯度很高，当前的学习率会触发爆炸式策略更新。由于学习率对地形不敏感，策略梯度算法的收敛问题非常严重。

#figure(
  image("rl-figures/策略梯度法问题-1.png"),
  caption: [梯度爆炸更新],
)

#danger(title: [策略梯度法存在的问题是训练出来的新策略可能比旧策略还要差])[
  能否找到一个算法，使得训练出来的新策略#underline("一定")比旧策略好呢？也就是能否保证下面的式子成立呢？
  $
    J(theta_"new") - J(theta_"old") >= 0
  $
]

== 近端策略优化

在最优化理论中，优化方法主要有两种：*线搜索*和*置信域*。梯度下降算法是一种线搜索。我们首先确定下降方向，然后朝该方向迈出一步。

#tip[
  Trust Region Method: 置信域方法
]

#figure(
  image("rl-figures/线搜索和置信域比较.png"),
  caption: [左图为线搜索，右图为置信域],
)

而在置信域中，我们确定想要探索的最大步长，然后在该置信域内找到最优点。

#figure(
  image("rl-figures/置信域.png"),
  caption: [置信域方法],
)

也就是，我们在梯度上升的时候，先周围看一圈，然后找一个安全的步长，然后进行梯度上升。

为了实现这个目的，我们的目标函数$J(theta)$就要改变了。

#tip(title: [原始目标函数不好优化，转而去优化一个替代目标函数，这种方法很常见吗？])[
  是的，非常常见，而且几乎是现代机器学习、优化、统计推断和强化学习里的"基本套路"之一。很多时候，原始目标函数要么不可导、不可直接计算、方差太大、非凸、约束复杂，要么优化起来不稳定，所以我们会构造一个更容易优化的替代目标函数，也常叫surrogate objective/surrogate loss/proxy objective。
]



PPO算法是Actor-Critic架构的，也就是演员-评论家架构。

所以我们先来回顾一下策略梯度法的演员-评论家架构的梯度公式。

策略梯度法的目标函数如下

$
  J(theta) = EE_(tau tilde pi_theta) [G(tau)]
$

策略梯度法的目标函数的梯度如下

$
  nabla_theta J(theta) = EE_(tau tilde pi_theta) [sum_(t=0)^T A_t^(pi_theta) nabla_theta log pi_theta (a_t|s_t)]
$

其中$A_t^(pi_theta)$为估计的优势。这里的$A$表示优势，也就是Advantage。表示当前策略采取动作$a_t$之后，相对于价值函数评估的价值的优势是多少。

#danger[
  为了不和前面的公式混淆，我们将动作的数学符号由$A_t$改为$a_t$，将状态的数学符号由$S_t$改为$s_t$。这样$A$就代表优势了。
]

而PPO的目标函数变成了下式，也就是*替代目标函数*（surrogate objection function）。

#theorem(name: [带裁剪的PPO目标函数])[
  #math.equation(
    $
      \
      \
      \
      \
      \
      \
      \
      J(theta)^"ppo-clip" = EE_(markhl(tau tilde pi_(theta_"old"), tag: #<tau>, color: #blue)) [sum_(t=0)^T [min ( markhl((pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t)), tag: #<ratio1>) markhl(A_t^(pi_(theta_"old")), tag: #<piold1>, color: #green), markhl("clip", tag: #<clip>, color: #gray)( markhl((pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t)), tag: #<ratio2>), 1-epsilon, 1+epsilon) markhl(A_t^(pi_(theta_"old")), tag: #<piold2>, color: #green))]]
      \
      \
      \
      \
      \
      \
      \
      #annot((<ratio1>, <ratio2>), pos: top, dy: -1.5em, leader-connect: [elbow])[针对同一个状态$s_t$，\ 新策略采取动作$a_t$的概率 \ 和旧策略采取动作$a_t$的概率的比值]
      #annot((<piold1>, <piold2>), dx: -1.5em, dy: 1.5em, leader-connect: [elbow])[旧策略 \ 采取动作$a_t$ \ 的优势估计]
      #annot((<clip>), dx: 4.5em, dy: 1.5em, leader-connect: [elbow])[将比值裁剪到范围$(1-epsilon,1+epsilon)$]
      #annot((<tau>), pos: bottom + left, dx: -1.5em, dy: 2.5em, leader-connect: [elbow])[旧策略产生的轨迹]
    $,
    number-align: bottom,
    block: true,
  )
]

只要计算出这个目标函数，然后进行反向传播（`.backward()`也就是求$nabla_theta J(theta)^"ppo-clip"$）就可以更新策略神经网络了。

当我们运行旧策略$pi_(theta_"old")$玩一局倒立摆游戏时，会产生一条轨迹$tau=(s_0,a_0,R_0,s_1,a_1,R_1,dots.c)$。将这条轨迹保存下来，我们可以使用新策略$pi_theta$来计算在每个状态$s_t$，新策略采取动作$a_t$的概率$pi_theta (a_t|s_t)$是多少。

例如：当我们使用旧策略玩了一把倒立摆游戏之后，保存的轨迹中

$
  pi_(theta_"old") (a_1|s_1)=["向左推的概率"=0.3, "向右推的概率"=0.7]
$

那么新策略也可以针对$s_1$计算一下采取动作的概率分布。

#danger[
  在训练最开始时，新策略和旧策略相同，所以比值为$(pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t)) =1$，也就是这个比值不会被裁剪。

  所以在训练最开始时，也就是PPO的目标函数第一次进行反向传播时，有如下成立：
  $
    J(theta)^"ppo-clip" = EE_(tau tilde pi_(theta_"old")) [sum_(t=0)^T (pi_theta (a_t|s_t)|_(theta=theta_"old"))/(pi_(theta_"old") (a_t|s_t)) A_t^(pi_(theta_"old"))]
  $
  对两边进行求导，注意这里求导时$theta=theta_"old"$。所以有如下：
  $
    nabla_theta J(theta)^"ppo-clip" &= nabla_theta EE_(tau tilde pi_(theta_"old")) [sum_(t=0)^T (pi_theta (a_t|s_t)|_(theta=theta_"old"))/(pi_(theta_"old") (a_t|s_t)) A_t^(pi_(theta_"old"))] \
    &= EE_(tau tilde pi_(theta_"old")) [sum_(t=0)^T (nabla_theta pi_theta (a_t|s_t)|_(theta=theta_"old"))/(pi_(theta_"old") (a_t|s_t)) A_t^(pi_(theta_"old"))] colblue("（log梯度技巧）") \
    &= EE_(tau tilde pi_(theta_"old")) [sum_(t=0)^T A_t^(pi_(theta_"old")) nabla_theta log pi_theta (a_t|s_t) ]
  $
  我们会发现，PPO的第一次反向传播，等价于原始的策略梯度法#emoji.swimming。
]

如果新策略与旧策略的概率比超出$(1-epsilon)$和$(1+epsilon)$的范围，则优势将被剪裁。在PPO论文中的实验中，$epsilon$设置为$0.2$。

#figure(
  image("rl-figures/PPO目标函数中的比值.svg"),
  caption: [PPO目标函数中的比值],
)

实际上，如果策略发生重大变化超出了我们的舒适区，那么这将阻碍策略的实现。

我们假设比值（重要性权重）

$
  rho_t (theta) = (pi_theta (a_t|s_t)) / (pi_(theta_"old") (a_t|s_t))
$

#figure(
  table(
    columns: 7,
    table.cell(
      fill: red.lighten(60%),
    )[],
    table.cell(
      fill: red.lighten(60%),
    )[$p_t (theta) > 0$],
    table.cell(
      fill: red.lighten(60%),
    )[$A_t$],
    table.cell(
      fill: red.lighten(60%),
    )[min函数的结果],
    table.cell(
      fill: red.lighten(60%),
    )[目标函数是否被裁剪？],
    table.cell(
      fill: red.lighten(60%),
    )[目标函数的符号],
    table.cell(
      fill: red.lighten(60%),
    )[梯度],
    [1], [$rho_t (theta) in [1-epsilon,1+epsilon]$], [+], [$rho_t (theta) A_t$], [否], [+], [$checkmark$],
    [2], [$rho_t (theta) in [1-epsilon,1+epsilon]$], [-], [$rho_t (theta) A_t$], [否], [-], [$checkmark$],
    [3], [$rho_t (theta) < 1-epsilon$], [+], [$rho_t (theta) A_t$], [否], [+], [$checkmark$],
    [4], [$rho_t (theta) < 1-epsilon$], [-], [$(1-epsilon)A_t$], [是], [-], [$0$],
    [5], [$rho_t (theta) > 1+epsilon$], [+], [$(1+epsilon)A_t$], [是], [+], [$0$],
    [6], [$rho_t (theta) > 1+epsilon$], [-], [$rho_t (theta) A_t$], [否], [-], [$checkmark$],
  ),
  caption: [对PPO目标函数根据比值的不同进行分情况讨论],
)


我们有6种不同的情况，见上面的表。首先记住，我们取裁剪目标和非裁剪目标中的最小值。

- *情况1和2：比值介于裁剪范围之内*

在情况1和2中，由于比例介于范围$[1-epsilon,1+epsilon]$之间，因此目标函数不会被裁剪。

在情况1中，我们具有正优势：该动作优于该状态下所有动作的平均值。因此，我们应该鼓励当前策略提高在该状态下采取该动作的概率。

由于该比值是在区间之间的，因此我们可以增加我们的策略在该状态下采取该行动的概率。

在情况2中，我们有一个负优势：该动作比该状态下所有动作的平均值更差。因此，我们应该阻止当前策略在该状态下采取该动作。

由于该比值是在区间之间的，因此我们可以降低我们的策略在该状态下采取该行动的概率。

- *情况3和4：比值小于$1-epsilon$*

如果概率比低于$1-epsilon$，则当前策略在该状态下采取该行动的概率比旧策略低得多。

如果像情况3中一样，优势估计为正（$A>0$），那么我们希望增加在该状态下采取该行动的概率。

但是，如果像情况4那样，优势估计为负，我们不想进一步降低在该状态下采取该行动的概率。因此，$"梯度"=0$（因为我们在一条平线上），所以我们不会更新权重。

- *情况5和6：比值大于$1+epsilon$*

如果概率比高于$1+epsilon$，则当前策略中该状态下采取该行动的概率远高于前一策略。

如果像情况5那样，优势为正，我们不应该太贪心。因为在当前状态下，我们采取该行动的概率已经比之前的策略更高了。因此，梯度等于0（因为我们在一条平线上），所以我们不会更新权重。

如果像情况6中那样，优势是负的，我们希望降低在该状态下采取该行动的概率。

总结一下，我们只用未裁剪的目标函数部分来更新策略。当最小值是裁剪后的目标函数部分时，我们不会更新策略权重，因为梯度将等于0。

因此，我们仅在以下情况下更新我们的策略：

- 比值在 $(1-epsilon,1+epsilon)$ 之内
- 比值不在 $(1-epsilon,1+epsilon)$ 范围内，但优势使我们更接近这个范围。
  - 比值小于 $1-epsilon$ ，但优势 > 0 。
  - 比值大于 $1+epsilon$ ，但优势 < 0 。

你可能会想，为什么当最小值是截断比值时，梯度为0。当比值被截断时，在这种情况下的导数将不是 $rho_t (theta) A_t$ 的导数。而是 $(1-epsilon)A_t$ 或者 $(1+epsilon)A_t$ 的导数，而两者的导数都是0。

#tip[
  $A_t$不是$theta$的函数。所以对$theta$求导为0。
]

总而言之，得益于这个裁剪的替代目标函数（替换了原来的简单优美的目标函数（$J(theta)=EE_(tau tilde pi_theta) [G(tau)]$），我们限制了当前策略与旧策略之间的差异范围。因为我们消除了比值超出区间的诱因，因为裁剪会对梯度产生影响。如果比值为$>1+epsilon$或$<1-epsilon$，则梯度将等于$0$。

PPO增加了一个软约束（裁剪机制），可以通过一阶优化器（求一阶导数）进行优化。我们偶尔可能会做出一些错误的决策，但它在优化速度上取得了良好的平衡。实验结果证明，这种平衡能够以最简单的方式实现最佳性能。

#figure(
  algo(
    line-numbers: true,
    strong-keywords: false,
    comment-prefix: [#sym.triangle.stroked.r ],
    comment-styles: (fill: rgb(100%, 0%, 0%)),
    header: [#box(stroke: 0.1em, inset: 0.3em)[带裁剪的PPO算法]],
  )[
    输入：初始化策略参数$theta_0$ \
    初始化裁剪阈值参数$epsilon=0.2$ \
    *for* $k=0,1,2,dots.c$ *do* #i \
    使用策略$pi_(theta_k)$采集一条轨迹数据$cal(D)_k$ \
    计算优势$A_t^(pi_(theta_k))$ #comment[使用广义优势估计（GAE）] \
    更新策略 #comment[执行N步的梯度下降] \
    $theta'_1=theta_k$ #comment[将$theta_k$作为梯度更新的起点]\
    *for* $i=1,2,dots.c,N$ *do* #comment[更新N次梯度] #i\
    $rho=(pi_(theta'_i) (a_t|s_t))/(pi_(theta_k) (a_t|s_t))$ #comment[计算比值]\
    $"clipped"="clip"(rho,1-epsilon,1+epsilon)$ #comment[裁剪比值]\
    $J(theta'_i)=sum_(t=0)^T [min (rho dot.c A_t^(pi_(theta_k)), "clipped" dot.c A_t^(pi_(theta_k)))]$ #comment[累加所有时间步的最小值]\
    $theta'_(i+1)=theta'_i+alpha nabla_(theta'_i) J(theta'_i)$ #d #comment[更新梯度]\
    $theta_(k+1)=theta'_N$ #comment[将$theta'_N$作为下一次更新的起点]\
  ],
  caption: [带裁剪的PPO算法伪代码],
)

#chapter("PPO实战", image: image("./orange2.jpg"), l: "rl-ppo-cart-pole")

== 网络结构的定义

- 演员网络：策略网络
- 评论家网络：价值函数网络

#codly(header: [策略网络和价值函数网络的定义])
```python
class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
```

== PPO算法的实现

我们来实现Agent，代码如下：

```python
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.001
        self.lr_v = 0.02
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs

    def collect_trajectory(self, env):
        """采集一条轨迹"""
        state = env.reset()
        states, next_states, actions, action_probs, rewards, dones = [], [], [], [], [], []
        done = False

        while not done:
            action, probs = self.get_action(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state) # $s_t$
            actions.append(action) # $a_t$
            action_probs.append(probs[action]) # $pi(a_t|s_t)$
            next_states.append(next_state) # $s_(t+1)$
            rewards.append(reward) # $R_t$
            dones.append(done) # $"done"_t$
            state = next_state # 状态转移

        return states, next_states, actions, action_probs, rewards, dones
```

接下来我们要实现智能体的更新（`update`）代码，在PPO算法中，我们需要用到旧策略采样的轨迹。所以需要保存旧策略采样的轨迹的完整信息。

我们采样的轨迹是：

$
  tau = [ s_0, a_0, R_0, s_1, a_1, R_1, dots.c, s_T, a_T, R_T ]
$

那么我们使用元组来保存一条轨迹的信息。结构如下

$
  "一条轨迹的详细信息" = cases(
    "states": [s_0,s_1,dots,s_(T-1)],
    "next_states": [s_1,s_2,dots,s_T],
    "actions": [a_0,a_1,dots,a_(T-1)],
    "action_probs": [pi_(theta_"old") (a_0|s_0),pi_(theta_"old") (a_1|s_1),dots,pi_(theta_"old") (a_(T-1)|s_(T-1))],
    "rewards": [R_0,R_1,dots,R_(T-1)],
    "dones": ["False","False",dots,"True"]
  )
$

有了旧策略采样的轨迹的所有详细信息，我们就可以来实现PPO算法了。代码如下。

```python
class Agent:
    ...

    def update(self, trajectory):
        states, next_states, actions, action_probs, rewards, dones = trajectory
        # $[s_0,s_1,dots,s_(T-1)]$
        states = torch.tensor(states)
        # $[a_0,a_1,dots,a_(T-1)]$
        actions = torch.tensor(actions).view(-1, 1)
        # $[R_0,R_1,dots,R_(T-1)]$
        rewards = torch.tensor(rewards).view(-1, 1)
        # $[s_1,s_2,dots,s_T]$
        next_states = torch.tensor(next_states)
        # $["False"_1, "False"_2, dots, "True"_T]$
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1)
        # $[V(s_0),V(s_1),dots,V(s_(T-1))]$
        V = self.v(states).detach()
        # $"TD-target"_t=R_t+gamma V(s_(t+1))$
        # $["TD-target"_0, "TD-target"_1, dots, "TD-target"_(T-1)]$
        td_target = rewards + self.gamma * self.v(next_states) * (1 - dones)
        # 一步TD误差：$delta_t=R_t+gamma V(s_(t+1))-V(s_t)$
        # $[delta_0, delta_1, dots, delta_(T-1)]$
        td_delta = td_target - V
        # 计算每个时刻t的广义优势估计（GAE）
        # $[A^"GAE"_0,A^"GAE"_1,dots,A^"GAE"_(T-1)]$
        gae = compute_gae(self.gamma, td_delta.cpu())
        # 冻结一份旧策略采取动作的对数概率$log pi_(theta_"old") (a_t|s_t)$
        # $[log pi_(theta_"old") (a_0|s_0),log pi_(theta_"old") (a_1|s_1),dots,log pi_(theta_"old") (a_(T-1)|s_(T-1))]$
        old_probs = torch.tensor(action_probs).view(-1, 1)
        old_log_probs = torch.log(old_probs).detach()
        # 每条轨迹使用10次，也就是更新10次策略网络
        for _ in range(10):
            # 新策略采取动作的对数概率：$log pi_theta (a_t|s_t)$
            # $[log pi_theta (a_0|s_0),log pi_theta (a_1|s_1),dots,log pi_theta (a_(T-1)|s_(T-1))]$
            log_probs = torch.log(self.pi(states).gather(1, actions))
            # 计算比值：$p_t (theta)=(pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t))$
            ratio = torch.exp(log_probs - old_log_probs)
            # $p_t (theta)A^(pi_(theta_"old"))_t$
            # $[p_0 A^(pi_(theta_"old"))_0,p_1 A^(pi_(theta_"old"))_1,dots,p_(T-1) A^(pi_(theta_"old"))_(T-1)]$
            surr1 = ratio * gae
            # $"clip"(p_t (theta),1-epsilon,1+epsilon)A^(pi_(theta_"old"))_t|_(epsilon=0.2)$
            # $["clip"(p_0)A^"GAE"_0,"clip"(p_1)A^"GAE"_1,dots,"clip"(p_(T-1))A^"GAE"_(T-1)]$
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * gae
            # $pi_"loss"=-1/T sum_(t=0)^(T-1) min("surr1"_t,"surr2"_t)$
            pi_loss = torch.mean(-torch.min(surr1, surr2))
            # $V_"loss"=1/T sum_(t=0)^(T-1)(V(s_t)-(A_t^"GAE"+V(s_t)))$
            v_loss = torch.mean(F.mse_loss(self.v(states), gae + V))
            self.optimizer_pi.zero_grad()
            self.optimizer_v.zero_grad()
            pi_loss.backward()
            v_loss.backward()
            self.optimizer_pi.step()
            self.optimizer_v.step()
```

#tip(title: [比值的计算方法])[
  $
    & because A=e^(log A) \
    & therefore A/B = e^(log A - log B) = (e^(log A))/(e^(log B))
  $
]

== 广义优势估计的计算

```python
def compute_gae(gamma, td_delta):
    # $delta_t$
    td_delta = td_delta.detach().numpy()
    gae_list = []
    last_gae = 0.0
    lmbda = 0.95
    # $A_t=delta_t+gamma lambda A_(t+1)$
    for delta in td_delta[::-1]:
        last_gae = gamma * lmbda * last_gae + delta
        gae_list.append(last_gae)
    gae_list.reverse()
    return torch.tensor(gae_list)
```

== 训练循环

```python
def train(env, agent):
    return_list = []
    episode_list = []
    for episode in range(500):
        trajectory = agent.collect_trajectory(env)
        agent.update(trajectory)
        # 统计信息
        episode_reward = sum(trajectory[4])
        return_list.append(episode_reward)
        episode_list.append(episode)

        if (episode + 1) % 10 == 0:
            print(f"回合：{episode}, 回报：{episode_reward}")

    return return_list, episode_list

def main():
    env = gym.make("CartPole-v0")
    env.seed(0)
    torch.manual_seed(0)
    agent = Agent()
    return_list, episode_list = train(env, agent)
    plot_loss(episode_list, return_list, "ppo-loss.pdf")
    test_agent(agent, env)

if __name__ == "__main__":
    main()
```

获得的奖励可视化如下：

#figure(
  image("rl-figures/ppo-loss.svg"),
  caption: [PPO算法获得的奖励],
)

#chapter("PPO背后的数学", image: image("./orange2.jpg"), l: "rl-ppo-math")

在使用策略梯度算法训练智能体时，一个挑战是它们容易出现性能崩溃：智能体会突然表现很差。这种情况很难恢复，因为智能体会开始生成质量较差的轨迹，而这些轨迹又被用于进一步训练策略。我们还看到，on-policy（同策略）算法的样本效率较低，因为它们无法重用数据。

Schulman等人提出的近端策略优化（Proximal Policy Optimization, PPO）是一类优化算法，用来解决这两个问题。PPO的核心思想是在目标函数中引入一个替代（surrogate）目标，它通过保证策略的单调改进来避免性能崩溃。该目标还有一个好处，就是可以在训练过程中重用离策略（off-policy）数据。

PPO可以通过将原始的目标函数$J(theta)$替换为修改后的PPO目标，来扩展REINFORCE或者Actor-Critic。此修改带来了更稳定、且样本效率更高的训练过程。

在本章中，我们首先讨论性能崩溃问题。随后通过单调改进理论来应对该问题。我们将该理论应用于策略梯度目标，将其改造为一个代理目标。

在介绍完理论基础后，我们讨论PPO算法。

== 数学背景知识

=== KL散度

KL散度（Kullback-Leibler Divergence, KLD）非常重要。

假设有一个连续型随机变量$x$，其概率密度表示为$p(x)$。此时，函数$f(x)$的期望值可以用下面的数学式表示。

$
  EE_(p(x)) [f(x)]=integral f(x)p(x) upright(d) x
$

关于概率分布$q(x)$的期望值可以用下面的数学式表示。

$
  EE_(q(x)) [f(x)]=integral f(x)q(x) upright(d) x
$

*KL散度的定义*

衡量两个概率分布之间差异的一种方法是KL散度。当给定两个概率分布$p(x)$和$q(x)$时，KL散度可以用下面的数学式表示。

#definition(name: [连续型随机变量的KL散度])[
  $
    D_"KL" (p parallel q) = integral p(x) log p(x)/q(x) upright(d) x
  $
]

上面的式子是当$x$为连续型随机变量时的KL散度。当$x$为离散型随机变量时，数学式如下所示。

#definition(name: [离散型随机变量的KL散度])[
  $
    D_"KL" (p parallel q) = sum_x p(x) log p(x)/q(x)
  $
]

KL散度具有以下特性：

- 两个概率分布的差异越大，KL散度的值就越大
- KL散度的值大于或等于0，且仅当两个概率分布相同时，其值才为0
- KL散度是非对称的衡量指标，因此$D_"KL" (p parallel q)$和$D_"KL" (q parallel p)$的值不同

这些特性使得KL散度可以用来衡量两个概率分布的差异程度。下面我们通过具体的例子来了解这些特性。这里以抛硬币为例进行说明。假设一枚硬币正面朝上和反面朝上的概率是确定的，如下表所示。

#figure(
  table(
    columns: 2,
    [正面朝上的概率], [70%],
    [反面朝上的概率], [30%],
  ),
  caption: [硬币的真实概率分布],
)

上表是硬币的"真实概率分布"，在这里我们用符号$p$来表示。假设有人对这枚硬币的概率分布做了如下估计。

#figure(
  table(
    columns: 2,
    [正面朝上的概率], [50%],
    [反面朝上的概率], [50%],
  ),
  caption: [第一个人对硬币的概率分布做出的估计],
)

我们用$q$来表示这个估计的概率分布。此时，"真实概率分布$p$"和"估计概率分布$q$"之间的KL散度可以做如下计算。

$
  D_"KL" (p parallel q) & = 0.7 log 0.7/0.5 + 0.3 log 0.3/0.5 \
                        & = 0.082 dots.c
$

KL散度约为0.082。假设另一个人对这枚硬币的概率分布做了如下估计。

#figure(
  table(
    columns: 2,
    [正面朝上的概率], [20%],
    [反面朝上的概率], [80%],
  ),
  caption: [第二个人对硬币的概率分布做出的估计],
)

这与真实的概率分布值大相径庭。此时的KL散度的值如下所示。

$
  D_"KL" (p parallel q) & = 0.7 log 0.7/0.2 + 0.3 log 0.3/0.8 \
                        & = 0.58 dots.c
$

这个值比第一个KL散度的值要大。最后，假设又有一个人对这枚硬币的概率分布做了如下估计。

#figure(
  table(
    columns: 2,
    [正面朝上的概率], [70%],
    [反面朝上的概率], [30%],
  ),
  caption: [第三个人对硬币的概率分布做出的估计],
)

这个分布与真实的概率分布相同。此时的KL散度的值如下所示。

$
  D_"KL" & = 0.7 log 0.7/0.7 + 0.3 log 0.3/0.3 \
         & = 0.7 log 1 + 0.3 log 1 space space space colblue((log 1 = 0)) \
         & = 0
$

由于$p$和$q$是相同的概率分布，因此KL散度为0。

从以上结果可以看出，KL散度可以用来衡量两个概率分布的差异程度。当两个概率分布相同时，KL散度取最小值0。两个概率分布的差异越大，KL散度的值就越大。

=== 信息论

信息量（Self-Information）：

- 一个事件发生的信息量定义为：$I(x) = -log P(x)$
- 概率越小的事件，包含的信息量越大
- 例如："太阳从东边升起"（高概率）vs "中彩票"（低概率）

熵（Entropy）：

- 衡量随机变量的不确定性：$H(P) = -sum_x P(x) log P(x)$
- 熵越大，不确定性越大

交叉熵（Cross-Entropy）：

- 衡量两个概率分布之间的差异：$H(P, Q) = -sum_x P(x) log Q(x)$
- 其中 $P$ 是真实分布，$Q$ 是预测分布

极大似然估计和交叉熵

假设我们有训练数据 ${(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}$，其中 $y_i$ 是真实标签。

似然函数：

$
  L(theta) = product_(i=1)^n P(y_i | x_i; theta)
$

对数似然：

$
  log L(theta) = sum_(i=1)^n log P(y_i | x_i; theta)
$

最大化对数似然 = 最小化负对数似然：

$
  "Loss" = -1/n sum_(i=1)^n log P(y_i | x_i; theta)
$

这就是*交叉熵损失*！

KL散度和交叉熵

KL散度衡量两个分布的差异：

$
  D_"KL" (P parallel Q) & = sum_x P(x) log (P(x))/(Q(x)) \
                        & = sum_x P(x) log P(x) - sum_x P(x) log Q(x)
$

$
  D_"KL" (P parallel Q) = -H(P) + H(P,Q)
$

$
  (partial D_"KL" (P parallel Q))/(partial theta) = (partial H(P, Q))/(partial theta)
$

其中：

- $H(P) = -sum_x P(x) log P(x)$ 是真实分布的熵（常数）
- $H(P,Q) = -sum_x P(x) log Q(x)$ 是交叉熵

最小化KL散度 $arrow.l.r.long.double$ 最小化交叉熵（因为真实分布的熵是常数）

=== 重要性采样（Importance Sampling）

==== 什么是重要性采样？

重要性采样利用从其他概率分布中采样的数据来计算某个概率分布的期望值。为了说明重要性采样，下面来看一个计算期望值$EE_pi [x]$的简单例子。设$x$是随机变量，$x$的概率用$pi(x)$表示。期望值的式子如下所示。

$
  EE_pi [x] = sum x pi(x)
$

我们先复习一下学过的内容。为了用蒙特卡洛方法近似这个期望值，我们从概率分布$pi$对$x$进行采样，并取其平均值。数学式如下所示：

$
  "采样": x^((i)) tilde pi space space space (i=1,2,dots.c,n) \
  EE_pi [x] tilde.eq (x^((1)) + x^((2)) + dots.c + x^((n)))/n
$

式子中的符号$x^((i)) tilde pi$表示第$i$个数据$x(i)$是从概率分布$pi$中采样的。

接下来回到正题。我们现在探讨的是从别的概率分布中采样$x$的场景。假设$x$是从概率分布$b$（而不是$pi$）中采样的。在这种情况下，期望值$EE_pi [x]$如何被近似呢？解决这个问题的关键在于下面这个式子的变形。

$
  EE_pi [x] & = sum x pi(x) \
            & = sum x b(x)/b(x) pi(x) \
            & = sum x pi(x)/b(x) b(x)
$

这里的要点是插入$b(x)/b(x)$。$b(x)/b(x)$总是为$1$，所以等式成立。然后，我们将式子变为$sum dots.c b(x)$的形式，如上面的式子所示，这样它就可以被视为概率分布$b(x)$的期望值了。实际上对上面的式子变形，可以得到以下式子。

$
  EE_pi [x] & = sum x pi(x)/b(x) b(x) \
            & = EE_b [x pi(x)/b(x)]
$

这里需要注意的是式子中的$EE_b$，它表示的是概率分布$b$的期望值。另外每个$x$都要乘以$pi(x)/b(x)$这一点也很重要。如果$rho(x)=pi(x)/b(x)$，那么我们就可以把式子看作每个$x$都乘以权重$rho(x)$。根据以上探讨，基于式的蒙特卡洛方法如下所示。

$
  "采样": x^((i)) tilde b space space space (i=0,1,dots.c,n) \
  EE_pi [x] tilde.eq (rho(x^((1)))x^((1)) + rho(x^((2)))x^((2)) + dots.c + rho(x^((n)))x^((n)))/n
$

这样我们就使用从不同于$pi$的概率分布$b$中采样的数据计算出了$EE_pi [x]$。下面来实现重要性采样的代码 。这里我们对下图所示的概率分布进行重要性采样。

#figure(
  image("rl-figures/重要性采样图1.svg"),
  caption: [概率分布$pi$和$b$],
)

这里的目标是求出期望值$EE_pi [x]$。我们先尝试用普通的蒙特卡洛方法求概率分布$pi$的期望值。代码如下所示。

#codly(header: [用普通的蒙特卡洛方法求概率分布$pi$的期望值])
```python
import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

# 期望值$EE_pi [x]$
e = np.sum(x * pi)
print("E_pi[x]", e) # 输出：E_pi[x] 2.7

# 蒙特卡洛方法
n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi) # 使用pi进行采样
    samples.append(s)

mean = np.mean(samples)
var = np.var(samples)
print("MC: {:.2f} (var: {:.2f})".format(mean, var)) # 输出：MC: 2.78 (var: 0.27)
```

首先，套用定义式求期望值。求出的结果是$2.7$（这是真实的值）。然后，使用蒙特卡洛方法来求这个值。这里只基于概率分布`pi`采样100个数据，然后求平均值。为此，我们用NumPy的`np.mean`方法求出了平均值。结果是$2.78$，接近于真实值。作为参考，我们还使用NumPy的`np.var`方法求出了方差。方差的值是$0.27$。在与下一次重要性采样的结果进行比较时，这个值将被用作参考。方差用于表示数据的离散程度。期望值和方差之间的关系可以用以下式子表示。

$
  "Var"[X] = EE[(X-EE[X])^2]
$

方差是数据$X$和$X$的平均值$EE[X]$之间的差的平方的期望值。直观地说，它代表了数据的离散程度，如下图所示。

#figure(
  image("rl-figures/重要性采样图2.svg"),
  caption: [将每个数据作为二维平面上的点时的方差的示意图（圆心是平均值）],
)

下面尝试用重要性采样的方法来求期望值。代码如下所示。

#codly(header: [用重要性采样的方法来求期望值])
```python
b = np.array([1/3, 1/3, 1/3])
n = 100
samples = []

for _ in range(n):
    idx = np.arange(len(b)) # [0, 1, 2]
    i = np.random.choice(idx, p=b) # 使用 b 进行采样
    s = x[i]
    rho = pi[i] / b[i] # $rho$
    samples.append(rho * s)

mean = np.mean(samples)
var = np.var(samples)
print("IS: {:.2f} (var: {:.2f})".format(mean, var)) # 输出：IS: 2.95 (var: 10.63)
```

这里使用概率分布`b`进行采样，不过进行采样的目标是`b`的索引`([0, 1, 2])`。这是因为在计算权重`rho`时，要用到的是采样得到的索引。

接下来看看上面的结果。平均值是$2.95$，虽然与真实值$2.7$有一些差距，但也还算接近。此外，方差是$10.63$，这说明数据的离散程度大于蒙特卡洛方法的结果（使用蒙特卡洛方法时的方差是$0.27$）。

==== 如何减小方差

方差越小，用较少的样本得到的近似值就越准确。反之，方差越大，越需要更多的样本才能得到准确的近似值。下面介绍如何用重要性采样来减小方差。首先，我们结合下图来了解一下为什么重要性采样方法的方差会比较大。

#figure(
  image("rl-figures/重要性采样图3.svg"),
  caption: [基于概率分布$b$采样$3$的例子],
)

上图展示了选择$3$作为样本数据的例子。此时的权重$rho$为$2.4$。因此，$3$这个值要乘以$2.4$。这意味着尽管我们得到的值是$3$，但实际得到的值是$3 times 2.4 = 7.2$。听起来可能有些奇怪，但这是有道理的。原因如下。

- 由于$3$是概率分布$pi$的代表值，因此本来它应该被更频繁地采样。
- 但在概率分布$b$中，$3$这个值并不经常出现。
- 为了填补这个落差，当$3$被采样时，它会被调整为乘以权重，从而使该值变得更大。

考虑到概率分布$pi$和$b$之间的差异，我们将采样值乘以其权重来调整采样值，这么做是合理的。可是，明明采样的值是$3$，它却被当作了$7.2$，如果这是第一个采样数据，那么此时的估计值就是$7.2$。与真正的值$2.7$相比，$7.2$的偏离较大。因此，通过权重$rho$对实际得到的值的填补越大，方差与真正的值的差异就越大。

那么，怎样才能减小方差呢？一种方法是使两个概率分布（$b$和$pi$）更加接近。这样可以使权重$rho$的值更接近于$1$。下面来做个实验。与之前的代码相比，这次只有概率分布中的$b$的值被改变了。

#codly(header: [减小方差的方法])
```python
b = np.array([0.2, 0.2, 0.6])
n = 100
samples = []

for _ in range(n):
    idx = np.arange(len(b)) # [0, 1, 2]
    i = np.random.choice(idx, p=b) # 使用 b 进行采样
    s = x[i]
    rho = pi[i] / b[i] # $rho$
    samples.append(rho * s)

mean = np.mean(samples)
var = np.var(samples)
print("IS: {:.2f} (var: {:.2f})".format(mean, var)) # 输出：IS: 2.72 (var: 2.48)
```

上面的代码假设$b$的概率分布为`[0.2, 0.2, 0.6]`，其形状更接近于`pi`的概率分布。结果是，平均值为$2.72$，更接近于正确答案。另外，方差是$2.48$，比以前小。

因此，在进行重要性采样时，可以通过使两个概率分布更加接近来减小方差。不过，强化学习的重点是让一个策略（概率分布）进行"探索"，另一个策略进行"利用"。在满足这个条件的基础上，就可以通过使两个概率分布尽可能地接近来减小方差了。

以上就是对重要性采样的介绍。

== 替代目标

直观上，原始策略梯度法的问题出在步长上，因此可以通过加入一个约束，将步长限制在安全范围以防止性能崩塌。

我们的目的是：

$
  J(theta) - J(theta_"old") >= 0
$

所以首先要将差值计算出来。差值计算出的结果如下：

$
  J(theta) - J(theta_"old") = EE_(tau tilde pi_theta) [sum_(t=0)^T gamma^t A_t^(pi_(theta_"old"))]
$

上式的证明如下。我们从反方向证明。

$
  & EE_(tau tilde pi_theta) [sum_(t=0)^T gamma^t A_t^(pi_(theta_"old"))] && (1) \
  & = EE_(tau tilde pi_theta) [sum_(t=0)^T gamma^t (R_t + gamma V(S_(t+1)) - V(S_t))] && (2) \
  & = EE_(tau tilde pi_theta) [sum_(t=0)^T gamma^t R_t + sum_(t=0)^T gamma^(t+1) V(S_(t+1)) - sum_(t=0)^T gamma^t V(S_t)] && (3) \
  & = EE_(tau tilde pi_theta) [sum_(t=0)^T gamma^t R_t] + EE_(tau tilde pi_theta) [sum_(t=0)^T gamma^(t+1) V(S_(t+1)) - sum_(t=0)^T gamma^t V(S_t)] space space && (4) \
  & = J(theta) + EE_(tau tilde pi_theta) [sum_(colred(t=1))^T gamma^colred(t) V(S_colred(t)) - sum_(t=0)^T gamma^t V(S_t)] && (5) \
  & = J(theta) - EE_(tau tilde pi_theta) [ V(S_0) ] && (6) \
  & = J(theta) - EE_(tau tilde pi_theta) [ J(theta_"old") ] && (7) \
  & = J(theta) - J(theta_"old") && (8)
$

第(6)步到第(7)步依赖了状态价值函数的定义：$V(S_0)=EE_(tau tilde pi_(theta_"old")) [G_0]=EE_(tau tilde pi_(theta_"old")) [G(tau)]=J(theta_"old")$。

第(7)步到第(8)步依赖了期望的性质，也就是由于策略$pi_theta$和$pi_(theta_"old")$相互独立，所以$EE_(tau tilde pi_theta) [J(theta_"old")]=J(theta_"old")$。

$J(theta)-J(theta_"old")$可作为度量策略改进的指标。若该差值为正，则新策略 $pi_theta$ 比旧策略 $pi_(theta_"old")$ 更优。在一次策略迭代过程中，理想情况下应选择使这一差值最大化的新策略$pi_theta$。因此，最大化目标 $J(theta)$ 等价于最大化该差值，两者均可通过梯度上升实现。

$
  max_(theta) J(theta) arrow.l.r.double.long max_(theta) {J(theta) - J(theta_"old")}
$

以这种方式刻画目标也意味着，每次策略迭代都应保证非负（单调）的改进——即$J(theta) - J(theta_"old")>=0$——因为在最坏情况下我们可以简单地令$pi_theta=pi_(theta_"old")$，也就是不改进策略。在此条件下，整个训练过程中将不会发生性能崩塌，这正是我们所期望的性质。

然而，这一差值作为目标函数有一个限制使其无法直接使用。注意，在表达式$EE_(tau tilde pi_theta) [sum_(t=0)^T gamma^t A_t^(pi_(theta_"old"))]$中，期望要求使用新策略$pi_theta$采样轨迹以进行更新，但在完成更新之前，新策略$pi_theta$并不可用。为了解决这一悖论，我们需要设法将其改写为使用已可用的旧策略$pi_(theta_"old")$。

#tip(title: [重要性采样复习])[
  $
    EE_(x tilde P) [f(x)] = EE_(x tilde Q) [P(x)/Q(x) f(x)] approx 1/(|D|) sum_(x in D) P(x)/Q(x) f(x), space space space D tilde Q
  $
  比值$P(x)/Q(x)$是$x$的*重要性采样权重*。
  重要性采样估计器的方差是什么呢？
  $
    "var"(hat(mu)_Q) & = 1/N "var"(P(x)/Q(x) f(x)) \
                     & = 1/N (EE_(x tilde Q) [(P(x)/Q(x) f(x))^2] - EE_(x tilde Q) [P(x)/Q(x) f(x)]^2) \
                     & = 1/N (EE_(x tilde P) [Q(x)/P(x) (P(x)/Q(x) f(x))^2] - EE_(x tilde Q) [P(x)/Q(x) f(x)]^2) \
                     & = 1/N (colred(EE_(x tilde P) [P(x)/Q(x) f(x)^2]) - EE_(x tilde P) [f(x)]^2)
  $
  红色部分很有问题！——如果$P(x)/Q(x)$在错误的地方很大，那么估计的方差直接炸了。
  #tip(title: [方差的计算])[
    $
      "var"(X) = EE[X^2] - EE[X]^2
    $
  ]
]

这里就要用到重要性采样了。我们使用的重要性采样权重为$(pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t))$。也就是我们使用新旧策略采取动作的概率的比值来改进旧策略$pi_(theta_"old")$产生的轨迹的回报值。在新策略$pi_theta$下更可能发生的动作所对应的回报会被上调权重，而在$pi_theta$下相对不太可能的动作所对应的回报会被下调权重。该近似在下面的式子中给出。

$
  J(theta) - J(theta_"old") & = EE_(tau tilde colred(pi_theta)) [sum_(t=0)^T A_t^(pi_(theta_"old"))] space space space colblue("（为简单起见，将"gamma"设置为1）") \
  & approx EE_(tau tilde colred(pi_(theta_"old"))) [sum_(t=0)^T A_t^(pi_(theta_"old")) colred((pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t)))] \
  & = J^"CPI"_(pi_(theta_"old")) (theta)
$ <cpiformula>

上面的式子也就是$J^"CPI"_(pi_(theta_"old")) (theta)$叫做*替代目标*（surrogate objective）。因为新目标中包含了新旧策略的比值，所以叫做"替代目标"。上标CPI的意思是"保守策略迭代"（conservative policy iteration）。

#danger(title: [为什么是约等于])[
  把真实性能差分展开，由期望的定义
  $
    J(theta) - J(theta_"old") &= sum_(t=0)^T EE_(s_t tilde d_(pi_(theta))^t, a_t tilde pi_theta (dot.c|s_t)) [A_t^(pi_(theta_"old")) ] \
    &= sum_(t=0)^T EE_(s_t tilde d_(pi_(theta))^t, a_t tilde pi_(theta_"old") (dot.c|s_t)) [A_t^(pi_(theta_"old")) (pi_theta (a_t|s_t)) / (pi_(theta_"old") (a_t|s_t))] \
    & approx sum_(t=0)^T EE_(s_t tilde d_(pi_(theta_"old"))^t, a_t tilde pi_(theta_"old") (dot.c|s_t)) [A_t^(pi_(theta_"old")) (pi_theta (a_t|s_t)) / (pi_(theta_"old") (a_t|s_t))] \
    & = EE_(tau tilde pi_(theta_"old"))[sum_(t=0)^T A_t^(pi_(theta_"old")) dot.c ( pi_theta (a_t|s_t) )/( pi_(theta_"old") (a_t|s_t) )]
  $
  CPI的近似假设是：新旧策略足够接近，因此它们诱导的状态访问分布也接近：
  $
    d_(pi_theta)^t (s) approx d_(pi_(theta_"old"))^t (s)
  $
]

现在我们有了一个新的目标函数。要将其用于策略梯度算法，需要检查在该目标下进行优化是否仍然是在执行策略梯度上升。幸运的是，我们可以证明替代目标的梯度等于策略梯度，如下式所述。

$
  nabla_theta J^"CPI"_(pi_(theta_"old")) (theta) |_(theta=theta_"old") = nabla_theta J(theta)|_(theta=theta_"old") = nabla_theta {J(theta)-J(theta_"old")}|_(theta=theta_"old")
$

上面的式子证明如下。也就是我们要证明代理目标的梯度等于策略梯度。

$
  nabla_theta J^"CPI"_(pi_(theta_"old")) (theta)|_(theta=theta_"old") &= nabla_theta EE_(tau tilde pi_(theta_"old")) [sum_(t=0)^T A^(pi_(theta_"old"))_t (pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t))]|_(theta=theta_"old") \
  &=EE_(tau tilde pi_(theta_"old")) [sum_(t=0)^T A^(pi_(theta_"old"))_t (nabla_theta pi_theta (a_t|s_t)|_(theta=theta_"old"))/(pi_(theta_"old") (a_t|s_t))] space space space colblue("log梯度技巧")\
  &=EE_(tau tilde pi_(theta_"old")) [sum_(t=0)^T A^(pi_(theta_"old"))_t nabla_theta log pi_theta (a_t|s_t)|_(theta=theta_"old")] \
  &= nabla_theta J(theta)|_(theta=theta_"old") \
  &= nabla_theta {J(theta)-J(theta_"old")}|_(theta=theta_"old")
$

上式表明，代理目标的梯度等于策略梯度。这保证了在代理目标下进行的优化仍然是在执行策略梯度上升。这也很有用，因为现在可以直接度量策略改进，最大化该度量就意味着最大化策略改进。此外，我们还知道，在上式中，$J^"CPI"_(pi_(theta_"old")) (theta)$是对$J(theta)-J(theta_"old")$的线性近似，因为她们的一阶导数（梯度）相等。

由于我们推导出了$J^"CPI"_(pi_(theta_"old")) (theta) approx J(theta)-J(theta_"old")$，注意这里是约等于。也就是

$
  (J(theta)-J(theta_"old")) - J^"CPI"_(pi_(theta_"old")) (theta) approx 0
$

这里的误差是因为新旧策略诱导出的状态访问分布之间有差异导致的。所以新策略$pi_theta$和旧策略$pi_(theta_"old")$越接近，诱导出的状态访问分布就越接近，那么误差就越小。

而衡量两个分布的差异的数学手段是KL散度。所以上面的约等于0，可以建模为

$
  |(J(theta)-J(theta_"old")) - J^"CPI"_(pi_(theta_"old")) (theta)| <= C sqrt(1/T sum_(t=0)^T ("KL"(pi_theta (a_t|s_t) parallel pi_(theta_"old") (a_t|s_t))))
$

将绝对值展开可以得到如下式子

$
  J(theta)-J(theta_"old") >= underbrace(J^"CPI"_(pi_(theta_"old")) (theta) - C sqrt(1/T sum_(t=0)^T ("KL"(pi_theta (a_t|s_t) parallel pi_(theta_"old") (a_t|s_t)))), "下界函数")
$

其中$C$是超参数。如果我们想让上式中的左手边大于0，那么只需要保证右手边大于0就可以了。右手边的式子叫做*下界函数*。

当我们优化下界函数时，我们能保证优化完毕之后，右手边的式子大于等于0吗？

现在，让我们看看策略迭代某一步中的最坏情况。考虑对新策略$pi_theta$的所有选择，其中也包含旧策略$pi_(theta_"old")$（参数更新幅度为0）。如果没有候选策略表现更好，就直接设定#text(baseline: -2pt)[$pi_theta=pi_(theta_"old")$]，并在该次迭代中不进行更新。在这种情况下，@cpiformula 告诉我们$J^"CPI"_(pi_(theta_"old")) (theta) = EE_(tau tilde colred(pi_(theta_"old"))) [sum_(t=0)^T A_t^(pi_(theta_"old")) colred((pi_(theta_"old") (a_t|s_t))/(pi_(theta_"old") (a_t|s_t)))] = 0$。因为一个策略相对于自身没有期望优势。KL散度满足$"KL"(pi parallel pi)=0$。

为了接受策略的变更，上面的式子表明，估计的策略改进$J^"CPI"_(pi_(theta_"old")) (theta)$必须大于$C sqrt(1/T sum_(t=0)^T ("KL"(pi_theta (a_t|s_t) parallel pi_(theta_"old") (a_t|s_t))))$。

如果将误差上界作为惩罚项加入到优化问题中，我们就能保证策略的单调改进。此时优化问题变成了如下形式：

$
  op(
    "argmax",
    limits: #true
  )_(pi_theta) (J^"CPI"_(pi_(theta_"old")) (theta) - C sqrt(1/T sum_(t=0)^T ("KL"(pi_theta (a_t|s_t) parallel pi_(theta_"old") (a_t|s_t))))) \
  arrow.double.long J(theta) - J(theta_"old") >= 0
$

这一结果满足了我们的最终要求。它使我们能够避免在使用原始目标 $J(theta_"old")$ 时可能出现的性能崩溃。需要注意的一个关键区别是，单调改进并不保证收敛到最优策略 $pi^*$。例如，策略优化仍可能停留在一个局部极大值处，在该处每次策略迭代都不产生改进——即$J(theta)-J(theta_"old")=0$。保证收敛仍是一个困难的没有解决的问题。

最后一步是考虑如何在实践中实现上式提出的优化问题。一个思路是直接约束KL散度的期望，如下式所示。

$
  1/T sum_(t=0)^T ("KL"(pi_theta (a_t|s_t) parallel pi_(theta_"old") (a_t|s_t))) <= delta
$

$delta$限制了KL散度的大小，因此实际上约束了新策略 $pi_theta$ 相对旧策略 $pi_(theta_"old")$ 能偏离的程度。只有策略空间中位于 $pi_(theta_"old")$ 附近的小邻域内的候选策略才会被考虑。这个邻域称为置信域（trust region），而上式称为置信域约束。需要注意的是，$delta$是一个需要调参的超参数。

#figure(
  image("rl-figures/使用下界函数优化目标函数.svg"),
  caption: [使用下界函数优化目标函数],
)

将约束和替代目标合并以后，置信域策略优化问题如下所示。

$
  max_theta J^"CPI"_(pi_(theta_"old")) (theta) = max_theta EE_(tau tilde pi_(theta_"old")) [sum_(t=0)^T A_t^(pi_(theta_"old")) (pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t))] \
  "需满足约束条件" space space 1/T sum_(t=0)^T ("KL"(pi_theta (a_t|s_t) parallel pi_(theta_"old") (a_t|s_t))) <= delta
$

上式中的约束条件非常的难以求解（需要求解二阶导数黑塞矩阵），所以我们不如将约束条件直接做进目标函数，从而有了PPO算法！

#math.equation(
  $
    \
    \
    \
    \
    \
    \
    \
    J(theta)^"ppo-clip" = EE_(markhl(tau tilde pi_(theta_"old"), tag: #<tau>, color: #blue)) [sum_(t=0)^T [min ( markhl((pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t)), tag: #<ratio1>) markhl(A_t^(pi_(theta_"old")), tag: #<piold1>, color: #green), markhl("clip", tag: #<clip>, color: #gray)( markhl((pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t)), tag: #<ratio2>), 1-epsilon, 1+epsilon) markhl(A_t^(pi_(theta_"old")), tag: #<piold2>, color: #green))]]
    \
    \
    \
    \
    \
    \
    \
    #annot((<ratio1>, <ratio2>), pos: top, dy: -1.5em, leader-connect: [elbow])[针对同一个状态$s_t$，\ 新策略采取动作$a_t$的概率 \ 和旧策略采取动作$a_t$的概率的比值]
    #annot((<piold1>, <piold2>), dx: -1.5em, dy: 1.5em, leader-connect: [elbow])[旧策略 \ 采取动作$a_t$ \ 的优势估计]
    #annot((<clip>), dx: 4.5em, dy: 1.5em, leader-connect: [elbow])[将比值裁剪到范围$(1-epsilon,1+epsilon)$]
    #annot((<tau>), pos: bottom + left, dx: -1.5em, dy: 2.5em, leader-connect: [elbow])[旧策略产生的轨迹]
  $,
  number-align: bottom,
  block: true,
)

#danger[
  由于我们将硬约束条件变成软约束条件做进了目标函数，所以约束就没那么强了。这也是PPO偶尔会训练退化的原因了。
]

== 补充知识

=== 状态访问分布

状态访问分布描述的是：在某个策略下，智能体在环境中运行时，访问各个状态的概率有多大。

也就是说，一个策略不只是决定动作，还会决定智能体未来更可能到哪些状态。

给定一个策略：

$
  pi(a|s)
$

初始状态分布为：

$
  rho_0 (s)
$

环境的状态转移概率为：

$
  P(s'|s,a)
$

那么在策略$pi$下，第$t$步处于状态$s$的概率记作：

$
  d_pi^t (s) = "Pr"(S_t=s|pi)
$

这就是#underline[时刻$t$的状态访问分布]。

例如：

$
  d^0_pi (s) = rho_0 (s)
$

因为第0步状态来自初始状态分布。

第1步：

$
  d^1_pi (s') = sum_s d_pi^0 (s) sum_a pi(a|s) P(s'|s,a)
$

第2步：

$
  d^2_pi (s'') = sum_s' d_pi^1 (s') sum_a pi(a|s) P(s'|s,a)
$

一般递推为：

$
  d^(t+1)_pi (s') = sum_s d_pi^t (s) sum_a pi(a|s) P(s'|s,a)
$

假设一个简单的走廊：A --- B --- C。初始状态在B，那么如果策略总是向左走，那么产生的轨迹如下：

$
  B arrow A arrow A arrow A arrow dots.c
$

所以状态访问分布是：

$
  d_pi^0(B) & = 1 \
  d_pi^1(A) & = 1 \
  d_pi^2(A) & = 1 \
$

如果策略总是向右走，那么产生的轨迹如下：

$
  B arrow C arrow C arrow C dots.c
$

所以状态访问分布是：

$
  d_pi^0(B) & = 1 \
  d_pi^1(C) & = 1 \
  d_pi^2(C) & = 1 \
$

所以即使两个策略只是在状态B的动作不同，之后访问的状态分布也完全不同。

所以策略会影响状态访问分布。

因为状态的转移满足

$
  S_t attach(limits(arrow.long), t: A_t) S_(t+1)
$

而动作来自策略：

$
  A_t tilde pi(dot.c | S_t)
$

所以策略改变动作概率，动作又改变下一状态概率。

完整的链条如下：

策略$pi(a|s) arrow.long$动作分布变化$arrow.long$环境状态转移$P(s'|s,a) arrow.long$未来状态分布变化$arrow.long$未来采样到的数据也变化。

这就是强化学习和普通监督学习的一个核心区别：策略不仅影响输出动作，还影响之后的数据分布。

如果任务长度为$T$，那么每个时间步都有一个状态分布：

$
  d^0_pi(s), d^1_pi(s), ..., d^T_pi(s)
$

目标函数通常写作

$
  J(theta) & = EE_(tau tilde pi_theta)[G(tau)] \
           & = sum_(t=0)^T EE_(s_t tilde d_pi^t, a_t tilde pi) [r(s_t,a_t)] \
           & = sum_(t=0)^T sum_s d^t_pi (s) sum_a pi(a|s)r(s,a)
$

这表示：总收益等于每个时刻访问某状态的概率 × 在该状态选动作的概率 × 奖励。

=== 全变差距离（Total Variation Distance）

全变差距离表示：两个分布在所有事件上的概率差异，最大能有多大。

假设有两个离散概率分布$P$和$Q$，定义在同一个样本空间$cal(X)$上。

$
  D_"TV" (P, Q) = 1/2 sum_(x in cal(X)) abs(P(x) - Q(x))
$

例如：

$
  P & = (0.3, 0.7) \
  Q & = (0.4, 0.6)
$

那么全变差距离为：

$
  D_"TV" (P, Q) = 1/2 (abs(0.3-0.4)+abs(0.7-0.6)) = 0.1
$

TV距离和KL散度之间有一个重要的关系，叫做#underline[Pinsker不等式]。

$
  D_"TV" (P, Q) <= sqrt(1/2 D_"KL" (P parallel Q))
$

这说明：如果KL很小，那么TV一定很小。

=== 误差上界的推导

将性能差分展开得到

$
  J(theta) - J(theta_"old") = sum_(t=0)^T sum_s d_(pi_theta)^t (s) sum_a pi_theta (a|s) A_(pi_(theta_"old"))^t
$

而CPI surrogate是：

$
  J_(pi_(theta_"old"))^"CPI" (theta) = sum_(t=0)^T sum_s d_(pi_(theta_"old"))^t (s) sum_a pi_theta (a|s) A_(pi_(theta_"old"))^t
$

所以误差定义为：

$
  Delta & = (J(theta) - J(theta_"old")) - J_(pi_(theta_"old"))^"CPI" (theta) \
  & = sum_(t=0)^T sum_s (d_(pi_theta)^t (s) - d_(pi_(theta_"old"))^t (s)) sum_a pi_theta (a|s) A^(pi_(theta_"old"))_t \
  & = sum_(t=0)^T sum_s (d_(pi_theta)^t (s) - d_(pi_(theta_"old"))^t (s)) EE_(a tilde pi(dot.c|s)) [A^(pi_(theta_"old"))_t]
$

我们定义：

$
  g_(pi_theta)^t (s) := EE_(a tilde pi(dot.c|s)) [A^(pi_(theta_"old"))_t]
$

利用一般不等式：若$P,Q$是两个分布，$f$有界，则

$
  abs(EE_(x tilde P) [f(x)] - EE_(x tilde Q) [f(x)]) <= 2 norm(f)_infinity D_"TV" (P, Q)
$

#tip(title: [$norm(f)_infinity$])[
  在所有可能输入$x$上，$f(x)$的绝对值最大是多少。

  例子：如果$f=(2,-5,3)$，那么$norm(f)_infinity = 5$
]

因此对于每个时间步t，

$
  abs(sum_s (d_(pi_theta)^t (s) - d_(pi_(theta_"old"))^t (s)) g_(pi_theta)^t (s)) <= 2 norm(g_(pi_theta)^t)_infinity D_"TV" (d_(pi_theta)^t, d_(pi_(theta_"old"))^t)
$

也就是

$
  abs(Delta) <= sum_(t=0)^T 2 norm(g_(pi_theta)^t)_infinity D_"TV" (d_(pi_theta)^t, d_(pi_(theta_"old"))^t)
$

#chapter("组相对策略优化（GRPO）", image: image("./orange2.jpg"), l: "rl-grpo")

== GRPO原理

#tip(title: [GRPO])[
  Group Relative Policy Optimization：组相对策略优化
]

#theorem(name: [GRPO的目标函数])[
  $
    J(theta)^"GRPO" = 1/G sum_(i=1)^G 1/abs(tau_i) sum_(t=1)^abs(tau_i) min [ rho A_(tau_i,t), "clip"(p, 1-epsilon,1+epsilon) A_(tau_i,t)] \
    "其中比值（重要性权重）" space space rho = (pi_theta (a_(tau_i,t)|s_(tau_i,t)))/(pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t)))
  $
]

先来说明一下GRPO目标函数中每个数学符号的含义：

- $pi_theta$表示正在更新的策略。
- $pi_(theta_"old")$表示上一轮训练好的旧策略。
- $G$表示使用旧策略$pi_(theta_"old")$采样的一组轨迹的数量，也就是如果我们使用旧策略采样了10条轨迹，那么$G=10$。
- $tau_i$表示第$i$条轨迹。
- $abs(tau_i)$表示第$i$条轨迹的动作数量。
- $pi_theta (a_(tau_i,t)|s_(tau_i,t))$表示第$i$条轨迹的第$t$个时刻的状态为$s_(tau_i,t)$，以及在这个状态下正在更新的策略$pi_theta$采取动作$a_(tau_i,t)$的概率。
- $pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t))$表示第$i$条轨迹的第$t$个时刻的状态为$s_(tau_i,t)$，以及在这个状态下正在更新的策略$pi_(theta_"old")$采取动作$a_(tau_i,t)$的概率。
- $A_(tau_i,t)$表示第$i$条轨迹的第$t$个时刻的动作的优势。

== 使用GRPO玩倒立摆游戏

#codly(header: [策略网络])
```python
class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x
```

智能体代码如下：

```python
class Agent:
    def __init__(self):
        self.lr = 0.0002
        self.action_size = 2
        self.pi = PolicyNet(self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)
        m = Categorical(probs)
        action = m.sample().item()

        return action, probs
```

将采样一条轨迹封装为一个函数

```python
class Agent:
    ...

    def collect_trajectory(self, env):
        """采样一条轨迹"""
        state = env.reset()
        states, log_probs, actions = [], [], []
        episode_reward = 0
        done = False

        while not done:
            action, probs = self.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_prob = torch.log(probs)[action]
            log_probs.append(log_prob.item())

            state = next_state
            episode_reward += reward

        # 归一化奖励
        normalized_reward = episode_reward / 200.0

        return states, log_probs, actions, normalized_reward
```

这里的归一化奖励需要说一下，我们已经知道木杆坚持200步不倒下，游戏就成功结束了。那么如果木杆坚持了3步就倒下，这条轨迹的奖励应该如何计算呢？这里我们选择$3/200=0.015$。这就是我们给这条轨迹的奖励。

GRPO的优势计算是和PPO的优势计算有区别的地方。PPO使用了价值函数网络评估每个动作的价值，并且使用了广义优势估计（GAE）。而GRPO创造性的提出了组相对优势。

也就是轨迹$tau_i$相对于*组内*其它轨迹的优势是多少？也就是如下

轨迹$tau_i$的归一化奖励是：

$
  R_(tau_i)^"normalized" = G(tau_i)/200
$

而一组轨迹的平均奖励是

$
  "reward"_"mean" = 1/G sum_(i=0)^G R_(tau_i)^"normalized"
$

那么轨迹$tau_i$相对于组内其它轨迹的优势为：

$
  A_(tau_i) = (R_(tau_i)^"normalized" - "reward"_"mean")/("reward"_"std")
$

轨迹中每一个动作的优势就等于这个动作所在轨迹的优势，也就是：

$
  A_(tau_i,t) = A_(tau_i)
$

#figure(
  image("rl-figures/组内优势计算.svg"),
  caption: [组内优势的计算，要注意的是动作的优势等于所在轨迹的组内优势],
)

```python
class Agent:
    ...

    def calc_advantages_with_grpo(self, trajectories):
        """使用一组轨迹计算某条轨迹的组内优势"""
        # [轨迹0的归一化奖励，轨迹1的归一化奖励，...]
        rewards = [r for o, l, a, r in trajectories]
        mean_reward = sum(rewards) / len(rewards)
        std_reward = np.std(rewards) + 1e-8
        # [轨迹0的组相对优势，轨迹1的组相对优势，...]
        advantages = [(r - mean_reward) / std_reward for r in rewards]

        return advantages
```

使用GRPO算法更新策略的代码如下

```python
class Agent:
    ...

    def update(self, trajectories):
        advantages = self.calc_advantages_with_grpo(trajectories)

        for step in range(20):
            loss = 0.0
            for traj, advantage in zip(trajectories, advantages):
                """遍历组里面的每一条轨迹和对应的组内优势"""
                states, log_probs, actions, _ = traj
                states = torch.tensor(states)
                log_probs = torch.tensor(log_probs).view(-1, 1)
                actions = torch.tensor(actions).view(-1, 1)
                new_log_probs = torch.log(self.pi(states).gather(1, actions))
                ratio = torch.exp(new_log_probs - log_probs)
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                traj_loss = torch.mean(
                    -torch.min(ratio * advantage, clipped_ratio * advantage))

                loss += traj_loss
            loss = loss / len(trajectories)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return None
```

训练循环如下：

```python
def train(agent, env):
    G = 5  # 一组轨迹有5条
    trial_num = 0
    while True:
        for episode in range(20):
            trajectories, episode_rewards = [], []
            for _ in range(G):
                states, log_probs, actions, normalized_reward = agent.collect_trajectory(
                    env)
                trajectories.append(
                    (states, log_probs, actions, normalized_reward))
                episode_rewards.append(normalized_reward * 200)
            agent.update(trajectories)

        # 一组轨迹的平均奖励
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        trial_num += 1

        if avg_reward > 195:
            print("训练结束，训练回合数：", trial_num)
            return
        else:
            print(f"训练回合数：{trial_num}，平均奖励：{avg_reward}")
```

#part("基于人类反馈的强化学习")

#chapter("大语言模型训练概述", image: image("./orange2.jpg"), l: "rlhf-llm-pipeline")

== LLM训练流程

#figure(
  image("rl-figures/大语言模型训练流程.svg"),
  caption: [大语言模型训练流程],
)

=== 预训练

假设训练数据是"abcd"，那么预测下一个token的原理如下

#table(
  columns: 3,
  [a], [b], [c],
  [#sym.arrow.b], [#sym.arrow.b], [#sym.arrow.b],
  [b], [c], [d],
)

也就是我们希望如下的目标函数越大越好

$
  J(theta) = pi_theta (b|a) pi_theta (c|a,b) pi_theta (d|a,b,c)
$

从这里我们可以看出，在LLM这个环境中

- 策略$pi_theta$是LLM
- 状态是上下文，例如$pi_theta (d|a,b,c)$的状态$S_t="字符串abc"$
- 动作：$pi_theta (d|a,b,c)$表示在状态$S_t="字符串abc"$的情况下，$A_t$为输出字符"d"。
- 动作概率：$pi_theta (d|a,b,c)$为采取动作的概率。

=== 监督微调

假设训练数据是"How are you? I am fine.\<eos_token\>"。

其中指令部分为"How are you?"。

回答部分为"I an fine.\<eos_token\>"。

那么训练方式如下：

#table(
  columns: 8,
  [How],
  [are],
  [you],
  table.cell(
    fill: green.lighten(60%),
  )[?],
  table.cell(
    fill: green.lighten(60%),
  )[I],
  table.cell(
    fill: green.lighten(60%),
  )[am],
  table.cell(
    fill: green.lighten(60%),
  )[fine],
  table.cell(
    fill: green.lighten(60%),
  )[.],
  [#sym.arrow.b],
  [#sym.arrow.b],
  [#sym.arrow.b],
  table.cell(
    fill: green.lighten(60%),
  )[#sym.arrow.b],
  table.cell(
    fill: green.lighten(60%),
  )[#sym.arrow.b],
  table.cell(
    fill: green.lighten(60%),
  )[#sym.arrow.b],
  table.cell(
    fill: green.lighten(60%),
  )[#sym.arrow.b],
  table.cell(
    fill: green.lighten(60%),
  )[#sym.arrow.b],

  [are],
  [you],
  [?],
  table.cell(
    fill: green.lighten(60%),
  )[I],
  table.cell(
    fill: green.lighten(60%),
  )[am],
  table.cell(
    fill: green.lighten(60%),
  )[fine],
  table.cell(
    fill: green.lighten(60%),
  )[.],
  table.cell(
    fill: green.lighten(60%),
  )[\<eos_token\>],
)

只有绿色部分计算损失。也就是我们要让下面的式子越大越好

$
  J(theta) = & pi_theta ("I"|"How are you?") times \
             & pi_theta ("am"|"How are you? I") times \
             & pi_theta ("fine"|"How are you? I am") times \
             & pi_theta ("."|"How are you? I am fine") times \
             & pi_theta ("<eos_token>"|"How are you? I am fine.")
$

通过观察上面的目标函数，我们发现，如果SFT训练的轮数很多，那么会过拟合监督微调数据。模型会发生灾难性遗忘，因为在目标函数中没有约束新旧模型的偏差（KL散度）。

#danger[
  SFT一般不会训练很多轮！否则会出现灾难性遗忘。
]

监督微调（Supervised Fine-Tuning，SFT）通常也是采用"预测下一个词"（predict next token）的训练方式。

- 基础模型训练：大多数语言模型（如GPT系列）本身就是通过自回归方式训练的，即在给定上下文的情况下预测下一个词（token）。
- 监督微调（SFT）：在微调阶段，模型通常会使用带有标签的输入-输出对进行训练，比如对话数据、问答对等。训练目标依然是让模型在给定输入（上下文）的条件下，预测正确的下一个词。

换句话说，SFT通过*有监督的数据*指导模型生成更符合特定任务或风格的输出，但训练目标仍然是最大化正确预测下一个词的概率。


#figure(
  table(
    columns: 3,
    [训练阶段], [训练方式], [目标],
    [预训练], [自回归预测下一个词], [学习语言的通用统计规律],
    [监督微调], [给定输入，预测下一个词], [让模型生成更符合特定任务的输出],
  ),
  caption: [预训练和监督微调],
)

所以我们要使用"强化学习"来对大语言模型进行微调，这就是"基于人类反馈的强化学习"。

#tip(title: [RLHF])[
  RLHF:  #underline([R])einforcement #underline[L]earning From #underline[H]uman #underline[F]eedback
]

当我们谈论大语言模型（LLM）的强化学习时，我们进入了一个完全不同的世界。我们不再训练智能体在倒立摆环境中的表现，而是对预训练好的大语言模型进行微调，使其符合人类的偏好。该模型不会与外部环境交互——它本质上是在探索自身的输出空间。

正如OpenAI和其他科研人员所发现的，这种方法对于将原始语言模型转化为辅助系统至关重要。正如IBM研究人员指出的那样，"RLHF特别适合于目标复杂、定义不明确或难以指定的任务。"毕竟，如何用数学来定义"乐于助人"或"诚实"这样的概念呢？

这里的根本转变是：

+ 我们正在优化大语言模型以适应人类的偏好，而不是对环境的掌控
+ 我们的数据来自人类的判断，而不是环境互动
+ 我们需要在奖励最大化与保持接近原始预训练行为之间取得平衡

这种平衡行为使得LLM强化学习特别棘手，但也特别令人着迷！

== RLHF的关键技术

#figure(
  image("rl-figures/RLHF常用算法.svg"),
  caption: [RLHF常用算法],
)

=== 近端策略优化（PPO）

PPO是LLM强化学习对齐技术的*重量级冠军*，因OpenAI开发的InstructGPT和ChatGPT而闻名。PPO于2017年开发，解决了强化学习中的一个关键挑战：如何在不破坏训练稳定性的情况下进行有意义的更新。

PPO成功的秘诀在于其"近端"特性——它对策略进行保守更新，防止模型在单次迭代中发生过大变化。这是通过其目标函数中巧妙的裁剪机制实现的：

$
  J_"PPO" (theta)=EE[min((pi_theta (a|s))/(pi_theta_"old" (a|s))A, "clip"((pi_theta (a|s))/(pi_theta_"old" (a|s)),1-epsilon,1+epsilon)A)]
$

通过限制新旧策略之间的比例（通常在 $1 plus.minus 0.2$ 以内），PPO 可以确保模型在训练过程中不会偏离目标。

PPO 一直是实现人类反馈强化学习 (RLHF) 的首选算法，该算法遵循以下三个步骤：

+ 从预训练模型开始
+ 监督微调
+ 根据人类偏好训练奖励模型
+ 使用PPO微调LLM，来最大化奖励，同时保持接近原来的LLM。

=== 直接偏好优化（DPO）

如果说PPO是一位谨慎的外科医生，能够精准地进行手术，那么DPO则是一位效率专家，他找到了通往同一目标的捷径。DPO于2023年在一篇题为《直接偏好优化：你的语言模型其实是一个奖励模型》的论文中首次提出，它彻底消除了对单独奖励模型的需求。

DPO的精妙之处在于其数学洞察力：奖励函数与最优策略之间存在直接映射。通过利用这种关系，DPO将强化学习问题转化为基于人类偏好数据的更简单的分类问题。

与传统的三步RLHF流程不同，DPO只需一个训练阶段即可实现相同的目标。这就像跳过中间环节，直接到达源头。

DPO对从业者特别有吸引力的原因是：

+ 简单：无需训练单独的奖励模型
+ 效率：无需在训练期间进行昂贵的采样
+ 稳定性：组件越少，出错的可能性就越小
+ 性能：在控制输出属性方面，通常与PPO-RLHF相当或超过PPO-RLHF

DPO的目标函数如下：

$
  \
  \
  \
  \
  \
  \
  J^"DPO" (markhl(pi_theta, tag: #<dpo1>);markhl(pi_"ref", tag: #<dpo2>)) = EE_((y_w,y_l,x) tilde markhl(cal(D), tag: #<dpo3>)) [log(markhl(sigma, tag: #<dpo4>)(markhl(beta, tag: #<dpo5>)(log (pi_theta (markhl(y_w, tag: #<dpo6>)|x))/(pi_theta (markhl(y_l, tag: #<dpo7>)|markhl(x, tag: #<dpo8>))) - log (pi_"ref" (y_w|x))/(pi_"ref" (y_l|x)))))]
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  #annot(<dpo1>, [正在训练的LLM], leader-connect: "elbow", pos: left + bottom, dx: -2em, dy: 1em)
  #annot(<dpo2>, [冻结的参考LLM], leader-connect: "elbow", pos: left + top, dx: -2em, dy: -1em)
  #annot(<dpo3>, [偏好数据集], leader-connect: "elbow", pos: left + top, dx: -2em, dy: -2em)
  #annot(<dpo4>, [sigmoid函数], leader-connect: "elbow", pos: left + bottom, dx: -2em, dy: 2em)
  #annot(<dpo5>, [超参数], leader-connect: "elbow", pos: left + top, dx: -2em, dy: -2em)
  #annot(<dpo6>, [人类偏好的回答], leader-connect: "elbow", pos: right + top, dx: 2em, dy: -2em)
  #annot(<dpo7>, [人类厌恶的回答], leader-connect: "elbow", pos: right + bottom, dx: 2em, dy: 4em)
  #annot(<dpo8>, [提示词], leader-connect: "elbow", pos: right + bottom, dx: 2em, dy: 2em)
$

DPO的目标是让$(pi_theta (y_w|x))/(pi_theta (y_l|x))$越大越好！

=== 组相对策略优化（GRPO）

#figure(
  $
    \
    \
    \
    \
    \
    \
    \
    J^"GRPO" (theta) = EE_(markhl(q tilde P(Q), tag: #<grpo1>, color: #blue), markhl({o_i}^G_(i=1) tilde pi_(theta_"old") (Q|q), tag: #<grpo2>, color: #red)) [markhl(cal(L)_"GRPO" (theta), tag: #<grpo3>)]
    \
    \
    \
    \
    \
    \
    \
    #annot(<grpo1>, pos: bottom, leader-connect: "elbow", dx: 2em, dy: 2em, [从分布$P(Q)$中采样一些问题$q$。\
      $P(Q)$表示用来训练模型的问题数据集])
    #annot(<grpo2>, pos: top, leader-connect: "elbow", dx: 2em, dy: -2em, [针对每个采样出来的问题$q$。\
      使用旧策略生成$G$个不同的回答${o_1,o_2,dots,o_G}$])
    #annot(<grpo3>, pos: right + bottom, leader-connect: "elbow", dx: 2em, dy: 2em, [替代损失函数])
  $,
  caption: [GRPO目标函数],
)

#figure(
  $
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    cal(L)_"GRPO" (theta) = markhl(1/G sum^G_(i=1), tag: #<grpo4>, color: #red) markhl(1/abs(o_i) sum_(t=1)^abs(o_i), tag: #<grpo5>, color: #blue) min [p markhl(hat(A)_(i,t), tag: #<grpo6>), "clip"(p, 1-epsilon,1+epsilon)hat(A)_(i,t)] - beta D_"KL" [pi_theta parallel pi_"ref"]
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    \
    "其中比值" space space space p = (pi_theta (o_(i,t)|q,o_(i,<t)))/(pi_(theta_"old") (o_(i,t)|q,o_(i,<t)))
    #annot(<grpo4>, pos: top, leader-connect: "elbow", dx: 2em, dy: -2em, [裁剪替代损失的token平均损失
      - 遍历$G$个采样的回答，并将每个回答的token平均损失相加
      - 除以$G$，来获得所有采样的回答的平均值，这样保证了每个回答$o_i$的贡献的单位都一样])
    #annot(<grpo5>, pos: right + bottom, leader-connect: "elbow", dx: 2em, dy: 4em, [某个回答的每个token都计算损失，并取平均值
      - 将回答$o_i$的所有token都计算一遍损失，然后相加
      - 除以token的数量：$abs(o_i)="len"(o_i)$，这样保证了每个token都有相同单位的贡献])
    #annot(<grpo6>, pos: right + bottom, leader-connect: "elbow", dx: 2em, dy: 2em, [组内优势])
  $,
  caption: [GRPO替代损失函数],
)

现在，如果我们能将 PPO 的可靠性与更高的效率以及对推理能力提升的专注结合起来，会怎么样呢？GRPO 应运而生，它是强化学习领域的最新成果之一，由 DeepSeek 开发，并用于训练其令人印象深刻的 DeepSeek-Math 和 DeepSeek-R1 模型。

GRPO 建立在 PPO 的基础上，但引入了几项巧妙的修改：

- GRPO去掉了价值函数模型，减少了内存开销。
- GRPO评估输出的一组回答而不是单个token。
- GRPO直接将KL散度纳入损失函数。

这种基于组（Group）的方法尤其巧妙。GRPO 不是单独评估每个token，而是将完整的答案作为一个整体来看待 #sym.arrow.double.long 这是一种评估推理能力更自然的方式，其中整个解答过程都很重要，而不仅仅是单个步骤。

用 AWS 社区文章的话来说，"GRPO 用于计算优势的组相对方式与奖励模型的比较性质非常吻合，因为奖励模型通常是在同一问题的输出比较数据集上进行训练的。"

#chapter("使用DPO微调大语言模型", image: image("./orange2.jpg"), l: "rlhf-dpo")

== DPO理论介绍以及DPO存在的问题

#tip[
  - DPO：Direct Preference Optimization
  - 直接偏好优化：你的大语言模型实际上是一个奖励模型
]

=== 偏好数据集

DPO需要偏好数据集来微调LLM。偏好数据集的格式如下：

```json
{
  "prompt": "这部电影怎么样？",
  "chosen": "这部电影很好看。",
  "rejected": "这部电影不好看。"
}
```

数据集构建的方法：

- 针对同一个prompt，通过调整温度，让LLM输出不同的回答。然后让数据标注工程师来标注对不同回答的偏好。#link("https://blog.lukesalamone.com/posts/what-is-temperature/")[温度示例网站]。
- 在使用ChatGPT时，你可能注意到，偶尔会被要求在两个相似的答案中选择一个来继续对话。这个偏好会被记录下来，并用于在未来的偏好调整中改进模型。同样。
- 手工标注
- 合成数据：通过写提示词来让LLM生成偏好数据集（需要人工审核）。
- 使用网上的开源数据集
- ...

数据集容易存在的问题：

+ 正负例区分不明显

```json
{
  "prompt": "这部电影怎么样？",
  "chosen": "这部电影很好看。",
  "rejected": "这部电影挺好看。"
}
```

人类都无法识别哪个应该是正例，哪个应该是负例。

+ 数据集中存在偏好循环

```json
{
  "prompt": "这部电影怎么样？",
  "chosen": "这部电影很好看。",
  "rejected": "这部电影很差。"
},
{
  "prompt": "这部电影怎么样？",
  "chosen": "这部电影很差。",
  "rejected": "这部电影很好看。"
}
```

当模型看到以上数据时，就不知道人类的偏好是什么了。无法学到任何东西。因为第一个人标注的数据的偏好是$A succ B$，第二个人标注的数据的偏好是$B succ A$，那么模型看到的偏好是$A succ B succ A$。这就是偏好循环。

=== DPO目标函数

$
  \
  \
  \
  \
  \
  \
  J^"DPO" (markhl(pi_theta, tag: #<dpo1>);markhl(pi_"ref", tag: #<dpo2>)) = EE_((y_w,y_l,x) tilde markhl(cal(D), tag: #<dpo3>)) [log(markhl(sigma, tag: #<dpo4>)(markhl(beta, tag: #<dpo5>)(log (pi_theta (markhl(y_w, tag: #<dpo6>)|x))/(pi_theta (markhl(y_l, tag: #<dpo7>)|markhl(x, tag: #<dpo8>))) - log (pi_"ref" (y_w|x))/(pi_"ref" (y_l|x)))))]
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  \
  #annot(<dpo1>, [正在训练的LLM], leader-connect: "elbow", pos: left + bottom, dx: -2em, dy: 1em)
  #annot(<dpo2>, [冻结的参考LLM], leader-connect: "elbow", pos: left + top, dx: -2em, dy: -1em)
  #annot(<dpo3>, [偏好数据集], leader-connect: "elbow", pos: left + top, dx: -2em, dy: -2em)
  #annot(<dpo4>, [sigmoid函数], leader-connect: "elbow", pos: left + bottom, dx: -2em, dy: 2em)
  #annot(<dpo5>, [超参数], leader-connect: "elbow", pos: left + top, dx: -2em, dy: -2em)
  #annot(<dpo6>, [人类偏好的回答], leader-connect: "elbow", pos: right + top, dx: 2em, dy: -2em)
  #annot(<dpo7>, [人类厌恶的回答], leader-connect: "elbow", pos: right + bottom, dx: 2em, dy: 4em)
  #annot(<dpo8>, [提示词], leader-connect: "elbow", pos: right + bottom, dx: 2em, dy: 2em)
$

这里我们思考一下，给定提示词$x$的情况下，模型输出补全$y$的概率是多少呢？如下所示：

$
  pi_theta (y|x) & = P_("LLM"_theta) (y_0|x) P_("LLM"_theta) (y_1|x,y_0) dots.c P_("LLM"_theta) (y_t|x,y_(<t)) \
                 & = product_(t=0)^abs(y) P_("LLM"_theta) (y_t|x,y_(<t))
$

然后根据log的性质也就是$log A/B = log A - log B$。可以推导出目标函数最内部的表达式的另一种形式。

$
  & log (pi_theta (y_w|x))/(pi_theta (y_l|x)) - log (pi_"ref" (y_w|x))/(pi_"ref" (y_l|x)) \
  & = log pi_theta (y_w|x) - log pi_theta (y_l|x) - log pi_"ref" (y_w|x) + log log (pi_"ref" (y_l|x)) \
  & = log pi_theta (y_w|x) - log pi_"ref" (y_w|x) - (log pi_theta (y_l|x) - log log (pi_"ref" (y_l|x))) \
  & = log (pi_theta (y_w|x))/(pi_"ref" (y_w|x)) - log (pi_theta (y_l|x))/(pi_"ref" (y_l|x))
$

通过观察目标函数，我们知道在最开始训练时：$pi_theta=pi_"ref"$，所以$log (pi_theta (y_w|x))/(pi_"ref" (y_w|x)) - log (pi_theta (y_l|x))/(pi_"ref" (y_l|x)) = 0$。但由于外层有$sigma$函数，所以不影响反向传播更新网络（$sigma(0)=0.5$）。

通过观察目标函数，我们可以知道随着训练的进行，$(pi_theta (y_w|x))/(pi_theta (y_l|x))$会越来越大。因为$(pi_"ref" (y_w|x))/(pi_"ref" (y_l|x))$是一个常数（作为正则化项存在）。

#danger(title: [DPO存在的问题])[
  + $(pi_theta (y_w|x))/(pi_theta (y_l|x))$的分子和分母可能同时增大或者减小。例如分母增大了1.5倍，分子增大了3倍。那么就导致了LLM输出人类偏好的回答的概率和输出人类讨厌的回答的概率都增加了。
  + $(pi_theta (y_w|x))/(pi_theta (y_l|x))$随着训练，分子越来越大，分母越来越小，结果导致了$pi_theta (y_w|x)$趋近于1，$pi_theta (y_l|x)$趋近于0，正则化项$(pi_"ref" (y_w|x))/(pi_"ref" (y_l|x))$没有起到作用，最终LLM彻底失去了探索能力。也就是出现了"过拟合"的问题。DPO容易过拟合，泛化能力不足。所以DPO的训练轮数不能太多。
  + 数据集中存在的偏好循环会导致LLM学不到东西。例如数据集中存在$A succ B$和$B succ C$，那么模型可以学习到人类的偏好是$A succ B succ C$。如果这是还有一条数据中$C succ A$，那么模型就懵了，不知道人类的偏好是什么了。因为模型看到的是$A succ B succ C succ A$。也就是偏好循环。
]

尽管存在这些缺点，DPO仍然是一种非常有效的工具；许多最成功和性能最好的开源 LLM 都是使用 DPO 进行指令微调的。

#danger[
  个人认为DPO不算是强化学习，因为DPO需要训练数据集。而真正的强化学习的训练数据是由策略模型自己采样得来的。
]

== DPO实战

我们遵循RLHF的标准流程：预训练LLM #sym.arrow.double.long SFT #sym.arrow.double.long DPO

=== 第一步：对预训练模型进行基于指令的监督微调（SFT）

#tip[
  - 基于指令的监督微调：Instruct SFT
  - 目标：在给定问题（Question）的前提下，使得模型产生该回答（Answer）的概率最大化。
]

#figure(
  image("rl-figures/sft-object.svg"),
  caption: [SFT的损失函数],
)

我们要注意的是在*基于指令的监督微调*中，我们只针对*答案*部分计算损失。

#figure(
  image("rl-figures/只针对答案计算损失.svg"),
  caption: [基于指令的监督微调只针对答案部分计算损失],
)

我们将损失的计算推广到批次（batch）。

#figure(
  image("rl-figures/推广到批次计算损失.svg"),
  caption: [将只针对答案计算损失推广到批次],
)

首先导入需要的依赖。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import os
from dataclasses import dataclass
import datasets
import time
```

接下来我们导入模型和分词器。

```python
device = "cuda"
model_path = "./Qwen3-0.6B-Base"

model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

然后我们设置一下生成文本的参数。保证测试的一致性。

其中151645为`<|im_end|>`，151643为`<|endoftext|>`。

#figure(
  ```python
  model.generation_config.do_sample = True
  model.generation_config.eos_token_id = [151645, 151643]
  model.generation_config.pad_token_id = 151643
  model.generation_config.temperature = 0.7
  model.generation_config.top_p = 0.8
  model.generation_config.top_k = 20
  model.generation_config.repetition_penalty = 1.05
  ```,
  caption: [模型生成的配置],
) <模型生成的配置>

然后我们定义一下SFT阶段的训练超参数。

```python
@dataclass
class SFTConfig:
    max_length = 2500
    batch_size = 2
    gradient_accumulation_steps = 8
    log_iter = 400
    max_lr = 2e-5
    min_lr = 2e-6
    warmup_steps = 1000
```

接下来我们导入训练数据并将训练数据转换成`input_ids`。

```python
ultrachat_200k_data = datasets.load_dataset("./ultrachat_200k")

def tokenize_and_format(data):
    """使用模型自带的聊天模板格式化训练数据"""
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize = True,
        add_generation_prompt = False,
        truncation = True,
        max_length = 2500,
    )

    return input_ids

## 生成训练数据的input_ids
train_data = []
i = 0
while True:
    data = ultrachat_200k_data["train_sft"][i]["messages"]
    # 添加系统提示词
    data.insert(
        0,
        {"content": "You are a helpful assistant", "role": "system"}
    )
    input_ids = tokenize_and_format(data)
    train_data.append(input_ids)
    i += 1
    if i % 1000 == 0:
        print(f"已处理{i}条数据")
    if i == 50000:
        break
```

接下来我们编写一下学习率的线性预热和余弦衰减的函数。

#figure(
  ```python
  def linear_warmup(current_step, warmup_steps, max_lr):
      if current_step < warmup_steps:
          return max_lr * current_step / warmup_steps
      else:
          return max_lr

  def cosine_decay(current_step, warmup_steps, total_steps, max_lr, min_lr):
      if current_step < warmup_steps:
          return linear_warmup(current_step, warmup_steps, max_lr)
      else:
          progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
          decay = 0.5 * (1 + np.cos(np.pi * progress))
          return (max_lr - min_lr) * decay + min_lr
  ```,
  caption: [学习率调度：线性预热和余弦衰减],
) <学习率调度>

#tip(title: [基于指令SFT的掩码设置])[
  - SFT和预训练的区别核心就是掩码掉"问题"部分的损失，而只看"回答"部分的损失，并仅基于回答部分的损失进行优化
  - 实现方式：构造损失掩码，仅针对每轮对话（含多轮）的模型"输出"部分（也就是回答部分）进行损失计算
]

假设我们的多轮对话数据是

```json
data = [
    {
        "content": "你是谁？",
        "role": "user"
    },
    {
        "content": "我是左元。",
        "role": "assistant"
    },
    {
        "content": "你会强化学习吗？",
        "role": "user"
    },
    {
        "content": "略知一二。",
        "role": "assistant"
    }
]
```

经过模型的对话模板格式化之后是

```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
你是谁？<|im_end|>
<|im_start|>assistant
我是左元。<|im_end|>
<|im_start|>user
你会强化学习吗？<|im_end|>
<|im_start|>assistant
<think></think>略知一二。<|im_end|>
```

代码如下：


#show figure: set block(breakable: true)
#figure(
  ```python
  def create_answer_mask(input_ids, tokenizer):
      """
      创建仅对助手回答部分计算损失的掩码

      Args:
          input_ids: 输入token序列 [batch_size, seq_len]
          tokenizer: 分词器

      Returns:
          answer_mask: 助手回答部分为1，其他部分为0的掩码
      """
      batch_size, seq_len = input_ids.shape
      answer_mask = torch.zeros_like(input_ids)

      # 获取结束标记的token id
      eos_token_id = tokenizer.encode("<|im_end|>")[0]

      for batch_idx in range(batch_size):
          # 找到所有 <|im_end|> 的位置
          eos_positions = torch.where(
              input_ids[batch_idx] == eos_token_id
          )[0].tolist()

          if len(eos_positions) < 2:  # 至少需要user和assistant各一个结束标记
              continue

          # 解析对话轮次
          user_ends, assistant_ends = \
              _parse_conversation_turns(eos_positions)

          # 为每个助手回答设置掩码
          _set_answer_masks(
              answer_mask[batch_idx],
              user_ends,
              assistant_ends,
              seq_len
          )

      return answer_mask


  def _parse_conversation_turns(eos_positions):
      """
      解析对话轮次，分离用户和助手的结束位置

      对话格式：
      <|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>\n

      eos_positions[0]: system结束 (如果有)
      eos_positions[1]: 第1轮user结束
      eos_positions[2]: 第1轮assistant结束
      eos_positions[3]: 第2轮user结束
      eos_positions[4]: 第2轮assistant结束
      ...
      """
      # 跳过system系统提示词部分，从第一个user开始
      conversation_eos = eos_positions[1:]  # 去掉system的<im_end>

      # 偶数索引：user结束位置，奇数索引：assistant结束位置
      # pos + 1 为跳过<im_end>，此时pos+1指向`\n`
      user_ends = [pos + 1 for pos in conversation_eos[::2]] # 每隔2个取一个，从0开始，顺便跳过`<im_end>`
      assistant_ends = [pos + 1 for pos in conversation_eos[1::2]] # 每隔2个取一个，从1开始，顺便跳过`<im_end>`

      return user_ends, assistant_ends


  def _set_answer_masks(mask, user_ends, assistant_ends, seq_len):
      """
      为助手回答部分设置掩码

      Args:
          mask: 当前样本的掩码 [seq_len]
          user_ends: 用户消息结束位置列表
          assistant_ends: 助手消息结束位置列表
          seq_len: 序列长度
      """
      num_user_turns = len(user_ends)
      num_assistant_turns = len(assistant_ends)

      if num_user_turns == num_assistant_turns:
          # 完整对话：每轮都有用户问题和助手回答
          for user_end, assistant_end in zip(user_ends, assistant_ends):
              answer_start = user_end + 3  # 跳过 <|im_start|>assistant 这2个token，answer_start指向assistant后面的`\n`
              answer_end = assistant_end - 1  # 不包含<im_end>后面的`\n`
              mask[answer_start:answer_end] = 1 # 左闭右开区间，所以掩码不包含<im_end>

      elif num_user_turns == num_assistant_turns + 1:
          # 未完成对话：最后一轮助手回答被截断

          # 处理完整的对话轮次
          for user_end, assistant_end in zip(user_ends[:-1], assistant_ends):
              answer_start = user_end + 3
              answer_end = assistant_end - 1
              mask[answer_start:answer_end] = 1

          # 处理最后一轮被截断的助手回答
          last_user_end = user_ends[-1]
          last_answer_start = last_user_end + 3
          mask[last_answer_start:] = 1  # 到序列结尾
  ```,
  caption: [答案掩码],
) <对答案掩码>
#show figure: set block(breakable: false)

设置模型的训练超参数：

```python
batch_size = SFTConfig.batch_size
gradient_accumulation_steps = SFTConfig.gradient_accumulation_steps
log_iter = SFTConfig.log_iter
max_lr = SFTConfig.max_lr
min_lr = SFTConfig.min_lr
warmup_steps = SFTConfig.warmup_steps
total_steps = len(train_data) // batch_size
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
```

编写日志记录函数

#figure(
  ```python
  with open(f"log.txt", "a") as my_file:
      my_file.write(f" \
          time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, \
          batch_size:{batch_size}, \
          warmup_steps:{warmup_steps}, \
          max_lr:{max_lr}, \
          min_lr:{min_lr}\n")

  #定义一个日志记录函数
  def log_call(iters, iters_average_loss):
      with open(f"log.txt", "a") as my_file:
          my_file.write(f" \
              time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, \
              iters:{iters+1}, \
              iters_average_Loss:{iters_average_loss:.4f}\n")
  ```,
  caption: [日志相关配置],
) <日志配置>

主训练循环如下

#codly(header: [SFT训练循环])
```python
model.train()
training_losses = []
model.zero_grad()  # 训练开始时清空梯度
skipped_batches_count = 0

total_batches = len(train_data) // batch_size

for batch_idx in range(total_batches):
    ## ==================== 数据准备阶段 ====================

    # 获取当前批次的原始数据
    current_batch_sequences = train_data[
        batch_idx * batch_size : (batch_idx + 1) * batch_size
    ]

    # 计算当前批次的最大序列长度，用于padding对齐
    max_sequence_length = max([len(sequence) for sequence in current_batch_sequences])

    ### 对批次数据进行右填充，使所有序列长度一致以便并行计算
    padded_sequences_list = []
    ### 将<|endoftext|>设置为填充符
    pad_token_id = model.generation_config.eos_token_id[-1]

    for seq_idx in range(batch_size):
        # 原始的一条训练数据
        original_sequence = current_batch_sequences[seq_idx]
        # 要填充的长度
        padding_length = max_sequence_length - len(original_sequence)

        # 使用EOS token进行右填充
        padded_sequence = torch.nn.functional.pad(
            torch.tensor(original_sequence),
            (0, padding_length),
            mode="constant",
            value=pad_token_id
        ).tolist()

        padded_sequences_list.append(padded_sequence)

    # 转换为张量
    batch_input_tensor = torch.tensor(padded_sequences_list)

    ## ==================== 构建输入输出对 ====================

    # 构建因果语言模型的输入输出对：x->y（下一个词预测）
    model_inputs = batch_input_tensor[:, :-1].to(device)    # 输入：前n-1个token
    target_labels = batch_input_tensor[:, 1:].to(device)    # 标签：后n-1个token

    ## ==================== 构建训练掩码 ====================

    # 构建掩码矩阵来控制损失计算范围
    # 1. padding_mask：标识哪些位置是填充token（不计算损失）
    # 2. answer_mask：标识哪些位置是助手回答部分（只对回答计算损失）

    ### 【填充掩码】：非填充token为1，填充token为0
    ### padding_mask中的问题部分的掩码也是1
    padding_mask = torch.where(target_labels == pad_token_id, 0, 1)

    ### 【回答掩码】：只有助手回答部分为1，其他部分为0
    assistant_answer_mask = create_answer_mask(model_inputs, tokenizer)

    ### 【组合掩码】：同时满足"非填充"且"是回答部分"的token才计算损失
    ### 取出交集，就是真正要计算的回答部分
    final_loss_mask = (assistant_answer_mask & padding_mask)

    ## ==================== 批次有效性检查 ====================

    # 检查当前批次是否有效：如果某个样本的回答部分完全为空，则跳过该批次
    # 这种情况通常发生在问题过长导致回答部分被截断时
    tokens_per_sample = final_loss_mask.sum(dim=-1)  # 每个样本的有效回答token数
    min_answer_tokens = tokens_per_sample.min().item()  # 最少的有效token数

    if min_answer_tokens == 0:
        print(f"跳过第{batch_idx + 1}批次：回答部分数据不足")
        skipped_batches_count += 1
        continue  # 跳过当前批次

    ## ==================== 模型前向传播 ====================

    # 执行前向传播，获取模型预测的logits
    # [batch_size, seq_length, vocab_size]
    model_logits = model(model_inputs).logits

    ## ==================== 损失计算 ====================

    # 计算带掩码的交叉熵损失
    # 步骤：logits -> softmax -> log -> gather -> 负对数似然 -> 掩码过滤 -> 平均

    # 1. 计算每个token的负对数似然损失，
    # 形状：[batch_size, seq_len, vocab_size]
    log_probabilities = torch.log(torch.softmax(model_logits, dim=-1))
    # 使用真正的目标token取出vocab_size长度的数组中token对应的对数概率
    # 形状：[batch_size, seq_len]
    gathered_log_probs = torch.gather(
        log_probabilities,
        dim=-1,
        index=target_labels.unsqueeze(2)
    )
    negative_log_likelihood = gathered_log_probs * (-1)  # 负对数似然
    token_losses = negative_log_likelihood.squeeze(2)

    # 2. 应用掩码并计算每个样本的平均损失
    masked_token_losses = torch.mul(token_losses, final_loss_mask)
    sample_losses = masked_token_losses.sum(dim=-1) \
                  / final_loss_mask.sum(dim=-1)

    # 3. 计算批次平均损失并应用梯度累积
    batch_average_loss = torch.nanmean(sample_losses) \
                       / gradient_accumulation_steps

    ## ==================== 反向传播和优化 ====================

    # 反向传播计算梯度
    batch_average_loss.backward()

    # 动态调整学习率（余弦衰减 + 预热）
    current_learning_rate = cosine_decay(
        batch_idx,
        warmup_steps,
        total_steps,
        max_lr,
        min_lr
    )

    # 更新优化器的学习率
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_learning_rate

    # 梯度累积：只在累积步数达到或最后一个批次时更新权重
    is_accumulation_step = (batch_idx + 1) \
                         % gradient_accumulation_steps == 0
    is_final_batch = (batch_idx + 1) == total_batches

    if is_accumulation_step or is_final_batch:
        optimizer.step()        # 更新模型权重
        optimizer.zero_grad()   # 清空梯度缓存

    ## ==================== 训练日志记录 ====================

    # 记录当前批次的损失（还原梯度累积的缩放）
    actual_batch_loss =                   \
        batch_average_loss.item()         \
        *                                 \
        gradient_accumulation_steps
    training_losses.append(actual_batch_loss)

    # 定期输出训练进度
    should_log = (batch_idx + 1) % log_iter == 0 or is_final_batch

    if should_log:
        # 计算最近几个批次的平均损失
        recent_losses = training_losses[-log_iter:]
        recent_average_loss = np.nanmean(recent_losses)

        # 输出训练状态
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"时间: {current_time} | "
              f"批次: {batch_idx + 1}/{total_batches} | "
              f"最近{len(recent_losses)}批次平均损失: {recent_average_loss:.4f} | "
              f"学习率: {current_learning_rate:.2e}")

        # 调用外部日志记录函数
        log_call(batch_idx, recent_average_loss)

## ==================== 训练完成总结 ====================

print("训练完成!")
print(f"训练统计:")
print(f"   - 总批次数: {total_batches}")
print(f"   - 跳过批次数: {skipped_batches_count}")
print(f"   - 有效批次数: {total_batches - skipped_batches_count}")
print(f"   - 最终平均损失: {np.nanmean(training_losses[-100:]):.4f}")

if skipped_batches_count > 0:
    skip_ratio = skipped_batches_count / total_batches * 100
    print(f"跳过批次占比: {skip_ratio:.2f}%")
    if skip_ratio > 10:
        print("建议: 跳过批次过多，考虑增加最大序列长度或优化数据预处理")

model.save_pretrained("./Qwen3-0.6B-SFT/")
tokenizer.save_pretrained("./Qwen3-0.6B-SFT/")
```

#figure(
  image("rl-figures/只对答案部分计算损失示意图.svg"),
  caption: [我们是从`model_inputs`中寻找回答部分，然后计算预测的损失],
)

为什么要取交集？也就是代码```python final_loss_mask = (assistant_answer_mask & padding_mask)```为什么要这么写？

注意到```python padding_mask = torch.where(target_labels == pad_token_id, 0, 1)```是将`target_labels`中的填充符`pad_token`置为0。因为我们是*右*填充逻辑。

在绝大多数标准场景下，这个"取交集"的操作确实是冗余的，但保留它也无伤大雅，甚至是一种防御性编程的好习惯。

下面为你详细拆解为什么它是"冗余"的，以及为什么在某些边缘情况下它又是"必要"的。

1. 为什么理论上是"没必要"的？（冗余性分析）

SFT（有监督微调）的核心逻辑是：只计算 Assistant（助手）回答部分的 Loss。

我们来看看这两个掩码的定义：

- padding_mask：标记哪些是真实内容（1），哪些是填充内容（0）。
- assistant_answer_mask：标记哪些是助手的回答内容（1），其他都是 0。

在标准的*右填充（Right-padding）*数据处理流程中：

- 所有的 Padding 都在句子的最末尾。
- 所有的 Assistant 回答都在 Padding 之前。
- 结论：`assistant_answer_mask` 为 1 的区域，必然是真实内容，因此该区域的 `padding_mask` 必然也是 1。

数学上来说：$"Answer_Set" subset "Content_Set"$。

既然回答集合是内容集合的子集，那么 `Answer & Content` 自然就等于 Answer。所以取交集不会改变 `assistant_answer_mask` 本身的值。

2. 为什么保留它也是好的？（防御性编程）

虽然逻辑上冗余，但在工程实践中，保留 `& padding_mask` 有两个好处：

A. 防止掩码生成函数的Bug

`create_answer_mask`是一个复杂的逻辑函数（涉及查找`<|im_start|>`，`<|im_end|>`索引）。如果这个函数写得有瑕疵，比如：

- 索引计算错误，导致掩码划多了，划到了 Padding 区域。
- 数据集本身有脏数据，导致解析错位。

此时，`& padding_mask` 就像一道安全闸，强制把 Loss 计算限制在非 Padding 区域，防止模型去学习"预测 Padding Token"，避免模型变傻。

B. 处理截断（Truncation）的边缘情况

在代码中，我们使用了`truncation=True`。如果一条数据非常长，恰好在 Assistant 回答的一半被截断了，且后面紧接着就是 Padding（这种情况在我们的代码中不会出现！虽然在truncation后通常没有padding，但也取决于具体实现），双重掩码能确保万无一失。

=== 第二步：使用DPO算法对SFT后的模型进行微调

#danger[
  我们使用DPO算法要微调的模型是上一节中保存的"Qwen3-0.6B-SFT"。
]

加载我们要微调的模型。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import datasets
import torch
import numpy as np
import os

device = "cuda"
model_path = "./Qwen3-0.6B-SFT"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto"
)
# 冻结的参考模型
ref_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto"
)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

生成回答的配置和SFT时相同，见@模型生成的配置。

定义模型训练的超参数。

```python
@dataclass
class DPOConfig:
    max_length = 1700
    batch_size = 2
    gradient_accumulation_steps = 8
    beta = 0.5 # $beta$是DPO目标函数中的超参数
    log_iter = 100
    max_lr = 1e-6
    min_lr = 1e-7
    warmup_steps:int = 300
```

接下来导入*正负例数据集*。

```python
binarized_data = datasets.load_dataset("./ultrafeedback_binarized")
```

处理数据集

```python
def tokenize_and_format(data):
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize = True,
        add_generation_prompt = False,
        truncation = True,
        max_length = DPOConfig.max_length,
    )

    return input_ids

## 生成偏好数据的input_ids
chosen_input_ids_list = []
i = 0
while True:
    data = binarized_data["train_sft"][i]["chosen"]
    data.insert(
        0,
        {"content": "You are a helpful assistant", "role": "system"}
    )
    input_ids = tokenize_and_format(data)
    chosen_input_ids_list.append(input_ids)
    i += 1
    if i % 10000 == 0 or i == len(binarized_data["train_sft"]):
        print(f"偏好数据已处理{i}条数据")
    if i == 30000:
        break
print("-" * 70)

#############################################################################
## 生成不偏好数据的input_ids
rejected_input_ids_list = []
i = 0
while True:
    data = binarized_data["train_sft"][i]["rejected"]
    data.insert(
        0,
        {"content": "You are a helpful assistant", "role": "system"}
    )
    input_ids = tokenize_and_format(data)
    rejected_input_ids_list.append(input_ids)
    i += 1
    if i % 10000 == 0 or i == len(binarized_data["train_sft"]):
        print(f"非偏好数据已处理{i}条数据")
    if i == 30000:
        break

## 确保数据条数一致
assert len(chosen_input_ids_list) == len(rejected_input_ids_list)
```

设置一下训练的超参数

```python
beta = DPOConfig.beta # $beta$超参数
batch_size = DPOConfig.batch_size
gradient_accumulation_steps = DPOConfig.gradient_accumulation_steps
log_iter = DPOConfig.log_iter
max_lr = DPOConfig.max_lr
min_lr = DPOConfig.min_lr
warmup_steps = DPOConfig.warmup_steps
total_steps = len(chosen_input_ids_list) // batch_size
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
```

- 配置logging日志记录模型训练过程参见@日志配置。
- 复用SFT阶段设置的余弦衰减学习率曲线，参见@学习率调度。
- 将回答部分进行掩码的代码，参见@对答案掩码。

计算平均对数概率

```python
def _compute_average_log_probability(logits, target_labels, mask):
    """
    计算带掩码的平均对数概率

    Args:
        logits: 模型输出 [batch_size, seq_len, vocab_size]
        target_labels: 目标标签 [batch_size, seq_len]
        mask: 计算掩码 [batch_size, seq_len]

    Returns:
        average_log_prob: 每个样本的平均对数概率 [batch_size]
    """
    # 计算softmax概率分布
    probabilities = torch.softmax(logits, dim=-1)

    # 计算对数概率
    log_probabilities = torch.log(probabilities)

    # 获取目标token的对数概率：
    # $[log pi(y_0|x),log pi(y_1|x,y_0), log pi(y_2|x,y_(<2)),dots]$
    gathered_log_probs = torch.gather(
        log_probabilities,
        dim=-1,
        index=target_labels.unsqueeze(2)
    ).squeeze(2)

    # 应用掩码并计算平均值
    masked_log_probs = torch.mul(gathered_log_probs, mask)

    # $1/T (log pi(y_0|x) + log pi(y_1|x,y_0) + log pi(y_2|x,y_(<2)) + dots.c)$
    # $=1/T (sum_(t=0)^T log pi(y_t|x,y_(<t)))=log (product_(t=0)^T pi(y_t|x,y_(<t)))^(1/T)$
    # 这就是在提示词为x的条件下，生成一条回答y的概率。用强化学习的术语来说就是一条轨迹的概率。
    average_log_prob = masked_log_probs.sum(dim=-1) / mask.sum(dim=-1)

    return average_log_prob
```

DPO训练循环如下：

#codly(header: [DPO训练循环])
```python
model.train()

# ==================== 训练指标记录列表 ====================
training_losses = []
# 偏好的回答的概率
preferred_log_probabilities = []
# 讨厌的回答的概率
rejected_log_probabilities = []
# 偏好的回答的奖励
preferred_rewards = []
# 讨厌的回答的奖励
rejected_rewards = []
reward_margins = []

model.zero_grad()  # 训练开始时清空梯度
skipped_batches_count = 0
total_batches = len(chosen_input_ids_list) // batch_size

for batch_idx in range(total_batches):
    ## ==================== 获取批次数据 ====================

    # 获取当前批次的偏好对数据
    preferred_batch_sequences = chosen_input_ids_list[
        batch_idx * batch_size:(batch_idx + 1) * batch_size
    ]
    rejected_batch_sequences = rejected_input_ids_list[
        batch_idx * batch_size:(batch_idx + 1) * batch_size
    ]

    ## ==================== 数据填充对齐 ====================

    # 计算各自批次的最大序列长度
    preferred_max_length = max([len(sequence) for sequence in preferred_batch_sequences])
    rejected_max_length = max([len(sequence) for sequence in rejected_batch_sequences])
    # 使用eos token作为pad token
    pad_token_id = model.generation_config.eos_token_id[-1]

    ### 偏好数据填充处理
    preferred_padded_sequences = []
    for seq_idx in range(batch_size):
        original_sequence = preferred_batch_sequences[seq_idx]
        # 计算要填充多少个pad
        padding_length = preferred_max_length - len(original_sequence)
        # 在训练数据的末尾填充pad
        padded_sequence = torch.nn.functional.pad(
            torch.tensor(original_sequence),
            (0, padding_length),
            mode="constant",
            value=pad_token_id
        ).tolist()
        # 将填充过的数据放入列表
        preferred_padded_sequences.append(padded_sequence)

    preferred_batch_tensor = torch.tensor(preferred_padded_sequences)

    ### 拒绝数据填充处理
    rejected_padded_sequences = []
    for seq_idx in range(batch_size):
        original_sequence = rejected_batch_sequences[seq_idx]
        padding_length = rejected_max_length - len(original_sequence)

        padded_sequence = torch.nn.functional.pad(
            torch.tensor(original_sequence),
            (0, padding_length),
            mode="constant",
            value=pad_token_id
        ).tolist()

        rejected_padded_sequences.append(padded_sequence)

    rejected_batch_tensor = torch.tensor(rejected_padded_sequences)

    ## ==================== 构建输入输出对 ====================

    # 构建因果语言模型的输入输出对：x->y（下一个词预测）
    # 模型的输入：偏好的回答
    preferred_model_inputs = preferred_batch_tensor[:, :-1].to(device)
    # 真实的标签$y_w$
    preferred_target_labels = preferred_batch_tensor[:, 1:].to(device)

    rejected_model_inputs = rejected_batch_tensor[:, :-1].to(device)
    # 真实的标签$y_l$
    rejected_target_labels = rejected_batch_tensor[:, 1:].to(device)

    ## ==================== 构建训练掩码 ====================

    # 构建掩码矩阵：padding_mask（忽略填充token）+ answer_mask（只关注回答部分）

    # pad_token_id 对应的置为 0 ，其它置为 1 。
    preferred_padding_mask = torch.where(
        preferred_target_labels == pad_token_id,
        0,
        1
    )
    rejected_padding_mask = torch.where(
        rejected_target_labels == pad_token_id,
        0,
        1
    )

    # 助手回答的掩码：将助手回答的部分掩码为 1 。其它都是 0 。
    preferred_answer_mask = create_answer_mask(
        preferred_model_inputs,
        tokenizer
    )
    rejected_answer_mask = create_answer_mask(
        rejected_model_inputs,
        tokenizer
    )

    # 最终掩码：取交集
    preferred_final_mask = (preferred_answer_mask & preferred_padding_mask)
    rejected_final_mask = (rejected_answer_mask & rejected_padding_mask)

    ## ==================== 批次有效性检查 ====================

    # 检查偏好对数据是否都有有效的回答部分
    preferred_min_tokens = preferred_final_mask.sum(dim=-1).min().item()
    rejected_min_tokens = rejected_final_mask.sum(dim=-1).min().item()

    if preferred_min_tokens == 0 or rejected_min_tokens == 0:
        print(f"跳过第{batch_idx + 1}批次：偏好对数据回答部分不足")
        skipped_batches_count += 1
        continue  # 跳过当前批次

    ## ==================== 模型前向传播 ====================

    # 训练模型对偏好数据的前向传播
    preferred_logits = model(preferred_model_inputs).logits
    torch.cuda.empty_cache()  # 清理GPU显存
    torch.cuda.ipc_collect()

    # 训练模型对拒绝数据的前向传播
    rejected_logits = model(rejected_model_inputs).logits
    torch.cuda.empty_cache()  # 清理GPU显存
    torch.cuda.ipc_collect()

    # 参考模型的前向传播（不计算梯度）
    with torch.no_grad():
        reference_preferred_logits = ref_model(preferred_model_inputs) \
            .logits                                                    \
            .detach()
        reference_rejected_logits = ref_model(rejected_model_inputs)   \
            .logits                                                    \
            .detach()

    ## ==================== DPO损失计算 ====================
    """
    DPO (Direct Preference Optimization) 论文: https://arxiv.org/pdf/2305.18290.pdf
    核心思想：通过偏好对比学习，无需显式奖励模型
    """

    # 计算平均对数概率 (average_log_prob = True)
    # 参考: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L924

    ### 训练模型的对数概率
    ### 正在微调的模型，接收到正例的logits，计算对数概率
    ### $log pi_theta (y_w|x)$
    preferred_log_prob = _compute_average_log_probability(
        preferred_logits,
        preferred_target_labels,
        preferred_final_mask
    )
    ### $log pi_theta (y_l|x)$
    rejected_log_prob = _compute_average_log_probability(
        rejected_logits,
        rejected_target_labels,
        rejected_final_mask
    )

    ### 参考模型的对数概率
    ### $log pi_"ref" (y_w|x)$
    reference_preferred_log_prob = _compute_average_log_probability(
        reference_preferred_logits,
        preferred_target_labels,
        preferred_final_mask
    )
    ### $log pi_"ref" (y_l|x)$
    reference_rejected_log_prob = _compute_average_log_probability(
        reference_rejected_logits,
        rejected_target_labels,
        rejected_final_mask
    )

    # $beta log (pi_theta (y_w|x))/(pi_"ref" (y_w|x))$
    preferred_implicit_reward =                              \
        beta *                                               \
        (preferred_log_prob - reference_preferred_log_prob)
    # $beta log (pi_theta (y_l|x))/(pi_"ref" (y_l|x))$
    rejected_implicit_reward =                               \
        beta *                                               \
        (rejected_log_prob - reference_rejected_log_prob)

    # $beta(log (pi_theta (y_w|x))/(pi_"ref" (y_w|x)) - log (pi_theta (y_l|x))/(pi_"ref" (y_l|x)))$
    reward_margin = preferred_implicit_reward - rejected_implicit_reward

    # DPO损失：
    # $-log(sigma(beta(log (pi_theta (y_w|x))/(pi_"ref" (y_w|x)) - log (pi_theta (y_l|x))/(pi_"ref" (y_l|x)))))$
    preference_probability = torch.nn.functional.sigmoid(reward_margin)
    sample_losses = -torch.log(preference_probability)

    # 批次平均损失 + 梯度累积
    batch_average_loss =                          \
        torch.nanmean(sample_losses) /            \
        gradient_accumulation_steps

    ## ==================== 反向传播和优化 ====================

    batch_average_loss.backward()

    # 动态学习率调整
    current_learning_rate = cosine_decay(
        batch_idx,
        warmup_steps,
        total_steps,
        max_lr,
        min_lr
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_learning_rate

    # 梯度累积和权重更新
    is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
    is_final_batch = (batch_idx + 1) == total_batches

    if is_accumulation_step or is_final_batch:
        optimizer.step()        # 更新权重
        optimizer.zero_grad()   # 清空梯度

    ## ==================== 训练指标记录 ====================

    # 记录各项训练指标（detach避免梯度追踪）
    training_losses.append(
        batch_average_loss.detach().item() * gradient_accumulation_steps)
    preferred_log_probabilities.append(
        torch.nanmean(preferred_log_prob.detach()).item())
    rejected_log_probabilities.append(
        torch.nanmean(rejected_log_prob.detach()).item())
    preferred_rewards.append(
        torch.nanmean(preferred_implicit_reward.detach()).item())
    rejected_rewards.append(torch.nanmean(
        rejected_implicit_reward.detach()).item())
    reward_margins.append(
        torch.nanmean(reward_margin.detach()).item())

    ## ==================== 训练日志输出 ====================

    should_log = (batch_idx + 1) % log_iter == 0 or is_final_batch

    if should_log:
        # 计算最近批次的平均指标
        recent_loss = np.nanmean(training_losses[-log_iter:])
        recent_preferred_logprob = np.nanmean(
            preferred_log_probabilities[-log_iter:])
        recent_rejected_logprob = np.nanmean(
            rejected_log_probabilities[-log_iter:])
        recent_preferred_reward = np.nanmean(preferred_rewards[-log_iter:])
        recent_rejected_reward = np.nanmean(rejected_rewards[-log_iter:])
        recent_margin = np.nanmean(reward_margins[-log_iter:])

        # 格式化输出训练状态
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"时间: {current_time}")
        print(f"批次: {batch_idx + 1}/{total_batches}")
        print(f"最近{log_iter}批次指标：")
        print(f"   - 平均损失: {recent_loss:.4f}")
        print(f"   - 偏好对数概率: {recent_preferred_logprob:.4f}")
        print(f"   - 拒绝对数概率: {recent_rejected_logprob:.4f}")
        print(f"   - 偏好奖励: {recent_preferred_reward:.4f}")
        print(f"   - 拒绝奖励: {recent_rejected_reward:.4f}")
        print(f"   - 奖励边际: {recent_margin:.4f}")
        print(f"学习率: {current_learning_rate:.2e}")
        print("-" * 80)

        # 调用外部日志记录
        log_call(batch_idx, recent_loss)

## ==================== 训练完成总结 ====================

print("DPO训练完成！")
print(f"训练统计：")
print(f"   - 总批次数: {total_batches}")
print(f"   - 跳过批次数: {skipped_batches_count}")
print(f"   - 有效批次数: {total_batches - skipped_batches_count}")

# 输出最终训练指标
if training_losses:
    final_metrics = {
        "loss": np.nanmean(training_losses[-100:]),
        "preferred_logprob": np.nanmean(preferred_log_probabilities[-100:]),
        "rejected_logprob": np.nanmean(rejected_log_probabilities[-100:]),
        "preferred_reward": np.nanmean(preferred_rewards[-100:]),
        "rejected_reward": np.nanmean(rejected_rewards[-100:]),
        "margin": np.nanmean(reward_margins[-100:])
    }

    print(f"最终指标 (最近100批次平均)：")
    for metric_name, metric_value in final_metrics.items():
        print(f"   - {metric_name}: {metric_value:.4f}")

if skipped_batches_count > 0:
    skip_ratio = skipped_batches_count / total_batches * 100
    print(f"跳过批次占比: {skip_ratio:.2f}%")
    if skip_ratio > 10:
        print("建议: 跳过批次过多，考虑增加最大序列长度或优化数据预处理")
```

#chapter("使用PPO微调大语言模型--复刻InstructGPT", image: image("./orange2.jpg"), l: "rlhf-instruct-gpt")

== InstructGPT训练流程

- Step-1：SFT，Supervised Fine-Tuning，有监督微调。顾名思义，它是在有监督（有标注）数据上微调训练得到的。这里的监督数据其实就是输入Prompt，输出相应的回复，只不过这里的回复是人工编写的。这个工作要求比一般标注要高，其实算是一种创作了。
- Step-2：RM，Reward Model，奖励模型。具体来说，一个Prompt丢给前一步的SFT，输出若干个（4-9个）回复，由标注人员对这些回复进行排序。然后从4-9个中每次取2个，因为是有序的，就可以用来训练这个奖励模型，让模型学习到这个好坏评价。这一步非常关键，它就是所谓的Human Feedback，引导下一步模型的进化方向。
- Step-3：RL，Reinforcement Learning，强化学习，使用PPO策略进行训练。PPO，Proximal Policy Optimization，近端策略优化，是一种强化学习优化方法，它背后的主要思想是避免每次太大的更新，提高训练的稳定性。具体过程如下：首先需要初始化一个语言模型，然后丢给它一个Prompt，它生成一个回复，上一步的奖励模型给这个回复一个打分，这个打分回传给模型更新参数。这里的这个模型在强化学习视角下就是一个策略。这一步有个很重要的动作，就是更新模型时会考虑模型每一个token的输出和第一步SFT输出之间的差异性，要让它俩尽量相似。这是为了缓解强化学习可能的过度优化。

#figure(
  image("rl-figures/instruct-gpt.png"),
  caption: [InstructGPT训练流程],
)

== 即时奖励$R_t$的计算

$
  J_"PPO" (theta)=EE[min((pi_theta (a|s))/(pi_theta_"old" (a|s))A, "clip"((pi_theta (a|s))/(pi_theta_"old" (a|s)),1-epsilon,1+epsilon)A)]
$

PPO的目标函数我们已经很熟悉了，优势$A$如果使用1步TD误差的话是：$delta=R_t+gamma V(S_(t+1)) - V(S_t)$。

在倒立摆环境中，只要木杆不倒下，那么$R_t=1$。但是在大语言模型这个环境中，情况就要复杂多了。在大语言模型中，策略模型LLM采取的动作是输出一个token，那么输出一个token，我们应该给什么奖励$R_t$呢？

#figure(
  image("rl-figures/输出的每个token的奖励怎么给.svg"),
  caption: [输出一个token（采取动作），针对输出的这个token，怎么给即时奖励$R_t$？],
)

在RLHF中，奖励模型（Reward Model）的作用是针对一条完整的补全给出分数的。

#figure(
  image("rl-figures/奖励模型原理.svg"),
  caption: [奖励模型是针对一条完整的补全给出得分的],
)

在 InstructGPT (Ouyang et al., 2022) 的 PPO 阶段，奖励函数的设计非常精妙。它不仅仅是奖励模型（Reward Model, RM）给出的分数，还包含了一个至关重要的惩罚项。

即时奖励的计算公式分两种情况：

- `<eos_token>`之前的token的奖励如下计算：

$
  R_t = -beta log (pi_theta (y_t|x,y_(<t)))/(pi_"ref" (y_t|x,y_(<t)))
$

- 最后一个token也就是`<eos_token>`的奖励如下：

$
  R_"<eos_token>" = R_T = "奖励模型给的得分" - beta log (pi_theta (y_T|x,y_(<T)))/(pi_"ref" (y_T|x,y_(<T)))
$

InstructGPT成功的关键秘诀是式子中的KL散度惩罚项$beta log (pi_theta (y_t|x,y_(<t)))/(pi_"ref" (y_t|x,y_(<t)))$。

- 直观理解：我们不希望强化学习后的模型$pi_theta$偏离原始的SFT模型$pi_"ref"$太远。
- 为什么需要KL散度惩罚项？
  - 防止Reward Hacking（奖励黑客）：如果没有约束，RL模型会利用奖励模型的漏洞，生成一些人类读起来不通顺、但奖励模型误判为高分的奇怪句子（例如重复乱码）。
  - 保持语言流畅性：SFT 模型通常能生成通顺的自然语言。通过锚定 SFT 模型，保证了 RL 模型生成的句子依然是"人话"。
- 计算细节：这个惩罚是*逐token计算*的，也就是如果$pi_theta$对某个token的生成概率远高于$pi_"ref"$，说明模型在"冒险"偏离原来的轨道，惩罚就会变大。
- 奖励模型的打分鼓励模型往"人类喜欢"的方向走。
- KL惩罚拽住模型，不让它跑得太偏，避免它为了高分而变成"怪兽"。

在标准的InstructGPT实现中，*奖励模型（RM）的分数通常只加在最后一个 token 上*，而 *KL惩罚是加在每一个 token 上的*。

让我们把这个过程像切蛋糕一样切开来看：

1. 奖励的时间步分配

假设模型生成了一个长度为 $T$ 的句子：$y = [y_1, y_2, ..., y_T]$。

在 PPO 的每一个时间步 $t$，智能体获得的即时奖励 $R_t$ 是这样计算的：

#table(
  columns: 3,
  [时间步 (token)], [即时奖励 $R_t$ 的构成], [解释],
  [中间token($t < T$)],
  [只有KL惩罚 \ $R_t = -beta log (pi_theta (y_t|x,y_(<t)))/(pi_"ref" (y_t|y_(<t)))$],
  [此时句子还没写完，奖励模型无法打分。我们只关心这一步有没有"偏离初心"（KL散度）。],

  [最后一个token($t = T$)],
  [KL惩罚+RM总分
    #math.equation(
      $
        R_T = & -beta log (pi_theta (y_t|x,y_(<t)))/(pi_"ref" (y_t|y_(<t))) \
              & + "奖励模型给的分数"
      $,
      numbering: none,
      block: true,
    )],
  [句子结束（遇到`<eos_token>`），奖励模型RM终于看完了整句话，给出一个得分，叠加在最后一步上。],
)

2. 为什么要这样做？

你可能会问："如果只奖励最后一个词，前面的词怎么知道自己做得好不好？"

虽然物理上的奖励（奖励模型给的得分）只发生在最后一步，但 PPO 使用了广义优势估计（GAE）来回传这个信号：

1. *价值函数的预测*：Critic 网络（价值网络）会预测每一个 token 的 $V(S_t)$。它会学到："虽然我现在只拿到了微小的 KL 惩罚，但我知道只要我顺着这个方向走，最后能拿到巨大的*奖励模型给的得分*。"
2. *信号回传*：通过折扣因子 $gamma$ 和 GAE 参数 $lambda$，最后一步的巨大奖励会沿着时间轴*向前传播*。
  - 如果奖励模型给的得分很高，那么 $y_(T-1), y_(T-2)...$ 的优势函数（Advantage）都会变高。
  - 这就告诉了前面的 token："你们铺垫得很好，导致最后结局很棒。"
3. 举个具体的例子

- Prompt: "你好"
- Response: "你好呀`<eos_token>`" (4个token)

假设 $beta=0.1$，奖励模型RM给整句打分 $1.0$。

- token 1 ("你"):
  - $R_1 = -0.1 times "KL"("你")$
  - （没有 RM 分数）
- token 2 ("好"):
  - $R_2 = -0.1 times "KL"("好")$
  - （没有 RM 分数）
- token 3 ("呀"):
  - $R_3 = -0.1 times "KL"("呀")$
  - （没有 RM 分数）
- token 4 ("`<eos_token>`"):
  - $R_4 = -0.1 times "KL"("<eos_token>") + 1.0$
  - （这里加上了奖励模型RM的最终大奖）

💡 总结

- 奖励模型RM给的分数是"稀疏奖励"（Sparse Reward）：只在结局出现一次。
- KL惩罚是"稠密奖励"（Dense Reward）：每一步都有，时刻约束模型不要乱说话。
- 依靠GAE：把最后的稀疏奖励"抹匀"分摊给前面的每一个动作，让前面的词也能获得梯度更新。

== PPO微调LLM实战

应用场景：在电商场景中，很多商品由于没有人买而变成了"长尾商品"。而我们需要为这些长尾商品"自动"生成一些*正向*评论，来吸引买家。所以需要训练一个能够编写正向评论的LLM。

+ 首先对一个预训练LLM（gpt2, Qwen2.5-0.5B等）进行SFT，让LLM可以编写商品评论。这里的问题在于这个LLM虽然能够写电商评论，但写出来的可能是正向评论也可能是负向评论。
+ 训练一个奖励模型（Reward Model），可以针对正向评论打高分，针对负向评论打低分。
+ 使用奖励模型给SFT后的LLM输出的电商评论打分，如果分数高（说明是正向评论），则鼓励SFT后的LLM输出这条评论（提升输出这条评论的概率）。如果分数低（说明是负向评论），则抑制SFT模型输出这条评论的概率。

=== 监督微调（SFT）

我们的目标是训练一个可以写电商评论的大语言模型。所以我们对gpt2进行微调，让它可以写电商评论。

由于我们使用的文本，既有正向情感的文本，也有负向情感的文本，所以训练出来的模型，在给定提示词（例如："这本书"）时，可能写出正向情感的评论也可能写出负向情感的评论。

#tip(title: [预训练模型])[
  我们使用的底座模型是中文版gpt2："gpt2-chinese-cluecorpussmall"。
]

#codly(header: [对gpt2进行SFT])
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, pipeline, set_seed
from torch.utils.data import DataLoader
from datasets import load_dataset
from pprint import pprint

dataset = load_dataset("csv", data_files="online_shopping_10_cats.csv")
model_path = "./gpt2-chinese-cluecorpussmall"

ds_train = dataset["train"]
# 将评论长度为大于1024的过滤掉，因为gpt2的最大上下文长度为1024。
ds_train = ds_train.filter(lambda x : x["review"] != None and len(x["review"]) > 20 and len(x["review"]) < 1024)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def tokenize(batch):
    return tokenizer(batch["review"])

map_kwargs = {
    "batched": True,
    "batch_size": 512,
    "remove_columns": ["cat", "label", "review"]
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)

tokenized_dataset_train.set_format(type="torch")
# 将eos_token设置为pad_token
tokenizer.eos_token = tokenizer.pad_token
# 将mlm设置为False，那么数据会被整理为因果注意力模型使用的格式，
# 也就是预测下一个token任务需要的数据格式
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False
)

dataloader_params = {
    "batch_size": 2,
    "collate_fn": data_collator
}

train_dataloader = DataLoader(
    tokenized_dataset_train,
    **dataloader_params
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# 一般sft会训练1个epoch，也就是把训练数据看一遍就可以了
# 否则容易过拟合，造成灾难性遗忘
num_epochs = 1

device = torch.device("cuda")
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Step: {i}, Loss: {loss.item()}")

model.save_pretrained("./gpt2-sft")
tokenizer.save_pretrained("./gpt2-sft")

# 测试微调后的模型
g = pipeline("text-generation", model="./gpt2-sft")
set_seed(42)
pprint(g("这本书真是", max_length=30, num_return_sequences=10))
```

=== 训练奖励模型（Reward Model）

我们会将gpt2预训练模型微调成一个分类模型（奖励模型）。

我们要训练一个模型作为奖励模型，也就是给模型输入文本，模型可以给一个评分出来。

#figure(
  image("rl-figures/奖励模型训练流程示意图.svg"),
  caption: [奖励模型训练流程示意图],
)

我们的评论在输入奖励模型之前会先在末尾添加一个`reward_token`，作为标记。作用和Bert用来训练分类模型时添加的`CLS_TOKEN`是一样的。

我们将评论输入gpt2模型，然后提取gpt2输出的最后一层隐藏层（`last_hidden_state`），并送入一个我们自己定义的线性层（`reward_head`），然后取输出的最后一个元素作为评分。

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

model_path = "./gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.eos_token = tokenizer.pad_token
# reward_token设置为eos_token
REWARD_TOKEN_ID = tokenizer.eos_token_id

ds = load_dataset("csv", data_files="online_shopping_10_cats.csv")
ds_train = ds["train"]

ds_train = ds_train.filter(lambda x : x["review"] != None and len(x["review"]) > 20 and len(x["review"]) < 1024)

def tokenize(batch):
    # 提取出文本内容
    outputs = tokenizer(batch["review"])
    # 每条数据一个评分，初始化为 0 。
    outputs["score"] = [0] * len(outputs["input_ids"])
    # 对每条数据的最后的reward_token进行评分
    outputs["score_index"] = [0] * len(outputs["input_ids"])
    for i in range(len(outputs["input_ids"])):
        # 第 i 条数据的末尾添加一个eos_token，作为reward_token
        outputs["input_ids"][i].append(REWARD_TOKEN_ID)
        # reward_token的掩码设置为1。
        outputs["attention_mask"][i].append(1)
        # 正向情感的文本评分为1。负向情感的评分为0。
        # 也就是使用数据集中的标签作为评分。
        outputs["score"][i] = float(batch["label"][i])
        # 对 reward_token 进行评分，也就是评分的索引为 reward_token 的索引。
        outputs["score_index"][i] = len(outputs["input_ids"][i]) - 1
    return outputs

map_kwargs = {
    "batched": True,
    "batch_size": 512,
    "remove_columns": ["cat", "label", "review"]
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)

tokenized_dataset_train.set_format(type="torch")

class RewardModel(nn.Module):
    """奖励模型的结构"""
    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # gpt2的前向传播，但是还要输出隐藏层
        transformer_outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # 提取最后一层隐藏层
        last_hidden_state = transformer_outputs.hidden_states[-1]
        # 使用reward_head线性层对last_hidden_state给出奖励
        reward = self.reward_head(last_hidden_state).squeeze(-1)
        # sigmoid用来将奖励搞到(0,1)范围内
        return torch.sigmoid(reward)

model = RewardModel(model_path)

data_collator = DataCollatorWithPadding(tokenizer)

dataloader_params = {
    "batch_size": 16,
    "shuffle": True,
    "collate_fn": data_collator
}

train_dataloader = DataLoader(
    tokenized_dataset_train,
    **dataloader_params
)

device = torch.device("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# 二分类交叉熵损失
criterion = nn.BCELoss()
# 这里训练1轮，遵循InstructGPT论文中的训练设置。
num_epochs = 1 # N+ Implementation Detail paper

model.to(device)

for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        inputs = batch.to(device)
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        # 模型针对训练数据的打分
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0])
        # 模型对reward_token的打分
        score = scores[batch_indices, inputs["score_index"]]
        # 真实分数：0或者1
        target = inputs["score"]
        loss = criterion(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Loss: ", loss.item())

torch.save(model.state_dict(), "reward_model.pt")

#########################################
############ 对模型进行评估 #############
#########################################

model.eval()

all_predictions = []
all_labels = []

for i, batch in enumerate(train_dataloader):
    inputs = batch.to(device)
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    with torch.no_grad():
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0])
        score = scores[batch_indices, inputs["score_index"]]
        target = inputs["score"]
    # 对评论的打分大于0.5分，视为预测为1。
    predictions = (score > 0.5).int()

    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(target.cpu().numpy())

# 计算混淆矩阵
print(confusion_matrix(all_labels, all_predictions))
```

结果的格式如下：

```python
混淆矩阵:
[[364,  60],   # 第一行：真实标签为0的情况：364个预测正确，60个预测错误
 [ 31, 412]]   # 第二行：真实标签为1的情况：31个预测错误，412个预测正确
```

=== 对gpt2-sft进行PPO微调

==== 导入依赖

```python
from transformers import DataCollatorWithPadding, AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from torch.utils.data import DataLoader
import random
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

model_path = "./gpt2-sft"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
```

==== 准备提示词

我们需要搞一些提示词，给到gpt2-sft，让它补全。补全的内容交给奖励模型评分。然后决定惩罚gpt2-sft还是奖励gpt2-sft。

#danger[
  自己编写prompt太麻烦了，所以我们准备提示词的方式，是把数据集中文本的前几个字截取出来作为提示词。*后面的文本不再使用*！
  #tip(title: [截取前几个字作为提示词])[
    #underline[东西]#text(fill: gray)[太小了，光有数量没有质量！很失望！]
  ]
]

#codly(header: [为PPO准备提示词])
```python
ds = load_dataset("csv", data_files="online_shopping_10_cats.csv")
ds_train = ds["train"]

ds_train = ds_train.filter(lambda x: x["review"] != None and len(
    x["review"]) > 20 and len(x["review"]) < 1024)

# 截取评论数据的前2～8个token作为提示词
input_min_token_length = 2
input_max_token_length = 8
input_token_length_range = list(range(
    input_min_token_length,
    input_max_token_length))

def tokenize(sample):
    # 提示词token的数量随机选择一个
    input_size = random.choice(input_token_length_range)
    # 如果input_size=3，截取review字段文本的前3个token出来
    sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size]
    # 提示词掩码为1
    sample["attention_mask"] = [1] * len(sample["input_ids"])
    # 提示词对应的文本
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


map_kwargs = {
    "batched": False,
    "remove_columns": ["cat", "review", "label"]
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)

tokenized_dataset_train.set_format(type="torch")

batch_size = 32

def collator(batch):
    return dict((key, [d[key] for d in batch]) for key in batch[0])

# 提示词组成的数据集
train_dataloader = DataLoader(
    tokenized_dataset_train,
    batch_size=batch_size,
    collate_fn=collator,
    shuffle=True
)
```

==== ActorCritic模型结构

在倒立摆环境中，我们的Actor是`PolicyNet`，Critic是`ValueNet`。

在InstructGPT中，演员模型和价值模型合并成了一个模型`ActorCriticModel`。这样actor和critic可以共享gpt2-sft模型的权重。

#codly(header: [演员评论家模型])
```python
class ActorCriticModel(nn.Module):
    """GPT2模型+价值头"""
    def __init__(self, model_path):
        super().__init__()
        # 这个要初始化为我们微调出来的gpt2-sft模型
        # actor演员模型：策略模型$pi_theta$
        self.llm = AutoModelForCausalLM.from_pretrained(model_path)
        # 添加价值头$V_omega$
        # critic评论家模型：价值函数模型，价值头，线性层
        self.v_head = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # gpt2-sft模型的输出
        transformer_outputs = self.llm.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states = True,
        )
        # 输出的token的logits
        logits = transformer_outputs.logits
        # 获取最后一层隐藏层
        last_hidden_state = transformer_outputs.hidden_states[-1]

        # 评估token的价值，评估的是最后一个隐藏层的价值
        value = self.v_head(last_hidden_state).squeeze(-1)
        # 返回输出的token的logits和token的价值
        return logits, value

    def generate(self, *args, **kwargs):
        """生成文本"""
        return self.llm.generate(*args, **kwargs)
```

==== 加载模型

#codly(header: [将需要的模型都加载到显存])
```python
device = torch.device("cuda")

# 将奖励模型加载
reward_model = RewardModel(model_path)
reward_model.load_state_dict(torch.load(
    "reward_model.pt",
    map_location="cpu"))
# 加载演员评论家模型
model = ActorCriticModel(model_path)
model = model.to(device)
reward_model = reward_model.to(device)
# 加载冻结的参考模型$pi_"ref"$，只用来计算即时奖励$R_t$
ref_model = deepcopy(model)

# 设置reward_token
REWARD_TOKEN_ID = tokenizer.eos_token_id
```

==== 奖励$R_t$的计算

#figure(
  image("rl-figures/提示词不计算奖励.svg"),
  caption: [提示词为"这本书"，补全为"真好看"的$R_t$的计算],
)

下面的函数`compute_rewards`在计算输出的*每个*token的$R_t$的同时，*顺便*计算了每个token的价值，也就是$V(S_t)$。

```python
def compute_rewards(
    input_data, # 输入数据（提示词+补全）
    query_tensors, # 提示词张量
    response_tensors, # 补全的张量
    score_tensors # 奖励模型给出的分数的张量
):
    with torch.no_grad():
        # 正在微调的模型所输出的token的logits和token的价值（$[V(S_0),V(S_1),...,V(S_T)]$）
        logits, values = model(**input_data) # (B, T, vocab_size)
        # 冻结的参考模型的输出
        ref_logits, _ = ref_model(**input_data)
        # 正在微调的模型的输出的对数概率log_softmax
        # 去掉最后一个token，因为是预测下一个token的任务
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        # 冻结的模型的输出的对数概率
        ref_logp = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
        # 实际生成的token序列
        # 自回归模型是预测下一个token，所以去掉第一个token
        labels = input_data["input_ids"][:, 1:] # (B, T-1)
        # 使用gather提取实际token的概率
        # logp 是 vocab_size 大小的张量
        logp = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1) # (B, T-1)
        ref_logp = torch.gather(ref_logp, 2, labels.unsqueeze(-1)).squeeze(-1) # (B, T-1)
        # KL散度：$[log (pi_theta (y_0|x))/(pi_"ref" (y_0|x)), log (pi_theta (y_1|x,y_0))/(pi_"ref" (y_1|x,y_0)),...]$
        kl = logp - ref_logp
        # kl散度的权重$beta$
        beta = 0.2
        # 最终奖励的计算的右边式子：$[-beta log (pi_theta (y_0|x))/(pi_"ref" (y_0|x)), -beta log (pi_theta (y_1|x,y_0))/(pi_"ref" (y_1|x,y_0)),...]$
        rewards = - beta * kl
        attention_mask = input_data["attention_mask"]
        # 预测下一个token，所以去掉第一个mask
        masks = torch.zeros_like(attention_mask[:, 1:])
        masks[:,:] = attention_mask[:, 1:]
        # 遍历批次中的每一个提示词张量
        for j in range(len(query_tensors)):
            # 补全开始的索引
            start = len(query_tensors[j]) - 1
            # 补全结束的索引
            end = start + len(response_tensors[j])
            # 提示词部分掩码为0
            masks[j, :start] = 0
            # 补全后面的填充token掩码为0
            masks[j, end:] = 0
            # 将奖励模型给出的分数加到补全的最后一个token的奖励上面，得到
            # $[-beta log (pi_theta (y_0|x))/(pi_"ref" (y_0|x)), -beta log (pi_theta (y_1|x,y_0))/(pi_"ref" (y_1|x,y_0)),...,"score"-beta log (pi_theta (y_T|x,y_(<T)))/(pi_"ref" (y_T|x,y_(<T)))]$
            rewards[j, end - 1] += score_tensors[j]
            # 只留下掩码为1的部分的奖励
            rewards[j, :] *= masks[j, :]
            # 只留下掩码为1的部分的价值
            # 注意最后一个token的价值$V(S_"end")$去掉！
            values[j, :-1] *= masks[j, :]

    return logp, rewards, values[:, :-1], masks
```

`values[j, :-1] *= masks[j, :]`是为什么呢？

#figure(
  image("rl-figures/每个token的价值.svg"),
  caption: [去掉最后一个的价值],
)

==== 广义优势估计（GAE）的计算

```python
def masked_mean(values, mask):
    # 计算带掩码的平均值
    return (values * mask).sum() / mask.sum()

def masked_var(values, mask):
    # 计算带掩码的方差
    mean = masked_mean(values, mask)
    centred_values = values - mean
    return masked_mean(centred_values ** 2, mask)

def masked_whiten(values, mask):
    """对数据进行带掩码的白化处理，让有效数据的方差变为1，但均值保持不变"""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    whitened += mean
    return whitened

def compute_advantage(rewards, values, masks):
    """广义优势估计（GAE）"""
    lastgae = 0.0
    advantage_reversed = []
    # 补全的token数量
    seq_length = rewards.shape[-1]
    # 折扣因子$gamma=1$，指数加权超参数$lambda=0.95$
    gamma, lam = 1.0, 0.95

    for t in reversed(range(seq_length)): # 逆序计算
        # $V(S_(t+1))$
        nextvalues = values[:, t + 1] if t < seq_length - 1 else 0.0
        # $delta_t = R_t + gamma V(S_(t+1)) - V(S_t)$
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        # $A_t^"GAE" = delta_t + gamma lambda A_(t+1)^"GAE"$
        lastgae = delta + gamma * lam * lastgae
        advantage_reversed.append(lastgae)
    # $[A_0^"GAE",A_1^"GAE",A_2^"GAE",dots]$
    advantages = torch.stack(advantage_reversed[::-1], dim=1)
    # 对广义优势估计进行了白化处理
    advantages = masked_whiten(advantages, masks)
    # $"GAETarget"_t=A_t^"GAE"+V(S_t)$
    gae_targets = advantages + values
    return advantages, gae_targets
```

==== 损失的计算

```python
def compute_loss(
    old_logprobs, # 旧策略输出的概率$log pi_(theta_"old") (a_t|s_t)$
    logprobs, # 正在微调的模型输出的对数概率$log pi_theta (a_t|s_t)$
    vpreds, # 价值 $V(S_t)$
    masks, # 掩码
    advantages, # 广义优势估计$A_t^"GAE"$
    gae_targets # gae目标$A_t^"GAE"+V(S_t)$
):
    ratio = torch.exp(logprobs - old_logprobs) # $(pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t))$
    pg_loss1 = - ratio * advantages # $-(pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t)) A_t^"GAE"$
    # $-"clip"((pi_theta (a_t|s_t))/(pi_(theta_"old") (a_t|s_t)),1-epsilon,1+epsilon )A_t^"GAE"$
    pg_loss2 = - torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
    # 求掩码部分的平均值
    pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), masks)
    # $"MSELoss"(V(S_t), "GAETarget") arrow 0$
    v_loss = masked_mean((vpreds - gae_targets) ** 2, masks)
    # $"total"_"loss"="loss"_(pi_theta)+0.1 times "loss"_(V_omega)$
    loss = pg_loss + 0.1 * v_loss

    return loss
```

==== PPO训练循环

训练流程：
- 一条轨迹：提示词 + 策略模型输出的补全。
- 批次大小：32。所以我们针对32个提示词生成32条轨迹。那么还是一个提示词*一条轨迹*。
- 每条轨迹通过ppo使用4次更新策略模型。
- 更新后的策略模型作为下一轮训练的$pi_(theta_"old")$。
- 由于32条轨迹同时计算新旧策略的概率比值，比较消耗GPU。所以我们4条4条的计算。也就是`mini_batch_size=4`。
- 使用32条轨迹更新32次策略模型$pi_theta$。更新后的策略模型作为下一轮ppo微调的$pi_(theta_"old")$。

#codly(header: [使用旧策略采集的轨迹更新4次策略])
```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# 使用旧策略产生的轨迹（补全）更新4次模型
ppo_epochs = 4
num_epochs = 1
mini_batch_size = 4

def ppo_update(input_data, old_logprobs, masks, advantages, gae_targets):
    for ep in range(ppo_epochs): # $pi_theta$被更新了32次
        # range(0, 32)
        batch_inds = list(range(batch_size)) # batch_size=32
        for start in range(0, batch_size, mini_batch_size):
            # 4条4条的使用
            mini_batch_inds = batch_inds[start:start+mini_batch_size]

            mb_model_inputs = {
                "input_ids": input_data["input_ids"][mini_batch_inds],
                "attention_mask": input_data["attention_mask"][mini_batch_inds]
            }
            # 模型的输出是token的logits和value
            # mb_vpreds：$[V(S_0),V(S_1),...,V(S_T)]$
            mb_logits, mb_vpreds = model(**mb_model_inputs)
            # 去掉最后一个token
            mb_logits = F.log_softmax(mb_logits[:, :-1, :], dim=-1)
            # 取出真实标签对应的概率：$[log pi_theta (y_0|x), log pi_theta (y_1|x,y_0), ...]$
            mb_logprobs = torch.gather(
                mb_logits,
                2,
                mb_model_inputs["input_ids"][:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            loss = compute_loss(
                old_logprobs[mini_batch_inds], # $log pi_(theta_"old") (y_t|x,y_(<t))$
                mb_logprobs, # $log pi_theta (y_t|x,y_(<t))$
                mb_vpreds[:, :-1], # $V(S_t)$
                masks[mini_batch_inds],
                advantages[mini_batch_inds], # $A_t^"GAE"$
                gae_targets[mini_batch_inds] # $A_t^"GAE" + V(S_t)$
            )

            optimizer.zero_grad()
            # 使用4条轨迹更新了一次$pi_theta$
            loss.backward()
            optimizer.step()
            print("loss/total", loss.item())
    print("ppo update finished")
```

主训练循环代码如下

```python
count = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        if count == 1:
            break
        count += 1
        # 生成补全内容（回复）
        query_tensors = batch["input_ids"]  # 提示词的张量
        query_attention_masks = batch["attention_mask"]

        response_tensors = []  # 补全的张量
        query_response_tensors = []  # 提示词+补全的张量
        score_tensors = []  # 分数的张量

        for i, query in enumerate(query_tensors):
            query = query.to(device)
            query_attention_mask = query_attention_masks[i].to(device)
            # 随机挑一个补全的长度
            new_tokens = random.choice(list(range(
                output_min_length,
                output_max_length)))
            # 设置补全长度属性
            generation_kwargs["max_new_tokens"] = new_tokens
            # 使用$pi_(theta_"old")$采样一条轨迹：提示词 + 补全
            query_response = model.generate(
                input_ids=query.unsqueeze(0),
                attention_mask=query_attention_mask.unsqueeze(0),
                **generation_kwargs
            ).squeeze(0)
            # 补全的长度
            response_len = len(query_response) - len(query)
            # 补全的张量
            response_tensors.append(query_response[-response_len:])
            query_response_tensors.append(query_response)
            # 从奖励模型拿分数
            with torch.no_grad():
                # 提示词 + 补全 + reward_token
                query_response_score = torch.cat([
                    query_response,
                    torch.tensor([REWARD_TOKEN_ID]).to(device)])
                attention_mask = torch.ones_like(
                    query_response_score,
                    dtype=torch.long)
                # 奖励模型的评分
                score = reward_model(
                    query_response_score.unsqueeze(0),
                    attention_mask.unsqueeze(0)
                ).squeeze(0)[-1]
                # 将奖励模型的评分从(0,1)缩放到(-1,1)
                score = 2 * (score - 0.5)
            score_tensors.append(score)

        input_data = data_collator([
            {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids)
            }
            for ids in query_response_tensors
        ]).to(device)

        # 奖励和优势
        old_logprobs, rewards, values, masks = compute_rewards(
            input_data,
            query_tensors,
            response_tensors,
            score_tensors
        )
        # 计算GAE和GAE Target
        advantages, gae_targets = compute_advantage(rewards, values, masks)

        # 小批次训练
        if input_data["input_ids"].shape[0] != 32:
            break
        # 使用采集到的32条轨迹更新策略
        ppo_update(input_data, old_logprobs, masks, advantages, gae_targets)
```

==== 评估模型

主要对比PPO微调后的模型和微调前的模型。对比方法：

- 选择一些提示词
- 使用PPO微调后的模型补全
- 送入奖励模型，打平均分
- 使用PPO微调前的模型补全
- 送入奖励模型，打平均分
- 对比打分情况

#codly(header: [对比微调前后的模型的补全的打分情况])
```python
train_gen_lengths = [0] * len(tokenized_dataset_train)
for i in range(len(tokenized_dataset_train)):
    train_gen_lengths[i] = random.choice(list(range(
        output_min_length,
        output_max_length)))


def validate():
    scores = []
    count = 0
    for b, batch in enumerate(train_dataloader):
        if count == 1:
            break
        count += 1
        # 生成补全内容
        query_tensors = batch["input_ids"]
        query_attention_masks = batch["attention_mask"]
        for i, query in enumerate(query_tensors):
            query = query.to(device)
            query_attention_mask = query_attention_masks[i].to(device)
            new_tokens = train_gen_lengths[b * len(query_tensors) + i]
            generation_kwargs["max_new_tokens"] = new_tokens
            query_response = model.generate(
                input_ids=query.unsqueeze(0),
                attention_mask=query_attention_mask.unsqueeze(0),
                **generation_kwargs
            ).squeeze(0)
            query_response_score = torch.cat([
                query_response,
                torch.tensor([REWARD_TOKEN_ID]).to(device)])
            attention_mask = torch.ones_like(
                query_response_score, dtype=torch.long)
            score = reward_model(
                query_response_score.unsqueeze(0),
                attention_mask.unsqueeze(0)
            ).squeeze(0)[-1]
            score = 2 * (score - 0.5)
            scores.append(score.item())
    print("平均分数：", sum(scores) / len(scores))


validate()

model_path = "./gpt2-sft"
model = ActorCriticModel(model_path).to(device)
validate()
```

#chapter("使用GRPO微调大语言模型——复刻DeepSeek-R1", image: image("./orange2.jpg"), l: "rlhf-dapo")

强化学习已成为增强大语言模型初始训练效果的强大工具，尤其是在推理密集型任务中。DeepSeek 最近在 DeepSeek-Math 和 DeepSeek-R1 模型上取得的突破，展现了 RL 在提升 LLM 数学推理和问题解决能力方面的巨大潜力。

这些成就得益于一种名为"组相对策略优化"（GRPO）的创新强化学习方法，该方法解决了将强化学习应用于语言模型的独特挑战。我们将深入探讨 GRPO 的工作原理，以及它为何代表了 LLM 训练的重大进步。

== GRPO简介

GRPO目标函数如下：

$
  J(theta)^"GRPO" = 1/G sum_(i=1)^G 1/abs(tau_i) sum_(t=1)^abs(tau_i) min [ p A_(tau_i,t), "clip"(p, 1-epsilon,1+epsilon) A_(tau_i,t)] - beta D_"KL" [pi_theta parallel pi_"ref"] \
  "其中比值" space space p = (pi_theta (a_(tau_i,t)|s_(tau_i,t)))/(pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t)))
$

先来说明一下GRPO目标函数中每个数学符号的含义：

- $pi_theta$表示正在更新的策略。
- $pi_(theta_"old")$表示上一轮训练好的旧策略。
- $pi_"ref"$表示冻结的参考模型。
- $G$表示使用旧策略$pi_(theta_"old")$采样的一组轨迹的数量，也就是如果我们使用旧策略采样了10条轨迹，那么$G=10$。
- $tau_i$表示第$i$条轨迹。
- $abs(tau_i)$表示第$i$条轨迹的动作数量。
- $pi_theta (a_(tau_i,t)|s_(tau_i,t))$表示第$i$条轨迹的第$t$个时刻的状态为$s_(tau_i,t)$，以及在这个状态下正在更新的策略$pi_theta$采取动作$a_(tau_i,t)$的概率。
- $pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t))$表示第$i$条轨迹的第$t$个时刻的状态为$s_(tau_i,t)$，以及在这个状态下正在更新的策略$pi_(theta_"old")$采取动作$a_(tau_i,t)$的概率。
- $A_(tau_i,t)$表示第$i$条轨迹的第$t$个时刻的动作的优势。
- $beta$是超参数。$D_"KL" [pi_theta|pi_"ref"]$表示$pi_theta$和$pi_"ref"$而偏离程度，也就是KL散度。

== 使用DAPO（GRPO的变种）微调Qwen2.5-3B-Instruct

=== 实现思路及任务要求

#danger(title: [我们要从零复刻一个类似DeepSeek-R1的模型，产生思维链！])[
  我们的实现不采用原始的GRPO算法，而是采用字节提出的 *DAPO* 算法，DAPO 对原始的GRPO有如下几项改进：

  + *token级的策略梯度损失*：每个token在策略梯度损失中具有同等权重。也就是每个token的优势等于token所在轨迹的组相对优势！
  + *移除KL散度*：策略梯度损失中不再使用KL散度。由于我们不再需要参考策略网络$pi_"ref"$，这可以减少 GPU 内存的使用。
]

我们的算法伪代码如下：

#tip(title: [DAPO算法伪代码])[
  + 对于每个训练步骤，随机选取$N$个问题：$q_1,q_2,...,q_N$。
  + 对于每个问题$q_i$，采样$G$条回答（轨迹）：$a_(i,1),a_(i,2),...,a_(i,G)$ 。   |> $G$为一组轨迹中轨迹的数量
  + 计算每个回答$a_(i,j)$的奖励$r_(i,j)$。
    - $a_(i,j)$为第i个问题的第j条回答。
    - $r_(i,j)$为第i个问题的第j条回答的回报。
  + 计算每个问题 $q_i$ 的奖励的平均值和标准差。
  $
       mu_i & arrow.l "mean"(r_(i,1),r_(i,2),dots.c,r_(i,G)) \
    sigma_i & arrow.l "std"(r_(i,1),r_(i,2),dots.c,r_(i,G))
  $
  5. 对于回答$a_(i,j)$中的每个token，也就是$t$计算优势：
    - $A_(i,j)[t]$为第i个问题的第j条回答的第t个token的优势
    - #text(fill: red)[重要！输出的token（动作）的优势等于token所在回答的组相对优势！]
  $
    A_(i,j) [t] arrow.l (r_(i,j)-mu_i)/sigma_i
  $
  6. 使用GRPO的目标函数计算策略梯度。为了简单起见，我们每次迭代只进行一次策略更新，这样GRPO的目标函数的梯度等价于原始策略梯度法中的梯度估计方法，针对每个token都进行梯度估计。
  $
    nabla_theta log pi_theta (a_(i,j)[t]) dot.c A_(i,j)[t]
  $
  7. 使用梯度更新策略网络$pi_theta$。
  + goto 1
]

#danger(title: [伪代码中第6步的解释])[
  GRPO目标函数最内层为
  $
    min [ (pi_theta (a_(tau_i,t)|s_(tau_i,t)))/(pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t))) A_(tau_i,t), "clip"((pi_theta (a_(tau_i,t)|s_(tau_i,t)))/(pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t))), 1-epsilon,1+epsilon) A_(tau_i,t)]
  $
  如果GRPO使用旧策略采集的轨迹只更新一次策略的话，相当于在$theta=theta_"old"$处进行求导（梯度）。由于此时$theta=theta_"old"$，所以裁剪不会发生。也就是如下：
  $
    & nabla_theta min [ (pi_theta (a_(tau_i,t)|s_(tau_i,t)))/(pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t))) A_(tau_i,t), "clip"((pi_theta (a_(tau_i,t)|s_(tau_i,t)))/(pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t))), 1-epsilon,1+epsilon) A_(tau_i,t)]|_(theta=theta_"old") \
    & = nabla_theta {(pi_theta (a_(tau_i,t)|s_(tau_i,t)))/(pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t))) A_(tau_i,t)}|_(theta=theta_"old") \
    & = (nabla_theta pi_theta (a_(tau_i,t)|s_(tau_i,t))|_(theta=theta_"old"))/(pi_(theta_"old") (a_(tau_i,t)|s_(tau_i,t))) A_(tau_i,t) space space space space colblue("（log梯度技巧）")\
    & = nabla_theta {A_(tau_i,t) dot.c log pi_theta (a_(tau_i,t)|s_(tau_i,t))}|_(theta=theta_"old")
  $
]

我们的任务是什么？我们想要训练一个能够玩类似24点游戏的LLM。叫做`CountDown Task`。

我们将在`CountDown Task`中微调 Qwen2.5-3B-Instruct 模型。给定一个包含 3 个或 4 个数字的列表和一个目标数字，模型需要使用简单的算术运算（`+、-、*、/`）生成一个数学表达式，该表达式的求值结果等于目标数字。例如：

我们用的数据格式如下：

```
nums: [37, 81, 10], target: 34
```

我们会先将上面的数据格式化成如下数据：

#codly(
  zebra-fill: none,
  number-format: none,
)
```
<|im_start|>system
你是一个有用的助手。你首先在脑海中思考推理过程，然后为用户提供答案。<|im_end|>
<|im_start|>user
使用这些数字 [37 81 10]，创建一个等于 34 的等式。你可以使用基本算术运算（+、-、*、/），每个数字只能使用一次。在 <think> </think> 标签中展示你的解题过程。并在 <answer> </answer> 标签中返回最终答案，例如 <answer> (1 + 2) / 3 </answer>。<|im_end|>
<|im_start|>assistant
让我一步步来解决这个问题。
<think>
```

上面的数据输入给我们微调后的模型，模型应该输出的补全如下：

```
可以通过组合给定的数字 37 81 10 使它们满足运算为 34 。首先通过 81 减去 37 得到 44 ，再进行减法运算 44 减去 10 等于 34 ，最后将这两步结果利用小括号组合在一起形成数学表达式为 (81 - 37) - 10 等于 34 。
</think>
<answer> (81 - 37) - 10 </answer><|im_end|>
```

完整的提示词+补全的数据如下：

```
<|im_start|>system
你是一个有用的助手。你首先在脑海中思考推理过程，然后为用户提供答案。<|im_end|>
<|im_start|>user
使用这些数字 [37 81 10]，创建一个等于 34 的等式。你可以使用基本算术运算（+、-、*、/），每个数字只能使用一次。在 <think> </think> 标签中展示你的解题过程。并在 <answer> </answer> 标签中返回最终答案，例如 <answer> (1 + 2) / 3 </answer>。<|im_end|>
<|im_start|>assistant
让我一步步来解决这个问题。
<think>可以通过组合给定的数字 37 81 10 使它们满足运算为 34 。首先通过 81 减去 37 得到 44 ，再进行减法运算 44 减去 10 等于 34 ，最后将这两步结果利用小括号组合在一起形成数学表达式为 (81 - 37) - 10 等于 34 。
</think>
<answer> (81 - 37) - 10 </answer><|im_end|>
```

=== 奖励函数的设计

在使用GRPO玩倒立摆游戏时，我们的奖励很简单，例如一条轨迹执行了5步动作，轨迹就结束了，那么这条轨迹的回报是5。

那么在LLM这个环境中，我们应该如何给一条轨迹（模型输出的补全）奖励呢？因为GRPO是需要给一整条轨迹计算回报的。

在GRPO的实现中，奖励是两部分的总和：

1. *格式奖励*：当模型正确遵循指定的格式并带有思考和答案标签时，模型获得`0.1`的奖励，否则`0`的奖励。
2. *答案奖励*：如果模型的最终答案恰好使用了提供的数字，并且每个数字只使用了一次，并正确求值为目标值，则模型将获得`1`的奖励，否则将获得`0`的奖励。

例如下面的补全只能拿到0.1分的回报。因为格式是没问题的，但是`<answer></answer>`标签内的答案有问题，包含了不该有的`= 34`。

```
可以通过组合给定的数字 37 81 10 使它们满足运算为 34 。首先通过 81 减去 37 得到 44 ，再进行减法运算 44 减去 10 等于 34 ，最后将这两步结果利用小括号组合在一起形成数学表达式为 (81 - 37) - 10 等于 34 。
</think>
<answer> (81 - 37) - 10 = 34</answer><|im_end|>
```

只有下面这种补全才能拿到1.1的满分。既带有`<think></think>`标签和`<answer></answer>`标签，`<answer></answer>`标签内的答案`(81 - 37) - 10`也是完全正确的。

```
可以通过组合给定的数字 37 81 10 使它们满足运算为 34 。首先通过 81 减去 37 得到 44 ，再进行减法运算 44 减去 10 等于 34 ，最后将这两步结果利用小括号组合在一起形成数学表达式为 (81 - 37) - 10 等于 34 。
</think>
<answer> (81 - 37) - 10</answer><|im_end|>
```

所以可以看到，我们的奖励函数是很严格的！

=== 代码实现

创建文件夹

```bash
$ mkdir GRPO-Zero
```

==== 模型结构

Qwen2 的模型结构，保存在文件 `qwen2_model.py` 中。

#codly(
  header: [Qwen2.5模型结构],
  zebra-fill: luma(240),
  number-format: number => [ #number ],
)
```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Qwen2Config:
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 32768
    max_window_layers: int = 70
    model_type: str = "qwen2"
    num_attention_heads: int = 16
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = self._norm(x).type_as(x)
        x = self.weight * x.to(input_dtype)
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, args: Qwen2Config):
        super().__init__()
        self.n_kv_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = self.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=False,
        )
        self.args = args

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        cache_shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        cache_k = torch.zeros(cache_shape, dtype=dtype, device=device)
        cache_v = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

    def del_kv_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        cos, sin = pos_embed
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, unsqueeze_dim=2)
        if start_pos is not None:
            # inference mode
            end_pos = start_pos + seqlen
            self.cache_k[:bsz, start_pos:end_pos, :, :] = xk
            self.cache_v[:bsz, start_pos:end_pos, :, :] = xv
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=self.cache_k[:bsz, :end_pos].transpose(1, 2),
                value=self.cache_v[:bsz, :end_pos].transpose(1, 2),
                is_causal=True if seqlen > 1 else False,
                enable_gqa=True,
            ).transpose(1, 2)
        else:
            # training mode
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                is_causal=True,
                enable_gqa=True,
            ).transpose(1, 2)
        output = output.reshape(bsz, seqlen, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x):
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: Qwen2Config):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.dim = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(
            dim=args.hidden_size,
            intermediate_size=args.intermediate_size,
        )
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        h = x + self.self_attn(self.input_layernorm(x), pos_embed, start_pos=start_pos)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device: torch.device):
        super().__init__()
        self.config = config
        base = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, pos):
        inv_freq = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
        pos = pos[:, None, :].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float().to(x.device) @ pos.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Transformer(nn.Module):
    def __init__(self, params: Qwen2Config, device: torch.device):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_hidden_layers

        self.embed_tokens = torch.nn.Embedding(params.vocab_size, params.hidden_size)
        with torch.device(device):
            self.rotary_emb = Qwen2RotaryEmbedding(config=params, device=device)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        if not params.tie_word_embeddings:
            self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    def output_proj(self, x):
        if self.params.tie_word_embeddings:
            return x @ self.embed_tokens.weight.T
        else:
            return self.lm_head(x)

    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.embed_tokens(tokens)
        pos = torch.arange(0, seqlen, device=tokens.device, dtype=torch.int32)
        pos_emb = self.rotary_emb(h, pos[None, :])

        pipe = []
        for layer in self.layers:
            pipe.append(lambda x, layer=layer: layer(x, pos_emb))
        pipe.append(self.norm.forward)
        pipe.append(self.output_proj)
        return torch.utils.checkpoint.checkpoint_sequential(
            pipe, len(pipe), h, use_reentrant=False
        )

    def inference(self, tokens: torch.Tensor, start_pos: Union[int, torch.Tensor]):
        _bsz, seqlen = tokens.shape
        del _bsz
        h = self.embed_tokens(tokens)

        pos = torch.arange(0, seqlen, device=tokens.device, dtype=torch.int32)[None, :]
        if isinstance(start_pos, torch.Tensor):
            pos = pos + start_pos[:, None]
        else:  # int
            pos.add_(start_pos)
        pos_emb = self.rotary_emb(h, pos)

        for layer in self.layers:
            h = layer(h, pos_emb, start_pos=start_pos)

        # only need the hidden state of the last token
        # to predict the next token
        h = h[:, -1:, :]
        h = self.norm(h)

        output = self.output_proj(h)
        return output

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        for layer in self.layers:
            layer.self_attn.init_kv_cache(
                max_batch_size, max_seq_len, dtype=dtype, device=device
            )

    def del_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.del_kv_cache()

    @classmethod
    def from_pretrained(cls, ckpt_path, device: torch.device):
        config_file = Path(ckpt_path) / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
        args = Qwen2Config(
            attention_dropout=config["attention_dropout"],
            bos_token_id=config["bos_token_id"],
            eos_token_id=config["eos_token_id"],
            hidden_act=config["hidden_act"],
            hidden_size=config["hidden_size"],
            initializer_range=config["initializer_range"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            max_window_layers=config["max_window_layers"],
            model_type=config["model_type"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            vocab_size=config["vocab_size"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            sliding_window=config["sliding_window"],
            use_sliding_window=config["use_sliding_window"],
            use_cache=config["use_cache"],
            tie_word_embeddings=config["tie_word_embeddings"],
            torch_dtype=config["torch_dtype"],
        )
        with torch.device("meta"):
            model = cls(params=args, device=device)

        import safetensors.torch

        model_weight_files = sorted(Path(ckpt_path).glob("model*.safetensors"))
        weights = {}
        for file in model_weight_files:
            weights.update(safetensors.torch.load_file(file, device="cpu"))
        # remove "model." prefix from keys
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True, assign=True)
        return model.to(device)
```

==== 分词器

分词器代码保存在文件 `tokenizer.py` 中。

#codly(header: [分词器实现])
```python
import json
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment
from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase


class Tokenizer:
    """Tokenizer with chat template supported using jinja2 engine"""

    def __init__(self, tokenizer_path: str):
        super().__init__()
        tokenizer_config_path = Path(tokenizer_path).parent / "tokenizer_config.json"
        self.tokenizer_config = json.load(open(tokenizer_config_path))
        self.tokenizer = TokenizerBase.from_file(tokenizer_path)
        self.chat_template = Environment().from_string(
            self.tokenizer_config["chat_template"]
        )
        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        self.pad_token = self.tokenizer_config["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
```

==== 任务定义

首先创建 `data_types.py` ，内容如下：

```python
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Episode:
    """存储一个回合（Episode）或者说一条轨迹的所有相关信息"""
    """一个回合 = 问题 + 一条回答"""
    prefix: str # 问题
    text: str # "问题+回答"整个文本
    prefix_token_ids: List[int] # 问题的input_ids
    prefix_tokens: List[str] # 问题的token组成的列表
    generated_token_ids: List[int] # 生成的回答的token列表
    is_finished: bool # 回答是否结束标志位
    reward: float # 奖励
    reward_info: Dict[str, float] # 详细的奖励信息


@dataclass
class MiniBatch:
    """每个Step训练所需的微批次数据"""
    prefix: List[str] # 问题列表
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    numbers: List[List[int]] # 问题的数字列表
    target: List[int] # 问题对应的答案数字
```

创建文件 `countdown_task.py` ，内容如下

```python
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer

SYSTEM_MESSAGE = (
    "你是一个有用的助手。你首先在脑海中思考推理过程，"
    "然后为用户提供答案。"
)
# `{numbers}` 和 `{target}` 是占位符，构建训练数据时会被替换
USER_TEMPLATE = (
    "使用这些数字 {numbers}，创建一个等于 {target} 的等式。"
    "你可以使用基本算术运算（+、-、*、/），每个数字只能使用一次。"
    "在 <think> </think> 标签中展示你的解题过程。"
    "并在 <answer> </answer> 标签中返回最终答案，例如 <answer> (1 + 2) / 3 </answer>。"
)

RESPONSE_PROMPT = "让我一步步来解决这个问题。\n<think>"


class CountdownTasksDataset(Dataset):
    """准备训练数据集"""

    def __init__(
        self,
        tokenizer: Tokenizer, # 分词器
        data_path: str, # 数据集的路径
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path) / "data")
        # 索引 `test_size` 后面的数据用作测试数据
        self.data = (
            data.iloc[:-test_size]               \
            if split == "train"                  \
            else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(
            self.encode_prefix(
                item["nums"], # 数字列表
                item["target"] # 目标数字
            )
        )
        return item

    def encode_prefix(self, numbers: List[int], target: int):
        """Prefix 是模型 *真正的* 输入，也就是问题"""
        # 格式化对话模板
        user_message = USER_TEMPLATE.format(
            numbers=numbers,
            target=target
        )
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        # 将问题切分
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix, # 问题字符串
            "prefix_tokens": tokens.tokens, # 问题切分后的字符串列表
            "prefix_token_ids": tokens.ids, # input_ids
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """将数据整理到一个批次中"""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [
            item["prefix_tokens"] for item in batch
        ]
        prefix_token_ids = [
            item["prefix_token_ids"] for item in batch
        ]
        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


def format_reward_function(
    response: str, # 模型的回答
    end_token: Optional[str] = None # 结尾token
) -> float:
    """
    检查模型的回复是否符合格式 <think>...</think><answer>...</answer>
    """
    # 如果存在end token，则去掉
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = \
        r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(
        full_format_regex,
        response,
        re.DOTALL
    )
    # 如果完全匹配，则给1分
    if full_format_match:
        return 1.0

    reward = 0.0
    # 如果有<think></think>标签对，则奖励加0.1分
    if think_match:
        reward += 0.1
    # 如果有<answer></answer>标签对，则奖励加0.5分
    if answer_match:
        reward += 0.5
    # 返回奖励
    return reward


def answer_reward_function(
    response: str, # 模型给出的回答
    numbers: List[int] = None, # 数字列表
    target: int = None # 目标数字
) -> float:
    """
    检查答案中：
    1. 是否使用了所有给的数字
    2. 每个数字是否使用了一次
    3. 答案中包含的表达式的求值结果是否等于目标数字
    """
    # 答案的正则表达式
    answer_regex = r"<answer>(.*?)<\/answer>"
    # 回答中是否有答案标签对
    answer_match = re.search(answer_regex, response, re.DOTALL)
    # 如果在回答中没有搜索到答案，那么给0分
    if not answer_match:
        return 0.0
    # 提取出答案的文本
    answer_content = answer_match.group(1)
    # 如果答案标签内没有东西，给0分
    if not answer_content:
        return 0.0
    # 如果答案标签中，除了表达式以外，还有其它内容，给0分
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # 检查答案中，每个数字是否只使用了一次
    used_numbers = [
        int(n) for n in re.findall(r"\d+", answer_content)
    ]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # 检查答案中包含的表达式的求值结果是否为目标数字
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        pass

    return 0.0


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Countdown Task 的奖励函数。

    总奖励 = 0.1 * 格式奖励 + 答案准确性奖励
    """
    format_reward = format_reward_function(
        "<think>" + response,
        end_token
    )
    answer_reward = answer_reward_function(
        response,
        numbers,
        target
    )
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
```

==== GRPO（DAPO）算法的实现

创建文件 `grpo.py` ，内容如下：

```python
import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer

# 采集轨迹，也就是回答
# 根据一个问题，采样多条回答
# 假设5个问题，每个问题8个回答，那么要并行采样40条轨迹
# 可能碰到的实现的坑，都来自并行推理
# 1. 问题的长度不一样
# 2. 有的补全结束的早，有的结束的晚。
@torch.no_grad()
def rollout(
    model: Transformer, # 生成回答的llm模型
    batch: MiniBatch, # N个问题
    tokenizer: Tokenizer,
    max_gen_len: int, # 最大生成长度
    num_answer_per_question: int, # 每个问题产生多少个回答
    reward_function: Callable, # 奖励函数
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    # 问题：List[input_ids]
    prefix_token_ids = batch.prefix_token_ids
    # 批次中的问题数量 x 每个问题生成的回答数量 = 批次中的数据量
    bsz = len(batch.prefix) * num_answer_per_question
    # 最短问题长度
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    # 最长问题长度
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    # 总长度 = 最大生成长度 + 最大问题长度
    total_len = max_gen_len + max_prompt_len
    # 开启KV Cache，加速生成回答的速度
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    # 将所有token先初始化为填充符pad_token_id：批次中数据量 x 每条数据的总长度
    tokens = torch.full(
        (bsz, total_len),
        pad_token_id,
        dtype=torch.long,
        device=device
    )
    # 将问题部分填入
    # 第 k 个问题假设生成 num_answer_per_question 条回答
    # 那么 num_answer_per_question 条训练数据的前缀都是第k个问题
    for k, t in enumerate(prefix_token_ids):
        # 第k个问题的数据在批次中的偏移量
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            # 第k个问题的第i条完整数据的问题部分
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    # 文本的掩码，填充符置为False
    input_text_mask = tokens != pad_token_id
    # 确保最小的问题长度小于总长度
    assert min_prompt_len < total_len
    # 标志位，标志一条回答（一条轨迹）是否结束，初始化为0
    is_finished = torch.zeros(
        (bsz,), dtype=torch.bool, device=device)
    # 并行的预测下一个token
    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* 生成轨迹:{cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        # 针对批次中的所有训练数据，并行采样下一个token
        # 根据文本的 prev_pos~cur_pos 部分生成下一个token
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(
                tokens[:, prev_pos:cur_pos],
                prev_pos
            )
        # logits ---> probs
        probs = torch.softmax(logits[:, -1], dim=-1)
        # 采样下一个token，具体使用了多项分布来采样
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(-1)
        # 如果cur_pos这个索引已经有token了，那么直接作为下一个token
        # cur_pos这个所有已经有token，说明是长问题，那么这个token不需要预测
        # 注意：这里cur_pos对应的token不能是pad
        next_token = torch.where(
            input_text_mask[:, cur_pos], # cur_pos是否已经存在token了
            tokens[:, cur_pos], # 对于长的问题，cur_pos对应的已经有token了
            next_token # 对于最小长度的问题，选择预测出来的next_token
        )
        # 如果生成回答已经结束，那么下一个token是pad，
        # 如果没有结束，那么是next_token
        next_token = torch.where(
            is_finished,
            pad_token_id, # 对于短回答，回答已经结束，需要继续填充pad
            next_token # 对于长回答，回答没有结束，需要使用预测出来的token
        )
        # 将cur_pos赋值为下一个token
        tokens[:, cur_pos] = next_token
        # 如果有结尾标记
        if end_token_id is not None:
            # 检查这个结尾标记是否为生成下一个token得到的
            is_end_token = next_token == end_token_id
            # 如果cur_pos对应的是False，说明cur_pos是填充符
            # 说明这个token是生成的next token
            is_generated_token = ~input_text_mask[:, cur_pos]
            # 如果eos token是生成的，那么结束。
            is_finished = is_finished \
                        | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        # 如果全部结束，那么跳出循环
        if is_finished.all():
            break
    # 删除kv cache
    model.del_kv_cache()
    # 手动垃圾回收
    gc.collect()
    # 清空cuda显存
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # 准备存放输出回合的数组
    episodes = []
    # 遍历批次中的问题数量
    for i in range(bsz // num_answer_per_question):
        # 遍历第i条问题的第j条回答
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            # 截取出回答部分
            generated_token_ids =                   \
                tokens_list                         \
                [idx]                               \
                [len(batch.prefix_token_ids[i]):]
            # 删除填充token
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    :generated_token_ids.index(pad_token_id)
                ]
            # 生成的文本
            generated_text = \
                tokenizer.detokenize(generated_token_ids)
            # 计算第i个问题的第j条回答的奖励
            rewards = reward_function(
                # 生成的文本
                response=generated_text,
                # 数字列表
                numbers=batch.numbers[i],
                # 正确答案数字
                target=batch.target[i],
                end_token=end_token,
            )
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    # 清除输出内容
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(
    episodes: List[Episode]
) -> List[Episode]:
    """归一化每个组的奖励. 使用 prefix（问题） 区分不同的组."""
    """每条轨迹的reward字段替换为轨迹的优势"""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    output = []
    # 遍历每个组，一个问题对应一组回答
    for group in groups.values():
        # $[r_(i,0), r_(i,1), ...]$
        group_rewards = [item.reward for item in group]
        # 每个组的回答的奖励的平均值
        mean_reward = np.mean(group_rewards)
        # 每个组的回答的奖励的标准差
        std_reward = np.std(group_rewards)
        # 遍历组中的每一条回答，然后计算这条回答的优势
        # $(r_i - "mean"(r))/("std"(r)+epsilon)$
        for episode in group:
            normalized_reward =                \
                (episode.reward - mean_reward) \
                /                              \
                (std_reward + 1e-4)
            # reward字段，使用回答的组内优势替换掉奖励
            episode = dataclasses.replace(
                episode,
                reward=normalized_reward
            )
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """计算熵，熵越小不确定性越小，用来监控模型训练的稳定性，不参与反向传播"""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy =                           \
        torch.logsumexp(logits, dim=-1) \
        -                               \
        torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model, # 微调的模型
    optimizer, # 优化器
    episodes: List[Episode], # 轨迹（问题+回答）的数组
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float, # 梯度裁剪，1.0
    device: torch.device,
    dtype: torch.dtype,
):
    """使用GRPO算法更新策略."""
    # 计算出每一条回答的组内优势
    episodes = normalize_rewards_per_group(episodes)
    # 按照回合的token数量排序，更有效的微批次训练
    episodes.sort(
        key=lambda x:                   \
        len(x.prefix_token_ids)         \
        +                               \
        len(x.generated_token_ids))
    num_target_tokens = sum(
        len(episode.generated_token_ids)
        for episode in episodes
    )
    entropy = 0.0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* 计算策略梯度: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids)     \
            +                                 \
            len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        # 微批次中最长的轨迹长度
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids      # 问题的input_ids
            + episode.generated_token_ids # 生成的回答的input_ids
            + [pad_token_id] * ( # 添加填充符pad
                  batch_max_length - batch_lengths[i]
              )
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            # 问题部分掩码是0
            [0] * len(episode.prefix_token_ids)
            # 回答部分掩码为1
            + [1] * len(episode.generated_token_ids)
            # 填充符掩码为0
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        # 取出每个回合的优势$(r_i-"mean"(r))/("std"(r)+epsilon)$
        batch_advantages = [
            episode.reward for episode in batch_episodes
        ]
        batch_token_ids = torch.tensor(
            batch_token_ids,
            device=device,
            dtype=torch.long
        )
        batch_masks = torch.tensor(
            batch_masks,
            device=device,
            dtype=torch.bool
        )
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            # 去掉最后一个token，输入
            input_token_ids = batch_token_ids[:, :-1]
            # 去掉第一个token，目标token
            # 真实的目标token是来自上一轮的模型输出的回答
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            # logits是预测的下一个token
            logits = model.forward(input_token_ids).float()
        # 在 one-hot 分类里，
        # 交叉熵等于对正确类别概率取负对数，
        # 所以"负对数概率"与"交叉熵"指的是同一个目标函数。
        # $log pi_theta (a|s) = -"cross_entropy"$
        # $-sum_j a_j dot.c log hat(a)_j = - log hat(a)_t$，$a_t$是真实标签，$hat(a)_t$是模型预测为$a_t$的概率。
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy                            \
                    +                                    \
                    (token_entropy * target_masks).sum() \
                    /                                    \
                    num_target_tokens
        # 对数概率乘以优势$log pi_theta (a|s) dot.c A$
        obj = log_probs * batch_advantages[:, None]
        # 计算每个token的平均目标
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        # 每一轮都要进行反向传播，计算模型参数的导数，但不更新模型的参数
        loss.backward()

    # 梯度裁剪
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    # 清空梯度
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
```

==== 训练循环

创建文件 `train.py` ，内容如下：

```python
import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from countdown_task import CountdownTasksDataset, reward_function
from grpo import rollout, update_policy
from qwen2_model import Transformer
from tokenizer import Tokenizer


def evaluate(model, tokenizer, device, dtype):
    """每隔10个step，使用测试数据集评估一下，看能做对多少题"""
    test_dataset = CountdownTasksDataset(
        data_path="Countdown-Tasks-3to4",
        tokenizer=tokenizer,
        split="test",
        test_size=128, # 128条测试数据
    )
    generator = torch.Generator(device=device)
    # 批次大小减半，我们就可以生成2倍长的轨迹了
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        # 批次大小为256，减半
        batch_size=256 // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            # 最大生成长度为1024，乘以2
            max_gen_len=1024 * 2,
            # 评估时，针对每个问题只生成1个回答
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([
            episode.reward_info["answer_reward"] \
            for episode in episodes
        ])
    return np.mean(success)


def main():
    pretrained_model_path = Path(
        "./Qwen2.5-3B-Instruct/"
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.set_default_device(device)
    torch.random.manual_seed(1337)
    # 批次大小
    BATCH_SIZE = 256
    # 每个批次32个问题
    NUM_QUESTIONS_PER_BATCH = 32
    # 每个问题产生8条回答
    NUM_ANSWERS_PER_QUESTION = \
        BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"./logs/{current_time}")
    tokenizer = Tokenizer("./Qwen2.5-3B-Instruct/tokenizer.json")

    train_dataset = CountdownTasksDataset(
        data_path="./Countdown-Tasks-3to4/",
        tokenizer=tokenizer,
        split="train",
        test_size=128,
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    model = Transformer.from_pretrained(
        pretrained_model_path,
        device=device
    ).train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-5,
        weight_decay=0.0,
        betas=[0.9, 0.999],
    )

    start_time = time.time()
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(train_dataloader, start=1):
        # 生成轨迹（问题+回答）数据
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=1024,
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        # 更新策略
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=2, # 微批次大小为2
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=1.0, # 梯度裁剪到1.0
            device=device,
            dtype=dtype,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # 计算一些统计信息，然后保存到日志文件中
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] \
            for episode in episodes
        ]
        answer_reward = [
            episode.reward_info["answer_reward"] \
            for episode in episodes
        ]
        num_finished_episodes = sum(
            episode.is_finished for episode in episodes
        )
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) \
             for episode in episodes]
        )
        print(
            f"\r步骤 {step}, 平均奖励: {mean_reward:.2f}, "
            f"计算正确率: {success_rate:.2f}, "
            f"梯度裁剪: {grad_norm:.2f}, 时长: {duration:.2f}, "
            f"结束的回合的数量: {num_finished_episodes}, "
            f"平均回答长度: {mean_response_len:.2f}, "
            f"熵: {entropy:.2f}"
        )
        # 每隔10步评估一次
        if step % 10 == 0:
            eval_success_rate = evaluate(
                model,
                tokenizer,
                device,
                dtype
            )
            print(f"\r评估数据集回答正确率: \
                   {eval_success_rate:.2f}" \
                   + " " * 100)
            tb_writer.add_scalar(
                "回答正确率/评估",
                eval_success_rate,
                step
            )

        tb_writer.add_scalar("损失", loss, step)
        tb_writer.add_scalar("平均奖励", mean_reward, step)
        tb_writer.add_scalar("奖励的标准差", std_reward, step)
        tb_writer.add_scalar(
            "回答正确率/训练", success_rate, step)
        tb_writer.add_scalar("格式奖励", format_reward, step)
        tb_writer.add_scalar("梯度裁剪", grad_norm, step)
        tb_writer.add_scalar("时长", duration, step)
        tb_writer.add_scalar(
            "结束的回合数量",
            num_finished_episodes,
            step
        )
        tb_writer.add_scalar("学习率", lr, step)
        tb_writer.add_scalar(
            "平均回答长度", mean_response_len, step)
        tb_writer.add_scalar("熵", entropy, step)
        for i, episode in enumerate(episodes):
            # TensorBoard 将文本处理为markdown格式
            text = html.escape(episode.text)
            tb_writer.add_text(
                f"text_{i}",
                f"<pre>{text}</pre>",
                step
            )

        # 每隔100步保存模型的检查点
        if step % 100 == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"将检查点保存到 {output_file}")


if __name__ == "__main__":
    main()
```

执行如下命令开始训练！

```bash
$ uv run train.py
```

== GRPO应用场景

=== 医疗思维链

我们使用 Hugging Face 的 `interleave_datasets` 混合三个关键数据集：

*PubMedQA*（占总数据的70%）：

- 临床问答，答案为yes/no/maybe

*GSM8K*：

- 数学应用题，以保持数学推理能力

*Health Benchmarks*：

- 50+道医学专业选择题
- 从心脏病学到疫苗接种的类别

权重应该反映数据集的复杂性——PubMedQA的曝光度增加了3倍，以处理其细微差别。我们这里对数据集进行了打乱。由于PubMedQA样本量增加了3倍，因此将这些样本展示给模型的几率也增加了3倍。

模型在训练过程中的答案

```
Prompt（提示）：

在以下科学背景下：

我们此前已经展示了对志愿者进行 4 周、共 19 次、每日亚红斑量紫外线（DUVR，模拟非极端暴露条件）照射所产生的不利影响。该光源的 UVA（320-400 nm）与 UVB（290-320 nm）辐照度比为 25，而典型用于代表夏季晴朗近天顶太阳辐照的模拟太阳光（SSR）的该比值接近 10。
在此，我们报告对之前研究的拓展，评估了一款广谱日常护肤产品（低 SPF：SPF 8，UVA-PF 7，UVA 3* 级防护）所提供的光防护。我们检测了与皮肤癌与光老化相关的细胞与分子光损伤标志物。
本研究显示，重复 DUVR 暴露所产生的生物学效应可以被一款广谱日常护理产品所预防，且其所提供的防护水平会随具体检测终点而变化。

回答下列问题：
一款广谱防晒是否能预防代表温带地区的日常亚红斑量太阳紫外线重复暴露所导致的累积性损伤？请用 "yes" / "no" / "maybe" 作答。你需要仔细阅读背景并推理后回答。

Model Response（模型回答）：

<reasoning>
提供的背景讨论了一项研究：重复的亚红斑量 UV（DUVR）暴露在 4 周内会产生不利影响。它还提到使用了一款广谱日常护理产品（SPF 8，UVA 防护因子 7，UVA 3* 评级）进行评估。研究的目的是判断该广谱产品是否能预防这类重复暴露引起的累积性损伤。文本中说明，该研究显示重复 DUVR 的生物学效应可以被这款产品预防，说明其具有保护作用。虽然不同终点的防护程度不同，但整体表述支持其确实能预防累积损伤。因此答案应为"yes"。
</reasoning>

<answer>
yes
</answer>
```

TRL代码如下：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset, interleave_datasets, concatenate_datasets
import re


# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting


def get_datasets(split="train") -> Dataset:
    data = load_dataset('gsm8k',
                        'main')[split]  # type: ignore
    data = data.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer']),
        'db_set': 'gsm8k'
    })  # type: ignore
    data = data.remove_columns(['question'])

    # two times more than other datasets
    data_qa = load_dataset(
        "PubMedQA", "pqa_artificial")[split]
    data_qa = data_qa.filter(lambda x: len(
        "\n".join(x['context']['contexts'])) < 1024)  # avoid long traces
    data_qa = data_qa.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Given the scientific context below:\n" +
                "\n".join(x['context']['contexts']) +
                "\n\nAnswer the following question:\n" +
                x['question'] +
                " with 'yes', 'no' or 'maybe'. You need to carefully review the context and reason before answering."
            },
        ],
        'answer': x['final_decision'],
        'db_set': 'pubmedqa'
    })  # type: ignore
    data_qa = data_qa.remove_columns(
        ['pubid', 'question', 'context', 'long_answer', 'final_decision'])

    categories = [
      'Lab_Medicine',
      'Wearables',
      'Dermatology',
      'Gastroenterology',
      'Internal_Medicine',
      'Oncology',
      'Orthopedics',
      'General_Surgery',
      'Ophthalmology',
      'Audiology',
      'Head_Neck_Surgery',
      'Elderly_Care',
      'Pediatrics',
      'Allergy_Immunology',
      'Rheumatology',
      'Pharmacy',
      'Obstetrics_Gynecology',
      'Microbiology',
      'Dentistry',
      'Physical_Medicine_and_Rehabilitation',
      'Neurology',
      'Psychiatry',
      'Pathology',
      'Genetics',
      'Rare_Diseases',
      'Hematology',
      'Emergency',
      'Endocrinology',
      'Radiology',
      'Cardiology',
      'Pulmonology',
      'Infectious_Diseases',
      'Critical_Care',
      'Pediatric_Surgery',
      'Neuroscience',
      'Epidemiology',
      'Fitness_Sports',
      'Health_Education',
      'Health_Economics',
      'Health_Entrepreneurship',
      'Hospital_Management',
      'Mental_Health',
      'Nutrition',
      'Palliative_Care',
      'Preventive_Medicine',
      'Public_Health',
      'Social_Media_Addiction',
      'Sleep',
      'Supplements',
      'Vaccination',
      'Work_Health',
      'Wellbeing'
    ]
    data_mc = concatenate_datasets(
        [load_dataset("Health_Benchmarks", i)[i] for i in categories])
    data_mc = data_mc.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\nAnswer the following question:\n" +
                x['Questions'] +
                "\n With 'A', 'B', 'C' or 'D'. You need to carefully review the context and reason before answering."
            },
        ],
        'answer': x['Answers'],
        'db_set': 'med_mc'
    })  # type: ignore
    data_mc = data_mc.remove_columns(['Answers', 'Questions'])

    dataset = concatenate_datasets([data, data_qa, data_mc])
    return dataset


dataset = get_datasets()
dataset = dataset.shuffle(seed=42)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]
print(f"train size: {len(train_dataset)}, test size: {len(test_dataset)}")


# Reward functions
def correctness_reward_func(prompts, completions, answer, db_set, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    rewards = []
    for r, a, dt in zip(extracted_responses, answer, db_set):
        if dt == "gsm8k":
            if a in r:
                rewards.append(1.0)
            elif r == a:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(2.0 if r.lower() == a.strip().lower() else 0.0)
    return rewards


def int_reward_func(completions, db_set, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r, dt in zip(extracted_responses, db_set):
        if dt == "gsm8k":
            rewards.append(0.5 if r.isdigit() else 0.0)
        elif dt == "pubmedqa":
            rewards.append(0.5 if (
                'yes' in r.lower() or 'no' in r.lower() or 'maybe' in r.lower()) else 0.0)
        else:
            rewards.append(0.5 if ('a' in r.lower() or 'b' in r.lower(
            ) or 'c' in r.lower() or 'd' in r.lower()) else 0.0)
    return rewards


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


training_args = GRPOConfig(output_dir="outputs", num_generations=2, per_device_train_batch_size=4)


# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "../GRPO/Qwen2.5-3B-Instruct/",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "../GRPO/Qwen2.5-3B-Instruct/")


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()
```

=== Text-To-SQL

#link("https://github.com/yai333/Text-to-SQL-GRPO-Fine-tuning-Pipeline/tree/main")[GITHUB地址]

底座模型：`Qwen2.5-Coder-7B-Instruct`

任务

- 输入：请帮我写一段SQL，要求查询出部门35岁以上的程序员
- 输出：`SELECT * FROM database WHERE ...`

该模型展现出强大的整体性能，SQL 生成准确率高（44/50 分，得分 4 或 5 分），推理质量优异（48/50 分，得分 4 或 5 分），格式遵循近乎完美（49/50 分，得分 5 分），且具有明确的教育价值。总体而言，88% 的输出得分达到 4.0 分或更高，反映出模型结果的一致性、结构良好且易于解释。

#tip(title: [奖励函数编写要点])[
  1. 使用正则表达式来实现格式奖励
  2. 创建一个sqlite3数据库，用来校验SQL语句的正确性
  3. 调用DeepSeek或者GPT接口，让外部大语言模型评估一下CoT的质量。
]




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
# 取出测试数据集中的第1000张图片
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

上面的过程被称为*正向扩散*：我们通过添加少量其他液体来改变环境状态。然而，进行*反向扩散*——将混合物恢复到其原始状态——是否同样容易？事实证明并非如此。即使在最好的情况下，实现这一点也需要高度复杂的机制。

==== 将类比应用于机器学习

扩散也可以应用于图像。想象一下一张高质量的狗狗照片。我们可以通过逐渐添加随机噪声来轻松地变换这幅图像。结果，像素值会发生变化，使图像中的狗狗变得不那么明显，甚至无法辨认。这个变换过程称为*正向扩散*。

#figure(
  image("小狗图像的正向扩散和反向扩散过程.svg"),
  caption: [高清图像的扩散过程],
)

我们也可以考虑反向操作：给定一张噪声图像，目标是重建原始图像。这项任务更具挑战性，因为与大量可能的噪声变化相比，可高度识别的图像状态要少得多。用前面提到的物理类比，这个过程称为*反向扩散*。

在本文中，我将通过示意图来解释它的工作原理。

=== 扩散模型的架构

为了更好地理解扩散模型的结构，让我们分别检查两个扩散过程。

==== 正向扩散

如前所述，前向扩散涉及逐步向图像添加噪声。然而，在实践中，这个过程要更加微妙一些。

最常见的方法是从均值为0的*高斯分布*中为图片中的每个像素采样一个随机值。然后将这个采样值（可以是正值也可以是负值）添加到像素的原始值中。对所有像素重复此操作会得到原始图像的噪声版本。

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

通常，迭代次数选择在50到1000之间。

==== 神经网络架构

最常见的是，U-Net架构被用作扩散模型的主干。以下是一些原因：

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

实际上，这意味着我们使用一个具有共享权重的U-Net模型，该模型基于来自不同扩散步骤的图像对进行训练。在推理过程中，含噪图像会多次通过同一个经过训练的U-Net模型，逐步进行优化，直到生成高质量的图像。

#figure(
  image("共享模型.svg"),
  caption: [共享模型],
)

虽然由于仅使用单一模型，生成质量可能会略有下降，但训练速度的提升却非常显著。

== 扩散模型理论简介

=== 概述

#figure(
  image("正向扩散和逆向扩散.png"),
  caption: [正向扩散和反向扩散],
)

扩散模型的训练可以分为两部分：

+ 正向扩散过程 #sym.arrow 给图像添加噪声。
+ 反向扩散过程 #sym.arrow 从图像中去除噪声。

=== 正向扩散过程

#figure(
  image("正向扩散公式图解.svg"),
  caption: [正向扩散公式图解],
)

#tip(title: "正态分布")[
  $
    f(x) = 1/(sigma sqrt(2 pi)) e^(-(x-mu)^2/(2 sigma^2))
  $
]

#notify(title: "方差计划")[
  假设$beta_#text[start] = 0.0002, beta_#text[end] = 0.04$，则第1步添加的高斯噪声的方差是0.0002, 第2步添加的高斯噪声的方差是$0.0002 + (0.04 - 0.0002) / 1000$。
]

前向扩散过程逐步将高斯噪声添加到输入图像$x_0$中，总共会有$T$步。该过程将产生一系列带噪声的图像样本$x_1,x_2,...,x_T$。

当$T arrow infinity$时，最终结果将变成完全噪声图像，就像从*各向同性*的高斯分布中采样出来的噪声一样。

首先，如果$z tilde cal(N) (mu, sigma^2)$的话，那么正态分布可以写成如下公式：

$
  z=mu+sigma epsilon space "其中" epsilon tilde cal(N) (0,1)
$

利用这个技巧，我们可以将采样图像$x_t$表示如下：

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
  根据上面的公式，如果想要从原始图片$x_0$得到添加了500步噪声的图片$x_500$需要迭代500次！
]

但我们不需要设计一种算法来迭代地向图像中添加噪声，而是可以使用闭式公式（解析解）在特定的时间步$t$直接对噪声图像进行采样。

给定原始图片$x_0$和时间步$t$可以直接得到添加了$t$步噪声的图像$x_t$。公式如下：

#tip(title: "给定原始图片" + $x_0$ + "和时间步" + $t$ + "直接采样出" + $x_t$ + "的公式")[
  $
    x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon
  $ <closed-form-add-noise>
  其中：
  - $overline(alpha)_t=alpha_t alpha_(t-1) dots alpha_1$
  - $alpha_t = 1 - beta_t$
  - $epsilon tilde cal(N) (0,1)$
]

现在我们可以使用此公式在任何时间步直接对$x_t$进行采样，这使得前向扩散过程更快。

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

# 一步得到$x_t$,使用闭式解（closed form）
def add_noise(x_0, t, betas):
    T = len(betas)

    alphas = 1 - betas  # $[alpha_1, alpha_2, dots.c, alpha_T]$
    # cumprod功能：[1,2,3,4] --> [1,2,6,24]
    alpha_bars = torch.cumprod(alphas, dim=0)
    t_idx = t - 1
    alpha_bar = alpha_bars[t_idx]  # $overline(alpha)_t$

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

+ 将为每个训练样本（图像）选择一个随机时间步$t$。
+ 对每幅图像添加高斯噪声（对应于$t$）。
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

请注意，在最后一步，我们只是输出学习到的平均值$mu_theta (x_1,1)$，而不向其中添加噪声。

== 代码实现

=== 扩散过程相关代码

我们需要针对扩散的时间步来制定一个方差的调度计划。每个时间步，都在前面一个时间步的图像中添加一个高斯噪声。在开始向图像中添加噪声的时候，需要高斯噪声的方差小一些，不至于一开始就把图像变得很模糊。而到了后面，添加噪声就可以大胆一些了，反正已经模糊了。所以后面添加的高斯噪声需要方差大一些。

#tip(title: "高斯噪声的方差")[
  公式中的$beta_t$就是第$t$个时间步要添加的噪声的方差。
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

而由于$alpha_t=1-beta_t$，所以有如下代码

```python
self.alphas = 1 - self.betas
```

这样就计算出了每个时间步的$alpha_t$。

而由于$overline(alpha)_t=product_(s=1)^t alpha_s$，所以使用`torch.cumprod`来计算

```python
self.alpha_bars = torch.cumprod(self.alphas, dim=0)
```

由于我们已经使用重参数技巧来给图像添加噪声，也就是通过解析解可以直接得到添加了$t$个时间步的噪声的图像。所以前向扩散过程就用了这个公式。

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

接下来编写反向去噪过程的代码，也就是从$x_t$预测$x_(t-1)$。

#figure(
  image("预测噪声.svg"),
  caption: [以噪声为预测目标],
)

给定了时间步和添加噪声的图像，模型可以预测出这幅图片中的噪声有多少，那么我们从$x_t$中将预测出的噪声减掉，就可以去噪了！

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

这段代码使用最大池化（`nn.MaxPool2d`）来缩小数据。这会使张量的大小缩小$1/2$。而在扩大数据的处理中，代码使用了双线性插值的上采样（`nn.Upsample`）。这会使张量的大小扩大$2$倍。

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
  caption: [正向扩散和反向扩散],
)

扩散模型的训练可以分为两部分：

1. 正向扩散过程 #sym.arrow.long 给图像添加噪声。
2. 反向扩散过程 #sym.arrow.long 从图像中去除噪声。

== 正向扩散过程

#figure(
  image("正向扩散公式图解.svg"),
  caption: [正向扩散公式图解],
)

正向扩散过程逐步将高斯噪声添加到输入图像$x_0$中，总共会有$T$步。该过程将产生一系列带噪声的图像样本$x_1, dots, x_T$。

当$T arrow infinity$时，最终结果将变成完全噪声图像，就像从*各向同性*的高斯分布中采样出来的噪声一样。

但我们不需要设计一种算法来迭代地向图像中添加噪声，而是可以使用闭式公式（解析解）在特定的时间步$t$直接对噪声图像进行采样。

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

利用这个技巧，我们可以将采样图像$x_t$表示如下：

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
  所有$epsilon$都是`i.i.d.`（独立同分布）标准正态随机变量。

  使用不同的符号和下标来区分它们非常重要，因为它们是独立的，并且在采样后它们的值可能会有所不同。
]

重复这些步骤将为我们提供以下仅取决于输入图像$x_0$的公式：

$
  x_t = sqrt(overline(alpha)_t)x_(0)+sqrt(1-overline(alpha)_t) epsilon
$

现在我们可以使用此公式在任何时间步骤直接对$x_t$进行采样，这使得前向过程更快。

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

贝叶斯推断为什么要忽略掉分母$q(x_t)$呢？因为我们要计算的是，在*固定*$x_t$的情况下，求$x_(t-1)$的概率，所以$q(x_t)$是个常数。可以忽略掉。

现在的问题是，$q(x_t|x_(t-1))$我们已经知道公式了，但$q(x_(t-1))$我们不知道怎么计算。

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
    tilde(beta)_t = (1-overline(alpha)_(t-1)) / (1-overline(alpha)_t) beta_t = ((1-overline(alpha)_(t-1))(1-alpha_t)) / (1-overline(alpha)_t)
  $
]

*证明* 使用贝叶斯定理，利用前向过程的转移概率是高斯分布的性质，可以推导出反向过程的条件转移概率分布也是高斯分布，并得到其概率密度函数。

$
  & q(x_(t-1)|x_t,bold(upright(x))_0) \
  & = (q(x_(t-1)|bold(upright(x))_0)q(x_t|x_(t-1),bold(upright(x))_0)) / q(x_t|bold(upright(x))_0) \
  & = (q(x_(t-1)|bold(upright(x))_0)q(x_t|x_(t-1))) / q(x_t|bold(upright(x))_0) \
  & = (cal(N)(x_(t-1);sqrt(overline(alpha)_(t-1))bold(upright(x))_0,(1-overline(alpha)_(t-1))bold(I))cal(N)(x_t;sqrt(alpha)_t x_(t-1),(1-alpha_t)bold(I))) / (cal(N)(x_t;sqrt(overline(alpha)_t)bold(upright(x))_0, (1-overline(alpha)_t)bold(I))) \
  & prop exp { - 1/2 [ ( (x_t - sqrt(alpha_t)x_(t-1))^2 ) / beta_t + ( (x_(t-1) - sqrt(1-overline(alpha)_(t-1))bold(upright(x))_0)^2 ) / (1-overline(alpha)_(t-1)) - (x_t - sqrt(overline(alpha)_t)bold(upright(x))_0)^2 / (1-overline(alpha)_t) ] } \
  & = exp { - 1/2 [ ( x_t^2 - 2 sqrt(alpha_t) x_t x_(t-1) + alpha_t x_(t-1)^2 ) / beta_t + ( x^2_(t-1) - 2 sqrt(overline(alpha)_(t-1))x_(t-1) bold(upright(x))_0 + overline(alpha)_(t-1) bold(upright(x))_0^2 ) / (1-overline(alpha)_(t-1)) - (x_t - sqrt(overline(alpha)_t)bold(upright(x))_0)^2 / (1-overline(alpha)_t) ] } \
  & = exp { - 1/2 [ (alpha_t / beta_t + 1 / (1-overline(alpha)_(t-1)))x_(t-1)^2 - 2 ( sqrt(alpha_t) / beta_t x_t + sqrt(overline(alpha)_(t-1)) / (1 - overline(alpha)_(t-1)) bold(upright(x))_0 )x_(t-1) + C(x_t,bold(upright(x))_0) ] }
$

$C(x_t,bold(upright(x))_0)$是对于$x_(t-1)$的常数。高斯分布的方差系数是

$
  tilde(beta)_t & = 1 / (alpha_t/beta_t + 1/(1-overline(alpha)_(t-1))) \
  & = (1-overline(alpha)_(t-1)) / (1-overline(alpha)_t) beta_t space space space space space space colblue("（"beta_t=1-alpha_t"）") \
  & = ((1-overline(alpha)_(t-1))(1-alpha_t)) / (1-overline(alpha)_t)
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
  & = 1/sqrt(alpha_t) (x_t - (1-alpha_t)/sqrt(1-overline(alpha)_t)epsilon)
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

直接最大化左边是不可行的，也就是不可求导，无法优化。所以DDPM通过最大化变分下界来优化目标。所以损失函数得到以下结果，其中用$p_theta (x_(t-1)|x_t)$近似$q(x_(t-1)|x_t,x_0)$。

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
device = "cuda" if torch.cuda.is_available() else "cpu"


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

def _pos_encoding(time_idx, output_dim, device="cpu"):
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

根据式@epsilon近似 ，$epsilon$可以近似表示为 $nabla_(x_t) log p(x_t)$。值得注意的是，$epsilon$ 与 $nabla_(x_t) log p(x_t)$之间只相差负常数倍（$-sqrt(1-overline(alpha)_t)$）。这说明以$-sqrt(1-overline(alpha)_t) epsilon$来代替$epsilon$作为训练数据的神经网络也是可行的。

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
device = "cuda" if torch.cuda.is_available() else "cpu"


def show_images(images, labels=None, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            if labels is not None:
                ax.set_xlabel(labels[i].item())
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            i += 1
    plt.tight_layout()
    plt.show()

def _pos_encoding(time_idx, output_dim, device="cpu"):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(timesteps, output_dim, device="cpu"):
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
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

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
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
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
dataset = torchvision.datasets.MNIST(root="./data", download=True, transform=preprocess)
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
    print(f"Epoch {epoch} | Loss: {loss_avg}")

# plot losses
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
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
  + 先验模型：Decoder-Only Transformer，先验模型的目标是生成高质量的*条件*。
  + 条件扩散模型：UNet
]

== CLIP

为了从文本创建扩散图像（文生图），我们将使用CLIP模型生成的嵌入（文本特征向量，图像特征向量）。从CLIP获得的文本嵌入用于调节先验模型，使其扩散相应的图像嵌入。然后，这些图像嵌入用于调节解码器模型，用来指导解码器生成对应的图像。

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

#figure(
  image("rl-figures/先验模型的训练流程.svg"),
  caption: [先验模型的训练原理],
)

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

#figure(
  image("rl-figures/先验模型的推理流程.svg"),
  caption: [先验模型的推理流程],
)

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

时间步嵌入是扩散的重要组成部分。这是因为不同时间步的图像具有不同的噪声量。为了在我们的模型中利用这些信息，我们将使用正弦位置编码。这些位置编码与Transformer中常用的位置编码相同。主要区别在于，我们的输入时间步很可能不会按顺序排列，并且包含所有可能的时间步，因此我们只需要获取与输入时间步对应的位置编码。

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

        self.register_buffer("pe", pe)

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

然后将时间步通过MLP以获得时间步嵌入。

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


== 条件扩散模型

#danger[
  训练用于生成图片的条件扩散模型时，CLIP模型和先验模型都必须*冻结*！
]

=== 概述

#figure(
  image("unet-decoder.png"),
  caption: [解码器架构],
)

条件扩散模型是模型中生成图像的部分。它通过预测每个时间步应该去除的噪声，并迭代地从噪声图像中去除预测的噪声来实现这一点。

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

对于残差块的归一化部分，我们将使用GroupNorm（组归一化）。之所以使用GroupNorm而不是BatchNorm，是因为它的性能与批次大小无关，这使得它在小批次或可变批次大小上表现更佳。使用GroupNorm也有助于提高训练期间的稳定性。

残差块的激活部分用于实现非线性特性。虽然ReLU通常用于残差网络，但我们将使用SiLU作为激活函数。在Ramachandran等人的论文中，尽管模型和超参数是专门为ReLU设置的，但SiLU的表现优于ReLU和其他激活函数。由于SiLU的简单性以及与ReLU的相似性，因此只需用它来代替ReLU即可轻松实现。

对于卷积，我们将使用`3x3`核大小和SAME填充的Conv2d来保留空间维度。SAME填充的工作原理是在输入的边界添加0，以确保输出形状与未填充时的输入形状相同。第一个卷积会将输入从`d_in`通道投影到`d_out`通道，而第二个卷积则仅保留`d_out`通道。

残差块也将根据输入的嵌入信息进行调节。执行调节的主要方法之一是将嵌入添加到输入中。对于我们的模型，我们将执行线性投影以获得比例和偏差值。然后将输入乘以比例，然后添加偏差。Nichol和Dhariwal表明，与加法相比，使用这种调节方法可以提高FID分数。编写此部分代码时需要注意的一点是，调节嵌入的维度可能比输入的维度少。因此，我们需要在嵌入的末尾添加维度。例如，如果输入的形状为`(B, C, L)`，而嵌入的形状为`(B, C)`，则需要添加一个维度以使其形状为`(B, C, 1)`。

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

最后，通过将ResNet模块的原始输入添加到输出来创建跳跃连接。如果需要更改通道数以匹配输出，则对输入执行`1x1`卷积。这些跳跃连接用于增强特征传播。

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

有多种方法可以为注意力模块实现注意力机制。其中一些选项包括自注意力和交叉注意力。我们将使用的方法是GLIDE模型中用到的两种方法的混合。对于这种方法，我们将从输入和条件信息中获取K和V，并将它们连接在一起以获得最终的K和V。

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

需要注意的是，输入图像的形状为`(B, C, H, W)`，但对于我们的注意力机制，我们希望其形状为`(B, L, C)`。因此，我们需要将`H`和`W`维度合并，并将其与`C`维度转置。执行注意力机制后，我们需要将输出转换回其原始形状。

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

对于注意力条件信息，我们首先需要将文本标题传入文本Transformer编码器。在unCLIP论文中，之所以使用这些文本编码，是因为Ramesh等人认为这有助于学习CLIP无法学习的自然语言知识。但在测试过程中，他们发现这种方法效果不佳，因此这部分是可选的。

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

现在，含噪图像已达到所需的通道数，它们可以与条件信息一起通过UNet层。UNet将包含四个编码器层和解码器层，中间有一个瓶颈层。对于每层的通道数，我们将使用`[1, 2, 4, 8]`作为层通道数与模型通道数的比值。UNet的编码器、解码器和瓶颈层都将包含两个残差块。在瓶颈残差块之间以及编码器和解码器内层的每个残差块之后，放置了注意力块。跳跃连接连接位于编码器和解码器的残差块之间。

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


训练解码器模型的一个重要部分是损失函数。我们的损失函数首先要做的就是对批次中的每个图像进行随机时间步的采样。

```python
timesteps = torch.randint(
    0,
    config.decoder.max_time,
    (image.shape[0],),
    device=config.device,
    dtype=torch.long
)
```

然后使用这些时间步和方差计划值来获取通过前向扩散添加的噪声图像和噪声。

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

然后将噪声图像、时间步、文本标题和文本掩码传递到模型中以获得预测噪声。

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

对于数据集，所有图像的尺寸都从`(28,28)`调整为`(32,32)`。图像也进行了归一化，平均值和标准差设置为训练样本的平均值和标准差。

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

在训练这三个模型时，当使用权重衰减时，我使用AdamW作为优化器，否则使用Adam。

```python
if config.clip.weight_decay == 0:
    optimizer = Adam(clip.parameters(), lr=config.clip.lr)
else:
    optimizer = AdamW(clip.parameters(), lr=config.clip.lr, weight_decay=config.clip.weight_decay)
```

对于所有模型，我还使用了带有线性预热的余弦退火学习率调度器。该调度器在每个epoch结束时更新。

```python
if config.clip.warmup_epochs > 0:
    warmup = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=(1 / config.clip.warmup_epochs), end_factor=1.0, total_iters=(config.clip.warmup_epochs - 1), last_epoch=-1)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.clip.epochs - config.clip.warmup_epochs), eta_min=config.clip.lr_min)
```

所有模型还使用了最大范数为1.0的梯度剪裁，以帮助防止梯度爆炸。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.[model].grad_max_norm)
```

在训练CLIP模型时，训练数据被随机分成训练集和验证集。当某个epoch的验证损失小于或等于前一个最低验证损失时，模型权重会被保存。

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

然后，我们需要以相反的顺序迭代遍历所有时间步，从0到`max_time-1`。我们还需要扩展时间步，以便批次中的每条数据都有一个时间步。

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

之后，我们使用解码器模型来预测时间步$t$的噪声。

```python
# Predicting noise at timestep t with decoder
pred_noise = decoder(img, timesteps, caption=prompt, mask=mask)
```

如果不是最终时间步，我们还需要生成随机噪声$z$。我们需要这个值和$sigma_t$，因为我们的模型直接预测从$x_t$到$x_0$的噪声，而不是从$x_t$到$x_(t-1)$的噪声。因此，模型会预测从$x_t$到$x_0$的噪声，并将从$x_(t-1)$到$x_0$的噪声加回去。这样一来，只有从$x_t$到$x_(t-1)$的噪声被移除。最后一步不需要$sigma_t$和$z$值，因为从$x_t$到$x_0$的预测噪声与从$x_t$到$x_(t-1)$的预测噪声相同。

```python
# Generating random noise
z = torch.randn_like(img) if t > 0 else 0
```

利用生成的噪声、方差计划值和预测噪声，我们可以计算时间步t-1的图像。

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

为了查看模型的结果，我将展示每个标题的反向扩散过程。为了查看这一点，我们创建了一个修改版的`sample_image`函数，用于绘制反向扩散过程中10个时间步的图像。

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

#show: appendices.with("Appendices", hide-parent: false)

#chapter("uv教程", image: image("./orange2.jpg"))


```bash
$ pip install uv
$ uv venv rlhf-env # 创建虚拟环境
$ source rlhf-env/bin/activate # 进入虚拟环境
$ uv pip install jinja2
$ uv pip install pandas
$ uv pip install pyarrow
$ uv pip install pyyaml
$ uv pip install safetensors
$ uv pip install tensorboard
$ uv pip install tokenizers
$ uv pip install torch
$ uv pip install transformers
```

配置国内源

```bash
# 推荐使用清华源
$ echo 'export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"'>> ~/.bashrc

# 让配置立即生效
$ source ~/.bashrc

# 检查环境变量
$ echo $UV_DEFAULT_INDEX
```

```bash
$ uv pip install modelscope
$ uv pip install addict
$ uv pip install datasets==2.21.0
$ uv pip install transformers[serving]

$ modelscope download --model Qwen/Qwen3-0.6B-Base --local_dir ./
```
