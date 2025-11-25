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

#part("强化学习")

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



=== 时间步位置编码

由于训练网络需要时间步的信息，所以我们需要将时间步进行编码然后注入到网络中。

时间步信息通过正弦位置编码注入网络：

$
  bold("v")_i = cases(
    sin (t/(10000^(i/D))) ",  " i"为偶数时", ,
    cos (t/(10000^(i/D))) ",  " i"为奇数时"
  )
$

可以看到编码方式和Transformer中的几乎一样。


```python
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
```

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



完整的 U-Net 结果如下


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
        """unet的输入：图片x_t和时间步t"""

        # 时间步的位置编码嵌入
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        x1 = self.down1(x, v) # 下采样
        x = self.maxpool(x1) # 最大池化
        x2 = self.down2(x, v) # 下采样
        x = self.maxpool(x2) # 最大池化

        x = self.bot1(x, v)

        x = self.upsample(x) # 上采样
        x = torch.cat([x, x2], dim=1) # 跳跃连接
        x = self.up2(x, v) # 上采样
        x = self.upsample(x) # 上采样
        x = torch.cat([x, x1], dim=1) # 跳跃连接
        x = self.up1(x, v) # 上采样
        x = self.out(x)
        return x
```

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

首先，如果 $z tilde cal(N) (mu,sigma^2)$ 的话，那么有下面的结论

$
  z = mu + sigma epsilon space "其中" epsilon tilde cal(N)(0, 1)
$

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

$
  x_t &= sqrt(1-beta_t)x_(t-1) + sqrt(beta)_t colred(epsilon_(t-1)) \
  &= sqrt(alpha_t) markrect(x_(t-1), color: #red, tag: #<xtminus1>) + sqrt(1-alpha_t) colred(epsilon_(t-1)) \
  #linebreak()
  #linebreak()
  #linebreak()
  #linebreak()
  #linebreak()
  &= sqrt(alpha_t)markrect((sqrt(alpha_(t-1))x_(t-2) + sqrt(1-alpha_(t-1)) colred(epsilon_(t-2))), color: #red, tag: #<xtminus2>) + sqrt(1-alpha_t) colred(epsilon_(t-1)) \
  &= sqrt(alpha_t alpha_(t-1))x_(t-2) + markrect(sqrt(alpha_t (1 - alpha_(t-1))) colred(epsilon_(t-2)) + sqrt(1-alpha_t) colred(epsilon_(t-1)), color: #red, tag: #<xtminus3>) &&\
  #linebreak()
  #linebreak()
  #linebreak()
  && colred("how?")
  #linebreak()
  #linebreak()
  #linebreak()
  &= sqrt(alpha_t alpha_(t-1))x_(t-2) + markrect(sqrt(1-alpha_t alpha_(t-1)) colred(overline(epsilon)_(t-2)), color: #red, tag: #<xtminus4>) \
  & space dots.v \
  &= sqrt(alpha_t alpha_(t-1) dots.c alpha_1)x_0 + sqrt(1-alpha_t alpha_(t-1) dots.c alpha_1)epsilon \
  &= sqrt(overline(alpha)_t)x_0 + sqrt(1-overline(alpha)_t)epsilon
$

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

但是我们如何从第 4 行跳到第 5 行呢？

也就是公式中的 #text(red)[how?] 怎么解决呢？

#figure(
  image("重参数技巧.svg"),
  caption: [重参数技巧的使用],
)

我们用 $X$ 和 $Y$ 来表示这两个项。它们可以被视为来自两个不同正态分布的样本。即

$
  X tilde cal(N)(0,alpha_t(1-alpha_(t-1))I)
$

和

$
  Y tilde cal(N)(0,(1-alpha_(t))I)
$

回想一下，两个正态分布（独立）随机变量的和也是正态分布的。即，如果 $Z=X+Y$ ，那么有下面的公式

$
  Z tilde cal(N)(0, sigma^2_X+sigma^2_Y)
$

因此，我们可以将它们合并在一起，并以重新参数化的形式表示合并后的正态分布。这就是我们将这两个项合并的方法。

重复这些步骤将为我们提供以下仅取决于输入图像 $x_0$ 的公式：

$
  x_t = sqrt(overline(alpha)_t)x_(0)+sqrt(1-overline(alpha)_t) epsilon
$

现在我们可以使用此公式在任何时间步骤直接对 $x_t$ 进行采样，这使得前向过程更快。

== 逆向扩散过程

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

代码的主要修改有两处。在代码❶处，我们准备了处理标签的嵌人层（`nn.Embedding`）。具体来说，就是通过`nn.Embedding(num_labels, time_embed_dim)`将共计`num_labels`个不同整数变换为`time_embed_dim`维的向量。在代码❷处，通过`nn.Embedding`层处理标签，然后再加到变换后的时间数据的向量中。

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

❷ `labels.to(device`

在device上准备标签数据。

❸ `model(x_noisy, t, labels)`

向模型提供labels进行训练。

以上就是条件扩散模型的实现。现在运行代码。经过10轮训练后，最终生成了下图所示的图像。

虽然仍有改进的余地，但生成的图像是符合条件的。在下一节中，我们将探讨进一步改进这个条件扩散模型的方法。

== 得分函数

在上面的内容里面，我们实现了一个简单的条件扩散模型。这个"简单"的意思是我们只是向模型提供了条件而已。因此，模型有可能会轻视我们提供的条件，甚至在最坏的情况下有可能会忽略我们提供的条件。接下来，我们将介绍一种称为"指引"的方法。指引是一种将给定条件纳入模型并给予更多重视的机制。

要理解指引机制，首先需要了解得分函数。

=== 什么是得分函数

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
分类器指引的理念是使用得分和分类器来表示条件得分。通常，我们会向分类器指引中引人权重$gamma$，这样就能调整分类器的贡献度。引入后的数学式如下所示。

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

经过以上修改之后，我们就完成了无分类器指引的实现。这里以$gamma = 3$（`gamma=3.0`）声称了数据。

无分类器指引可以通过$gamma$来调整条件的重视程度。而且由于无需另外训练分类器，因此在资源和时间方面更为高效。

== Stable Diffusion

到目前为止，我们已经实现了处理MNIST的小型扩散模型以及小型条件扩散模型。当然，现代的扩散模型规模相当庞大，而且经过了多项改进。下面是一些进一步改进扩散模型的指引。这里我们介绍一个著名的模型——Stable Diffusion，它能够生成下图所示的高清图像。另外，它的代码和预训练的权重数据是公开的，任何人都可以运行这些代码。

#figure(
  image("sd-image.png"),
  caption: [Stable Diffusion生成的高清图像],
) <sd-image>

```python
# $#text[@sd-image]$
```

