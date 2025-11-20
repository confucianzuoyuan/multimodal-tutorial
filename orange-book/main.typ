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

在 `forward` 方法中，我们通过 `linear_project/Conv2D` 方法传递具有形状 `(B, C, H, W)` 的输入，并输出形状 `(B, d_model, P_col, P_row)` 的张量。

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
  caption: [左图：使用 `Conv2D` 运算分成 `16` 个 `8x8` 块的 `32x32` MNIST 图像。右图：添加位置编码和类别token后的 `16` 个图像补丁，使用随机数据初始化。],
)

请注意，我们已经用 64 个卷积核初始化了 `Conv2D` 运算，每个卷积核中的每个补丁只占用一个像素，以免扭曲图像。

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

        # Q和K的点积
        attention = Q @ K.transpose(-2,-1)

        # 缩放
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
# Q和K的点积
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
        # 第一次层归一化之后的残差
        out = x + self.mha(self.ln1(x))
        # 第二次层归一化之后的残差
        out = out + self.mlp(self.ln2(out))
        return out
```

Transformer编码器由两个子层组成：第一个子层执行多头注意力，第二个子层包含MLP。多头注意力子层执行token之间的通信，而MLP子层允许token单独"思考"与它们通信的内容。

层归一化是一种优化技术，可跨其特征独立归一化批处理中的每个输入。对于我们的模型，我们将在每个子层的开头通过层归一化模块传递我们的输入。

```python
# 第一个层归一化模块
self.ln1 = nn.LayerNorm(d_model)
# 第二个层归一化模块
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
    # 残差
    out = x + self.mha(self.ln1(x))
    # 残差
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
        # 补丁的数量 = (32x32) // (4x4)
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
        # 前向传播
        outputs = ViT(inputs)
        # 交叉熵损失
        loss = criterion(outputs, labels)
        # 求导数
        loss.backward()
        # 梯度下降
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
        # 计算K，Q，V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Q和K的点积
        attention = Q @ K.transpose(-2,-1)
        # 缩放
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
            AttentionHead(width, self.head_size) for _ in range(n_heads)
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
        # d_model(width) --- emb_dim
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
        # 将文本特征嵌入到联合嵌入空间中（多模态嵌入空间）
        # 文本编码器输出的张量的维度和图像编码器输出的张量的维度必须一致
        if self.projection is not None:
            x = x @ self.projection

        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x
```

对于文本编码器，我们将使用常规Transformer模型。创建文本编码器的第一步是创建大小为 `（vocab_size, width)` 的嵌入表。此嵌入表包含一个向量表示，其大小等于词汇表中每个 token 的 Transformer 模型的 `width` 。

```python
self.encoder_embedding = nn.Embedding(vocab_size, width)
```

在输出 Transformer 的结果之前，我们需要将特征嵌入到联合嵌入空间中。我们将通过获取文本特征的点积以及使用 `nn.Parameter` 创建的可学习的投影来实现这一点。

```python
# 可学习的投影
self.projection = nn.Parameter(torch.randn(width, emb_dim))
```

在 `forward` 方法中，我们要做的第一件事是通过嵌入表传递文本的token。

```python
# 文本嵌入
x = self.encoder_embedding(text)
```

然后，我们需要将位置编码添加到嵌入表的输出中。

```python
# 位置嵌入
x = self.positional_embedding(x)
```

添加位置编码后，我们现在可以将其与掩码一起通过编码器层。

```python
# Transformer编码器
for encoder_layer in self.encoder:
    x = encoder_layer(x, mask=mask)
```

编码器层的输出是文本的特征。我们将使用从 `EOT` 的嵌入中抽取的特征。

```python
# 从EOT的嵌入抽取特征
x = x[torch.arange(text.shape[0]),torch.sub(torch.sum(mask[:,0],dim=1),1)]
```

最后，我们通过计算特征和投影之间的点积，将文本特征嵌入到联合嵌入空间中，并通过除以归一化的点积对其进行归一化。

```python
# 将文本特征嵌入到联合嵌入空间中（多模态嵌入空间）
if self.projection is not None:
    x = x @ self.projection
x = x / torch.norm(x, dim=-1, keepdim=True)
return x
```

=== 图像编码器


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

        # 可学习的投影
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

        # 多模态嵌入
        # 保证文本编码器的输出的维度和图像编码器的输出的维度相等
        if self.projection is not None:
            x = x @ self.projection

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
        # Iₑ是图像嵌入，形状 [B, D=emb_dim]
        I_e = self.image_encoder(image)
        # Tₑ是文本嵌入，形状 [B, D=emb_dim]
        T_e = self.text_encoder(text, mask=mask)

        # 缩放逐点余弦相似度[n, n]
        # 形状 I_e @ T_e^T : [B, D] @ [D, B] --> [B, B]
        logits = (I_e @ T_e.transpose(-2,-1)) * torch.exp(self.temperature)

        # 对称损失函数 labels形状为[B]，值为 [0, 1, 2, ..., B-1]
        labels = torch.arange(logits.shape[0]).to(self.device)
        # 从文本 --> 图像方向，以文本嵌入 T₃ 为例子，
        # 交叉熵损失的目标是让 T₃⋅I₃ 越大越好
        loss_i = nn.functional.cross_entropy(
            logits.transpose(-2,-1),
            labels
        )
        # 从图像 --> 文本方向，以图像嵌入 I₃ 为例子，
        # 交叉熵损失的目标是让 I₃⋅T₃ 越大越好
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
self.dataset = load_dataset("./../datasets/clip-mnist")
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

#tip[
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
        # 归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # I_e @ T_e^T
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

#tip[
  具体步骤如下：
  + 将数据库中的所有图片使用clip的图像编码器进行抽取特征
  + 将抽取的图片特征存入向量数据库
  + 将输入的搜索文本通过clip的文本编码器抽取文本特征
  + 使用余弦相似度找出向量数据库中和文本特征最接近的几张图片
]

#chapter("扩散模型", image: image("./orange2.jpg"), l: "multimodal-chap3")

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

当 $T → ∞$ 时，最终结果将变成完全噪声图像，就像从 *各向同性* 的高斯分布中采样出来的噪声一样。

首先，如果 $z tilde cal(N) (mu, sigma^2)$ 的话，那么正态分布可以写成如下公式：

$
  z=mu+sigma epsilon space #text[其中] epsilon tilde cal(N) (0,1)
$

利用这个技巧，我们可以将采样图像 $x_t$ 表示如下：

$
  x_t = sqrt(1-beta_t) x_(t-1) + sqrt(beta_t) epsilon_(t-1)
$

#danger(title: "一步一步的添加噪声太麻烦了！")[
  根据上面的公式，如果想要从原始图片 $x_0$ 得到添加了 500 步噪声的图片 $x_500$ 需要迭代 500 次！
]

但我们不需要设计一种算法来迭代地向图像中添加噪声，而是可以使用闭式公式（解析解）在特定的时间步长 $t$ 直接对噪声图像进行采样。

给定原始图片 $x_0$ 和时间步 $t$ 可以直接得到添加了 $t$ 步噪声的图像 $x_t$ 。公式如下：

#tip(title: "给定原始图片" + $x_0$ + "和时间步" + $t$ + "直接采样出" + $x_t$ + "的公式")[
  $
    x_t = sqrt(overline(alpha)_t) x_0 + sqrt(1-overline(alpha)_t) epsilon
  $
  其中：
  - $overline(alpha)_t=alpha_t alpha_(t-1) dots alpha_1$
  - $alpha_t = 1 - beta_t$
  - $epsilon tilde cal(N) (0,1)$
]

现在我们可以使用此公式在任何时间步直接对 $x_t$ 进行采样，这使得前向扩散过程更快。

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
    beta_start, # beta_start: 起始β值，论文中等于0.0001
    beta_end, # beta_end: 结束β值，论文中等于0.02
    num_timesteps, # num_timesteps: 时间步的数量：1000
    device=device
)
```

而由于 $α_t=1-beta_t$ ，所以有如下代码

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

#show raw.where(lang: "python"): it => {
  show regex("\$(.*?)\$"): re => {
    eval(re.text, mode: "markup")
  }
  it
}

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

#codly(
  annotations: (
    (
      start: 25,
      end: 31,
      content: block(
        width: auto,
        align(
          left,
        )[$mu_theta (x_t, t)=1/sqrt(alpha_t) (bold("x")_t - (1-alpha_t)/sqrt(1-overline(alpha)_t) epsilon_theta (bold("x")_t,t)) space$],
      ),
    ),
    (
      start: 32,
      end: 34,
      content: block(
        width: auto,
        align(left)[$sigma_t = sqrt(((1-alpha_t)(1-overline(alpha)_(t-1)))/(1-overline(alpha)_t))$],
      ),
    ),
  ),
)
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
	# 均值
	mu = (x                                \
        -                                \
        ((1-alpha)                       \
         /                               \
         torch.sqrt(1-alpha_bar)) * eps) \
	   / torch.sqrt(alpha)
	# 标准差
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
    # 偶数位置
    v[0::2] = torch.sin(t / div_term[0::2])
    # 奇数位置
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

#figure(
  image("u-net-illustration-correct-scale2.svg"),
  caption: [U-net模型],
)

先来编写卷积块。


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
device = 'cuda'


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
        beta_start=0.0001, # 方差的起始值β_0 = 0.0001
        beta_end=0.02, # 方差的最终值β_T = 0.02
        device='cpu'
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
        # α_t = 1 - β_t
        self.alphas = 1 - self.betas
        # α_bar_t = α_t * α_{t-1} * ... * α_1
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
        """去除一步噪声"""
        # x_t, t
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
        # x_{t-1}
        return mu + noise * std

    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 28, 28)):
        """从纯噪声图片x_1000反向扩散出x_0"""
        batch_size = x_shape[0]
        # 采样一张白噪声图片x_1000出来
        x = torch.randn(x_shape, device=self.device)
        # for t = T, T-1, ..., 0
        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor(
                [i] * batch_size,
                device=self.device,
                dtype=torch.long
            )
            # 一步去噪，x_t --> x_{t-1}
            x = self.denoise(model, x, t)

        images = [
            self.reverse_to_img(x[i])
            for i in range(batch_size)
        ]
        return images


preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(
    root='./../datasets',
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
        # x_noisy是x_t，noise是添加的真正的噪声
        x_noisy, noise = diffuser.add_noise(x, t)
        # 模型根据x_t和时间步t，预测给x_t添加的噪声
        noise_pred = model(x_noisy, t)
        # 添加的真实噪声和预测噪声之间进行均方误差计算
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')

# 画出损失
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 从完全噪声的图片反向扩散
images = diffuser.sample(model)
show_images(images)
```

#chapter("条件扩散模型", image: image("./orange2.jpg"), l: "multimodal-chap4")

我们之前对数据$x$的概率$p(x)$进行了建模。但在实用层面，我们更希
望对条件概率$p(x|y)$进行建模（其中$y$表示条件）。如果能成功建立$p(x|y)$的模型，那么就可以通过条件$y$控制想生成的$x$。

条件$y$可以是文本、图像或标签等。如果$y$是一张低分辨率的图像，那么可以考虑将其变换为高分辨率的图像。这就是被称为超分辨率成像（super-resolution imaging）的技术。使用扩散模型进行超分辨率处理的模型包括级联扩散模型（cascaded diffusion model）等。

本节将以$y$作为标签的例子进行说明。具体来说，我们将创建一个模型，在给定MNIST数据的数字标签后，该模型能够生成与该标签相对应的图像。



#part("Part Two Title")

#chapter("Mathematics", image: image("./orange2.jpg"))

== Theorems
#index("Theorems")
=== Several equations<heading2>
#index("Theorems!Several equations")
This is a theorem consisting of several equations.
#theorem(name: "Name of the theorem")[
  In $E=bb(R)^n$ all norms are equivalent. It has the properties:
  $ abs(norm(bold(x)) - norm(bold(y))) <= norm(bold(x-y)) $
  $ norm(sum_(i=1)^n bold(x)_i) <= sum_(i=1)^n norm(bold(x)_i) quad "where" n "is a finite integer" $
]

=== Single Line
#index("Theorems!Single Line")
This is a theorem consisting of just one line.
#theorem()[
  A set $scr(D)(G)$  in dense in $L^2(G)$, $|dot|_0$.
]
== Definitions
#index("Definitions")
A definition can be mathematical or it could define a concept.
#definition(name: "Definition name")[
  Given a vector space $E$, a norm on $E$ is an application, denoted $norm(dot)$, $E$ in $bb(R)^+ = [0,+∞[$ such that:
  $ norm(bold(x)) = 0 arrow.r.double bold(x) = bold(0) $
  $ norm(lambda bold(x)) = abs(lambda) dot norm(bold(x)) $
  $ norm(bold(x) + bold(y)) lt.eq norm(bold(x)) + norm(bold(y)) $
]
== Notations
#index("Notations")

#notation()[
  Given an open subset $G$ of $bold(R)^n$, the set of functions $phi$ are:
  #v(0.5em, weak: true)
  + Bounded support $G$;
  + Infinitely differentiable;
  #v(0.5em, weak: true)
  a vector space is denoted by $scr(D)(G)$.
]
== Remarks
#index("Remarks")
This is an example of a remark.

#remark()[
  The concepts presented here are now in conventional employment in mathematics. Vector spaces are taken over the field $bb(K)=bb(R)$, however, established properties are easily extended to $bb(K)=bb(C)$.
]

== Corollaries
#index("Corollaries")
#corollary(name: "Corollary name")[
  The concepts presented here are now in conventional employment in mathematics. Vector spaces are taken over the field $bb(K)=bb(R)$, however, established properties are easily extended to $bb(K)=bb(C)$.
]
== Propositions
#index("Propositions")
=== Several equations
#index("Propositions!Several equations")

#proposition(name: "Proposition name")[
  It has the properties:
  $ abs(norm(bold(x)) - norm(bold(y))) <= norm(bold(x-y)) $
  $ norm(sum_(i=1)^n bold(x)_i) <= sum_(i=1)^n norm(bold(x)_i) quad "where" n "is a finite integer" $
]
=== Single Line
#index("Propositions!Single Line")

#proposition()[
  Let $f,g in L^2(G)$; if $forall phi in scr(D) (G)$, $(f,phi)_0=(g,phi)_0$ then $f = g$.
]
== Examples
#index("Examples")
=== Equation Example
#index("Examples!Equation")
#example()[
  Let $G=\(x in bb(R)^2:|x|<3\)$ and denoted by: $x^0=(1,1)$; consider the function:

  $
    f(x) = cases(
      e^(abs(x)) quad & "si" |x-x^0| lt.eq 1 slash 2,
      0 & "si" |x-x^0| gt 1 slash 2
    )
  $

  The function $f$ has bounded support, we can take $A={x in bb(R)^2:|x-x^0| lt.eq 1 slash 2+ epsilon}$ for all $epsilon in lr(\] 0\;5 slash 2-sqrt(2) \[, size: #70%)$.
]

=== Text Example
#index("Examples!Text")



== Exercises
#index("Exercises")
#exercise()[
  This is a good place to ask a question to test learning progress or further cement ideas into students' minds.
]
== Problems
#index("Problems")

#problem()[
  What is the average airspeed velocity of an unladen swallow?
]

== Vocabulary
#index("Vocabulary")

Define a word to improve a students' vocabulary.

#vocabulary(name: "Word")[
  Definition of word.
]

#chapter("Presenting Information and Results with a Long Chapter Title", image: image("./orange3.jpg"))
== Table
#index("Table")
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent porttitor arcu luctus, imperdiet
urna iaculis, mattis eros. Pellentesque iaculis odio vel nisl ullamcorper, nec faucibus ipsum molestie.
Sed dictum nisl non aliquet porttitor. Etiam vulputate arcu dignissim, finibus sem et, viverra nisl.
Aenean luctus congue massa, ut laoreet metus ornare in. Nunc fermentum nisi imperdiet lectus
tincidunt vestibulum at ac elit. Nulla mattis nisl eu malesuada suscipit.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [*Treatments*], [*Response 1*], [*Response 2*],
    [Treatment 1], [0.0003262], [0.562],
    [Treatment 2], [0.0015681], [0.910],
    [Treatment 3], [0.0009271], [0.296],
  ),
  caption: [Table caption.],
) <table>

Referencing @table in-text using its label.

== Figure
#index("Figure")

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent porttitor arcu luctus, imperdiet
urna iaculis, mattis eros. Pellentesque iaculis odio vel nisl ullamcorper, nec faucibus ipsum molestie.
Sed dictum nisl non aliquet porttitor. Etiam vulputate arcu dignissim, finibus sem et, viverra nisl.
Aenean luctus congue massa, ut laoreet metus ornare in. Nunc fermentum nisi imperdiet lectus
tincidunt vestibulum at ac elit. Nulla mattis nisl eu malesuada suscipit.

#figure(
  image("creodocs_logo.svg", width: 50%),
  caption: [Figure caption.],
) <figure>

Referencing @figure in-text using its label and referencing @figure1 in-text using its label.

#figure(
  placement: top,
  table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [*Treatments*], [*Response 1*], [*Response 2*],
    [Treatment 1], [0.0003262], [0.562],
    [Treatment 2], [0.0015681], [0.910],
    [Treatment 3], [0.0009271], [0.296],
  ),
  caption: [Floating table.],
) <table1>

#figure(
  placement: bottom,
  image("creodocs_logo.svg", width: 100%),
  caption: [Floating figure.],
) <figure1>

#my-bibliography(bibliography("sample.bib"))

#make-index(title: "Index")

#show: appendices.with("Appendices", hide-parent: false)

#chapter("Appendix Chapter Title", image: image("./orange2.jpg"))

== Appendix Section Title

#lorem(50)
#chapter("Appendix Chapter Title", image: image("./orange2.jpg"))

== Appendix Section Title

#lorem(50)
