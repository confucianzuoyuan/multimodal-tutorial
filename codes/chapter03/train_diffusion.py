import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Optional, Sequence
from tqdm import tqdm

# ================================
# 1. 扩散过程相关函数
# ================================


def get_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    生成线性的beta调度表，从beta_start到beta_end，共timesteps步

    Args:
        timesteps: 扩散步数
        beta_start: 起始beta值
        beta_end: 结束beta值

    Returns:
        torch.Tensor: beta调度表
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    前向扩散过程：给原始图像x_0添加噪声，得到时间步t的噪声图像x_t

    Args:
        x_0: 原始清晰图像
        t: 时间步
        sqrt_alphas_cumprod: sqrt(alpha_bar_t)，图像保留比例
        sqrt_one_minus_alphas_cumprod: sqrt(1-alpha_bar_t)，噪声比例

    Returns:
        x_t: 噪声图像
        noise: 添加的噪声
    """
    # 生成与输入图像相同形状的高斯噪声
    noise = torch.randn_like(x_0)

    # 获取当前时间步的缩放因子，调整维度以匹配图像(B,C,H,W)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(
        -1, 1, 1, 1)

    # 应用扩散公式: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise
    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t, noise


@torch.no_grad()
def denoising_step(x_t, t, model, alphas, alphas_cumprod, betas):
    """
    反向去噪过程：从x_t预测x_{t-1}

    Args:
        x_t: 当前时间步的噪声图像
        t: 当前时间步
        model: 训练好的UNet模型
        alphas: alpha调度表
        alphas_cumprod: alpha累积乘积
        betas: beta调度表

    Returns:
        x_{t-1}: 去噪后的图像
    """
    # 使用模型预测噪声
    predicted_noise = model(x_t, t)

    # 获取当前时间步的参数，调整维度匹配图像
    alpha_t = alphas[t].view(-1, 1, 1, 1)
    beta_t = betas[t].view(-1, 1, 1, 1)
    alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1, 1)

    # 计算噪声的标准差
    sigma_t = torch.sqrt(beta_t)

    # 应用DDPM反向采样公式计算均值
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
    )

    # 如果不是最后一步，添加随机噪声；最后一步输出确定性结果
    if t[0] > 0:
        z = torch.randn_like(x_t)
        return mean + sigma_t * z
    else:
        return mean

# ================================
# 2. 时间位置编码模块
# ================================


class TimePositionalEncoding(nn.Module):
    """
    计算正弦位置编码，用于时间步嵌入
    类似Transformer中的位置编码：
        PE[t, 2i]   = sin(t / 10000^{2i/d})
        PE[t, 2i+1] = cos(t / 10000^{2i/d})
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: 嵌入维度（必须是偶数）
        """
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: 形状为(batch,)的时间步张量

        Returns:
            形状为(batch, dim)的正弦嵌入
        """
        device = timesteps.device
        half_dim = self.dim // 2

        # 计算指数衰减因子
        scale = math.log(10000) / (half_dim - 1)
        exponents = torch.exp(torch.arange(half_dim, device=device) * -scale)

        # 计算正弦和余弦值
        args = timesteps[:, None] * exponents[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)

        return emb

# ================================
# 3. UNet构建模块
# ================================


class UNetStage(nn.Module):
    """
    UNet的单个编码器或解码器阶段

    下采样阶段: (B, C_in, H, W) -> (B, C_out, H/2, W/2)
    上采样阶段: (B, 2*C_in, H, W) -> (B, C_out, 2H, 2W)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        use_label_cond: bool = False,
        kernel_size: int = 3,
        downsample: bool = True
    ):
        """
        Args:
            in_ch: 输入通道数
            out_ch: 输出通道数
            time_dim: 时间嵌入维度
            use_label_cond: 是否使用标签条件
            kernel_size: 卷积核大小
            downsample: True为下采样，False为上采样
        """
        super().__init__()

        self.use_label_cond = use_label_cond
        self.is_down = downsample

        # 时间编码器
        self.time_encoder = TimePositionalEncoding(time_dim)
        self.time_proj = nn.Linear(time_dim, out_ch)

        # 标签条件投影（如果启用）
        if use_label_cond:
            self.label_proj = nn.Linear(1, out_ch)

        # 卷积层
        if downsample:
            # 下采样：保持空间尺寸 -> 缩小空间尺寸
            self.conv_in = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1)
            self.spatial = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
        else:
            # 上采样：连接跳跃连接，因此输入通道翻倍
            self.conv_in = nn.Conv2d(2 * in_ch, out_ch, kernel_size, padding=1)
            self.spatial = nn.ConvTranspose2d(
                out_ch, out_ch, 4, stride=2, padding=1)

        # 批归一化和激活函数
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv_feat = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入特征图
            timesteps: 时间步张量，形状(B,)
            labels: 标签张量，形状(B, 1)（如果启用标签条件）

        Returns:
            变换后的特征图
        """
        # 第一个卷积层
        out = self.bn1(self.act(self.conv_in(x)))

        # 添加时间嵌入
        t_emb = self.act(self.time_proj(self.time_encoder(timesteps)))
        out = out + t_emb[..., None, None]

        # 添加标签嵌入（如果启用）
        if self.use_label_cond:
            if labels is None:
                raise ValueError("启用了标签条件但未提供标签")
            lbl_emb = self.act(self.label_proj(labels))
            out = out + lbl_emb[..., None, None]

        # 第二个卷积层和空间变换
        out = self.bn2(self.act(self.conv_feat(out)))
        return self.spatial(out)

# ================================
# 4. 完整的UNet模型
# ================================


class DiffusionUNet(nn.Module):
    """
    用于扩散模型的UNet架构
    支持时间步条件和可选的标签条件
    """

    def __init__(
        self,
        img_channels: int = 3,
        time_dim: int = 128,
        label_conditioning: bool = False,
        channel_sequence: Sequence[int] = (64, 128, 256, 512, 1024)
    ):
        """
        Args:
            img_channels: 输入/输出图像通道数
            time_dim: 时间嵌入维度
            label_conditioning: 是否启用标签条件
            channel_sequence: 特征通道宽度序列
        """
        super().__init__()

        self.time_dim = time_dim
        self.label_conditioning = label_conditioning

        # 输入卷积层
        self.stem = nn.Conv2d(img_channels, channel_sequence[0], 3, padding=1)

        # 下采样路径（编码器）
        self.down_blocks = nn.ModuleList([
            UNetStage(c_in, c_out, time_dim,
                      label_conditioning, downsample=True)
            for c_in, c_out in zip(channel_sequence, channel_sequence[1:])
        ])

        # 上采样路径（解码器）
        rev_channels = channel_sequence[::-1]
        self.up_blocks = nn.ModuleList([
            UNetStage(c_in, c_out, time_dim,
                      label_conditioning, downsample=False)
            for c_in, c_out in zip(rev_channels, rev_channels[1:])
        ])

        # 输出投影层
        self.head = nn.Conv2d(channel_sequence[0], img_channels, 1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入图像批次，形状(B, C, H, W)
            timesteps: 时间步索引，形状(B,)
            labels: 标签张量，形状(B, 1)（如果启用标签条件）

        Returns:
            与输入x相同形状的输出张量
        """
        skips = []
        h = self.stem(x)

        # 编码器路径（存储跳跃连接）
        for down in self.down_blocks:
            h = down(h, timesteps, labels=labels)
            skips.append(h)

        # 解码器路径（使用反向的跳跃连接）
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(torch.cat([h, skip], dim=1), timesteps, labels=labels)

        return self.head(h)

# ================================
# 5. 可视化函数
# ================================


def plot_noise_distribution(noise, predicted_noise):
    """绘制真实噪声和预测噪声的分布对比"""
    plt.figure(figsize=(10, 6))
    plt.hist(noise.cpu().numpy().flatten(), density=True,
             alpha=0.7, label="真实噪声", bins=50)
    plt.hist(predicted_noise.cpu().numpy().flatten(), density=True,
             alpha=0.7, label="预测噪声", bins=50)
    plt.xlabel("噪声值")
    plt.ylabel("密度")
    plt.title("噪声分布对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_noise_prediction(noise, predicted_noise):
    """并排显示真实噪声和预测噪声"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 转换为可显示的格式
    def tensor_to_image(tensor):
        img = (tensor + 1) / 2  # 从[-1,1]缩放到[0,1]
        img = img.permute(1, 2, 0)  # CHW到HWC
        img = torch.clamp(img, 0, 1)
        return img.cpu().numpy()

    axes[0].imshow(tensor_to_image(noise))
    axes[0].set_title("真实噪声")
    axes[0].axis('off')

    axes[1].imshow(tensor_to_image(predicted_noise))
    axes[1].set_title("预测噪声")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_generation_process(generated_images):
    """显示生成过程的中间结果"""
    fig, axes = plt.subplots(1, len(generated_images),
                             figsize=(len(generated_images) * 3, 3))
    if len(generated_images) == 1:
        axes = [axes]

    for ax, (step, image) in zip(axes, generated_images):
        ax.imshow(image)
        ax.set_title(f"步骤 {step}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# ================================
# 6. 主训练和生成代码
# ================================


def main():
    """主函数：训练模型并生成图像"""

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 超参数
    T = 1000  # 扩散步数
    IMAGE_SHAPE = (48, 48)  # 图像尺寸
    LR = 0.001  # 学习率
    NO_EPOCHS = 1000  # 训练轮数
    BATCH_SIZE = 256  # 批大小
    PRINT_FREQUENCY = 10  # 打印频率
    VERBOSE = False  # 是否显示详细信息

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SHAPE),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),  # 缩放到[-1, 1]
    ])

    # 加载图像
    photo_path = "image.jpg"  # 替换为你的图像路径
    try:
        image = Image.open(photo_path)
        transformed_image = transform(image).to(device)
        print(f"成功加载图像: {photo_path}")
    except FileNotFoundError:
        print(f"未找到图像文件: {photo_path}")
        print("将使用随机图像进行演示")
        transformed_image = torch.randn(3, *IMAGE_SHAPE).to(device)

    # 初始化扩散调度表
    betas = get_beta_schedule(T).to(device)
    alphas = (1. - betas).to(device)
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    # 初始化模型和优化器
    unet = DiffusionUNet(img_channels=3, label_conditioning=False).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=LR)

    print(f"模型参数数量: {sum(p.numel() for p in unet.parameters()):,}")

    # 训练循环
    print("开始训练...")
    unet.train()

    for epoch in range(NO_EPOCHS):
        # 创建批次数据
        batch = torch.stack([transformed_image] * BATCH_SIZE)
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

        # 前向扩散：添加噪声
        batch_noisy, noise = forward_diffusion_sample(
            batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
        )

        # 模型预测噪声
        predicted_noise = unet(batch_noisy, t)

        # 计算损失并更新参数
        optimizer.zero_grad()
        loss = F.mse_loss(noise, predicted_noise)
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if epoch % PRINT_FREQUENCY == 0:
            print(f"轮次 {epoch:4d} | 训练损失: {loss.item():.6f}")

            # 可视化预测结果（如果启用详细模式）
            if VERBOSE:
                with torch.no_grad():
                    plot_noise_prediction(noise[0], predicted_noise[0])
                    plot_noise_distribution(noise, predicted_noise)

    print("训练完成！")

    # 生成新图像
    print("开始生成图像...")
    generated_images = []
    show_every = 100  # 每隔多少步保存一次中间结果

    with torch.no_grad():
        unet.eval()
        # 从纯噪声开始
        img = torch.randn((1, 3) + IMAGE_SHAPE).to(device)

        # 逐步去噪
        for i in tqdm(reversed(range(T)), desc="去噪进度"):
            t_tensor = torch.full((1,), i, dtype=torch.long, device=device)
            img = denoising_step(img, t_tensor, unet,
                                 alphas, alphas_cumprod, betas)

            # 保存中间结果
            if i % show_every == 0 or i == 0:
                # 转换为PIL图像
                img_display = (img[0] + 1) / 2  # 从[-1,1]到[0,1]
                img_display = torch.clamp(img_display, 0, 1)
                img_display = img_display.permute(1, 2, 0).cpu().numpy()
                img_display = (img_display * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_display)

                generated_images.append((T - i, img_pil))

    # 显示生成过程
    print("显示生成过程...")
    generated_images.sort()
    plot_generation_process(generated_images)

    # 保存最终生成的图像
    final_image = generated_images[-1][1]
    final_image.save("generated_image.png")
    print("最终生成的图像已保存为 'generated_image.png'")


if __name__ == "__main__":
    main()
