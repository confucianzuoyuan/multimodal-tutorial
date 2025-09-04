import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from scipy.stats import multivariate_normal

plt.rcParams['font.sans-serif'] = 'WenQuanYi Zen Hei'  # 替换为你选择的字体


def simulate_diffusion_process():
    """模拟扩散过程"""
    print("\n3. 扩散过程模拟:")

    # 原始图像（简化为1D信号）
    original_signal = np.sin(np.linspace(0, 4*np.pi, 100)) + \
        0.5*np.cos(np.linspace(0, 8*np.pi, 100))

    # 扩散参数
    T = 1000  # 总步数
    beta_start, beta_end = 1e-4, 0.02
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)

    # 前向扩散过程
    timesteps_to_show = [0, 100, 300, 500, 999]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, t in enumerate(timesteps_to_show):
        if t == 0:
            noisy_signal = original_signal
        else:
            # 直接采样 x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise
            noise = np.random.normal(0, 1, len(original_signal))
            noisy_signal = np.sqrt(
                alpha_bars[t]) * original_signal + np.sqrt(1 - alpha_bars[t]) * noise

        axes[i].plot(original_signal, 'b-', alpha=0.7, label='原始信号')
        axes[i].plot(noisy_signal, 'r-', alpha=0.8, label=f'噪声信号 (t={t})')
        axes[i].set_title(
            f'时间步 t={t}\n信噪比: {10*np.log10(alpha_bars[t]/(1-alpha_bars[t])):.1f} dB' if t > 0 else '原始信号')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # 最后一个子图显示噪声调度
    axes[-1].plot(betas, label='β_t (噪声调度)')
    axes[-1].plot(alpha_bars, label='ᾱ_t (累积信号保持率)')
    axes[-1].set_xlabel('时间步')
    axes[-1].set_ylabel('值')
    axes[-1].set_title('扩散调度参数')
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)

    plt.suptitle('扩散过程：从信号到噪声', fontsize=14)
    plt.tight_layout()
    plt.show()


simulate_diffusion_process()


def demonstrate_reparameterization():
    """演示重参数化技巧"""
    print("\n4. 重参数化技巧:")
    print("目标：从 N(μ, σ²) 采样")
    print("方法：z = μ + σ * ε, 其中 ε ~ N(0, 1)")

    # 参数
    mu, sigma = 2.5, 1.8
    n_samples = 10000

    # 方法1：直接采样
    samples_direct = np.random.normal(mu, sigma, n_samples)

    # 方法2：重参数化
    epsilon = np.random.normal(0, 1, n_samples)  # 标准正态分布
    samples_reparam = mu + sigma * epsilon

    # 比较结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 标准正态分布 ε
    axes[0].hist(epsilon, bins=50, density=True, alpha=0.7,
                 color='lightblue', edgecolor='black')
    x = np.linspace(-4, 4, 100)
    axes[0].plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
    axes[0].set_title('标准正态分布 ε ~ N(0,1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 直接采样结果
    axes[1].hist(samples_direct, bins=50, density=True, alpha=0.7,
                 color='lightgreen', edgecolor='black')
    x = np.linspace(mu-4*sigma, mu+4*sigma, 100)
    axes[1].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                 label=f'N({mu},{sigma}²)')
    axes[1].set_title('直接采样')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 重参数化结果
    axes[2].hist(samples_reparam, bins=50, density=True, alpha=0.7,
                 color='lightcoral', edgecolor='black')
    axes[2].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                 label=f'N({mu},{sigma}²)')
    axes[2].set_title('重参数化: μ + σ * ε')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('重参数化技巧演示', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 统计验证
    print(
        f"直接采样 - 均值: {np.mean(samples_direct):.3f}, 标准差: {np.std(samples_direct):.3f}")
    print(
        f"重参数化 - 均值: {np.mean(samples_reparam):.3f}, 标准差: {np.std(samples_reparam):.3f}")
    print(f"理论值 - 均值: {mu}, 标准差: {sigma}")


demonstrate_reparameterization()
