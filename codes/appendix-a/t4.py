import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from scipy.stats import multivariate_normal

plt.rcParams['font.sans-serif'] = 'WenQuanYi Zen Hei'  # 替换为你选择的字体


def demonstrate_gaussian_properties():
    """演示高斯分布的优良性质"""
    print("=" * 60)
    print("高斯分布在扩散模型中的优势")
    print("=" * 60)

    # 1. 中心极限定理
    print("1. 中心极限定理验证:")
    sample_sizes = [1, 5, 10, 30]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, n in enumerate(sample_sizes):
        # 从均匀分布中采样，然后求均值
        samples = []
        for _ in range(10000):
            uniform_samples = np.random.uniform(0, 1, n)
            samples.append(np.mean(uniform_samples))

        axes[i].hist(samples, bins=50, density=True, alpha=0.7,
                     color='skyblue', edgecolor='black')

        # 理论正态分布
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        x = np.linspace(min(samples), max(samples), 100)
        y = norm.pdf(x, sample_mean, sample_std)
        axes[i].plot(x, y, 'r-', linewidth=2, label='理论正态分布')

        axes[i].set_title(f'样本大小 n={n}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('中心极限定理：均匀分布的样本均值趋向正态分布', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 2. 可加性
    print("\n2. 高斯分布的可加性:")
    mu1, sigma1 = 1, 2
    mu2, sigma2 = 3, 1.5

    # 两个独立高斯分布的和
    mu_sum = mu1 + mu2
    sigma_sum = np.sqrt(sigma1**2 + sigma2**2)

    print(f"X₁ ~ N({mu1}, {sigma1}²)")
    print(f"X₂ ~ N({mu2}, {sigma2}²)")
    print(f"X₁ + X₂ ~ N({mu_sum}, {sigma_sum:.3f}²)")

    # 验证
    samples1 = np.random.normal(mu1, sigma1, 100000)
    samples2 = np.random.normal(mu2, sigma2, 100000)
    samples_sum = samples1 + samples2

    print(f"理论均值: {mu_sum}, 实际均值: {np.mean(samples_sum):.3f}")
    print(f"理论标准差: {sigma_sum:.3f}, 实际标准差: {np.std(samples_sum):.3f}")


demonstrate_gaussian_properties()
