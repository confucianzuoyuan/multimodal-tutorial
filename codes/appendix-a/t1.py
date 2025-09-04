import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_univariate_gaussian():
    """一元高斯分布可视化"""
    x = np.linspace(-10, 10, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('一元高斯分布特性分析', fontsize=16, fontweight='bold')
    
    # 1. 不同均值，相同方差
    ax1 = axes[0, 0]
    means = [0, 2, -2]
    variance = 1
    colors = ['blue', 'red', 'green']
    
    for mu, color in zip(means, colors):
        y = norm.pdf(x, mu, np.sqrt(variance))
        ax1.plot(x, y, color=color, linewidth=2, 
                label=f'μ={mu}, σ²={variance}')
        ax1.axvline(mu, color=color, linestyle='--', alpha=0.7)
    
    ax1.set_title('不同均值的影响 (σ² = 1)', fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('概率密度 f(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 相同均值，不同方差
    ax2 = axes[0, 1]
    mu = 0
    variances = [0.5, 1, 2, 4]
    colors = ['purple', 'blue', 'orange', 'red']
    
    for var, color in zip(variances, colors):
        y = norm.pdf(x, mu, np.sqrt(var))
        ax2.plot(x, y, color=color, linewidth=2, 
                label=f'μ={mu}, σ²={var}')
    
    ax2.set_title('不同方差的影响 (μ = 0)', fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('概率密度 f(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 累积分布函数 (CDF)
    ax3 = axes[1, 0]
    mu, sigma = 0, 1
    y_pdf = norm.pdf(x, mu, sigma)
    y_cdf = norm.cdf(x, mu, sigma)
    
    ax3.plot(x, y_pdf, 'b-', linewidth=2, label='概率密度函数 (PDF)')
    ax3.plot(x, y_cdf, 'r-', linewidth=2, label='累积分布函数 (CDF)')
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.7)
    
    ax3.set_title('PDF vs CDF (标准正态分布)', fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('概率')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 68-95-99.7规则可视化
    ax4 = axes[1, 1]
    mu, sigma = 0, 1
    x_fine = np.linspace(-4, 4, 1000)
    y = norm.pdf(x_fine, mu, sigma)
    ax4.plot(x_fine, y, 'b-', linewidth=2, label='N(0,1)')
    
    # 填充不同标准差范围
    x_1sigma = x_fine[np.abs(x_fine) <= 1]
    y_1sigma = norm.pdf(x_1sigma, mu, sigma)
    ax4.fill_between(x_1sigma, y_1sigma, alpha=0.3, color='green', 
                     label='±1σ (68.27%)')
    
    x_2sigma = x_fine[np.abs(x_fine) <= 2]
    y_2sigma = norm.pdf(x_2sigma, mu, sigma)
    ax4.fill_between(x_2sigma, y_2sigma, alpha=0.2, color='orange', 
                     label='±2σ (95.45%)')
    
    x_3sigma = x_fine[np.abs(x_fine) <= 3]
    y_3sigma = norm.pdf(x_3sigma, mu, sigma)
    ax4.fill_between(x_3sigma, y_3sigma, alpha=0.1, color='red', 
                     label='±3σ (99.73%)')
    
    ax4.set_title('68-95-99.7规则', fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('概率密度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.rcParams['font.sans-serif'] = 'WenQuanYi Zen Hei'  # 替换为你选择的字体
    plt.tight_layout()
    plt.show()

# 运行可视化
plot_univariate_gaussian()