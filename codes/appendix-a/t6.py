import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'WenQuanYi Zen Hei'  # 替换为你选择的字体

def demonstrate_natural_noise():
    """演示自然界中的高斯噪声"""
    
    # 模拟相机传感器噪声
    clean_signal = np.ones(1000) * 100  # 理想信号值
    
    # 各种噪声源
    thermal_noise = np.random.normal(0, 2, 1000)      # 热噪声
    shot_noise = np.random.normal(0, 1.5, 1000)       # 散粒噪声  
    read_noise = np.random.normal(0, 1, 1000)         # 读取噪声
    
    # 总噪声 = 各种噪声的叠加
    total_noise = thermal_noise + shot_noise + read_noise
    noisy_signal = clean_signal + total_noise
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 噪声分布
    ax1.hist(total_noise, bins=50, alpha=0.7, density=True)
    ax1.set_title('真实世界噪声分布')
    ax1.set_xlabel('噪声值')
    ax1.set_ylabel('概率密度')
    
    # 拟合高斯分布
    mu, sigma = np.mean(total_noise), np.std(total_noise)
    x = np.linspace(-10, 10, 100)
    gaussian = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*((x-mu)/sigma)**2)
    ax1.plot(x, gaussian, 'r-', linewidth=2, label='理论高斯分布')
    ax1.legend()
    
    # 信号对比
    ax2.plot(clean_signal[:100], 'b-', label='理想信号', linewidth=2)
    ax2.plot(noisy_signal[:100], 'r-', alpha=0.7, label='含噪信号')
    ax2.set_title('信号 vs 噪声信号')
    ax2.set_xlabel('样本')
    ax2.set_ylabel('信号值')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"噪声均值: {mu:.3f}")
    print(f"噪声标准差: {sigma:.3f}")
    print("可以看到，多种噪声叠加后确实呈现高斯分布！")

demonstrate_natural_noise()