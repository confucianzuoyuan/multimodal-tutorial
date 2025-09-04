import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from scipy.stats import multivariate_normal

plt.rcParams['font.sans-serif'] = 'WenQuanYi Zen Hei'  # 替换为你选择的字体

def plot_bivariate_gaussian():
    """二元高斯分布可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('二元高斯分布特性分析', fontsize=16, fontweight='bold')
    
    # 创建网格
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # 不同的协方差矩阵配置
    configs = [
        {'mean': [0, 0], 'cov': [[1, 0], [0, 1]], 'title': '独立变量\n(ρ=0)'},
        {'mean': [0, 0], 'cov': [[1, 0.8], [0.8, 1]], 'title': '正相关\n(ρ=0.8)'},
        {'mean': [0, 0], 'cov': [[1, -0.8], [-0.8, 1]], 'title': '负相关\n(ρ=-0.8)'},
        {'mean': [0, 0], 'cov': [[2, 0], [0, 0.5]], 'title': '不同方差\n(σ₁²=2, σ₂²=0.5)'},
        {'mean': [1, -1], 'cov': [[1, 0], [0, 1]], 'title': '不同均值\n(μ=[1,-1])'},
        {'mean': [0, 0], 'cov': [[1, 0.5], [0.5, 2]], 'title': '一般情况\n(混合效应)'}
    ]
    
    for i, config in enumerate(configs):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # 创建多元正态分布
        rv = multivariate_normal(config['mean'], config['cov'])
        
        # 计算概率密度
        pdf_values = rv.pdf(pos)
        
        # 绘制等高线
        contour = ax.contour(X, Y, pdf_values, levels=8, colors='black', alpha=0.6, linewidths=1)
        contourf = ax.contourf(X, Y, pdf_values, levels=20, cmap='viridis', alpha=0.8)
        
        # 添加颜色条
        plt.colorbar(contourf, ax=ax, shrink=0.8)
        
        # 标记均值点
        ax.plot(config['mean'][0], config['mean'][1], 'r*', markersize=15, 
                markeredgecolor='white', markeredgewidth=1)
        
        # 设置标题和标签
        ax.set_title(config['title'], fontweight='bold')
        ax.set_xlabel('X₁')
        ax.set_ylabel('X₂')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

plot_bivariate_gaussian()

def plot_3d_gaussian():
    """3D可视化二元高斯分布"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # 创建网格
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # 三种不同的配置
    configs = [
        {'mean': [0, 0], 'cov': [[1, 0], [0, 1]], 'title': '独立变量 (ρ=0)'},
        {'mean': [0, 0], 'cov': [[1, 0.8], [0.8, 1]], 'title': '正相关 (ρ=0.8)'},
        {'mean': [0, 0], 'cov': [[1, -0.8], [-0.8, 1]], 'title': '负相关 (ρ=-0.8)'}
    ]
    
    for i, config in enumerate(configs):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # 创建分布并计算PDF
        rv = multivariate_normal(config['mean'], config['cov'])
        Z = rv.pdf(pos)
        
        # 3D表面图
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # 添加等高线投影
        ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', alpha=0.5)
        
        ax.set_title(config['title'], fontweight='bold')
        ax.set_xlabel('X₁')
        ax.set_ylabel('X₂')
        ax.set_zlabel('概率密度')
    
    plt.tight_layout()
    plt.show()

plot_3d_gaussian()

def analyze_covariance_matrix():
    """分析协方差矩阵的数学性质"""
    print("=" * 60)
    print("协方差矩阵分析")
    print("=" * 60)
    
    # 定义几个协方差矩阵
    covariance_matrices = {
        '独立变量': np.array([[1, 0], [0, 1]]),
        '正相关': np.array([[1, 0.8], [0.8, 1]]),
        '负相关': np.array([[1, -0.8], [-0.8, 1]]),
        '不同方差': np.array([[2, 0], [0, 0.5]])
    }
    
    for name, cov in covariance_matrices.items():
        print(f"\n{name}:")
        print(f"协方差矩阵:\n{cov}")
        
        # 计算特征值和特征向量
        eigenvals, eigenvecs = np.linalg.eig(cov)
        print(f"特征值: {eigenvals}")
        print(f"特征向量:\n{eigenvecs}")
        
        # 计算相关系数
        correlation = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        print(f"相关系数 ρ: {correlation:.3f}")
        
        # 计算行列式（用于概率密度函数）
        det = np.linalg.det(cov)
        print(f"行列式 |Σ|: {det:.3f}")
        
        print("-" * 40)

analyze_covariance_matrix()