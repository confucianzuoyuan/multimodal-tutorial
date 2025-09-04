import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def verify_gaussian_properties():
    """验证高斯分布的数学性质"""
    print("=" * 60)
    print("一元高斯分布数学性质验证")
    print("=" * 60)
    
    # 生成样本
    mu, sigma = 2, 1.5
    samples = np.random.normal(mu, sigma, 100000)
    
    # 1. 均值和方差
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)
    
    print(f"理论均值: {mu:.3f}")
    print(f"样本均值: {sample_mean:.3f}")
    print(f"误差: {abs(mu - sample_mean):.6f}")
    print()
    
    print(f"理论方差: {sigma**2:.3f}")
    print(f"样本方差: {sample_var:.3f}")
    print(f"误差: {abs(sigma**2 - sample_var):.6f}")
    print()
    
    # 2. 概率计算
    # P(X ≤ μ) = 0.5
    prob_less_than_mean = norm.cdf(mu, mu, sigma)
    print(f"P(X ≤ μ) = {prob_less_than_mean:.6f} (理论值: 0.5)")
    
    # P(μ-σ ≤ X ≤ μ+σ) ≈ 0.6827
    prob_1sigma = norm.cdf(mu + sigma, mu, sigma) - norm.cdf(mu - sigma, mu, sigma)
    print(f"P(μ-σ ≤ X ≤ μ+σ) = {prob_1sigma:.6f} (理论值: 0.6827)")
    
    # 3. 概率密度函数验证
    def gaussian_pdf(x, mu, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    test_x = mu
    theoretical_pdf = gaussian_pdf(test_x, mu, sigma)
    scipy_pdf = norm.pdf(test_x, mu, sigma)
    
    print(f"\n在x={test_x}处的概率密度:")
    print(f"手工计算: {theoretical_pdf:.6f}")
    print(f"SciPy计算: {scipy_pdf:.6f}")
    print(f"误差: {abs(theoretical_pdf - scipy_pdf):.10f}")

verify_gaussian_properties()