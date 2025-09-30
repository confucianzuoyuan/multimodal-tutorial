$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

$$
y = \sigma(wx+b)
$$

$$
\mathcal{L}=(y-\hat{y})^2
$$


求导

$$
\frac{\partial \mathcal{L}}{\partial w} = ?
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = ?
$$

训练数据：$(1,2)$ 和 $(2,3)$

初始化了一个神经网络的参数 $w=2.0, b=3.0$ 。

## 前向传播

中间结果：$m_{1} = wx+b=2.0\times 1.0 + 3.0 = 5.0$

中间结果：$m_{2} = \sigma (m_{1}) = 0.9933071490757268$

中间结果：$m_{3} = m_{2}-2.0=0.9933071490757268-2.0=-1.0066928509242732$

最终结果：$\mathcal{L}=m_{3}^2$

## 反向传播

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial w} &= 2\times m_{3}\times 1 \times \sigma(m_{1})(1-\sigma(m_{1}))\times x \\
&= 2\times(-1.0066928509242732)\times 0.9933071490757268\times(1-0.9933071490757268)\times 1.0 \\
&= -0.013385102246024565
\end{aligned}

$$

$\alpha = 50$

$w = 2.0 - 50 * -0.013385102246024565$

