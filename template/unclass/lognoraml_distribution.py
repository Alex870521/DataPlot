import numpy as np
import matplotlib.pyplot as plt
from config.custom import figure_set
# 定义分布的参数
geoMean = 0.8
geoStd = 1.5


mu = np.log(geoMean)
sigma = np.log(geoStd)

# 生成等距分布的数据点
x = np.geomspace(0.01, 10, 10000)
y = np.log(x)
y2 = np.log10(x)

# 计算在 x 和 y 上的概率密度函数
fx = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
fy = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(y - mu)**2 / (2 * sigma**2))
fy2 = (1 / (np.log10(geoStd) * np.sqrt(2 * np.pi))) * np.exp(-(y2 - np.log10(geoMean))**2 / (2 * np.log10(geoStd)**2))
print(x[fy.argmax()])
# 绘制概率密度函数
fig, ax1 = plt.subplots(figsize=(8,8))
plt.plot(x, fx, label='Normal Scale', lw=4)
plt.plot(x, fy, label='Ln Scale', lw=4)
plt.plot(x, fy2, label='Log Scale', lw=4)
# 添加图例和轴标签
plt.legend()
plt.title(rf'$\bf geoMean = {geoMean}, geoStd = {geoStd}$')
plt.xlabel('X')
plt.ylabel('Density')
ax1.set_xlim(x.min(), x.max())
plt.semilogx()


