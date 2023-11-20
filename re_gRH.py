import numpy as np
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pandas import read_csv


PATH_MAIN = Path("C:/Users/alex/PycharmProjects/DataPlot/Data")

with open(PATH_MAIN / 'RH80.csv', 'r', encoding='utf-8', errors='ignore') as f:
    df = read_csv(f, parse_dates=['x'])

x_data = np.array(df['x'], dtype='float')
y_data = np.array(df['gRH'], dtype='float')

def lognormal_pdf(x, mu, sigma, scale):
    return scale * lognorm.pdf(x, sigma, loc=0, scale=np.exp(mu))


# def muti_lognormal_pdf(x, g1, g2, g3):
#     log1 = lognormal_pdf(x, *g1)
#     log2 = lognormal_pdf(x, *g2)
#     log3 = lognormal_pdf(x, *g3,)
#     return log1 + log2 + log3

# 猜測初始參數值
initial_guess = [0.03, 1, 1, 0.8, 1, 1]
x_plot = np.linspace(0.01, 2, 1000)

def cost(parameters):
    g_0 = parameters[:3]

    g_1 = parameters[3:6]

    return np.sum(np.power(lognormal_pdf(x_data, *g_0) + lognormal_pdf(x_data, *g_1) - y_data, 2)) / len(x_data)

result = optimize.minimize(cost, initial_guess)
# 使用curve_fit進行擬合
# params, cov = curve_fit(lognormal_pdf, x_data, y_data, p0=initial_guess, maxfev=1400000)

# 提取擬合參數

# mu, sigma, scale, mu2, sigma2, scale2, mu3, sigma3, scale3,= params

# 打印擬合參數
# print('mu:', mu)
# print('sigma:', sigma)
# print('scale:', scale)
#
# # 繪製數據和擬合曲線
#
y_plot = cost(result.x)
fig, ax = plt.subplots(1, 1, dpi=150, constrained_layout=True)
plt.scatter(x_data, y_data, label='data')
plt.xlim(0.01, )
ax.set_xscale('log')
plt.plot(x_plot, y_plot, 'r-', label='fit')
plt.legend()
plt.show()



# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import optimize
# from pytictoc import TicToc
# t = TicToc()
# def g(x, A, μ, σ):
#     return A / (σ * math.sqrt(2 * math.pi)) * np.exp(-(x - μ) ** 2 / (2 * σ ** 2))
# def f(x):
#     return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)
# A = 100.0  # intensity
# μ = 4.0  # mean
# σ = 4.0  # peak width
# n = 500  # Number of data points in signal
# x = np.linspace(-10, 10, n)
# y = g(x, A, μ, σ) + np.random.randn(n)
# t.tic()  # start clock
# def cost(parameters):
#     g_0 = parameters[:3]
#
#     g_1 = parameters[3:6]
#
#     return np.sum(np.power(g(x, *g_0) + g(x, *g_1) - y, 2)) / len(x)
#
# initial_guess = [5, 10, 4, -5, 10, 4]
# result = optimize.minimize(cost, initial_guess)
# g_0 = [250.0, 4.0, 5.0]
# g_1 = [20.0, -5.0, 1.0]
# x = np.linspace(-10, 10, n)
# y = g(x, *g_0) + g(x, *g_1) + np.random.randn(n)
# fig, ax = plt.subplots()
# ax.scatter(x, y, s=1)
# ax.config(x, g(x, *g_0))
# ax.config(x, g(x, *g_1))
# ax.config(x, g(x, *g_0) + g(x, *g_1))
# ax.config(x, y)
# t.toc()  # stop clock and print elapsed time