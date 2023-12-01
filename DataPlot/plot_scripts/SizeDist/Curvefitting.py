# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:05:51 2023
@author: shinj
"""
import pandas as pd
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math
from numpy import log as ln

# Reset time index & Read file
time = pd.date_range(start = '2020-09-04 2300', end = '2021-05-31 2300', freq = '1h')
df = pd.read_excel('Merge_Size distribution.xlsx', sheet_name = 2).set_index(time)

# Define Event & Clean
dic = {'Event': df.loc[df['Ext'] >= df['Ext'].quantile(0.8)], 
	   'Clean': df.loc[df['Ext'] <= df['Ext'].quantile(0.2)]}

# Mean & SD of each size bin during event & clean period
dp = df.columns[2:232].astype(float)
dlogdp = ln(dp)
df_event_mean  = dic['Event'][dp].mean()
df_event_std   = dic['Event'][dp].std()
df_clean_mean  = dic['Clean'][dp].mean()
df_clean_std   = dic['Clean'][dp].std()
Enhancement    = df_event_mean/df_clean_mean


def fitting(dp, dist, cut=None):
    Num = np.sum(dist * dlogdp)
    data = dist / Num

    # 定義多個對數常態分佈的函數
    def lognorm_func(dp, N1, mu1, sigma1, N2, mu2, sigma2, N3, mu3, sigma3, N4, mu4, sigma4):
        return (N1 / (np.log(sigma1) * np.sqrt(2 * np.pi)) * np.exp(-(np.log(dp) - np.log(mu1)) ** 2 / (2 * np.log(sigma1) ** 2)) +
                N2 / (np.log(sigma2) * np.sqrt(2 * np.pi)) * np.exp(-(np.log(dp) - np.log(mu2)) ** 2 / (2 * np.log(sigma2) ** 2)) +
                N3 / (np.log(sigma3) * np.sqrt(2 * np.pi)) * np.exp(-(np.log(dp) - np.log(mu3)) ** 2 / (2 * np.log(sigma3) ** 2)) +
                N4 / (np.log(sigma4) * np.sqrt(2 * np.pi)) * np.exp(-(np.log(dp) - np.log(mu4)) ** 2 / (2 * np.log(sigma4) ** 2)))

    # 使用 curve_fit 函數進行擬合
    x = np.arange(1, len(data) + 1)
    popt, pcov = curve_fit(lognorm_func, dp, data, p0=[9.59e7, 35, 1.84, 1.34e8, 233.8, 1.62, 1e8, 800, 2.03, 5e7, 2500, 2.8])
    perr = np.sqrt(np.diag(pcov))
    print(perr)

    # 獲取擬合的參數
    N1, mu1, sigma1_fit, N2, mu2, sigma2_fit, N3, mu3, sigma3_fit, N4, mu4, sigma4_fit = popt

    print("擬合結果:")
    print("第一個對數常態分佈:")
    print("Number 1:", N1 * Num)
    print("平均值 (mu1):", mu1)
    print("標準差 (sigma1):", sigma1_fit)
    print()
    print("第二個對數常態分佈:")
    print("Number 2:", N2 * Num)
    print("平均值 (mu2):", mu2)
    print("標準差 (sigma2):", sigma2_fit)
    print()
    print("第三個對數常態分佈:")
    print("Number 3:", N3 * Num)
    print("平均值 (mu3):", mu3)
    print("標準差 (sigma3):", sigma3_fit)
    print()
    print("第四個對數常態分佈:")
    print("Number 4:", N4 * Num)
    print("平均值 (mu4):", mu4)
    print("標準差 (sigma4):", sigma4_fit)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 27
    plt.rcParams["mathtext.fontset"] = 'custom'
    fig, ax1 = plt.subplots(1,1, figsize=(10,9.5), dpi = 200, constrained_layout = True)
    plt.plot(dp, Num*lognorm_func(dp, N1, mu1, sigma1_fit, N2, mu2, sigma2_fit, N3, mu3, sigma3_fit, N4, mu4, sigma4_fit), color='#c41b1b', zorder=10, lw=2.5)  # 擬合曲線 #1d6f0b #c41b1b  
    plt.errorbar(x=dp, y=dist, yerr=df_event_std, fmt="o",color="r",ecolor='r', elinewidth=1, capsize = 4, markersize = 12, alpha=0.45, markerfacecolor='none')
    
    plt.xlim(10,20000)
    plt.ylim(bottom = 0)

    font_label = {'family': 'Times New Roman','weight': 'bold','size': 35}
    plt.title('Surface-based PSDs', weight='bold', fontsize = 50, fontname = 'Times New Roman', pad = 15)
    plt.xlabel('', **font_label)
    # plt.ylabel('dS/dlogdp',labelpad = 10, **font_label)
    plt.yticks([0,0.4E9,0.8E9,1.2e9,1.6e9])
    plt.grid(which='minor', alpha=0.5)
    plt.grid(which='major', alpha=0.5)
    ax = plt.gca()
    ax.tick_params(axis='both', length = 8)
    ax.tick_params(which='minor', length = 6)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,5), useMathText = True)
    prop_legend = {'family':'Times New Roman','weight': 'normal','size': 22}
    plt.legend(['Fitting curve (Event)','Mean ± SD (Event)'], prop = prop_legend, loc='upper right', frameon=False) #bbox_to_anchor=(1.03, 0.89)

    # Add yaxis on the right hand side
    ax2 = ax1.twinx()
    ax2.plot(dp,Enhancement,'b',linewidth=3.87,linestyle="dashed")
    ax2.set_ylabel('Enhancement ratio',**font_label,color = 'b', rotation = 270, labelpad = 40)
    ax2.tick_params(axis = 'y',labelcolor = 'b',length = 10)
    ax2.spines['right'].set_color('blue')
    ax2.set_ylim(bottom = 0)
    ax2.set_yticks([0,4,8,12,16])

    plt.semilogx()
    plt.savefig('CurveFit_EventPSSDs.png', transparent = True, bbox_inches="tight")
    plt.show()

fitting(dp, df_event_mean)