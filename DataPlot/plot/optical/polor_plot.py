from os.path import join as pth
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from PyMieScatt import ScatteringFunction
from DataPlot.plot import set_figure

prop_legend = {'size': 12, 'family': 'Times New Roman', 'weight': 'bold'}
textprops = {'fontsize': 12, 'fontfamily': 'Times New Roman', 'fontweight': 'bold'}


@set_figure(figsize=(10, 6))
def polor_plot(inner, outer):
    employee = ['AS', 'AN', 'OM', 'Soil', 'SS', 'BC']
    actual = np.append(inner, inner[0])
    expected = np.append(outer, outer[0])

    plt.figure()
    plt.subplot(polar=True)

    theta = np.linspace(0, 2 * np.pi, len(actual))

    lines, labels = plt.thetagrids(range(0, 360, int(360 / len(employee))), employee)

    line1, = plt.plot(theta, actual, 'o-', linewidth=0.1, color='#115162')
    plt.fill(theta, actual, '#afe0f5', alpha=0.5)
    line2, = plt.plot(theta, expected, 'o-', linewidth=0.1, color='#7FAE80')
    plt.fill(theta, expected, '#b5e6c5', alpha=0.5)

    plt.legend(handles=[line1, line2], labels=['Dry', 'Amb'], prop=prop_legend, loc='best', bbox_to_anchor=(1, 0, 0.2, 1), frameon=False)
    plt.title(r'$\bf clean$')
    plt.show()


if __name__ == '__main__':
    dp = [50, 200, 500]
    m = [1.55 + 0.00*1j]*len(dp)
    wave = [600]*len(dp)

    for _dp, _m, _wave in zip(dp, m, wave):
        theta, SL, SR, SU = ScatteringFunction(_m, _wave, _dp)

        SL_r = np.flipud(SL)
        SR_r = np.flipud(SR)
        SU_r = np.flipud(SU)
        SL_t = np.append(SL, SL_r)
        SR_t = np.append(SR, SR_r)
        SU_t = np.append(SU, SU_r)

        polor_plot(SR_t, SU_t)
