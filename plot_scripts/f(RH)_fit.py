from os.path import join as pth
import numpy as np
import math
import pandas as pd
from pandas import date_range, read_csv, read_excel, concat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


PATH_MAIN = Path('Data')

with open(PATH_MAIN/'f(RH).csv', 'r', encoding='utf-8', errors='ignore') as f:
    frh = read_csv(f).set_index('RH')[['fRH', 'fRHSS', 'fRHs', 'fRHl']]


split = [36, 75]
x1 = frh.index.to_numpy()[split[0]:split[1]]
y1 = frh['fRHs'].values[split[0]:split[1]]
x2 = frh.index.to_numpy()[split[1]:]
y2 = frh['fRHs'].values[split[1]:]


def f__RH(RH, a, b, c):
    f = a + b * (RH/100)**c
    return f


popt, pcov = curve_fit(f__RH, x1, y1)
a, b, c = popt
yvals = f__RH(x1, a, b, c) #擬合y值
print(u'係數a:', a)
print(u'係數b:', b)
print(u'係數c:', c)

popt2, pcov2 = curve_fit(f__RH, x2, y2)
a2, b2, c2 = popt2
yvals2 = f__RH(x2, a2, b2, c2) #擬合y值
print(u'係數a2:', a2)
print(u'係數b2:', b2)
print(u'係數c2:', c2)

fig, axes = plt.subplots(1, 1, figsize=(5, 5), dpi=150, constrained_layout=True)
plot1 = plt.scatter(x1, y1, label='original values')
plot2 = plt.plot(x1, yvals, 'r', label='polyfit values')
plot3 = plt.scatter(x2, y2, label='original values')
plot4 = plt.plot(x2, yvals2, 'r', label='polyfit values')
plt.xlabel(r'$\bf RH$')
plt.ylabel(r'$\bf f(RH)$')
plt.legend(loc='best')
plt.title(r'$\bf Curve fit$')
plt.savefig('test.png')
plt.show()
