import numpy as np
import pandas as pd
from tqdm import tqdm

from DataPlot.process import DataBase, DataReader, ParticleSizeDistProcessor
from DataPlot.process.method import Mie_PESD

df = DataBase[['Extinction', 'Scattering', 'Absorption']]
PNSD = DataReader('PNSD_dNdlogdp.csv')
df_ = pd.concat([df, PNSD], axis=1).dropna()

hour_real = []
hour_imaginary = []

process_bar = tqdm(total=len(df_.index), desc="Time process")


for time, ser in df_.head(50).iterrows():
    nMin = 1.33
    nMax = 1.60
    kMin = 0.00
    kMax = 0.60
    spaceSize = 31

    nRange = np.linspace(nMin, nMax, num=spaceSize)
    kRange = np.linspace(kMin, kMax, spaceSize)
    Delta_array = np.zeros((spaceSize, spaceSize))
    # 同一時間除了折射率其餘數據皆相同 因此在折射率的迴圈外
    bext_mea = ser['Extinction']
    bsca_mea = ser['Scattering']
    babs_mea = ser['Absorption']

    dp = ParticleSizeDistProcessor().dp
    for ki, k in enumerate(kRange):
        for ni, n in enumerate(nRange):
            m = n + (1j * k)
            ndp = np.array(ser[3:])

            ext_dist, sca_dist, abs_dist = Mie_PESD(m, 550, dp, ndp)

            bext_cal = sum(ext_dist) * 0.014
            bsca_cal = sum(sca_dist) * 0.014
            babs_cal = sum(abs_dist) * 0.014

            Delta_array[ni][ki] = ((babs_mea - babs_cal) / (18.23)) ** 2 + ((bsca_mea - bsca_cal) / 83.67) ** 2

    min_delta = Delta_array.argmin()
    next_n = nRange[(min_delta // spaceSize)]
    next_k = kRange[(min_delta % spaceSize)]

    # 將網格變小
    nMin_small = next_n - 0.02 if next_n > 1.33 else 1.33
    nMax_small = next_n + 0.02
    kMin_small = next_k - 0.04 if next_k > 0.04 else 0
    kMax_small = next_k + 0.04
    spaceSize_small = 41

    nRange_small = np.linspace(nMin_small, nMax_small, spaceSize_small)
    kRange_small = np.linspace(kMin_small, kMax_small, spaceSize_small)
    Delta_array_small = np.zeros((spaceSize_small, spaceSize_small))
    # 所有數據與大網格一致所以使用上方便數即可
    for ki, k in enumerate(kRange_small):
        for ni, n in enumerate(nRange_small):
            m = n + (1j * k)
            ndp = np.array(ser[3:])
            ext_dist, sca_dist, abs_dist = Mie_PESD(m, 550, dp, ndp)

            bext_cal = sum(ext_dist) * 0.014
            bsca_cal = sum(sca_dist) * 0.014
            babs_cal = sum(abs_dist) * 0.014

            Delta_array_small[ni][ki] = ((bext_mea - bext_cal) / (18.23)) ** 2 + ((bsca_mea - bsca_cal) / 83.67) ** 2

    min_delta_small = Delta_array_small.argmin()
    hour_real.append(nRange_small[(min_delta_small // spaceSize_small)])
    hour_imaginary.append(kRange_small[(min_delta_small % spaceSize_small)])

    process_bar.update(1)

data = pd.DataFrame({'real': hour_real, 'imaginary': hour_imaginary}, index=df_.index[:50])
