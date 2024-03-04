from Mie import MieQ
import numpy as np
import math
import pandas as pd


def getQ_optical(m,dp_SMPS,dp_APS):                     #計算不同粒徑的Qext並儲存在Q_ext中
    Q_ext = np.zeros((130,7))
    for i in range(0,110):
        Q_ext[i] = MieQ(m,550,dp_SMPS[i])
    for k in range(0,20):
        Q_ext[110+k] = MieQ(m,550,dp_APS[k])
    return Q_ext

ext_12                     = dfext.loc[3675:4346,0:]
re_12                      = df2_Recon.loc[:,1]                    # =df2_Recon.value[:,1]
time1                      = ext_12.values[:,0]                     #array of object
Ext_from_NEPH_AE33         = ext_12.values[:,1].astype('float')     #array of float
Sca_from_NEPH_AE33         = ext_12.values[:,2].astype('float')
Abs_from_NEPH_AE33         = ext_12.values[:,9].astype('float')
Ext_Recon                  = re_12.values.astype('float')
# Ext_of_PM1_Mie             = np.array(Mie_ext_PM1)                  #list to array
# Ext_of_PM25_Mie            = np.array(Mie_ext_PM25)
# Contribution_of_PM1        = Ext_of_PM1_Mie/Ext_of_PM25_Mie*100
hour_real      = np.zeros((672,))
hour_imaginary = np.zeros((672,))


for T in range(1,50):
    nMin=1.00
    nMax=1.60
    kMin=0.01
    kMax=0.07
    spaceSize=31
    
    nRange       = np.linspace(nMin,nMax,num = spaceSize)
    kRange       = np.linspace(kMin,kMax,spaceSize)
    Delta_array  = np.zeros((spaceSize,spaceSize))
    #同一時間除了折射率其餘數據皆相同 因此在折射率的迴圈外
    bext_mea     = Ext_from_NEPH_AE33[T-1]
    bsca_mea     = Sca_from_NEPH_AE33[T-1]
    babs_mea     = Abs_from_NEPH_AE33[T-1]
    
    Const_array  = np.full((130,1),(math.pi)/(2.303*4))
    ddp_array    = np.array(ddp).reshape((130,1))
    dp_All_array = np.array(dp_All).reshape((130,1))
           
    SMPS_array   = df2_SMPS.values[T,1:111].astype('float').reshape((110,1))
    APS_array    = df2_APS.values[T,4:24].astype('float').reshape((20,1))
    nlogdp_array = np.concatenate([SMPS_array,APS_array])
    for ki,k in enumerate(kRange):
        for ni,n in enumerate(nRange):
            m            = n + (1j*k) 
            Q            = getQ_optical(m,dp_SMPS,dp_APS)
            Q_ext        = Q[:,0]                              #存Qext的部分
            Q_sca        = Q[:,1]
            Q_abs        = Q[:,2]
            Q_ext_array  = Q_ext.reshape((130,1))
            Q_sca_array  = Q_sca.reshape((130,1))
            Q_abs_array  = Q_abs.reshape((130,1))
            
            Ext_array    = Const_array * Q_ext_array * dp_All_array * 10**(-6) * nlogdp_array * ddp_array
            Sca_array    = Const_array * Q_sca_array * dp_All_array * 10**(-6) * nlogdp_array * ddp_array
            Abs_array    = Const_array * Q_abs_array * dp_All_array * 10**(-6) * nlogdp_array * ddp_array
            bext_cal     = sum(Ext_array)
            bsca_cal     = sum(Sca_array)
            babs_cal     = sum(Abs_array)
            Delta_array[ni][ki] = ((babs_mea-babs_cal)/(18.23))**2 + ((bsca_mea-bsca_cal)/83.67)**2
    
    min_delta            = Delta_array.argmin()
    next_n               = nRange[(min_delta // spaceSize)]
    next_k               = kRange[(min_delta %  spaceSize)]

    #將網格變小
    nMin_small = next_n - 0.02
    nMax_small = next_n + 0.02
    kMin_small = next_k - 0.008
    kMax_small = next_k + 0.008
    spaceSize_small=41
    
    nRange_small = np.linspace(nMin_small,nMax_small,spaceSize_small)
    kRange_small = np.linspace(kMin_small,kMax_small,spaceSize_small)
    Delta_array_small = np.zeros((spaceSize_small,spaceSize_small))
    #所有數據與大網格一致所以使用上方便數即可
    for ki,k in enumerate(kRange_small):
        for ni,n in enumerate(nRange_small):
            m            = n + (1j*k) 
            Q            = getQ_optical(m,dp_SMPS,dp_APS)
            Q_ext        = Q[:,0]                              #存Qext的部分
            Q_sca        = Q[:,1]
            Q_abs        = Q[:,2]
            Q_ext_array  = Q_ext.reshape((130,1))
            Q_sca_array  = Q_sca.reshape((130,1))
            Q_abs_array  = Q_abs.reshape((130,1))
            
            Ext_array    = Const_array * Q_ext_array * dp_All_array * 10**(-6) * nlogdp_array * ddp_array
            Sca_array    = Const_array * Q_sca_array * dp_All_array * 10**(-6) * nlogdp_array * ddp_array
            Abs_array    = Const_array * Q_abs_array * dp_All_array * 10**(-6) * nlogdp_array * ddp_array
            bext_cal     = sum(Ext_array)
            bsca_cal     = sum(Sca_array)
            babs_cal     = sum(Abs_array)
            Delta_array_small[ni][ki] = ((bext_mea-bext_cal)/(18.23))**2 + ((bsca_mea-bsca_cal)/83.67)**2
    
    min_delta_small           = Delta_array_small.argmin()
    hour_real     [T-1] = nRange_small[(min_delta_small // spaceSize_small)]
    hour_imaginary[T-1] = kRange_small[(min_delta_small %  spaceSize_small)]

data = pd.DataFrame({'Time':time1,'real':hour_real,'imaginary':hour_imaginary})
data.to_excel('重建折射率消光_改成散吸光.xls')
