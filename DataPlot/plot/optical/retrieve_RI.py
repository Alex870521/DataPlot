from Mie import MieQ
import numpy as np
import math
import pandas as pd
# from numba import njit
import time
start = time.time()
df2_SMPS  = pd.read_excel("C:\\Users\\alex\\OneDrive\\桌面\\202102月SMPS_dN逐時數據.xlsx",header=None,sheet_name=0)
df2_APS   = pd.read_excel("C:\\Users\\alex\\OneDrive\\桌面\\202102月APS_dN逐時數據.xlsx",header=None,sheet_name=0)
df2_Recon = pd.read_excel("C:\\Users\\alex\\OneDrive\\桌面\\第2次期中報告\\IMPR_Teom(RH)\\IMPR2_NEW.xlsx",header=None,sheet_name=0)
dfext     = pd.read_excel("C:\\Users\\alex\\OneDrive\\桌面\\數據整理.xlsx",header=None,sheet_name=0)

dp_SMPS  = [11.8,12.2,12.6,13.1,13.6,14.1,14.6,15.1,15.7,16.3,16.8,17.5,18.1,18.8,19.5,20.2,20.9,21.7,22.5,23.3,24.1,25,25.9,26.9,27.9,28.9,30,31.1,32.2,33.4,34.6,35.9,37.2,38.5,40,41.4,42.9,44.5,46.1,47.8,49.6,51.4,53.3,55.2,57.3,59.4,61.5,63.8,66.1,68.5,71,73.7,76.4,79.1,82,85.1,88.2,91.4,94.7,98.2,101.8,105.5,109.4,113.4,117.6,121.9,126.3,131,135.8,140.7,145.9,151.2,156.8,162.5,168.5,174.7,181.1,187.7,194.6,201.7,209.1,216.7,224.7,232.9,241.4,250.3,259.5,269,278.8,289,299.6,310.6,322,333.8,346,358.7,371.8,385.4,399.5,414.2,429.4,445.1,461.4,478.3,495.8,514,532.8,552.3,572.5,593.5]
dp_APS   = [626,673,723,777,835,898,965,1037,1114,1197,1286,1382,1486,1596,1715,1843,1981,2129,2288,2458]
dp_All   = [11.8,12.2,12.6,13.1,13.6,14.1,14.6,15.1,15.7,16.3,16.8,17.5,18.1,18.8,19.5,20.2,20.9,21.7,22.5,23.3,24.1,25,25.9,26.9,27.9,28.9,30,31.1,32.2,33.4,34.6,35.9,37.2,38.5,40,41.4,42.9,44.5,46.1,47.8,49.6,51.4,53.3,55.2,57.3,59.4,61.5,63.8,66.1,68.5,71,73.7,76.4,79.1,82,85.1,88.2,91.4,94.7,98.2,101.8,105.5,109.4,113.4,117.6,121.9,126.3,131,135.8,140.7,145.9,151.2,156.8,162.5,168.5,174.7,181.1,187.7,194.6,201.7,209.1,216.7,224.7,232.9,241.4,250.3,259.5,269,278.8,289,299.6,310.6,322,333.8,346,358.7,371.8,385.4,399.5,414.2,429.4,445.1,461.4,478.3,495.8,514,532.8,552.3,572.5,593.5,626,673,723,777,835,898,965,1037,1114,1197,1286,1382,1486,1596,1715,1843,1981,2129,2288,2458]
ddp      = [0.393,0.400,0.449,0.500,0.500,0.500,0.500,0.549,0.600,0.551,0.598,0.651,0.649,0.700,0.700,0.664,0.785,0.800,0.800,0.800,0.849,0.900,0.949,1.000,1.000,1.049,1.100,1.100,1.149,1.200,1.249,1.323,1.278,1.398,1.451,1.449,1.549,1.600,1.649,1.749,1.800,1.849,1.900,1.999,2.100,2.100,2.199,2.292,2.358,2.449,2.599,2.700,2.700,2.799,2.999,3.101,3.150,3.250,3.399,3.550,3.650,3.799,3.950,4.125,4.223,4.350,4.548,4.750,4.850,5.048,5.250,5.448,5.650,5.848,6.099,6.299,6.499,6.748,6.999,7.282,7.466,7.798,8.100,8.349,8.698,9.049,9.349,9.649,9.998,10.398,10.798,11.198,11.598,11.998,12.448,12.888,13.358,13.848,14.397,14.948,15.448,15.997,16.597,17.197,17.847,18.498,19.147,19.847,20.596,21.382,44.788,48.477,51.962,55.964,60.449,64.970,69.455,74.458,79.944,85.948,92.434,99.921,106.961,114.412,123.418,132.906,142.913,153.401,164.409,176.863]
# @njit(nopython=True)
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

end = time.time()
print (end-start)