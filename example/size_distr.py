from datetime import datetime as dtm
from pathlib import Path

from DataPlot.dataProcess import *
from DataPlot.rawDataReader import *

# sys.path.insert(1, '/path/to/the/package')

start, end = dtm(2024, 2, 1), dtm(2024, 4, 30, 23, 54)

path_raw = Path('/Users/chanchihyu/NTU/KSvis能見度計畫/FS/data')
path_prcs = Path('/Users/chanchihyu/NTU/KSvis能見度計畫/FS/prcs')

# read data
dt_smps = RawDataReader('SMPS_TH', path_raw / 'NZ_SMPS', reset=True, start=start, end=end, mean_freq='1h', csv_out=True)
dt_smps = RawDataReader('SMPS_genr', path_raw / 'NZ_SMPS', reset=True, start=start, end=end, mean_freq='1h',
                        csv_out=True)
dt_aps = RawDataReader('APS_3321', path_raw / 'FS_APS', reset=True, start=start, end=end, mean_freq='1h')

# process data
# size 2023-12-03 16:15:00
path_prcs = Path('prcs_FSaps')
distr_prcs = SizeDistr(path_prcs, excel=False, csv=True)

# dt_smps_prcs = distr_prcs.basic(dt_smps,nam='distr_smps')
dt_aps_prcs = distr_prcs.basic(dt_aps, nam='distr_aps', unit='um')

# merge

# '''
# merge_prcs   = distr_prcs.merge_SMPS_APS(dt_smps,dt_aps,nam='merge_data')
# # merge_prcs   = distr_prcs.merge_SMPS_APS_v2(dt_smps,dt_aps,nam='merge_data')
# dt_merge_prcs = distr_prcs.basic(merge_prcs['data_all'],nam='merge_all_basic')
# dt_merge_prcs = distr_prcs.basic(merge_prcs['data_qc'],nam='merge_qc_basic')
# merge_dt = merge_prcs['data_qc']
# pm1, pm25 = merge_dt.keys()[merge_dt.keys()<1000], merge_dt.keys()[merge_dt.keys()<2500]
# pm1_minus = merge_dt.keys()[merge_dt.keys()>1000]
# pm0_1 = merge_dt.keys()[merge_dt.keys()<100]

# pm0_1_bsc = distr_prcs.basic(merge_dt[pm0_1], nam='pm01_basic')
# pm1_bsc  = distr_prcs.basic(merge_dt[pm1], nam='pm1_basic')
# pm25_bsc = distr_prcs.basic(merge_dt[pm25], nam='pm25_basic')
# pm1_minus_bsc = distr_prcs.basic(merge_dt[pm1_minus], nam='bigger_pm1_basic')
# '''
