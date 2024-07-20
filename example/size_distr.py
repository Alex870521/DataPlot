import sys
from pathlib import Path

# sys.path.insert(1,str(Path('C:/')/'Users'/os.getlogin()/'Desktop'/'yrr'/'program'/'my'/'application'/'ContainerHandle'))

# sys.path.insert(1,str(Path('pkg')))

sys.path.insert(1, str(Path('ContainerHandle-main')))

from ContainerHandle.rawDataReader import *
from ContainerHandle.dataProcess import *
from datetime import datetime as dtm

start = dtm(2024, 2, 1, 0)
end = dtm(2024, 4, 30, 23, 54)
path_raw = Path('_RawData')

# dt_smps = pd.read_csv(path_raw/("smps")/('tunnel')/('test')/r"_read_smps_th_raw.csv",parse_dates=['time']).set_index(['time'])

## read data
# th_all = Table.reader(path_raw/'th_all',reset=True)
# th_all_oth = Table.reader(path_raw/'th_oth',reset=True)
# dt_th = th_all(start,end)
# dt_th_oth = th_all_oth(start,end)

# smps = SMPS_TH.reader(path_raw/'NZ_SMPS',reset=True)
aps = APS_3321.reader(path_raw / 'FS_APS', reset=True)
# smps = SMPS_genr.reader(path_raw/'NZ_SMPS', reset=True, rate=True, update_meta=dict(freq='1.5T'), QC=True, )
# dt_smps = smps(start,end,mean_freq='1h')
dt_aps = aps(start, end, mean_freq='1h')

## process data
# # # ## size 2023-12-03 16:15:00
path_prcs = Path('prcs_FSaps')
distr_prcs = SizeDistr(path_prcs, excel=False, csv=True)

# dt_smps_prcs = distr_prcs.basic(dt_smps,nam='distr_smps')
dt_aps_prcs = distr_prcs.basic(dt_aps, nam='distr_aps', unit='um')

## merge

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
