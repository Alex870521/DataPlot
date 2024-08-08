from datetime import datetime as dtm
from pathlib import Path

from DataPlot import *

# sys.path.insert(1, '/path/to/the/package')

start, end = dtm(2024, 4, 1), dtm(2024, 6, 28)

path_raw = Path('/Users/chanchihyu/NTU/KSvis能見度計畫/FS/data')
path_prcs = Path('/Users/chanchihyu/NTU/KSvis能見度計畫/FS/prcs')

# read data
# th_all = Table.reader(path_raw/'th_all',reset=True)
# th_all_oth = Table.reader(path_raw/'th_oth',reset=True)
# dt_th = th_all(start,end)
# dt_th_oth = th_all_oth(start,end)

dt_ae33 = RawDataReader('AE33', path_raw / 'AE33', reset=True, start=start, end=end)
# dt_neph = RawDataReader('NEPH', path_raw / 'neph', reset=True, start=start, end=end, mean_freq='1h', csv_out=True)
# BC1054 = RawDataReader('BC1054', path_raw / 'BC1054', reset=True, start=start, end=end, mean_freq='1h', csv_out=True)
# MA350 = RawDataReader('MA350', path_raw / 'MA350', reset=True, start=start, end=end, mean_freq='1h', csv_out=True)

# process data
# optical
# opt_prcs = Optical(path_out=path_prcs, excel=False, csv=True)

# dt_abs = opt_prcs.absCoe(dt_ae33)
# dt_opt = opt_prcs.basic(dt_abs['abs'], dt_neph['G'], df_ec=dt_abs['eBC'], df_mass=dt_th['PM2.5'], df_no2=dt_th['NO2'],
#                         nam='opt_basic')


# read other data
# with (Path(path_prcs)/'reconstrc_basic.pkl').open('rb') as f:
# 	chem_basic = pkl.load(f)
# with (Path(path_prcs)/'merge_qc_basic.pkl').open('rb') as f:
# 	dt_merge = pkl.load(f)

# IMPROVE
# opt_imp = opt_prcs.IMPROVE(chem_basic['mass'],dt_th['RH'])

# mie
# mie = opt_prcs.Mie(dt_merge['number'],chem_basic['RI']['RI_dry'])
