from datetime import datetime as dtm
from pathlib import Path

from DataPlot.rawDataReader import *

# sys.path.insert(1, '/path/to/the/package')

start = dtm(2024, 5, 21)
end = dtm(2024, 6, 28)

path_raw = Path('/Users/chanchihyu/PycharmProjects/DataPlot/data')
path_prcs = Path('prcs')

## read data
# th_all = Table.reader(path_raw/'th_all',reset=True)
# th_all_oth = Table.reader(path_raw/'th_oth',reset=True)
# dt_th = th_all(start,end)
# dt_th_oth = th_all_oth(start,end)

# dt_neph = NEPH.reader(path_raw/'neph',reset=True)(start,end,mean_freq='1h',csv_out=True)
dt_ae33 = AE33.Reader(path_raw / 'FS_AE33', reset=True)(start, end, mean_freq='1h', csv_out=True)
# BC1054 = BC1054.reader(path_raw / 'BC1054', reset=True)
# MA350 = MA350.reader(path_raw / 'MA350', reset=True)

## process data
## optical
# opt_prcs = Optical(path_out=path_prcs, excel=False, csv=True)

# dt_abs = opt_prcs.absCoe(dt_ae33)
# dt_opt = opt_prcs.basic(dt_abs['abs'], dt_neph['G'], df_ec=dt_abs['eBC'], df_mass=dt_th['PM2.5'], df_no2=dt_th['NO2'],
#                         nam='opt_basic')


## read other data
# with (Path(path_prcs)/'reconstrc_basic.pkl').open('rb') as f:
# 	chem_basic = pkl.load(f)
# with (Path(path_prcs)/'merge_qc_basic.pkl').open('rb') as f:
# 	dt_merge = pkl.load(f)

## IMPROVE
# opt_imp = opt_prcs.IMPROVE(chem_basic['mass'],dt_th['RH'])

## mie
## https://pymiescatt.readthedocs.io/en/latest/
# mie = opt_prcs.Mie(dt_merge['number'],chem_basic['RI']['RI_dry'])
