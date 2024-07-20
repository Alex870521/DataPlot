import sys
from pathlib import Path

# sys.path.insert(1,str(Path('C:/')/'Users'/os.getlogin()/'Desktop'/'yrr'/'program'/'my'/'application'/'ContainerHandle'))

sys.path.insert(1, str(Path('ContainerHandle-main')))

from ContainerHandle.rawDataReader import *
from ContainerHandle.dataProcess import *
from datetime import datetime as dtm

start = dtm(2023, 12, 18)
end = dtm(2023, 12, 20)

path_raw = Path('_RawData')
path_prcs = Path('other')

## read data
th_all = Table.reader(path_raw / 'th_all', reset=True)
th_all_oth = Table.reader(path_raw / 'th_oth', reset=True)
dt_th = th_all(start, end)
dt_th_oth = th_all_oth(start, end)

teom = TEOM.reader(path_raw / 'teom', reset=True)
# ocec_lcres = OCEC_LCRES.reader(path_raw/'ocec',reset=True)
# ocec_res   = OCEC_RES.reader(path_raw/'ocec',reset=True)
dt_teom = teom(start, end, mean_freq='1h', csv_out=True)
# dt_ocec_lcres = ocec_lcres(start,end)
# dt_ocec_res = ocec_res(start,end)

## process data
## chemical
chem_prcs = Chemistry(path_out=path_prcs, excel=False, csv=True)

dt_teom = chem_prcs.TEOM_basic(dt_teom, df_check=None)
# dt_ocec = chem_prcs.OCEC_basic(dt_ocec_lcres,dt_ocec_res,df_mass=dt_th['PM2.5'],least_square_range=(0.1,5.0,0.1),nam='ocec_0.1-5')

# ## reconstruc
chem_basic = chem_prcs.ReConstrc_basic(dt_th[['NH4+', 'SO42-', 'NO3-', 'Fe', 'Na+']],
                                       dt_ocec['basic'][['Optical_OC', 'Optical_EC']],
                                       df_ref=dt_th['PM2.5'], df_water=dt_th_oth['ALWC'])

# ## isoropia
dt_iso = chem_prcs.ISOROPIA(
    dt_th[['Na+', 'SO42-', 'NH4+', 'NO3-', 'Cl-', 'Ca2+', 'K+', 'Mg2+', 'NH3', 'HNO3', 'HCl', 'RH', 'AT']])

# ## partition
dt_part = chem_prcs.Partition(dt_th[['NH4+', 'SO42-', 'NO3-', 'Cl-', 'NO2', 'HNO3', 'SO2', 'NH3', 'HCl', 'AT']])

## EPA
# epa = EPA_vertical.reader(path_raw/'epa',reset=False,append_data=True)
# dt_epa = epa(start,end,csv_out=True)
