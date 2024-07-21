from datetime import datetime as dtm
from pathlib import Path

from DataPlot.dataProcess import *
from DataPlot.rawDataReader import *

# sys.path.insert(1, '/path/to/the/package')

start, end = dtm(2023, 12, 18), dtm(2023, 12, 20)

path_raw = Path('_RawData')
path_prcs = Path('other')

# read data
dt_th = RawDataReader('Table', path_raw / 'th_all', reset=True, start=start, end=end)
dt_th_oth = RawDataReader('Table', path_raw / 'th_oth', start=start, end=end)

dt_teom = RawDataReader('TEOM', path_raw / 'teom', reset=True, start=start, end=end, mean_freq='1h', csv_out=True)
dt_ocec_lcres = RawDataReader('OCEC_LCRES', path_raw / 'ocec', reset=True, start=start, end=end, mean_freq='1h',
                              csv_out=True)
dt_ocec_res = RawDataReader('OCEC_RES', path_raw / 'ocec', reset=True, start=start, end=end, mean_freq='1h',
                            csv_out=True)

# process data
# chemical
chem_prcs = Chemistry(path_out=path_prcs, excel=False, csv=True)

dt_teom = chem_prcs.TEOM_basic(dt_teom, df_check=None)
dt_ocec = chem_prcs.OCEC_basic(dt_ocec_lcres, dt_ocec_res, df_mass=dt_th['PM2.5'], least_square_range=(0.1, 5.0, 0.1),
                               nam='ocec_0.1-5')

# reconstruc
chem_basic = chem_prcs.ReConstrc_basic(dt_th[['NH4+', 'SO42-', 'NO3-', 'Fe', 'Na+']],
                                       dt_ocec['basic'][['Optical_OC', 'Optical_EC']],
                                       df_ref=dt_th['PM2.5'], df_water=dt_th_oth['ALWC'])

# isoropia
dt_iso = chem_prcs.ISOROPIA(
    dt_th[['Na+', 'SO42-', 'NH4+', 'NO3-', 'Cl-', 'Ca2+', 'K+', 'Mg2+', 'NH3', 'HNO3', 'HCl', 'RH', 'AT']])

# ## partition
dt_part = chem_prcs.Partition(dt_th[['NH4+', 'SO42-', 'NO3-', 'Cl-', 'NO2', 'HNO3', 'SO2', 'NH3', 'HCl', 'AT']])

# EPA
# epa = EPA_vertical.reader(path_raw/'epa',reset=False,append_data=True)
# dt_epa = epa(start,end,csv_out=True)
