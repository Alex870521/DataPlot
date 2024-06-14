import pysplit as py

# TODO: Hybrid Single-Particle Lagrangian Integrated Trajectory (HYSPLIT) model

working_dir = r'/Users/chanchihyu/hysplit/working'
# breakpoint()
storage_dir = r'/Users/chanchihyu/hysplit4'
meteo_dir = r'E:/gdas'

basename = 'colgate'

years = [2007, 2011]
months = [1, 8]
hours = [11, 17, 23]
altitudes = [500, 1000, 1500]
location = (42.82, -75.54)
runtime = -120

py.generate_bulktraj(basename, working_dir, storage_dir, meteo_dir,
                     years, months, hours, altitudes, location, runtime,
                     monthslice=slice(0, 32, 2), get_reverse=True,
                     get_clipped=True)
