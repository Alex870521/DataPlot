from pandas import read_csv

from .core import AbstractReader


class reader(AbstractReader):
    nam = 'AE43'

    def _raw_reader(self, _file):
        with open(_file, 'r', encoding='utf-8', errors='ignore') as f:
            _df = read_csv(f, parse_dates={'time': ['StartTime']}, index_col='time')
            _df_id = _df['SetupID'].iloc[-1]

            ## get last SetupID data
            _df = _df.groupby('SetupID').get_group(_df_id)[
                ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'Status']].copy()

            ## remove data without Status=0
            _df = _df.where(_df['Status'] == 0).copy()

            return _df[['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7']]

    ## QC data
    def _QC(self, _df):
        ## remove negative value
        _df = _df.mask((_df < 0).copy())

        ## call by _QC function
        ## QC data in 1 hr
        def _QC_func(_df_1hr):
            _df_ave = _df_1hr.mean()
            _df_std = _df_1hr.std()
            _df_lowb, _df_highb = _df_1hr < (_df_ave - _df_std * 1.5), _df_1hr > (_df_ave + _df_std * 1.5)

            return _df_1hr.mask(_df_lowb | _df_highb).copy()

        return _df.resample('1h').apply(_QC_func).resample('5T').mean()
