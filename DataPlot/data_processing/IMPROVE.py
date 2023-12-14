from pandas import read_csv, concat
from core import DataProcessor, DataReader


class ImproveProcessor(DataProcessor):
    """
    A class for processing improved chemical data.

    Parameters:
    -----------
    reset : bool, optional
        If True, resets the processing. Default is False.
    filename : str, optional
        The name of the file to process. Default is None.
    version : str, optional
        The version of the data processing. Should be one of 'revised' or 'modified'.
        Default is None.

    Methods:
    --------
    revised(_df):
        Calculate revised version of particle contribution.

    modified(_df):
        Calculate modified version of particle contribution.

    gas(_df):
        Calculate gas contribution.

    frh(_RH, version=None):
        Helper function to get frh values based on relative humidity (RH) and version.

    process_data():
        Process data and save the result.

    Attributes:
    -----------
    DEFAULT_PATH : Path
        The default path for data files.

    Examples:
    ---------
    >>> df = ImproveProcessor(reset=True, filename='revised_IMPROVE.csv', version='revised').process_data()

    """

    def __init__(self, reset=False, filename=None, version=None):
        super().__init__(reset)
        self.file_path = super().DEFAULT_PATH / 'Level2' / filename

        if version not in ['revised', 'modified']:
            raise ValueError("Invalid data_type. Allowed values are 'revised' and 'modified'.")
        else:
            self.version = version or 'revised'

    @staticmethod
    def revised(_df):
        def mode(Mass):
            if Mass < 20:
                L_mode = Mass ** 2 / 20
                S_mode = Mass - L_mode
            if Mass >= 20:
                L_mode = Mass
                S_mode = 0

            return L_mode, S_mode

        _frh, _frhss, _frhs, _frhl = ImproveProcessor.frh(_df['RH'], 'revised')

        L_AS, S_AS = mode(_df['AS'])
        L_AN, S_AN = mode(_df['AN'])
        L_OM, S_OM = mode(_df['OM'])

        _df['AS_ext_dry'] = 2.2 * 1 * S_AS + 4.8 * 1 * L_AS
        _df['AN_ext_dry'] = 2.4 * 1 * S_AN + 5.1 * 1 * L_AN
        _df['OM_ext_dry'] = 2.8 * S_OM + 6.1 * L_OM
        _df['Soil_ext_dry'] = 1 * _df['Soil']
        _df['SS_ext_dry'] = 1.7 * 1 * _df['SS']
        _df['EC_ext_dry'] = 10 * _df['EC']
        _df['total_ext_dry'] = sum(_df['AS_ext_dry':'EC_ext_dry'])

        _df['AS_ext'] = (2.2 * _frhs * S_AS) + (4.8 * _frhl * L_AS)
        _df['AN_ext'] = (2.4 * _frhs * S_AN) + (5.1 * _frhl * L_AN)
        _df['OM_ext'] = (2.8 * S_OM) + (6.1 * L_OM)
        _df['Soil_ext'] = (1 * _df['Soil'])
        _df['SS_ext'] = (1.7 * _frhss * _df['SS'])
        _df['EC_ext'] = (10 * _df['EC'])
        _df['total_ext'] = sum(_df['AS_ext':'EC_ext'])

        _df['ALWC_AS_ext'] = _df['AS_ext'] - _df['AS_ext_dry']
        _df['ALWC_AN_ext'] = _df['AN_ext'] - _df['AN_ext_dry']
        _df['ALWC_SS_ext'] = _df['SS_ext'] - _df['SS_ext_dry']
        _df['ALWC_ext'] = _df['total_ext'] - _df['total_ext_dry']

        _df['fRH_IMPR'] = _df['total_ext'] / _df['total_ext_dry']

        return _df['AS_ext_dry':]

    @staticmethod
    def modified(_df):
        _frh, _frhss, _frhs, _frhl = ImproveProcessor.frh(_df['RH'], 'modified')

        _df['AS_ext_dry'] = 3 * 1 * _df['AS']
        _df['AN_ext_dry'] = 3 * 1 * _df['AN']
        _df['OM_ext_dry'] = 4 * _df['OM']
        _df['Soil_ext_dry'] = 1 * _df['Soil']
        _df['SS_ext_dry'] = 1.7 * 1 * _df['SS']
        _df['EC_ext_dry'] = 10 * _df['EC']
        _df['total_ext_dry'] = sum(_df['AS_ext_dry':'EC_ext_dry'])

        _df['AS_ext'] = (3 * _frh * _df['AS'])
        _df['AN_ext'] = (3 * _frh * _df['AN'])
        _df['OM_ext'] = (4 * _df['OM'])
        _df['Soil_ext'] = (1 * _df['Soil'])
        _df['SS_ext'] = (1.7 * _frhs * _df['SS'])
        _df['EC_ext'] = (10 * _df['EC'])
        _df['total_ext'] = sum(_df['AS_ext':'EC_ext'])

        _df['ALWC_AS_ext'] = _df['AS_ext'] - _df['AS_ext_dry']
        _df['ALWC_AN_ext'] = _df['AN_ext'] - _df['AN_ext_dry']
        _df['ALWC_SS_ext'] = _df['SS_ext'] - _df['SS_ext_dry']
        _df['ALWC_ext'] = _df['total_ext'] - _df['total_ext_dry']

        _df['fRH_IMPR'] = _df['total_ext'] / _df['total_ext_dry']

        return _df['AS_ext_dry':]

    @staticmethod
    def gas(_df):
        _df['ScatteringByGas'] = (11.4 * 293 / (273 + _df['AT']))
        _df['AbsorptionByGas'] = (0.33 * _df['NO2'])
        _df['ExtinctionByGas'] = _df['ScatteringByGas'] + _df['AbsorptionByGas']
        return _df['ScatteringByGas':]

    @staticmethod
    def frh(_RH, version=None):
        frh = DataReader('fRH.json')
        if _RH is not None:
            if _RH > 95:
                _RH = 95
            _RH = round(_RH)
            return frh.loc[_RH].values.T

        return 1, 1, 1, 1
        pass

    def process_data(self):
        if self.file_path.exists() and not self.reset:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return read_csv(f, parse_dates=['Time']).set_index('Time')
        else:
            df = concat([DataReader('EPB.csv'), DataReader('IMPACT.csv'), DataReader('chemical.csv')], axis=1)

            # particle contribution '銨不足不納入計算'
            improve_input_df = df.loc[df['NH4_status'] != 'Deficiency', ['AS', 'AN', 'OM', 'Soil', 'SS', 'EC', 'RH']]

            if self.version == 'revised':
                df_improve = improve_input_df.dropna().copy().apply(self.revised, axis=1)

            if self.version == 'modified':
                df_improve = improve_input_df.dropna().copy().apply(self.modified, axis=1)

            # gas contribution
            df_ext_gas = df[['NO2', 'AT']].dropna().copy().apply(self.gas, axis=1)

            _df = concat([df_improve, df_ext_gas], axis=1).reindex(df.index.copy())
            _df.to_csv(self.file_path)

            return _df
