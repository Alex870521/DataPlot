from pathlib import Path
from pandas import read_csv, concat
from Data_processing.processDecorator import save_to_csv

PATH_MAIN = Path("C:/Users/Alex/PycharmProjects/DataPlot/Data")

with open(PATH_MAIN / 'level1' / 'fRH.csv', 'r', encoding='utf-8', errors='ignore') as f:
    frh = read_csv(f).set_index('RH')


def f_RH(RH, version=None):
    """
    :param RH: int
    :param version: str, (fRH, fRHs, fRHl, fRHss)
    :return: float
    """

    if RH > 95:
        val = frh[version][95]
    else:
        val = frh[version][int(RH)]
    return val


def original_IMPROVE(_df):
    _df['AS_ext_dry'] = (3 * 1 * _df['AS_mass'])
    _df['AN_ext_dry'] = (3 * 1 * _df['AN_mass'])
    _df['OM_ext_dry'] = (4 * _df['OM_mass'])
    _df['Soil_ext_dry'] = (1 * _df['Soil_mass'])
    _df['EC_ext_dry'] = (10 * _df['EC_mass'])
    _df['total_ext_dry'] = sum(_df['AS_ext_dry':'EC_ext_dry'])

    _df['AS_ext'] = (3 * f_RH(_df['RH'], version='fRH') * _df['AS_mass'])
    _df['AN_ext'] = (3 * f_RH(_df['RH'], version='fRH') * _df['AN_mass'])
    _df['OM_ext'] = (4 * _df['OM_mass'])
    _df['Soil_ext'] = (1 * _df['Soil_mass'])
    _df['EC_ext'] = (10 * _df['EC_mass'])
    _df['total_ext'] = sum(_df['AS_ext':'EC_ext'])

    _df['ALWC_AS_ext'] = _df['AS_ext'] - _df['AS_ext_dry']
    _df['ALWC_AN_ext'] = _df['AN_ext'] - _df['AN_ext_dry']
    _df['ALWC_ext'] = _df['total_ext'] - _df['total_ext_dry']

    _df['fRH_IMPR'] = _df['total_ext'] / _df['total_ext_dry']

    return _df['AS_ext_dry':]


def revised_IMPROVE(_df):
    def mode(Mass):
        if Mass < 20:
            L_mode = Mass ** 2 / 20
            S_mode = Mass - L_mode
        if Mass >= 20:
            L_mode = Mass
            S_mode = 0

        return L_mode, S_mode

    L_AS, S_AS = mode(_df['AS_mass'])
    L_AN, S_AN = mode(_df['AN_mass'])
    L_OM, S_OM = mode(_df['OM_mass'])

    _df['AS_ext_dry'] = (2.2 * 1 * S_AS + 4.8 * 1 * L_AS)
    _df['AN_ext_dry'] = (2.4 * 1 * S_AN + 5.1 * 1 * L_AN)
    _df['OM_ext_dry'] = (2.8 * S_OM + 6.1 * L_OM)
    _df['Soil_ext_dry'] = (1 * _df['Soil_mass'])
    _df['SS_ext_dry'] = (1.7 * 1 * _df['SS_mass'])
    _df['EC_ext_dry'] = (10 * _df['EC_mass'])
    _df['total_ext_dry'] = sum(_df['AS_ext_dry':'EC_ext_dry'])

    _df['AS_ext'] = (2.2 * f_RH(_df['RH'], version='fRHs') * S_AS + 4.8 * f_RH(_df['RH'], version='fRHl') * L_AS)
    _df['AN_ext'] = (2.4 * f_RH(_df['RH'], version='fRHs') * S_AN + 5.1 * f_RH(_df['RH'], version='fRHl') * L_AN)
    _df['OM_ext'] = (2.8 * S_OM + 6.1 * L_OM)
    _df['Soil_ext'] = (1 * _df['Soil_mass'])
    _df['SS_ext'] = (1.7 * f_RH(_df['RH'], version='fRHSS') * _df['SS_mass'])
    _df['EC_ext'] = (10 * _df['EC_mass'])
    _df['total_ext'] = sum(_df['AS_ext':'EC_ext'])

    _df['ALWC_AS_ext'] = _df['AS_ext'] - _df['AS_ext_dry']
    _df['ALWC_AN_ext'] = _df['AN_ext'] - _df['AN_ext_dry']
    _df['ALWC_SS_ext'] = _df['SS_ext'] - _df['SS_ext_dry']
    _df['ALWC_ext'] = _df['total_ext'] - _df['total_ext_dry']

    _df['fRH_IMPR'] = _df['total_ext'] / _df['total_ext_dry']

    return _df['AS_ext_dry':]


def modify_IMPROVE(_df):
    _df['AS_ext_dry'] = (3 * 1 * _df['AS_mass'])
    _df['AN_ext_dry'] = (3 * 1 * _df['AN_mass'])
    _df['OM_ext_dry'] = (4 * _df['OM_mass'])
    _df['Soil_ext_dry'] = (1 * _df['Soil_mass'])
    _df['SS_ext_dry'] = (1.7 * 1 * _df['SS_mass'])
    _df['EC_ext_dry'] = (10 * _df['EC_mass'])
    _df['total_ext_dry'] = sum(_df['AS_ext_dry':'EC_ext_dry'])

    _df['AS_ext'] = (3 * f_RH(_df['RH'], version='fRH') * _df['AS_mass'])
    _df['AN_ext'] = (3 * f_RH(_df['RH'], version='fRH') * _df['AN_mass'])
    _df['OM_ext'] = (4 * _df['OM_mass'])
    _df['Soil_ext'] = (1 * _df['Soil_mass'])
    _df['SS_ext'] = (1.7 * f_RH(_df['RH'], version='fRHSS') * _df['SS_mass'])
    _df['EC_ext'] = (10 * _df['EC_mass'])
    _df['total_ext'] = sum(_df['AS_ext':'EC_ext'])

    _df['ALWC_AS_ext'] = _df['AS_ext'] - _df['AS_ext_dry']
    _df['ALWC_AN_ext'] = _df['AN_ext'] - _df['AN_ext_dry']
    _df['ALWC_SS_ext'] = _df['SS_ext'] - _df['SS_ext_dry']
    _df['ALWC_ext'] = _df['total_ext'] - _df['total_ext_dry']

    _df['fRH_IMPR'] = _df['total_ext'] / _df['total_ext_dry']

    return _df['AS_ext_dry':]


def gas_IMPROVE(_df):
    _df['ScatteringByGas'] = (11.4 * 293 / (273 + _df['AT']))
    _df['AbsorptionByGas'] = (0.33 * _df['NO2'])
    _df['ExtinctionByGas'] = _df['ScatteringByGas'] + _df['AbsorptionByGas']
    return _df['ScatteringByGas':]


@save_to_csv(PATH_MAIN / 'level2' / 'IMPROVE' / 'revised_IMPROVE.csv')
def improve_process(reset=False, version='revised', filename=None):
    if filename.exists() & (~reset):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            return read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'level1' / 'EPB.csv', 'r', encoding='utf-8', errors='ignore') as f:
        minion = read_csv(f, parse_dates=['Time'], na_values=['-', 'E', 'F']).set_index('Time')

    with open(PATH_MAIN / 'level1' / 'IMPACT.csv', 'r', encoding='utf-8', errors='ignore') as f:
        impact = read_csv(f, parse_dates=['Time']).set_index('Time')

    with open(PATH_MAIN / 'level2' / 'mass_volume_VAM.csv', 'r', encoding='utf-8', errors='ignore') as f:
        mass = read_csv(f, parse_dates=['Time']).set_index('Time')

    df = concat([minion, impact, mass], axis=1)

    _index = df.index.copy()
    # 銨不足不納入計算
    IMPROVE_input_df = concat([df[['AS_mass', 'AN_mass', 'OM_mass', 'Soil_mass', 'SS_mass', 'EC_mass']],
                               df['NH4_status'].mask(df['NH4_status'] == 'Deficiency'), df['RH']], axis=1)

    # gas contribution
    df_ext_gas = df[['NO2', 'AT']].dropna().copy().apply(gas_IMPROVE, axis=1).reindex(_index)
    df_ext_gas.index.name = 'Time'

    if version == 'original':
        df_IMPROVE = IMPROVE_input_df.dropna().copy().apply(original_IMPROVE, axis=1).reindex(_index)

    if version == 'modify':
        df_IMPROVE = IMPROVE_input_df.dropna().copy().apply(modify_IMPROVE, axis=1).reindex(_index)

    if version == 'revised':
        df_IMPROVE = IMPROVE_input_df.dropna().copy().apply(revised_IMPROVE, axis=1).reindex(_index)

    df_IMPROVE.index.name = 'Time'
    df_IMPROVE_v2 = concat([df_IMPROVE, df_ext_gas], axis=1)

    return df_IMPROVE_v2


if __name__ == '__main__':
    df = improve_process(reset=True)