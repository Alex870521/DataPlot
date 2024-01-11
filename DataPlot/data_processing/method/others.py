import numpy as np


def kappa(df, dp=0.5):
    water_surface_tension = 0.072
    water_Mw = 18
    water_density = 1
    universal_gas_constant = 8.314  # J/mole*K
    diameter = dp  # um
    A = 4 * (water_surface_tension * water_Mw) / (water_density * universal_gas_constant * (df['AT'] + 273))
    power = A / diameter
    a_w = (df['RH'] / 100) * (np.exp(-power))
    Kappa = (df['gRH'] ** 3 - 1) * (1 - a_w) / a_w
    return Kappa


def other_process(df):
    df['PG'] = df[['Scattering', 'Absorption', 'ScatteringByGas', 'AbsorptionByGas']].dropna().copy().apply(np.sum, axis=1)
    # df['kappa'] = kappa(df.copy(), dp=df['GMDs'])
    df['MAC'] = df['Absorption'] / df['T_EC']
    df['Ox'] = df['NO2'] + df['O3']
    df['N2O5_tracer'] = df['NO2'] * df['O3']
    df['Vis_cal'] = 1096 / df['Extinction']
    # df['size'] = df['gRH'] * df['GMDs']
    # df['fRH_Mix'] = df['Bext'] / df['Extinction']
    # df['fRH_PNSD'] = df['Bext_internal'] / df['Bext_dry']
    df['fRH_IMPR'] = df['total_ext'] / df['total_ext_dry']
    df['OCEC_ratio'] = df['O_OC'] / df['O_EC']
    df['PM1/PM25'] = np.where(df['PM1'] / df['PM25'] < 1, df['PM1'] / df['PM25'], np.nan)
    df['MEE_PNSD'] = df['Bext_internal'] / df['PM25']
    # df['MEE_dry_PNSD'] = df['Bext_dry'] / df['PM25']
    return df