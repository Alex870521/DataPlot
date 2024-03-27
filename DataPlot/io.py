from DataPlot.plot.templates import Pie


def default_data():
    import pandas as pd
    # The following data is from the Denver Aerosol Sources and Health study.
    # See doi:10.1016/j.atmosenv.2008.12.017
    #
    # The data are pollution source profile estimates for five modeled
    # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
    # species. The radar charts are experimented with here to see if we can
    # nicely visualize how the modeled source profiles change across four
    # scenarios:
    #  1) No gas-phase species present, just seven particulate counts on
    #  2) Inclusion of gas-phase specie carbon monoxide (CO)
    #  3) Inclusion of gas-phase specie ozone (O3).
    #  4) Inclusion of both gas-phase species is present...
    data = {
        'Sulfate': [0.01, 0.34, 0.02, 0.71, 0.74, 0.70],
        'Nitrate': [0.88, 0.13, 0.34, 0.13, 0.04, 0.06],
        'OC': [0.07, 0.95, 0.04, 0.05, 0.05, 0.02],
        'EC': [0.20, 0.02, 0.85, 0.19, 0.05, 0.10],
        'Soil': [0.20, 0.10, 0.07, 0.01, 0.21, 0.12],
    }

    return pd.DataFrame(data, index=['Case_1', 'Case_2', 'Case_3', 'Case_4', 'Case_5', 'Case_6'])


if __name__ == '__main__':
    Pie.pieplot(default_data(), labels=default_data().columns, unit='PM25', style='donut')
