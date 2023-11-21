import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import DataPlot.plot_templates.basic as plot
# from DataPlot.Data_processing import integrate
from DataPlot.plot_templates import set_figure
from pathlib import Path
# df = integrate()

PATH_MAIN = Path(__file__).parent.parent / 'Data'

df1 = pd.read_csv(PATH_MAIN / '20231120-CPC1.csv', usecols=range(7), header=17, skipfooter=8, engine='python').set_index('Elapsed Time')
df2 = pd.read_csv(PATH_MAIN / '20231120-CPC2.csv', usecols=range(7), header=17, skipfooter=8, engine='python').set_index('Elapsed Time')

column_mapping = {
    'Conc (#/cm?'  : '60 nm',
    'Conc (#/cm?.1': '60* nm',
    'Conc (#/cm?.2': '60** nm',
    'Conc (#/cm?.3': '80 nm',
    'Conc (#/cm?.4': '100 nm',
    'Conc (#/cm?.5': '150 nm',
}

df1.rename(columns=column_mapping, inplace=True)
df2.rename(columns=column_mapping, inplace=True)

dic = {f'{value}': pd.concat([df1[f'{value}'], df2[f'{value}']], axis=1, ignore_index=True) for key, value in column_mapping.items()}


def test():
    @set_figure
    def test_scatter(_df, title):
        # df = pd.DataFrame(
        #     {'CPC_1': np.linspace(0, 10, 500) + np.random.randn(500) * 1 * np.random.randn(500),
        #      'CPC_2': np.linspace(0, 10, 500) + np.random.randn(500) * 1 * np.random.randn(500),
        #      'Absorption': np.linspace(0, 8, 500) + np.random.randn(500) * 1 * np.random.randn(500),
        #      'RH': np.random.random(500) * 100,
        #      'diversity': np.random.random(500) * 1})

        _df.columns = ['CPC_1', 'CPC_2']

        fig, ax = plot.scatter(_df, x='CPC_1', y='CPC_2', x_range=[_df['CPC_1'].min()*0.95, _df['CPC_1'].max()*1.05], y_range=[_df['CPC_2'].min()*0.95, _df['CPC_2'].max()*1.05], title=title, regression=True)

        # plot.scatter_mutiReg(df, x='Extinction', y1='Scattering', y2='Absorption', regression=True)
        # sns.jointplot(df, x='Extinction', y='Scattering')

    for key, value in dic.items():
        test_scatter(value, key)
