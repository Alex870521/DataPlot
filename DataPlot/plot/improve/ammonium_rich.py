import matplotlib.pyplot as plt
from pandas import DataFrame
from DataPlot.plot.core import *


@set_figure
def ammonium_rich(_df: DataFrame, title='') -> plt.Axes:
    print('Plot: ammonium_rich')
    df = _df[['NH4+', 'SO42-', 'NO3-', 'PM25']].dropna().copy().div([18, 96, 62, 1])
    df['required_ammonium'] = df['NO3-'] + 2 * df['SO42-']

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150, constrained_layout=True)

    scatter = ax.scatter(df['required_ammonium'], df[['NH4+']], c=df[['PM25']].values,
                         vmin=0, vmax=70, cmap='jet', marker='o', s=10, alpha=1)
    ax.axline((0, 0), slope=1., color='r', lw=3, ls='--', label='1:1')
    ax.set_xlabel(r'$\bf NO_{3}^{-}\ +\ 2\ \times\ SO_{4}^{2-}\ (mole\ m^{-3})$')
    ax.set_ylabel(r'$\bf NH_{4}^{+}\ (mole\ m^{-3})$')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(ax.get_yticks())
    ax.set_title(title)

    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best')

    color_bar = plt.colorbar(scatter, extend='both')
    color_bar.set_label(label=Unit('PM25'), size=14)

    # fig.savefig(f'Ammonium_rich_{title}')
    return ax

