from windrose import WindroseAxes
from matplotlib.pyplot import subplots, rcParams, show
from pathlib import Path
from datetime import datetime as dtm
from Data_processing import integrate
from Data_classify import state_classify

color_lst = ['#8ecae6', '#f1dca7', '#f4a261', '#bc3908']

df = integrate()

Seasons = {'2020-Summer': (dtm(2020, 9, 4), dtm(2020, 9, 21, 23)),
           '2020-Autumn': (dtm(2020, 9, 22), dtm(2020, 12, 29, 23)),
           '2020-Winter': (dtm(2020, 12, 30), dtm(2021, 3, 25, 23)),
           '2021-Spring': (dtm(2021, 3, 26), dtm(2021, 5, 6, 23))}
           # '2021-Summer': (dtm(2021, 5, 7), dtm(2021, 10, 16, 23)),
           # '2021-Autumn': (dtm(2021, 10, 17), dtm(2021, 12, 31, 23))}


# for _, (_st, _ed) in Seasons.items():
#     fig, ax = subplots(figsize=(4, 4), dpi=150.)
#     ax.axis('off')
#
#     fs = 15.
#     font_fam = 'Times New Roman'
#     # font_fam = 'DejaVu Sans'
#     rcParams['font.sans-serif'] = font_fam
#     rcParams['mathtext.fontset'] = 'core'
#     font_dic = dict(fontsize=fs, math_fontfamily='core')
#
#     dt_met = df[_st:_ed]
#     ws, wd = dt_met['WS'], dt_met['WD']
#
#     ax = WindroseAxes.from_ax(fig=fig)
#     ax.bar(wd.loc[_st:_ed].values, ws.loc[_st:_ed].values, bins=[0, 1, 2, 3], nsector=16, normed=True, colors=color_lst)
#
#     ax.set(ylim=(0, 30), yticks=[0, 15, 30], yticklabels=['', '15 %', '30 %'])
#     ax.tick_params(labelsize=fs - 3, pad=-2)
#     show()
#     # ax.set_legend(framealpha=0,bbox_to_anchor=[-.05,-.05], fontsize=fs-2.,loc='lower left',ncol=3)
#
#     fig.savefig(f'windrose_{_st.strftime("%Y%m%d")}_{_ed.strftime("%Y%m%d")}.png')

dic = state_classify(df)

for state, _df in dic.items():
    fig, ax = subplots(figsize=(4, 4), dpi=150.)
    ax.axis('off')

    fs = 15.

    dt_met = _df
    ws, wd = dt_met['WS'], dt_met['WD']

    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(wd.values, ws.values, bins=[0, 1, 2, 3], nsector=16, normed=True, colors=color_lst)

    ax.set(ylim=(0, 30), yticks=[0, 15, 30], yticklabels=['', '15 %', '30 %'])
    ax.tick_params(labelsize=fs - 3, pad=-2)
    show()
    # ax.set_legend(framealpha=0,bbox_to_anchor=[-.05,-.05], fontsize=fs-2.,loc='lower left',ncol=3)

    # fig.savefig(f'windrose/windrose_{state}.png')