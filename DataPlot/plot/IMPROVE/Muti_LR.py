
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from pandas import read_csv, concat
from sklearn.linear_model import LinearRegression

from DataPlot.process import *
from DataPlot.plot.templates import scatter, scatter_mutiReg
from DataPlot.plot.core import set_figure, unit, getColor


def residual_PM(_df):
    _df['residual_PM'] = _df['PM25'] - _df['AS'] - _df['AN'] - _df['OM'] - _df['SS'] - _df['EC']

    return _df[['residual_PM', 'Ti', 'Fe', 'Si']]


def residual_ext(_df):
    _df['residual_ext'] = _df['total_ext_dry'] - _df['AS_ext_dry'] - _df['AN_ext_dry'] - _df['Soil_ext_dry'] - _df['SS_ext_dry']

    return _df[['residual_ext', 'POC', 'SOC']]


if __name__ == '__main__':
    df = DataBase
    dic_grp_sta = Classifier(df, 'state')

    species = ['Extinction', 'Scattering', 'Absorption', 'total_ext_dry', 'AS_ext_dry', 'AN_ext_dry',
               'OM_ext_dry', 'Soil_ext_dry', 'SS_ext_dry', 'EC_ext_dry',
               'AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'EC', 'OM']

    for key, values in dic_grp_sta.items():
        if key == 'Total':
            # df_cal = dic_grp_sta[key][species].dropna().copy().apply(residual_ext, axis=1)
            df_cal = dic_grp_sta[key][species].dropna().copy()
            df_cal['Const'] = 1
            df_cal['Const2'] = 1
            x = df_cal[['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'Const']].to_numpy()
            y = df_cal[['Scattering']].to_numpy()

            model = LinearRegression(positive=True).fit(x, y)
            r_sq = model.score(x, y)
            # print(f"coefficient of determination: {r_sq}")
            # print(f"intercept: {model.intercept_}")
            print(f"coefficients: {model.coef_}")
            y_pred = model.predict(x)
            df_ = pd.DataFrame(np.concatenate([y, y_pred], axis=1), columns=['y_calculated', 'y_predict'])
            scatter(df_, 'y_calculated', 'y_predict', regression=True, title=f'{key}')

            result = df_cal[['AS', 'AN', 'POC', 'SOC', 'Soil', 'SS', 'Const']].mul(model.coef_[0]).mean()

            # --------
            x2 = df_cal[['POC', 'SOC', 'EC', 'Const2']].to_numpy()
            y2 = df_cal[['Absorption']].to_numpy()

            model2 = LinearRegression(positive=True).fit(x2, y2)
            r_sq2 = model2.score(x2, y2)
            # print(f"coefficient of determination: {r_sq2}")
            # print(f"intercept: {model2.intercept_}")
            print(f"coefficients: {model2.coef_}")
            y_pred2 = model2.predict(x2)
            df_ = pd.DataFrame(np.concatenate([y2, y_pred2], axis=1), columns=['y_calculated', 'y_predict'])
            scatter(df_, 'y_calculated', 'y_predict', regression=True, title=f'{key}')

            result2 = df_cal[['POC', 'SOC', 'EC', 'Const2']].mul(model2.coef_[0]).mean()


            n_df = df_cal[['AS', 'AN', 'POC', 'SOC', 'EC']].mul([2.74, 4.41, 11.5, 7.34, 12.27])
            new_df = concat([df_cal['Extinction'], n_df], axis=1)
            new_dic = Classifier(new_df, 'state')

            ext_dry_dict = {state: [new_dic[state][specie].mean() for specie in
                                    ['AS', 'AN', 'POC', 'SOC', 'EC']] for state in
                            ['Total', 'Clean', 'Transition', 'Event']}

            df_cal['Localized'] = df_cal[['AS', 'AN', 'POC', 'SOC', 'EC']].mul([2.74, 4.41, 11.5, 7.34, 12.27]).sum(axis=1)

    modify_IMPROVE = DataReader('modify_IMPROVE.csv')['total_ext_dry'].rename('Modified')
    revised_IMPROVE = DataReader('revised_IMPROVE.csv')['total_ext_dry'].rename('Revised')

    df = concat([df_cal, revised_IMPROVE, modify_IMPROVE], axis=1)

    @set_figure(figsize=(10, 6))
    def donuts_ext(data_set, labels, style='donut', title='', symbol=True):

        prop_legend = {'size': 12, 'family': 'Times New Roman', 'weight': 'bold'}
        textprops = {'fontsize': 14, 'fontfamily': 'Times New Roman', 'fontweight': 'bold'}

        labels = 'AS', 'AN', 'POA', 'SOA', 'EC'

        values1 = np.array(list(data_set.values()))[3]
        values2 = np.array(list(data_set.values()))[2]
        values3 = np.array(list(data_set.values()))[1]

        colors1 = getColor(kinds='3-4')

        def adjust_opacity(color, alpha):
            # 將顏色轉換為RGB表示
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            # 調整透明度
            r_new = int(alpha * r + (1 - alpha) * 255)
            g_new = int(alpha * g + (1 - alpha) * 255)
            b_new = int(alpha * b + (1 - alpha) * 255)
            # 轉換為新的色碼
            new_color = '#{:02X}{:02X}{:02X}'.format(r_new, g_new, b_new)
            return new_color

        colors2 = [adjust_opacity(color, 0.8) for color in colors1]
        colors3 = [adjust_opacity(color, 0.6) for color in colors1]

        fig, ax = plt.subplots(1, 1)
        ax.pie(values1, labels=None, colors=colors1, textprops=textprops,
               autopct='%1.1f%%',
               pctdistance=0.9, radius=14, wedgeprops=dict(width=3, edgecolor='w'))

        ax.pie(values2, labels=None, colors=colors2, textprops=textprops,
               autopct='%1.1f%%',
               pctdistance=0.85, radius=11, wedgeprops=dict(width=3, edgecolor='w'))

        ax.pie(values3, labels=None, colors=colors3, textprops=textprops,
               autopct='%1.1f%%',
               pctdistance=0.80, radius=8, wedgeprops=dict(width=3, edgecolor='w'))

        text = 'Average (1/Mm)' + '\n\n' + 'Event : ' + "{:.2f}".format(np.sum(values1)) + '\n' + \
               'Transition : ' + "{:.2f}".format(np.sum(values2)) + '\n' + \
               'Clean : ' + "{:.2f}".format(np.sum(values3))

        ax.text(0, 0, text, fontdict=textprops, ha='center', va='center')
        ax.axis('equal')
        ax.set_title(f'{title}', size=20, weight='bold')

        ax.legend(labels, loc='center', prop=prop_legend, title_fontproperties=dict(weight='bold'),
                  title='Outer : Event' + '\n' + 'Middle : Transition' + '\n' + 'Inner : Clean',
                  bbox_to_anchor=(0.66, 0, 0.5, 1), frameon=False)
        plt.show()
        # fig.savefig(f"IMPROVE_ext_donuts_{title}", transparent=True)


    @set_figure(figsize=(6, 6))
    def scatter_mutiReg(df, x, y1, y2, y3, regression=None, diagonal=False, **kwargs):
        # create fig
        fig, ax = plt.subplots()

        df = df.dropna(subset=[x, y1, y2, y3])

        x_data = np.array(df[x])

        y_data1 = np.array(df[y1])
        y_data2 = np.array(df[y2])
        y_data3 = np.array(df[y3])

        color1 = {'line': '#1a56db', 'edge': '#0F50A6', 'face': '#5983D9'}
        color2 = {'line': '#046c4e', 'edge': '#1B591F', 'face': '#538C4A'}
        color3 = {'line': '#c81e1e', 'edge': '#f05252', 'face': '#f98080'}

        scatter1 = ax.scatter(x_data, y_data1, s=25, color=color1['face'], alpha=0.8, edgecolors=color1['edge'],
                              label='Revised')
        scatter2 = ax.scatter(x_data, y_data2, s=25, color=color2['face'], alpha=0.8, edgecolors=color2['edge'],
                              label='Modified')
        scatter3 = ax.scatter(x_data, y_data3, s=25, color=color3['face'], alpha=0.8, edgecolors=color3['edge'],
                              label='Localized')

        xlim = kwargs.get('xlim')
        ylim = kwargs.get('ylim')
        xlabel = kwargs.get('xlabel') or unit(x) or ''
        ylabel = kwargs.get('ylabel') or unit(y1) or ''
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

        title = kwargs.get('title') or ''

        title_format = fr'$\bf {title}$'
        ax.set_title(title_format, fontdict={'fontweight': 'bold', 'fontsize': 20})

        if regression:
            x_2d = x_data.reshape(-1, 1)
            y_2d = y_data1.reshape(-1, 1)

            model = LinearRegression().fit(x_2d, y_2d)
            slope = model.coef_[0][0].__round__(3)
            intercept = model.intercept_[0].__round__(3)
            r_square = model.score(x_2d, y_2d).__round__(3)

            plt.plot(x_2d, model.predict(x_2d), linewidth=3, color=color1['line'], alpha=1, zorder=3)

            text = np.poly1d([slope, intercept])
            func1 = '\n' + 'y = ' + str(text).replace('\n', "") + '\n' + r'$\bf R^2 = $' + str(r_square)


            x_2d = x_data.reshape(-1, 1)
            y_2d2 = y_data2.reshape(-1, 1)
            model2 = LinearRegression().fit(x_2d, y_2d2)
            slope = model2.coef_[0][0].__round__(3)
            intercept = model2.intercept_[0].__round__(3)
            r_square = model2.score(x_2d, y_2d2).__round__(3)

            plt.plot(x_2d, model2.predict(x_2d), linewidth=3, color=color2['line'], alpha=1, zorder=3)

            text = np.poly1d([slope, intercept])
            func2 = '\n' + 'y = ' + str(text).replace('\n', "") + '\n' + r'$\bf R^2 = $' + str(r_square)


            x_2d = x_data.reshape(-1, 1)
            y_2d3 = y_data3.reshape(-1, 1)
            model3 = LinearRegression().fit(x_2d, y_2d3)
            slope = model3.coef_[0][0].__round__(3)
            intercept = model3.intercept_[0].__round__(3)
            r_square = model3.score(x_2d, y_2d3).__round__(3)

            plt.plot(x_2d, model3.predict(x_2d), linewidth=3, color=color3['line'], alpha=1, zorder=3)

            text = np.poly1d([slope, intercept])
            func3 = '\n' + 'y = ' + str(text).replace('\n', "") + '\n' + r'$\bf R^2 = $' + str(r_square)

        if diagonal:
            ax.axline((0, 0), slope=1., color='k', lw=2, ls='--', alpha=0.5, label='1:1')
            plt.text(0.97, 0.95, r'$\bf 1:1\ Line$', color='k', ha='right', va='top', transform=ax.transAxes)

        leg = plt.legend(handles=[scatter1, scatter2, scatter3],
                         labels=[f'Revised: {func1}', f'Modified: {func2}', f'Localized: {func3}'],
                         loc='upper left', prop={'weight': 'bold', 'size': 12})

        for text, color in zip(leg.get_texts(), [color1['line'], color2['line'], color3['line']]):
            text.set_color(color)
        # savefig

        return fig, ax

    fig, ax = scatter_mutiReg(df, x='Extinction', y1='Revised', y2='Modified', y3='Localized', xlim=[0, 400], ylim=[0, 400],
                              title='', regression=True, diagonal=True)

    donuts_ext(ext_dry_dict, labels=['AS', 'AN', 'POC', 'SOC', 'EC'])