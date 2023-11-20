
cmap=['Haline_r','Haline']
## get_xticklabels 有點問題  如果加上pplt xtick會不一致

#-----------------------------------------------------------------------------------------------------------------------
#消光 PM1 PM2.5 整年度時序
fig2, (axes1,axes2) = plt.subplots(2, 1, figsize=(12, 5), dpi=150, constrained_layout=True)
sc = axes1.scatter(df.index, df.Extinction, c=df.PM25, norm=plt.Normalize(vmin=0,vmax=50), cmap='jet',
              marker='o', s=10, facecolor="b", edgecolor=None, alpha=1)

axes1.set_title('Extinction & $\mathregular{PM_{2.5}}$ Sequence Diagram', font_label_bold)
axes1.set_xlabel('',font_label_bold)
axes1.set_ylabel('Ext (1/Mm)',font_label_bold)
axes1.set_ylim(0.,600)
axes1.set_xlim(18506.,18871.)

axins = inset_axes(axes1, width="40%",height="5%", loc=1)
color_bar = plt.colorbar(sc, cax=axins, orientation='horizontal')
color_bar.set_label(label='$\mathregular{PM_{2.5}}$' + ' ($\mathregular{\mu}$g/$\mathregular{m^3}$)',family='Times New Roman', weight='bold',size=14)

color_bar.ax.set_xticklabels(color_bar.ax.get_xticks().astype(int),size=14)
###
sc2 = axes2.scatter(df.index, df.Extinction, c=df.PM1, norm=plt.Normalize(vmin=0,vmax=30), cmap='jet',
              marker='o', s=10, facecolor="b", edgecolor=None, alpha=1)

axes2.set_title('Extinction & $\mathregular{PM_{1.0}}$ Sequence Diagram', font_label_bold)
axes2.set_xlabel('',font_label_bold)
axes2.set_ylabel('Ext (1/Mm)',font_label_bold)
axes2.set_ylim(0.,600)
axes2.set_xlim(18506.,18871.)

axins2 = inset_axes(axes2, width="40%",height="5%", loc=1)
color_bar2 = plt.colorbar(sc2, cax=axins2, orientation='horizontal')
color_bar2.set_label(label='$\mathregular{PM_{1.0}}$' + ' ($\mathregular{\mu}$g/$\mathregular{m^3}$)',family='Times New Roman', weight='bold',size=14)
color_bar2.ax.set_xticklabels(color_bar2.ax.get_xticks().astype(int),size=14)

plt.show()

#消光散光吸光逐月or逐季 時序-------------------------------------------------------------------------------------------------
for _key, _df in df_group_season:
    print(f'Plot : {_df.Season[0]}')
    st_tm, fn_tm = _df.index[0], _df.index[-1]
    tick_time = date_range(st_tm, fn_tm, freq='10d')  ## set tick

    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(11, 5), dpi=150)

    sc1 = ax1.scatter(_df.index, _df.Scatter,
                      marker='o', s=15, facecolor="g", edgecolor='k', linewidths = 0.3, alpha=0.9)
    sc2 = ax1.scatter(_df.index, _df.Absorption,
                      marker='o', s=15, facecolor="r", edgecolor='k', linewidths = 0.3, alpha=0.9)

    ax1.set_xlabel('Date',font_label_bold)
    ax1.set_ylabel('Scattering & \n Absorption (1/Mm)',font_label_L_bold)
    ax1.set_xticks(tick_time)
    ax1.set_xticklabels(tick_time.strftime('%m/%d'))
    ax1.set_ylim(0, _df.Scatter.max()+10)
    ax1.set_xlim(st_tm, fn_tm)
    [ax1.spines[axis].set_visible(False) for axis in ['top']]

    sc3 = ax2.scatter(_df.index, _df.Extinction,
                       marker='o', s=15, facecolor="b", edgecolor='k', linewidths=0.3, alpha=0.9)
    therosold = ax2.plot(_df.index, np.full(len(_df.index), _df.Extinction.quantile([0.90])), color='r', ls='--', lw=2)
    ax2.set_ylabel('Extinction (1/Mm)', font_label_L_bold, loc='bottom')
    ax2.set_xticks(tick_time)
    ax2.set_xticklabels(tick_time.strftime(''))
    ax2.set_ylim(0,_df.Scatter.max()+10)
    ax2.set_xlim(st_tm, fn_tm)

    ax2.set_title(str(_df.Season[0]) + ' Sequence Diagram', font_label_L_bold)

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    [ax2.spines[axis].set_visible(False) for axis in ['bottom']]
    ax2.get_xaxis().set_visible(False)

    ax2.legend(handles = [sc1, sc2, sc3], labels=['Scattering', 'Absorption', 'Extinction'], bbox_to_anchor=(0, 1.), loc = 'upper left', prop=prop_legend)
    plt.subplots_adjust(hspace=0.0)
    # fig.savefig(pth(f"Optical{_df.Season[0]}"))
    # plt.show()

#四象限圓餅圖-------------------------------------------------------------------------------------------------------------
for quadrant in ['1', '2', '3', '4']:
    print('Quadrant = ' + quadrant)
    for x in ['Extinction', 'Scatter', 'Absorption', 'MEE', 'MSE', 'MAE', 'VC',
              'AS_ext', 'AN_ext', 'OM_ext', 'Soil_ext', 'SS_ext', 'EC_ext',
              'AS_ext_d', 'AN_ext_d', 'OM_ext_d', 'Soil_ext_d', 'SS_ext_d', 'EC_ext_d']:
        print(x + ' = ' + '{:.2f} \u00B1 {:.2f}'.format(dic_four[quadrant][x].mean(), dic_four[quadrant][x].std()))


    sizes = [dic_four[quadrant]['AS_ext_d'].mean(), (dic_four[quadrant]['AS_ext']-dic_four[quadrant]['AS_ext_d']).mean(),
             dic_four[quadrant]['AN_ext_d'].mean(), dic_four[quadrant]['AN_ext'].mean()-dic_four[quadrant]['AN_ext_d'].mean(),
             dic_four[quadrant]['OM_ext_d'].mean(), dic_four[quadrant]['Soil_ext_d'].mean(),
             dic_four[quadrant]['SS_ext_d'].mean(), dic_four[quadrant]['SS_ext'].mean()-dic_four[quadrant]['SS_ext_d'].mean(),
             dic_four[quadrant]['EC_ext_d'].mean()]

    sizes2 = [dic_four[quadrant]['AS_ext'].mean(),
              dic_four[quadrant]['AN_ext'].mean(),
              dic_four[quadrant]['OM_ext'].mean(), dic_four[quadrant]['Soil_ext'].mean(),
              dic_four[quadrant]['SS_ext'].mean(), dic_four[quadrant]['EC_ext'].mean()]
    labels = 'Ammonium Sulfate, AS', 'Hygroscopic growth by AS', 'Ammonium Nitrate, AN', 'Hygroscopic growth by AN',\
             'Organic Matter, OM', 'Soil', 'Sea Salt, SS', 'Hygroscopic growth by SS', 'Elemental Carbon, EC'

    labels_sizes = ['{:^20}'.format(labels[0]) + '\n' + str(sizes[0]) + ' (1/Mm)',
                    '{:^20}'.format(labels[1]) + '\n' + str(sizes[1]) + ' (1/Mm)',
                    '{:^17}'.format(labels[2]) + '\n' + str(sizes[2]) + ' (1/Mm)',
                    '{:^18}'.format(labels[3]) + '\n' + str(sizes[3]) + ' (1/Mm)',
                    '{:^18}'.format(labels[4]) + '\n' + str(sizes[4]) + ' (1/Mm)',
                    '{:^20}'.format(labels[5]) + '\n' + str(sizes[5]) + ' (1/Mm)']

    colors = ['#FF3333', '#FFB5B5', '#33FF33', '#BBFFBB', '#FFFF33', '#5555FF', '#B94FFF', '#FFBFFF', '#AAAAAA']
    colors2 = ['#FF3333', '#33FF33', '#FFFF33', '#5555FF', '#B94FFF', '#AAAAAA']
    explode = (0, 0, 0, 0, 0, 0, 0, 0, 0)  # explode a slice if required 圓餅外凸
    explode2 = (0, 0, 0, 0, 0, 0)
    # Intended to serve something like a global variable
    textprops = {'fontsize': 16, 'fontname': 'Times New Roman', 'weight': 'bold'}
    prop_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    fig1, ax1 = plt.subplots(figsize=(14, 8), constrained_layout=True)
    ax1.pie(sizes, explode=explode, labels=None, colors=colors,
            autopct='%1.1f%%', shadow=False, textprops=textprops, radius=10, labeldistance=None, pctdistance=0.85,
            startangle=0, wedgeprops=dict(width=3, edgecolor='w'))

    ax1.pie(sizes2, explode=explode2, labels=None, colors=colors2,
            autopct='%1.1f%%', shadow=False, textprops=textprops, radius=13, labeldistance=None, pctdistance=0.90,
            startangle=0, wedgeprops=dict(width=3, edgecolor='w'))

    sumstr = 'Extinction' + '\n\n' + "{:.2f}".format(np.sum(sizes)) + '(1/Mm)'
    plt.text(0, 0, sumstr, horizontalalignment='center', verticalalignment='center', size=18, family='Times New Roman',
             weight='bold')

    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title(quadrant, family='Times New Roman', size=25, weight='bold')
    plt.show()


#四象限 violine----------------------------------------------------------------------------------------------------------
for col, x in zip(['MEE','MSE','MAE'], label_[-4:-2]):  #自己打要的欄位跟label
    fig, axes = plt.subplots(1, 1, figsize=(8, 5), dpi=150, constrained_layout=True)
    plt.title(col+' violin config', font_label_bold)
    plt.xlabel('', font_label_bold)
    plt.ylabel(x, font_label_bold)
    violin = sns.violinplot(data=[dic_four['1'][col], dic_four['2'][col], dic_four['3'][col], dic_four['4'][col]],
                            scale='area', palette='husl', inner='quartile')
    for violin, alpha in zip(axes.collections[:], [0.8, 0.8, 0.8, 0.8]):
        violin.set_alpha(alpha)
        violin.set_edgecolor('k')

    plt.xticks([0, 1, 2, 3],
               ['High Visibility\nHigh $\mathregular{PM_{2.5}}$', 'High Visibility\nLow $\mathregular{PM_{2.5}}$',
                'Low Visibility\nLow $\mathregular{PM_{2.5}}$', 'Low Visibility\nHigh $\mathregular{PM_{2.5}}$'],
               fontsize=18, fontname="Times New Roman", weight='bold')
    plt.ylim(0, )
    Mean  = plt.scatter([0, 1, 2, 3], [dic_four['1'][col].mean(), dic_four['2'][col].mean(),
                                      dic_four['3'][col].mean(), dic_four['4'][col].mean()],
                        marker='o', s=50, facecolor="white", edgecolor="black")
    Event = plt.scatter([0, 1, 2, 3], [dic_four['1'][col].quantile(0.9), dic_four['2'][col].quantile(0.9),
                                       dic_four['3'][col].quantile(0.9), dic_four['4'][col].quantile(0.9)],
                        marker='o', s=50, facecolor="red", edgecolor="black")
    Clean = plt.scatter([0, 1, 2, 3], [dic_four['1'][col].quantile(0.1), dic_four['2'][col].quantile(0.1),
                                       dic_four['3'][col].quantile(0.1), dic_four['4'][col].quantile(0.1)],
                        marker='o', s=50, facecolor="green", edgecolor="black")
    plt.legend(handles=[Event, Mean, Clean], labels=['Event', 'Mean', 'Clean'], loc='best', prop=prop_legend)
    plt.show()
    fig.savefig(pth(f"4_{col}"))


# code to give the average values of each data
for season in ['2020-Summer', '2020-Autumn', '2020-Winter', '2021-Spring', '2021-Summer']:
    for cond in ['Total', 'Event', 'Clean']:
        print(dic_obs[season][cond]['Season'][0], cond)
        # print('{:.2f}'.format(dic_obs[season][cond].Extinction.mean()))
        for x in ['Extinction', 'Scatter', 'Absorption', 'MEE', 'MSE', 'MAE', 'VC',
                  'AS_ext', 'AS_ext', 'AN_ext',	'OM_ext', 'Soil_ext', 'SS_ext',	'EC_ext',
                  'AS_ext_d', 'AN_ext_d', 'OM_ext_d', 'Soil_ext_d',	'SS_ext_d',	'EC_ext_d']:

            print(x + ' = ' + '{:.2f} \u00B1 {:.2f}'.format(dic_obs[season][cond][x].mean(), dic_obs[season][cond][x].std()))


# GF_PM_Ext
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=0)
ax.set_position([0.125,0.125,0.65,0.8])

npoints=1000
xreg = np.linspace(df.PM25.min(),df.PM25.max(),83)
yreg = np.linspace(df.GF.min(),df.GF.max(),34)
X,Y = np.meshgrid(xreg,yreg)

d_f = df.copy()
df['GF'] = d_f['GF'].round(2)
df['PM25'] = d_f['PM25'].round(2)
table = df.pivot_table(index=['PM25'], columns=['GF'], values=['Extinction'], aggfunc=np.mean)

def func(para, a, b):
    PM, GF = para
    return a * (PM * GF)**(b)


fit_df = df[['PM25', 'GF', 'Extinction']].dropna()
popt, pcov = curve_fit(func, (fit_df['PM25'], fit_df['GF']), fit_df['Extinction'])
print(popt)
def f(x, y):
    return popt[0]*(x*y)**(popt[1])

def fmt(x):
    s = f"{x:.0f} 1/Mm"
    return rf"{s}"

plt.xlabel(label_[4], font_label_bold)
plt.ylabel('GF(RH)', font_label_bold)
plt.xlim(df.PM25.min(),df.PM25.max())
plt.ylim(df.GF.min(),df.GF.max())
plt.title('', font_label_bold)

# pcolor =  ax.pcolormesh(X, Y, (X*4.5*Y**(1/3)), cmap= 'jet',shading='auto',vmin=0,vmax = 843, alpha=0.8)
cont = ax.contour(X, Y, f(X,Y), colors= 'black', levels=5, vmin=0, vmax=f(X,Y).max(), linewidths=2)
conf = ax.contourf(X, Y, f(X,Y), cmap='YlGnBu', levels=100, vmin=0, vmax=f(X,Y).max())
ax.clabel(cont, colors=['black'], fmt=fmt, fontsize=16)

plt.scatter(df['PM25'], df['GF'], c=df.Extinction, norm=plt.Normalize(vmin=df.Extinction.min(), vmax=df.Extinction.max()), cmap='jet',
              marker='o', s=20, facecolor="b", edgecolor=None, alpha=0.5)

box = ax.get_position()
cax = fig.add_axes([0.8, box.y0, 0.03, box.y1-box.y0])
color_bar = plt.colorbar(conf, cax=cax)
color_bar.set_label(label='Extinction (1/Mm)',family='Times New Roman', weight='bold',size=16)
color_bar.ax.set_xticklabels(color_bar.ax.get_xticks().astype(int),size=16)

