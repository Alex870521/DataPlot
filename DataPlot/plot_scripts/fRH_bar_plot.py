



if __name__ == '__main__':

    @set_figure(figsize=(6, 6), fw=16, fs=16)
    def test(**kwargs):
        data = np.array([1.01, 2.20, 1.47, 1.88, 1.42, 3.78])
        fig, ax = plt.subplots(1, 1)
        col = ['#FF3333', '#33FF33', '#FFFF33', '#5555FF', '#B94FFF', '#FFA500']
        x_position = np.array([0, 1, 2, 3, 4, 5])
        plt.bar(x_position, data, color=getColor(6))

        xlim = kwargs.get('xlim') or (-0.5, 5.5)
        ylim = kwargs.get('ylim') or (1, 4)
        xlabel = kwargs.get('xlabel') or unit.Factor
        ylabel = kwargs.get('ylabel') or unit.fRH
        title = kwargs.get('title') or ''

        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
        ax.set_xticks(x_position, labels=['F1', 'F2', 'F3', 'F4', 'F5', 'F6'], )
        ax.set_yticks(ax.get_yticks())
        ax.set_title(title)
        # ax.legend(prop=dict(weight='bold', size=12))

    test(title='test the title')