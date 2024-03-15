import matplotlib.pyplot as plt
import numpy as np
from DataPlot.plot.core import *
from typing import Literal


class Pie:
    @staticmethod
    def inner_pct(pct, symbol=True):
        if symbol:
            if pct < 8:
                return ''
            else:
                return '{:.1f}%'.format(pct)
        else:
            return ''

    @staticmethod
    def outer_pct(pct, symbol=True):
        if symbol:
            if pct > 8:
                return ''
            else:
                return '{:.1f}%'.format(pct)
        else:
            return ''

    @staticmethod
    def auto_pct(pct, symbol=True):
        if symbol:
            if pct < 3:
                return ''
            else:
                return '{:.1f}%'.format(pct)
        else:
            return ''

    @staticmethod
    @set_figure(fs=8, fw='bold')
    def pie(data_set: dict[str, list],
            labels: list[str],
            unit: str,
            style: Literal["pie", 'donut'],
            ax: plt.Axes | None = None,
            symbol=True,
            **kwargs):
        """
        Create a pie or donut chart based on the provided data.

        Parameters
        ----------
        data_set : dict[str, list]
            A dictionary mapping category names to a list of species.
            It is assumed that all lists contain the same number of entries as the *labels* list.
        labels : list of str
            The labels for each category.
        unit : str
            The unit to display in the center of the donut chart.
        style : Literal["pie", 'donut']
            The style of the chart, either 'pie' for a standard pie chart or 'donut' for a donut chart.
        ax : plt.Axes or None, optional
            The Axes object to plot the chart onto. If None, a new figure and Axes will be created.
        symbol : bool, optional
            Whether to display values for each species in the chart.
        **kwargs
            Additional keyword arguments to be passed to the plotting function.

        Returns
        -------
        None
            The function plots the pie or donut chart but does not return any value.

        Notes
        -----
        - The *data_set* dictionary should contain lists of species that correspond to each category in *labels*.
        - The length of each list in *data_set* should match the length of the *labels* list.

        Examples
        --------
        >>> data_set = {'Category 1': [10, 20, 30], 'Category 2': [15, 25, 35]}
        >>> labels = ['Species 1', 'Species 2', 'Species 3']
        >>> pie(data_set, labels, unit='kg', style='pie', symbol=True)
        """
        category_names = list(data_set.keys())
        data = np.array(list(data_set.values()))

        pies, species = data.shape

        colors = kwargs.get('colors') or (Color.colors1 if species == 6 else Color.getColor(num=species))

        radius = 4
        width = 4 if style == 'pie' else 1

        text = [''] * pies if style == 'pie' else [Unit(unit) + '\n\n' + '{:.2f}'.format(x) for x in data.sum(axis=1)]
        pct_distance = 0.6 if style == 'pie' else 0.88

        if ax is None:
            fig, ax = plt.subplots(1, pies, figsize=(pies * 3, 3))

        for i in range(pies):
            ax[i].pie(data[i], labels=None, colors=colors, textprops=None,
                      autopct=lambda pct: Pie.inner_pct(pct, symbol=symbol),
                      pctdistance=pct_distance, radius=radius, wedgeprops=dict(width=width, edgecolor='w'))

            ax[i].pie(data[i], labels=None, colors=colors, textprops=None,
                      autopct=lambda pct: Pie.outer_pct(pct, symbol=symbol),
                      pctdistance=1.3, radius=radius, wedgeprops=dict(width=width, edgecolor='w'))
            ax[i].axis('equal')
            ax[i].text(0, 0, text[i], ha='center', va='center')
            ax[i].set_title(category_names[i])

        ax[-1].legend(labels, loc='center', prop={'weight': 'bold'}, bbox_to_anchor=(1.2, 0, 0.5, 1))

        # fig.savefig(f"pie_{style}_{title}")
        plt.show()

        return ax

    @staticmethod
    @set_figure(figsize=(7, 5), fs=8, fw='bold')
    def donuts(data_set: dict[str, list],
               labels: list[str],
               unit: str,
               ax: plt.Axes | None = None,
               symbol=True,
               **kwargs):
        """
        Plot a donut chart based on the data set.

        Parameters
        ----------
        data_set : dict
            A mapping from category_names to a list of species. It is assumed all lists
            contain the same number of entries and that it matches the length of *labels*.
        labels : list of str
            The category labels.
        unit : str
            The unit to be displayed in the center of the donut chart.
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, the current axes will be used (default).
        symbol : bool, optional
            Whether to display values for each species (default is True).
        **kwargs : dict, optional
            Additional keyword arguments to pass to the matplotlib pie chart function.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the donut chart.
        """
        category_names = list(data_set.keys())
        data = np.array(list(data_set.values()))

        pies, species = data.shape

        colors1 = kwargs.get('colors') or (Color.colors1 if species == 6 else Color.getColor(num=species))
        colors2 = Color.adjust_opacity(colors1, 0.8)
        colors3 = Color.adjust_opacity(colors1, 0.6)

        if ax is None:
            fig, ax = plt.subplots()

        ax.pie(data[2], labels=None, colors=colors1, textprops=None,
               autopct=lambda pct: Pie.auto_pct(pct, symbol=symbol),
               pctdistance=0.9, radius=14, wedgeprops=dict(width=3, edgecolor='w'))

        ax.pie(data[1], labels=None, colors=colors2, textprops=None,
               autopct=lambda pct: Pie.auto_pct(pct, symbol=symbol),
               pctdistance=0.85, radius=11, wedgeprops=dict(width=3, edgecolor='w'))

        ax.pie(data[0], labels=None, colors=colors3, textprops=None,
               autopct=lambda pct: Pie.auto_pct(pct, symbol=symbol),
               pctdistance=0.80, radius=8, wedgeprops=dict(width=3, edgecolor='w'))

        text = (Unit(f'{unit}') + '\n\n' +
                'Event : ' + "{:.2f}".format(np.sum(data[2])) + '\n' +
                'Transition : ' + "{:.2f}".format(np.sum(data[1])) + '\n' +
                'Clean : ' + "{:.2f}".format(np.sum(data[0])))

        ax.text(0, 0, text, ha='center', va='center')
        ax.axis('equal')

        ax.set_title(kwargs.get('title') or '')

        ax.legend(labels, loc='center', prop={'weight': 'bold'}, title_fontproperties={'weight': 'bold'},
                  title=f'Outer : {category_names[2]}' + '\n' + f'Middle : {category_names[1]}' + '\n' + f'Inner : {category_names[0]}',
                  bbox_to_anchor=(0.75, 0, 0.5, 1))

        # fig.savefig(f"donuts_{title}")
        plt.show()

        return ax
