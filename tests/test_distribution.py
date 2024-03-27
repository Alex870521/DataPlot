import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from DataPlot import *

PNSD = DataReader('PNSD_dNdlogdp.csv')
PSSD = DataReader('PSSD_dSdlogdp.csv')
PVSD = DataReader('PVSD_dVdlogdp.csv')
PESD_inter = DataReader('PESD_dextdlogdp_internal.csv')
PESD_dry_inter = DataReader('PESD_dextdlogdp_dry_internal.csv')
PESD_exter = DataReader('PESD_dextdlogdp_external.csv')

Ext_amb_dis_internal, Ext_amb_dis_std_internal = DataClassifier(PESD_inter, by='State', statistic='Table')
Ext_dry_dis_internal, Ext_dry_dis_std_internal = DataClassifier(PESD_dry_inter, by='State', statistic='Table')
Ext_amb_dis_external, Ext_amb_dis_std_external = DataClassifier(PESD_exter, by='State', statistic='Table')

PNSD_amb_dis, PNSD_amb_dis_std = DataClassifier(PNSD, by='State', statistic='Table')
PSSD_amb_dis, PSSD_amb_dis_std = DataClassifier(PSSD, by='State', statistic='Table')
PVSD_amb_dis, PVSD_amb_dis_std = DataClassifier(PVSD, by='State', statistic='Table')

ext_grp, _ = DataClassifier(PESD_inter, by='Extinction', statistic='Table', qcut=10)

df_mean_all, df_std_all = DataClassifier(DataBase, by='Hour', statistic='Table')


class TestDataPlot(unittest.TestCase):
    def setUp(self):
        self.data = {'Clean': np.array([1, 2, 3, 4]),
                     'Transition': np.array([2, 3, 4, 5]),
                     'Event': np.array([3, 4, 5, 6])}
        self.data_set = pd.DataFrame.from_dict(self.data, orient='index', columns=['11.8', '12.18', '100.58', '1200.00'])

    def tearDown(self):
        plt.close()
        pass

    def test_distribution(self):
        ax = plot.distribution.heatmap(PNSD)
        self.assertIsInstance(ax, Axes)

    def test_heatmap_tms(self):
        plot.distribution.heatmap_tms(PNSD, freq='60d')
        plot.distribution.heatmap_tms(PSSD, unit='Surface', freq='60d')
        plot.distribution.heatmap_tms(PVSD, unit='Volume', freq='60d')
        plot.distribution.heatmap_tms(PESD_inter, unit='Extinction', freq='60d')

    def test_curve_fitting(self):
        plot.distribution.curve_fitting(np.array(Ext_amb_dis_internal.columns, dtype=float), Ext_amb_dis_internal.loc['Transition'], mode=3)

    def test_std_additional(self):
        ax = plot.distribution.plot_dist(self.data_set, data_std=self.data_set * 0.1)
        self.assertEqual(len(ax.get_legend().legend_handles), 3)  # Assuming 3 states with std

    def test_enhancement_additional(self):
        ax = plot.distribution.plot_dist(self.data_set, additional='enhancement')
        self.assertEqual(len(ax.get_legend().legend_handles), 5)  # Assuming 2 enhancement ratios added

    def test_error_additional(self):
        ax = plot.distribution.plot_dist(self.data_set, additional='error')
        self.assertEqual(len(ax.get_legend().legend_handles), 5)  # Assuming 2 error curves added

    def test_(self):
        plot.distribution.three_dimension(ext_grp, unit='Extinction')
        plot.distribution.ls_mode()
        plot.distribution.lognorm_dist()


if __name__ == '__main__':
    unittest.main()
