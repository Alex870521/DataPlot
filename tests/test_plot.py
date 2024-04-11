import unittest

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import Axes

import DataPlot.plot as plot
from DataPlot import *


class TestDataPlot(unittest.TestCase):
    def setUp(self):
        # 在每個測試之前執行的設置代碼
        print("Running setUp method")

    def tearDown(self):
        # 在每個測試結束後執行的清理代碼
        print("Running tearDown method")

    def test_font_properties(self):
        @set_figure
        def dummy_plot():
            plt.plot([1, 2, 3], [4, 5, 6])

        # Call the decorated function to apply the decorator settings
        dummy_plot()

        # Assert that the font properties are set as expected
        self.assertEqual(plt.rcParams['mathtext.fontset'], 'custom')
        self.assertEqual(plt.rcParams['font.family'], ['Times New Roman'])
        self.assertEqual(plt.rcParams['font.size'], 12.)

    def test_figure_size(self):
        @set_figure(figsize=(8, 6))
        def dummy_plot():
            plt.plot([1, 2, 3], [4, 5, 6])

        # Call the decorated function to apply the decorator settings
        dummy_plot()

        # Assert that the figure size is set as expected
        self.assertEqual(plt.rcParams['figure.figsize'], [8., 6.])

    def test_linear_regression(self):
        # Create a sample DataFrame
        data = {'X': [1, 2, 3, 4, 5], 'Y1': [2, 4, 6, 8, 10], 'Y2': [3, 6, 9, 12, 15]}
        df = pd.DataFrame(data)

        # Call the linear_regression function
        ax = plot.linear_regression(df, x='X', y=['Y1', 'Y2'], labels=['Label1', 'Label2'],
                                    diagonal=True, xlim=(0, 10), ylim=(0, 20),
                                    xlabel="X-axis", ylabel="Y-axis", title="Scatter Plot with Regressions")

        # Check if the return value is an instance of Axes
        self.assertIsInstance(ax, Axes)

        # Check if the title of the plot matches the expected title
        self.assertEqual(ax.get_title(), "Scatter Plot with Regressions")

        # Add more assertions as needed

    def test_multiple_linear_regression(self):
        # Create a sample DataFrame for testing
        data = {'X1': [1, 2, 3, 4, 5],
                'X2': [2, 3, 4, 5, 6],
                'Y': [3, 5, 7, 9, 11]}
        df = pd.DataFrame(data)

        # Call the function and get the Axes object
        ax = plot.multiple_linear_regression(df, x=['X1', 'X2'], y='Y',
                                             labels=['X1 Label', 'X2 Label'],
                                             diagonal=True, add_constant=True,
                                             xlabel="X-axis", ylabel="Y-axis",
                                             title="Multiple Linear Regression Plot")

        # Assert that the returned object is a Matplotlib Axes object
        self.assertIsInstance(ax, plt.Axes)

        # Add more assertions as needed to check the plot properties or data used for plotting

    def test_scatter(self):
        # Create a sample DataFrame for testing
        data = {'X': [1, 2, 3, 4, 5],
                'Y': [3, 4, 5, 6, 7],
                'C': [10, 20, 30, 40, 50],
                'S': [100, 200, 300, 400, 500]}
        df = pd.DataFrame(data)

        # Call the function and get the Axes object
        ax = plot.scatter(df, x='X', y='Y', c='C', s='S', cmap='jet', regression=True,
                          diagonal=True, box=True, figsize=(8, 6))

        # Assert that the returned object is a Matplotlib Axes object
        self.assertIsInstance(ax, plt.Axes)

        # Add more assertions as needed to check the plot properties or data used for plotting


if __name__ == '__main__':
    unittest.main()
