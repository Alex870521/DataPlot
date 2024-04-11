import unittest

from DataPlot.plot.core import Color


class TestColor(unittest.TestCase):
    def test_getColor(self):
        colors = Color.getColor(num=6, cmap='jet_r')
        self.assertEqual(len(colors), 6)  # Check if the returned list has 6 colors

    def test_adjust_opacity(self):
        colors = ['#FF0000', '#00FF00', '#0000FF']
        adjusted_colors = Color.adjust_opacity(colors, alpha=0.5)
        self.assertEqual(len(adjusted_colors), 3)  # Check if the length of adjusted colors is the same as input colors

    def test_color_maker(self):
        obj = [1, 2, 3, 4, 5]
        scalar_map, colors = Color.color_maker(obj, cmap='Blues')
        self.assertIsNotNone(scalar_map)  # Check if scalar map is not None
        self.assertEqual(len(colors), len(obj))  # Check if the length of colors is the same as the input object


if __name__ == '__main__':
    unittest.main()
