import unittest
from unittest.mock import patch
from DataPlot import *


class TestOptical(unittest.TestCase):

    @patch('DataPlot.plot.optical.Q_plot')
    def test_key_error_handling(self, mock_Q_plot):
        # 模拟在调用 plot.optical.Q_plot(['AR', 'AN'], x='dp', y='Q') 时产生 KeyError
        mock_Q_plot.side_effect = KeyError("Key not found")

        with self.assertRaises(KeyError):
            ax = plot.optical.Q_plot(['AR', 'AN'], x='dp', y='Q')


if __name__ == '__main__':
    unittest.main()
