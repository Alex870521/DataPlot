from DataPlot.Data_processing.IMPACT import impact_process
from DataPlot.Data_processing.csv_decorator import *


def test():
    print(__file__)
    # impact_process()
    PATH_MAIN = Path(__file__).parent.parent

    @save_to_csv(PATH_MAIN / 'output1.csv')
    def my_function(filename=None, reset=False):

        # 這個函式返回一個或多個DataFrame
        df1 = pd.DataFrame({'Time': [pd.Timestamp('2020-04-11 01:00:00'), pd.Timestamp('2020-04-11 02:00:00'),
                                     pd.Timestamp('2020-04-11 03:00:00')],
                            'A': [4, 5, 6], 'B': [4, 5, 6]}).set_index('Time')
        # df2 = pd.DataFrame({'Time': [pd.Timestamp('2020-04-11 01:00:00'), pd.Timestamp('2020-04-11 02:00:00'), pd.Timestamp('2020-04-11 03:00:00')],
        # 'C': [4, 5, 6], 'D': [10, 11, 12]}).set_index('Time')
        return df1

    abc = my_function(reset=True)