from datetime import datetime
from data_processing import integrate


# set_group
Seasons = {'2020-Summer': (datetime(2020, 9, 4), datetime(2020, 9, 21, 23)),
           '2020-Autumn': (datetime(2020, 9, 22), datetime(2020, 12, 29, 23)),
           '2020-Winter': (datetime(2020, 12, 30), datetime(2021, 3, 25, 23)),
           '2021-Spring': (datetime(2021, 3, 26), datetime(2021, 5, 6, 23))}
           # '2021-Summer': (datetime(2021, 5, 7), datetime(2021, 10, 16, 23))
           # '2021-Autumn': (datetime(2021, 10, 17), datetime(2021, 12, 31, 23))


def season_classify(df) -> dict:
    """
    Classifying by 'Seasons'.

    Parameters
    ----------
    df : pandas.DataFrame
        The data you want to classify.

    Returns
    -------
    dict
        A mapping from classify_name to a dataframe.

    """

    # set classify tags
    df['Month'] = df.index.strftime('%Y-%m')
    df['Hour'] = df.index.hour
    df.loc[(df['Hour'] <= 18) & (df['Hour'] >= 7), 'Diurnal'] = 'Day'
    df.loc[(df['Hour'] <= 23) & (df['Hour'] >= 19), 'Diurnal'] = 'Night'
    df.loc[(df['Hour'] <= 6) & (df['Hour'] >= 0), 'Diurnal'] = 'Night'

    for season, (season_start, season_end) in Seasons.items():
        df.loc[season_start: season_end, 'Season'] = season

    # parameter
    dic_grp_sea = {}
    df_group_season = df.groupby('Season')

    # Output
    for _grp, _df in df_group_season:
        # get season event and clean threshold
        clean_upp_boud, event_low_boud = _df.Extinction.quantile([0.2, 0.8])
        _df_ext = _df.Extinction.copy()

        # event classify
        _df.loc[_df_ext >= event_low_boud, 'State'] = 'Event'
        _df.loc[(_df_ext < event_low_boud) & (_df_ext > clean_upp_boud), 'State'] = 'Transition'
        _df.loc[_df_ext <= clean_upp_boud, 'State'] = 'Clean'

        # condition
        cond_event = _df.State == 'Event'
        cond_transition = _df.State == 'Transition'
        cond_clean = _df.State == 'Clean'

        # origin output
        dic_grp_sea[_grp] = {'Total': _df.copy(),
                             'Clean': _df.loc[cond_clean].copy(),
                             'Transition': _df.loc[cond_transition].copy(),
                             'Event': _df.loc[cond_event].copy()}

    return dic_grp_sea


def state_classify(df):
    """
    Classifying by 'States'.

    Parameters
    ----------
    df : pandas.DataFrame
        The data you want to classify.

    Returns
    -------
    dict
        A mapping from classify_name to a dataframe.

    """

    # set classify tags
    df['Month'] = df.index.strftime('%Y-%m')
    df['Hour'] = df.index.hour
    df.loc[(df['Hour'] <= 18) & (df['Hour'] >= 7), 'Diurnal'] = 'Day'
    df.loc[(df['Hour'] <= 23) & (df['Hour'] >= 19), 'Diurnal'] = 'Night'
    df.loc[(df['Hour'] <= 6) & (df['Hour'] >= 0), 'Diurnal'] = 'Night'

    clean_upp_boud, event_low_boud = df.Extinction.quantile([0.2, 0.8])
    _df_ext = df.Extinction.copy()

    # event classify
    df.loc[_df_ext >= event_low_boud, 'State'] = 'Event'
    df.loc[(_df_ext < event_low_boud) & (_df_ext > clean_upp_boud), 'State'] = 'Transition'
    df.loc[_df_ext <= clean_upp_boud, 'State'] = 'Clean'

    # condition
    cond_event = df.State == 'Event'
    cond_transition = df.State == 'Transition'
    cond_clean = df.State == 'Clean'

    # origin output
    dic_grp_sta = {'Total': df.copy(),
                   'Clean': df.loc[cond_clean].copy(),
                   'Transition': df.loc[cond_transition].copy(),
                   'Event': df.loc[cond_event].copy()}

    return dic_grp_sta


def four_quadrant(df):
    _dic = {}
    _df = df.copy()
    _dic['1'] = df.loc[(_df.Extinction < 68.22) & (_df.PM25 >= 16)]
    _dic['2'] = df.loc[(_df.Extinction < 68.22) & (_df.PM25 < 16)]
    _dic['3'] = df.loc[(_df.Extinction >= 68.22) & (_df.PM25 < 16)]
    _dic['4'] = df.loc[(_df.Extinction >= 68.22) & (_df.PM25 >= 16)]
    return _dic


if __name__ == '__main__':
    df = integrate(reset=False)
