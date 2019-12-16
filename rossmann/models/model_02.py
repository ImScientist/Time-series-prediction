import time
import pandas as pd
import datetime
from typing import List, Dict, Tuple, Union, Any

from sklearn.metrics import mean_absolute_error
from ..preprocess import load_and_preprocess
from ..feature_generator import promo2_running
from ..utils import rmspe
from ..model_selection import get_best_model_and_predict_01


def model_02(data_dir: str,
             p_range: Tuple[int, int] = (0, 6),
             d_range: Tuple[int, int] = (0, 1),
             q_range: Tuple[int, int] = (0, 1),
             n_stores: int = None):
    """ Eliminate the Christmas effects by adding new features.
    """

    exogen_vars = [
        'Promo', 'Promo2', 'SchoolHoliday', 'any_state_holiday',
        'last_competitor_here', 'days_since_start',
        'day__1', 'day__2', 'day__3', 'day__4', 'day__5', 'day__6', 'day__7',
        'weeks_left_yr__0', 'weeks_left_yr__1', 'weeks_left_yr__2', 'weeks_left_yr__3', 'weeks_left_yr__4',
        'weeks_left_yr__5', 'weeks_left_yr__6', 'weeks_left_yr__7', 'weeks_left_yr__8',
        'weeks_left_yr__9', 'weeks_left_yr__10', 'weeks_left_yr__11', 'weeks_left_yr__12',
        'weeks_since_yr__0', 'weeks_since_yr__1', 'weeks_since_yr__2', 'weeks_since_yr__3']

    df = load_and_preprocess(data_dir)

    #
    # Generate new features
    #

    # Is promo2 running?
    df['Promo2'] = df[['Date', 'promos2']].apply(lambda x: promo2_running(x[0], x[1]), 1)

    # Has the last competitor already arrived?
    df['last_competitor_here'] = (df['competition_since'] <= df['Date']).astype(int)

    # I will drop the `Date` column and use this column as my the trend generator
    df['days_since_start'] = df['Date'].apply(lambda x: (x - datetime.datetime(2013, 1, 1)).days)

    df['days_left_yr'] = df['Date'].apply(lambda x: (datetime.datetime(x.year, 12, 31) - x).days)
    df['days_since_yr'] = df['Date'].apply(lambda x: (x - datetime.datetime(x.year, 1, 1)).days)

    df['weeks_left_yr'] = df['days_left_yr'] // 7
    df['weeks_since_yr'] = df['days_since_yr'] // 7

    df['any_state_holiday'] = ((df['StateHoliday'] != '0') | (df['DayOfWeek'] == 7)).astype(int)

    df['weeks_left_yr'] = df['weeks_left_yr'] * (df['weeks_left_yr'] < 12).astype(int) + 12 * (
                df['weeks_left_yr'] >= 12).astype(int)
    df['weeks_since_yr'] = df['weeks_since_yr'] * (df['weeks_since_yr'] < 3).astype(int) + 3 * (
                df['weeks_since_yr'] >= 3).astype(int)

    # OHE of some variables

    # this will be replaced with any_state_holiday
    # df = pd.concat([df.drop(['StateHoliday'], 1),
    #                 pd.get_dummies(df['StateHoliday'], prefix='state_h_')], axis=1, sort=False)

    df = pd.concat([df.drop(['DayOfWeek'], 1),
                    pd.get_dummies(df['DayOfWeek'], prefix='day_')], axis=1, sort=False)

    df = pd.concat([df.drop(['weeks_left_yr'], 1),
                    pd.get_dummies(df['weeks_left_yr'], prefix='weeks_left_yr_')], axis=1, sort=False)

    df = pd.concat([df.drop(['weeks_since_yr'], 1),
                    pd.get_dummies(df['weeks_since_yr'], prefix='weeks_since_yr_')], axis=1, sort=False)

    #
    # Prediction
    #

    start = time.time()

    # We can immediately predict the 'Sales' if the store is closed.
    df['Sales'] = df[['Open', 'Sales']].apply(lambda x: 0 if x[0] == 0 else x[1], 1)

    # get the indices of all stores
    all_stores = list(df['Store'].drop_duplicates().values)
    if n_stores is not None:
        all_stores = all_stores[:n_stores]

    # store metrics related to the predictions for every store
    all_metrics = []

    t = datetime.datetime(2015, 7, 31)

    for store in all_stores:
        df.loc[(df['Store'] == store) &
               (df['Open'] == 1) &
               (df['Date'] > t), 'Sales'], metrics = get_best_model_and_predict_01(df=df[df['Store'] == store],
                                                                                   exogen_vars=exogen_vars,
                                                                                   endogen_var='Sales',
                                                                                   p_range=p_range,
                                                                                   d_range=d_range,
                                                                                   q_range=q_range,
                                                                                   metrics={'mae': mean_absolute_error,
                                                                                            'rmspe': rmspe},
                                                                                   order_metric='rmspe_valid')

        metrics['store'] = int(store)
        all_metrics.append(metrics)

        print(f'store {store} done; \t rel. time(s):', time.time() - start)

    result = df[df['Id'].notnull()][['Id', 'Sales']]
    result['Id'] = result['Id'].astype(int)
    result = result.set_index(['Id']).sort_index()

    return result, all_metrics
