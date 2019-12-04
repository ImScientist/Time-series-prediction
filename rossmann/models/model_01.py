import pandas as pd
import datetime
from typing import List, Dict, Tuple, Union, Any

from ..preprocess import load_and_preprocess
from ..feature_generator import promo2_running
from ..model_selection import get_best_model_and_predict_01


def model_01(data_dir: str,
             p_range: Tuple[int, int] = (0, 6),
             d_range: Tuple[int, int] = (0, 1),
             q_range: Tuple[int, int] = (0, 4),
             n_stores: int = None):
    df = load_and_preprocess(data_dir)

    # Generate new features

    # Is promo2 running?
    df['Promo2'] = df[['Date', 'promos2']].apply(lambda x: promo2_running(x[0], x[1]), 1)

    # Has the last competitor already arrived?
    df['last_competitor_here'] = (df['competition_since'] <= df['Date']).astype(int)

    # I will drop the `Date` column and use this column as my the trend generator
    df['days_since_start'] = df['Date'].apply(lambda x: (x - datetime.datetime(2013, 1, 1)).days)

    # ohe of some variables
    df = pd.concat([df.drop(['StateHoliday'], 1),
                    pd.get_dummies(df['StateHoliday'], prefix='state_h_')], axis=1, sort=False)

    df = pd.concat([df.drop(['DayOfWeek'], 1),
                    pd.get_dummies(df['DayOfWeek'], prefix='day_')], axis=1, sort=False)

    # Prediction

    # We can immediately predict the 'Sales' if the store is closed.
    df['Sales'] = df[['Open', 'Sales']].apply(lambda x: 0 if x[0] == 0 else x[1], 1)

    # get the indices of all stores
    all_stores = list(df['Store'].drop_duplicates().values)
    if n_stores is not None:
        all_stores = all_stores[:n_stores]

    for store in all_stores:
        df.loc[(df['Store'] == store) &
               (df['Open'] == 1) &
               (df['Date'] > datetime.datetime(2015, 7, 31)), 'Sales'] = \
            get_best_model_and_predict_01(df[df['Store'] == store], p_range, d_range, q_range)

    result = df[df['Id'].notnull()][['Id', 'Sales']]
    result['Id'] = result['Id'].astype(int)
    result = result.set_index(['Id'])

    return result
