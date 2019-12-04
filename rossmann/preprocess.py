import os
import datetime
import pandas as pd
from typing import List, Dict, Tuple, Union, Any

months_to_num = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sept': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}


def get_promo_months(x: Union[List[str], Any]) -> List[int]:
    """
    :param x: string of comma separated name of months, e.g. 'Jul,Aug,Sep,Nov'
    :return: list of integers
    """
    if type(x) != str:
        return []
    else:
        months = x.split(',')
        months = map(lambda y: months_to_num[y], months)
        return list(months)


def get_promo_date(yr, week):
    """
    :param yr: year
    :param week: week
    :return: date
    """
    try:
        date = datetime.datetime(int(yr), 1, 1) + datetime.timedelta(days=7 * (int(week) - 1))
        date = datetime.date(date.year, date.month, 1)
        return date
    except (TypeError, ValueError):
        return None


def get_all_promo(since, intervals):
    if since is not None:
        promos = [datetime.date(yr, m, 1)
                  for yr in [2013, 2014, 2015]
                  for m in intervals
                  if datetime.date(yr, m, 1) >= since]
        return promos
    else:
        return None


def load_data(data_dir: str):
    dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')

    df_store = pd.read_csv(os.path.join(data_dir, 'store.csv'))
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'), parse_dates=[2], date_parser=dateparse,
                           low_memory=False)
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'), parse_dates=[3], date_parser=dateparse)

    return df_store, df_train, df_test


def fill_nans(df_store, df_train, df_test):
    df_test['Open'] = df_test['Open'].fillna(1)
    df_store['CompetitionOpenSinceMonth'] = df_store['CompetitionOpenSinceMonth'].fillna(1)
    df_store['CompetitionOpenSinceYear'] = df_store['CompetitionOpenSinceYear'].fillna(2020)

    return df_store, df_train, df_test


def preprocess_store(df_store):
    # reduce competition info to single date column
    df_store['competition_since'] = df_store[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']] \
        .apply(lambda x: datetime.date(int(x[0]), int(x[1]), 1) if x[0] > 0 and x[1] > 0 else pd.NaT, 1)

    df_store['competition_since'] = pd.to_datetime(df_store['competition_since'], format='%Y-%m-%d')
    df_store = df_store.drop(columns=['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'])

    # get dates when a promotion starts
    df_store['Promo2Since'] = df_store[['Promo2SinceYear', 'Promo2SinceWeek']] \
        .apply(lambda x: get_promo_date(x[0], x[1]), 1)

    # jan, mar, ... -> 1, 3, ...
    df_store['PromoInterval'] = df_store['PromoInterval'].apply(lambda x: get_promo_months(x))

    # get list of dates when promo2 starts
    df_store['promos2'] = df_store[['Promo2Since', 'PromoInterval']].apply(lambda x: get_all_promo(x[0], x[1]), 1)

    # drop junk
    df_store = df_store.drop(['PromoInterval', 'Promo2Since', 'Promo2SinceYear', 'Promo2SinceWeek', 'Promo2'], 1)

    return df_store


def load_and_preprocess(data_dir: str):
    """
    """
    df_store, df_train, df_test = load_data(data_dir)

    df_store, df_train, df_test = fill_nans(df_store, df_train, df_test)

    df_store = preprocess_store(df_store)

    # concat train and test data ...
    df = pd.concat([df_train, df_test], sort=False).reset_index(drop=True) \
        .sort_values(by=['Store', 'Date'])

    # ... and merge it with the stores data
    df = df.set_index(['Store']) \
        .join(df_store.set_index(['Store'])) \
        .reset_index() \
        .sort_values(by=['Store', 'Date']) \
        .reset_index(drop='True')

    return df
