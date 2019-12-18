import xgboost as xgb
import datetime
from typing import List, Dict, Tuple, Union, Any

from ..preprocess import load_and_preprocess
from ..utils import rmspe_xg
from ..feature_generator import dt_since_last_promo, dt_since_first_promo2


def model_03(data_dir: str,
             num_boost_round: int = 2,
             early_stopping_rounds: int = None,
             params: Dict = None):
    """ Use xgboost
    """

    df = load_and_preprocess(data_dir)

    #
    # Generate new features
    #

    df['dt_since_last_promo2'] = df[['Date', 'promos2']].apply(lambda x: dt_since_last_promo(x[0], x[1]), 1)
    df['dt_since_first_promo2'] = df[['Date', 'promos2']].apply(lambda x: dt_since_first_promo2(x[0], x[1]), 1)

    # I will drop the `Date` column and use this column as my the trend generator
    df['days_since_start'] = df['Date'].apply(lambda x: (x - datetime.datetime(2013, 1, 1)).days)

    df['days_left_yr'] = df['Date'].apply(lambda x: (datetime.datetime(x.year, 12, 31) - x).days)
    df['days_since_yr'] = df['Date'].apply(lambda x: (x - datetime.datetime(x.year, 1, 1)).days)

    df['weeks_left_yr'] = df['days_left_yr'] // 7
    df['weeks_since_yr'] = df['days_since_yr'] // 7

    # two options for 'competition_since'
    # competition already here (binary) or days since competition is here (continuous variable)
    df['competition_since_days'] = df['Date'] - df['competition_since']
    df['competition_since_days'] = df['competition_since_days'].apply(lambda x: x.days)

    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(100000).astype(int)

    # no OHE of some variables
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    df['StateHoliday'] = df['StateHoliday'].replace(mappings).astype(int)
    df['Assortment'] = df['Assortment'].replace(mappings).astype(int)
    df['StoreType'] = df['StoreType'].replace(mappings).astype(int)


    #
    # Training
    #

    feature_pairs = [
        ('Store', 'int'),

        ('DayOfWeek', 'int'),
        ('days_since_start', 'q'),
        ('days_since_yr', 'q'),
        ('weeks_since_yr', 'q'),

        ('days_left_yr', 'q'),
        ('weeks_left_yr', 'q'),

        ('Promo', 'i'),
        ('dt_since_last_promo2', 'q'),
        ('dt_since_first_promo2', 'q'),

        ('StateHoliday', 'int'),
        ('SchoolHoliday', 'i'),

        ('StoreType', 'int'),
        ('Assortment', 'int'),

        ('CompetitionDistance', 'q'),
        ('competition_since_days', 'q')
    ]

    features = list(map(lambda x: x[0], feature_pairs))
    feature_types = list(map(lambda x: x[1], feature_pairs))

    t1, t2 = datetime.datetime(2015, 4, 1), datetime.datetime(2015, 7, 31)

    df_tr = df[(df['Date'] < t1) & (df['Open'] == 1)].copy(deep=True)
    df_va = df[(df['Date'].between(t1, t2, inclusive=True)) & (df['Open'] == 1)].copy(deep=True)
    df_te = df[df['Date'] > t2].copy(deep=True)

    label_col = 'Sales'
    train_log = dict()

    dtr = xgb.DMatrix(data=df_tr[features].to_numpy(),
                      label=df_tr[label_col].to_numpy(),
                      feature_names=features,
                      feature_types=feature_types)

    dva = xgb.DMatrix(data=df_va[features].to_numpy(),
                      label=df_va[label_col].to_numpy(),
                      feature_names=features,
                      feature_types=feature_types)

    watchlist = [(dtr, 'train'), (dva, 'eval')]

    gbm = xgb.train(params, dtr,
                    num_boost_round,
                    evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds,
                    feval=rmspe_xg,
                    evals_result=train_log,
                    verbose_eval=True)

    #
    # Predictions
    #

    # We can immediately predict the 'Sales' if the store is closed.
    df_te.loc[df_te['Open'] == 0, 'Sales'] = 0
    df_te.loc[df_te['Open'] == 1, 'Sales'] = gbm.predict(
        xgb.DMatrix(data=df_te[df_te['Open'] == 1][features].to_numpy(),
                    feature_names=features))

    result = df_te[['Id', 'Sales']].copy(deep=True)
    result['Id'] = result['Id'].astype(int)
    result = result.set_index(['Id'], drop=True).sort_index()

    return result, gbm, train_log
