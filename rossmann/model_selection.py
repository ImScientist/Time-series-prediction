import itertools
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List, Dict, Tuple, Union, Any


def get_arima_ic(p_range: Tuple = (0, 5),
                 d_range: Tuple = (0, 1),
                 q_range: Tuple = (0, 5),
                 endog=None,
                 exog=None,
                 criterion: str = 'aic',
                 trend: str = 'c') \
        -> List[Dict]:
    """ Pick the right order (pdq) for an ARIMA model.
    Do not compare combinations with different `d`.

    :param p_range: lower and upper limit for possible p-values;
        All pdq combinations where p is in the range [lower limit, upper limit)
        will be generated.
    :param d_range:
    :param q_range:
    :param endog: endogen variables
    :param exog: exogen variables
    :param criterion: aic (Akaike information criterion) or bic
        (Bayesian information criterion)
    :param trend:
    :return:
        list of dicts with scores for different information criteria;
        list is ordered based on the scores of one of the information criteria.
    """

    rp, rd, rq = range(*p_range), range(*d_range), range(*q_range)

    ics = []

    for pdq in list(itertools.product(rp, rd, rq)):

        try:
            mod = sm.tsa.statespace.SARIMAX(endog=endog, exog=exog, order=pdq, trend=trend)
            fit_res = mod.fit(disp=False)
            ics.append({
                'pdq': pdq,
                'aic': fit_res.aic,
                'bic': fit_res.bic
            })
        except:
            ics.append({
                'pdq': pdq,
                'aic': np.nan,
                'bic': np.nan
            })

    ics = sorted(ics, key=lambda x: x[criterion])

    return ics


def get_best_model_and_predict_01(df: pd.DataFrame = None,
                                  p_range=(0, 6),
                                  d_range=(0, 1),
                                  q_range=(0, 4)):
    """
    :param df: pd.DataFrame containing train and test data
        from a particular store
    :param p_range: [min, max) values of p the grid search
    :param d_range: [min, max) values of d the grid search
    :param q_range: [min, max) values of q the grid search
    :return:
        prediction
    """
    exogen_vars = ['Promo', 'Promo2', 'SchoolHoliday',
                   'last_competitor_here', 'days_since_start',
                   'state_h__0', 'state_h__a', 'state_h__b', 'state_h__c',
                   'day__1', 'day__2', 'day__3', 'day__4', 'day__5', 'day__6', 'day__7']

    tt = df[df['Open'] == 1] \
        .copy(deep=True) \
        .reset_index(drop=True)

    train = tt[(tt['Date'] <= datetime.datetime(2015, 7, 31))]
    test = tt[(tt['Date'] > datetime.datetime(2015, 7, 31))]

    # Model selection using AIC information criterion.
    ics = get_arima_ic(p_range=p_range,
                       d_range=d_range,
                       q_range=q_range,
                       endog=train['Sales'],
                       exog=train[exogen_vars],
                       criterion='aic',
                       trend='c')

    mod = sm.tsa.statespace.SARIMAX(endog=train['Sales'],
                                    exog=train[exogen_vars],
                                    order=ics[0]['pdq'],
                                    trend='c')
    fit_res = mod.fit(disp=False)

    prediction = fit_res.forecast(steps=len(test), exog=test[exogen_vars])

    return prediction.to_numpy()
