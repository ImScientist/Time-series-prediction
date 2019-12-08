import itertools
import datetime
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from .utils import rmspe
from typing import List, Dict, Tuple, Union, Any


def get_best_model_and_predict_01(df: pd.DataFrame,
                                  exogen_vars: List[str] = None,
                                  endogen_var: str = None,
                                  p_range: Tuple = (0, 5),
                                  d_range: Tuple = (0, 1),
                                  q_range: Tuple = (0, 5),
                                  metrics: Union[Dict, Any] = None,
                                  order_metric: str = 'rmspe'):
    """ Compare the predictions of different ARIMA models.
    The models could be (p,d,q) ARIMA models with different d - value.

    Training for the time period [2013/01/01, 2014/12/31]
    Validation for the time period [2015/01/01, 2015/07/31]

    :param df:
    :param exogen_vars:
    :param endogen_var:
    :param p_range:
    :param d_range:
    :param q_range:
    :param metrics:
    :param order_metric:
    :return:
    """

    rp, rd, rq = range(*p_range), range(*d_range), range(*q_range)

    # validation interval
    t1, t2 = datetime.datetime(2015, 1, 1), datetime.datetime(2015, 7, 31)

    # train sub-interval; used to compare predictions from the same time range
    t1_, t2_ = datetime.datetime(2014, 1, 1), datetime.datetime(2014, 7, 31)

    if metrics is None:
        metrics = {'mae': mean_absolute_error,
                   'rmspe': rmspe}

    tt = df[df['Open'] == 1] \
        .copy(deep=True) \
        .reset_index(drop=True)

    train = tt[tt['Date'] < t1]
    valid = tt[tt['Date'].between(t1, t2, inclusive=True)]
    test = tt[tt['Date'] > t2]

    train_sub_idx = tt[tt['Date'].between(t1_, t2_, inclusive=True)].index.to_list()

    if len(test) == 0:
        return None, dict()

    metric_values = []

    for pdq in list(itertools.product(rp, rd, rq)):

        try:
            # fit on training data
            mod = sm.tsa.statespace.SARIMAX(endog=train[endogen_var],
                                            exog=train[exogen_vars],
                                            order=pdq)
            fit_res = mod.fit(disp=False)

            # predictions on validation data
            prediction_valid = fit_res.forecast(steps=len(valid),
                                                exog=valid[exogen_vars])

            # prediction on train sub-interval
            prediction_train = fit_res.get_prediction(start=train_sub_idx[0],
                                                      end=train_sub_idx[-1],
                                                      dynamic=0,
                                                      full_results=False).predicted_mean

            res = dict()
            res.update(
                dict((f'{k}_valid', float(v(valid[endogen_var], prediction_valid)))
                     for k, v in metrics.items()))
            res.update(
                dict((f'{k}_train', float(v(tt.loc[train_sub_idx, endogen_var], prediction_train)))
                     for k, v in metrics.items()))

            res['n_valid'] = int(len(prediction_valid))
            res['n_train'] = int(len(prediction_train))

            res['pdq'] = pdq
            res['aic'] = float(fit_res.aic)
            res['bic'] = float(fit_res.bic)
            # tmp['prediction'] = prediction_valid
            metric_values.append(res)

        except Exception as e:
            print(e)

    metric_values = sorted(metric_values, key=lambda x: x[order_metric])

    # Take the pdq-order with the lowest error on the validation data set,
    # fit the model on the train+valid data and predict on the test data.

    mod = sm.tsa.statespace.SARIMAX(endog=tt[tt['Date'] <= t2][endogen_var],
                                    exog=tt[tt['Date'] <= t2][exogen_vars],
                                    order=metric_values[0]['pdq'])
    fit_res = mod.fit(disp=False)

    prediction = fit_res.forecast(steps=len(test),
                                  exog=test[exogen_vars])

    return prediction.to_numpy(), metric_values[0]
