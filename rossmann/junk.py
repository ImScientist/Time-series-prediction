import datetime
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from statsmodels.tsa.ar_model import AR, ARResults, ARResultsWrapper



with pm.Model() as model:
    alpha = 1.0 / count_data.mean()
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)

    idx = np.arange(n_count_data)  # Index
    lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)

    observation = pm.Poisson("obs", lambda_, observed=count_data)



# TODO
def dt_from_last_promotion(curr_date, prom_months, first_prom_start):
    """
    :param curr_date: date formatted to datetime.datetime
    :param prom_months: list of integers
    :param first_prom_start: date formatted to datetime.datetime
    :return: number of days since the last promotion has started
    """

    if prom_months and pd.notnull(first_prom_start) and pd.notnull(curr_date):

        if curr_date.year >= first_prom_start.year:

            curr_year = curr_date.year
            prev_year = curr_date.year - 1

            curr_year_candidates = list(map(lambda x: datetime.datetime(curr_year, x, 1), prom_months))
            prev_year_candidate = datetime.datetime(prev_year, prom_months[-1],
                                                    1)  # look at the last promotion from the last year (if exists)

            all_candidates = curr_year_candidates + [prev_year_candidate]

            all_candidates = filter(lambda x: first_prom_start <= x <= curr_date, all_candidates)
            all_candidates = list(all_candidates)

            if all_candidates:
                latest_prom = max(all_candidates)
                return (curr_date - latest_prom).days

    return None


def dt_from_comptetion_opening_event(curr_date, competition_opening):
    """
    :param curr_date:
    :param competition_opening:
    :return: time difference (in days) between the current date
        and the date when the competition store has opened.
    """

    if pd.notnull(competition_opening) and pd.notnull(curr_date):
        return (curr_date - competition_opening).astype('timedelta64[D]').item().days
    else:
        return None


def modify_store_open(store_open, n_customers):
    """
    Set `open` status of store from 1 to 0 if there are no customers.
    :param store_open: 1 or 0
    :param n_customers:
    :return:
    """
    if n_customers == 0 and store_open == 1:
        return 0
    else:
        return store_open


def preprocess_store_data(df_store):
    # get start of first promotion
    promo_2_since_trf = FunctionTransformer(lambda x: np.array([get_promo_date(*el) for el in x]).reshape(-1, 1),
                                            validate=False)

    df_store['Promo2Since'] = promo_2_since_trf.fit_transform(
        df_store[['Promo2SinceYear', 'Promo2SinceWeek']].values).reshape(-1)

    df_store.drop(['Promo2SinceYear', 'Promo2SinceWeek'], inplace=True, axis=1)

    # get months when promotion is restarted
    promo_2_months = FunctionTransformer(lambda x: np.array([get_promo_months(*el) for el in x]).reshape(-1, 1),
                                         validate=False)

    df_store['PromoInterval'] = promo_2_months.fit_transform(df_store[['PromoInterval']].values).reshape(-1)

    # get date when the competition has appeared
    competition_since_trf = FunctionTransformer(
        lambda x: np.array([get_competition_date(*el) for el in x]).reshape(-1, 1), validate=False)

    df_store['CompetitionSince'] = competition_since_trf.fit_transform(
        df_store[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']].values).reshape(-1)

    df_store.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], inplace=True, axis=1)

    # data type conversion
    df_store['CompetitionSince'] = pd.to_datetime(df_store['CompetitionSince'])
    df_store['Promo2Since'] = pd.to_datetime(df_store['Promo2Since'])

    df_store.set_index('Store', inplace=True)

    return df_store


def preprocess_train_test_data(df_train, df_test, df_store):
    # store open transformation (only for test data based on our observations)
    store_open_imputer = SimpleImputer(strategy='constant', fill_value=1, copy=True)

    df_test['Open'] = store_open_imputer.fit_transform(df_test[['Open']]).reshape(-1)

    # set all values of `Open=1` to `0` if `Customers==0`
    # df_test does not have a `Customers` column.
    store_open_trf = FunctionTransformer(lambda x: np.array([modify_store_open(*el) for el in x]).reshape(-1, 1),
                                         validate=False)

    df_train['Open'] = store_open_trf.fit_transform(df_train[['Open', 'Customers']].values).reshape(-1)

    # join train/test with store data (df_store)
    df_train = df_train.join(df_store, on='Store')
    df_test = df_test.join(df_store, on='Store')

    # time after last promotion (if exists)
    dt_last_prom_trf = FunctionTransformer(lambda x: np.array([dt_from_last_promotion(*el) for el in x]).reshape(-1, 1),
                                           validate=False)

    df_train['dtLastProm'] = dt_last_prom_trf.fit_transform(
        df_train[['Date', 'PromoInterval', 'Promo2Since']].values).reshape(-1)
    df_test['dtLastProm'] = dt_last_prom_trf.transform(
        df_test[['Date', 'PromoInterval', 'Promo2Since']].values).reshape(-1)

    # time since the competition has opened
    dt_competition_opened_trf = FunctionTransformer(
        lambda x: np.array([dt_from_comptetion_opening_event(*el) for el in x]).reshape(-1, 1), validate=False)

    df_train['dtCompetitionOpen'] = dt_competition_opened_trf.fit_transform(
        df_train[['Date', 'CompetitionSince']].values).reshape(-1)
    df_test['dtCompetitionOpen'] = dt_competition_opened_trf.transform(
        df_test[['Date', 'CompetitionSince']].values).reshape(-1)

    return df_train, df_test










##################################################
### pipeline uils



import sklearn


def get_pipelines(pipes_union):
    pipelines = [x[1] for x in pipes_union.get_params()['transformer_list']]
    return pipelines


def get_columns_from_pipeline(pipeline):
    """
    Fucntion to extract the column names of the pipeline.
    Use it only when the pipeline is fitted.
    """

    relevant_columns = []

    if type(pipeline) == sklearn.pipeline.Pipeline:

        # get initial column name from the column selector
        relevant_columns = pipeline[0].kw_args['relevant_columns']

        # check if the pipeline has a one hot encoder
        ohe = [trf for trf in pipeline
               if type(trf) == sklearn.preprocessing._encoders.OneHotEncoder]

        if ohe:
            categories = ohe[0].categories_

            relevant_columns = ['{0:s}_{1:s}'.format(col, str(cat))
                                for idx, col in enumerate(relevant_columns)
                                for cat in categories[idx]]

    elif type(pipeline) == sklearn.preprocessing.FunctionTransformer:
        relevant_columns = pipeline.kw_args['relevant_columns']

    if not relevant_columns:
        print('Columns not extracted from pipeline')

    return relevant_columns
