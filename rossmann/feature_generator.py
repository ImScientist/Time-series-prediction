import pandas as pd
import datetime


def promo2_running(date, promos):
    """ Is promo2 running?
    """
    if promos is not None and promos[0] <= date:
        return 1
    else:
        return 0


def dt_since_last_promo(date_cur, promos_start, fillna=-360):
    if type(promos_start) == list:
        dts = [(date_cur - pd.Timestamp(el)).days for el in promos_start]
    else:
        dts = []

    dt = max(
        max(filter(lambda x: x < 0, dts), default=fillna),
        min(filter(lambda x: x >= 0, dts), default=fillna)
    )

    return dt


def dt_since_first_promo2(date_cur, promos_start,
                          promo_na=datetime.date(2020, 1, 1)):
    if type(promos_start) == list:
        promo_start = min(promos_start)
    else:
        promo_start = promo_na

    days = (date_cur - pd.Timestamp(promo_start)).days

    return days
