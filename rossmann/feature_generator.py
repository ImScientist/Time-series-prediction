
def promo2_running(date, promos):
    """ Is promo2 running?
    """
    if promos is not None and promos[0] <= date:
        return 1
    else:
        return 0
