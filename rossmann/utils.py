import numpy as np


def rmspe(y, yhat):
    """Root Mean Square Percentage Error
    """
    # Assume that y>0 is fulfilled
    return np.sqrt(np.mean((yhat / y - 1) ** 2))


def rmspe_xg(yhat, y):
    y = y.get_label()
    #     y = np.expm1(y.get_label())
    #     yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)
