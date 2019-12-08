import numpy as np


def rmspe(y, y_hat):
    """Root Mean Square Percentage Error
    """
    # Assume that y>0 is fulfilled
    return np.sqrt(sum(((y - y_hat) / y) ** 2 / len(y)))
