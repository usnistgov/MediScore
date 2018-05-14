import math
import numpy as np

"""
Description: a function used to round numbers. A list of options may be provided
  to control how rounding is implemented:
    * 'sd' may be added to round by significant digits instead of by absolute digit values.
    * 't' may be added to truncate at a certain digit value instead of rounding.
"""

def myround(n,precision,mode=[]):
    if isinstance(n,str):
        return n

    if n is None:
        return np.nan
    elif np.isnan(n):
        return n

    prec = precision
    if 'sd' in mode: #significant digits
        sd1 = 0
        if n != 0:
            sd1 = -int(math.floor(math.log10(abs(n))))
        prec = precision - 1 + sd1
    if 't' in mode: #truncation
        n_r = math.floor(n*math.pow(10,prec))/math.pow(10,prec)
    else:
        n_r = round(n,prec)
    return n_r
