# File of all channels parameters. In the process
"""  Module for computation channels parameters. It can be used for getting signs."""
import math
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt


def meanpar(param, timewindow):
    """ Function  for calculate mean value in time window series.

        Args:
            param (list): Series data.
            timewindow(int): Time window for separation series.
        Returns:
            list: The return value. Size: series/timewindow.
    """
    parm = []
    for i in range(math.floor(len(param)/timewindow)): # lost data
        sum1 = 0
        for j in range(timewindow):
                sum1 += param[j+(timewindow *(i))]
        parm.append(sum1/timewindow)
    return parm


def meantime(param, timewindow):
    """ Function  for calculate mean value(time) in time window series.

            Args:
                param (list): All time series.
                timewindow(int): Time window for separation series.
            Returns:
                list: The return value. Size: time series/timewindow.
    """
    parm = []
    sum = 0
    itinsec = 55
    for i in range(math.floor(len(param) / timewindow)):
        sum += timewindow/itinsec
        parm.append(sum)

    return parm


def sd(param, timewindow):
    """ Function  for calculate standard deviation value in time window series .

            Args:
                param (list): Series data.
                timewindow(int): Time window for separation series.
            Returns:
                list: The return value. Size: series/timewindow.
    """
    sd1 = subfunc(param, timewindow)
    for i in range(math.floor(len(param) / timewindow)):
        sd1[i] = np.std(sd1[i])
    return sd1


def odba(x, y, z):
    """ Function  for calculate overall dynamic body acceleration(odba) in time window series .

                Args:
                    x (list): Series data from x channel.
                    y (list): Series data from y channel.
                    z (list): Series data from z channel.
                Returns:
                    list: The return value odba.Size: len(x)
    """
    res = []
    for i in range(len(x)): # test for length x=y=z
        res.append(abs(x[i] + y[i] + z[i]))
    return res


def vedba(x, y, z):
    """ Function  for calculate vectorial dynamic body acceleration(vedba) in time window series .

                    Args:
                        x (list): Series data from x channel.
                        y (list): Series data from y channel.
                        z (list): Series data from z channel.
                    Returns:
                        list: The return value vedba. Size: len(x)
    """
    res = []
    for i in range(len(x)):
        res.append(math.sqrt(pow(x[i], 2)+pow(y[i], 2)+pow(z[i], 2)))
    return res


def quantiles(param, timewindow, k=0.5):
    """ Function for calculate quantiles in time series.

            Args:
                param (list): Series for filtering.
                timewindow(int): Time window for separation series.
                k (float): Quantile coefficient. Default 0.5(median).
            Returns:
                list: The return value. Size: series/timewindow.
    """
    res = subfunc(param, timewindow)
    for i in range(math.floor(len(param) / timewindow)):
        res[i]= pd.Series(res[i]).quantile(k)
    return res


def skewness(param, timewindow):
    """ Function  for calculate skewness in time window series.

                Args:
                    param (list): All time series.
                    timewindow(int): Time window for separation series.
                Returns:
                    list: The return value. Size: time series/timewindow.
    """
    res = subfunc(param, timewindow)
    for i in range(math.floor(len(param) / timewindow)):
            res[i] = pd.Series(res[i]).skew()
    return res


def kurtosis(param, timewindow):
    """ Function  for calculate kurtosis in time window series.

                    Args:
                        param (list): All time series.
                        timewindow(int): Time window for separation series.
                    Returns:
                        list: The return value. Size: time series/timewindow.
    """
    res = subfunc(param, timewindow)
    for i in range(math.floor(len(param) / timewindow)):
        res[i] = pd.Series(res[i]).kurtosis()
    return res


def paintkde(par, timewindow, n):
    """ Function  for print normal distribution graph in time window series.

                        Args:
                            param (list): All time series.
                            timewindow(int): Time window for separation series.
                            n(int): Subseries index.

    """
    res = []
    try:
        print(par[(timewindow-1) + (timewindow * (n))])
    except:
        print('Out of range, n =0')
        n = 0
    for j in range(timewindow):
        res.append(par[j + (timewindow * (n))])
    pd.Series(res).plot(kind='kde')
    plt.show()


def signalMagnitudeArea(x, y, z, timewindow):
    """ Function  for calculate signal Magnitude Area in time window series.

                        Args:
                            x (list): Series data from x channel.
                            y (list): Series data from y channel.
                            z (list): Series data from z channel.
                            timewindow(int): Time window for separation series.
                        Returns:
                            list: The return value sma. Size: time series/timewindow.
    """
    res = []
    for i in range(math.floor(len(x) / timewindow)):
        subpar = []
        for j in range(timewindow):
            subpar.append((abs(x[j + (timewindow * (i))])+abs(y[j + (timewindow * (i))])
                           + abs(z[j + (timewindow * (i))])))
        res.append(subpar[i]/timewindow)
    return res

def movementVariation(x, y, z, timewindow):
    """ Function  for calculate Movement Variation in time window series.

                        Args:
                            x (list): Series data from x channel.
                            y (list): Series data from y channel.
                            z (list): Series data from z channel.
                            timewindow(int): Time window for separation series.
                        Returns:
                            list: The return value sma. Size: time series/timewindow.
    """
    res = []
    for i in range(math.floor(len(x) / timewindow)):
        subpar = []
        for j in range(timewindow):
            subpar.append((abs(x[j + (timewindow * (i))])+abs(y[j + (timewindow * (i))])
                           + abs(z[j + (timewindow * (i))])))
        res.append(subpar[i]/timewindow)
    return res


def activity(param, timewindow):
    """ Function  for calculate activity in time window series.

                        Args:
                            param (list): All time series.
                            timewindow(int): Time window for separation series.
                        Returns:
                            list: The return value. Size: time series/timewindow.
    """
    res = subfunc(param, timewindow)
    for i in range(math.floor(len(param) / timewindow)):
        res[i] = np.var(res[i])
    return res


def mobility(param, timewindow):
    """ Function  for calculate mobility in time window series.

                            Args:
                                param (list): All time series.
                                timewindow(int): Time window for separation series.
                            Returns:
                                list: The return value. Size: time series/timewindow.
    """
    res = subfunc(param, timewindow)
    for i in range(math.floor(len(param) / timewindow)):
        res[i] = np.std(np.diff(res[i])/np.std(res[i]))
    return res


def complexity(param, timewindow):
    """ Function  for calculate complexity in time window series.

                            Args:
                                param (list): All time series.
                                timewindow(int): Time window for separation series.
                            Returns:
                                list: The return value. Size: time series/timewindow.
        """
    res = subfunc(param, timewindow)
    for i in range(math.floor(len(param) / timewindow)):
        res[i] = (np.std(np.diff(np.diff(res[i]))) / np.std(np.diff(res[i]))
                   / (np.std(np.diff(res[i])/np.std(res[i]))))
    return res


def dwa(param, timewindow):
    """ Function  for calculate Durbin-Watson in time window series.

                            Args:
                                param (list): All time series.
                                timewindow(int): Time window for separation series.
                            Returns:
                                list: The return value. Size: time series/timewindow.
        """
    res = subfunc(param, timewindow)
    for i in range(math.floor(len(param) / timewindow)):
        res[i] = durbin_watson(res[i])
    return res


def subfunc(param, timewindow):
    """ Function  for generate subseries list in time window series.

                            Args:
                                param (list): All time series.
                                timewindow(int): Time window for separation series.
                            Returns:
                                list: The return value.  Size: time series/timewindow.
        """
    res = []
    for i in range(math.floor(len(param) / timewindow)):
        subpar = []
        for j in range(timewindow):
            subpar.append(param[j + (timewindow * (i))])
        res.append(subpar)
    return res

