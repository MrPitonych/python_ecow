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
    for i in range(math.floor(len(param) / timewindow)):  # lost data
        sum1 = 0
        for j in range(timewindow):
            sum1 += param[j + (timewindow * (i))]
        parm.append(sum1 / timewindow)
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
        sum += timewindow / itinsec
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
                    list: The return value odba. Size: len(x)
    """
    res = []
    for i in range(len(x)):  # test for length x=y=z
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
        res.append(math.sqrt(pow(x[i], 2) + pow(y[i], 2) + pow(z[i], 2)))
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
        res[i] = pd.Series(res[i]).quantile(k)
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
        print(par[(timewindow - 1) + (timewindow * (n))])
    except:
        print('Out of range, n =0')
        n = 0
    for j in range(timewindow):
        res.append(par[j + (timewindow * (n))])
    pd.Series(res).plot(kind='kde')
    plt.show()


# new
def sumQuantilesMIN(param, timewindow, q=0.5, mode=1):
    """ Function for sum of values that are less than quantiles in time series.

            Args:
                param (list): Series for filtering.
                timewindow(int): Time window for separation series.
                q (float): Quantile coefficient. Default 0.5(median).
                mode (int): Operation mode, 1 = normal sum of values that are less than quantiles,
                                            2 = sum of squares of values that are less than quantiles.
                                            Default 1.
            Returns:
                list: The return value. Size: series/timewindow.
    """
    param = np.array(subfunc(param, timewindow))
    res = np.zeros(len(param))
    quantiles = np.zeros(len(param))

    for i in range(len(param)):
        quantiles[i] = np.quantile(param[i], q)

    if mode == 1:

        for i in range(len(param)):
            for j in range(len(param[i])):
                if param[i, j] < quantiles[i]:
                    res[i] += param[i, j]
    elif mode == 2:

        for i in range(len(param)):
            for j in range(len(param[i])):
                if param[i, j] < quantiles[i]:
                    res[i] += (param[i, j] ** 2)
    return res

# new
def sumQuantilesMAX(param, timewindow, q=0.5, mode=1):
    """ Function for sum of values that are less than quantiles in time series.

            Args:
                param (list): Series for filtering.
                timewindow(int): Time window for separation series.
                q (float): Quantile coefficient. Default 0.5(median).
                mode (int): Operation mode, 1 = normal sum of values that are larger than quantiles,
                                            2 = sum of squares of values that are larger than quantiles.
                                            Default 1.
            Returns:
                list: The return value. Size: series/timewindow.
    """
    param = np.array(subfunc(param, timewindow))
    res = np.zeros(len(param))
    quantiles = np.zeros(len(param))

    for i in range(len(param)):
        quantiles[i] = np.quantile(param[i], q)

    if mode == 1:

        for i in range(len(param)):
            for j in range(len(param[i])):
                if param[i, j] > quantiles[i]:
                    res[i] += param[i, j]
    elif mode == 2:

        for i in range(len(param)):
            for j in range(len(param[i])):
                if param[i, j] > quantiles[i]:
                    res[i] += (param[i, j] ** 2)
    return res


def pairwiseDifferences(param, timeWindow):
    """ Function for pairwise quantile differences in time series.

               Args:
                   param (list): Series for filtering.
                   timewindow(int): Time window for separation series.
               Returns:
                   list: The return value. Size: series/timewindow.
       """
    res = subfunc(param, timeWindow)
    q = np.quantile(res, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    counter = 0
    for i in range(len(q) - 1):
        for j in range((i + 1), len(q)):
            res.insert(counter, q[j] - q[i])
            counter = counter + 1
    return res


# new
def AAD(x, y, z, timewindow):
    """ Function  for calculate Average Absolute Difference (AAD)  in time window series .

                       Args:
                           x (list): Series data from x channel.
                           y (list): Series data from y channel.
                           z (list): Series data from z channel.
                       Returns:
                           list: The return value AAD. Size: len(x)
       """
    resX = np.array(subfunc(x, timewindow))
    resY = np.array(subfunc(y, timewindow))
    resZ = np.array(subfunc(z, timewindow))
    res = np.zeros(len(resX))
    s = np.sqrt((resX ** 2) + (resY ** 2) + (resZ ** 2))
    for i in range(len(resX)):
        res[i] = np.mean(abs(s[i] - np.mean(s[i])))
    return res


# new
def averageIntensity(x, y, z):
    """ Function  for calculate Average Intensity (AI) in time window series .

                    Args:
                        x (list): Series data from x channel.
                        y (list): Series data from y channel.
                        z (list): Series data from z channel.
                    Returns:
                        list: The return value AI. Size: len(x)
    """
    res = []
    for i in range(len(x)):
        res.append((math.sqrt(pow(x[i], 2) + pow(y[i], 2) + pow(z[i], 2))) / len(x))
    return res


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
            subpar.append(
                (abs(x[j + (timewindow * (i))]) + abs(y[j + (timewindow * (i))])  # тут бы на кол-во элементов поделить
                 + abs(z[j + (timewindow * (i))])))
        res.append(subpar[i] / timewindow)
    return res


# new
def movementVariation(x, y, z, timewindow):
    """ Function  for calculate Movement Variation in time window series.

                        Args:
                            x (list): Series data from x channel.
                            y (list): Series data from y channel.
                            z (list): Series data from z channel.
                            timewindow(int): Time window for separation series.
                        Returns:
                            list: The return value MV. Size: time series/timewindow.
    """
    resX = np.array(subfunc(x, timewindow))
    resY = np.array(subfunc(y, timewindow))
    resZ = np.array(subfunc(z, timewindow))
    res = np.zeros(len(resX))
    for i in range(len(resX)):
        res[i] = (sum(abs(resX[i, 1:len(resX[i])] - resX[i, 0:len(resY[i]) - 1])) +
                  sum(abs(resY[i, 1:len(resY[i])] - resY[i, 0:len(resY[i]) - 1])) +
                  sum(abs(resZ[i, 1:len(resZ[i])] - resZ[i, 0:len(resZ[i]) - 1]))) / (len(resX[i]))
    return res


# new
def entropy(x, y, z, timewindow):
    """ Function for calculate entropy in time window series.

                        Args:
                            x (list): Series data from x channel.
                            y (list): Series data from y channel.
                            z (list): Series data from z channel.
                            timewindow(int): Time window for separation series.
                        Returns:
                            list: The return value entropy. Size: time series/timewindow.
    """
    resX = np.array(subfunc(x, timewindow))
    resY = np.array(subfunc(y, timewindow))
    resZ = np.array(subfunc(z, timewindow))
    Ts = resX + resY + resZ
    res = np.zeros(len(resX))
    for i in range(len(resX)):
        res[i] = sum((1 + Ts[i, 0:len(Ts[i])]) * np.log(1 + abs(Ts[i, 0:len(Ts[i])]))) / len(resX[i])
    return res


# new
def energy(x, y, z, timewindow):
    """ Function for calculate energy in time window series.

                        Args:
                            x (list): Series data from x channel.
                            y (list): Series data from y channel.
                            z (list): Series data from z channel.
                            timewindow(int): Time window for separation series.
                        Returns:
                            list: The return value energy. Size: time series/timewindow.
    """
    resX = np.array(subfunc(x, timewindow))
    resY = np.array(subfunc(y, timewindow))
    resZ = np.array(subfunc(z, timewindow))
    TSS = (resX ** 2) + (resY ** 2) + (resZ ** 2)
    res = np.zeros(len(resX))
    for i in range(len(resX)):
        res[i] = sum((TSS[i, 0:len(TSS[i])]) ** 2) / len(resX[i])
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
        res[i] = np.std(np.diff(res[i]) / np.std(res[i]))
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
                  / (np.std(np.diff(res[i]) / np.std(res[i]))))
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
