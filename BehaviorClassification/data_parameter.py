import math
import numpy as np
import pandas as pd

def meanpar(par, timeWindow):
    parM=[]

    for i in range(math.floor(len(par)/timeWindow)):
        sum = 0
        for j in range(timeWindow):
                sum += par[j+(timeWindow*(i))]
        parM.append(sum/timeWindow)
    return parM


def meantime(par, timeWindow):
    parM=[]
    sum=0
    itinsec=55
    for i in range(math.floor(len(par) / timeWindow)):
        sum+=timeWindow/itinsec
        parM.append(sum)

    return parM

def sd(par, timeWindow):
    sd1 = []
    for i in range(math.floor(len(par) / timeWindow)):
        subpar = []
        for j in range(timeWindow):
            subpar.append(par[j+(timeWindow*(i))])
        sd1.append(np.std(subpar))
    return sd1


def odba(par1,par2, par3):
    res=[]
    for i in range(len(par1)):
        res.append(abs(par1[i] + par2[i] + par3[i]))
    return res


def vedba(par1,par2, par3):
    res=[]
    for i in range(len(par1)):
        res.append(math.sqrt(pow(par1[i], 2)+pow(par2[i], 2)+pow(par3[i], 2)))
    return res


def quantiles(par,timeWindow,k):
    res=[]
    for i in range(math.floor(len(par) / timeWindow)):
        subpar = []
        for j in range(timeWindow):
            subpar.append(par[j+(timeWindow*(i))])
        res.append(pd.Series(subpar).quantile(k))
    return res


def skewness(par, timeWindow):
    res = []
    for i in range(math.floor(len(par) / timeWindow)):
        subpar = []
        for j in range(timeWindow):
            subpar.append(par[j + (timeWindow * (i))])
        res.append(pd.Series(subpar).skew())
    return res

def kurtosis(par, timeWindow):
    res = []
    for i in range(math.floor(len(par) / timeWindow)):
        subpar = []
        for j in range(timeWindow):
            subpar.append(par[j + (timeWindow * (i))])
        res.append(pd.Series(subpar).kurtosis())
    return res
