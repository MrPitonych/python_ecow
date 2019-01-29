import math
import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt

def meanpar(par, timeWindow):
    parM=[]

    for i in range(math.floor(len(par)/timeWindow)): # lost data
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
    sd1 = subfunc(par,timeWindow)
    for i in range(math.floor(len(par) / timeWindow)):
        sd1[i] =np.std(sd1[i])
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


def quantiles(par,timeWindow,k=0.5):
    res=subfunc(par,timeWindow)
    for i in range(math.floor(len(par) / timeWindow)):
        res[i]= pd.Series(res[i]).quantile(k)
    return res


def skewness(par, timeWindow):
    res = subfunc(par,timeWindow)
    for i in range(math.floor(len(par) / timeWindow)):
            res[i] = pd.Series(res[i]).skew()
    return res


def kurtosis(par, timeWindow):
    res = subfunc(par,timeWindow)
    for i in range(math.floor(len(par) / timeWindow)):
        res[i] = pd.Series(res[i]).kurtosis()
    return res

def paintkde(par, timeWindow,n):
    res =[]
    try:
      print( par[(timeWindow-1) + (timeWindow * (n))])
    except:
        print('Out of range, n =0')
        n=0
    for j in range(timeWindow):
         res.append(par[j + (timeWindow * (n))])
    pd.Series(res).plot(kind='kde')
    plt.show()

def signalMagnitudeArea(par1,par2, par3, timeWindow):
    res = []
    for i in range(math.floor(len(par1) / timeWindow)):
        subpar = []
        for j in range(timeWindow):
            subpar.append((abs(par1[j + (timeWindow * (i))])+abs(par2[j + (timeWindow * (i))])
                           +abs(par3[j + (timeWindow * (i))])))
        res.append(subpar[i]/timeWindow)
    return res

def activity(par,timeWindow):
    res = subfunc(par,timeWindow)
    for i in range(math.floor(len(par) / timeWindow)):
        res[i] = np.var(res[i])
    return res

def  mobility(par,timeWindow):
    res = subfunc(par,timeWindow)
    for i in range(math.floor(len(par) / timeWindow)):
        res[i] = np.std(np.diff(res[i])/np.std(res[i]))
    return res

def complexity(par, timeWindow):
    res = subfunc(par,timeWindow)
    for i in range(math.floor(len(par) / timeWindow)):
        res[i]=(np.std(np.diff(np.diff(res[i]))) / np.std(np.diff(res[i]))
                   /(np.std(np.diff(res[i])/np.std(res[i]))))
    return res

def dwa(par,timeWindow):
    res = subfunc(par,timeWindow)
    for i in range(math.floor(len(par) / timeWindow)):
        res[i] = durbin_watson(res[i])
    return res

def subfunc(par,timeWindow):
    res = []
    for i in range(math.floor(len(par) / timeWindow)):
        subpar = []
        for j in range(timeWindow):
            subpar.append(par[j + (timeWindow * (i))])
        res.append(subpar)
    return res

