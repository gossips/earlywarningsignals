#!/bin/python3

import scipy.stats
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

## Description of the collection of functions

def isUniformSpacing(vector):
    """Checks if the spacing is uniform

    :param vector: the vector to be checked
    """

    vector=np.asarray(vector)
    diff = vector[1:]-vector[0:-1]

    # Spacing is uniform if and only if maximum and minimum steps are the same
    return (np.max(diff) == np.min(diff))

def check_time_series(data, timeindex=None):
    ## Dummy function.
    """Check if the timeseries are in the suitable format (pd DataFrame with only numeric values).

    :param timeindex: the timeindex of the data
    :param data: the data, can be univariate or multivariate, as Pandas DataFrame

    """
    timeseries = pd.DataFrame(data=data)
    if timeindex is None:
        timeindex = np.linspace(0,timeseries.shape[0]-1, timeseries.shape[0])
    else:
        if isinstance(timeindex, np.ndarray):
            timeindex = pd.DataFrame(timeindex, columns=['Time'])
        if isinstance(timeindex, pd.DataFrame):
            timeindex = np.asarray(timeindex)
            evenly = isUniformSpacing(spaced)
            if evenly == False:
                print("time index is not evenly spaced.")
        if timeseries.shape[0] == timeindex.shape[0]:
            print("right format for analysis")
        else:
            print("timeindex and data do not have the same length")
    return timeseries, timeindex

def logtransform(df):

    """Compute the logtransform of the original timeseries

    :param df: dataframe with original timeseries
    :return: new dataframe with logtransformed timeseries


    Created by Ingrid van de Leemput
    """
    df=df+1
    df_log=df.apply(np.log)
    return df_log

def detrend(timeseries, detrending='gaussian', bandwidth=None, span=None, degree=None):

    """Detrend time series.

    :param timeseries: array with time indices in the first column and time series values in the second.
    :param detrending: either
        'gaussian' = Gaussian detrending
        'linear' = linear regression
        'loess' = local nonlinear regression
        'first_diff' = first-difference filtering
        'no' = no detrending
    :param bandwidth: bandwidth for Gaussian detrending. If None, chooses default bandwidth (using Silverman's rule of thumb).
    :param span: window size in case of loess, in percentage of time series length. If None, chooses default span (25%).
    :param degree: degree of polynomial in case of loess. If None, chooses default degree of 2.
    :return: trend and residuals. In case of first_diff, returns residuals and difference between consecutive time values.

    Created by Arie Staal
    """

    ts = timeseries[:,1]
    time_index = timeseries[:,0]

    if detrending == 'gaussian':

        if bandwidth == None:
            # Silverman's rule of thumb
            bw = 0.9 * min(np.std(ts), (np.percentile(ts, 75) - np.percentile(ts, 25)) / 1.34) * len(ts)**(-0.2)
        else:
            bw = round(len(ts) * bandwidth/100)

        trend = gaussian_filter1d(ts, bw, axis=0) # smY in R code
        resid = ts - trend                        # nsmY in R code

    elif detrending == 'linear':

        x = np.linspace(0, len(ts), len(ts))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,ts)
        trend = intercept + slope * x
        resid = ts - trend

    elif detrending == 'loess':

        if span == None:
            span = 25/100
        else:
            span = span/100

        if degree == None:
            degree = 2
        else:
            degree = degree

        # Here include code for local nonlinear regression

    elif detrending == 'first_diff':

        resid = np.diff(ts, n=1, axis=0)
        time_index_diff = time_index[0:(len(time_index) - 1)]

    elif detrending == 'no':

        trend = ts
        resid = ts

    if detrending == 'first_diff':
        return resid, time_index_diff
    else:
        return trend, resid

def EWS(ts,window_size=None,autocorrelation=False,variance=False,skewness=False,
        kurtosis=False, CV=False):

    """Function that calculates early warning signals

    :param timeseries: Original timeseries (column for every variable)
    :param window_size: Set to size of rolling window, default setting does not use a rolling window
    :param autocorrelation: Set to True if autocorrelation is required in output
    :param variance: Set to True if variance is required in output
    :param skewness: Set to True if skewness is required in output
    :param kurtosis: Set to True if kurtosis is required (Fishers definition used,
    so it is zero for a normal distribution)
    :param CV: Set to True if coefficient of variation is required
    :return: dict with the chosen output and for every output an array with the
    values for each variable (every column).
    """

    timeseries=pd.DataFrame.as_matrix(ts)
    nr_vars=len(timeseries[0,:])

    result={}

    if window_size == None:
        libs = 1
        window_size = len(timeseries[:,0])
    else:
        libs = len(timeseries[:,0]) - window_size + 1

    if autocorrelation == True:
        AC=np.zeros((libs,nr_vars))
    if variance == True:
        Var=np.zeros((libs,nr_vars))
    if skewness == True:
        Skews=np.zeros((libs,nr_vars))
    if kurtosis == True:
        Kurt=np.zeros((libs,nr_vars))
    if CV == True:
        CVs=np.zeros((libs,nr_vars))

    for j in range(libs):
        lib_ts=timeseries[j:j+window_size,:]

        for i in range(nr_vars):

            if autocorrelation == True:
                AC[j,i]=np.corrcoef(lib_ts[1:,i],lib_ts[:-1,i])[1,0]
                result.update({'autocorrelation' : AC})

            if variance == True:
                Var[j,i]=np.var(lib_ts[:,i])
                result.update({'variance' : Var})

            if skewness == True:
                Skews[j,i]=scipy.stats.skew(lib_ts[:,i])
                result.update({'skewness' : Skews})

            if kurtosis == True:
                Kurt[j,i]=scipy.stats.kurtosis(lib_ts[:,i])
                result.update({'kurtosis' : Kurt})

            if CV == True:
                CVs[j,i]=np.std(lib_ts[:,i])/np.mean(lib_ts[:,i])
                result.update({'CV' : CVs})

    return result


def EWS_rolling_window(df,winsize=50):

    """Use a rolling window on which EWS are calculated

    :param df: dataframe with original timeseries
    :param winsize:  the size of the rolling window expressed as percentage of the timeseries length (must be numeric between 0 and 100). Default is 50\%.
    :return: dataframe with result

    Created by Ingrid van de Leemput
    """

    mw=round(len(df.index) * winsize/100) # length moving window
    df.rolling(window=mw).apply(func=EWS)

    #How to do this nicely? I think we should plug this into the EWS function!

def kendalltrend(ts):
    """Calculates the Kendall trend statistic for a EW indicators

    :ts: timeseries for the indicator. Can be array, series or a dataframe
    :return: Kendall tau value and p_value

    """
    ti = range(len(ts))
    tau, p_value = scipy.stats.kendalltau(ti,ts)
    return [tau, p_value]


#Interpolation function
def interp(x, y, new_x, method = 'linear', spline = False, k = 3, s = 0, der = 0):
    """
    .. function:: interp(x, y, new_x, dim, method, spline, k, s, der)
        Function interpolates timeseries with different methods.
        :param x: time index. In case of dataframe use df.iloc to refer the correct column. Required value.
        :param y: Original data values. Must be the same lenght as x. In case of dataframe use df.iloc to refer the correct column. Required value.
        :param new_x: Index values at which to interpolate data. Required value.
        :param method: Specifies interpolation method used. One of
            'nearest': return the value at the data point closest to the point of interpolation.
            'linear': interpolates linearly between data points on new data points
            'quadratic': Interpolated values determined from a quadratic spline
            'cubic': Interpolated values determined from a cubic spline
            Default is 'linear'
        :param spline: Spline interpolation. Can be True or False. If True then the function ignores the method call. Default is False.
        :param k: Degree of the spline fit. Must be <= 5. Default is k=3, a cubic spline.
        :param s: Smoothing value. Default is 0. A rule of thumb can be s = m - sqrt(m) where m is the number of data-points being fit.
        :param der: The order of derivative of the spline to compute (must be less than or equal to k)
        Created by M Usman Mirza
    """
    if spline == False:
        f = interpolate.interp1d(x = x, y = y, kind = method)
        i = f(new_x)
        return i
    elif spline == True:
        f = interpolate.splrep(x = x, y = y, k = k, s = s)
        i = interpolate.splev(x = new_x, tck = f, der = der)
        return i
