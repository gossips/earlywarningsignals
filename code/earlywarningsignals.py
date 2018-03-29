#!/bin/python3

import scipy.stats
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy import interpolate

## Description of the collection of functions

def checkSpacing(iterator):
    iterator=np.asarray(iterator)
    return len(set(iterator)) <= 1 #set builds an unordered collection of unique elements.

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
            spaced = timeindex[1:]-timeindex[0:-1]
            evenly = checkSpacing(spaced)
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

def EWS(ts,autocorrelation=False,variance=False,skewness=False,
        kurtosis=False, CV=False):
      
    """Function that calculates early warning signals
    
    :param timeseries: Original timeseries (column for every variable)
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
    
    if autocorrelation == True:
        AC=[0]*nr_vars
    if variance == True:
        Var=[0]*nr_vars
    if skewness == True:
        Skews=[0]*nr_vars
    if kurtosis == True:
        Kurt=[0]*nr_vars
    if CV == True:
        CVs=[0]*nr_vars
    
    for i in range(nr_vars):
        
        if autocorrelation == True:
            AC[i]=np.corrcoef(timeseries[1:,i],timeseries[:-1,i])[1,0]
            result.update({'autocorrelation' : AC})            

        if variance == True:
            Var[i]=np.var(timeseries[:,i])
            result.update({'variance' : Var})            

        if skewness == True:
            Skews[i]=scipy.stats.skew(timeseries[:,i])
            result.update({'skewness' : Skews})            

        if kurtosis == True:
            Kurt[i]=scipy.stats.kurtosis(timeseries[:,i])
            result.update({'kurtosis' : Kurt})

        if CV == True:
            CVs[i]=np.std(timeseries[:,i])/np.mean(timeseries[:,i])
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

def kendalltau(indicatorvec):
    ## Kendall trend statistic
    timevec = range(len(indicatorvec))
    tau, p_value = scipy.stats.kendalltau(timevec,indicatorvec)
    return [tau, p_value]

# temporary input for Kendall trend statistic (remove when other functions are ready)
nARR = [2,3,4,6,7,8,5,9,10,20]
nACF = [2,3,4,6,7,8,5,9,10,20]
nSD = [2,3,4,6,7,8,5,9,10,20]
nSK = [2,3,4,6,7,8,5,9,10,20]
nKURT = [2,3,4,6,7,8,5,9,10,20]
nDENSITYRATIO = [2,3,4,6,7,8,5,9,10,20]
nRETURNRATE = [2,3,4,6,7,8,5,9,10,20]
nCV = [2,3,4,6,7,8,5,9,10,20]

# Estimate Kendall trend statistic for indicators (ouput: [Tau,p_value])
KtAR=kendalltau(nARR)
KtACF=kendalltau(nACF)
KtSD=kendalltau(nSD)
KtSK=kendalltau(nSK)
KtKU=kendalltau(nKURT)
KtDENSITYRATIO=kendalltau(nDENSITYRATIO)
KtRETURNRATE=kendalltau(nRETURNRATE)
KtCV=kendalltau(nCV)

#print Kendall output (to be removed later?)
print('\nKtAR (Tau,p_value): %.4f, %.4f' % (KtAR[0],KtAR[1]))
print('\nKtACF (Tau,p_value): %.4f, %.4f' % (KtACF[0],KtACF[1]))
print('\nKtSD (Tau,p_value): %.4f, %.4f' % (KtSD[0],KtSD[1]))
print('\nKtSK (Tau,p_value): %.4f, %.4f' % (KtSK[0],KtSK[1]))
print('\nKtKU (Tau,p_value): %.4f, %.4f' % (KtKU[0],KtKU[1]))
print('\nKtDENSITYRATIO (Tau,p_value): %.4f, %.4f' % (KtDENSITYRATIO[0],KtDENSITYRATIO[1]))
print('\nKtRETURNRATE (Tau,p_value): %.4f, %.4f' % (KtRETURNRATE[0],KtRETURNRATE[1]))
print('\nKtCV (Tau,p_value): %.4f, %.4f' % (KtCV[0],KtCV[1]))

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
        :rtype: array of interpolated values 
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