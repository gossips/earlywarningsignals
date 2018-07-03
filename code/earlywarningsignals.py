#!/bin/python3

import scipy.stats
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy import interpolate

## Description of the collection of functions

def checkSpacing(iterator):
    iterator=np.asarray(iterator)
    spaced  = iterator[1:]-iterator[0:-1]
    spaced = np.reshape(spaced,(spaced.shape[0],1)) #reshape such that input for set is correct   
    return len(set(spaced[:,0])) <= 1 #set builds an unordered collection of unique elements.

def check_time_series(data, timeindex=None):
    ## Dummy function.
    """Check if the timeseries are in the suitable format (pd DataFrame with only numeric values). Function also works for non-numeric data, because I did not check for that. 
    
    :param timeindex: the timeindex of the data
    :param data: the data, can be univariate or multivariate, as Pandas DataFrame
    
    """ 
    if isinstance(data, np.ndarray):
        timeseries = pd.DataFrame(data=data)
    elif isinstance(data, pd.DataFrame):
        timeseries = data
    else:
        raise ValueError("data should be numpy array or a pandas dataframe")
    
    if (timeindex is not None) and (isinstance(timeindex,np.ndarray)):    
        evenly = checkSpacing(timeindex)
        
        if evenly == False:
            raise ValueError("time index is not evenly spaced.")
            
        if timeseries.shape[0] == timeindex.shape[0]:
            print("right format for analysis")
            timeseries.setindex(timeindex)
        else:
            raise ValueError("timeindex and data do not have the same length")
    elif timeindex is not None:
        raise ValueError("timeindex should be numpy array")   
            
    return timeseries

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
            
        trend = loess(time_index, ts, degree, span)
        resid = ts - trend
        
    elif detrending == 'first_diff':
        
        resid = np.diff(ts, n=1, axis=0)
        time_index_diff = time_index[0:(len(time_index) - 1)]
        
    if detrending == 'first_diff':
        return resid, time_index_diff
    else:
        return trend, resid

def loess(x, y, degree, span):
    
    """Local polynomial regression.
    
    Uses weighting of data points after R function loess. 
    
    :param x: times series indices.
    :param y: time series.
    :param degree: degree of polynomial.
    :param span: window size in fraction of time series length.
    :return: trend.
    
    Created by Arie Staal
    """
    
    no_points = int(np.round(span*len(y)))
    half_no_points = int(np.round(0.5*span*len(y)))
    
    maxdist = 0.5*span*len(y)
    
    p = np.empty(np.shape(y))
    
    for i in range(0,len(y)):
        
        if i < half_no_points:
            x_span = x[0:no_points]
            y_span = y[0:no_points]
        
        if (i >= half_no_points) & (i <= len(y) - half_no_points):
            x_span = x[i - half_no_points : i + half_no_points]
            y_span = y[i - half_no_points : i + half_no_points]
            
        if i > (len(y) - half_no_points):
            x_span = x[len(y)-no_points+1:]
            y_span = y[len(y)-no_points+1:]
        
        wi = np.empty(np.shape(y_span))
        
        cnt = 0
        for x_i in x_span:
            dist = np.absolute(x[i] - x_i) / (np.max(x) - np.min(x))
            w_i[cnt] = (1 - (dist/maxdist)**3)**3
            cnt = cnt + 1
            
        fit = np.poly1d(np.polyfit(x_span, y_span, deg=degree, w=w_i))
        p[i] = fit(i)
        
    return p

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