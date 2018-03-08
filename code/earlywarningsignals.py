#!/bin/python3

import scipy.stats
import numpy as np
from scipy.ndimage import gaussian_filter1d

## Description of the collection of functions

def check_time_series(input):
    ## Dummy function.
    return 'Hello world!'

def logtransform(ts):
    
    """Compute the logtransform of the original timeseries
    
    :param ts: original timeseries with variables in rows
    :return: logtransform of ts 

    
    Created by Ingrid van de Leemput
    """
    ts_log = np.log(ts+1)
    return ts_log

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

def EWS(timeseries,autocorrelation=False,variance=False,skewness=False,
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
    
    one loop
    
    """
    
    nr_vars=len(timeseries[0,:])
    result={}
    
    if autocorrelation == True:
        AC=[0]*nr_vars
        for i in range(nr_vars):
            AC[i]=np.corrcoef(timeseries[1:,i],timeseries[:-1,i])[1,0]
            result.update({'autocorrelation' : AC})
            
    if variance == True:
        Var=[0]*nr_vars
        for i in range(nr_vars):
            Var[i]=np.var(timeseries[:,i])
            result.update({'variance' : Var})
            
    if skewness == True:
        Skews=[0]*nr_vars
        for i in range(nr_vars):
            Skews[i]=scipy.stats.skew(timeseries[:,i])
            result.update({'skewness' : Skews})
            
    if kurtosis == True:
        Kurt=[0]*nr_vars
        for i in range(nr_vars):
            Kurt[i]=scipy.stats.kurtosis(timeseries[:,i])
            result.update({'kurtosis' : Kurt})

    if CV == True:
        CVs=[0]*nr_vars
        for i in range(nr_vars):
            CVs[i]=np.std(timeseries[:,i])/np.mean(timeseries[:,i])
            result.update({'CV' : CVs})        
        
    return result

def apply_rolling_window(ts,winsize=50):
    
    """Re-arrange time series for rolling window
    
    :param ts: original timeseries (one-dimensional!!)
    :param winsize:  the size of the rolling window expressed as percentage of the timeseries length (must be numeric between 0 and 100). Default is 50\%.
    :return: matrix of different windows

    !! This function can only handle one-dimensional timeseries, and we don't check for that (yet)
    Created by Ingrid van de Leemput
    """
    
    # WE SHOULD CHECK HERE THAT ts is one-dimensional.. 
    
    mw=round(ts.size * winsize/100) # length moving window
    omw=ts.size-mw+1 # number of moving windows
    nMR = np.empty(shape=(omw,mw))
    nMR[:] = np.nan
    
    #not needed in this function: 
    low=2 
    high=mw 
    x = range(1,mw) 
    
    for i in range(0,omw):
        nMR[i,:]=ts[i:i+mw]  
    return nMR

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
def interpolate(x, y, new_x = None, dim = 1, method = 'linear', spline = False, k = 3, s = 0, der = 0):
    """
    Function interpolates data with in one or two dimension. Returns interpolated data.
    x: Original data point coordinates or time in case of time series. If dim = 2 then it should be a 2-dim array/float/tuple. Required value.
    y: Original data values. Must be the same dimension as x. If dim = 2 then it should a 2-dim array/float/tuple. Required value.
    new_x: Points at which to interpolate data. For 2-dim it should a grid. Required value.
    dim: Specifies dimension of data. Currently only for 1 or 2 dimensions. Default is 1
    method: Specifies interpolation method used. One of
    	‘nearest’: return the value at the data point closest to the point of interpolation.
        'linear’: interpolates linearly between data points on new data points
        'cubic’: Interpolated values determined from a cubic spline
        Default is ‘linear’
    spline: Spline interpolation. Can be True or False. If True then the function ignores the method call. Default is False. 
    k: Degree of the smoothing spline. Must be <= 5. Default is k=3, a cubic spline. 
    der: The order of derivative of the spline to compute (must be less than or equal to k)
    Created by M Usman Mirza
    """
    if dim == 1 & spline == False:
        f = interp1d(x = x, y = y, kind = method)
        i = f(new_x)
        return i
    elif dim == 2 & spline == False:
        i = griddata(points = x, values = y, xi = new_x, method = method)
        return i
    elif dim == 1 & spline == True:
        f = splrep(x = x, y = y, k = k, s = s)
        i = splev(x = new_x, tck = f, der = der)
        return i
    elif dim == 2 & spline == True:
        f = bisplrep(x = x[:,0], y = x[:,1], z = y, k = k, s = s)
        i = bisplev(x = new_x[:,0], y = new_x[0,:], tck = f)
        return i
    else:
        print('Dimension > 2 not supported')