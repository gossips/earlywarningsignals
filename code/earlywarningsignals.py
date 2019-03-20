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
            evenly = isUniformSpacing(timeindex)
            if evenly == False:
                raise ValueError("time index is not evenly spaced")
        if timeseries.shape[0] == timeindex.shape[0]:
            print("right format for analysis")
        else:
            raise ValueError("timeindex and data do not have the same length")

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

def detrend(ts, detrending='gaussian', bandwidth=None, span=None, degree=None):
    
    """Detrend time series.
    
    :param ts: data frame with time indices as index and time series values in the columns.
    :param detrending: either
        'gaussian' = Gaussian detrending
        'linear' = linear regression
        'loess' = local nonlinear regression
    :param bandwidth: bandwidth for Gaussian detrending. If None, chooses default bandwidth (using Silverman's rule of thumb).
    :param span: window size in case of loess, in percentage of time series length. If None, chooses default span (25%).
    :param degree: degree of polynomial in case of loess. If None, chooses default degree of 2.
    :return: trend and residuals. In case of first_diff, returns residuals and difference between consecutive time values.
    
    Created by Arie Staal
    """
    
    ts_trend = pd.DataFrame().reindex_like(ts)
    ts_residual = pd.DataFrame().reindex_like(ts)
    
    for column in ts:
    
        if detrending == 'gaussian':
        
            if bandwidth == None:
                # Silverman's rule of thumb
                bw = round(ts.shape[0] * 0.9 * min(ts[column].std(), (ts[column].quantile(0.75) - ts[column].quantile(0.25)) / 1.34) * ts.shape[0]**(-0.2))
            else:
                bw = round(ts.shape[0] * bandwidth/100)
            
            trend = gaussian_filter1d(ts[column], bw, axis=0) # smY in R code
            resid = ts[column] - trend                        # nsmY in R code
        
        elif detrending == 'linear': 
        
            x = np.linspace(0, ts.shape[0], ts.shape[0])
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,ts[column])
            trend = intercept + slope * x
            resid = ts[column] - trend
        
        elif detrending == 'loess':
        
            if span == None:
                span_i = 25/100
            else:
                span_i = span/100
            
            if degree == None:
                degree = 2
            else:
                degree = degree
            
            trend = loess(ts[column], degree=degree, span=span_i)
            resid = ts[column] - trend
        
        # first difference is left out compared to the original R code
        
        ts_residual[column] = resid
        ts_trend[column] = trend
    
    return ts_trend, ts_residual


def loess(y, degree=None, span=None):
    
    """Local polynomial regression.
    
    Uses weighting of data points after R function loess. 
    
    :param y: time series.
    :param degree: degree of polynomial.
    :param span: window size in fraction of time series length.
    :return: trend.
    
    Created by Arie Staal
    """
    
    x = y.index.values
    
    no_points = int(np.round(span*len(y)))
    half_no_points = int(np.round(0.5*span*len(y)))
    
    maxdist = 0.5*span*len(y)
    
    p = np.empty(np.shape(y))
    
    for i in range(0, len(y)):

        
        if i < half_no_points:
            x_span = x[0:no_points]
            y_span = y[0:no_points]
        
        if (i >= half_no_points) & (i <= (len(y) - half_no_points)):
            x_span = x[i - half_no_points : i + half_no_points]
            y_span = y[i - half_no_points : i + half_no_points]
            
        if i > (len(y) - half_no_points):
            x_span = x[len(y)-no_points+1:]
            y_span = y[len(y)-no_points+1:]
        
        w_i = np.empty(np.shape(y_span))
        
        cnt = 0
        for x_i in x_span:
            dist = np.absolute(x[i] - x_i) / (np.max(x) - np.min(x))
            w_i[cnt] = (1 - (dist/maxdist)**3)**3
            cnt = cnt + 1
        
        if len(x_span) <= degree:
            raise TypeError("Window length should be higher than the degree of the fitted polynomial.")
        
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
                
    if autocorrelation == True:
        plt.plot(result['autocorrelation'])
        plt.title('autocorrelation')
        plt.show()
    if variance == True:
        plt.plot(result['variance'])
        plt.show()
    if skewness == True:
        plt.plot(result['skewness'])
        plt.show()
    if kurtosis == True:
        plt.plot(result['kurtosis'])
        plt.show()
    if CV == True:
        plt.plot(result['CV'])
        plt.show()
        
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

    :ts: Dataframe of timeseries for the indicator
    :return: Kendall tau value and p_value per column
    Created by M Usman Mirza

    """
    k_tau = []
    for y in ts:
        ti = range(len(df[y]))
        tau, p_value = scipy.stats.kendalltau(ti, df[y])
        k_tau.append([y, tau, p_value])
    return k_tau
        

#Interpolation function
def interp(df, new_x, method = 'linear', spline = False, k = 3, s = 0, der = 0):
    """Function interpolates timeseries with different methods.
        :param df: Dataframe of timeseries with index. Required value.
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
        :rtype: Dataframe of interpolated values
        Created by M Usman Mirza
    """
    df_x = df.index.tolist()
    df_new = pd.DataFrame(columns = df.columns, index = new_x)
    for y in df:
        if spline == False:
            f = interpolate.interp1d(x = df_x, y = df[y].tolist(), kind = method)
            i = f(new_x)
            df_new[y] = i
            
        elif spline == True:
            f = interpolate.splrep(x = x, y = df[y].tolist(), k = k, s = s)
            i = interpolate.splev(x = new_x, tck = f, der = der)
            df_new[y] = i
    return df_new

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
