    #!/bin/python3

import scipy.stats
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy import spatial, optimize, interpolate
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
        kurtosis_ews=False, CV=False,plots=True):

    """Function that calculates early warning signals

    :param timeseries: Original timeseries (column for every variable)
    :param window_size: Set to size of rolling window, default setting does not use a rolling window
    :param autocorrelation: Set to True if autocorrelation is required in output
    :param variance: Set to True if variance is required in output
    :param skewness: Set to True if skewness is required in output
    :param kurtosis: Set to True if kurtosis is required (Fishers definition used,
    so it is zero for a normal distribution)
    :param CV: Set to True if coefficient of variation is required
    :return: pandas dataframe with variable, and the chosen early warning indicators for every window
    """

    nr_vars=ts.shape[1]

    if window_size == None:
        window_size = len(ts.iloc[:,0])

    # The functions that calculate the EWS
    a=lambda x: np.corrcoef(x[1:],x[:-1])[1,0]
    b=lambda x: np.var(x)
    c=lambda x: scipy.stats.skew(x[:])
    d=lambda x: scipy.stats.kurtosis(x[:])
    e=lambda x: np.std(x[:])/np.mean(x[:])

    functions=[a,b,c,d,e]
    indicators=[autocorrelation,variance,skewness,kurtosis_ews,CV]
    strindicators=['autocorrelation','variance','skewness','kurtosis_ews','CV']
    idx=np.where(indicators)[0]
    strindicators=[strindicators[i] for i in idx] #Only calculate the selectd indicators

    for n in range(nr_vars):
        df1=ts.iloc[:,n]
        res=df1.rolling(window_size).agg([functions[i] for i in idx])
        res.columns=[strindicators]
        res.insert(0,'variable',n)
        if n == 0:
            result = res
        else:
            result=pd.concat([result,res],axis=0)
    if plots == True:
        for i in strindicators:
            result.groupby('variable')[i].plot(legend=True,title=i)
            plt.show()
    return result



def kendalltrend(ts):
    """Calculates the Kendall trend statistic for a EW indicators

    :ts: Dataframe of timeseries for the indicator
    :return: Kendall tau value and p_value per column
    """
    ti = range(len(ts))
    tau, p_value = scipy.stats.kendalltau(ti,ts)
    return [tau, p_value]



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


def krig(df, buf, new_index, N):
    """Interpolates data using Kriging.
    :param df: Dataframe to interpolate with time as the index.
    :param buf: The bandwidth for the empirical variogram
    :param new_index: The new time index on which to get interpolated points
    :param N: Number of nearest points to consider for interpolation.
    :return point estimates, lags, emprircal variogram, model variogram

    """
    #empirical variogram
    def vario(df, buf):
        fr_i = 0
        to_i = ptp(array(df.index))
        lags = arange(fr_i, to_i, buf)
        df_l = df.shape[0]
        dist = spatial.distance.pdist(array(df.index).reshape(df_l, 1))
        sq_dist = spatial.distance.squareform(dist)
        sv_lag = []
        sv = []
        for k in lags:
            for i in range(df_l):
                for j in range(i + 1, df_l):
                    if (sq_dist[i, j] >= k - buf) and (sq_dist[i, j] <= k + buf):
                        sv_lag.append((df.iloc[i] - df.iloc[j])**2)
            sv.append(sum(sv_lag)/(2*len(sv_lag)))
        return array(sv), lags

    #sill
    def c_f(df, lag, lag_i, sv):
        sill = var(df)
        if sv[lag] == 0:
            return sill
        return sill - sv[lag_i]

    #spherical variogram model
    def sph_m(lags, a, c, nugget):
        sph = []
        for i in range(lags.size):
            if lags[i] <= a:
                sph.append(c*( 1.5*lags[i]/a - 0.5*(lags[i]/a)**3.0) + nugget)
            if lags[i] > a:
                sph.append(c + nugget)
        return sph


    def vario_fit(df, buf):
        sv, lags = vario(df, buf) #empirical variogram
        c = c_f(df, lags[0], 0, sv) #sill - nugget
        nugget = sv[0]
        sill = var(df)
        #Fitting the variogram
        sph_par, sph_cv = optimize.curve_fit(sph_m, lags, sv, p0 = [int(lags.size/2), c, nugget])
        sv_model = sph_m(lags, sph_par[0], sph_par[1], sph_par[2])
        return lags, sv, sv_model, sph_par


    lags, sv, sv_model, sph_par = vario_fit(df, buf)
    mu = array(mean(df))
    coord_df = array([repeat(0, array(df.index).size), array(df.index)]).T
    coord_new_index = array([repeat(0, array(new_index).size), new_index]).T
    dist_mat = spatial.distance.cdist(coord_df, coord_new_index)
    dist_mat = c_[df.index, df, dist_mat]
    int_e = []
    for i in range(len(new_index)):
        dist_mat_1 = dist_mat[dist_mat[:,i+2].argsort()]
        dist_mat_1 = dist_mat_1[:N,:]
        k = sph_m(dist_mat_1[:,i+2], sph_par[0], sph_par[1], sph_par[2])
        k = matrix(k).T
        dist_mat_df = spatial.distance.squareform(spatial.distance.pdist(dist_mat_1[:,0].reshape(N,1)))
        K = array(sph_m(dist_mat_df.ravel(), sph_par[0], sph_par[1], sph_par[2])).reshape(N, N)
        K = matrix(K)
        weights = inv(K)*k
        resid = mat(dist_mat_1[:,1] - mu)
        int_e.append(resid*weights + mu)
    int_e = ravel(int_e)
    return int_e, lags, sv, sv_model
