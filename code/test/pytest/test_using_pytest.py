from pytest import approx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest as pt

def test_logtransform():

    from earlywarningsignals import logtransform

    # Test 1. Trying a known value

    # Input and expected output
    x_in = pd.DataFrame(data=[0])
    y_computed = logtransform(x_in)
    y_expected = pd.DataFrame(data=[0.0])

    # Check it
    assert (y_computed.equals(y_expected))

    # Test 2. Trying a known value

    # Input and expected output
    x_in = pd.DataFrame(data=[1.0])
    y_computed = logtransform(x_in)
    y_expected = pd.DataFrame(data=[0.69314718055994529])

    # Check it
    assert (y_computed.equals(y_expected))

def test_EWS():
    from earlywarningsignals import EWS

    # Test 1. Trying a known value
    np_input_ts=np.array([[1,6],[3,5],[4,4],[6,1]])
    input_ts=pd.DataFrame(data=np_input_ts)
    output_autocorrelation=[0.92857142857142838, 0.96076892283052284]
    output_variance=[3.25, 3.5]
    output_skewness=[0.0, -0.6872431934890912]
    output_CV=[0.51507875363771272, 0.46770717334674267]
    output_kurtosis=[-1.1479289940828403, -1.0]
    result=EWS(input_ts,autocorrelation=True,variance=True,skewness=True,
               kurtosis=True,CV=True)
    assert(np.all(result['autocorrelation'] == output_autocorrelation))
    assert(np.all(result['variance'] == output_variance))
    assert(np.all(result['skewness'] == output_skewness))
    assert(np.all(result['kurtosis'] == output_kurtosis))
    assert(np.all(result['CV'] == output_CV))

def test_isUniformSpacing():
    from earlywarningsignals import isUniformSpacing

    input_good = np.arange(10) # 0, 1, ..., 10
    input_bad = np.arange(10) # 0, 1, 9, 3, ..., 10
    input_bad[3]=9

    assert(isUniformSpacing(input_good)) # Expected True
    assert(~isUniformSpacing(input_bad)) # Expected not True

def test_check_timeseries_vector():
    from earlywarningsignals import check_time_series

    N = 10
    input_vector = np.arange(N)
    [ts, indices] = check_time_series(input_vector)

    # The indices are correct
    assert(np.min(indices) == 0)
    assert(np.max(indices) == N-1)
    # The number of elements is correct
    assert(np.size(indices) == N)
    assert(np.size(ts) == N)

    # The output is a dataframe
    assert(isinstance(ts, pd.DataFrame))

    # The output is a dataframe
    assert(isinstance(ts, pd.DataFrame))

def test_check_timeseries_array():
    from earlywarningsignals import check_time_series

    input_array = np.array([[1,6],[3,5],[4,4],[6,1]])
    [ts, indices] = check_time_series(input_array)

    # The indices are correct
    assert(np.min(indices) == 0)
    assert(np.max(indices) == 4-1)

    # The number of elements is correct
    assert(np.size(indices) == 4)

    # The output is a dataframe
    assert(isinstance(ts, pd.DataFrame))

def test_check_timeseries_dataframe():
    from earlywarningsignals import check_time_series

    N = 10
    input_vector = np.arange(N)
    input_df = pd.DataFrame(input_vector)
    [ts, indices] = check_time_series(input_df)

    # The indices are correct
    assert(np.min(indices) == 0)
    assert(np.max(indices) == N-1)
    # The number of elements is correct
    assert(np.size(indices) == N)
    assert(np.size(ts) == N)

def test_check_timeseries_customtimes():
    from earlywarningsignals import check_time_series

    N = 10
    input_vector = np.arange(N)
    input_times = np.linspace(0, 0.9, N) # 0, 0.1, 0.2, ..., 0.9
    input_df = pd.DataFrame(input_vector, input_times)
    [ts, indices] = check_time_series(input_df)

    # The indices are correct
    assert(indices.all() == input_times.all())

    # The number of elements is correct
    assert(np.size(indices) == N)
    assert(np.size(ts) == N)

def test_timeseries():
    from earlywarningsignals import check_time_series

    # Test 1: Trying a known value
    input_1a = np.arange(10)*2
    input_1b = np.arange(10)
    input_2a = np.array([[1,6],[3,5],[4,4],[6,1]])
    input_2aa = np.array([['a','b'],['b','x'],['l','i'],['g','z']])
    input_2b = np.arange(4)
    input_2bb = np.random.randint(low=0, high=14, size=(4,1))
    input_3a = pd.DataFrame(input_1a)
    input_3b = pd.DataFrame(input_1b)
    input_4a = pd.DataFrame(input_2a)
    input_4b = pd.DataFrame(input_2b)
    [output_1a, output_1b] = check_time_series(input_1a)
    #still have to write asserts
    assert(isinstance(output_1a, pd.DataFrame))

    #Check if evenly spaced check works
    with pt.raises(ValueError):
        check_time_series(input_2a, input_2bb)

    #Check if timeseries and timeindex have same length
    with pt.raises(ValueError):
        check_time_series(input_3a, input_4b)

def test_kendalltrend():
    from earlywarningsignals import kendalltrend
    x = np.linspace(0, 100, 101)
    in_ts_trend = np.exp(x*0.1) #test of timeseries with trend
    in_ts_notrend = np.random.normal(0, 1, 101) #test of timeseries with trend
    result_tr = kendalltrend(in_ts_trend)
    result_ntr = kendalltrend(in_ts_notrend)

    assert(result_tr[1] < 0.01 and result_ntr[1] > 0.01)

def test_interp():
    import pandas as pd
    from earlywarningsignals import interp

    # Generate the data
    fun = lambda x: np.cos(-x**2/9.0)
    x = np.linspace(0, 10, 25)
    y = fun(x)

    # Write in the form of a data frame
    df = pd.DataFrame(y, index = x)
    
    # Interpolate
    x_n = np.linspace(0, 10, 100)
    df_n = interp(df, x_n, method = 'cubic')

    # Extract interpolated ys
    y_n = df_n.values.T
    
    # Check equality under tolerance
    absTol = 0.05
    absErrs = np.abs(y_n - fun(x_n))
    assert(absErrs.max() < absTol)
    
def test_detrend():
    from earlywarningsignals import detrend
    from pandas.util.testing import assert_frame_equal
    
    # Test 1: linear trend
    
    #import data generated with R Toolbox
    in_ts = pd.read_csv("ts_test_detrend/x1.txt", sep=" ", header=None) 
    in_ts.columns = ["original"]
    in_ts.index=np.arange(1,10002)
    in_ts1= in_ts[0:101]
    
    detrend_methods= ['gaussian', 'loess', 'linear']
    
    for method in detrend_methods:  
        out_detrend = pd.read_csv('ts_test_detrend/x1_' + method +'.txt', sep='\t')
        out_detrend=out_detrend.drop(['timeindex'], axis=1)
        out_detrend=out_detrend.join(in_ts1)
        out_detrend.head()
    
        # Test function
        ts= pd.DataFrame({'ts1': out_detrend['original'], 'ts2':out_detrend['original']})
        output_detrend= pd.DataFrame({'ts1': out_detrend['smoothed'], 'ts2':out_detrend['smoothed']})  
        ts_trend, ts_resid = detrend(ts, detrending= method)
        
        # test whether R output is same as Python output
        print(method)
        assert_frame_equal(ts_trend, output_detrend, check_less_precise=2)
        
        # test whether ts - ts_trend = ts_resid
        assert_frame_equal(ts - ts_trend, ts_resid)
    

