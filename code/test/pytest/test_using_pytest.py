from pytest import approx
import numpy as np
import matplotlib.pyplot as plt

def test_dummy_pass():
    assert True

def test_dummy_fail():
    assert False
    
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
    
def test_spaced():
    from earlywarningsignals import checkSpacing
    
    # Test 1: Trying a known value
    input_1 = np.arange(10)
    input_2 = np.arange(10)
    input_2[3]=9
    output_1= True
    output_2= False
    
    assert(checkSpacing(input_1) == output_1)
    assert(checkSpacing(input_2) == output_2)
    
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
    
    #check if evenly spaced check works
    with raises(ValueError):
        check_time_series(input_2a, input_2bb)
        
    #check if check if timeseries and timeindex have same length works
    with raises(ValueError):
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
    from earlywarningsignals import interp
    x = np.linspace(0, 10, 11)
    x_n = np.linspace(0, 10, 100)
    y = np.cos(-x**2/9.0)
    z_linear = interp(x, y, x_n, method = 'linear')
    z_quadratic = interp(x, y, x_n, method = 'cubic')
    plt.figure(1)
    plt.plot(x, y, 'o-')
    plt.plot(x_n, z_linear, 'o-')
    plt.figure(2)
    plt.plot(x, y, 'o-')
    plt.plot(x_n, z_quadratic, 'o-')

    
    
    
    
    
    