from pytest import approx
import numpy as np

def test_dummy_pass():
    assert True

def test_dummy_fail():
    assert False
    
def test_logtransform():
    
    from earlywarningsignals import logtransform
    
    # Test 1. Trying a known value
    
    # Input and expected output
    x_in = 0.0
    y_expected = 0.0
    
    # Check it
    assert (logtransform(x_in) == y_expected)
    
    # Test 2. Trying a known value
    
    # Input and expected output
    x_in = 1.0    
    y_expected = 0.69314718055994529
    
    # Check it
    assert (logtransform(x_in) == y_expected)
    
def test_EWS():
    from earlywarningsignals import EWS
    
    # Test 1. Trying a known value
    input_ts=np.array([[1,6],[3,5],[4,4],[6,1]])
    output_autocorrelation=[0.92857142857142838, 0.96076892283052284]
    output_variance=[3.25, 3.5]
    output_skewness=[0.0, -0.6872431934890912]
    output_CV=[0.51507875363771272, 0.46770717334674267]
    output_kurtosis=[-1.1479289940828403, -1.0]
    result=EWS(input_ts,autocorrelation=True,variance=True,skewness=True,
               kurtosis=True,CV=True)
    assert(result['autocorrelation'] == output_autocorrelation)
    assert(result['variance'] == output_variance)
    assert(result['skewness'] == output_skewness)
    assert(result['kurtosis'] == output_kurtosis)
    assert(result['CV'] == output_CV)