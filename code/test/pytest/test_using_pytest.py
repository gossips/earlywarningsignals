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