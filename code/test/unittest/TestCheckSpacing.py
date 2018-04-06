# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:07:01 2018

@author: bolt008
"""

import unittest
from  earlywarningsignals import CheckSpacing
import numpy as np
import pandas as pd

class CheckSpacingTest(unittest.TestCase):
    """Tests for 'CheckSpacing.py'"""
    
    def test_is_evenly_even(self):
        """is  an array of 1-10 evenly spaced?"""
        input_1 = np.arange(10)
        spaced_1 = input_1[1:]-input_1[0:-1]
        self.assertTrue(CheckSpacing(spaced_1))
        
    def test_is_evenly_uneven(self):
        input_2 = np.random.randint(low=0, high=10, size=(10,1))
        spaced_2 =  input_2[1:]-input_2[0:-1]
        self.assertFalse(CheckSpacing(spaced_2))
        
if __name__ == '__main__':
    unittest.main()