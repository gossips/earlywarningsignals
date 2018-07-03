# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:56:41 2018

@author: Weina005
"""
import earlywarningsignals
import pandas as pd
import numpy as np


df = pd.DataFrame(np.random.randint(low=0, high=10, size=(7, 5)), columns=['a', 'b', 'c', 'd', 'e'])

earlywarningsignals.check_time_series(df) #ok
earlywarningsignals.logtransform(df) #ok
earlywarningsignals.detrend(df)
earlywarningsignals.EWS(df,autocorrelation=True) #ok
earlywarningsignals.apply_rolling_window(df)
