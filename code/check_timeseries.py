# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:37:05 2018

@author: Bregje van der Bolt

"""
import numpy as np
import pandas as pd

def checkSpacing(iterator):
    return len(set(iterator)) <= 1 #set builds an unordered collection of unique elements.
    
    
def check_timeseries(data, timeindex=None):
    """Check if the timeseries are in the suitable format (pd DataFrame with only numeric values).
    
    :param timeindex: the timeindex of the data
    :param data: the data, can be univariate or multivariate
    
    """
    
    timeseries = pd.DataFrame(data=data)
    
    if timeindex == None:
         timeindex = np.linspace(0,timeseries.shape[0]-1, timeseries.shape[0])
    else:
         timeindex = np.asarray(timeindex)
         spaced = timeindex[1:]-timeindex[0:-1]
         evenly = checkSpacing(spaced)
         if evenly == False:
             print("time index is not evenly spaced.")
         
     

    if timeseries.shape[0] == timeindex.shape[0]:
        Y = timeseries
        print("right format for analysis")
    else:
        print("timeindex and data do not have the same length")
        
    return Y, timeindex



    

                
df = pd.DataFrame(np.random.randint(low=0, high=10, size=(7, 5)), columns=['a', 'b', 'c', 'd', 'e'])
[Y,timeindex]=check_timeseries(df)