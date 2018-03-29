# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:37:05 2018

@author: Bregje van der Bolt

"""
import numpy as np
import pandas as pd

## Description of the collection of functions

def checkSpacing(iterator):
    iterator=np.asarray(iterator)
    return len(set(iterator)) <= 1 #set builds an unordered collection of unique elements.

def check_timeseries(data, timeindex=None):
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
            spaced = timeindex[1:]-timeindex[0:-1]
            evenly = checkSpacing(spaced)
            if evenly == False:
                print("time index is not evenly spaced.")
        if timeseries.shape[0] == timeindex.shape[0]:
            print("right format for analysis")
        else:
            print("timeindex and data do not have the same length")
    return timeseries, timeindex


                
#df = pd.DataFrame(np.random.randint(low=0, high=10, size=(7, 5)), columns=['a', 'b', 'c', 'd', 'e'])
#df2 = pd.DataFrame(np.random.randint(low=0, high=14, size=(14,1)))
#timeindex1 = pd.DataFrame(np.random.randint(low=0, high=14, size=(13,1)))
#timeindex2=np.random.randint(low=0, high=14, size=(14,1))
#[Y,timeindex4]=check_timeseries(df)
#[y2, timeindex5]=check_timeseries(df2, timeindex1)
#[y3, timeindex6]=check_timeseries(df2, timeindex2)