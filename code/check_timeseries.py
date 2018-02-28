# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:37:05 2018

@author: Bregje van der Bolt

"""

def checkSpacing(iterator):
    return len(set(iterator)) <= 1 #set builds an unordered collection of unique elements.
    
    
def check_timeseries(data, timeindex=None):
    """Check if the timeseries are in the suitable format (np array with only numeric values).
    
    :param timeindex: the timeindex of the data
    :param data: the data, can be univariate or multivariate
    
    """
    import numpy as np
    
    timeseries = np.asarray(data)
    
    if timeindex == None:
         timeindex = np.linspace(0,data.shape[0]-1, data.shape[0])
    else:
         timeindex = np.asarray(timeindex)
         spaced = timeindex[1:]-timeindex[0:-1]
         evenly = checkSpacing(spaced)
         if evenly == False:
             print("time index is not evenly spaced.")
         
     

    if data.shape[0] == timeindex.shape[0]:
        Y = timeseries
        print("right format for analysis")
    else:
        print("timeindex and data do not have the same length")
        
    return Y, timeindex



    

                
