#!/bin/python3

import scipy.stats
import numpy as np

## Description of the collection of functions

def check_time_series(input):
    ## Dummy function.
    return 'Hello world!'

def logtransform(ts):
    
    """Compute the logtransform of the original timeseries
    
    :param ts: original timeseries with variables in rows
    :return: logtransform of ts 

    
    Created by Ingrid van de Leemput
    """
    ts_log = np.log(ts+1)
    return ts_log

def apply_rolling_window(ts,winsize=50):
    
    """Re-arrange time series for rolling window
    
    :param ts: original timeseries (one-dimensional!!)
    :param winsize:  the size of the rolling window expressed as percentage of the timeseries length (must be numeric between 0 and 100). Default is 50\%.
    :return: matrix of different windows

    !! This function can only handle one-dimensional timeseries, and we don't check for that (yet)
    Created by Ingrid van de Leemput
    """
    
    # WE SHOULD CHECK HERE THAT ts is one-dimensional.. 
    
    mw=round(ts.size * winsize/100) # length moving window
    omw=ts.size-mw+1 # number of moving windows
    nMR = np.empty(shape=(omw,mw))
    nMR[:] = np.nan
    
    #not needed in this function: 
    low=2 
    high=mw 
    x = range(1,mw) 
    
    for i in range(0,omw):
        nMR[i,:]=ts[i:i+mw]  
    return nMR

def kendalltau(indicatorvec):
    ## Kendall trend statistic
    timevec = range(len(indicatorvec))
    tau, p_value = scipy.stats.kendalltau(timevec,indicatorvec)
    return [tau, p_value]

# temporary input for Kendall trend statistic (remove when other functions are ready)
nARR = [2,3,4,6,7,8,5,9,10,20]
nACF = [2,3,4,6,7,8,5,9,10,20]
nSD = [2,3,4,6,7,8,5,9,10,20]
nSK = [2,3,4,6,7,8,5,9,10,20]
nKURT = [2,3,4,6,7,8,5,9,10,20]
nDENSITYRATIO = [2,3,4,6,7,8,5,9,10,20]
nRETURNRATE = [2,3,4,6,7,8,5,9,10,20]
nCV = [2,3,4,6,7,8,5,9,10,20]

# Estimate Kendall trend statistic for indicators (ouput: [Tau,p_value])
KtAR=kendalltau(nARR)
KtACF=kendalltau(nACF)
KtSD=kendalltau(nSD)
KtSK=kendalltau(nSK)
KtKU=kendalltau(nKURT)
KtDENSITYRATIO=kendalltau(nDENSITYRATIO)
KtRETURNRATE=kendalltau(nRETURNRATE)
KtCV=kendalltau(nCV)

#print Kendall output (to be removed later?)
print('\nKtAR (Tau,p_value): %.4f, %.4f' % (KtAR[0],KtAR[1]))
print('\nKtACF (Tau,p_value): %.4f, %.4f' % (KtACF[0],KtACF[1]))
print('\nKtSD (Tau,p_value): %.4f, %.4f' % (KtSD[0],KtSD[1]))
print('\nKtSK (Tau,p_value): %.4f, %.4f' % (KtSK[0],KtSK[1]))
print('\nKtKU (Tau,p_value): %.4f, %.4f' % (KtKU[0],KtKU[1]))
print('\nKtDENSITYRATIO (Tau,p_value): %.4f, %.4f' % (KtDENSITYRATIO[0],KtDENSITYRATIO[1]))
print('\nKtRETURNRATE (Tau,p_value): %.4f, %.4f' % (KtRETURNRATE[0],KtRETURNRATE[1]))
print('\nKtCV (Tau,p_value): %.4f, %.4f' % (KtCV[0],KtCV[1]))

