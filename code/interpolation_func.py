# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:22:54 2018

@author: mirza009
"""

import numpy as np
from scipy.interpolate import *
import matplotlib.pyplot as plt
 
#Interpolation function
def interpolate(x, y, new_x = None, dim = 1, method = 'linear', spline = False, k = 3, s = 0, der = 0):
    """
    Function interpolates data with in one or two dimension. Returns interpolated data.
    x: Original data point coordinates or time in case of time series. If dim = 2 then it should be a 2-dim array/float/tuple. Required value.
    y: Original data values. Must be the same dimension as x. If dim = 2 then it should a 2-dim array/float/tuple. Required value.
    new_x: Points at which to interpolate data. For 2-dim it should a grid. Required value.
    dim: Specifies dimension of data. Currently only for 1 or 2 dimensions. Default is 1
    method: Specifies interpolation method used. One of
    	‘nearest’: return the value at the data point closest to the point of interpolation.
        'linear’: interpolates linearly between data points on new data points
        'cubic’: Interpolated values determined from a cubic spline
        Default is ‘linear’
    spline: Spline interpolation. Can be True or False. If True then the function ignores the method call. Default is False. 
    k: Degree of the smoothing spline. Must be <= 5. Default is k=3, a cubic spline. 
    der: The order of derivative of the spline to compute (must be less than or equal to k)
    Created by M Usman Mirza
    """
    if dim == 1 & spline == False:
        f = interp1d(x = x, y = y, kind = method)
        i = f(new_x)
        return i
    elif dim == 2 & spline == False:
        i = griddata(points = x, values = y, xi = new_x, method = method)
        return i
    elif dim == 1 & spline == True:
        f = splrep(x = x, y = y, k = k, s = s)
        i = splev(x = new_x, tck = f, der = der)
        return i
    elif dim == 2 & spline == True:
        f = bisplrep(x = x[:,0], y = x[:,1], z = y, k = k, s = s)
        i = bisplev(x = new_x[:,0], y = new_x[0,:], tck = f)
        return i
    else:
        print('Dimension > 2 not supported')

 
 
