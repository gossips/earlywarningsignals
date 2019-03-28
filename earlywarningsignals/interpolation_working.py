# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:22:54 2018

@author: mirza009
"""

import numpy as np
from scipy.interpolate import *
import matplotlib.pyplot as plt
%matplotlib inline 
%matplotlib qt #for polots in new window

 x = np.linspace(0, 10, 20)
 y = np.cos(x)*np.sin(x)
 plt.plot(x, y, '.')
 f = interp1d(x, y, kind = 'linear')
 f1 = interp1d(x, y, kind = 'cubic')
 plt.plot(x, y, '.', x1, f(x1), '--', x1, f1(x1), '-')
 
 #2D interpolation
 def func(x, y): return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
 grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
 points = np.random.rand(1000, 2)
 values = func(points[:,0], points[:,1])
 grid_l = griddata(points, values, (grid_x, grid_y), method='linear')
 grid_c = griddata(points, values, (grid_x, grid_y), method='cubic')
 
 plt.imshow(func(grid_x, grid_y), extent = (0, 1, 0 , 1), origin = 'lower')
 plt.plot(points[:,0], points[:,1], 'k.')
 plt.imshow(grid_l, extent = (0, 1, 0 , 1), origin = 'lower')
 plt.imshow(grid_c, extent = (0, 1, 0 , 1), origin = 'lower')

#Splines
 x = np.linspace(0, 2*np.pi+np.pi/4, 10)
 
#Interpolation function
def interpolate(x, y, new_x = None, dim = 1, method = 'linear', spline = False, k = 3, s = 0, der = 0):
    """
    Function interpolates data with in one or two dimension. Returns interpolated data.
    x: Original data point coordinates or time in case of time series
    y: Original data values. Must be the same dimension as x
    new_x: Points at which to interpolate data. Must be the same dimension as x
    dim: Specifies dimension of data. Currently only for 1 or 2 dimensions
    method: Specifies interpolation method used. One of
    	‘nearest’ return the value at the data point closest to the point of interpolation.
    	‘linear’ interpolates linearly between data points on new data points
    	‘cubic’ Interpolated values determined from a cubic spline
    Spline
    
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
        f = bisplrep(x = x[0], y = x[1], z = y, k = k, s = s)
        i = bisplev(x = new_x[:,0], y = new_x[0,:], tck = f)
        return i
    else:
        print('Dimension > 2 not supported')

 
 
 i = interpolate(x, y, x_i, 1, 'cubic')
 i = interpolate(points, values, (grid_x, grid_y), 2, 'cubic')
 i = interpolate(x = x, y = y, new_x = x_i, dim = 1, spline = True)
 x_i = np.linspace(0, 10, 100)
 plt.plot(x_i, i, '--')
 plt.imshow(i, extent = (0, 1, 0 , 1), origin = 'lower')
 ynew = splev(x_i, i, der=3)
 plt.plot(x_i, i, 'b')