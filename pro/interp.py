from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import numpy as np

def interp1(x, y, kind="cubic"):
    '''
    INPUT : X and Y need to be arrays of numbers of the same dimensions.
    You don't have to include a kind, but if you want a method other 
    than cubic, you can substitute it here.
    
    OUTPUT : An object of the interpolated data. You want to call this
    function by saying "test = interp1d(x, y)" , then "test(24.2)" 
    for example. 
    '''
    return interp1d(x, y, kind)

def interp2(x, y, val, kind="cubic"):
    '''
    INPUT : Same as above, but a third input array for the value at each point 
    is required. For example, for our Moon table, we have:
    (Moon Phase, Wavelength, Value)
    Make sure the value input array contains a point for every x,y combination.
    You can input as a 2D array for val, or as a flattened array with at least one
    point for every x,y.

    OUTPUT : Similar to above, but the call would be "test(.65, 24.2)" and
    produces a value interpolated between those points. 
    '''
    return interp2d(x, y, val, kind)


def integrate(inter_y, dx, xmin, xmax):
    '''
    INPUT : Give it an interpolated function, your step size, and your bounds.
    
    OUTPUT : The integrated value over your bounds.
    '''
    i = xmin
    summer = 0
    while i < xmax:
        summer = summer + (inter_y(i) * dx)
        i = i + dx
    return summer

'''
x = [0, 1, 2, 3, 4]
y = [0, 1, 2, 3, 4]
z = [[0, 1, 4, 9, 16], [0, 1, 4, 9, 16], [0, 1, 4, 9, 16], [0, 1, 4, 9, 16], [0, 1, 4, 9, 16]]

test1 = interp1(x, y)

print test1(2.5)

test2 = interp2(x, y, z)

print test2(3.5, 3.5)

print integrate(test1, 0.01, 0, 4)

OUTPUT FROM TEST:
2.5
[ 12.25]
8.02
'''
