'''
Programmer:     Laurel Farris
Last updated:   4/21/16
Description:    Calculate Atmospheric transmission, a
                (extinction, refraction, etc.)
'''

import numpy as np
import math
from astropy import constants as const

''' Define constants '''
c = const.c.cgs.value
h = const.h.cgs.value

def atm_transmission(bandpass, airmass):
    '''
    Calculate the atmospheric transmission (a),
    '''

    ''' Open files '''
    f = open('../Input_files/extinction.txt')
    for line in f:
        ''' Disregard comments '''
        if not line.strip().startswith("#"):
            wavelength,extinction = np.loadtxt(f, unpack=True)
            break
    for i in range(0,len(wavelength)+1):
        if bandpass == wavelength[i]:
            k = extinction[i]
            break

    ''' Calculate and return atmospheric transmission '''
    return 10.**(k*airmass / -2.5)


a = atm_transmission(-3, 1.2)
print a
