'''
System effeciency, q
Telescope throughput
Instrument throughput
Filter throughput
Grating specification
'''

import numpy as np
import math
from astropy import constants as const

''' Define constants '''
c = const.c.cgs.value
h = const.h.cgs.value


def sys_transmission(wavelength):
    '''
    Calculate the system transmission (q), using the telescope
    throughput, filter throughput, and detector efficiency; all three of
    which are calculated within their own function embedded below.
    '''
    def telescope_throughput():
        return 0
    def filter_throughput():
        return 0
    def detector_efficiency():
        return 0
    return q


