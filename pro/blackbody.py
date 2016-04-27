import numpy as np
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt

'''
       The reference spectrum will be Vega, as taken from a table of values 
       and formatted in an array for this program.
'''
h = const.h.cgs.value
c = const.c.cgs.value
k = const.k_B.cgs.value
bandpasses = {'U':(3000,4500,3656), 'B':(3500,6000,4353), 'V':(4500,7500,5477), 
           'R':(5000,9500,6349),'I':(6500,12500,8797)}

def planck(teff,refflux,mag,band):
    '''
       Inputs:  Target's effective temperature and magnitude at a specific wavelength, the 
                desired bandpass, and a reference flux.
       Ouputs:  Array of wavelengths within the bandpass and the corresponding fluxes.
    '''
    abscenter = refflux*10**(-0.4*mag)
    lolim,hilim,center = bandpasses[band]
    wave = np.arange(lolim,hilim)
    modwave = wave*1e-8
    unflux = (2.0*h*c**2/modwave**5) * (np.exp(h*c/(modwave*k*teff))-1.0)**(-1)

    pairs = dict(zip(wave,unflux))                # combines two lists into dictionary
    uncenter = pairs[center]                      # unnormalized flux at band's central wavelength
    N = abscenter/uncenter                        # normalization constant
    normflux = N*unflux                           # normalized flux curve
    return wave,normflux

def graph():
    x,y = planck(5000,3.6e-9,0,'U')
    plt.plot(x,y)
    plt.show()


graph()
