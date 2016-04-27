import numpy as np

# Module to calculate q (system efficiency)

# Open files from which to read in system data
r = open('reflect.txt')
waveR,reflect = np.loadtxt(r, unpack=True)
f = open('filter.txt')
waveF,filt = np.loadtxt(f,unpack=True)
qe = open('qeff.txt')
waveQE,qe = np.loadtxt(f,unpack=True)


def mirrorThru(waveR,reflect):
    '''
    Calculates reflectivity of mirrors as a function of wavelength.
    Returns: Transmission of mirrors
    '''
    mirrors = (interpolate(waveR,reflect))**3.0
    # To the third power - for each of the three mirrors on the 3.5m
    return mirrors

def filThru(waveF,filt):
    '''
    Returns: filter transmission as a function of wavelength.
    '''
    filt = interpolate(waveF,filt)
    return filt

def detecThru(waveQE,qe):
    '''
    Calculates detector transmission as a function of wavelength.
    Takes into consideration: Quantum efficiency, readout noise,
    electron gain, plate scale, seeing.
    '''
    plate = 0.228  # Plate scale for ARCTIC (arcsec/pix) (2x2 binning)
    gain = 2.00    # electrons/DN
    rn = 3.7       # electrons
    qeff = interpolate(waveQE,qe)
    detec = ???
    return detec

def calc_q(mirrors,filt,detec):
    '''
    Multiplies all factors of system efficiency.
    Returns: q as a function of wavelength.
    '''
    net_q = mirrors*filt*detec
    return net_q

