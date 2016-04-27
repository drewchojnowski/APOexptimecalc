# *** 4 Feb 2016 edits to Caitlin's Exposure Time Calculator ***



import numpy as np
import math
from astropy.io import ascii

h = 6.62e-27
c = 3e10
waves=diam=mO=mB=pscale=zp=instr=filter=grating=airmass=moonph=seeing=None

# ** All variables need "reasonable" default values **
# TASKS
# 1) Required exposure time accurate to 10%
# 2) Signal-to-noise at specified exposure time
# 3) Calculated flux for specified exposure time
# Optional: 4) Graphical output + table ** Ask Jon **
# 5) Separate functions for all noise functions (Poisson from object, sky, readout noise)

# CODE SPECS
# 1) Flexible, easily transferable to other telescopes (only 'gratings' are not transferable)
# 2) Reasonable numbers come out!

def fileRead(filename):
    inputs = ascii.read(filename)
    global waves   ;  waves = inputs['lamb']
    global diam    ;  diam = inputs['diam']
    global mO      ;  mO = inputs['ob_mag']
    global mB      ;  mB = inputs['bg_mag']
    global pscale  ;  pscale = inputs['pscale']
    global zp      ;  zp = inputs['zp']
    global instr   ;  instr = inputs['instr']
    global filt    ;  filt = inputs['filt']
    global grating ;  grating = inputs['grating']       # Name of grating *Do dictionary
    global airmass ;  airmass = inputs['airmass']
    global moonph  ;  moonph = inputs['moonph']
    global seeing  ;  seeing = inputs['seeing']

    return calc_SN()

def calc_SN(width=2,height=2,sigRN=5):
    '''
        Inputs: Width and height of image in arcsec, 
                and detector readnoise.
    '''
    sourceS = calc_S(True)
    backgS = calc_S(False)
    area = width*height
    Npix = area/pscale**2
    SoverN = sourceS/np.sqrt(sourceS+backgS*area+Npix*sigRN)
    return SoverN

def calc_S(source):
    ''' 
        Inputs: Boolean to check if the signal is from the 
                target of interest.
        Outputs: Total signal.
    '''
    T = calc_T()
    q = calc_q()
    a = calc_atmos()
    t = expTime()
    P = calc_P(source)
    S = T*q*a*P*t
    return S

def calc_P(source):
    '''
        Outputs: Signal per unit 1 sec.
    '''
    E = (h*c)/(waves)
    F = calc_F(source)
    P = F/E
    return P

def calc_F(source):
    '''
        Outputs: Flux in erg/cm2/s/A using optical Vega as 
                 the reference.
    '''
    Fref = 3.6e-9*5500**2*waves**(-2)
    M = None
    if source:
        M = mO
    else:
        M = mB
    F = 10**(-M/2.5)*Fref
    return F

def expTime():
    time = np.empty(len(waves))
    time.fill(1.0)
    return time

def calc_T():
    '''
        Outputs: Aperture area in cm2.
    '''
    T = math.pi*(50*diam)**2
    return T

def calc_atmos():
    '''
        Atmospheric transmission, IP.
    '''
    atmos = np.empty(len(waves))
    atmos.fill(0.8)
    return atmos

def calc_q():
    net_q = np.empty(len(waves))
    net_q.fill(0.5)
    return net_q

def telThru():
    ins = np.empty(len(waves))
    ins.fill(1.0)
    return tel

def insThru():
    ins = np.empty(len(waves))
    ins.fill(1.0)
    return ins

def filThru():
    fil = np.empty(len(waves))
    fil.fill(1.0)
    return fil

def effic():
    eff = np.empty(len(waves))
    eff.fill(0.5)
    return eff
