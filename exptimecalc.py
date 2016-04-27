import numpy as np
import math
from astropy.io import ascii
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

# Vega spectrum from ftp://ftp.stsci.edu/cdbs/grid/k93models/standards/
vega_spec_file='data/vega_c95.txt'

# Aluminum reflectivity from 
# https://laserbeamproducts.wordpress.com/2014/06/19/reflectivity-of-aluminium-uv-visible-and-infrared/
al_reflect_file='data/al_reflectivity.txt'

# DIS info from http://www.apo.nmsu.edu/arc35m/Instruments/DIS/#7
dis_grating_file='data/dis_gratings.txt'

#bandpasses = {'U':(3000,4500,3656),'B':(3500,6000,4353),'V':(4500,7500,5477),'R':(5000,9500,6349),'I':(6500,12500,8797)}


#################################################################################################
'''
The main code
'''
def exptimecalc(instr=None,grating=None,mag=None,snr=100.0,teff=None,filt=None,wcent=6250.0,wspan=750.0,
                seeing=None,airmass=None,moonphase=None):

    # get wavelength limits based on center and span
    w1 = wcent-(wspan/2.); w2 = wcent+(wspan/2.)

    # get instrument parameters if not already supplied
    instr,gain,readnoise,platescale = get_instr_params(instr=instr,grating=grating)

    # get parameters specific to object & observing conditions
    obsinfo=get_obsinfo(mag=mag,teff=teff,seeing=seeing,airmass=airmass,moonphase=moonphase)
    mag=obsinfo[0]; teff=obsinfo[1]; seeing=obsinfo[2]; airmass=obsinfo[3]; moonphase=obsinfo[4]

    # get a flux-calibrated blackbody of target, based on V mag, teff, and vega spectrum
    wave,objflux = get_blackbody_flux(teff=teff,mag=mag,w1=w1,w2=w2,wcent=wcent)

    # get the throughput of 3 aluminum mirros
    mirror_throughput = get_mirror_throughput(w1=w1,w2=w2)

    exptime='???????'

    print('='*70)
    print('The exposure time needed for a S/N = '+str(snr)+' observation of a')
    print('V = '+str(mag)+', Teff = '+str(teff)+' K star observed at')
    print('airmass = '+str(airmass)+', during seeing = '+str(seeing)+'",')
    print(str(moonphase)+' days after new moon, using the '+instr)
    print('instrument on the APO 3.5m telescope is:')
    print('\n'+exptime+' seconds')

    return instr


#################################################################################################
'''
Exit code message:

During any of the user input, entering "q" will quit out of the code
with the following message.
'''
def exit_code():
    sys.exit('Exiting the APO exposure time calculator.')


#################################################################################################
'''
Get Instrument parameters (gain, readnoise, and platescale)
'''
def get_instr_params(instr=None,grating=None):
    # establish instrument if not specified
    print('='*70)
    print('* Which instrument are you using?')
    if instr is None:
        print(' 1. ARCTIC (default)')
        print(' 2. DIS')
        a = raw_input('')
        if a=='':  a='1'
        if a[:1]=='1': instr='arctic'
        if a[:1]=='2': instr='dis'
        if a=='q': exit_code()
    print('Ok, you are using '+instr+'.')

    # get arctic parameters - pretty simple
    if instr=='arctic':
        gain=2.0
        readnoise=3.7
        platescale=0.228 # (arcseconds/pixel)

    # get dis parameters - more complicated, in case other info from DIS grating table is needed.
    if instr=='dis':
        # read in table of DIS grating info
        distable = ascii.read(dis_grating_file)
        disgratings = np.array(distable['grating'])
        # establish grating if not specified
        print('='*70)
        print('* Which DIS grating are you using?')
        if grating is None:
            # list off all the gratings in the dis table
            for i in range(len(distable['grating'])):
                if i==0: 
                    print(' - enter '+str(i+1)+' for '+disgratings[i]+' (default)')
                else:
                    print(' - enter '+str(i+1)+' for '+disgratings[i])
            a = raw_input('')
            if a=='':  a='1'
            if a=='q': exit_code()
            grating=disgratings[int(a)]
        print('Ok, you are using '+grating+'.')
        

        # now that grating is established, look up info in table
        gd=np.where(grating==disgratings)
        if len(gd[0])==0: sys.exit('The specified grating does not exist.')
        gain=distable['gain'][gd]
        readnoise=distable['readnoise'][gd]
        platescale=distable['platescale'][gd]

    print('Ok. gain = '+str(gain)+', readnoise='+str(readnoise)+', platescale='+str(platescale)+'.')

    return instr,gain,readnoise,platescale


#################################################################################################
'''
Get observation parameters (Vmag, Teff, seeing, airmass, moonphase)
'''
def get_obsinfo(mag=None,teff=None,seeing=None,airmass=None,moonphase=None):
    # get the V mag... default is V=9
    print('='*70)
    print('* What is the V magnitude of the target? (default is 9.0)')
    if mag is None:
        mag=raw_input('')
        if mag=='q': exit_code()
        if mag=='':  
            mag=9.0
        else:
            mag=float(mag)
    print("Ok, your target has V = "+str(mag)+".")

    # get the effective temp... default is Teff=5000 K
    print('='*70)
    print('* What is the Teff of the object? (default is 5000)')
    if teff is None:
        teff=raw_input('')
        if teff=='q': exit_code()
        if teff=='':  
            teff=5000.0
        else:
            teff=float(teff)
    print("Ok, your target has Teff = "+str(teff)+".")

    # get the airmass... default is 1.2
    print('='*70)
    print('* What is the expected airmass? (default is 1.2)')
    if airmass is None:
        airmass=raw_input('')
        if airmass=='q': exit_code()
        if airmass=='':  
            airmass=1.2
        else:
            airmass=float(airmass)
    else:
        if airmass>4: print("Airmass = "+str(airmass)+"? That's a pretty high number...")
    print("Ok, your target will be observed at airmass = "+str(airmass)+".")

    # get the seeing... default is 1.0"
    if seeing is None:
        print('='*70)
        print('* What is the expected seeing? (default is 1.0")')
        seeing=raw_input('')
        if seeing=='q': exit_code()
        if seeing=='':  
            seeing=1.0
        else:
            seeing=float(seeing)
    else:
        if (seeing>3): print("Seeing = "+str(seeing)+"? That's a pretty high number...")
    print("Ok, your target will be observed while seeing = "+str(seeing)+".")

    # get the moonphase... default is 0.0
    if moonphase is None:
        print('='*70)
        print('* What is the moon phase? Enter a value between 0 and 14 days from new (default is 0):')
        moonphase=raw_input('')
        if moonphase=='q': exit_code()
        if moonphase=='':  
            moonphase=0
        else:
            moonphase=int(moonphase)
    else:
        if moonphase>14:
            print('Moonphase must be between 0-14. Please try again.')
            exit_code()
    print("Ok, your target will be observed "+str(moonphase)+" days after new moon.")
    
    return mag,teff,airmass,seeing,moonphase


#################################################################################################
'''
Blackbody calculator

   Inputs:  Target's effective temperature and magnitude at a specific wavelength, the 
            desired bandpass, and a reference flux.
   Ouputs:  Array of wavelengths within the bandpass and the corresponding fluxes.
'''
def get_blackbody_flux(teff=None,mag=None,filt=None,w1=None,w2=None,wcent=None):
    h = const.h.cgs.value
    c = const.c.cgs.value
    k = const.k_B.cgs.value

    # read in the vega spectrum
    vega = ascii.read(vega_spec_file)
    wvega = np.array(vega['wavelength'])
    fvega = np.array(vega['flux'])
    # get the vega flux at wcent
    gd = np.where((wvega>wcent-0.6) & (wvega<wcent+0.6))
    fvega_gd = fvega[gd][0]

    abscenter = fvega_gd*10.0**(-0.4*mag)

    wave = np.arange(w1,w2)
    modwave = wave*1e-8
    unflux = (2.0*h*c**2/modwave**5) * (np.exp(h*c/(modwave*k*teff))-1.0)**(-1)

    pairs = dict(zip(wave,unflux))                # combines two lists into dictionary
    uncenter = pairs[wcent]                      # unnormalized flux at band's central wavelength
    N = abscenter/uncenter                        # normalization constant
    objflux = N*unflux                           # normalized flux curve

    return wave,objflux


#################################################################################################
'''
Interpolator
'''
def interp(inx,iny):
    f=interp1d(inx,iny)
    outx=np.arange(np.min(inx),np.max(inx))
    outy=f(outx)

    return outx,outy

#################################################################################################
#################################################################################################
#################################################################################################
# everything below here has not been incorporated into the code yet
#################################################################################################
#################################################################################################
#################################################################################################


#################################################################################################
'''
Calculates reflectivity of mirrors as a function of wavelength.
Returns: Transmission of mirrors
'''

def get_mirror_throughput(w1=None,w2=None):
    # read in the file of aluminum reflectivity
    aldata = ascii.read(al_reflect_file)
    # get interpolated arrays of wavelength and reflectivity
    # reflectivity is given in percentage, so divide by 100
    alwave,alreflect = interp(aldata['wavelength'],aldata['reflectivity']/100.)

    # restrict to the specified wavelength limits
    gd=np.where((alwave>w1) & (alwave<w2))
    if len(gd[0])<2:
        print('There is an error in mirror relectivity calculation.')
        exit_code()
    mirror_throughput=np.mean(alreflect[gd])
    # To the third power - for each of the three mirrors on the 3.5m
    mirror_throughput=mirror_throughput**3.0

    return mirror_throughput


#################################################################################################
'''
Returns: filter transmission as a function of wavelength.
'''
def get_filter_throughput():
    filterdata = ascii.read(filterfile)
    filtwave,filtbla = interp(filterdata['wavelength'],filterdata['?'])

    return filter_throughput


#################################################################################################
'''
Calculates detector transmission as a function of wavelength.
Takes into consideration: Quantum efficiency, readout noise,
electron gain, plate scale, seeing.
'''
#def get_detector_throughput():

#    plate = 0.228  # Plate scale for ARCTIC (arcsec/pix) (2x2 binning)
#    gain = 2.00    # electrons/DN
#    rn = 3.7       # electrons
#    qeff = interpolate(waveQE,qe)
##    detec = ???
#    return detec


#################################################################################################
'''
Multiplies all factors of system efficiency.
Returns: q as a function of wavelength.
'''
def calc_q(mirrors,filt,detec):
    qe = open('qeff.txt')
    waveQE,qe = np.loadtxt(f,unpack=True)
    net_q = mirrors*filt*detec
    return net_q









