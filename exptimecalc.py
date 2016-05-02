import numpy as np
import math
import glob
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

# Atmospheric extinction file from Jon
atmos_extinct_file='data/atmosphere.dat'

# DIS detector quantum efficiency
qe_disR_file='data/QE_dis_red'
qe_disB_file='data/QE_dis_blue'

# ARCTIC detector quantum efficiency
qe_arctic_file='data/QE_arctic'

#################################################################################################
'''
The main code
'''
def exptimecalc(instr=None,grating=None,mag=None,snr=100.0,teff=None,filt=None,wcent=5500.0,wspan=100.0,
                seeing=None,airmass=None,moonphase=None):

    # get wavelength limits based on center and span
    w1 = wcent-(wspan/2.)
    w2 = wcent+(wspan/2.)

    # get the telescope collecting area (in angstroms currently... maybe should be in m)
    collecting_area = get_collecting_area()

    # get instrument parameters if not already supplied
    instrinfo = get_instr_params(instr=instr,grating=grating,filt=filt)
    instr=instrinfo[0]; grating=instrinfo[1]; filt=instrinfo[2]
    gain=instrinfo[3]; readnoise=instrinfo[4]; platescale=instrinfo[5]

    # get parameters specific to object & observing conditions
    obsinfo=get_obsinfo(mag=mag,teff=teff,seeing=seeing,airmass=airmass,moonphase=moonphase)
    mag=obsinfo[0]; teff=obsinfo[1]; seeing=obsinfo[2]; airmass=obsinfo[3]; moonphase=obsinfo[4]

    # get a flux-calibrated blackbody of target, based on V mag, teff, and vega spectrum
    wave,objflux = get_blackbody_flux(teff=teff,mag=mag,w1=w1,w2=w2,wcent=wcent)
#    objflux=(objflux*wave)/(h*(c*10**8))

    # get the throughput of 3 aluminum mirros
    mirror_wave,mirror_trans = get_mirror_throughput(w1=w1,w2=w2)

    # get the filter transmission for arctic
    if instr=='arctic':
        filt_wave,filt_trans = get_filter_transmission(filt=filt,w1=w1,w2=w2)

    # get the quantum efficiency
    qe_wave,qe = get_qe(instr=instr,grating=grating,w1=w1,w2=w2)

    # get the mean extinction
    atmos_wave,atmos_extinct = get_mean_extinction(w1=w1,w2=w2,airmass=airmass)

#    sys_eff = get_sys_eff()

#    signal=do_count_equation(objflux=objflux,collecting_area=collecting_area,sys_eff=sys_eff,atm_trans=atm_trans)

    exptime='???????'

    print('='*70)
    print('The exposure time needed for a S/N = '+str(snr)+' observation of a')
    print('V = '+str(mag)+', Teff = '+str(teff)+' K star observed at')
    print('airmass = '+str(airmass)+', during seeing = '+str(seeing)+'",')
    print(str(moonphase)+' days after new moon, using the '+instr)
    print('instrument on the APO 3.5m telescope is:')
    print('\n'+exptime+' seconds')

    return wave,objflux,atmos_wave,atmos_extinct


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
Get collecting area
'''
def get_collecting_area():
    central_hole=0.77e10 #angstroms
    primary_mirror=3.5e10 #angstroms

    area=(math.pi*(primary_mirror/2.)**2.0)-(math.pi*(central_hole/2.)**2.0)

    return area


#################################################################################################
'''
Get atmospheric extinction
'''
def get_mean_extinction(w1=None,w2=None,airmass=None):
    # read in the atmosphere.dat file from Jon
    extinctdata=ascii.read(atmos_extinct_file)
    ext_wave,ext = interp(extinctdata['wavelength'],extinctdata['extinction'])

    # restrict to the specified wavelength limits
    gd=np.where((ext_wave>=w1) & (ext_wave<w2))
    if len(gd[0])==0:
        print('There is an error in get_mean_extinction.')
        exit_code()
    else:
        ext_wave=ext_wave[gd]; ext=ext[gd]

    return ext_wave,ext


#################################################################################################
'''
Get filter transmission
'''
def get_filter_transmission(filt=None,w1=None,w2=None):
    # read in the filter transmission curve
    filtfile='data/filt_'+filt
    filtdata=ascii.read(filtfile)

    # get the interpolated arrays
    filt_wave,filt_trans = interp(filtdata['wavelength'],filtdata['transmission'])

    # restrict to the specified wavelength limits
    gd=np.where((filt_wave>=w1) & (filt_wave<w2))
    if len(gd[0])==0:
        print('\n'+'*'*70)
        print('There is an error in get_filter_transmission.')
        exit_code()
    else:
        filt_wave=filt_wave[gd]; filt_trans=filt_trans[gd]

    return filt_wave,filt_trans


#################################################################################################
'''
Get detector quantum efficiency
'''
def get_qe(instr=None,grating=None,w1=None,w2=None):
    # figure out which qe file to read in
    if instr=='dis':
        tmp=grating[:1].lower()
        if tmp=='r': qedata=ascii.read(qe_disR_file)
        if tmp=='b': qedata=ascii.read(qe_disB_file)
    if instr=='arctic': qedata=ascii.read(qe_arctic_file)

    # get the interpolated arrays
    qe_wave,qe = interp(qedata['wavelength'],qedata['qe'])

    # restrict to the specified wavelength limits
    gd=np.where((qe_wave>=w1) & (qe_wave<w2))
    if len(gd[0])==0:
        print('\n'+'*'*70)
        print('There is an error in get_qe.')
        exit_code()
    else:
        qe_wave=qe_wave[gd]; qe=qe[gd]

    return qe_wave,qe


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

    # To the third power - for each of the three mirrors on the 3.5m
    alreflect=alreflect**3.0

    # restrict to the specified wavelength limits
    gd=np.where((alwave>=w1) & (alwave<w2))
    if len(gd[0])==0:
        print('\n'+'*'*70)
        print('There is an error in get_mirror_throughput.')
        exit_code()
    else:
        alwave=alwave[gd]; alreflect=alreflect[gd]

    return alwave,alreflect


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
Get Instrument parameters (grating, filter, gain, readnoise, and platescale)
'''
def get_instr_params(instr=None,grating=None,filt=None):
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

        if filt is not None:
            tmp='data/filt_'+filt
            filtfile=glob.glob(tmp)
            if len(filtfile)==0:
                print('A transmission curve file ('+tmp+') was not found for the specified filter.')
                exit_code()
        else:
            filt_files=np.array(glob.glob('data/filt_*'))
            nfilt=len(filt_files)
            print('='*70)
            print('* Which ARCTIC filter are you using?')
            tmpfilt=[]
            for i in range(nfilt):
                bla=filt_files[i].split('_'); tmpfilt.append(bla[len(bla)-1])
                if i==0: 
                    print(' - enter '+str(i+1)+' for '+tmpfilt[i]+' (default)')
                else:
                    print(' - enter '+str(i+1)+' for '+tmpfilt[i])
            tmpfilt=np.array(tmpfilt)
            a = raw_input('')
            if a=='':  a='1'
            if a=='q': exit_code()
            filt=tmpfilt[int(a)-1]
        print('Ok, you are using the '+filt+' filter.')


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
            grating=disgratings[int(a)-1]
        print('Ok, you are using '+grating+'.')
        

        # now that grating is established, look up info in table
        gd=np.where(grating==disgratings)
        if len(gd[0])==0: sys.exit('The specified grating does not exist.')
        gain=distable['gain'][gd]
        readnoise=distable['readnoise'][gd]
        platescale=distable['platescale'][gd]
        gain=gain.tolist(); gain=gain[0]
        readnoise=readnoise.tolist(); readnoise=readnoise[0]
        platescale=platescale.tolist(); platescale=platescale[0]

    print('Ok. gain = '+str(gain)+', readnoise='+str(readnoise)+', platescale='+str(platescale)+'.')

    return instr,grating,filt,gain,readnoise,platescale


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
Do count equation
'''
def do_count_equation(objflux=None,collecting_area=None,sys_eff=None,atm_trans=None):
    signal=collecting_area*atm_trans*sys_eff*objflux


    return signal


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









