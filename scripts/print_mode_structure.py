from scipy import constants as u
from math import sqrt, pi
from numpy import pi, sqrt
from gate_simulations.mode_frequencies import Mode_frequencies
from gate_simulations.lamb_dicke_factor import Lamb_Dicke_Factor

###############################################################################
### Set up gate and trap parameters
###############################################################################

#species='8843'
##freq_vec = [2*pi*1.3414*1e6,2*pi*1.92*1e6,2*pi*2.06*1e6] # # freqs single Sr
species='4388'
#species='4343'
freq_vec = [2*pi*1.919*1e6,2*pi*3.93*1e6,2*pi*4.22*1e6] # freqs single Ca
#species='2409'
#freq_vec = [2*pi*1.65*1e6,2*pi*3.72*1e6,2*pi*4.82*1e6]
#species='0924'
#freq_vec = [2*pi*2.69*1e6,2*pi*11.19*1e6,2*pi*12.26*1e6]
modes = Mode_frequencies(species=species,freqs=freq_vec)

freqs = [2*pi*1.505*1e6,2*pi*2.92*1e6,2*pi*4.115*1e6,2*pi*4.56*1e6,2*pi*1.475*1e6,2*pi*1.63*1e6] # Ca-Sr
modes.set_frequencies_manually(*freqs)

ld = Lamb_Dicke_Factor(species=species,epsilon=modes.epsilon)


for mode in modes.freqs:
    print('Freq {} mode: {:.2f}MHz'.format(mode,modes.freqs[mode]/1e6/(2*pi)))
    b1,b2 = ld._calc_zeta(mode_name=mode)
    ldf = ld.calc_lamb_dicke(modes.freqs[mode],mode)
    print('Eigenvector element Ca: {:.3f}, Sr: {:.3f}'.format(b1,b2))
    print('Lamb Dicke factor {}: {:.3f}'.format(species[0:2],ldf))
    print(' ')
    
    
# Notes
# (i)   the radial mode frequencies are only calculated approximately, mode splitting is not included
#       for most accurate results manually set all mode frequencies from measured values
# (ii)  The printed out lamb-Dicke factor is always for the ion first listed in the species vecor
#       i.e. for species='4388' for Calcium and for species='8843' for Strontium
#       when changing the ion species don't forget to change the single ion mode frequencies
# (iii) Lamb-Dicke factor makes assumptions about the beam geometry and lasers used,
#       adjust these for your experiment
# (iv)  axial mode Lamb-Dicke factors agree well with experimentally measured ones in Ca-Sr crystal
#       radial ones have not yet been tested