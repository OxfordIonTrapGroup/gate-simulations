import numpy as np
from math import pi, sqrt

class Mode_frequencies():
    def __init__(self,species='8888',single_ion_freqs=None,m1=None,m2=None):
        """ 
        Calculate motional mode frequencies of ion crystals
        TODO: implement mixed species radial modes
        All frequencies in rad/s, not in 1/s
        freqs in order: ax_ip, rad_ip_l, rad_ip_u (single ion frequencies of ion listed first in species)
        calculated Lamb-Dicke parameters are also those of first ion species
        """
        self.freqs = {}
        
        if len(species) == 4:
            self.mode_names = ['ax_ip','ax_oop','rad_ip_l','rad_ip_u','rad_oop_l','rad_oop_u']
            self.is_single_ion = False
            if species[0:2] == species[2:4]:
                self.is_mixed_species = False
            else:
                self.is_mixed_species = True
            self.m1 = int(species[0:2])
            self.m2 = int(species[2:4])
        elif len(species) == 2:
            self.mode_names = ['ax_ip','rad_ip_l','rad_ip_u']
            self.is_mixed_species = False
            self.is_single_ion = True
            self.m1 = int(species)
            self.epsilon = 0 # not relevant

        if species == '8888':
            ax_freq = 2*pi*1.3414*1e6
            rad_freq_l = 2*pi*1.92*1e6
            rad_freq_u = 2*pi*2.06*1e6
            self.set_frequencies(ax_freq,rad_freq_l,rad_freq_u)
        elif species == '4343':
            ax_freq = 2*pi*1.924e6
            rad_freq_l = 2*pi*3.93e6 
            rad_freq_u = 2*pi*4.22e6 
            self.set_frequencies(ax_freq,rad_freq_l,rad_freq_u)
        elif species == '4040':
            ax_freq = 2*pi*1.997e6
            rad_freq_l = 2*pi*4.402e6 
            rad_freq_u = 2*pi*4.685e6 
            self.set_frequencies(ax_freq,rad_freq_l,rad_freq_u)
        elif species == '4388' or '8843':
            ax_freq = 2*pi*1.924e6
            rad_freq_l = 2*pi*3.93e6 
            rad_freq_u = 2*pi*4.22e6 
            self.set_frequencies(ax_freq,rad_freq_l,rad_freq_u)
        elif species == '88':
            self.freqs['ax_ip'] = 2*pi*1.34175e6
            self.freqs['ax_oop'] = self.f_oop(self.freqs['ax_ip'])
        elif species == '43':
            self.freqs['ax_ip'] = 2*pi*1.924e6
            self.freqs['ax_oop'] = self.f_oop(self.freqs['ax_ip'])
        elif single_ion_freqs == None:
            print('Species not implemented, set frequencies and masses manually!')            
        if single_ion_freqs != None:
            self.set_frequencies(ax_freq=single_ion_freqs[0],rad_freq_l=single_ion_freqs[1],rad_freq_u=single_ion_freqs[2])
        if m1 != None:
            self.m1 = m1
        if m2 != None:
            self.m2 = m2
        
            
    def calc_f_z_mixed_species(self,m1,m2,omega_z_si=1):
        mu= m2/m1
        omega_ip  = np.sqrt((1+mu-np.sqrt(1-mu+mu**2))/(mu))*omega_z_si
        omega_oop = np.sqrt((1+mu+np.sqrt(1-mu+mu**2))/(mu))*omega_z_si
        return omega_ip, omega_oop
    
    def calc_f_rad_mixed_species(self,m1,m2,omega_ip_si,omega_rad_u_si,omega_rad_l_si):
        mu= m2/m1
        #alpha = (1+(omega_rad_l**2-omega_rad_u**2)/omega_ip**2)/2
        epsilon = np.sqrt((1+(omega_rad_l_si**2+omega_rad_u_si**2)/omega_ip_si**2)/2)
        a = np.sqrt(epsilon**4*(mu**2-1)**2-2*epsilon**2*(mu-1)**2*mu*(1+mu)+
                    mu**2*(1+(mu-1)*mu))
        omega_rad_ip_l =  np.sqrt(-(mu+mu**2-epsilon**2*(1+mu**2)-a)/(2*mu**2))*omega_ip_si
        omega_rad_oop_u = np.sqrt(-(mu+mu**2-epsilon**2*(1+mu**2)+a)/(2*mu**2))*omega_ip_si
        return omega_rad_ip_l, omega_rad_oop_u
    
    def calc_epsilon(self,omega_ip_si,omega_rad_u_si,omega_rad_l_si):
        epsilon = np.sqrt((1+(omega_rad_l_si**2+omega_rad_u_si**2)/omega_ip_si**2)/2)
        return epsilon

    def f_rock(self,omega_ip,omega_rad):
        return sqrt(omega_rad**2-omega_ip**2)

    def f_oop(self,omega_ip):
        return sqrt(3)*omega_ip
    
    def set_frequencies(self,ax_freq,rad_freq_l,rad_freq_u,is_mixed_species=None,m1=None,m2=None):
        # all input frequencies are ip frequencies, from which oop freqs are calculated
        # for mixed species: single ion frequencies of ion 1
        if self.is_single_ion:
            if m1 is None:
                m1 = self.m1
            self.freqs['ax'] = ax_freq
            self.freqs['rad_l'] = rad_freq_l
            self.freqs['rad_u'] = rad_freq_u
            self.epsilon = self.calc_epsilon(ax_freq,rad_freq_u,rad_freq_l)
        else:
            if m1 is None:
                m1 = self.m1
            if m2 is None:
                m2 = self.m2
            #TODO find correct formula for upper mode
            omega_ip, omega_oop = self.calc_f_z_mixed_species(m1,m2,omega_z_si=ax_freq)
            omega_rad_ip_l, omega_rad_oop_l = self.calc_f_rad_mixed_species(m1,m2,ax_freq,rad_freq_u,rad_freq_l)
            self.freqs['ax_ip'] = omega_ip
            self.freqs['ax_oop'] = omega_oop
            self.freqs['rad_ip_l'] = omega_rad_ip_l
            self.freqs['rad_oop_l'] = omega_rad_oop_l
            self.freqs['rad_ip_u'] = omega_rad_ip_l
            self.freqs['rad_oop_u'] = omega_rad_oop_l
            self.epsilon = self.calc_epsilon(ax_freq,rad_freq_u,rad_freq_l)

            
    def set_frequencies_manually(self,ax_freq_ip,ax_freq_oop,rad_freq_ip_l,rad_freq_ip_u,rad_freq_oop_l,rad_freq_oop_u):
        self.freqs['ax_ip'] = ax_freq_ip
        self.freqs['ax_oop'] = ax_freq_oop
        self.freqs['rad_ip_l'] = rad_freq_ip_l
        self.freqs['rad_oop_l'] = rad_freq_oop_l
        self.freqs['rad_ip_u'] = rad_freq_ip_u
        self.freqs['rad_oop_u'] = rad_freq_oop_u
        # !Caution! epsilon not update here, bc need to single ion freqs for that
        # this will cause errors if programmed frequencies here are very different to default ones
        # but since usually only the radial requencies are different, it should be fine

                
            
            
            
            
        