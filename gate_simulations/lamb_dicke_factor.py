import numpy as np
from scipy import constants as u
from math import pi, sin, cos, sqrt


class Lamb_Dicke_Factor():
    def __init__(self,nu=755.222766e12,f_z=1.924e6,N=1,m1=43,m2=0,is_qudpl=False,
                 delta=0,species=None,epsilon=0,qudpl_and_raman=False, raman_misalignment=0, qdp_misalignment=0):
        """ Calculates Lamb-Dicke factors for mixed species crystals,
        according to derivations in
        Wübbena et al (2012): Sympathetic cooling of mixed-species two-ion crystals for precision spectroscopy
        nu: laser resonance frequency in Hz
        f_z: motional mode frequency OF SINGLE ION in Hz
        N: number of ions
        m1/m2: mass of ion 1/2, in amu
        is_qudpl: is quadrupole transition, if False: dipole transition
        delta: Raman detuning in Hz, 0 for quadrupole transitions
        epsilon: omega_z/omega_p, see Wubbena and Kielpinski papers
        species: set pre-defined parameters for typical species 
        """
        self.delta = delta
        self.qudpl_and_raman = qudpl_and_raman
        if species is not None:
            self.set_species(species)
        else:
            self.nu = nu + delta
            self.is_qudpl = is_qudpl
            self.is_qudpl_2 = self.is_qudpl
            self.f_z = f_z
            self.N = N
            self.m1 = m1
            self.m2 = m2
        self.beta_radial_qdp = (60-qdp_misalignment)/360*2*pi # 60 for quadrupole laser, 90 for Raman lasers
        self.beta_radial_raman = (90+raman_misalignment)/360*2*pi # 90 for Raman lasers
        self.beta_axial_qdp = (0-qdp_misalignment)/360*2*pi # 45 for qdp, 0 for raman lasers
        self.beta_axial_raman = (0+raman_misalignment)/360*2*pi # 45 for qdp, 0 for raman lasers
        self.alpha_raman = 90/360*2*pi # 45 for qdp, 0 for raman lasers
        self.alpha_qdp = pi/2 # formula different for quadrupole, but for these values and our geometry this gives correct result
        self.epsilon = epsilon
        
    def set_epsilon(self,epsilon):
        self.epsilon = epsilon
        
    def set_raman_misalignement(self,raman_misalignment):
        self.beta_radial_raman = (90+raman_misalignment)/360*2*pi # 90 for Raman lasers
        self.beta_axial_raman = (0+raman_misalignment)/360*2*pi
            
    def set_species(self,species):
        #self.mf = Mode_frequencies(species=species)
        if (species == '88') or (species == '8888'):
            self.nu = 444.779044095e12 + self.delta
            self.is_qudpl = True
            self.is_qudpl_2 = self.is_qudpl
            self.m1 = 88
            self.mixed_species = False
            if species == '88':
                self.N = 1
            elif species == '8888':
                self.N = 2
                self.m2 = 88
        elif (species == '43') or (species == '4343'):
            self.nu = 755.222766e12 + self.delta
            self.is_qudpl = False
            self.is_qudpl_2 = self.is_qudpl
            self.m1 = 43
            self.mixed_species = False
            if species == '43':
                self.N = 1
            elif species == '4343':
                self.N = 2
                self.m2 = 43
        elif species == '4388':
            self.nu = 755.222766e12 + self.delta
            self.is_qudpl = False
            if self.qudpl_and_raman:
                self.nu_2 = 444.779044095e12 + self.delta
                self.is_qudpl_2 = True
            else:
                self.is_qudpl_2 = self.is_qudpl
            self.N = 2
            self.m1 = 43
            self.m2 = 88
            self.mixed_species = True
        elif species == '8843':
            self.nu = 444.779044095e12 + self.delta
            self.is_qudpl = True
            if self.qudpl_and_raman:
                self.nu_2 = 755.222766e12 + self.delta
                self.is_qudpl_2 = False
            else:
                self.is_qudpl_2 = self.is_qudpl
            self.N = 2
            self.m1 = 88
            self.m2 = 43
            self.mixed_species = True
        elif (species == '40') or (species == '4040'):
            self.nu = 411.0421297763932e12 + self.delta
            self.is_qudpl = True
            self.is_qudpl_2 = self.is_qudpl
#            self.nu = 755.222766e12 + self.delta
#            self.is_qudpl = False
            #self.f_z = 1.23e6 # from J. Benhelm paper
            self.m1 = 40
            self.mixed_species = False
            if species == '40':
                self.N = 1
            elif species == '4040':
                self.N = 2
                self.m2 = 40
        elif species == '4043':
            self.nu = 755.222766e12 + self.delta
            self.is_qudpl = False
            self.is_qudpl_2 = self.is_qudpl
            #self.f_z = 1.998e6
            self.N = 2
            self.m1 = 40
            self.m2 = 43
            self.mixed_species = True
        elif species == '0924':
            self.nu = 755.222766e12 + self.delta #TODO change
            self.is_qudpl = False
            self.is_qudpl_2 = False
            #self.f_z = 1.998e6
            self.N = 2
            self.m1 = 9
            self.m2 = 24
            self.mixed_species = True
        else:
            print('Species not predefined. Set custom values.')
            
            
    def calc_lamb_dicke(self,mode_freq,mode_name='ax_ip',ion_ind=1,beta=0,alpha=0):
        if self.N == 1:
            return self._lamb_dicke(mode_freq)
        elif self.N == 2:
            return self._lamb_dicke_2_ions(mode_freq,mode_name,ion_ind,beta,alpha)

    def _calc_zeta(self,mode_name='ax_ip'):
        #a = sy.sqrt(eps**2*(mu**2-1)**2-2*eps**2*(mu-1)**2*mu*(1+mu)+mu**2*(1+(mu-1)*mu))
        mu= self.m2/self.m1
        if mode_name == 'ax_ip': # TODO extend for rocking modes
            b1z = np.sqrt((1-mu+np.sqrt(1-mu+mu**2))/(2*np.sqrt(1-mu+mu**2)))
            b2z = np.sqrt(1-b1z**2)
        elif mode_name == 'ax_oop':
            b2z = -np.sqrt((1-mu+np.sqrt(1-mu+mu**2))/(2*np.sqrt(1-mu+mu**2)))
            b1z = np.sqrt(1-b2z**2)
        elif (mode_name == 'rad_ip_l') or (mode_name == 'rad_ip_u'):
            if self.epsilon == 0:
                print('Error! Epsilon not set')
            a = np.sqrt(self.epsilon**4*(mu**2-1)**2-2*self.epsilon**2*(mu-1)**2*mu*(1+mu)+mu**2*(1+(mu-1)*mu))
            b1z = np.sqrt((mu-mu**2+self.epsilon**2*(mu**2-1)+a)/(2*a))
            b2z = np.sqrt(1-b1z**2)
        elif (mode_name == 'rad_oop_l') or (mode_name == 'rad_oop_u'):
            if self.epsilon == 0:
                print('Error! Epsilon not set')
            a = np.sqrt(self.epsilon**4*(mu**2-1)**2-2*self.epsilon**2*(mu-1)**2*mu*(1+mu)+mu**2*(1+(mu-1)*mu))
            b2z = -np.sqrt((mu-mu**2+self.epsilon**2*(mu**2-1)+a)/(2*a))
            b1z = np.sqrt(1-b2z**2)
        else:
            print('Error: mode not implemented')
            b1z = 0
            b1z = 0
        return b1z, b2z
    
    def _lamb_dicke(self,mode_freq):
        f_z = mode_freq/(2*pi)
        lamb = u.c/(self.nu)
        amu = 1/u.N_A*1e-3
        eta = np.sqrt(u.hbar/(self.m1*amu*pi*f_z))*2*pi/lamb/np.sqrt(2)
        if self.is_qudpl:
            eta = eta*sqrt(2)*cos(pi/4)/2 # for 45deg angle between z and k, factor /2 bc delta_k =2 sin(alpha/2)*k
        return eta
    
    def _lamb_dicke_2_ions(self,mode_freq,mode_name,ion_ind,beta_arg,alpha_arg):
        zeta_1, zeta_2 = self._calc_zeta(mode_name=mode_name)
        if ion_ind == 1:
            zeta = zeta_1
            m = self.m1
            is_qudpl = self.is_qudpl
        else:
            zeta = zeta_2
            m = self.m2
            is_qudpl = self.is_qudpl_2
        if beta_arg != 0:
            beta = beta_arg
        else:
            if (mode_name == 'ax_oop') or (mode_name == 'ax_ip'):
                if is_qudpl:
                    beta = self.beta_axial_qdp
                else:
                    beta = self.beta_axial_raman
            else:
                if is_qudpl:
                    beta = self.beta_radial_qdp
                else:
                    beta = self.beta_radial_raman
        if alpha_arg != 0:
            alpha = alpha_arg
        else:
            if is_qudpl:
                alpha = self.alpha_qdp
            else:
                alpha = self.alpha_raman
        f_z = mode_freq/(2*pi)
        if ion_ind == 2:
            if self.qudpl_and_raman:
                lamb = u.c/(self.nu_2)
            else:
                lamb = u.c/(self.nu)
        else:
            lamb = u.c/(self.nu)
        amu = 1/u.N_A*1e-3
        eta = np.sqrt(u.hbar/(m*amu*pi*f_z))*2*pi/lamb*zeta*cos(beta)*sin(alpha/2)
        if ion_ind == 2:
            if self.qudpl_and_raman:
                if self.is_qudpl_2:
                    eta = eta *sqrt(2)*cos(pi/4)/2
            else:
                if self.is_qudpl:
                    eta = eta *sqrt(2)*cos(pi/4)/2
        else:
            if self.is_qudpl:
                eta = eta *sqrt(2)*cos(pi/4)/2
        return eta



