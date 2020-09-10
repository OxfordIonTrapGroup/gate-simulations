#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VMS Dec 2017

import numpy as np
from numpy import pi, sqrt, sin, cos
from qutip import qeye, tensor, destroy, fock_dm
from qutip import expect, fidelity, ptrace, sigmam, sigmap, sigmax, sigmay, basis
import random
from gate_simulations.lamb_dicke_factor import Lamb_Dicke_Factor
from gate_simulations.mode_frequencies import Mode_frequencies
from gate_simulations.motional_modes import Mode_structure
from qutip import thermal_dm

# Two qubit gate simulation parent class
class Tqg_simulation():
    

    def __init__(self,nHO=15,nbar_mode=0.0,delta_g=2*pi*40e3,Omega_R=2*pi*80e3,
                 T=25e-6,two_loops=True,species='8888',ion_spacing=-1,mode=1,
                 factor=1,tau_mot=0,tau_spin=0,gamma_el=0,gamma_ram=0,
                 Delta_ca=-2*pi*10e12,Delta_sr=2*pi*30e12,Omega_R_2=2*pi*80e3,
                 phase_insensitive=False,sq_factor=1.0,qudpl_and_raman=False,
                 on_radial_modes=False,nbars = [0.1,0.1,5,5,5,5],
                 mode_freqs=None):  
        ''' nHO: dimension of HO space which is simulated
        nbar_mode: mean thermal population of motional mode
        delta_g: gate detuning
        Omega_R: Rabi frequency
        T: elapsed time
        two loops: do two-loop (True) or single loop (false) gate
        species: ion species of crystal
        ion-spacing: integer (+1) or half-integer (-1) standing wavelengths
        mode: ip (1) or oop (-1), this code assumes gate is performed an axial mode
        factor: square-root of ratio of Rabi frequencies for efficiency tests
        tau_mot: motional coherence time, =0 for intinity
        tau_spin: spin coherence time, =0 for intinity
        gamma_el: Rayleigh elastic dephasing rate
        gamma_ram: Raman scattering rate
        Delta_ca: calcium Raman detuning, from 397
        Delta_sr: strontium Raman detuning, from 422
        sq_factor: relative area offset of single qubit pulses (i.e. to simulate if 
                   Ramsey pi/2 and pi pulses aren't calibrated properly)
        Only for MS gate:
        Omega_R_2: Rabi frequency for second ion, for mixed species gate driven with different lasers
        phase_insensitive: wraps gate in single qubit operations driven by gate laser, to make it insensitive
                           relative to phase of other single qubit operations
        qudpl_and_raman: use two different lasers for MS gate, with different lamb-dicke factors
        on_radial_modes: does gate on radial modes, but still sets it up for axial modes, 
                         this is used for simulating off-resonant excitation of radial modes for a tilted ion crystal
        nbars: temperature of all modes, this is only used for calculating errors from off-resonantly exciting other modes
        ! Note that most gate parameters are class objects, and therefore parameters
        are changed by previously run scans !
        '''
        
        self.nq = 2 # number of qubit states
        
        self.nHO = nHO
        self.nbar_mode = nbar_mode
        self.delta_g = delta_g
        self.Omega_R = Omega_R
        self.Omega_R_2 = Omega_R_2
        self.T = T
        self.two_loops = two_loops
        self.species = species
        self.ion_spacing = ion_spacing
        self.mode = mode
        self.factor = factor
        self.tau_mot = tau_mot
        self.tau_spin = tau_spin
        self.gamma_el = gamma_el
        self.gamma_ram = gamma_ram
        self.Delta_ca = Delta_ca # detuning from 397 transition
        self.Delta_sr = Delta_sr # detuning from 422 transition!! (not 408, would require different matrix elements)
        self.delta_LS = 0 # frequency offset due to light shift
        self.ampl_asym_1 = 0 # amplitude asymmetry in sidebands
        self.ampl_asym_2 = 0 # amplitude asymmetry in sidebands of second species laser
        self.species_Rabi_asym_MS = 0 # Rabi frequency asymmetry btw the two species, for MS gate
        self.phase_insensitive = phase_insensitive # surround gate pulse with pi/2 pulses so that can use analysis pulse that is not phase-stabilised to gate
        self.sq_factor = sq_factor
        self.qudpl_and_raman = qudpl_and_raman
        self.on_radial_modes = on_radial_modes
        self.nbars = nbars

        # single qubit operators
        self.up_up = fock_dm(2,0) # |up><up|
        self.dn_dn = fock_dm(2,1) # |dn><dn|
        self.sz = self.up_up - self.dn_dn
        self.up = basis(2,0) # |up>
        self.dn = basis(2,1) # |dn>
        self.pl = (self.up+self.dn)/2 # |+>
        self.mn = (self.up-self.dn)/2 # |->
        self.pp = self.pl*self.pl.dag() # |+><+|
        self.mm = self.mn*self.mn.dag() # |+><+|
        self.pm = self.pl*self.mn.dag() # |+><+|
        self.mp = self.mn*self.pl.dag() # |+><+|
        
        # two qubit operators
        self.uu_uu = tensor(self.up_up,self.up_up) # |up,up><up,up|
        self.dd_dd = tensor(self.dn_dn,self.dn_dn) # |dn,dn><dn,dn|
        self.ud_ud = tensor(sigmap(),sigmap()) # |up,dn><up,dn|
        self.du_du = tensor(sigmam(),sigmam()) # |dn,up><dn,up|
        
        self.pp_pp = tensor(self.pp,self.pp) # |+,+><+,+|
        self.mm_mm = tensor(self.mm,self.mm) # |-,-><-,-|
        self.pm_pm = tensor(self.pm,self.pm) # |+,-><+,-|
        self.mp_mp = tensor(self.mp,self.mp) # |-,+><-,+|
        
        # two qubit operators with HO, o* means tensor product
        self.uu_id = tensor(self.up_up,qeye(self.nq),qeye(self.nHO)) # |up><up| o* id
        self.id_uu = tensor(qeye(self.nq),self.up_up,qeye(self.nHO)) # id o* |up><up|
        self.dd_id = tensor(self.dn_dn,qeye(self.nq),qeye(self.nHO)) # |dn><dn| o* id
        self.id_dd = tensor(qeye(self.nq),self.dn_dn,qeye(self.nHO)) # id o* |dn><dn|
        
        self.sp_id = tensor(sigmap(),qeye(self.nq),qeye(self.nHO)) # |up><up| o* id
        self.id_sp = tensor(qeye(self.nq),sigmap(),qeye(self.nHO)) # id o* |up><up|
        self.sm_id = tensor(sigmam(),qeye(self.nq),qeye(self.nHO)) # |dn><dn| o* id
        self.id_sm = tensor(qeye(self.nq),sigmam(),qeye(self.nHO)) # id o* |dn><dn|
        
        # readout projection operators   
        self.spin_dndn = tensor(self.dn_dn,self.dn_dn,qeye(self.nHO))
        self.spin_upup = tensor(self.up_up,self.up_up,qeye(self.nHO))
        self.spin_updn_dnup = tensor(self.up_up,self.dn_dn,qeye(self.nHO))+tensor(self.dn_dn,self.up_up,qeye(self.nHO))
        self.spin_updn = tensor(self.up_up,self.dn_dn,qeye(self.nHO))
        self.spin_dnup = tensor(self.dn_dn,self.up_up,qeye(self.nHO))
        
        self.spin_pp = tensor(self.pp,self.pp,qeye(self.nHO))
        self.spin_mm = tensor(self.mm,self.mm,qeye(self.nHO))
        self.spin_pm = tensor(self.pp,self.mm,qeye(self.nHO))
        self.spin_mp = tensor(self.mm,self.pp,qeye(self.nHO))
        
        # annihilation operator
        self.a = tensor(qeye(self.nq),qeye(self.nq),destroy(self.nHO))
        self.ad = self.a.dag()
        
        self.mw_offset_phase_1 = random.random()*2*pi # use this phase offset for mw pulses because they don't have a fixed phase relationship to gate lasers
        self.phi_sum_1 = random.random()*2*pi # sum frequency of phases of two sidebands, ie the phase that determines phi in sigma_phi
        self.phi_diff_1 = random.random()*2*pi # difference frequency of phases of two sidebands, constant and ~0 for our frequency geometry
            
        self.setup_mode_structure(mode_freqs)
        self.set_ion_parameters()
        
    def sig_phi(self,phi):
        # Pauli sigma phi operator
        sig_phi = sigmax()*cos(phi)+sigmay()*sin(phi)
        return sig_phi
    
    def sig_phi_i(self,i,phi=pi/2):
        # Pauli sigma phi operator either acting on ion 1, or ion 2
        if i is 1:
            sig_phi_i = tensor(self.sig_phi(phi),qeye(self.nq),qeye(self.nHO))
        elif i is 2:
            sig_phi_i = tensor(qeye(self.nq),self.sig_phi(phi),qeye(self.nHO))
        return sig_phi_i
        
    # single qubit rotation, on ion 1, ion 2 or both
    def U_rot(self,theta,ion_index=[1,1],phi=0): 
        # ion_index=[i1,i2], go gate on ion ii if ii==1, else, do identity
        if ion_index[0] == 1:
            op_1 = (cos(theta/2)*qeye(self.nq))-(1j*sin(theta/2)*(cos(phi)*sigmax()+sin(phi)*sigmay()))
        else:
            op_1 = qeye(self.nq)
        if ion_index[1] == 1:
            op_2 = (cos(theta/2)*qeye(self.nq))-(1j*sin(theta/2)*(cos(phi)*sigmax()+sin(phi)*sigmay()))
        else:
            op_2 = qeye(self.nq)
        rot=tensor(op_1,op_2,qeye(self.nHO))
        return rot
    
    def setup_mode_structure(self,mode_freqs):
        if self.mode == 1:
            self.nbars[0] = self.nbar_mode
        elif self.mode == -1:
            self.nbars[1] = self.nbar_mode
        if self.species == 'effic_test':
            ion_species = '8888'
        else:
            ion_species = self.species
        self.modes = Mode_structure(nbars=self.nbars,species=ion_species,
                                    qudpl_and_raman=self.qudpl_and_raman)
        if mode_freqs != None:
            self.modes.set_frequencies_manually(*mode_freqs)
    
    def set_frequencies_manually(self,freqs):
        self.modes.set_frequencies_manually(*freqs)
    
    def set_ion_parameters(self):
        # set approximate heating rate and lamb dicke parameters according to ions species
        self.set_lamb_dicke_factors()
        if self.species == '4388':
            if self.mode == 1: #ip mode
                self.n_dot = 120
            else: # oop mode
                self.n_dot = 8
            self.mixed_species = True
        else:
            if self.mode == 1: #ip mode
                self.n_dot = 1e2
            else: # oop mode
                self.n_dot = 1
            if self.species == '4043':
                self.mixed_species = True
            else:
                self.mixed_species = False

            
    def set_lamb_dicke_factors(self,mode_name=None,raman_misalignment=0):
        # set lamb dicke parameters according to ions species
        # look up in Lamb_Dicke_Factor what laser is pre-set for which species
        # (i.e. Raman or quadrupole)
        if mode_name == None:
            if self.on_radial_modes:
                if self.mode == 1:
                    mode_name = 'rad_ip_l'
                elif self.mode == -1:
                    mode_name = 'rad_oop_l'
            else:
                if self.mode == 1:
                    mode_name = 'ax_ip'
                elif self.mode == -1:
                    mode_name = 'ax_oop'
        self.modes.set_raman_misalignement(raman_misalignment)
        self.omega_z = self.modes.modes[mode_name].freq
        self.eta_1 = abs(self.modes.modes[mode_name].eta)
        self.eta_2 = abs(self.modes.modes[mode_name].eta_2)
        
        
    def return_omega_mode(self):
        # read out mode frequency used for simulation, programmed into Mode_frequencies
        if self.species == 'effic_test':
            species = '8888'
        else:
            species = self.species
        if self.mode == 1:
            mode_name = 'ax_ip'
        elif self.mode == -1:
            mode_name = 'ax_oop'
        modes = Mode_frequencies(species=species)
        mode_freq = modes.freqs[mode_name]
        return mode_freq

    
    def set_sq_pulse_miscalibration(self,sq_factor):
        self.sq_factor = sq_factor
        
    def set_gate_length(self,T):
        self.T = T
        
    def set_heating_rate(self,ndot):
        self.n_dot = ndot
        
    def set_mot_dephasing_time(self,tau):
        self.tau_mot = tau
        
    def set_spin_dephasing_time(self,tau):
        self.tau_spin = tau
        
    def set_ion_temp(self,nbar):
        self.nbar_mode = nbar
        
    def set_el_deph_rate(self,gamma):
        self.gamma_el = gamma
        
    def set_ram_scat_rate(self,gamma):
        self.gamma_ram = gamma
        
    def set_gate_detuning(self,delta_g):
        self.delta_g = delta_g
        
    def set_Rabi_freq(self,Omega_R):
        self.Omega_R = Omega_R
        self.set_relative_Rabi_frequencies()
        
    def set_Rabi_freq_2(self,Omega_R_2):
        self.Omega_R_2 = Omega_R_2
        self.set_relative_Rabi_frequencies()
        
    def set_delta_LS(self,delta_LS):
        self.delta_LS = delta_LS
        
    def set_ampl_asym(self,ampl_asym=None):
        self.ampl_asym_1 = ampl_asym
        
    def set_ampl_asym_2(self,ampl_asym_2=None):
        self.ampl_asym_2 = ampl_asym_2
        
    def set_species_Rabi_asym_MS(self,species_Rabi_asym_MS=None):
        self.species_Rabi_asym_MS = species_Rabi_asym_MS
        
    def calc_ideal_gate_detuning(self,T=None,verbose=True):
        if T is not None:
            T = self.t_g
        if self.two_loops:
            K = 2
        else:
            K = 1
        delta_g = 2*pi/T*K
        if verbose:
            print('Calculated ideal gate detuning: {:.2f}kHz'.format(delta_g/1e3))
        return delta_g
    
    def calc_ideal_gate_time(self,verbose=True):
        if self.two_loops:
            K = 2
        else:
            K = 1
        t_g = 2*pi/self.delta_g*K
        if verbose:
            print('Calculated ideal gate time: {:.2f}us'.format(t_g*1e6))
        return t_g
    
    def set_custom_parameters(self,delta_g=None,Omega_R=None,Omega_R_2=None,
                              T=None,ndot=None,delta_LS=None,ampl_asym=None,
                              ampl_asym_2=None,tau_mot=None,factor=None,gamma_el=None,
                              gamma_ram=None,tau_spin=None,species_Rabi_asym_MS=None):
        # if not set separately, t_g and Omega_R are calculated from self.delta_g
        # delta_g is not calculated, but remains its old value if it isn't set explicitly
        if delta_g is not None:
            self.set_gate_detuning(delta_g)
        if T is not None:
            self.set_gate_length(T)
        else:
            self.set_gate_length(self.calc_ideal_gate_time(verbose=False))
        if species_Rabi_asym_MS is not None:
            self.set_species_Rabi_asym_MS(species_Rabi_asym_MS)
        if Omega_R is not None:
            self.set_Rabi_freq(Omega_R)
        else:
            OR_1, OR_2 = self.calc_ideal_Rabi_freq(verbose=False)
            self.set_Rabi_freq(OR_1)
        if Omega_R_2 is not None:
            self.set_Rabi_freq_2(Omega_R_2)
        else:
            OR_1, OR_2 = self.calc_ideal_Rabi_freq(verbose=False)
            self.set_Rabi_freq_2(OR_2)
        if delta_LS is not None:
            self.set_delta_LS(delta_LS)
        if ampl_asym is not None:
            if ampl_asym_2 is None: # if only ampl_asym is changed, set both one and two to ampl_asym, because assuming one species
                self.set_ampl_asym_2(ampl_asym)
            self.set_ampl_asym(ampl_asym)
        if ampl_asym_2 is not None:
            self.set_ampl_asym_2(ampl_asym_2)
        if ndot is not None:
            self.set_heating_rate(ndot)
        if factor is not None:
            self.set_eff_factor(factor)
        if tau_mot is not None:
            self.set_mot_dephasing_time(tau_mot)
        if gamma_el is not None:
            self.set_el_deph_rate(gamma_el)
        if gamma_ram is not None:
            self.set_ram_scat_rate(gamma_ram)
        if tau_spin is not None:
            self.set_spin_dephasing_time(tau_spin)
    
    def timescan(self,nT=100,**kwargs):
        # do a gate dynamics time scan
        # !!! If parameters are not separately specified, chooses optimal gate parameters
        # for programmed gate detuning
        
        self.set_custom_parameters(**kwargs)
        
        # initialize result vectors
        end_populations_dndn = []
        end_populations_upup = []
        end_populations_updn = []
        end_populations_dnup = []
        fidelities = []
        
        times, final_rhos = self.do_gate(nT=nT)
        
        for ii in range(len(times)):
            
            end_populations_upup.append(expect(self.spin_upup,final_rhos[ii]))
            end_populations_dndn.append(expect(self.spin_dndn,final_rhos[ii]))
            end_populations_updn.append(expect(self.spin_updn,final_rhos[ii]))
            end_populations_dnup.append(expect(self.spin_dnup,final_rhos[ii]))
            
            rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
            fidelities.append(fidelity(rho_target,ptrace(final_rhos[ii],[0,1]))**2)
            
        return times, end_populations_upup, end_populations_dndn, end_populations_updn, end_populations_dnup, fidelities
    
    def detuning_scan(self,delta_g_vec,delta_g_0=None,**kwargs):
        # do a gate dynamics frequency scan
        # !!! If parameters are not separately specified, chooses optimal gate parameters
        # for programmed gate detuning
        
        self.set_custom_parameters(delta_g=delta_g_0,**kwargs)
        
        # initialize result vectors
        end_populations_dndn = []
        end_populations_upup = []
        end_populations_updn = []
        end_populations_dnup = []
        fidelities = []
        
        for delta_g in delta_g_vec:
            self.set_gate_detuning(delta_g)
            times, final_rhos = self.do_gate(nT=2)
            
            end_populations_upup.append(expect(self.spin_upup,final_rhos[-1]))
            end_populations_dndn.append(expect(self.spin_dndn,final_rhos[-1]))
            end_populations_updn.append(expect(self.spin_updn,final_rhos[-1]))
            end_populations_dnup.append(expect(self.spin_dnup,final_rhos[-1]))
            
            rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
            fidelities.append(fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return end_populations_upup, end_populations_dndn, end_populations_updn, end_populations_dnup, fidelities
    
    def parity_scan(self,analysis_phase,scramble_mw_phase=False,fixed_analysis_phase=True,
                    scramble_laser_phase=False,is_wobble_gate=False,**kwargs):
        # parity scan for simulated fidelity measurement
        # analysis_phase: vector of analysis pi/2 pulse phases to be evaluated
        # for MS gate only:
        # faster with scramble_mw_phase=False
        # scramble_mw_phase: for every phase phi, set mw phase to random value, in general different to laser phase
        # scramble_laser_phase: for every phase phi, set laser phase phi_sum to random value, in general different to mw phase (as would be the case in an experiment)
        # fixed_analysis_phase: use same phase for analysis pulse and gate, ie effectively do analysis pulse with gate laser, rather than with microwaves/rf
        # !!! If parameters are not separately specified, chooses optimal gate parameters
        # for programmed gate detuning
        
        self.set_custom_parameters(**kwargs)
        
        # initialize result vectors
        end_populations_dndn = []
        end_populations_upup = []
        end_populations_updn_dnup = []
        #fidelities = []
        times, final_rhos = self.do_gate(nT=2)
        
        for phi in analysis_phase:
            if is_wobble_gate:
                phi_analysis_1 = phi
                phi_analysis_2 = phi
            else: # for MS gate, have included all possible kinds of phase randomisations here
                if scramble_mw_phase:
                    self.mw_offset_phase_1 = random.random()*2*pi
                    if self.mixed_species:
                        self.mw_offset_phase_2 = random.random()*2*pi
                    else:
                        self.mw_offset_phase_2 = self.mw_offset_phase_1
                if scramble_laser_phase:
                    self.phi_sum_1 = random.random()*2*pi
                    if self.mixed_species:
                        self.phi_sum_2 = random.random()*2*pi
                    else:
                        self.phi_sum_2 = self.phi_sum_1
                if scramble_mw_phase or scramble_laser_phase:
                    times, final_rhos = self.do_gate(nT=2)
                if fixed_analysis_phase:
                    phi_analysis_1 = phi-self.phi_sum_1
                    phi_analysis_2 = phi-self.phi_sum_2
                else:
                    phi_analysis_1 = self.mw_offset_phase_1+phi
                    phi_analysis_2 = self.mw_offset_phase_2+phi
            rho_after_analysis = self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=phi_analysis_1)*\
                                 self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=phi_analysis_2)*\
                                 final_rhos[-1]* \
                                 self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=phi_analysis_1).dag() *\
                                 self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=phi_analysis_2).dag()
            final_rho = rho_after_analysis

            end_populations_upup.append(expect(self.spin_upup,final_rho))
            end_populations_dndn.append(expect(self.spin_dndn,final_rho))
            end_populations_updn_dnup.append(expect(self.spin_updn_dnup,final_rho))
#            rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
#            fidelities.append(fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
        return end_populations_upup, end_populations_dndn, end_populations_updn_dnup#, fidelities
    
        
    def scan_heating_rate(self,ndot_max=1e4,n_steps=10,**kwargs):
        # simulate gate fidelity for different heating rates
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        
        errors = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        n_dots = np.linspace(0, ndot_max, n_steps)

        for ndot in n_dots:
            self.set_heating_rate(ndot)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return n_dots, errors
    
    def scan_tilt_angle(self,angle_max=10/360*2*pi,angle_min=-10/360*2*pi,n_steps=10,mode_name='rad_ip_l',**kwargs):
        # simulate gate fidelity for different heating rates
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        
        errors = []
        rho_target = tensor(self.dn_dn,self.dn_dn)
        tilt_angles = np.linspace(angle_min, angle_max, n_steps)
        
        self.set_ion_temp(self.modes.modes[mode_name].nbar)
        detuning_offset = self.omega_z - self.modes.modes[mode_name].freq
        self.set_gate_detuning(self.delta_g+detuning_offset)

        for angle in tilt_angles:
            self.set_lamb_dicke_factors(mode_name=mode_name,raman_misalignment=angle)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return tilt_angles, errors
    
    def scan_sq_error(self,sq_factor_max=1.05,sq_factor_min=0.995,n_steps=10,**kwargs):
        # simulate gate fidelity for different heating rates
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        
        errors = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        sq_factors = np.linspace(sq_factor_min, sq_factor_max, n_steps)

        for sq_factor in sq_factors:
            self.set_sq_pulse_miscalibration(sq_factor)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return sq_factors, errors
    
    def scan_delta_g(self,delta_max=10e3,delta_min=300e3,n_steps=10,**kwargs):
        # simulate gate fidelity for different gate detunings, with t_g and Omega_R
        # optimised for this delta_g
        # i.e. determines scaling of errors due to eg heating with delta_g
        errors = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        deltas = np.linspace(delta_min, delta_max, n_steps)

        for delta in deltas:
            self.set_custom_parameters(delta_g=delta,**kwargs)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return deltas, errors
    
    def scan_t_g(self,t_max=50e-6,t_min=10e-6,n_steps=10,tau=0,
                     ndot=0,factor=1,gamma_el=0,gamma_ram=0,**kwargs):
        # simulate gate fidelity for different gate pulse durations, with delta_g
        # and Omega_R optimised for this t_g
        # i.e. determines scaling of errors due to eg heating with t_g
        errors = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        ts = np.linspace(t_min, t_max, n_steps)

        for t_g in ts:
            delta_g = self.calc_ideal_gate_detuning(T=t_g,verbose=False)
            self.set_custom_parameters(delta_g=delta_g,**kwargs)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
        return ts, errors
    
    def scan_mot_dephasing_rate(self,tau_max=10e-3,tau_min=10e-6,n_steps=10,**kwargs):
        # simulate gate fidelity for different motional dephasing rates
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        errors = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        taus = np.linspace(tau_min, tau_max, n_steps)
        self.set_heating_rate(0)
        for tau in taus:
            self.set_mot_dephasing_time(tau)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return taus, errors
    
    def scan_el_dephasing_rate(self,gamma_max=10e-3,n_steps=10,**kwargs):
        # simulate gate fidelity for different elastic scattering rates
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        errors = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        gammas = np.linspace(0, gamma_max, n_steps)
        self.set_heating_rate(0)
        for gamma in gammas:
            self.set_el_deph_rate(gamma)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return gammas, errors
    
    def scan_spin_dephasing_rate(self,tau_min=1e-3,tau_max=10e-3,n_steps=10,**kwargs):
        # simulate gate fidelity for different spin dephasing rates
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        errors = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        taus = np.linspace(tau_min, tau_max, n_steps)
        self.set_heating_rate(0)
        for tau in taus:
            self.set_spin_dephasing_time(tau)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return taus, errors
    
    def scan_raman_scattering_rate(self,gamma_max=10e-3,n_steps=10,**kwargs):
        # simulate gate fidelity for different raman scattering rates
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        errors = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        gammas = np.linspace(0, gamma_max, n_steps)
        self.set_heating_rate(0)
        for gamma in gammas:
            self.set_ram_scat_rate(gamma)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return gammas, errors

    def calc_fid(self,t_g=25e-6,tau=0,factor=1,delta_g=2*pi*80e3,Omega_r=2*pi*224.7e3,ndot=0,nbar=0.0):
        # calculate fidelity for given set of parameters
        # initialize result vectors
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        self.set_ion_temp(nbar)
        self.set_gate_detuning(delta_g)
        self.set_gate_length(t_g)
        self.set_Rabi_freq(Omega_r)
        self.set_eff_factor(factor)
#        self.set_mot_dephasing_time(tau)
#        self.set_el_deph_rate(gamma_el)
#        self.set_ram_scat_rate(gamma_ram)
        self.set_heating_rate(ndot)
        times, final_rhos = self.do_gate()
        error = (1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
        return error

    def scan_ion_temp(self,nbar_max=1,n_steps=10,**kwargs):
        # simulate gate fidelity for different hinitial ion temperatures
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        errors = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        nbars = np.linspace(0, nbar_max, n_steps)

        for nbar in nbars:
            self.set_ion_temp(nbar)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return nbars, errors
    
    
    
    
    