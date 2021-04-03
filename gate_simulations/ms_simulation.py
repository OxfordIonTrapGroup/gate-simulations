#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VMS Dec 2017

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, pi, sqrt, sin, cos
from qutip import qeye, tensor, destroy, fock_dm, rx, ry, mesolve, thermal_dm, expect, fidelity, ptrace, sigmam, sigmap, sigmax, sigmay
import pylab as pyl
import random
from gate_simulations.tqg_simulation import Tqg_simulation

class Ms_simulation(Tqg_simulation):

    def __init__(self,scramble_phases=True,do_center_pi=False,do_Walsh=False,
                 second_loop_phase_error=0,phi_comp_scnd_sb=0,**kwargs):  
        ''' nHO: dimension of HO space which is simulated
        nbar_mode: mean thermal population of motional mode
        delta_g: gate detuning
        Omega_R: Rabi frequency
        T: elapsed time
        two loops: do two-loop (True) or single loop (false) gate
        species: ion species of crystal
        ion-spacing: integer (+1) or half-integer (-1) standing wavelengths
        mode: ip (1) or oop (-1)
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
        ! Note that most gate parameters are class objects, and therefore parameters
        are changed by previously run scans !'''
        
        super().__init__(**kwargs)
        
        self.do_center_pi = do_center_pi
        self.do_Walsh = do_Walsh
        self.second_loop_phase_error = second_loop_phase_error # only != 0 if miscalibrate phase of second loop
        self.phi_comp_scnd_sb = phi_comp_scnd_sb # only != 0 if accidentally compensate the phase of one (R)SB differently than the other (B)SB
        
        if scramble_phases:
            self.mw_offset_phase_1 = random.random()*2*pi # use this phase offset for mw pulses because they don't have a fixed phase relationship to gate lasers
            self.phi_sum_1 = random.random()*2*pi
        else:
            self.mw_offset_phase_1 = 0
            self.phi_sum_1 = 0
        self.phi_diff_1 = 0
        
        self.set_ion_parameters()
        
        self.rho_target =  1/2*(self.uu_uu+self.dd_dd+1j*self.ud_ud-1j*self.du_du)
        
    
    def set_ion_parameters(self):
        super().set_ion_parameters()
        self.set_relative_Rabi_frequencies()
        self.set_phases()
        
    def set_relative_Rabi_frequencies(self):
        self.Omega_R_1 = (1-self.species_Rabi_asym_MS)*self.Omega_R
        if self.mixed_species:
            self.Omega_R_2 = (1+self.species_Rabi_asym_MS)*self.Omega_R_2 #TODO, check what this is set to currently
        else:
            self.Omega_R_2 = (1+self.species_Rabi_asym_MS)*self.Omega_R
            
    def set_phases(self):
        if self.mixed_species:
            self.phi_sum_2 = random.random()*2*pi
            self.phi_diff_2 = self.phi_diff_1 #random.random()*2*pi # TODO
            self.mw_offset_phase_2 = random.random()*2*pi
        else:
            self.phi_sum_2 = self.phi_sum_1
            self.phi_diff_2 = self.phi_diff_1
            self.mw_offset_phase_2 = self.mw_offset_phase_1
            
    
    def ms_force_asym(self,rho_in,times,phi_offset=0,scnd_loop=False):
        
        if scnd_loop:
            phi_comp_scnd_sb = self.phi_comp_scnd_sb/360*2*pi
        else:
            phi_comp_scnd_sb = 0
        
        # ms hamiltonian, time independent parts
        H1  = -1j/2*self.ad*(self.Omega_R_1*self.eta_1*(1-self.ampl_asym_1)*exp(1j*(self.phi_diff_1+self.phi_sum_1+phi_comp_scnd_sb))*self.sp_id+
                             self.Omega_R_2*self.eta_2*(1-self.ampl_asym_2)*exp(1j*(self.phi_diff_2+self.phi_sum_2+phi_comp_scnd_sb))*self.id_sp) # Omega_plus
        H1b =  1j/2*self.ad*(self.Omega_R_1*self.eta_1*(1+self.ampl_asym_1)*exp(1j*(self.phi_diff_1-self.phi_sum_1))*self.sm_id+
                             self.Omega_R_2*self.eta_2*(1+self.ampl_asym_2)*exp(1j*(self.phi_diff_2-self.phi_sum_2))*self.id_sm) # Omega_minus
        H2  =  1j/2*self.a *(self.Omega_R_1*self.eta_1*(1-self.ampl_asym_1)*exp(-1j*(self.phi_diff_1+self.phi_sum_1+phi_comp_scnd_sb))*self.sm_id+
                             self.Omega_R_2*self.eta_2*(1-self.ampl_asym_2)*exp(-1j*(self.phi_diff_2+self.phi_sum_2+phi_comp_scnd_sb))*self.id_sm) # Omega_plus
        H2b = -1j/2*self.a *(self.Omega_R_1*self.eta_1*(1+self.ampl_asym_1)*exp(-1j*(self.phi_diff_1-self.phi_sum_1))*self.sp_id+
                             self.Omega_R_2*self.eta_2*(1+self.ampl_asym_2)*exp(-1j*(self.phi_diff_2-self.phi_sum_2))*self.id_sp) # Omega_minus

#        # simplified Hamiltonian, without amplitude asymmetry:
#        # ms hamiltonian, time independent parts
#        H1 = 1/2*self.ad*(self.Omega_R_1*self.eta_1*self.sig_phi_i(1)+self.Omega_R_2*self.eta_2*self.sig_phi_i(2))
#        H2 = 1/2*self.a*(self.Omega_R_1*self.eta_1*self.sig_phi_i(1)+self.Omega_R_2*self.eta_2*self.sig_phi_i(2))
#        # wobble hamiltonian, time dependent parts
#        def H1_coeff(t,args):
#            return exp(-1j*((self.delta_g)*t+phi_offset))
#        def H2_coeff(t,args):
#            return exp(1j*((self.delta_g)*t+phi_offset))
#        # combine
#        H_ms = [[H1,H1_coeff],[H2,H2_coeff]]
        
        
        # MS hamiltonian, time dependent parts
        def H1_coeff(t,args):
            return exp(-1j*((self.delta_g+self.delta_LS)*t-phi_offset))
        def H2_coeff(t,args):
            return exp( 1j*((self.delta_g+self.delta_LS)*t-phi_offset))
        def H1b_coeff(t,args):
            return exp(-1j*((self.delta_g-self.delta_LS)*t-phi_offset))
        def H2b_coeff(t,args):
            return exp( 1j*((self.delta_g-self.delta_LS)*t-phi_offset))
        # combine
        H_ms = [[H1,H1_coeff],[H2,H2_coeff],[H1b,H1b_coeff],[H2b,H2b_coeff]]
        # decay operators
        c_ops_wobble = []
        if self.n_dot is not 0:
            c_ops_wobble.append(sqrt(self.n_dot)*self.a)
            c_ops_wobble.append(sqrt(self.n_dot)*self.ad)
        if self.tau_mot is not 0:
            c_ops_wobble.append(sqrt(2/self.tau_mot)*self.ad*self.a)
        if self.gamma_el is not 0:
            c_ops_wobble.append(sqrt(self.gamma_el/4)*(tensor(self.sz,qeye(2),qeye(self.nHO))))
            c_ops_wobble.append(sqrt(self.gamma_el/4)*(tensor(qeye(2),self.sz,qeye(self.nHO))))
        if self.gamma_ram is not 0:
            c_ops_wobble.append(sqrt(self.gamma_ram)*tensor(sigmap(),qeye(2),qeye(self.nHO)))
            c_ops_wobble.append(sqrt(self.gamma_ram)*tensor(qeye(2),sigmap(),qeye(self.nHO)))
            c_ops_wobble.append(sqrt(self.gamma_ram)*tensor(sigmam(),qeye(2),qeye(self.nHO)))
            c_ops_wobble.append(sqrt(self.gamma_ram)*tensor(qeye(2),sigmam(),qeye(self.nHO)))
        if self.tau_spin is not 0:
            c_ops_wobble.append(sqrt(1/self.tau_spin/2)*tensor(self.sz,qeye(2),qeye(self.nHO)))
            c_ops_wobble.append(sqrt(1/self.tau_spin/2)*tensor(qeye(2),self.sz,qeye(self.nHO)))
        # integrate Hamiltonian
        after_ms = mesolve(H_ms, rho_in, times, c_ops_wobble, [])
        return after_ms
    
    def do_gate(self,nT=2):
        if self.two_loops:
            times = np.linspace(0, self.T/2, nT)
        else:
            times = np.linspace(0, self.T, nT)
        # initialize result vector
        final_rhos = []
        # GATE
        # -------------------------------------------------------------------------
        # initial state
        rhoInitial = tensor(self.dn_dn,self.dn_dn,thermal_dm(self.nHO,self.nbar_mode))
        if self.phase_insensitive:
            rho_after_uw_piB2 = self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=self.mw_offset_phase_1)*\
                                self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=self.mw_offset_phase_2)*\
                                rhoInitial* \
                                self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=self.mw_offset_phase_1).dag() *\
                                self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=self.mw_offset_phase_2).dag()
            rho_after_laser_piB2 = self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1)* \
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2)* \
                                   rho_after_uw_piB2* \
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1).dag() *\
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2).dag()
            rho_before_MS_interaction = rho_after_laser_piB2
        elif self.ms_ls_exp:
            rho_after_piB2 = self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1)* \
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2)* \
                                   rhoInitial* \
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1).dag() *\
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2).dag()
            rho_before_MS_interaction = rho_after_piB2
        else:
            rho_before_MS_interaction = rhoInitial
        # do gate
        after_ms = self.ms_force_asym(rho_before_MS_interaction, times)
        #after_ms = self.ms_force(rho_before_MS_interaction, times)
        # finish Ramsey interferometer/ do second part of gate
        for ii in range(len(times)):
            rho_t = after_ms.states[ii]
            if self.two_loops:
                if self.do_center_pi:
                    rho_after_pi =  self.U_rot(pi*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1)*\
                                    self.U_rot(pi*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2)*\
                                    rho_t*\
                                    self.U_rot(pi*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1).dag() *\
                                    self.U_rot(pi*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2).dag()
                else:
                    rho_after_pi = rho_t
                if self.do_Walsh:
                    phi = pi+self.second_loop_phase_error+self.delta_LS*times[-1]#delta_g*times[ii]+pi
                else:
                    phi = 0+self.second_loop_phase_error+self.delta_LS*times[-1]
                after_second_loop = self.ms_force_asym(rho_after_pi, [0,times[ii]], phi_offset=phi, scnd_loop=True)
                #after_second_loop = self.ms_force(rho_after_pi, [0,times[ii]], phi_offset=phi)
                if self.phase_insensitive:
                    rho_after_2nd_laser_piB2 =  self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1)* \
                                                self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2)* \
                                                after_second_loop.states[-1]* \
                                                self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1).dag() *\
                                                self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2).dag()
                    rho_after_2nd_uw_piB2 = self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=self.mw_offset_phase_1)*\
                                            self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=self.mw_offset_phase_2)*\
                                            rho_after_2nd_laser_piB2* \
                                            self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=self.mw_offset_phase_1).dag() *\
                                            self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=self.mw_offset_phase_2).dag()
                    final_rho = rho_after_2nd_uw_piB2
                elif self.ms_ls_exp:
                    rho_after_2nd_piB2 = self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1)* \
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2)* \
                                   after_second_loop.states[-1]* \
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1).dag() *\
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2).dag()
                    final_rho = rho_after_2nd_piB2
                else:
                    final_rho = after_second_loop.states[-1]
            else: # only one loop
                #rho_final = self.U_rot(pi/2)*self.U_rot(pi)*rho_t*self.U_rot(pi).dag()*self.U_rot(pi/2).dag()
                if self.phase_insensitive:
                    rho_after_2nd_laser_piB2 =  self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=pi-self.phi_sum_1)* \
                                                self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=pi-self.phi_sum_2)* \
                                                rho_t* \
                                                self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=pi-self.phi_sum_1).dag() *\
                                                self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=pi-self.phi_sum_2).dag()
                    rho_after_2nd_uw_piB2 = self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=self.mw_offset_phase_1)*\
                                            self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=self.mw_offset_phase_2)*\
                                            rho_after_2nd_laser_piB2* \
                                            self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=self.mw_offset_phase_1).dag() *\
                                            self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=self.mw_offset_phase_2).dag()
                    final_rho = rho_after_2nd_uw_piB2
                elif self.ms_ls_exp:
                    rho_after_2nd_piB2 = self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1+pi/2)* \
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2+pi/2)* \
                                   rho_t* \
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[1,0],phi=-self.phi_sum_1+pi/2).dag() *\
                                   self.U_rot(pi/2*self.sq_factor,ion_index=[0,1],phi=-self.phi_sum_2+pi/2).dag()
                    final_rho = rho_after_2nd_piB2
                else:
                    final_rho = rho_t
            # final results
#            final_rhos.append(rho_final)
            final_rhos.append(final_rho)
        if self.two_loops:
            times *= 2
        return times, final_rhos
    
        
    def set_ampl_asym(self,ampl_asym, ampl_asym_2=None):
        self.ampl_asym_1 = ampl_asym
        if ampl_asym_2 is None:
            self.ampl_asym_2 = ampl_asym
        else:
            self.ampl_asym_2 = ampl_asym_2
            
    def scan_ampl_asym(self,ampl_asym_max=-0.1,ampl_asym_min=0.1,n_steps=10,**kwargs):
        # simulate gate fidelity for amplitude asymmetries
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        
        errors = []
        ampl_asyms = np.linspace(ampl_asym_min, ampl_asym_max, n_steps)

        for ampl_asym in ampl_asyms:
            self.set_ampl_asym(ampl_asym)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(self.rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            
        return ampl_asyms, errors
    
    def scan_LS(self,LS_max=-0.1,LS_min=0.1,n_steps=10,**kwargs):
        # simulate gate fidelity for amplitude asymmetries
        # initialize result vectors
        self.set_custom_parameters(**kwargs)
        
        errors = []
        pops = []
        
        LS_vec = np.linspace(LS_min, LS_max, n_steps)

        for LS in LS_vec:
            self.set_delta_LS(LS)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(self.rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            pops.append(ptrace(final_rhos[-1],[0,1])[0,0]+ptrace(final_rhos[-1],[0,1])[3,3])
            
        return LS_vec, errors, pops
    
        
        
    def calc_ideal_Rabi_freq(self,verbose=True):
        if self.two_loops:
            K = 2
        else:
            K = 1
        if self.mixed_species:
            OR_1 = self.delta_g/sqrt(K)/(2*self.eta_1)
            OR_2 = self.delta_g/sqrt(K)/(2*self.eta_2)
            if verbose:
                print('Calculated ideal Rabi frequency ion 1: 2*pi*{:.2f}kHz'.format(OR_1/1e3/(2*pi)))
                print('Calculated ideal Rabi frequency ion 2: 2*pi*{:.2f}kHz'.format(OR_2/1e3/(2*pi)))
        else:
            OR = self.delta_g/sqrt(K)/(2*self.eta_1)
            if verbose:
                print('Calculated ideal Rabi frequency: 2*pi*{:.2f}kHz'.format(OR/1e3/(2*pi)))
            OR_1 = OR
            OR_2 = OR
        return OR_1, OR_2
    

    
    