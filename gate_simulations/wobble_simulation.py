#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VMS Dec 2017

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, pi, sqrt, cos, sin
from qutip import qeye, tensor, destroy, fock_dm, rx, ry, mesolve, thermal_dm, expect, fidelity, ptrace, sigmam, sigmap, sigmax, sigmay
import pylab as pyl
from scipy import constants as u
from gate_simulations.tqg_simulation import Tqg_simulation

class Wobble_simulation(Tqg_simulation):

    def __init__(self,Delta_ca=-2*pi*10e12,Delta_sr=2*pi*30e12,**kwargs):  
        ''' 
        nHO: dimension of HO space which is simulated
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
        ! Note that most gate parameters are class objects, and therefore parameters
        are changed by previously run scans !
        '''
        
        self.Delta_ca = Delta_ca # detuning from 397 transition
        self.Delta_sr = Delta_sr # detuning from 422 transition!! (not 408, would require different matrix elements)

        super().__init__(**kwargs)

        
    
    def set_ion_parameters(self):
        super().set_ion_parameters() #TODO: include Raman detuning into eta for mixed species
        self.set_relative_Rabi_frequencies()
        
    def set_relative_Rabi_frequencies(self):
        if self.species is '4040':
            self.Omega_R_up_1 = self.Omega_R
            self.Omega_R_up_2 = self.Omega_R*self.ion_spacing*self.mode
            self.Omega_R_dn_1 = - self.Omega_R
            self.Omega_R_dn_2 = - self.Omega_R*self.ion_spacing*self.mode
        elif self.species is '4343':
            self.Omega_R_up_1 = self.Omega_R*(6/8)
            self.Omega_R_up_2 = self.Omega_R*self.ion_spacing*self.mode*(6/8)
            self.Omega_R_dn_1 = - self.Omega_R
            self.Omega_R_dn_2 = - self.Omega_R*self.ion_spacing*self.mode
        elif self.species is '4043':
            self.Omega_R_up_1 = self.Omega_R*(6/8)
            self.Omega_R_up_2 = self.Omega_R*self.ion_spacing*self.mode
            self.Omega_R_dn_1 = - self.Omega_R
            self.Omega_R_dn_2 = - self.Omega_R*self.ion_spacing*self.mode        
        elif self.species is '4388':
            omega_f_ca = 2*pi*6.68e12
            omega_f_sr = 2*pi*24.027e12
            Delta_ca_0 = -2*pi*9e12
            det_scal = omega_f_ca/(Delta_ca_0*(Delta_ca_0-omega_f_ca)) # detuning_scaling_normalisation, bc this is included in omega_rabi already
            self.Delta_ca = Delta_ca_0
            self.Delta_sr = self.sr_detuning()+Delta_ca_0
            self.Omega_R_up_1 = self.Omega_R*(6/8)*omega_f_ca/(self.Delta_ca*(self.Delta_ca-omega_f_ca))/det_scal
            self.Omega_R_up_2 = self.Omega_R*self.ion_spacing*self.mode*omega_f_sr/(self.Delta_sr*(self.Delta_sr-omega_f_sr))/det_scal
            self.Omega_R_dn_1 = - self.Omega_R*omega_f_ca/(self.Delta_ca*(self.Delta_ca-omega_f_ca))/det_scal
            self.Omega_R_dn_2 = - self.Omega_R*self.ion_spacing*self.mode*omega_f_sr/(self.Delta_sr*(self.Delta_sr-omega_f_sr))/det_scal
        elif self.species is 'effic_test':
            # assuming Delta is equal for Ca and Sr and also omega_f (which isn't true of course, but probably a good enough approximation for now)
            self.Omega_R_up_1 = self.Omega_R*self.factor
            self.Omega_R_up_2 = self.Omega_R*self.ion_spacing*self.mode/self.factor
            self.Omega_R_dn_1 = - self.Omega_R*self.factor
            self.Omega_R_dn_2 = - self.Omega_R*self.ion_spacing*self.mode/self.factor
        else:
            raise NameError('Invalid ion species')
        #self.Omega_rabi_scale = (abs(self.Omega_R_up_1*self.eta_1)+abs(self.Omega_R_up_2*self.eta_2)+abs(self.Omega_R_dn_1*self.eta_1)+abs(self.Omega_R_dn_2*self.eta_2))/(4*self.Omega_R*(self.eta_1+self.eta_2)/2)
        self.Omega_rabi_scale = np.sqrt(abs((self.Omega_R_up_1*self.eta_1+self.ion_spacing*self.mode*self.Omega_R_up_2*self.eta_2)**2+
                                         (self.Omega_R_dn_1*self.eta_1+self.ion_spacing*self.mode*self.Omega_R_dn_2*self.eta_2)**2-
                                         (self.Omega_R_up_1*self.eta_1+self.ion_spacing*self.mode*self.Omega_R_dn_2*self.eta_2)**2-
                                         (self.Omega_R_dn_1*self.eta_1+self.ion_spacing*self.mode*self.Omega_R_up_2*self.eta_2)**2
                                        )/(2*(self.Omega_R)**2))
        
    def sr_detuning(self):
        lambda_397 = 396.9589788e-9
        lambda_422 = 421.6712e-9
        #lambda_408 = 407.8865e-9
        delta_offset_sr = 2*np.pi*((u.c/lambda_397)-(u.c/lambda_422))
        return delta_offset_sr

    def wobble_force(self,rho_in,times,phi_0=0):
        # wobble hamiltonian, time independent parts
        H1 = -1j/2*self.ad*(self.Omega_R_up_1*self.eta_1*self.uu_id+self.Omega_R_up_2*self.eta_2*self.id_uu+
                     self.Omega_R_dn_1*self.eta_1*self.dd_id+self.Omega_R_dn_2*self.eta_2*self.id_dd)
        H2 = 1j/2*self.a*(self.Omega_R_up_1*self.eta_1*self.uu_id+self.Omega_R_up_2*self.eta_2*self.id_uu+
                   self.Omega_R_dn_1*self.eta_1*self.dd_id+self.Omega_R_dn_2*self.eta_2*self.id_dd)
        # wobble hamiltonian, time dependent parts
        def H1_coeff(t,args):
            return exp(-1j*(self.delta_g*t+phi_0))
        def H2_coeff(t,args):
            return exp(1j*(self.delta_g*t+phi_0))
        # combine
        H_wobble = [[H1,H1_coeff],[H2,H2_coeff]]
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
        after_wobble = mesolve(H_wobble, rho_in, times, c_ops_wobble, [])
        return after_wobble
    
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
        # apply pi/2
        rho_after_piB2 = self.U_rot(pi/2*self.sq_factor)*rhoInitial*self.U_rot(pi/2*self.sq_factor).dag()
        # do gate
        after_wobble = self.wobble_force(rho_after_piB2, times)
        # finish Ramsey interferometer/ do second part of gate
        for ii in range(len(times)):
            rho_t = after_wobble.states[ii]
            if self.two_loops:
                rho_after_pi = self.U_rot(pi*self.sq_factor)*rho_t*self.U_rot(pi*self.sq_factor).dag()
                phi = pi#delta_g*times[ii]+pi
                after_wobble_two = self.wobble_force(rho_after_pi, [0,times[ii]], phi_0=phi)
                rho_final = self.U_rot(pi/2*self.sq_factor)*after_wobble_two.states[-1]*self.U_rot(pi/2*self.sq_factor).dag()
            else: # only one loop
                rho_final = self.U_rot(pi/2*self.sq_factor)*rho_t*self.U_rot(pi/2*self.sq_factor).dag()
            # final results
            final_rhos.append(rho_final)
        if self.two_loops:
            times *= 2
        return times, final_rhos
        
        
    def set_eff_factor(self,factor):
        self.factor = factor
        self.set_relative_Rabi_frequencies()
        
    def calc_gate_eff(self):
        odd_phi = (self.Omega_R_up_1*self.eta_1+self.Omega_R_dn_2*self.eta_2)**2+\
                (self.Omega_R_dn_1*self.eta_1+self.Omega_R_up_2*self.eta_2)**2
        even_phi = (self.Omega_R_up_1*self.eta_1+self.Omega_R_up_2*self.eta_2)**2+\
                (self.Omega_R_dn_1*self.eta_1+self.Omega_R_dn_2*self.eta_2)**2
        eff = (odd_phi-even_phi)/(odd_phi+even_phi)
        return eff
        
    def calc_ideal_Rabi_freq(self,verbose=True):
        if self.two_loops:
            K = 2
        else:
            K = 1
        #OR = self.delta_g/(2*self.eta_1)/1e3/(2*pi)/sqrt(K)
        OR = self.delta_g/sqrt(K)/self.Omega_rabi_scale
        if verbose:
            print('Calculated ideal Rabi frequency: {:.2f}kHz'.format(OR/1e3/(2*pi)))
        return OR, OR
    
    def parity_scan(self,analysis_phase,**kwargs):
        end_populations_upup, end_populations_dndn, end_populations_updn_dnup = super().parity_scan(analysis_phase,is_wobble_gate=True,**kwargs)
        
        return end_populations_upup, end_populations_dndn, end_populations_updn_dnup#, fidelities
    
    def scan_efficiency(self,ndot=0,fact_vec=[1,2,3,4,5],tau_mot=0,gamma_el=0,gamma_ram=0,tau_spin=0):
        
        if self.species is not 'effic_test':
            raise TypeError('Need species effic_test for this to work (factor not implemented for other ions species)')
        
        # initialize result vectors
        errors = []
        efficiencies = []
        rho_target = 1/2*(self.uu_uu+self.dd_dd-1j*self.ud_ud+1j*self.du_du)
        self.set_mot_dephasing_time(tau_mot)
        self.set_el_deph_rate(gamma_el)
        self.set_ram_scat_rate(gamma_ram)
        self.set_spin_dephasing_time(tau_spin)

        for factor in fact_vec:
            self.set_eff_factor(factor)
            self.set_heating_rate(ndot)
            times, final_rhos = self.do_gate()
            errors.append(1-fidelity(rho_target,ptrace(final_rhos[-1],[0,1]))**2)
            efficiencies.append(self.calc_gate_eff())
            
        return fact_vec, errors, efficiencies



    
    
    
    
    