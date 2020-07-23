#from IPython import get_ipython
#get_ipython().magic('reset -sf')
from gate_simulations.ms_simulation import Ms_simulation
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, pi, sqrt


def off_res_coupling(omega_rabi,delta_g):
    epsilon = omega_rabi**2/(delta_g**2)
    return epsilon

    
#%%
## gate initialisation

## single loop
#two_loops = False
#delta_g_0 =2*pi*60e3
#Omega_R_0 =2*pi*242.5e3
#T_0 = 67e-6
#omega_z = 2*pi*1.924e6
#mode = 1
#species ='4343'
#phase_insensitive = True    
#delta_LS_0 = 0
#qudpl_and_raman = False
    
## single loop
#two_loops = False
#delta_g_0 =2*pi*60e3
#Omega_R_0 =2*pi*319.15e3
#T_0 = 67e-6
#omega_z = 2*pi*1.924e6
#mode = -1
#species ='4343'
#phase_insensitive = True    
#delta_LS_0 = 0
#qudpl_and_raman = False
    
## two loops
#two_loops = True
#delta_g_0=2*pi*30e3
#Omega_R_0=2*pi*348.03e3
#T_0 = 262e-6
#omega_z = 2*pi*1.3424e6
#mode = 1
#species='8888'
#phase_insensitive=True    
#delta_LS_0=0
#
#ms_1 = Ms_simulation(nHO=15,nbar_mode=0.0,delta_g=delta_g_0,Omega_R=Omega_R_0,
#                 two_loops=two_loops,species='4388',ion_spacing=-1,mode=mode,tau_spin=100e-3,
#                 phase_insensitive=True)

#%%
## gate initialisation

# Ca - Sr
two_loops = True
delta_g_0=2*pi*30e3
Omega_R_0=2*pi*119e3
Omega_R_2=2*pi*293.38e3
T_0 = 262e-6
omega_z = 2*pi*1.3424e6
mode = 1
species='4388'
phase_insensitive = False
delta_LS_0=0
qudpl_and_raman = True
ampl_asym_1 = 0.0
species_Rabi_asym_MS = 0.0
nbar_mode = 0.0


ms_1 = Ms_simulation(nHO=15,nbar_mode=nbar_mode,delta_g=delta_g_0,Omega_R=Omega_R_0,
                     Omega_R_2=Omega_R_2,two_loops=two_loops,species=species,
                     ion_spacing=-1,mode=mode,tau_spin=100e-3, 
                     phase_insensitive=phase_insensitive, qudpl_and_raman=qudpl_and_raman)


or_calculated = ms_1.calc_ideal_Rabi_freq()

#%% 

## time scan
# fidelity calculation in MS timescan not correct because coherences depend on randomly chosen phi, 
# target state doesn't track phase, populations are still fine
times, end_pop_upup, end_pop_dndn, end_pop_updn, end_pop_dnup, fidelities = ms_1.timescan(
        T=T_0,nT=100,delta_LS=delta_LS_0,ampl_asym=ampl_asym_1,ampl_asym_2=0,ndot=0,species_Rabi_asym_MS=species_Rabi_asym_MS)
    
fig, ax = plt.subplots()
ax.plot(times/1e-6, end_pop_upup,label='p11')
ax.plot(times/1e-6, end_pop_updn,label='p10')
ax.plot(times/1e-6, end_pop_dndn,label='p00')
ax.plot(times/1e-6, end_pop_dnup,label='p01')
plt.grid(True,which="both",ls=":", color='0.7')
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=5)
plt.legend(loc="upper left", numpoints=1)


#%%

## parity scan
#
#phi_vec = np.linspace(0,pi,100)
#
#end_pop_upup, end_pop_dndn, end_pop_updn_dnup = ms_1.parity_scan(phi_vec,
#        delta_LS=delta_LS_0,ampl_asym=0.0,
#        scramble_mw_phase=False,fixed_analysis_phase=True,scramble_laser_phase=False)
## faster with scramble_mw_phase=False
#    
#fig1, ax1 = plt.subplots()
#ax1.plot(phi_vec, end_pop_upup,label='p11')
#ax1.plot(phi_vec, end_pop_updn_dnup,label='p10_01')
#ax1.plot(phi_vec, end_pop_dndn,label='p00')
##ax.plot(detuning_vec/(2*pi)/1e3, fidelities)
#plt.grid(True,which="both",ls=":", color='0.7')
#plt.locator_params(axis='y', nbins=5)
#plt.locator_params(axis='x', nbins=5)
#ax1.set_ylim([0, 1])
#plt.legend(loc="upper left", numpoints=1)

##%%
## detuning scan
#
#delta_range = 2*pi*40e3
##detuning_vec = np.linspace(delta_g_0-delta_range,delta_g_0+delta_range,100)
#detuning_vec = np.linspace(-delta_range,delta_range,100)
#
#end_pop_upup, end_pop_dndn, end_pop_updn, end_pop_dnup, fidelities = ms_1.detuning_scan(detuning_vec,
#        T=T_0/4,Omega_R=Omega_R_0,delta_LS=delta_LS_0,ampl_asym=0.0)
#    
#fig1, ax1 = plt.subplots()
#ax1.plot(detuning_vec/(2*pi)/1e3, end_pop_upup,label='p11')
#ax1.plot(detuning_vec/(2*pi)/1e3, end_pop_updn,label='p10')
#ax1.plot(detuning_vec/(2*pi)/1e3, end_pop_dnup,label='p01')
#ax1.plot(detuning_vec/(2*pi)/1e3, end_pop_dndn,label='p00')
##ax.plot(detuning_vec/(2*pi)/1e3, fidelities)
#plt.grid(True,which="both",ls=":", color='0.7')
#plt.locator_params(axis='y', nbins=5)
#plt.locator_params(axis='x', nbins=5)
#ax1.set_ylim([0, 1])
#plt.legend(loc="upper left", numpoints=1)
#
