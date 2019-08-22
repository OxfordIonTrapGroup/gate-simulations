#from IPython import get_ipython
#get_ipython().magic('reset -sf')
from gate_simulations.wobble_simulation import Wobble_simulation
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, pi, sqrt

#%%
### gate initialisation

wobble_1 = Wobble_simulation(nHO=15,nbar_mode=0.0,delta_g=2*pi*40e3,
                 two_loops=True,species='4388',ion_spacing=-1,mode=1)
#wobble_1 = Wobble_simulation(nHO=15,nbar_mode=0.1,delta_g=2*pi*40e3,
#                 two_loops=True,species='effic_test',ion_spacing=-1,mode=1,
#                 factor=0.5,tau_spin=9e-3,sq_factor=1.0)


#%%
## Time scans


# example with correct parameters
times_2, end_pop_upup_2, end_pop_dndn_2, end_pop_updn_2, end_pop_dnup_2, fidelities_2 = \
    wobble_1.timescan(T=200e-6,nT=100)

## example with mis-set parameters
#times_2, end_pop_upup_2, end_pop_dndn_2, end_pop_updn_2, end_pop_dnup_2, fidelities_2 = \
#    wobble_1.timescan(T=200e-6,nT=100,Omega_R=2*pi*100e3)
    
rabi_freq, rf = wobble_1.calc_ideal_Rabi_freq()
    
wobble_1.calc_ideal_gate_time()

fig1, ax1 = plt.subplots()
ax1.plot(times_2/1e-6, end_pop_upup_2)
ax1.plot(times_2/1e-6, end_pop_updn_2)
ax1.plot(times_2/1e-6, end_pop_dndn_2)
ax1.plot(times_2/1e-6, fidelities_2)
ax1.plot(times_2/1e-6, end_pop_dnup_2,'-.')
plt.grid(True,which="both",ls=":", color='0.7')
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=5)
ax1.set_xlabel('$t_{g,tot}=2\cdot t_g$')

#%%
### parity scan
#phi_vec = np.linspace(0,2*pi,100)
#end_pop_upup, end_pop_dndn, end_pop_updn_dnup = wobble_1.parity_scan(phi_vec)
#fig1, ax1 = plt.subplots()
#ax1.plot(phi_vec, end_pop_upup,label='p11')
#ax1.plot(phi_vec, end_pop_dndn,'-.',label='p00')
#ax1.plot(phi_vec, np.array(end_pop_updn_dnup),label='p10+p01')
#plt.grid(True,which="both",ls=":", color='0.7')
#plt.locator_params(axis='y', nbins=5)
#plt.locator_params(axis='x', nbins=5)
#ax1.set_ylim([0, 1])
#plt.legend(loc="upper left", numpoints=1)

#%%
## heating rate scan
#n_dots, errors = wobble_1.scan_heating_rate(ndot_max=1e3,n_steps=10)
#
#fig2, ax2 = plt.subplots()
#ax2.plot(n_dots, errors)
#plt.grid(True,which="both",ls=":", color='0.7')
#plt.locator_params(axis='y', nbins=5)
#plt.locator_params(axis='x', nbins=5)
#ax2.set_xlabel('heating rate (quanta/s)')

#%%
## gate efficiency scan
#
#factor_vec = np.linspace(1, 4, 30)
#fact_vec, errors, efficiencies = wobble_1.scan_efficiency(ndot=1e2,fact_vec=factor_vec)
#
#fig2, ax2 = plt.subplots()
#ax2.plot(fact_vec, errors)
#plt.grid(True,which="both",ls=":", color='0.7')
#plt.locator_params(axis='y', nbins=5)
#plt.locator_params(axis='x', nbins=5)
#ax2.set_xlabel('factor')
#ax2.set_ylabel('gate error')
#
#errors_scaled = [er*(ef)/errors[0] for er,ef in zip(errors,efficiencies)]
#
#fig2, ax2 = plt.subplots()
#ax2.plot(fact_vec, errors_scaled)
#plt.grid(True,which="both",ls=":", color='0.7')
#plt.locator_params(axis='y', nbins=5)
#plt.locator_params(axis='x', nbins=5)
#ax2.set_xlabel('factor')
#ax2.set_ylabel('efficiency-scaled error')
#
#errors_norm = [er/errors[0] for er in errors]
#
#def fit_func(x):
#    y = [(1/xx)*errors[0] for xx in x]
#    return y
#
#def fit_func_2(x):
#    y = [2*xx/(1+xx**2) for xx in x]
#    return y
#
#
#fact_vec_real = [x**2 for x in fact_vec]
#
#fig2, ax2 = plt.subplots()
#ax2.plot(fact_vec_real,efficiencies)
#ax2.plot(fact_vec_real, fit_func_2(fact_vec_real),'-.')
#plt.grid(True,which="both",ls=":", color='0.7')
#plt.locator_params(axis='y', nbins=5)
#plt.locator_params(axis='x', nbins=5)
#ax2.set_xlabel('factor')
#ax2.set_ylabel('efficiency')