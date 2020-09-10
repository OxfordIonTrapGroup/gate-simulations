from gate_simulations.mode_frequencies import Mode_frequencies
from gate_simulations.lamb_dicke_factor import Lamb_Dicke_Factor

class Mode():
    def __init__(self,name='ax_ip',freq=2e6,nbar=0.05,eta=0.1,eta_2=-1):
        self.name = name
        self.freq = freq
        self.nbar = nbar
        self.eta = eta
        self.eta_2 = (eta if eta_2 == -1 else eta_2)
        
class Mode_structure():
    def __init__(self,single_ion_freqs=None,nbars=None,species='8888',N=2, raman_misalignment=0,qudpl_and_raman=False):
        # TODO: so far only works for two ions
        self.species = species
        self.n_ions = N
        self.mf = Mode_frequencies(single_ion_freqs=single_ion_freqs,species=self.species)
        self.ld = Lamb_Dicke_Factor(species=species,epsilon=self.mf.epsilon,
                                    raman_misalignment=raman_misalignment,qudpl_and_raman=qudpl_and_raman)
        
        if self.n_ions == 1:
            self.mode_names = ['ax','rad_l','rad_u']
        elif self.n_ions == 2:
            self.mode_names = ['ax_ip','ax_oop','rad_ip_l','rad_ip_u','rad_oop_l','rad_oop_u']
        
        self._setup_modes(nbars)
        
    def _setup_modes(self,nbars):
        modes = []
        if nbars == None:
            nbars = [0.05,0.05,0.05,0.05,0.05,0.05]
        freq_vec = [self.mf.freqs[mode] for mode in self.mode_names]
        eta_vec = self._calc_ld_factors(freq_vec,ion_index=1)
        eta_2_vec = self._calc_ld_factors(freq_vec,ion_index=2)
        for eta,eta_2,nbar,freq,name in zip(eta_vec,eta_2_vec,nbars,freq_vec,self.mode_names):
            modes.append(Mode(name=name,freq=freq,nbar=nbar,eta=eta,eta_2=eta_2))
        self.modes = {x.name: x for x in modes}
        
    def set_mode_nbar(self,mode_name,nbar):
        self.modes[mode_name].nbar = nbar
        
    def set_raman_misalignement(self,raman_misalignement):
        self.ld.set_raman_misalignement(raman_misalignement)
        freq_vec = [self.mf.freqs[mode] for mode in self.mode_names]
        self._update_modes(freq_vec)
        
    def _update_modes(self,freq_vec):
        eta_vec = self._calc_ld_factors(freq_vec,ion_index=1)
        eta_2_vec = self._calc_ld_factors(freq_vec,ion_index=2)
        self.ld.set_epsilon(self.mf.epsilon) # this doesn't do anything, bc epsilon not update at the moment
        for eta,eta_2,name,freq in zip(eta_vec,eta_2_vec,self.mode_names,freq_vec):
            self.modes[name].freq = freq
            self.modes[name].eta = eta
            self.modes[name].eta_2 = eta_2
        
    def _calc_ld_factors(self,freq_vec,ion_index=1):
        eta_vec = [self.ld.calc_lamb_dicke(mode_freq,mode_name,ion_ind=ion_index) for mode_freq,mode_name in zip(freq_vec,self.mode_names)]
        return eta_vec
        
    def set_frequencies(self,ax_freq,rad_freq_l,rad_freq_u):
        self.mf.set_frequencies(ax_freq,rad_freq_l,rad_freq_u)
        freq_vec = [self.mf.freqs[mode] for mode in self.mode_names]
        self._update_modes(freq_vec)
        
    def set_frequencies_manually(self,ax_freq_ip,ax_freq_oop,rad_freq_ip_l,rad_freq_ip_u,rad_freq_oop_l,rad_freq_oop_u):
        self.mf.set_frequencies_manually(ax_freq_ip,ax_freq_oop,rad_freq_ip_l,rad_freq_ip_u,rad_freq_oop_l,rad_freq_oop_u)
        freq_vec = [self.mf.freqs[mode] for mode in self.mode_names]
        self._update_modes(freq_vec)