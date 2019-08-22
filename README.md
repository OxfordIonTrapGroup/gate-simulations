# gate-simulations
Numerical siumlations for mixed species sigma-z and Molmer-Sorensen gates

This library contains code to numerically simulate sigma_z and Molmer-Sorensen gates
on (mixed-species) two-qubit trapped ion crystals.
The purpose of these simulations is to see how changes in gate parameters and error terms
affect gate dynamics and fidelities, as well as the role of phases in MS gates.
The set of errors included in these simulations is not complete, other significant 
errors can be for example Kerr-cross coupling or excitation of other modes.
All simulations are performed after the Lamb-Dicke and rotating wave approximations.
Corresponding theoretical derivations for the Hamiltonian and error terms can be found
in
https://www2.physics.ox.ac.uk/sites/default/files/page/2011/08/15/vmsthesis100119-44193.pdf
Example code of how to run the simulations, including notes about potential caveats
can be found in the scripts folder.
Many of the programmed in frequencies and beam geometries are specific for our experiment
and will have to be changed to fit to other experiments.
The library uses the QuTiP package, which can be downloaded and installed from
http://qutip.org/