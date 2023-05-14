from simulator import Simulator
from shells import Shell, Emitter
import matplotlib.pyplot as plt

"""
EXAMPLE SIMULATION FILE.

The purpose is to show how to create a simulation of the GRB prompt emission, and how various functions in the module are called.
"""

# create a central engine for the burst. This assumes a deceleration radius of 5.5*10^11 km, a distance from the innermost shell to the central engine of 1000 km, 1000 shells being emitted by the central engine, uptime of the central engine of 0.001 s, and a redshift of 2.
emitter = Emitter(5.5e+11, 1e+2, 1000, 1e-2, 2)
# emitter = Emitter(5.5e+11, 1e+6, 8000, 1e-2, 2)

# create a simulator object and input the central engine for the simulation
simulator = Simulator(emitter)

# setup the central engine with a first characteristic Lorentz factor of 500, following the profile of GRB 1 from Bustamante et al. (2017)
simulator.sim_emitter_setup(500, 0, 0, 1)

# plot the initial Lorentz factors of each shell
# simulator.plot_initial_lorentz_factors()

# start the simulation - everything is done for you in this function.
simulator.sim_start()

# plot the gamma-ray light curve of the prompt emission of the burst
simulator.plot_light_curve()

# plot the optical depth to Thomson scattering for each collision
simulator.plot_optical_depth()

simulator.plot_photosphere_radius_histogram()

# plot the proton maximum energy for each collision
simulator.plot_proton_maximum_energy()

# plot the final proton energy spectrum
simulator.plot_proton_energy_spectrum()

# save the proton spectrum to a .csv file
# simulator.save_to_csv_proto_spec()

# plot the proton energy loss timescales
simulator.plot_proton_energy_loss_timescales()

# show all plots
plt.show()
