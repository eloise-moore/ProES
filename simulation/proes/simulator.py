from shells import Shell, Emitter
import numpy as np
import matplotlib.pyplot as plt
import math
from astropy.cosmology import Planck18
import config
import os.path as path
from os.path import exists
import csv
import statistics
# import smplotlib

c = 3e+5

class Simulator():
    """
    A class to create a simulation of the GRB prompt emission.

    --------------------------------------------------------------------------------
                                Parameters contained:
    --------------------------------------------------------------------------------
    emitter:        an Emitter object to represent the central engine of the GRB.

    sim_time:       a list containing the timesteps within the simulation

    obs_time:       a list containing the time of shell collisions from an observer on Earth's frame of reference

    t_obs:          a list containing the time of shell collisions from an observer on Earth's frame of reference,
                    corrected for a time delay offset

    --------------------------------------------------------------------------------
    """
    def __init__(self, emitter):
        """
        Initialises the simulation

        Parameters:
            emitter:        an Emitter object to represent the central engine of the GRB.
        """
        self.emitter = emitter
        self.sim_time = [0]
        self.obs_time = []
        self.t_obs = []

        if config.ENABLE_LOGGING:
            if (os.path.exists(path.join(config.LOG_PATH, 'collision.log'))):
                os.remove(path.join(config.LOG_PATH, 'collision.log'))
                f = open(path.join(config.LOG_PATH, 'collision.log'), 'x')
                f.close()
            if (os.path.exists(path.join(config.LOG_PATH,'photons.log'))):
                os.remove(path.join(config.LOG_PATH,'photons.log'))
                f = open(path.join(config.LOG_PATH,'photons.log'), 'x')
                f.close()

            if (os.path.exists(path.join(config.LOG_PATH, 'protons.log'))):
                os.remove(path.join(config.LOG_PATH, 'protons.log'))
                f = open(path.join(config.LOG_PATH, 'protons.log'), 'x')
                f.close()

    def sim_emitter_setup(self, gamma1, gamma2, tp, dist):
        """
        Sets up the central engine of the burst, and sets up the shells emitted by the central engine - assigning indices, Lorentz factors, widths, and distances from central engine.

        Parameters:
            gamma1 (float): first characteristic Lorentz factor value

            gamma2 (float): second characteristic Lorentz factor value

            tp (float):     fraction of total emitted shells

            dist (int):     type of Lorentz factor distribution for shells. Possible inputs are
                            1 - profile for GRB1 of Bustamante et al. (2017) (DEFAULT DISTRIBUTION)
                            3 - profile for GRB3 of Bustamante et al. (2017)

        Returns:
            none
        """
        self.emitter.setup(gamma1, gamma2, tp, dist)

    def sim_start(self):
        """
        Starts the simulation. Each timestep is one shell collision. Within each timestep, the properties of the other shells in the central engine are updated (distance, volume, and density). Shell collisions result in a merged shell, which then propagates with the rest of the shells and may undergo subsequent collisions. The simulation ends if one of three conditions arise as is employed in Bustamante et al. (2017):

        * The shells of matter reach the circumbust medium. This signifies the end of the prompt emission phase as they shock into the interstellar medium (ISM) and produce the afterglow seen in all wavelengths.

        * The shells of matter have all merged into one large remaining shell. This means that no more collisions can occur, and signifies the end of the prompt emission phase.

        * The shells of matter are arranged such that the ordering of their Lorentz factors does not allow for any further collisions. Since fast shells cannot collide with slower shells, internal shocks can no longer occur, which signifies the end of the prompt emission phase due to lack of gamma-ray production.

        Output parameters from the simulation can be accessed via the plotting functions located within this class. If logs or console outputs are needed, these can be turned on or off in the config.py file under the ENABLE_LOGGING and VERBOSE variables respectively.

        Parameters:
            none

        Returns:
            none
        """
        st = 0
        lf = []
        while((self.emitter.nshells > 1) and (self.emitter.done == False)):

            # update simulation time to time of next collision
            st += self.emitter.next_collision_time
            print(self.emitter.nshells)
            # store simulation time
            self.sim_time.append(st)

            # update all shells in simulation
            for shell in self.emitter.shells:
                shell.update_shell(self.emitter.next_collision_time)

            # compute light travel time
            ltt = (Planck18.lookback_distance(self.emitter.z).to_value() * 3.08568e+24 / 3e+10) / (60 * 60 * 24 * 365)

            # compute observed time
            ot = (1 + self.emitter.z)*((((ltt * c) - self.emitter.collision_shells[0].radius) / c) + st)

            # let the emitter know what the simulation time is
            self.emitter.update_tobs(st)

            if config.VERBOSE:
                print("###############################################")
                print("                 SIM INFO")
                print("###############################################")
                print(f'sim time:   {st:.4e}')
                print(f'obs time:   {ot:.4e}')
                print(f'nshells:    {self.emitter.nshells}')
                print("###############################################")

            # create a shell collision and store whether or not it is subphotospheric
            res = self.emitter.collision(self.emitter.collision_shells[0], self.emitter.collision_shells[1])

            if (res == 0):
                if config.VERBOSE:
                    print("Super photospheric collision")
                self.obs_time.append(ot)
            else:
                if config.VERBOSE:
                    print("Subphotospheric collision")

            if config.VERBOSE:
                print(f"collisions occurred: {self.emitter.ncoll}")

            if self.emitter.nshells <=5:
                for i in self.emitter.shells:
                    lf.append(i.gamma)

            # calculate next collision time
            self.emitter.calculate_collision_times()

        # print(lf)
        # lf = [i.gamma for i in self.emitter.shells]
        print(f'Final Lorentz factor {statistics.mean(lf)}')

        offset = min(self.obs_time)
        self.t_obs = [i - offset for i in self.obs_time]
        t90 = self.emitter.calculate_t90(self.t_obs)
        tv = self.emitter.calculate_tv(t90)

        if config.VERBOSE:
            print()
            print("------------------------------------------------------------------")
            print("                     OUTPUT PARAMETERS")
            print("------------------------------------------------------------------")
            print(f"Ncoll:           {self.emitter.ncoll}")
            print(f"tv:              {tv * 1000:.1f} ms")
            print(f"T90:             {t90:.2f} s")
            print(f"E_gamma_tot:     {self.emitter.e_gamma_tot:.2e} erg")
            print()
            print(f"photospheric     {self.emitter.photo}")
            print(f"sub-photospheric {self.emitter.subphoto}")
            print("------------------------------------------------------------------")

        if config.ENABLE_LOGGING:
            f = open(path.join(config.LOG_PATH, 'output.log'), 'w')
            lines = ["------------------------------------------------------------------", "                     OUTPUT PARAMETERS", "------------------------------------------------------------------", f"Ncoll:           {self.emitter.ncoll}", f"tv:              {tv * 1000:.1f} ms", f"T90:             {t90:.2f} s", f"E_gamma_tot:     {self.emitter.e_gamma_tot:.2e} erg", " ", f"photospheric     {self.emitter.photo}", f"sub-photospheric {self.emitter.subphoto}", "------------------------------------------------------------------", " "]
            f.write('\n'.join(lines))
            f.close()

        f = open(path.join(config.LOG_PATH, 'output.log'), 'w')
        lines = ["------------------------------------------------------------------", "                     OUTPUT PARAMETERS", "------------------------------------------------------------------", f"Ncoll:           {self.emitter.ncoll}", f"tv:              {tv * 1000:.1f} ms", f"T90:             {t90:.2f} s", f"E_gamma_tot:     {self.emitter.e_gamma_tot:.2e} erg", " ", f"photospheric     {self.emitter.photo}", f"sub-photospheric {self.emitter.subphoto}", "------------------------------------------------------------------", " "]
        f.write('\n'.join(lines))
        f.close()

    def plot_initial_lorentz_factors(self):
        """
        Plots the initial Lorentz factors of the shells within the simulation. This function should be called before the sim_start function as it is dependent on intitial parameters. The image is saved to the da/im directory within the module, but the location can be altered in the config.py file. If you wish for the plot to be displayed in realtime, make sure that matplotlib.pyplot is imported within your code, and then call plt.show() or equivalent.

        Parameters:
            none

        Returns:
            none
        """
        x, y = self.emitter.retrieve_lorentz_factors()
        x = [i / 1e6 for i in x]

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)

        ax.scatter(x, y, marker='.')
        ax.set_xlabel(r'Initial shell radius $r_{k, 0}$ [$\times 10^6$ km]',fontsize=16)
        ax.set_ylabel(r'Initial Lorentz factor $\Gamma_{k, 0}$', fontsize=16)
        ax.set_yscale('log')
        ax.set(ylim=[10, 1e4])
        # ax.set(xlim=[0, 8])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
        plt.savefig(path.join(config.IM_PATH, 'initial_lorentz_factor_dist.png'))

    def plot_light_curve(self):
        """
        Plots the final prompt emission light curve of the GRB. The image is saved to the da/im directory within the module, but the location can be altered in the config.py file. If you wish for the plot to be displayed in realtime, make sure that matplotlib.pyplot is imported within your code, and then call plt.show() or equivalent.

        Parameters:
            none

        Returns:
            none
        """
        y = self.emitter.light_curve
        y.append(0)
        self.t_obs.append(0)

        tup = sorted(list(zip(self.t_obs, y)))
        x, y = zip(*tup)

        y2 = [i * 1e+6 for i in y]

        if config.VERBOSE:
            print(f"minimum flux {min(y):.4e} GeV/s/cm2 at time {x[y.index(min(y))]:.2f} s")
            print(f"maximum flux {max(y):.4e} GeV/s/cm2 at time {x[y.index(max(y))]:.2f} s")

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_axes((.2,.3,.6,.6))

        ax.plot(x, y)
        ax.set_xlabel(r't$_{obs}$ [s]',fontsize=16)
        ax.set_ylabel(r'Flux [GeV s$^{-1}$ cm$^{-2}$]', fontsize=16)
        ax.set_yscale('log')
        # ax.set(xlim=[-0.5, 80])
        ax.set(xlim=[-0.5, 1000])
        # ax.set(ylim=[1e-8, 1e-3])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)

        ax2=fig.add_axes((.2,.1,.6,.2), sharex=ax)
        ax2.plot(x, y2)
        ax2.set_xlabel(r't$_{obs}$ [s]',fontsize=16)
        ax2.set_ylabel(r'Counts [A.U.]', fontsize=16)
        # ax2.set(xlim=[-0.5, 80])
        ax2.set(xlim=[-0.5, 1000])
        # ax.set(ylim=[0, 10000])
        # ax2.set(ylim=[0, 800])
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.tick_params(axis='both', which='major', length=5)
        ax2.tick_params(axis='both', which='minor', length=2.5)
        ax2.tick_params(axis='both', which='both',direction='in',right=True,top=True)
        plt.savefig(path.join(config.IM_PATH, 'light_curve.png'))

    def plot_photosphere_radius_histogram(self):
        """
        Plots a histogram of the radii at which the optical depth to Thomson scattering is 1. By definition, this should be the photosphere of the burst, and is not at a singular radius but more of a narrow band of radii. This histogram allows for determining the most common photospheric radius of the burst. The image is saved to the da/im directory within the module, but the location can be altered in the config.py file. If you wish for the plot to be displayed in realtime, make sure that matplotlib.pyplot is imported within your code, and then call plt.show() or equivalent.

        Parameters:
            none

        Returns:
            none
        """
        x = self.emitter.r_coll
        # y = [1/i for i in self.emitter.op_depth]
        y = self.emitter.op_depth
        r_photo = []

        da = list(zip(x, y))

        for item in da:
            if round(item[1] * 2) / 2 == 1:
                r_photo.append(item[0])

        r_photo = [i / 1e+9 for i in r_photo]

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)

        plt.hist(r_photo)
        ax.set_xlabel(r'$R_{photo}$ [$\times 10^9$ km]',fontsize=16)
        ax.set_ylabel(r'Frequency', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
        plt.savefig(path.join(config.IM_PATH, 'photosphere_radius.png'))


    def plot_optical_depth(self):
        """
        Plots the optical depth to Thomson scattering for each shell collision at a given collisional radius. The image is saved to the da/im directory within the module, but the location can be altered in the config.py file. If you wish for the plot to be displayed in realtime, make sure that matplotlib.pyplot is imported within your code, and then call plt.show() or equivalent.

        Parameters:
            none

        Returns:
            none
        """
        x = self.emitter.r_coll
        # y = [1/i for i in self.emitter.op_depth]
        y = self.emitter.op_depth

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x, y, marker='.')
        ax.set_xlabel(r'$R_{C}$ [km]',fontsize=16)
        ax.set_ylabel(r'$\tau_{Th}$', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axhline(y=1, linestyle='--', color='k', linewidth=0.8)
        ax.text(1e+10, 1e+3, 'subphotospheric', fontsize=14)
        ax.fill_between([min(x)-1e+3, max(x)+1e+3], 1, max(y), color='k', alpha=0.2)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
        plt.savefig(path.join(config.IM_PATH, 'optical_depth.png'))

    def bin_proton_data(self, da):
        bin_x = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
        bin_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in da:
            for j in range(len(i[0])):
                match math.floor(math.log10(i[0][j])):
                    case -7:
                        bin_y[0] += i[1][j]
                    case -6:
                        bin_y[1] += i[1][j]
                    case -5:
                        bin_y[2] += i[1][j]
                    case -4:
                        bin_y[3] += i[1][j]
                    case -3:
                        bin_y[4] += i[1][j]
                    case -2:
                        bin_y[5] += i[1][j]
                    case -1:
                        bin_y[6] += i[1][j]
                    case 0:
                        bin_y[7] += i[1][j]
                    case 1:
                        bin_y[8] += i[1][j]
                    case 2:
                        bin_y[9] += i[1][j]
                    case 3:
                        bin_y[10] += i[1][j]
                    case 4:
                        bin_y[11] += i[1][j]
                    case 5:
                        bin_y[12] += i[1][j]
                    case 6:
                        bin_y[13] += i[1][j]
                    case 7:
                        bin_y[14] += i[1][j]
                    case 8:
                        bin_y[15] += i[1][j]
                    case 9:
                        bin_y[16] += i[1][j]
                    case 10:
                        bin_y[17] += i[1][j]
                    case 11:
                        bin_y[18] += i[1][j]
        return bin_x, bin_y

    def plot_proton_energy_spectrum(self):
        """
        Plots the final proton energy spectrum as a result of Fermi acceleration within shell collisions of the burst. Please note that the number densities have been binned for each energy. The image is saved to the da/im directory within the module, but the location can be altered in the config.py file. If you wish for the plot to be displayed in realtime, make sure that matplotlib.pyplot is imported within your code, and then call plt.show() or equivalent.

        Parameters:
            none

        Returns:
            none
        """
        x = self.emitter.e_prot_n_
        y = self.emitter.e_prot_n

        # fig = plt.figure(figsize=(8,6))
        # ax = fig.add_subplot(1,1,1)
        # ax.scatter(x, y)
        # # ax.scatter(x, y)
        # ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
        # ax.set_ylabel(r'E$^2$ N(E$_{p}$) [GeV cm$^{-2}$]', fontsize=16)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.tick_params(axis='both', which='major', labelsize=16)
        # ax.tick_params(axis='both', which='major', length=5)
        # ax.tick_params(axis='both', which='minor', length=2.5)
        # ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
        #

        # x1 = self.emitter.r_coll
        # # y = [1/i for i in self.emitter.op_depth]
        # y1 = self.emitter.op_depth
        # r_photo = []
        #
        # da1 = list(zip(x1, y1))
        #
        # for item in da1:
        #     if round(item[1] * 2) / 2 == 1:
        #         r_photo.append(item[0])
        #
        # da = list(zip(self.emitter.r_coll, x))
        # colours = plt.get_cmap('viridis')
        # fig = plt.figure(figsize=(8,6))
        # ax = fig.add_subplot(1,1,1)
        # # ax.scatter(x, y)
        # # for xe, ye in da:
        # #     plt.scatter([xe] * len(ye), ye, color=colours(da.index(list(zip(xe, ye))) / (len(x1) - 1)))
        # for index in range(len(da)):
        #     plt.scatter([da[index][0]] * len(da[index][1]), [da[index][1]], color=colours(index / (len(x1) - 1)))
        # ax.axhline(y=1e9, linestyle='--', color='k', linewidth=1.3)
        # ax.axvline(x=min(r_photo), linestyle='--', color='k', linewidth=1.3)
        # ax.axvline(x=max(r_photo), linestyle='--', color='k', linewidth=1.3)
        # ax.set_xlabel(r'$R_{C}$ [km]',fontsize=16)
        # ax.set_ylabel(r'$E_p$ [GeV]', fontsize=16)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.tick_params(axis='both', which='major', labelsize=16)
        # ax.tick_params(axis='both', which='major', length=5)
        # ax.tick_params(axis='both', which='minor', length=2.5)
        # ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)

        da = list(zip(x, y))

        bin_x, bin_y = self.bin_proton_data(da)

        # bin_y = [bin_y[i] * (bin_x[i])**2 for i in range(len(bin_y))]
        # print(bin_y)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(bin_x, bin_y)
        # ax.scatter(x, y)
        ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
        ax.set_ylabel(r'N(E$_{p}$) [GeV${-1}$ cm$^{-2}$]', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
        plt.savefig(path.join(config.IM_PATH, 'prot_energy_spec.png'))

        bin_y = [bin_y[i] * (bin_x[i])**2 for i in range(len(bin_y))]
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(bin_x, bin_y)
        # ax.scatter(x, y)
        ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
        ax.set_ylabel(r'E$^2$ N(E$_{p}$) [GeV cm$^{-2}$]', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
        plt.savefig(path.join(config.IM_PATH, 'prot_energy_spec_observer.png'))

        da = list(zip(bin_x, bin_y))

        with open(path.join(config.CSV_PATH, 'prot_energy_spec.csv'), 'w') as fp:
            output_fp = csv.writer(fp)

            for row in da:
                output_fp.writerow(row)
        return bin_x, bin_y

    def plot_proton_maximum_energy(self):
        """
        Plots the initial proton maximum energy for each collision, as well as the refined maximum proton energy taking into account losses as a function of collision radius. The image is saved to the da/im directory within the module, but the location can be altered in the config.py file. If you wish for the plot to be displayed in realtime, make sure that matplotlib.pyplot is imported within your code, and then call plt.show() or equivalent.

        Parameters:
            none

        Returns:
            none
        """
        x = self.emitter.r_coll
        y = self.emitter.e_prot
        z = self.emitter.e_prot_max

        tup = sorted(list(zip(x, y)))
        x, y = zip(*tup)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, y, label='with losses')

        tup = sorted(list(zip(x, z)))
        x, z = zip(*tup)
        ax.plot(x, z, label='without losses')
        ax.axhline(1e+9, linestyle='--', color='k')
        ax.set_xlabel(r'R$_{c}$ [km]',fontsize=16)
        ax.set_ylabel(r'$E_{p, max}$ [GeV]', fontsize=16)
        # ax.set(ylim=[1e3, 1e13])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
        ax.legend()
        plt.savefig(path.join(config.IM_PATH, 'prot_energy_rad_both.png'))

    def plot_proton_energy_loss_timescales(self):
        """
        Plots the proton energy loss timescales for each shell collision as a function of collision radius. The image is saved to the da/im directory within the module, but the location can be altered in the config.py file. If you wish for the plot to be displayed in realtime, make sure that matplotlib.pyplot is imported within your code, and then call plt.show() or equivalent.

        Parameters:
            none

        Returns:
            none
        """
        x = self.emitter.r_coll
        y = self.emitter.t_syn
        z = self.emitter.t_dyn
        a = self.emitter.t_py

        tup = sorted(list(zip(x, y)))
        x, y = zip(*tup)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, y, label='synchrotron losses timescale')

        tup = sorted(list(zip(x, z)))
        x, z = zip(*tup)
        ax.plot(x, z, label='dynamical loss timescale')
        ax.set_xlabel(r'R$_{c}$ [km]',fontsize=16)
        ax.set_ylabel(r't [s]', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # tup = sorted(list(zip(x, a)))
        # x, a = zip(*tup)
        # ax.plot(x, a, label='photohadronic loss timescale')
        # ax.set_xlabel(r'R$_{c}$ [km]',fontsize=16)
        # ax.set_ylabel(r't [s]', fontsize=16)
        # ax.set_xscale('log')
        # ax.set_yscale('log')

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
        ax.legend()
        plt.savefig(path.join(config.IM_PATH, 'loss_timescales.png'))

    def save_to_csv_proto_spec(self):
        x = self.emitter.e_prot_n_
        y = self.emitter.e_prot_n

        x, y = self.bin_proton_data(list(zip(x, y)))

        y = [y[i] * (x[i])**2 for i in range(len(y))]

        da = list(zip(x, y))

        with open(path.join(config.CSV_PATH, 'prot_energy_spec.csv'), 'w') as fp:
            output_fp = csv.writer(fp)

            for row in da:
                output_fp.writerow(row)
