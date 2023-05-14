import sys
sys.path.append('.')
import numpy as np
import math
from random import gauss
# from statistics import mean
from photons import Photons
from protons import Protons
from astropy.cosmology import Planck18
import config
import os.path as path

""" Constants """
c = 1
e_iso_kin = 1e+52       # erg
epsilon_b = 8.333e-2
epsilon_e = 8.333e-2
epsilon_p = 8.333e-1
proton_mass = 0.938     # GeV
sigma_th = 66.52e-36    # km^2



class Shell():
    """
    A class for storing information regarding a singular spherical shell of matter.

    --------------------------------------------------------------------------------
                                Parameters contained:
    --------------------------------------------------------------------------------

    index:      used to store index number of the shell. Higher indices are located
                closer to the central engine

    gamma:      shell Lorentz factor

    bulk_gamma: shell bulk Lorentz factor

    r0:         initial shell distance from the central engine (units: km)

    radius:     current shell distance from the central engine (units: km)

    l:          shell width (units: km)

    volume:     volume of the spherical shell (units: km^3)

    mass:       shell mass (units: GeV)

    density:    shell density (units: GeV/km^3)

    bulk_e_kin: shell bulk kinetic energy (units: erg)

    --------------------------------------------------------------------------------
    """

    def __init__(self, gamma, radius, l, index):
        """
        Initialises the Shell object.

        Parameters:
            gamma (float):     Lorentz factor of the shell

            radius (float):    shell distance from central engine

            l (float):         shell width

            index (int):       index number of shell

        Returns:
            none
        """
        self.index = index
        self.gamma = gamma
        self.bulk_gamma = math.sqrt(1 - gamma**-2)
        self.r0 = radius
        self.radius = radius
        self.l = l
        self.volume = 4 * np.pi * radius**2 * l
        self.mass = (e_iso_kin * 624.15) / (gamma * c**2)
        self.density = self.mass / self.volume
        self.bulk_e_kin = self.gamma * (self.mass/624.15) * c**2

    def update_shell(self, t):
        """
        Updates shell parameters at a point in time.

        Parameters:
            t (float): simulation time

        Returns:
            none
        """
        self.radius = self.radius + (3e+5 * self.bulk_gamma * t)
        self.volume = 4 * np.pi * self.radius**2 * self.l
        self.density = self.mass / self.volume


class Emitter():
    """
    A class to represent the central engine of the GRB. This class is used to setup the emitted shells and their collisions.

    --------------------------------------------------------------------------------
                                Parameters contained:
    --------------------------------------------------------------------------------

    r_dec:              deceleration radius of the burst (units: km)

    r_n_sh:             distance from innermost shell to central engine (units: km)

    nshells:            initial number of shells emitted

    dt_eng:             downtime of the central engine (units: s)

    shells:             list containing Shell objects of indivial shells

    next_collision_time:    time of next shell collision (units: s).
                            Default: 0

    t_obs:              collision time in observer's frame (units: s)

    collision shells:   list containing Shell objects of next two shells to collide

    ncoll:              number of shell collisions occurred

    light_curve:        list of flux values emitted by gamma rays during each
                        collision. Each entry corresponds to one shell collision

    e_gamma_tot:        total amount of energy released as gamma-rays (units: erg)

    r_coll:             list containing past collision radii of shells

    op_depth:           list containing optical depths of shells

    photo:              number of photospheric contributions to light curve

    subphoto:           number of subphotospheric collisions

    done:               keeps track of whether or not the GRB has finished.
                        Default: False
    --------------------------------------------------------------------------------
    """

    def __init__(self, r_dec, rn_sh, nshells, dt_eng, z, ep = epsilon_p):
        """
        Initialises the central engine.

        Parameters:
            r_dec (float):      deceleration radius of the burst

            rn_sh (float):      distance from innermost shell to central engine

            nshells (int):      initial number of shells emitted

            dt_eng (float):     uptime of the central engine

        Returns:
            none
        """
        self.r_dec = r_dec
        self.r_n_sh = rn_sh
        self.nshells = nshells
        self.dt_eng = dt_eng
        self.z = z
        self.ep = ep
        self.ee = ((1 - ep) / 2)
        self.eb = ((1 - ep) / 2)
        # print(self.ee)

        self.shells = []
        self.next_collision_time = 0
        self.t_obs = 0
        self.collision_shells = []
        self.ncoll = 0
        self.light_curve = []
        self.e_gamma_tot = 0
        self.r_coll = []
        self.r_coll_lc = []
        self.op_depth = []
        self.e_prot = []
        self.e_prot_max = []
        self.e_ind = []
        self.e_prot_n = []
        self.e_prot_n_ = []
        self.photo = 0
        self.subphoto = 0
        self.done = False

        self.t_syn = []
        self.t_dyn = []
        self.t_py = []

        self.shock_rat = []
        self.shock_rat2 = []

    #---------------------------------------------------------------------------
    #                       Lorentz factor distributions
    #---------------------------------------------------------------------------

    def gamma_dist_GRB1(self, gamma0):
        """
        Lorentz factor distribution for GRB1 of Bustamante et al. (2017). Lorentz factors are randomly samples from a log-normal distribution defined by a characteristic value gamma0.

        Parameters:
            gamma0 (float): characteristic Lorentz factor of distribution

        Returns:
            (float) randomly sampled Lorentz factor from distribution
        """
        return 1 + (gamma0 - 1) * np.exp(gauss(0, 1))

    def gamma_dist_GRB2(self, gamma1, gamma2):
        a1 = 1
        a2 = 0.1

        return 1 + (gamma2 - 1) * np.exp(a2*gauss(0, 1))

    def delta_step(self, k, val):
        """
        Dirac-delta distribution used to define amplitude values for gamma_dist_GRB3.

        Parameters:
            k:

            val:

        Returns:
            0 if k < val; else 1
        """
        if (k < val):
            return 0
        else:
            return 1

    def gamma_dist_GRB3(self, gamma1, gamma2, tp, k):
        """
        Lorentz factor distribution for GRB3 of Bustamante et al. (2017). Lorentz factors sampled form a sawtooth with narrow distribution and fluctuate between two characteristic values: gamma1 and gamma2.

        Parameters:
            gamma1 (float):     first characteristic Lorentz factor value

            gamma2 (float):     second characteristic Lorentz factor value

            tp (float):         fraction of total emitted shells

        Returns:
            (float) randomly sampled Lorentz factor from distribution
        """
        a = 0.1
        gamma = ((gamma1 - gamma2)/(self.nshells * tp)) * k + gamma2 - (gamma1 - gamma2) * (self.delta_step(k, self.nshells * tp) + self.delta_step(k, 2*self.nshells*tp))
        return  1 + (gamma - 1) * np.exp(a * gauss(0, 1))

    #---------------------------------------------------------------------------

    def setup(self, gamma1, gamma2, tp, dist):
        """
        Sets up the shells emitted by the central engine - assigning indices, Lorentz factors, widths, and distances from central engine.

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

        if config.VERBOSE:
            print("------------------------------------------------------------")
            print("                     SHELL GENERATION")
            print("------------------------------------------------------------")

        if config.ENABLE_LOGGING:
            f = open(path.join(config.LOG_PATH, 'shells.log'), 'w')

        # initialise shells
        for i in range(self.nshells):

            # setting up Lorentz factor of shell
            if (dist == 2):
                gamma = self.gamma_dist_GRB2(gamma1, gamma2)
            if (dist == 3):
                gamma = self.gamma_dist_GRB3(gamma1, gamma2, tp, i)
            else:
                gamma = self.gamma_dist_GRB1(gamma1)

            # shell width and distance from central engine
            # l = c * self.dt_eng
            l = 3e+5 * self.dt_eng
            r = self.r_n_sh + ((self.nshells - i) * (2 * l))

            # initialise shell and contain it within central engine's list of shells
            shell = Shell(gamma, r, l, i)
            self.shells.append(shell)

            # log shell parameters
            if config.ENABLE_LOGGING:
                lines = [f"shell:: {i}", f"          gamma: {gamma:.4f}", f"              l: {l:.4e}", f"         radius: {r:.4e}", f"           mass: {shell.mass:.4e}", "\n"]
                f.write('\n'.join(lines))
                f.close()

        # end of shell initialisation

        if config.VERBOSE:
            print(">>>>>>> shells generated")
            print("------------------------------------------------------------")
            print()

        self.calculate_collision_times()

    def retrieve_lorentz_factors(self):
        """
        Function used for plotting Lorentz factors of shells within the GRB. Retrieves the Lorentz factor and distance from the central engine.

        Parameters:
            none

        Returns:
            x (list): a list storing the distances from the central engine

            y (list): a list storing the shell Lorentz factors
        """
        x = []
        y = []

        for shell in self.shells:
            x.append(shell.radius)
            y.append(shell.gamma)

        return x, y

    def retrieve_radii(self):
        """
        Function used for plotting distance of shells from the central engine of the GRB. Retrieves the shell index and distance from the central engine.

        Parameters:
            none

        Returns:
            x (list): a list storing the shell indices

            y (list): a list storing the distances from the central engine
        """
        x = []
        y = []

        for shell in self.shells:
            x.append(shell.index)
            y.append(shell.radius)

        return x, y

    def calculate_collision_times(self):
        """
        Calculates collision times of the shells in simulation. Also removes any shells that have passed the GRB's deceleration radius.

        Parameters:
            none

        Returns:
            none
        """
        self.collision_shells = []
        count = 0
        # first, remove any shells past the deceleration radius from simulation
        for shell in self.shells:
            if (shell.radius > self.r_dec):
                self.shells.remove(shell)
                self.nshells -= 1
                count += 1

        if config.VERBOSE:
            print(f">>>>>>> removed {count} shells from simulation. {len(self.shells)} remaining")
            print()

        # make sure there is more than one shell remaining in simulation
        if (self.nshells > 1):

            # keep a track of shells to collide
            # first shell in collision is stored, and will be used as an index
            # to retrive the set of next shells to collide
            shells_to_collide = []
            collision_times = []

            for index in (range(len(self.shells)-1)):

                # determine if shell collision parameters
                shell_separation = self.shells[index].radius - self.shells[index+1].radius - self.shells[index+1].l
                bdiff = self.shells[index+1].bulk_gamma - self.shells[index].bulk_gamma

                # if no difference in gamma, shells will not collide
                if (bdiff == 0):
                    continue

                dt = abs(shell_separation / (3e+5 * bdiff))
                collision_times.append(dt)
                shells_to_collide.append(self.shells[index])

            if config.VERBOSE:
                print(f"number of shells to collide: {len(collision_times)}")
                print()

            # if there are collisions to occur, determine time of next collision
            # and shells to collide
            if (len(collision_times) > 0):
                self.next_collision_time = min(collision_times)
                index_next_collision = collision_times.index(self.next_collision_time)
                shell = shells_to_collide[index_next_collision]
                self.collision_shells.append(shell)
                next_shell = self.shells.index(shell) + 1
                self.collision_shells.append(self.shells[next_shell])

                if config.VERBOSE:
                    print(f"no. of shells to collide: {len(self.collision_shells)}")
                    print(f">>>>>>> collision of shells with gamma {self.collision_shells[1].gamma:.2f} and {self.collision_shells[0].gamma:.2f}")
                    print(f">>>>>>> at time interval {self.next_collision_time:.2e} s")
                    print()

            # if no more collisions can occur, simulation has finished
            else:
                self.done = True

                if config.VERBOSE:
                    print(">>>>>>> no more collisions!")


        # if there is only one shell remaining, simulation has finished as no
        # more shells can collide
        else:
            self.done = True

            if config.VERBOSE:
                print(">>>>>>> no more collisions!")

    def collision(self, shell1, shell2):
        """
        Simulate a collision between two shells (a fast and slow shell). A new merged shell is born and the slow shell is removed from the simulation. During the collision, a pulse of gamma rays is emitted corresponding to the
        prompt emission phase of the GRB. The flux of the pulse is contained within the light curve.

        Parameters:
            shell1 (object): first shell to collide

            shell2 (object): second shell to collide

        Returns:
            (int) 0 if the collision is above the photosphere

            (int) 1 if the collision is subphotospheric
        """
        # determine which shell is the fast moving shell and which is the slow moving shell
        if (shell1.gamma < shell2.gamma):
            slow_shell = shell1
            fast_shell = shell2
        else:
            slow_shell = shell2
            fast_shell = shell1

        if config.ENABLE_LOGGING:
            f = open(path.join(config.LOG_PATH, 'collision.log'), 'a')
            lines = ["------------------------------------------------------------", "                     COLLISION", f"fast shell::       gamma: {fast_shell.gamma:.4f}", f"                    mass: {fast_shell.mass:.4e}", f"                 density: {fast_shell.density:.4e}", f"                  volume: {fast_shell.volume:.4e}",f"                       l: {fast_shell.l:.4e}", f"                  radius: {fast_shell.radius:.4e}", f"              bulk gamma: {fast_shell.bulk_gamma:.4e}", " ", f"slow shell::       gamma: {slow_shell.gamma:.4f}", f"                    mass: {slow_shell.mass:.4e}", f"                 density: {slow_shell.density:.4e}", f"                  volume: {slow_shell.volume:.4e}", f"                       l: {slow_shell.l:.4e}", f"                  radius: {slow_shell.radius:.4e}", f"              bulk gamma: {slow_shell.bulk_gamma:.4e}", " "]
            f.write('\n'.join(lines))

        if config.VERBOSE:
            print("------------------------------------------------------------")
            print("                     COLLISION")
            print("------------------------------------------------------------")



        #-----------------------------------------------------------------------
        #                  calculate merged shell & shock parameters
        #                        and create new merged shell
        #-----------------------------------------------------------------------
        # merged shell Lorentz factor & bulk Lorentz factor
        merged_gamma = math.sqrt(((fast_shell.gamma * fast_shell.mass) + (slow_shell.gamma * slow_shell.mass)) / ((fast_shell.mass / fast_shell.gamma) + (slow_shell.mass / slow_shell.gamma)))
        bulk_gamma_merged = math.sqrt(1 - merged_gamma**-2)

        # forward shock Lorentz factor & bulk Lorentz factor
        gamma_fs = merged_gamma * math.sqrt((1 + ((2 * merged_gamma) / slow_shell.gamma)) / (2 + (merged_gamma / slow_shell.gamma)))
        bulk_gamma_fs = math.sqrt(1 - gamma_fs**-2)

        # reverse shock Lorentz factor & bulk Lorentz factor
        gamma_rs = merged_gamma * math.sqrt((1 + ((2 * merged_gamma) / fast_shell.gamma)) / (2 + (merged_gamma / fast_shell.gamma)))
        bulk_gamma_rs = math.sqrt(1 - gamma_rs**-2)

        self.shock_rat.append(fast_shell.gamma / slow_shell.gamma)
        self.shock_rat2.append(gamma_fs / gamma_rs)

        # merged shell width
        l_merged = slow_shell.l * ((bulk_gamma_fs - bulk_gamma_merged) / (bulk_gamma_fs - slow_shell.bulk_gamma)) + fast_shell.l * ((bulk_gamma_merged - bulk_gamma_rs) / (fast_shell.bulk_gamma - bulk_gamma_rs))

        # merged shell density
        # density_merged = ((fast_shell.l * fast_shell.density * fast_shell.gamma) + (slow_shell.l * slow_shell.density * slow_shell.gamma)) / (l_merged * merged_gamma)
        density_merged = ((fast_shell.l * fast_shell.density) + (slow_shell.l * slow_shell.density)) / l_merged

        # create new merged shell
        merged_shell = Shell(merged_gamma, slow_shell.radius, l_merged, slow_shell.index)

        # merged shell mass
        merged_mass = merged_shell.volume * density_merged

        # merged shell internal energy - to be radiated away
        e_iso_coll = (((fast_shell.gamma * (fast_shell.mass/624.15)) + (slow_shell.gamma * (slow_shell.mass/624.15))) * c**2) - (merged_gamma * ((slow_shell.mass/624.15) + (fast_shell.mass/624.15)) * c**2)

        # calculate energy dissipated in photons
        e_gamma_k = self.ee * e_iso_coll
        self.e_gamma_tot += e_gamma_k

        # calculate energy dissipated in protons
        e_proton_k = self.ep * e_iso_coll
        n_prot = merged_mass / proton_mass

        # merged shell kinetic energy
        e_kin_merged = (merged_gamma * (merged_mass/624.15) * c**2) #- e_iso_coll

        # update merged shell parameters
        merged_shell.mass = merged_mass
        merged_shell.density = density_merged
        merged_shell.bulk_e_kin = e_kin_merged

        if config.ENABLE_LOGGING:
            lines = [f"merged shell::     gamma: {merged_shell.gamma:.4f}", f"               mass (kg): {merged_shell.mass:.4e}", f"                 density: {merged_shell.density:.4e}", f"                  volume: {merged_shell.volume:.4e}", f"                       l: {merged_shell.l:.4e}", f"                  E_coll: {e_iso_coll:.4e}", f"                   E_kin: {e_kin_merged:.4e}", f"                  radius: {merged_shell.radius:.4e}", " "]
            f.write('\n'.join(lines))

            lines = [f"forward shock::        gamma: {gamma_fs:.4f}", f"                  bulk gamma: {bulk_gamma_fs:.4f}", " ", f"reverse shock::        gamma: {gamma_rs:.4f}", f"                  bulk gamma: {bulk_gamma_rs:.4e}", " "]
            f.write('\n'.join(lines))

        # record collision radius, optical depth
        self.r_coll.append(merged_shell.radius)
        self.op_depth.append(self.calculate_thomson_optical_depth(merged_shell))

        if config.ENABLE_LOGGING:
            lines = ["------------------------------------------------------------", f"optical depth: {self.calculate_thomson_optical_depth(merged_shell):.4f}", f"collision radius: {merged_shell.radius:.4e}", " "]
            f.write('\n'.join(lines))
            f.close()

        photons = Photons(self.t_obs, merged_shell.r0, fast_shell.l, fast_shell.bulk_gamma, bulk_gamma_rs, merged_shell.gamma, e_gamma_k, merged_shell.volume, self.z)

        if config.PROTON_PHYSICS:
            protons = Protons(e_proton_k, gamma_fs, bulk_gamma_fs, self.ep, self.ee, self.eb, merged_shell, photons)
            p, n_p, e_p, e_p_n = protons.generate_proton_spectrum()
        # self.e_prot.append(protons.emax)
            n_p = protons.to_GeV_s_cm(n_p)
            self.e_prot_max.append(protons.emax)
            self.e_prot.append(e_p)
            self.e_prot_n.append(n_p)
            self.e_ind.append(e_p_n)
            self.e_prot_n_.append(p)

            self.t_syn.append(protons.t_syn)
            self.t_dyn.append(protons.t_dyn)
            self.t_py.append(protons.t_py)

        # if the collision is above the photosphere, generate the light curve and
        # finish off the collision
        if (self.calculate_thomson_optical_depth(merged_shell) <= 1):
            self.photo += 1
            #-----------------------------------------------------------------------
            #                  generate synthetic light curve
            #-----------------------------------------------------------------------

            # generate light curve
            lc = photons.to_GeV_s_cm(photons.parametrise_luminosity())
            self.light_curve.append(lc)
            self.r_coll_lc.append(merged_shell.radius)

            if config.VERBOSE:
                print(f'flux = {lc:.4e} GeV/s/cm2')


            #-----------------------------------------------------------------------
            #                        finish off collision
            #-----------------------------------------------------------------------
            # collision efficiency
            eff = 1 - ((slow_shell.mass + fast_shell.mass) / np.sqrt(slow_shell.mass**2 + fast_shell.mass**2 + (slow_shell.mass * fast_shell.mass * ((fast_shell.gamma / slow_shell.gamma) + (slow_shell.gamma / fast_shell.gamma)))))

            if config.VERBOSE:
                print(f"shock efficiency = {eff:.4e}")

            # fast shell is removed from the simulation
            self.shells.remove(fast_shell)

            # replace slow shell with merged shell
            i = self.shells.index(slow_shell)
            self.shells[i] = merged_shell

            # update nshells and ncoll
            self.nshells -= 1
            self.ncoll += 1

            # compute next collision time
            self.collision_shells = []
            self.calculate_collision_times()
            return 0

        # collision is subphotospheric; do not generate the light curve but finish
        # off the collision
        else:
            self.subphoto += 1
            #-----------------------------------------------------------------------
            #                        finish off collision
            #-----------------------------------------------------------------------
            # collision efficiency
            eff = 1 - ((slow_shell.mass + fast_shell.mass) / np.sqrt(slow_shell.mass**2 + fast_shell.mass**2 + (slow_shell.mass * fast_shell.mass * ((fast_shell.gamma / slow_shell.gamma) + (slow_shell.gamma / fast_shell.gamma)))))

            if config.VERBOSE:
                print(f"shock efficiency = {eff:.4e}")

            # fast shell is removed from the simulation
            self.shells.remove(fast_shell)

            # replace slow shell with merged shell
            i = self.shells.index(slow_shell)
            self.shells[i] = merged_shell

            # update nshells and ncoll
            self.nshells -= 1
            self.ncoll += 1

            # compute next collision time
            self.collision_shells = []
            self.calculate_collision_times()
            return 1

    def calculate_thomson_optical_depth(self, shell):
        """
        Calculates the optical depth to Thomson scattering for a shell

        Parameters:
            shell (object):  input shell

        Returns:
            t_obs (float): optical depth to Thomson scattering
        """
        electron_density = shell.mass / (proton_mass * shell.volume)
        return (electron_density * sigma_th * shell.l)

    def update_tobs(self, tobs):
        """
        Update observer time.

        Parameters:
            tobs (float):  observer time (s)

        Returns:
            none
        """
        self.t_obs = tobs

    def calculate_t90(self, time):
        """
        Calculates T90 (the time elapsed between the detection of 5% and 95% of the total gamma-ray energy)

        Parameters:
            time (list):  a list containing the observed time of gamma-ray energy

        Returns:
            t90 (float):  the T90 value of the burst
        """
        # sorts the light curve and time in ascending order by time
        tup = sorted(list(zip(time, self.light_curve)))
        x, y = zip(*tup)

        # determine total energy of the burst by summing up the light curve
        total_energy = 0
        for i in y:
            total_energy += i

        # calculations of 5% and 95% of the total energy
        e5 = 0.05 * total_energy
        e95 = 0.95 * total_energy

        if config.VERBOSE:
            print(f"e5: {e5:.4e}")
            print(f"e95: {e95:.4e}")

        # determine the time in which 5% of the total energy is observed
        e = 0
        index = 0
        for i in y:
            e += i
            if (e >= e5):
                index = y.index(i)
                break
        t5 = x[index]

        # determine the time in which 95% of the total energy is observed
        e = 0
        index = 0
        for i in y:
            e += i
            if (e >= e95):
                index = y.index(i)
                break
        t95 = x[index]

        # calculate T90
        t90 = t95 - t5
        return t90

    def calculate_tv(self, t90):
        """
        Calculates the variability timescale of the burst (tv).

        Parameters:
            t90 (float):  the T90 value of the burst (seconds)

        Returns:
            tv (float):   the variability timescale of the burst (seconds)
        """
        return t90 / self.ncoll
