import sys
sys.path.append('.')
import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import config

# constants
c = 3e+5

class Photons():
    """
    A class to represent the prompt emission gamma-ray photons.

    --------------------------------------------------------------------------------
                                Parameters contained:
    --------------------------------------------------------------------------------
    t_obs:      the time in the observer's frame (units: s)

    r:          the collision radius (units: km)

    l:          the width of the fast shell (units: km)

    b_f:        the bulk Lorentz factor of the fast shell

    b_rs:       the bulk Lorentz factor of the reverse shock

    gamma:      the Lorentz factor of the merged shell

    e_iso:      the dissipated energy as gamma-rays (units: erg)

    dt_em:      the emission timescale (units: s)

    t_rise:     the rise time (units: s)

    h_k:        the peak luminosity of the spectrum (units: erg/s)

    t:          the time in the source frame (units: s)

    --------------------------------------------------------------------------------
    """
    # initialisation parameters retrieved from individual shell collisions
    def __init__(self, t, r_ck, l_fk, b_f, b_rs, gamma_mk, e_iso_gammak, v_iso, z):
        """
        Initialises the photons.

        Parameters:
            t (int):                source frame time

            r_ck (float):           collision radius

            l_fk (float):           fast shell width

            b_f (float):            bulk Lorentz factor of fast shell

            b_rs (float):           bulk Lorentz factor of reverse shock

            gamma_mk (float):       Lorentz factor of merged shell

            e_iso_gammak (float):   energy of dissipated gamma-rays
        """
        self.r = r_ck
        self.l = l_fk
        self.b_f = b_f
        self.b_rs = b_rs
        self.gamma = gamma_mk
        self.e_iso = e_iso_gammak
        self.v_iso = v_iso
        self.z = z
        self.t_obs = ((1+self.z) * t) / (2 * gamma_mk**2)

        # the time the reverse shock crosses the fast shell - the emission timescale
        self.dt_em = (self.l / (3e+5 * (self.b_f - self.b_rs)))

        # rise time
        self.t_rise = (self.dt_em / (2 * self.gamma**2)) * (1 + self.z)

        # peak luminosity of spectrum
        self.h_k = (self.e_iso / (1 + self.z)) * (1 / self.t_rise)

        # time in source frame
        self.t = (2 * self.gamma**2 * self.t_obs) / (1 + self.z)

        if config.ENABLE_LOGGING:
            f = open(path.join(config.LOG_PATH, 'photons.log'), 'a')
            lines = ["------------------------------------------------------------", "                     PHOTON PARAMETERS", "------------------------------------------------------------", f"t_obs               {self.t_obs:.4e}", f"r                   {self.r:.4e}", f"l                   {self.l:.4e}", f"gamma               {self.gamma:.4f}", f"e_iso               {self.e_iso:.4e}", f"dt_em               {self.dt_em:.4e}", f"t_rise              {self.t_rise:.4e}", f"h_k                 {self.h_k:.4e}", f"t                   {self.t:.4e}", "------------------------------------------------------------", " "]
            f.write('\n'.join(lines))
            f.close()

    # generate a broken power law spectrum; assume e_input in keV
    def generate_photon_spectrum(self, e_input):
        """
        Generate a broken power law spectrum of the GRB.

        Parameters:
            e_input:        input GRB energy in keV

        Returns:


        """
        e_break = 1
        a_gamma = 1
        b_gamma = 2
        C_gamma = self.e_iso / (self.gamma**2 * self.v_iso)

        if config.ENABLE_LOGGING:
            f = open(path.join(config.LOG_PATH, 'photons.log'), 'a')

            if (e_input < e_break):
                lines = ["------------------------------------------------------------", "              PHOTON SPECTRUM", "------------------------------------------------------------", f"e_input < e_break; ngamma =  {C_gamma * (e_input/e_break)**a_gamma}", "------------------------------------------------------------", " "]
                f.write('\n'.join(lines))
                f.close()
                return C_gamma * (e_input/e_break)**a_gamma

            else:
                lines = ["------------------------------------------------------------", "              PHOTON SPECTRUM", "------------------------------------------------------------", f"e_input > e_break; ngamma =  {C_gamma * (e_input/e_break)**b_gamma}", "------------------------------------------------------------", " "]
                f.write('\n'.join(lines))
                f.close()

        return C_gamma * (e_input/e_break)**b_gamma


    def parametrise_luminosity(self):
        """
        Parametrises the luminosity of the photon pulse as a peaked profile with a fast rise and exponential decay ("FRED" profile).

        Parameters:
            none

        Returns:
            a luminosity with units erg/s/km^2
        """

        if (self.t_obs < 0):
            if config.ENABLE_LOGGING:
                f = open(path.join(config.LOG_PATH, 'photons.log'), 'a')
                f.write(">>> obtained tobs < 0 \n")
                f.close()

            return 0

        if (self.t_obs >= 0) and (self.t_obs < self.t_rise):
            if config.ENABLE_LOGGING:
                f = open(path.join(config.LOG_PATH, 'photons.log'), 'a')
                f.write(">>> obtained 0 < t_obs < t_rise \n")
                f.close()

            return self.h_k * (1 - (1/(1 + ((2 * self.gamma**2 * c*self.t) / (self.r)))**2))

        else:
            if config.ENABLE_LOGGING:
                f = open(path.join(config.LOG_PATH, 'photons.log'), 'a')
                f.write(">>> obtained else \n")
                f.close()

            return self.h_k * ((1 / (1 + ((2 * self.gamma**2 * (self.t - self.dt_em))/(self.r)))**2) - (1/(1+((2 * self.gamma**2 * c * self.t)/(self.r)))**2))

    def to_GeV_s_cm(self, luminosity):
        """+
        Converts the luminosity into a flux with units GeV/s/cm^2

        Parameters:
            none

        Returns:
            a flux with units GeV/s/cm^2
        """
        return (luminosity * (624.15)) / (4 * math.pi * (Planck18.luminosity_distance(self.z).to_value()*3.08568e+24)**2)
