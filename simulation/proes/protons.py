import sys
sys.path.append('../lib')
from lib.pysophia import *
from scipy.integrate import quad#, LowLevelCallable
import numpy as np
import os, ctypes
import config
import math
from astropy.cosmology import Planck18

# constants
# eb = 1/12
# ee = 1/12
# ep = 5/6
c = 3e+10        # cm/s
e_charge = 4.803e-10   # Fr
m_p = 0.938     # GeV
e_th = 0.150    # GeV



class Protons():
    """
    A class to represent the prompt emission accelerated protons.

    --------------------------------------------------------------------------------
                                Parameters contained:
    --------------------------------------------------------------------------------
    e_prot:         the GRB energy injected into the protons (units: erg)

    e_prot_gev:     the GRB energy injected into the protons (units: GeV)

    mass:           the mass of the merged shell (units: GeV)

    n_prot:         number of protons contained within the merged shell

    e_prot_n_gev:   the energy per proton (units: GeV)

    gamma_m:        the Lorentz factor of the merged shell

    gamma_fs:       the Lorentz factor of the forward shock

    bulk_gamma_fs:  the bulk Lorentz factor of the forward shock

    e_gamma:        the GRB energy injected into the photons (units: erg)

    rc:             the collision radius (units: km)

    photons:        the photons object associated with the collision

    emax:           the maximum proton energy calculated (units: GeV)

    tsync:          the synchrotron loss timescale (units: s)

    tdyn:           the dynamical loss timescale (units: s)

    --------------------------------------------------------------------------------
    """
    def __init__(self, e_prot, gamma_fs, bulk_gamma_fs, ep, ee, eb, merged_shell, photons):
        """
        Initialises the protons.

        Parameters:
            e_prot:         the GRB energy injected into the protons (units: erg)

            mass:           the mass of the merged shell (units: GeV)

            gamma_m:        the Lorentz factor of the merged shell

            gamma_fs:       the Lorentz factor of the forward shock

            bulk_gamma_fs:  the bulk Lorentz factor of the forward shock

            e_gamma:        the GRB energy injected into the photons (units: erg)

            rc:             the collision radius (units: km)

            lm:             the width of the merged shell (units: km)

            photons:        the photons object associated with the collision
        """
        self.e_prot = e_prot
        self.e_prot_gev = e_prot * 624.15
        self.mass = merged_shell.mass
        self.n_prot = self.mass / m_p
        self.e_prot_n_gev = self.e_prot_gev / self.n_prot
        self.gamma_m = merged_shell.gamma
        self.gamma_fs = gamma_fs
        self.bulk_gamma_fs = bulk_gamma_fs
        self.ep = ep
        self.ee = ee
        self.eb = eb
        # self.ep = 5/6
        # self.ee = 1/12
        # self.eb = 1/12
        self.e_gamma = photons.e_iso
        self.rc = photons.r
        self.lm = merged_shell.r0
        self.photons = photons
        self.emax = 0
        self.t_sync = 0
        self.t_dyn = 0
        self.t_py = 0


    def calculate_magnetic_field(self):
        """
        Calculates the magnetic field associated with the merged shell.

        Parameters:
            none

        Returns:
            the magnetic field in kG

        """
        return 44.7 * (self.gamma_m / (10**2.5))**(-1) * (self.eb / self.ee)**(1/2) * (self.e_gamma / 1e+50)**(1/2) * (self.rc / 1e+9)**(-1) * (self.lm / 1e+3)**(-1/2)

    def calculate_acceleration_time(self, E, eff, b):
        """
        Calculates the proton acceleration timescale for a given energy and magnetic field.

        Parameters:
            E:  the proton energy (erg)

            b:  the input magnetic field (G)

        Returns:
            the acceleration time (units: s)
        """
        return E / (eff * c * e_charge * b)

    def calculate_synchrotron_loss_timescale(self, E, b):
        """
        Calculates the synchrotron loss timescale for the protons being accelerated for a given energy and magnetic field.

        Parameters:
            E:  the energy of the protons (units: erg)

            b:  the input magnetic field (units: G)

        Returns:
            the synchrotron loss timescale (units: s)
        """
        return (9 * (m_p / 624.15)**4) / (4 * c * e_charge**4 * b**2 * E)

    def calculate_dynamical_timescale(self):
        """
        Calculates the dynamical loss timescale for the protons within the merged shell.

        Parameters:
            none

        Returns:
            the dynamical loss timescale (units: s)
        """
        return self.lm / 3e+5

    def cross_sec_integ(self, E, e):
        """
        Helper function for determining the photohadronic loss timescale. This function provides the integral of the interaction cross sections of protons at a given energy. This function should not be called outside of the calculate_photohadronic_loss_timescale function. At this stage, I'm not sure how well this function is working, and so it is not called in the code at the moment.

        Parameters:
            E:  the maximum proton energy (units: GeV)

            e:  the integration energies (units: GeV)

        Returns:
            the result of the cross section integral
        """

        SI = SophiaInterface()
        # test = quad(lambda er: er * SI.crossection(er, 3, 13), e_th, (2 * E * e) / m_p, limit=100)
        #
        # if config.ENABLE_LOGGING:
        #     f = open(path.join(config.LOG_PATH, 'cross-sec.log'), 'a')
        #     lines = [f"{test[0]}", " "]
        #     f.write('\n'.join(lines))
        #     f.close()
        # # print(test[0])
        # return test
        return quad(lambda er: er * SI.crossection(er, 3, 13), e_th, (2 * E * e) / m_p, limit=100)

    def calculate_photohadronic_loss_timescale(self, E):
        """
        Calculates the photohadronic loss timescale for protons with a given energy. At this stage, I don't believe this function is working correctly, due to extremely large timescales being returned. Hence, this function is not called at any point in the code at the moment.

        Parameters:
            E:  The proton energy (units: erg)

        Returns:
            the photohadronic loss timescale (units: s)
        """

        return (1/2) * (m_p / E)**2 * quad(lambda e: (self.photons.generate_photon_spectrum(e * 1e+6) / e**2) * self.cross_sec_integ(E, e)[0], (e_th * m_p) / (2 * E), np.inf, limit=100)[0]
        # test = (1/2) * (m_p / E)**2 * quad(lambda e: (self.photons.generate_photon_spectrum(e * 1e+6) / e**2) * self.cross_sec_integ(E, e)[0], (e_th * m_p) / (2 * E), np.inf, limit=500)[0]
        # print(test)
        # if config.VERBOSE:
        #     print(".-.-.-.-.-.-.-.-.-.-.-.-")
        #     print(f"Proton energy: {E:.4f}")
        #     print(f" second integ: {test:.4e}")
        #     print(".-.-.-.-.-.-.-.-.-.-.-.-")
        # return test

    # def cross_sec_integ(self, E, e):
    #     lib = ctypes.CDLL(os.path.abspath('../lib/proton_integrator.so'))
    #
    #     lib.cross_sec_integ.restype = ctypes.c_double
    #     lib.cross_sec_integ.argtypes = (ctypes.POINTER(ctypes.c_double))
    #
    # def calculate_photohadronic_loss_timescale(self, E):
    #     lib = ctypes.CDLL(os.path.abspath('../lib/proton_integrator.so'))
    #
    #     lib.py_loss_timescale_integ.restype = ctypes.c_double
    #     lib.py_loss_timescale_integ.argtypes = (ctypes.POINTER(ctypes.c_double))
    #
    #     parm =

    def calculate_maximum_energy(self, eff):
        """
        Calculates the maximum energy that the protons can obtain via Fermi acceleration in the shell configuration given the energy losses that may occur.

        Parameters:
            eff:    the acceleration efficiency

        Returns:
            The maximum proton energy (units: erg)
        """
        b = self.calculate_magnetic_field() * 1e+3
        e_p_max = e_charge * b * self.gamma_fs * self.bulk_gamma_fs * self.rc * 1e+5
        self.emax = e_p_max * 624.15
        t_dyn = self.calculate_dynamical_timescale()
        t_syn = self.calculate_synchrotron_loss_timescale(e_p_max, b)
        self.t_syn = t_syn
        self.t_dyn = t_dyn
        # t_py = 1 / self.calculate_photohadronic_loss_timescale(e_p_max * 624.15)
        # self.t_py = t_py
        t_py = -1
        # print(t_py)

        if (t_py < 0):
            t_acc = min(t_dyn, t_syn)
        else:
            t_acc = min(t_dyn, t_syn, t_py)

        if config.ENABLE_LOGGING:
            f = open(path.join(config.LOG_PATH, 'protons.log'), 'a')

            if (t_acc == t_dyn):
                lines = ["------------------------------------------------------------", "              PROTON PARAMETERS", "------------------------------------------------------------", " dominant loss mechanism: adiabatic loss due to expansion of shell", f" loss timescale: {t_dyn}", f" max proton energy: {t_acc * eff * c * e_charge * b}", "------------------------------------------------------------", " "]
                f.write('\n'.join(lines))
                f.close()

            elif (t_acc == t_syn):
                lines = ["------------------------------------------------------------", "              PROTON PARAMETERS", "------------------------------------------------------------", " dominant loss mechanism: synchrotron", f" loss timescale: {t_syn}", f" max proton energy: {t_acc * eff * c * e_charge * b}", "------------------------------------------------------------", " "]
                f.write('\n'.join(lines))
                f.close()

            else:
                lines = ["------------------------------------------------------------", "              PROTON PARAMETERS", "------------------------------------------------------------", " dominant loss mechanism: photohadronic losses", f" loss timescale: {t_py}", f" max proton energy: {t_acc * eff * c * e_charge * b}", "------------------------------------------------------------", " "]
                f.write('\n'.join(lines))
                f.close()

        return t_acc * eff * c * e_charge * b

    def generate_proton_spectrum(self):
        """
        Generates a proton energy spectrum that can be obtained via Fermi acceleration within the shell collision.

        Parameters:
            none

        Returns:
            p:              a list containing the proton energies in the spectrum

            n_p:            a list containing the number of protons accelerated to a given energy in list p

            e_pmax:         the maximum proton energy used in the calculations (units: GeV)

            e_prot_n_gev:   the original proton energy (units: GeV)

        """
        # C_p = ((self.photons.e_iso * 624.15) / self.photons.v_iso) * (self.ep / self.ee)
        C_p = ((self.e_prot * 624.15) / self.photons.v_iso) * (self.ep / self.ee)

        e_pmax = self.calculate_maximum_energy(1) * 624.15
        p = np.logspace(math.log10(self.e_prot_n_gev), math.log10(e_pmax), 50)
        n_p = [C_p * (i)**(-2) * np.exp(-(i / e_pmax)**(2)) for i in p]

        return p, n_p, e_pmax, self.e_prot_n_gev #self.e_prot_n_gev

    def to_GeV_s_cm(self, spec):
        return (self.photons.v_iso) * ((spec) / (4*math.pi * (Planck18.luminosity_distance(self.photons.z).to_value()*3.08568e+24)**2))
