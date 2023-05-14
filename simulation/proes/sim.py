import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import math
from astropy.cosmology import Planck18
import os
from os.path import exists
# from os import remove as rm
from shells import Shell, Emitter

import config

c = 3e+5


emitter = Emitter(5.5e+11, 1e+3, 1000, 1e-2, 2)
avgm = []
avgv = []
avgd = []
# avge = [1]

def sim():
    emitter.setup(500, 0, 0, 1)
    # emitter.setup(500, 50, 0, 2)
    # emitter.setup(50, 500, 0.34, 3)
    lorentz_factors()

    sim_time = [0]
    obs_time = []
    st = 0
    avgm0 = calculate_mean(emitter.shells, 'mass')
    avgm.append(avgm0 / avgm0)
    avgv0 = calculate_mean(emitter.shells, 'volume')
    avgv.append(avgv0 / avgv0)
    avgd0 = calculate_mean(emitter.shells, 'density')
    avgd.append(avgd0 / avgd0)
    while((emitter.nshells > 1) and (emitter.done == False)):
        st += emitter.next_collision_time
        sim_time.append(st)

        for shell in emitter.shells:
            shell.update_shell(emitter.next_collision_time)

        # compute light travel time
        ltt = (Planck18.lookback_distance(emitter.z).to_value() * 3.08568e+24 / 3e+10) / (60*60*24*365)

        # compute observed time
        ot = (1+emitter.z)*((((ltt * c) - emitter.collision_shells[0].radius) / c) + st)

        emitter.update_tobs(st)

        print("###############################################")
        print("                 SIM INFO")
        print("###############################################")
        print(f'sim time:   {st:.4e}')
        print(f'obs time:   {ot:.4e}')
        print(f'nshells:    {emitter.nshells}')
        print("###############################################")

        avg = calculate_mean(emitter.shells, 'mass')
        avgm.append(avg / avgm0)
        avg = calculate_mean(emitter.shells, 'volume')
        avgv.append(avg / avgv0)
        avg = calculate_mean(emitter.shells, 'density')
        avgd.append(avg / avgd0)

        res = emitter.collision(emitter.collision_shells[0], emitter.collision_shells[1])

        if (res == 0):
            print("Super photospheric collision")
            obs_time.append(ot)
        else:
            print("Subphotospheric collision")

        print(f"collisions occurred: {emitter.ncoll}")
        emitter.calculate_collision_times()

    offset = min(obs_time)
    t_obs = [i - offset for i in obs_time]
    t90 = emitter.calculate_t90(t_obs)
    tv = emitter.calculate_tv(t90)

    f = open(path+'/output.log', 'w')

    print()
    print("------------------------------------------------------------------")
    print("                     OUTPUT PARAMETERS")
    print("------------------------------------------------------------------")
    print(f"Ncoll:           {emitter.ncoll}")
    print(f"tv:              {tv * 1000:.1f} ms")
    print(f"T90:             {t90:.2f} s")
    print(f"E_gamma_tot:     {emitter.e_gamma_tot:.2e} erg")
    print()
    print(f"photospheric     {emitter.photo}")
    print(f"sub-photospheric {emitter.subphoto}")
    print("------------------------------------------------------------------")

    lines = ["------------------------------------------------------------------", "                     OUTPUT PARAMETERS", "------------------------------------------------------------------", f"Ncoll:           {emitter.ncoll}", f"tv:              {tv * 1000:.1f} ms", f"T90:             {t90:.2f} s", f"E_gamma_tot:     {emitter.e_gamma_tot:.2e} erg", " ", f"photospheric     {emitter.photo}", f"sub-photospheric {emitter.subphoto}", "------------------------------------------------------------------", " "]
    f.write('\n'.join(lines))

    return t_obs, sim_time


def lorentz_factors():
    x, y = emitter.retrieve_lorentz_factors()
    x = [i / 1e6 for i in x]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    ax.scatter(x, y, marker='.')
    ax.set_xlabel(r'Initial shell radius $r_{k, 0}$ [$\times 10^6$ km]',fontsize=16)
    ax.set_ylabel(r'Initial Lorentz factor $\Gamma_{k, 0}$', fontsize=16)
    ax.set_yscale('log')
    # ax.set(xlim=[0, 8])
    ax.set(ylim=[10, 1e4])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    plt.savefig(impath+'/initial_lorentz_factor_dist.png')
    # plt.show()

def lc(t_obs):
    y = emitter.light_curve
    y.append(0)
    t_obs.append(0)

    tup = sorted(list(zip(t_obs, y)))
    x, y = zip(*tup)

    y2 = [i * 1e+6 for i in y]

    print(f"minimum flux {min(y):.4e} GeV/s/cm2 at time {x[y.index(min(y))]:.2f} s")
    print(f"maximum flux {max(y):.4e} GeV/s/cm2 at time {x[y.index(max(y))]:.2f} s")

    # fig = plt.figure(figsize=(8,6))
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes((.2,.3,.6,.6))

    ax.plot(x, y)

    # Plot cosmetics
    # ax.legend(loc="upper left")
    ax.set_xlabel(r't$_{obs}$ [s]',fontsize=16)
    ax.set_ylabel(r'Flux [GeV s$^{-1}$ cm$^{-2}$]', fontsize=16)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set(xlim=[-0.5, 80])
    # ax.set(ylim=[1e+2, 5e+3])
    # ax.set(ylim=[1e-8, 1e-3])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)

    ax2=fig.add_axes((.2,.1,.6,.2), sharex=ax)
    ax2.plot(x, y2)
    ax2.set_xlabel(r't$_{obs}$ [s]',fontsize=16)
    ax2.set_ylabel(r'Counts [A.U.]', fontsize=16)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax2.set(xlim=[-0.5, 80])
    # ax.set(ylim=[0, 10000])
    # ax2.set(ylim=[0, 700])
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', length=5)
    ax2.tick_params(axis='both', which='minor', length=2.5)
    ax2.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    plt.savefig(impath+'/light_curve.png')
    # plt.show()

def op_depth():
    # x = [i / 1e+3 for i in emitter.r_coll]
    x = emitter.r_coll
    y = [1/i for i in emitter.op_depth]
    # y = emitter.op_depth

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    ax.scatter(x, y, marker='.')
    ax.set_xlabel(r'$R_{C}$ [km]',fontsize=16)
    ax.set_ylabel(r'$\tau_{p \gamma}$', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(y=1, linestyle='--', color='k', linewidth=0.8)
    ax.text(1e+10, 1e+3, 'subphotospheric', fontsize=14)
    ax.fill_between([min(x)-1e+3, max(x)+1e+3], 1, max(y), color='k', alpha=0.2)

    # ax.set(xlim=[0, 8])
    # ax.set(ylim=[10, 1e4])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    plt.savefig(impath+'/optical_depth.png')
    # plt.show()
    return

def proton_energies():
    # x = emitter.r_coll
    # y = [i * 624.15 for i in emitter.e_prot]
    x = emitter.e_prot
    y = emitter.e_prot_n

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x, y, marker='.')
    # ax.axhline(1, marker='--')
    ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
    ax.set_ylabel(r'N(E$_{p}$) [GeV]', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)    # print(bin_x)
    # print(bin_y)

    plt.savefig(impath+'/prot_energy.png')
    # plt.show()
    return

def prot_spec():
    x = emitter.e_prot_n_
    y = emitter.e_prot_n

    # print(x)

    da = list(zip(x, y))
    bin_x = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
    bin_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in da:
        for j in range(len(i[0])):
            match math.floor(math.log10(i[0][j])):
                case -5:
                    bin_y[0] += i[1][j]
                case -4:
                    bin_y[1] += i[1][j]
                case -3:
                    bin_y[2] += i[1][j]
                case -2:
                    bin_y[3] += i[1][j]
                case -1:
                    bin_y[4] += i[1][j]
                case 0:
                    bin_y[5] += i[1][j]
                case 1:
                    bin_y[6] += i[1][j]
                case 2:
                    bin_y[7] += i[1][j]
                case 3:
                    bin_y[8] += i[1][j]
                case 4:
                    bin_y[9] += i[1][j]
                case 5:
                    bin_y[10] += i[1][j]
                case 6:
                    bin_y[11] += i[1][j]
                case 7:
                    bin_y[12] += i[1][j]
                case 8:
                    bin_y[13] += i[1][j]
                case 9:
                    bin_y[14] += i[1][j]
                case 10:
                    bin_y[15] += i[1][j]

    print(bin_x)
    print(bin_y)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(bin_x, bin_y)
    # ax.axhline(1, marker='--')
    ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
    ax.set_ylabel(r'N(E$_{p}$) [GeV]', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set(ylim=[0, 1e34])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    plt.savefig(impath+'/prot_energy_spec.png')

def e_vs_r():
    x = emitter.r_coll
    # y = [i * 1e4 for i in emitter.e_prot]
    # z = [i * 1e4 for i in emitter.e_prot_max]
    y = emitter.e_prot
    z = emitter.e_prot_max

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
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    ax.legend()
    plt.savefig(impath+'/prot_energy_rad_both.png')
    return

def loss_timescales():
    x = emitter.r_coll
    y = emitter.t_syn
    z = emitter.t_dyn

    tup = sorted(list(zip(x, y)))
    x, y = zip(*tup)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, label='synchrotron losses timescale')

    tup = sorted(list(zip(x, z)))
    x, z = zip(*tup)
    ax.plot(x, z, label='dynamical loss timescale')
    # ax.axhline(1e+9, linestyle='--')
    ax.set_xlabel(r'R$_{c}$ [km]',fontsize=16)
    ax.set_ylabel(r't$^{-1}$ [s$^{-1}$]', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    ax.legend()
    plt.savefig(impath+'/loss_timescales.png')
    return

def shock_ratio():
    x = emitter.shock_rat
    y = emitter.shock_rat2

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x, y, marker='.')
    ax.set_ylabel(r'Shock Ratio, $\Gamma_{fs} / \Gamma_{rs}$',fontsize=16)
    ax.set_xlabel(r'Shock Ratio, $\Gamma_f / \Gamma_s$', fontsize=16)
    ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    plt.savefig(impath+'/shock_rat.png')
    return

def print_avgs(sim_time):

    x = sim_time
    y = avgm
    y2 = avgv
    y3 = avgd

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.loglog(x, y, label=r'$\left< m \right> / \left< m_0 \right>$')
    ax.loglog(x, y2, label=r'$\left< V \right> / \left< V_0 \right>$')
    ax.loglog(x, y3, label=r'$\left< \rho \right> / \left< \rho_0 \right>$')
    ax.set_xlabel(r't [s]',fontsize=16)
    ax.set_ylabel(r'Avg', fontsize=16)
    ax.set_yscale('log')
    ax.set(xlim=[1, 1e+6])
    # ax.set(ylim=[10, 1e4])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=2.5)
    ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    ax.legend()
    # plt.savefig(impath+'/initial_lorentz_factor_dist.png')

def calculate_mean(list, val):
    sum = 0
    for i in list:
        sum += getattr(i, val)
    return sum / len(list)


def main():
    if (os.path.exists(path+('/collision.log'))):
        os.remove(path+('/collision.log'))
        f = open(path+('/collision.log'), 'x')
        f.close()
    if (os.path.exists(path+('/photons.log'))):
        os.remove(path+('/photons.log'))
        f = open(path+('/photons.log'), 'x')
        f.close()

    if (os.path.exists(path+('/protons.log'))):
        os.remove(path+('/protons.log'))
        f = open(path+('/protons.log'), 'x')
        f.close()

    t_obs, sim_time = sim()
    lc(t_obs)
    op_depth()
    # proton_energies()
    e_vs_r()
    # shock_ratio()
    loss_timescales()
    # print_avgs(sim_time)
    prot_spec()
    plt.show()


if __name__ ==  "__main__":
    main()
