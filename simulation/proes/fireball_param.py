from simulator import Simulator
from shells import Shell, Emitter
import config
import matplotlib.pyplot as plt
import shutil
import os.path as path
import os
import pandas as pd
from pathlib import Path
import numpy as np
from mpl_toolkits import mplot3d

im_dir = '/home/eloise/Documents/Masters_research/masters-research/Simulation/simulation/da/fireball_parm/'

def test_decel_radius():
    da_dir = '/home/eloise/Documents/Masters_research/masters-research/Simulation/simulation/da/fireball_parm/dec_rad/'
    parms = [1e+9, 5e+9, 1e+10, 5e+10, 1e+11, 5e+11, 1e+12, 5e+12, 1e+13, 5e+13]

    print('###############################################')
    print('             DECELERATION RADIUS')
    print('###############################################')

    for i in parms:
        print(f'TESTING VALUE::       {i:.1e}')
        emitter = Emitter(i, 1e+3, 1000, 1e-2, 2)
        simulator = Simulator(emitter)
        simulator.sim_emitter_setup(500, 0, 0, 1)
        simulator.sim_start()
        simulator.save_to_csv_proto_spec()
        shutil.move(path.join(config.CSV_PATH, 'prot_energy_spec.csv'), path.join(da_dir, f'prot_energy_spec_{i:.1e}.csv'))

    print('###############################################')

    files = sorted(Path(da_dir).iterdir(), key = os.path.getmtime)
    file_names = [str(i.stem) for i in files]
    colours = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    for index in range(len(files)):
        name = file_names[index]
        df = pd.read_csv(str(files[index]), index_col = 0)
        ax.plot(df, label=f'{name[17:]} km', color=colours(index / (len(files) - 1)))
        # ax.scatter(x, y)
        ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
        ax.set_ylabel(r'E$^2$ N(E$_{p}$) [GeV cm$^{-2}$]', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    ax.legend(bbox_to_anchor=(0.9, 1), borderaxespad=0)
    plt.savefig(path.join(im_dir, 'dec_rad.png'), bbox_inches = "tight")
    # plt.show()


def test_n_sh():
    da_dir = '/home/eloise/Documents/Masters_research/masters-research/Simulation/simulation/da/fireball_parm/n_sh/'
    parms = [10, 50, 100, 500, 1000, 5000]

    print('###############################################')
    print('             NO. OF SHELLS')
    print('###############################################')

    for i in parms:
        print(f'TESTING VALUE::       {i}')
        emitter = Emitter(5.5e+11, 1e+3, i, 1e-2, 2)
        simulator = Simulator(emitter)
        simulator.sim_emitter_setup(500, 0, 0, 1)
        simulator.sim_start()
        simulator.save_to_csv_proto_spec()
        shutil.move(path.join(config.CSV_PATH, 'prot_energy_spec.csv'), path.join(da_dir, f'prot_energy_spec_{i}.csv'))

    print('###############################################')

    files = sorted(Path(da_dir).iterdir(), key = os.path.getmtime)
    file_names = [str(i.stem) for i in files]
    colours = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    for index in range(len(files)):
        name = file_names[index]
        df = pd.read_csv(str(files[index]), index_col = 0)
        ax.plot(df, label=f'{name[17:]}', color=colours(index / (len(files) - 1)))
        # ax.scatter(x, y)
        ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
        ax.set_ylabel(r'E$^2$ N(E$_{p}$) [GeV cm$^{-2}$]', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    ax.legend(bbox_to_anchor=(0.9, 1), borderaxespad=0)
    plt.savefig(path.join(im_dir, 'n_sh.png'))
    # plt.show()

def test_dist_from_ce():
    da_dir = '/home/eloise/Documents/Masters_research/masters-research/Simulation/simulation/da/fireball_parm/dist_from_ce/'
    parms = [1e+0, 5e+0, 1+1, 5e+1, 1e+2, 5e+2, 1e+3, 5e+3, 1e+4, 5e+4, 1e+5, 5e+5]

    print('###############################################')
    print('DISTANCE FROM INNERMOST SHELL TO CENTRAL ENGINE')
    print('###############################################')

    for i in parms:
        print(f'TESTING VALUE::       {i:.1e}')
        emitter = Emitter(5.5e+11, i, 1000, 1e-2, 2)
        simulator = Simulator(emitter)
        simulator.sim_emitter_setup(500, 0, 0, 1)
        simulator.sim_start()
        simulator.save_to_csv_proto_spec()
        shutil.move(path.join(config.CSV_PATH, 'prot_energy_spec.csv'), path.join(da_dir, f'prot_energy_spec_{i:.1e}.csv'))

    print('###############################################')

    files = sorted(Path(da_dir).iterdir(), key = os.path.getmtime)
    file_names = [str(i.stem) for i in files]
    colours = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    for index in range(len(files)):
        name = file_names[index]
        df = pd.read_csv(str(files[index]), index_col = 0)
        ax.plot(df, label=f'{name[17:]} km', color=colours(index / (len(files) - 1)))
        # ax.scatter(x, y)
        ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
        ax.set_ylabel(r'E$^2$ N(E$_{p}$) [GeV cm$^{-2}$]', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    ax.legend(bbox_to_anchor=(0.9, 1), borderaxespad=0)
    plt.savefig(path.join(im_dir, 'dist_from_ce.png'), bbox_inches = "tight")
    # plt.show()

def test_ce_uptime():
    da_dir = '/home/eloise/Documents/Masters_research/masters-research/Simulation/simulation/da/fireball_parm/ce_uptime/'
    # parms = [5, 1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    parms = np.logspace(-6, 1.3, 50)

    print('###############################################')
    print('             CENTRAL ENGINE UPTIME')
    print('###############################################')

    for i in parms:
        print(f'TESTING VALUE::       {i:.1e}')
        emitter = Emitter(5.5e+11, 1e+3, 1000, i, 2)
        simulator = Simulator(emitter)
        simulator.sim_emitter_setup(500, 0, 0, 1)
        simulator.sim_start()
        simulator.save_to_csv_proto_spec()
        shutil.move(path.join(config.CSV_PATH, 'prot_energy_spec.csv'), path.join(da_dir, f'prot_energy_spec_{i:.1e}.csv'))

    print('###############################################')

    files = sorted(Path(da_dir).iterdir(), key = os.path.getmtime)
    file_names = [str(i.stem) for i in files]
    colours = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    # ax = plt.axes(projection='3d')

    for index in range(len(files)):
        name = file_names[index]
        df = pd.read_csv(str(files[index]), header=None, names=['x', 'y'])
        ax.loglog(df, label=f'{name[17:]}', color=colours(index / (len(files) - 1)))
        # ax.scatter(parms[index], df.y[16], label=f'{name[17:]}', color=colours(index / (len(files) - 1)))
        # ax.plot3D(np.log10(df.x), np.repeat(index, 19), np.log10(df.y), label=f'{name[17:]} s', color=colours(index / (len(files) - 1)))
        # ax.scatter(x, y)
        ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
        # ax.set_xlabel(r'Central engine uptime [s]',fontsize=16)
        ax.set_ylabel(r'E$^2$ N(E$_{p}$) [GeV cm$^{-2}$]', fontsize=16)
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    ax.legend(bbox_to_anchor=(0.9, 1), borderaxespad=0)
    plt.savefig(path.join(im_dir, 'ce_uptime-3d.png'), bbox_inches = "tight")
    plt.show()

def test_lorentz_factor_dist():
    da_dir = '/home/eloise/Documents/Masters_research/masters-research/Simulation/simulation/da/fireball_parm/lorentz_factor/'
    parms = [ 5e0, 1e1, 1.5e1, 2e1, 2.5e1, 5e1, 1e2, 5e2, 1e3, 5e3]

    print('###############################################')
    print('         LORENTZ FACTOR DISTRIBUTION')
    print('###############################################')

    for i in parms:
        print(f'TESTING VALUE::       {i:.1e}')
        emitter = Emitter(5.5e+11, 1e+2, 1000, 1e-2, 2)
        simulator = Simulator(emitter)
        simulator.sim_emitter_setup(i, 0, 0, 1)
        simulator.sim_start()
        simulator.save_to_csv_proto_spec()
        shutil.move(path.join(config.CSV_PATH, 'prot_energy_spec.csv'), path.join(da_dir, f'prot_energy_spec_{i}.csv'))

    print('###############################################')

    # files = os.listdir(da_dir)

    files = sorted(Path(da_dir).iterdir(), key = os.path.getmtime)
    file_names = [str(i.stem) for i in files]
    colours = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    for index in range(len(files)):
        name = file_names[index]
        df = pd.read_csv(str(files[index]), index_col = 0)
        ax.loglog(df, label=f'{name[17:]}', color=colours(index / (len(files) - 1)))
        # ax.scatter(x, y)
        ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
        ax.set_ylabel(r'E$^2$ N(E$_{p}$) [GeV cm$^{-2}$]', fontsize=16)
        ax.set(xlim=[1e-6, 1e11])
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    ax.legend(bbox_to_anchor=(0.9, 0.6), borderaxespad=0)
    plt.savefig(path.join(im_dir, 'lorentz_factor.png'))
    # plt.show()

def test_baryonic_load():
    da_dir = '/home/eloise/Documents/Masters_research/masters-research/Simulation/simulation/da/fireball_parm/baryonic_load/'
    parms = np.logspace(-5, -4.34297e-6, 20)

    print('###############################################')
    print('                 BARYONIC LOAD')
    print('###############################################')

    for i in parms:
        print(f'TESTING VALUE::       {i:.1e}')
        emitter = Emitter(5.5e+11, 1e+2, 1000, 1e-2, 2, ep = i)
        simulator = Simulator(emitter)
        simulator.sim_emitter_setup(500, 0, 0, 1)
        simulator.sim_start()
        simulator.save_to_csv_proto_spec()
        shutil.move(path.join(config.CSV_PATH, 'prot_energy_spec.csv'), path.join(da_dir, f'prot_energy_spec_{i:.1e}.csv'))

    print('###############################################')

    # files = os.listdir(da_dir)

    files = sorted(Path(da_dir).iterdir(), key = os.path.getmtime)
    file_names = [str(i.stem) for i in files]
    colours = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(8,6))
    # fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1,1,1)

    for index in range(len(files)):
        name = file_names[index]
        df = pd.read_csv(str(files[index]), index_col = 0)
        ax.loglog(df, label=f'{name[17:]}', color=colours(index / (len(files) - 1)))
        # ax.scatter(x, y)
        ax.set_xlabel(r'$E_{p}$ [GeV]',fontsize=16)
        ax.set_ylabel(r'E$^2$ N(E$_{p}$) [GeV cm$^{-2}$]', fontsize=16)
        # ax.set(xlim=[1e-6, 1e11])
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='major', length=5)
        ax.tick_params(axis='both', which='minor', length=2.5)
        ax.tick_params(axis='both', which='both',direction='in',right=True,top=True)
    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.savefig(path.join(im_dir, 'baryonic_load.png'), bbox_inches = "tight")


# test_decel_radius()
# test_n_sh()
# test_dist_from_ce()
# test_ce_uptime()
# test_lorentz_factor_dist()
# test_baryonic_load()
