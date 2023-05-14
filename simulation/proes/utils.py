import matplotlib.pyplot as plt
from shells import Shell, Emitter

def lorentz_factors(emitter):
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
