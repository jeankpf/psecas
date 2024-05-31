import numpy as np
from psecas import Solver, ChebyshevExtremaGrid
from psecas.systems.mti import MagnetoThermalInstability
from psecas import plot_solution

import matplotlib.pyplot as plt
from psecas import get_2Dmap

import os

"""
    The linear solution for the magnetothermal instability (MTI)
    in a quasi-global setup, i.e. periodic in x and non-periodic in z.

    Linearized equations for the MTI with anisotropic viscosity and heat
    conduction for a constant magnetic field in the x-direction.

    See the following paper for more details:

    Suppressed heat conductivity in the intracluster medium:
    implications for the magneto-thermal instability,
    Thomas Berlok, Eliot Quataert, Martin E. Pessah, Christoph Pfrommer
    https://arxiv.org/abs/2007.00018
"""

N = 64
zmin = 0
zmax = 1
grid = ChebyshevExtremaGrid(N, zmin, zmax)

beta = 1e6
Kn0 = 2000

current_folder = "eigenmodes"
if not os.path.exists(current_folder):
    os.mkdir(current_folder)

nmax = 15
mmax = 6
disps = np.ones((nmax,mmax))*-1
for n in range(1,nmax,3):
    kx = 2 * np.pi * n
    for mode in range(0, mmax):

        system = MagnetoThermalInstability(grid, beta, Kn0, kx)
        system.boundaries = [False, True, False, True, True]
        
        # No viscosity for now
        # system.nu0 = 0
        # system.make_background()
        
        solver = Solver(grid, system)
        
        Ns = np.hstack((np.arange(2, 5) * 16, np.arange(3, 12) * 32))
        omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=True, tol=1e-5)
        phi = np.arctan(vec[2].imag / vec[2].real)
        solver.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)
        
        # Normalize eigenmodes
        y = np.vstack(
            [
                system.result['dvx'].real,
                system.result['dvx'].imag,
                system.result['dvz'].real,
                system.result['dvz'].imag,
            ]
        )
        
        val = np.max(np.abs(y))
        for key in system.variables:
            system.result[key] /= val
        
        dvz = system.result['dvz'].real
        np.save(os.path.join(current_folder,'dvzn{0}m{1}.npy'.format(n,mode)), dvz)
        np.save(os.path.join(current_folder,'zn{0}m{1}.npy'.format(n,mode)), grid.zg)
        disps[n,mode] = omega
        
#        plot_solution(system, smooth=True, filename=os.path.join(current_folder,'eigenmodes.png'))
#        
#        plt.rc('image', origin='lower', cmap='RdBu')
#        plt.figure(2)
#        plt.clf()
#        fig, axes = plt.subplots(
#            num=2, sharex=True, sharey=True, nrows=2, ncols=3
#        )
#        xmin = 0.
##        xmax = 2 * np.pi / kx
#        xmax = 1.
#        Nx = 512
#        Nz = 512
#        extent = [xmin, xmax, system.grid.zmin, system.grid.zmax]
#        
#        dvx = get_2Dmap(system, 'dvx', xmin, xmax, Nx, Nz)
#        dvz = get_2Dmap(system, 'dvz', xmin, xmax, Nx, Nz)
#        dT = get_2Dmap(system, 'dT', xmin, xmax, Nx, Nz)
#        drho = get_2Dmap(system, 'drho', xmin, xmax, Nx, Nz)
#        
#        system.get_bx_and_by()
#        
#        dbx = get_2Dmap(system, 'dbx', xmin, xmax, Nx, Nz)
#        dbz = get_2Dmap(system, 'dbz', xmin, xmax, Nx, Nz)
#        
#        axes[0, 0].imshow(drho, extent=extent)
#        axes[0, 0].set_title(r'$\delta \rho/\rho$')
#        axes[0, 1].imshow(dvx, extent=extent)
#        axes[0, 1].set_title(r'$\delta v_x$')
#        axes[0, 2].imshow(dvz, extent=extent)
#        axes[0, 2].set_title(r'$\delta v_z$')
#        axes[1, 0].imshow(dT, extent=extent)
#        axes[1, 0].set_title(r'$\delta T/T$')
#        axes[1, 1].imshow(dbx, extent=extent)
#        axes[1, 1].set_title(r'$\delta b_x$')
#        axes[1, 2].imshow(dbz, extent=extent)
#        axes[1, 2].set_title(r'$\delta b_z$')
##        plt.show()
#        plt.savefig(os.path.join(current_folder,'map_eigenmodes.png'))
        
np.save(os.path.join(current_folder,'sigma.npy'), disps)
