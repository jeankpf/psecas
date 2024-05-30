import numpy as np
from psecas import Solver, ChebyshevExtremaGrid
from psecas.systems.hbi import HeatFluxDrivenBuoyancyInstability
from psecas import plot_solution

"""
    The linear solution for the heat-flux-driven buoyancy instability (HBI)
    in a quasi-global setup, i.e. periodic in x and non-periodic in z.

    See the following paper for more details:

    H. N. Latter, M. W. Kunz, 2012, MNRAS, 423, 1964
    The HBI in a quasi-global model of the intracluster medium

    The script gets the unstable solution shown in figure 3 in the paper.
    Changing mode between 0, 1, 2 and 3 gives the solution shown in the four panels.
"""

N = 2
zmin = 0
zmax = 1
grid = ChebyshevExtremaGrid(N, zmin, zmax)

beta = 1e5
Kn = 1 / 1500.
kx = 250

system = HeatFluxDrivenBuoyancyInstability(grid, beta, Kn, kx)

solver = Solver(grid, system)

mode = 0
Ns = np.hstack((np.arange(2, 5) * 16, np.arange(3, 12) * 32))
print(Ns)
#exit()
omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=True, tol=1e-5)
phi = np.arctan(vec[2].imag / vec[2].real)
phi = np.arctan(vec[mode].imag / vec[mode].real)
#phi = vec[mode].imag / vec[mode].real
print(phi)
solver.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)

print(omega)
plot_solution(system, smooth=True)

