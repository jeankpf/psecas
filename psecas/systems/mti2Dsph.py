class MagnetoThermalInstability:
    """
       The linear solution for the magnetothermal instability (MTI)
       in a quasi-global setup, i.e. periodic in phi and non-periodic in r.

       Linearized equations for the MTI with anisotropic viscosity and heat
       conduction for a constant magnetic field in the phi-direction.

       See the following paper for more details:

       Suppressed heat conductivity in the intracluster medium:
       implications for the magneto-thermal instability,
       Thomas Berlok, Eliot Quataert, Martin E. Pessah, Christoph Pfrommer
       https://arxiv.org/abs/2007.00018
    """

    def __init__(self, grid, beta, Kn0, m):
        # Problem parameters

        self._beta = beta
        self._Kn0 = Kn0

        self.m = m

        self.mu0 = 1.0
        self.p0 = 1.0
        self.rho0 = 1.0
        self.T0 = self.p0 / self.rho0

        self.H0 = 1.0
        self.Lr = 2.0

        self.set_va_and_B0()
        self.set_nu_and_chi()

        self.grid = grid
        self.grid.bind_to(self.make_background)

        # Variables to solve for
        self.variables = ["drho", "dA", "dvphi", "dvr", "dT"]

        self.labels = [
            r"$\delta \rho$",
            r"$\delta A$",
            r"$\delta v_{\varphi}$",
            r"$\delta v_r$",
            r"$\delta T$",
        ]

        # Boundary conditions
        self.boundaries = [True, False, False, False, False]

        # Create initial background
        self.make_background()

        # Number of equations in system
        self.dim = len(self.variables)

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "sigma"

        # Equations
        eq1 = "sigma*drho = -1j*m/r*dvphi -dlnrhodr*dvr -1.0*dr(dvr)"
        eq2 = "sigma*dA = 1.0*dvr"
        eq3 = (
            "sigma*dvphi = -1j*m/r*p/rho*drho -1j*m/r*p/rho*dT"
            + "-nu*4/3*(m/r)**2*dvphi -nu*1j*m/r*2/3*dr(dvr)"
        )
        eq4 = (
            "sigma*dvr = -T*dr(drho) -T*dr(dT) -T*dlnpdr*dT"
            + "+va**2*dr(dr(dA)) -va**2*(m/r)**2*dA"
            + "-1/rho*1j*m/r*drhonudr*2/3*dvphi -1j*m/r*nu*2/3*dr(dvphi)"
            + "+1/rho*drhonudr*1/3*dr(dvr) +nu*1/3*dr(dr(dvr))"
        )
        eq5 = (
            "sigma*dT = -1j*m/r*2/3*dvphi -2/3*dr(dvr) -dlnTdr*dvr"
            + "-2/3*(m/r)**2*kappa*dT -2/3*(m/r)**2*kappa*dlnTdr*dA"
        )

        self.equations = [eq1, eq2, eq3, eq4, eq5]

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        self.set_va_and_B0()
        self.make_background()

    @property
    def Kn0(self):
        return self._Kn0

    @Kn0.setter
    def Kn0(self, value):
        self._Kn0 = value
        self.set_nu_and_chi()
        self.make_background()

    def set_nu_and_chi(self):
        self.nu0 = 0.48 / self._Kn0
        self.chi0 = 24.0 / self._Kn0

    def set_va_and_B0(self):
        from numpy import sqrt

        self.B0 = sqrt(2 * self.p0 / self._beta)
        self.va = self.B0 / sqrt(self.mu0 * self.rho0)

    def get_bx_and_by(self):
        """Calculate dbphi and dbr. Requires a solution stored!"""
        import numpy as np

        self.grid.make_grid()
        self.result.update(
            {
                "dbphi": -np.matmul(self.grid.d1, self.result["dA"]),
                "dbr": 1j * self.m/r * self.result["dA"],
            }
        )

    def make_background(self):
        """Functing for creating the background profiles.
        Returns symbolic expressions (as a function of r) """
        import sympy as sym
        import numpy as np
        from sympy import exp, lambdify

        r = sym.symbols("r")

        rg = self.grid.zg

        globals().update(self.__dict__)

        # Define Background Functions
        rho_sym = rho0 * (1 - r / (3 * H0)) ** 2
        p_sym = p0 * (1 - r / (3 * H0)) ** 3

        T_sym = p_sym / rho_sym
        nu_sym = nu0 * T_sym ** (5 / 2)

        self.p = lambdify(r, p_sym)(rg)
        self.rho = lambdify(r, rho_sym)(rg)
        self.dpdr = lambdify(r, sym.diff(p_sym, r))(rg)
        self.dlnrhodr = lambdify(r, sym.diff(rho_sym, r) / rho_sym)(rg)
        self.dlnTdr = lambdify(r, sym.diff(T_sym, r) / T_sym)(rg)
        self.drhonudr = lambdify(r, sym.diff(rho_sym * nu_sym, r))(rg)

        self.T = self.p / self.rho
        self.chi = chi0 * self.T ** (5 / 2)
        self.nu = nu0 * self.T ** (5 / 2)
        self.kappa = self.chi * self.T / self.p
        self.dlnpdr = self.dpdr / self.p
