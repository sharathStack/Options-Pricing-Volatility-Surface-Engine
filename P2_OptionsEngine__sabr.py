"""
sabr.py  –  SABR stochastic volatility model (Hagan et al. 2002)

σ_SABR(F, K, T; α, β, ρ, ν)  ←  parametric vol smile

Reference:
  Hagan, Kumar, Lesniewski & Woodward (2002)
  "Managing Smile Risk", Wilmott Magazine
"""
from __future__ import annotations
import math

import numpy as np
from scipy.optimize import minimize

import config


class SABR:

    @staticmethod
    def implied_vol(F: float, K: float, T: float,
                    alpha: float, beta: float,
                    rho: float, nu: float) -> float:
        """Hagan SABR vol approximation."""
        if abs(F - K) < 1e-10:           # ATM approximation
            t1 = alpha / (F ** (1 - beta))
            t2 = 1 + ((1-beta)**2 * alpha**2 / (24 * F**(2*(1-beta)))
                       + rho*beta*nu*alpha / (4 * F**(1-beta))
                       + (2 - 3*rho**2) * nu**2 / 24) * T
            return t1 * t2

        ln_FK  = math.log(F / K)
        FK_b   = (F * K) ** ((1 - beta) / 2)
        z      = nu / alpha * FK_b * ln_FK
        x_z    = math.log((math.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

        A = alpha / (FK_b * (1 + (1-beta)**2/24 * ln_FK**2
                              + (1-beta)**4/1920 * ln_FK**4))
        B = z / x_z if abs(x_z) > 1e-10 else 1.0
        C = 1 + ((1-beta)**2 * alpha**2 / (24 * FK_b**2)
                  + rho*beta*nu*alpha / (4 * FK_b)
                  + (2 - 3*rho**2) * nu**2 / 24) * T
        return A * B * C

    @classmethod
    def calibrate(cls, F: float, T: float,
                  strikes: np.ndarray,
                  market_vols: np.ndarray,
                  beta: float = None) -> dict:
        """Calibrate (α, ρ, ν) by least-squares, with β fixed or free."""
        beta = beta if beta is not None else config.SABR_BETA

        def objective(params):
            alpha, rho, nu = params
            if alpha <= 0 or nu <= 0 or abs(rho) >= 0.999:
                return 1e10
            model_vols = np.array([cls.implied_vol(F, K, T, alpha, beta, rho, nu)
                                   for K in strikes])
            return float(np.sum((model_vols - market_vols)**2))

        # ATM vol as initial alpha guess
        atm_vol  = np.interp(F, strikes, market_vols)
        alpha0   = atm_vol * (F ** (1 - beta))
        res = minimize(objective, x0=[alpha0, -0.3, 0.4],
                       method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 8000})
        alpha, rho, nu = res.x
        return dict(alpha=round(alpha, 6), beta=beta,
                    rho=round(rho, 6), nu=round(nu, 6),
                    fit_error=round(res.fun, 10))

    @classmethod
    def smile(cls, F: float, T: float,
              strikes: np.ndarray, params: dict) -> np.ndarray:
        """Generate a model vol smile for a set of strikes."""
        return np.array([cls.implied_vol(F, K, T, params["alpha"],
                                         params["beta"], params["rho"],
                                         params["nu"])
                         for K in strikes])
