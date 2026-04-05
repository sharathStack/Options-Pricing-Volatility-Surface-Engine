"""
heston.py  –  Heston (1993) stochastic volatility model via COS method

dS = (r-q)S dt + √v S dW₁
dv = κ(θ-v)dt + ξ√v dW₂,   corr(dW₁,dW₂)=ρ

Reference:
  Fang & Oosterlee (2008) "A Novel Pricing Method for European Options
  Based on the Fourier-Cosine Series Expansion"
"""
from __future__ import annotations
import cmath
import math
from typing import Literal

import numpy as np

import config

OptionType = Literal["call", "put"]


class HestonModel:

    def __init__(self):
        self.kappa = config.HESTON_KAPPA
        self.theta = config.HESTON_THETA
        self.xi    = config.HESTON_XI
        self.rho   = config.HESTON_RHO
        self.v0    = config.HESTON_V0
        self.N     = config.HESTON_COS_N

    def char_func(self, u: complex, T: float, r: float, q: float) -> complex:
        """Heston characteristic function (Albrecher stable form)."""
        i    = 1j
        κ, θ, ξ, ρ, v0 = self.kappa, self.theta, self.xi, self.rho, self.v0
        d   = cmath.sqrt((ρ * ξ * i * u - κ)**2 + ξ**2 * (i * u + u**2))
        g   = (κ - ρ * ξ * i * u - d) / (κ - ρ * ξ * i * u + d)
        exp_dT = cmath.exp(-d * T)
        C   = ((r - q) * i * u * T
               + κ * θ / ξ**2 * (
                   (κ - ρ * ξ * i * u - d) * T
                   - 2 * cmath.log((1 - g * exp_dT) / (1 - g))
               ))
        D   = ((κ - ρ * ξ * i * u - d) / ξ**2
               * (1 - exp_dT) / (1 - g * exp_dT))
        return cmath.exp(C + D * v0)

    def price(self, S: float, K: float, T: float,
              r: float, q: float,
              option_type: OptionType = "call") -> float:
        """Price European option via COS truncation of the CF."""
        a, b = -6.0, 6.0
        N    = self.N
        ks   = np.arange(N, dtype=float)
        u_k  = ks * math.pi / (b - a)
        x    = math.log(S / K)

        # Evaluate characteristic function at each u_k
        phi = np.array([self.char_func(complex(uk, 0), T, r, q) * cmath.exp(1j * uk * (x - a))
                        for uk in u_k], dtype=complex)

        # COS payoff coefficients (call: integration from 0 to b)
        c, d = 0.0, b
        chi  = np.zeros(N)
        psi  = np.zeros(N)
        for k in range(N):
            uk = u_k[k]
            if k == 0:
                chi[k] = math.exp(d) - math.exp(c)
                psi[k] = d - c
            else:
                chi[k] = (1 / (1 + uk**2)) * (
                    math.cos(uk * (d - a)) * math.exp(d)
                    - math.cos(uk * (c - a)) * math.exp(c)
                    + uk * (math.sin(uk * (d - a)) * math.exp(d)
                            - math.sin(uk * (c - a)) * math.exp(c))
                )
                psi[k] = (math.sin(uk * (d - a)) - math.sin(uk * (c - a))) / uk

        V_k = (2 / (b - a)) * (chi - psi)
        w   = np.ones(N); w[0] = 0.5

        price_call = K * math.exp(-r * T) * float(np.sum(w * phi.real * V_k))
        price_call = max(price_call, 0.0)

        if option_type == "call":
            return price_call
        # Put via put-call parity
        return max(price_call - S * math.exp(-q * T) + K * math.exp(-r * T), 0.0)
