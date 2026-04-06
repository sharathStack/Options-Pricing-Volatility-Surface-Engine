"""
black_scholes.py  –  Analytical BSM pricing, full Greeks, and IV solver

Greeks returned:
  delta, gamma, vega, theta, rho, vanna, volga, charm
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm
from scipy.optimize import brentq

OptionType = Literal["call", "put"]


@dataclass
class OptionSpec:
    S:           float          # spot
    K:           float          # strike
    T:           float          # time to expiry (years)
    r:           float          # domestic risk-free rate
    q:           float          # foreign / carry rate
    sigma:       float          # volatility
    option_type: OptionType = "call"

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def forward(self) -> float:
        return self.S * math.exp((self.r - self.q) * self.T)


class BlackScholes:

    @staticmethod
    def _d1d2(S, K, T, r, q, sigma):
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return d1, d1 - sigma * math.sqrt(T)

    @classmethod
    def price(cls, opt: OptionSpec) -> float:
        d1, d2 = cls._d1d2(opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma)
        disc   = math.exp(-opt.r * opt.T)
        grow   = math.exp(-opt.q * opt.T)
        if opt.is_call:
            return opt.S * grow * norm.cdf(d1) - opt.K * disc * norm.cdf(d2)
        return opt.K * disc * norm.cdf(-d2) - opt.S * grow * norm.cdf(-d1)

    @classmethod
    def greeks(cls, opt: OptionSpec) -> dict[str, float]:
        S, K, T, r, q, σ = opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma
        sqT  = math.sqrt(T)
        d1, d2 = cls._d1d2(S, K, T, r, q, σ)
        nd1  = norm.pdf(d1)
        disc = math.exp(-r * T)
        grow = math.exp(-q * T)
        sign = 1 if opt.is_call else -1
        Nd1  = norm.cdf(sign * d1)
        Nd2  = norm.cdf(sign * d2)

        delta  = sign * grow * Nd1
        gamma  = grow * nd1 / (S * σ * sqT)
        vega   = S * grow * nd1 * sqT / 100         # per 1% vol
        theta  = ((-S * grow * nd1 * σ / (2 * sqT)
                   - sign * r * K * disc * Nd2
                   + sign * q * S * grow * Nd1) / 365)
        rho    = sign * K * T * disc * Nd2 / 100    # per 1bp

        # Second-order
        vanna  = -grow * nd1 * d2 / σ
        volga  = S * grow * nd1 * sqT * d1 * d2 / σ
        charm  = (-grow * (nd1 * ((r - q) / (σ * sqT) - d2 / (2 * T))
                           + sign * q * Nd1)) / 365

        return dict(price=cls.price(opt), delta=delta, gamma=gamma,
                    vega=vega, theta=theta, rho=rho,
                    vanna=vanna, volga=volga, charm=charm)

    @classmethod
    def implied_vol(cls, opt: OptionSpec, market_price: float,
                    tol: float = 1e-7) -> float:
        """Brent's method IV solver – typically < 15 iterations."""
        def f(sigma):
            o = OptionSpec(opt.S, opt.K, opt.T, opt.r, opt.q, sigma, opt.option_type)
            return cls.price(o) - market_price
        try:
            return brentq(f, 1e-6, 10.0, xtol=tol, maxiter=200)
        except ValueError:
            return float("nan")
