"""
vol_surface.py  –  Build, calibrate, and query an implied volatility surface

Tenors: 1W, 1M, 3M, 6M, 1Y
Moneyness grid: log(F/K) from -0.30 to +0.30
SABR is calibrated independently per tenor slice.
"""
from __future__ import annotations
import math

import numpy as np

import config
from sabr import SABR


TENOR_LABELS = ["1W", "1M", "3M", "6M", "1Y"]
TENORS = np.array([1/52, 1/12, 3/12, 6/12, 1.0])
MONEYNESS_GRID = np.linspace(-0.30, 0.30, 13)   # log-moneyness


class VolatilitySurface:

    def __init__(self):
        self.S    = config.SPOT
        self.r    = config.RATE_DOM
        self.q    = config.RATE_FOR
        self.sabr_params: dict[float, dict] = {}   # tenor → SABR params
        self.market_vols: np.ndarray | None = None

    def build_synthetic_market(self) -> np.ndarray:
        """
        Generate a realistic synthetic FX vol surface.
        Shape: (n_tenors, n_moneyness)
        """
        np.random.seed(1)
        surface = np.zeros((len(TENORS), len(MONEYNESS_GRID)))
        for i, T in enumerate(TENORS):
            atm = 0.09 + 0.025 * math.sqrt(T)
            for j, m in enumerate(MONEYNESS_GRID):
                skew   = -0.06 * m                    # negative skew (FX put premium)
                smile  =  0.04 * m**2                 # wings higher
                surface[i, j] = (atm + skew + smile
                                 + np.random.normal(0, 0.0008))
        self.market_vols = np.clip(surface, 0.02, 0.60)
        return self.market_vols

    def calibrate(self) -> None:
        """Fit SABR to each tenor slice of the market vol surface."""
        if self.market_vols is None:
            self.build_synthetic_market()

        print(f"\n  Calibrating SABR across {len(TENORS)} tenor slices…")
        for i, T in enumerate(TENORS):
            F       = self.S * math.exp((self.r - self.q) * T)
            strikes = F * np.exp(MONEYNESS_GRID)
            vols    = self.market_vols[i]
            params  = SABR.calibrate(F, T, strikes, vols)
            self.sabr_params[T] = params
            print(f"    {TENOR_LABELS[i]:4s}  α={params['alpha']:.4f}  "
                  f"ρ={params['rho']:.4f}  ν={params['nu']:.4f}  "
                  f"err={params['fit_error']:.2e}")

    def get_vol(self, K: float, T: float) -> float:
        """Interpolate / extrapolate a vol for arbitrary (K, T)."""
        tenor_arr = TENORS
        # Find nearest tenor bracket
        if T <= tenor_arr[0]:
            p = self.sabr_params[tenor_arr[0]]
        elif T >= tenor_arr[-1]:
            p = self.sabr_params[tenor_arr[-1]]
        else:
            idx  = np.searchsorted(tenor_arr, T)
            T_lo, T_hi = tenor_arr[idx-1], tenor_arr[idx]
            F_lo = self.S * math.exp((self.r - self.q) * T_lo)
            F_hi = self.S * math.exp((self.r - self.q) * T_hi)
            v_lo = SABR.implied_vol(F_lo, K, T_lo, **{k: v for k,v in
                                    self.sabr_params[T_lo].items() if k != "fit_error"})
            v_hi = SABR.implied_vol(F_hi, K, T_hi, **{k: v for k,v in
                                    self.sabr_params[T_hi].items() if k != "fit_error"})
            w    = (T - T_lo) / (T_hi - T_lo)
            return v_lo * (1 - w) + v_hi * w
        F = self.S * math.exp((self.r - self.q) * T)
        return SABR.implied_vol(F, K, T, **{k: v for k, v in p.items()
                                            if k != "fit_error"})

    def model_surface(self) -> np.ndarray:
        """Return the SABR model vol surface array (same shape as market)."""
        surface = np.zeros_like(self.market_vols)
        for i, T in enumerate(TENORS):
            F       = self.S * math.exp((self.r - self.q) * T)
            strikes = F * np.exp(MONEYNESS_GRID)
            p       = self.sabr_params[T]
            surface[i] = SABR.smile(F, T, strikes, p)
        return surface
