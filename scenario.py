"""
scenario.py  –  Delta-gamma P&L scenario analysis and parametric VaR

P&L ≈ Δ·ΔS + ½Γ·ΔS² + Vega·Δσ + Θ·Δt
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

import config
from black_scholes import BlackScholes, OptionSpec


class ScenarioEngine:

    def pnl_grid(self, opt: OptionSpec) -> pd.DataFrame:
        """Full scenario P&L table across spot and vol shocks."""
        g = BlackScholes.greeks(opt)
        rows = []
        for ds in config.SPOT_SHOCKS:
            for dv in config.VOL_SHOCKS:
                pnl = (g["delta"] * ds
                       + 0.5 * g["gamma"] * ds**2
                       + g["vega"] * dv
                       + g["theta"] * (1 / 365))
                rows.append({
                    "ΔSpot":    ds,
                    "ΔVol(%)":  round(dv * 100, 1),
                    "PnL":      round(pnl, 6),
                    "delta_pnl":round(g["delta"] * ds, 6),
                    "gamma_pnl":round(0.5 * g["gamma"] * ds**2, 8),
                    "vega_pnl": round(g["vega"] * dv, 6),
                })
        return pd.DataFrame(rows)

    def var_delta_gamma(self, opt: OptionSpec) -> dict:
        """Parametric VaR using delta-gamma-vega approximation."""
        g       = BlackScholes.greeks(opt)
        hold    = config.VAR_HOLDING_DAYS
        conf    = config.VAR_CONFIDENCE
        sigma_S = opt.sigma * opt.S * (hold / 252) ** 0.5
        z       = norm.ppf(conf)

        var_delta = abs(g["delta"]) * z * sigma_S
        var_gamma = 0.5 * g["gamma"] * sigma_S**2
        var_vega  = abs(g["vega"]) * 0.01 * z   # assume 1% vol move as shock
        var_total = var_delta + var_gamma + var_vega

        return {
            "confidence":    conf,
            "holding_days":  hold,
            "var_delta":     round(var_delta, 5),
            "var_gamma_adj": round(var_gamma, 5),
            "var_vega_adj":  round(var_vega, 5),
            "var_total":     round(var_total, 5),
        }
