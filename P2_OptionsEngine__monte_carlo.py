"""
monte_carlo.py  –  Monte Carlo pricing with variance reduction

Techniques:
  - Antithetic variates  (halves variance, no extra model calls)
  - Control variate      (BSM as control; ~10× variance reduction)
  - Barrier option path  (full trajectory, down-and-out / up-and-out)
"""
from __future__ import annotations
import math
import time

import numpy as np

import config
from black_scholes import BlackScholes, OptionSpec


class MonteCarlo:

    def __init__(self):
        np.random.seed(config.MC_SEED)
        self.n_paths = config.MC_N_PATHS
        self.n_steps = config.MC_N_STEPS

    def price_european(self, opt: OptionSpec,
                       antithetic: bool = True,
                       control_variate: bool = True) -> dict:
        S, K, T, r, q, σ = opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma
        dt   = T / self.n_steps
        disc = math.exp(-r * T)
        n    = self.n_paths // (2 if antithetic else 1)

        t0  = time.perf_counter_ns()
        Z   = np.random.standard_normal((n, self.n_steps))
        if antithetic:
            Z = np.vstack([Z, -Z])

        log_S = np.log(S) + np.cumsum(
            (r - q - 0.5 * σ**2) * dt + σ * math.sqrt(dt) * Z, axis=1)
        S_T   = np.exp(log_S[:, -1])
        payoff = np.maximum(S_T - K, 0) if opt.is_call else np.maximum(K - S_T, 0)

        if control_variate:
            Z_cv = np.random.standard_normal(len(S_T))
            S_T_cv = S * np.exp((r - q - 0.5*σ**2)*T + σ*math.sqrt(T)*Z_cv)
            cov_mat = np.cov(payoff, S_T_cv)
            beta    = cov_mat[0, 1] / max(cov_mat[1, 1], 1e-12)
            E_ST    = S * math.exp((r - q) * T)
            payoff  = payoff - beta * (S_T_cv - E_ST)

        mc_price = disc * payoff.mean()
        se       = disc * payoff.std() / math.sqrt(len(payoff))
        ci       = 1.96 * se
        bsm      = BlackScholes.price(opt)
        elapsed  = (time.perf_counter_ns() - t0) / 1e6

        return {
            "mc_price":    round(mc_price, 6),
            "bsm_price":   round(bsm, 6),
            "error_vs_bsm":round(abs(mc_price - bsm), 6),
            "std_error":   round(se, 7),
            "ci95_lower":  round(mc_price - ci, 6),
            "ci95_upper":  round(mc_price + ci, 6),
            "elapsed_ms":  round(elapsed, 2),
            "n_paths":     len(payoff),
        }

    def price_barrier(self, opt: OptionSpec, barrier: float,
                      barrier_type: str = "down-and-out") -> float:
        S, K, T, r, q, σ = opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma
        dt   = T / self.n_steps
        disc = math.exp(-r * T)
        Z    = np.random.standard_normal((self.n_paths, self.n_steps))
        paths = S * np.exp(np.cumsum(
            (r - q - 0.5*σ**2)*dt + σ*math.sqrt(dt)*Z, axis=1))

        if "down" in barrier_type:
            alive = paths.min(axis=1) > barrier
        else:
            alive = paths.max(axis=1) < barrier

        S_T    = paths[:, -1]
        payoff = np.where(alive,
                          np.maximum(S_T - K, 0) if opt.is_call else np.maximum(K - S_T, 0),
                          0)
        return round(disc * payoff.mean(), 6)

    def convergence_study(self, opt: OptionSpec) -> list[dict]:
        """Show how MC price converges as n_paths increases."""
        bsm = BlackScholes.price(opt)
        rows = []
        for n in [1_000, 5_000, 10_000, 50_000, 100_000]:
            orig = self.n_paths
            self.n_paths = n
            res  = self.price_european(opt, antithetic=True, control_variate=False)
            rows.append({"n_paths": n, "mc_price": res["mc_price"],
                          "bsm_price": bsm, "error": round(abs(res["mc_price"] - bsm), 6),
                          "se": res["std_error"]})
            self.n_paths = orig
        return rows
