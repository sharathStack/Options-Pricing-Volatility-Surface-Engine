"""
main.py  –  Options Pricing & Volatility Engine entry point
"""
import numpy as np
import pandas as pd

import config
from black_scholes import BlackScholes, OptionSpec
from heston        import HestonModel
from monte_carlo   import MonteCarlo
from vol_surface   import VolatilitySurface
from scenario      import ScenarioEngine
import dashboard


def main():
    print("═" * 60)
    print("  OPTIONS PRICING & VOLATILITY ENGINE")
    print("═" * 60)

    # Reference ATM call: GBPUSD 3-month
    opt = OptionSpec(S=config.SPOT, K=config.SPOT, T=0.25,
                     r=config.RATE_DOM, q=config.RATE_FOR,
                     sigma=config.ATM_VOL, option_type="call")

    # ── 1. BSM Pricing & Greeks ───────────────────────────────────────────────
    print("\n[1] BSM Greeks (ATM Call, T=3M)")
    g = BlackScholes.greeks(opt)
    for k, v in g.items():
        print(f"    {k:<10}  {v:>12.6f}")

    # ── 2. Implied Vol Solver ──────────────────────────────────────────────────
    print("\n[2] Implied Volatility Solver (Brent's method)")
    for mp in [0.020, 0.025, 0.030, 0.035, 0.040]:
        iv = BlackScholes.implied_vol(opt, mp)
        print(f"    Market price={mp:.3f}  →  IV={iv:.4%}")

    # ── 3. Heston Pricing ─────────────────────────────────────────────────────
    print("\n[3] Heston Model (COS method)")
    heston = HestonModel()
    for K in [1.27, 1.28, 1.29, 1.30, 1.31, 1.32, 1.33]:
        h_price = heston.price(config.SPOT, K, 0.25,
                               config.RATE_DOM, config.RATE_FOR, "call")
        b_price = BlackScholes.price(OptionSpec(config.SPOT, K, 0.25,
                                                config.RATE_DOM, config.RATE_FOR,
                                                config.ATM_VOL, "call"))
        print(f"    K={K:.3f}  Heston={h_price:.5f}  BSM={b_price:.5f}  "
              f"diff={h_price-b_price:+.5f}")

    # ── 4. Monte Carlo ─────────────────────────────────────────────────────────
    print("\n[4] Monte Carlo (100K paths, antithetic + CV)")
    mc     = MonteCarlo()
    mc_res = mc.price_european(opt)
    for k, v in mc_res.items():
        print(f"    {k:<22}  {v}")

    # Convergence study
    print("\n    Convergence Study:")
    for row in mc.convergence_study(opt):
        print(f"      n={row['n_paths']:>7,}  MC={row['mc_price']:.5f}  "
              f"BSM={row['bsm_price']:.5f}  err={row['error']:.5f}")

    # Barrier
    barrier_p = mc.price_barrier(opt, barrier=1.25, barrier_type="down-and-out")
    print(f"\n    Down-and-out barrier (B=1.25): {barrier_p:.5f}")

    # ── 5. Greeks across strikes ──────────────────────────────────────────────
    print("\n[5] Greeks across strike range")
    strikes = np.linspace(1.25, 1.35, 11)
    greek_rows = []
    for K in strikes:
        o = OptionSpec(config.SPOT, K, 0.25, config.RATE_DOM,
                       config.RATE_FOR, config.ATM_VOL, "call")
        gr = BlackScholes.greeks(o)
        greek_rows.append({"K": round(K,4), **{k: round(v,5) for k,v in gr.items()}})
    gdf = pd.DataFrame(greek_rows)
    print(gdf[["K","price","delta","gamma","vega","theta","vanna"]].to_string(index=False))

    # ── 6. Vol Surface calibration ────────────────────────────────────────────
    print("\n[6] Vol Surface Calibration (SABR)")
    vs = VolatilitySurface()
    vs.build_synthetic_market()
    vs.calibrate()

    # ── 7. Scenario & VaR ─────────────────────────────────────────────────────
    print("\n[7] Scenario P&L Grid & VaR")
    se  = ScenarioEngine()
    scen_df = se.pnl_grid(opt)
    print(scen_df.head(10).to_string(index=False))
    var = se.var_delta_gamma(opt)
    print(f"\n    Delta-Gamma VaR: {var}")

    # ── 8. Dashboard ──────────────────────────────────────────────────────────
    print("\n[8] Generating dashboard…")
    dashboard.plot_surface(vs, gdf, scen_df, var)

    print("\n  Done ✓")


if __name__ == "__main__":
    main()
