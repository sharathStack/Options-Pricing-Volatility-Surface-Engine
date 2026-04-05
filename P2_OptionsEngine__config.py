"""
config.py  –  Options Pricing & Volatility Engine parameters
"""

# ── Reference Instrument  (GBPUSD FX option) ─────────────────────────────────
SPOT         = 1.3000
RATE_DOM     = 0.05        # domestic risk-free rate
RATE_FOR     = 0.04        # foreign (carry) rate
ATM_VOL      = 0.12        # initial ATM volatility estimate

# ── Heston Stochastic Volatility ──────────────────────────────────────────────
HESTON_KAPPA = 2.0         # mean-reversion speed
HESTON_THETA = 0.04        # long-run variance (≈ 20% vol)
HESTON_XI    = 0.50        # vol-of-vol
HESTON_RHO   = -0.70       # spot–variance correlation
HESTON_V0    = 0.04        # initial variance
HESTON_COS_N = 200         # COS truncation order

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_N_PATHS   = 100_000
MC_N_STEPS   = 252
MC_SEED      = 42

# ── SABR ──────────────────────────────────────────────────────────────────────
SABR_BETA    = 0.5         # CEV exponent (0=normal, 1=lognormal); often fixed

# ── Scenario engine ───────────────────────────────────────────────────────────
VAR_CONFIDENCE   = 0.99
VAR_HOLDING_DAYS = 1
SPOT_SHOCKS      = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]
VOL_SHOCKS       = [-0.03, -0.01, 0, 0.01, 0.03]

# ── Output ────────────────────────────────────────────────────────────────────
CHART_OUTPUT = "vol_surface_dashboard.png"
CHART_DPI    = 150
