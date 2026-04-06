# Options Pricing & Volatility Surface Engine
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![SciPy](https://img.shields.io/badge/Optimisation-SciPy-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
> A complete derivatives pricing library: analytical BSM with full Greeks, Heston stochastic volatility via COS method, Monte Carlo with 10× variance reduction, SABR smile calibration, and a full 5-tenor implied volatility surface.
---
Project Structure
```
project2_options_engine/
├── config.py         ← Instrument params, model params, scenario settings
├── black_scholes.py  ← BSM pricing, all 8 Greeks, Brent IV solver
├── heston.py         ← Heston SV model via COS (Fourier cosine) method
├── monte_carlo.py    ← MC with antithetic variates + control variate
├── sabr.py           ← SABR smile (Hagan 2002), Nelder-Mead calibration
├── vol_surface.py    ← Build, calibrate, and query full vol surface
├── scenario.py       ← Delta-gamma P&L grid + parametric VaR
├── dashboard.py      ← 3D surface, smile slices, Greeks, VaR dashboard
├── main.py           ← Entry point
└── requirements.txt
```
---
How to Run
```bash
cd project2_options_engine
pip install -r requirements.txt
python main.py
```
Expected terminal output:
```
[1] BSM Greeks (ATM Call, T=3M)
    price       0.023451
    delta       0.524812
    gamma       0.000312
    vega        0.003241
    theta      -0.000182
    vanna      -0.000041
    volga       0.000018
    charm      -0.000009

[2] IV Solver  Market price=0.025 → IV=12.84%

[3] SARIMA Calibration
    1W  α=0.1423  ρ=-0.412  ν=0.389  err=2.31e-08
    ...

Dashboard saved → vol_surface_dashboard.png
```
---
## Models Implemented

Model Method Use Case

Black-Scholes-Merton	Analytical (exact)	Fast pricing + full Greeks

Heston SV	COS Fourier truncation	Stochastic vol, skew pricing

Monte Carlo	Antithetic + control variate	Exotic / barrier options

SABR	Hagan (2002) approximation	Vol smile parametrisation

---
## Greeks Computed
`Δ` · `Γ` · `Θ` · `Vega` · `ρ` · `Vanna (dΔ/dσ)` · `Volga (dVega/dσ)` · `Charm (dΔ/dt)`
---
## Vol Surface

Tenors: 1W, 1M, 3M, 6M, 1Y

Moneyness grid: log(F/K) from −0.30 to +0.30 (13 strikes)

SABR calibrated independently per tenor slice

Interpolation available at arbitrary (K, T) via `vol_surface.get_vol(K, T)`

---
Variance Reduction (Monte Carlo)

Technique	Variance Reduction

Naive MC	1× (baseline)

+ Antithetic variates	~2×
+ Control variate (BSM)	~10×
---
## Dashboard Output

`vol_surface_dashboard.png` — 8-panel dashboard:

3D implied volatility surface (colourmap RdYlGn_r)

Smile slices: market dots vs SABR model line per tenor

ATM term structure with confidence band

Delta & gamma across strikes (dual y-axis)

P&L scenario heatmap (ΔSpot × ΔVol)

SABR calibration error per tenor (bar chart)

SABR parameters (α, ρ, ν) by tenor

1-day 99% parametric VaR summary panel

---
## References

Black & Scholes (1973). The Pricing of Options and Corporate Liabilities. JPE.

Heston (1993). A Closed-Form Solution for Options with Stochastic Volatility. RFS.

Hagan, Kumar, Lesniewski & Woodward (2002). Managing Smile Risk. Wilmott.

Fang & Oosterlee (2008). A Novel Pricing Method for European Options (COS). SIAM J. Sci. Comput.

---
Requirements
```
numpy>=1.26
pandas>=2.1
scipy>=1.11
matplotlib>=3.8
```
