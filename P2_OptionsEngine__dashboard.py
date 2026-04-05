"""
dashboard.py  –  Volatility surface and options analytics visualisation
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
from vol_surface import VolatilitySurface, TENORS, TENOR_LABELS, MONEYNESS_GRID


COLORS = ["#ff4d6d","#f7b731","#26de81","#45aaf2","#a55eea"]
DARK, GRID_C = "#0f1117", "#1a1d2e"
WHITE = "#e8eaf6"


def _style(ax):
    ax.set_facecolor(GRID_C)
    ax.tick_params(colors=WHITE, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#2a2d3e")
    ax.title.set_color(WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.grid(alpha=0.18, color="#3a3d50")


def plot_surface(vs: VolatilitySurface,
                 greeks_df, scenario_df, var_result) -> None:

    market = vs.market_vols
    model  = vs.model_surface()

    fig = plt.figure(figsize=(20, 12), facecolor=DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── 1. 3D surface ─────────────────────────────────────────────────────────
    T_grid, M_grid = np.meshgrid(TENORS * 12, MONEYNESS_GRID, indexing="ij")
    ax1 = fig.add_subplot(gs[0, :2], projection="3d")
    ax1.set_facecolor(DARK)
    surf = ax1.plot_surface(M_grid, T_grid, market * 100,
                             cmap="RdYlGn_r", alpha=0.88, linewidth=0.2)
    ax1.set_xlabel("Log-Moneyness", color=WHITE, fontsize=7)
    ax1.set_ylabel("Tenor (M)", color=WHITE, fontsize=7)
    ax1.set_zlabel("IV (%)", color=WHITE, fontsize=7)
    ax1.set_title("Implied Volatility Surface (market)", color=WHITE, fontweight="bold")
    ax1.tick_params(colors=WHITE, labelsize=6)
    fig.colorbar(surf, ax=ax1, shrink=0.45, pad=0.05)

    # ── 2. Smile slices: market vs SABR ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2)
    for i, (T, label, col) in enumerate(zip(TENORS, TENOR_LABELS, COLORS)):
        F       = config.SPOT * np.exp((config.RATE_DOM - config.RATE_FOR) * T)
        strikes = F * np.exp(MONEYNESS_GRID)
        ax2.plot(MONEYNESS_GRID, market[i]*100, "o", color=col, markersize=3,
                 alpha=0.6)
        ax2.plot(MONEYNESS_GRID, model[i]*100, "-", color=col,
                 linewidth=1.5, label=label)
    ax2.set_title("Vol Smile: market ● vs SABR ─", fontweight="bold")
    ax2.set_xlabel("Log-Moneyness")
    ax2.set_ylabel("IV (%)")
    ax2.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE)

    # ── 3. ATM term structure ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3)
    atm_idx = len(MONEYNESS_GRID) // 2
    ax3.plot(TENORS*12, market[:, atm_idx]*100, "o-", color="#26de81",
             linewidth=2, markersize=6)
    ax3.fill_between(TENORS*12,
                     market[:, atm_idx]*100 - 0.4,
                     market[:, atm_idx]*100 + 0.4, alpha=0.2, color="#26de81")
    ax3.set_title("ATM Term Structure", fontweight="bold")
    ax3.set_xlabel("Tenor (months)")
    ax3.set_ylabel("ATM IV (%)")

    # ── 4. Greeks across strikes ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4)
    if greeks_df is not None:
        ax4.plot(greeks_df["K"], greeks_df["delta"], color="#45aaf2",
                 linewidth=2, label="Delta")
        ax4b = ax4.twinx()
        ax4b.plot(greeks_df["K"], greeks_df["gamma"]*1000,
                  color="#f7b731", linewidth=1.5, linestyle="--", label="Gamma×1000")
        ax4b.tick_params(colors=WHITE, labelsize=7)
        ax4.set_title("Delta & Gamma across Strikes", fontweight="bold")
        ax4.set_xlabel("Strike")
        ax4.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE, loc="upper left")
        ax4b.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE, loc="upper right")

    # ── 5. P&L Scenario heatmap (ΔSpot vs ΔVol) ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    _style(ax5)
    if scenario_df is not None:
        pivot = scenario_df.pivot(index="ΔVol(%)", columns="ΔSpot", values="PnL")
        im = ax5.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax5.set_xticks(range(len(pivot.columns)))
        ax5.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=6, color=WHITE)
        ax5.set_yticks(range(len(pivot.index)))
        ax5.set_yticklabels([f"{v:.1f}%" for v in pivot.index], fontsize=6, color=WHITE)
        ax5.set_title("P&L Scenario: ΔSpot vs ΔVol", fontweight="bold")
        fig.colorbar(im, ax=ax5, shrink=0.8)

    # ── 6. SABR calibration error ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    _style(ax6)
    errors = [vs.sabr_params[T]["fit_error"] * 1e8 for T in TENORS]
    bars   = ax6.bar(TENOR_LABELS, errors, color=COLORS, alpha=0.85)
    ax6.set_title("SABR Calibration Error (×1e-8)", fontweight="bold")
    ax6.set_ylabel("SSE ×1e-8")
    for bar, e in zip(bars, errors):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{e:.2f}", ha="center", fontsize=7, color=WHITE)

    # ── 7. SABR parameters ────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    _style(ax7)
    alphas = [vs.sabr_params[T]["alpha"] for T in TENORS]
    nus    = [vs.sabr_params[T]["nu"]    for T in TENORS]
    ax7.plot(TENOR_LABELS, alphas, "o-", color="#26de81", linewidth=2, label="α (vol level)")
    ax7.plot(TENOR_LABELS, nus,    "s-", color="#f7b731", linewidth=2, label="ν (vol-of-vol)")
    ax7.set_title("SABR Parameters by Tenor", fontweight="bold")
    ax7.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE)

    # ── 8. VaR summary ────────────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    _style(ax8)
    ax8.axis("off")
    if var_result:
        lines = [("Parametric VaR", ""),
                 (f"  Confidence",   f"{var_result['confidence']:.0%}"),
                 (f"  Holding days", f"{var_result['holding_days']}"),
                 ("", ""),
                 (f"  Delta VaR",    f"{var_result['var_delta']:.5f}"),
                 (f"  Gamma adj.",   f"{var_result['var_gamma_adj']:.5f}"),
                 (f"  Vega adj.",    f"{var_result['var_vega_adj']:.5f}"),
                 ("  ─────────────",""),
                 (f"  Total VaR",    f"{var_result['var_total']:.5f}"),]
        for i, (k, v) in enumerate(lines):
            color = "#f7b731" if k.startswith("  Total") else WHITE
            ax8.text(0.05, 0.95 - i*0.10, f"{k:<22}{v}", transform=ax8.transAxes,
                     fontsize=9, color=color, family="monospace",
                     fontweight="bold" if k.startswith("  Total") else "normal")
    ax8.set_title("Risk Metrics", fontweight="bold")

    fig.suptitle(f"Options Analytics Dashboard  ─  {config.SPOT} GBPUSD",
                 fontsize=14, fontweight="bold", color=WHITE, y=1.01)
    plt.savefig(config.CHART_OUTPUT, dpi=config.CHART_DPI,
                bbox_inches="tight", facecolor=DARK)
    print(f"\nDashboard saved → {config.CHART_OUTPUT}")
