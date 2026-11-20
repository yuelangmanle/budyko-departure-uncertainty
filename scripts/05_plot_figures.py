#!/usr/bin/env python3
"""
Generate 6 top-journal-grade, visually diverse 300 DPI figures (Fig1.png - Fig6.png):
  Fig1: 2x2 Spatial Atlas (GB & AUS maps with SE-AUS box) + Budyko density + Aridity-BFI regime contours
  Fig2: 1x2 Catchment rho1 violin+box distributions & variance ribbon + Theorem 1 1:1 parity & dumbbell decomposition
  Fig3: 1x2 S1 Hexbin parity density plot (n=10,432) + S3 GB->AUS extrapolation curves along Aridity Index
  Fig4: 1x2 S2 PUB Calibration-Winkler Pareto arrow plot + S1 Logit-CQR vs Split-CP heteroscedastic ribbon envelope
  Fig5: 1x2 1997-2021 Drought & Post-Drought dual-panel time series (Bias + ACI tracking) + SE vs North AUS dumbbell
  Fig6: 1x2 S2 PUB Lollipop feature importance + Binned BFI vs catchment memory rho1 curves (GB vs AUS) with partial-r inset
"""
import csv
import json
import math
import os
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection, LineCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# Colorblind-safe Okabe-Ito / Nature palette
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_DARK = "#222222"
C_GRAY = "#6E6E6E"


def fu_er(ai, omega):
    ai = np.maximum(np.asarray(ai, dtype=float), 1e-9)
    return 1.0 + ai - np.power(1.0 + np.power(ai, omega), 1.0 / omega)


def fu_rr(ai, omega):
    return np.clip(1.0 - fu_er(ai, omega), 0.0, 1.0)


def compute_s3_curves_if_needed():
    cache = os.path.join(RES_DIR, "s3_gb_to_aus_curves.npz")
    if os.path.exists(cache):
        return np.load(cache)

    attrs = {r["basin_id"]: r for r in csv.DictReader(open(os.path.join(DATA_DIR, "gb_aus_basin_attributes.csv"), encoding="utf-8"))}
    gb, aus = [], []
    for r in csv.DictReader(open(os.path.join(DATA_DIR, "gb_aus_annual_water_years.csv"), encoding="utf-8")):
        if r["has_lag1"] != "1":
            continue
        a = attrs[r["basin_id"]]
        def fl(k, d=np.nan):
            v = a.get(k, "")
            return float(v) if v not in ("", "nan", "None") else d
        rec = {
            "basin_id": r["basin_id"], "P": float(r["P_mm"]), "PET": float(r["PET_mm"]),
            "AI": float(r["AI_t"]), "RR": float(r["RR_t"]),
            "P1": float(r["P_lag1"]), "PET1": float(r["PET_lag1"]), "AI1": float(r["AI_lag1"]),
            "la": math.log10(max(fl("area_km2", 100.0), 0.1)), "el": fl("elev_mean_m"),
            "sl": fl("slope_proxy"), "ps": fl("p_seasonality"), "fo": fl("forest_frac"),
            "cl": fl("clay_pct"), "sa": fl("sand_pct"),
        }
        (gb if r["region"] == "GB" else aus).append(rec)

    def feat(recs):
        return np.array([[
            r["P"], r["PET"], r["AI"], math.log(max(r["AI"], 1e-4)),
            r["P1"], r["PET1"], r["AI1"], r["P"] / max(r["P1"], 1.0),
            r["la"], r["el"], r["sl"], r["ps"], r["fo"], r["cl"], r["sa"]
        ] for r in recs], dtype=float)

    X_gb, y_gb = feat(gb), np.array([r["RR"] for r in gb])
    X_aus, y_aus = feat(aus), np.array([r["RR"] for r in aus])
    ai_gb = np.array([r["AI"] for r in gb])
    ai_aus = np.array([r["AI"] for r in aus])

    w_gb = float(minimize_scalar(lambda w: float(np.mean((fu_rr(ai_gb, w) - y_gb) ** 2)), bounds=(1.1, 10.0), method="bounded").x)
    fu_gb_pred = fu_rr(ai_gb, w_gb)
    fu_aus_pred = fu_rr(ai_aus, w_gb)

    m_direct = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=42).fit(X_gb, y_gb)
    p_direct = np.clip(m_direct.predict(X_aus), 0.0, 1.0)

    m_res = HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.04, min_samples_leaf=40, l2_regularization=2.0, random_state=42).fit(X_gb, y_gb - fu_gb_pred)
    p_physres = np.clip(fu_aus_pred + m_res.predict(X_aus), 0.0, 1.0)

    np.savez_compressed(cache, ai_aus=ai_aus, y_aus=y_aus, fu_aus=fu_aus_pred, p_direct=p_direct, p_physres=p_physres)
    return np.load(cache)



def draw_basemap(ax, iso_codes, xlim, ylim, draw_states=False):
    geo_dir = os.path.join(DATA_DIR, "geo")
    c_path = os.path.join(geo_dir, "ne_50m_countries.geojson")
    s_path = os.path.join(geo_dir, "ne_50m_states.geojson")
    ax.set_facecolor("#EAF2F8")  # subtle ocean tint
    if os.path.exists(c_path):
        gj = json.load(open(c_path, encoding="utf-8"))
        patches = []
        for feat in gj["features"]:
            props = feat.get("properties", {})
            iso = props.get("ISO_A3") or props.get("ADM0_A3") or ""
            # Draw target countries in crisp warm ivory, neighbors in muted light gray
            is_tgt = iso in iso_codes
            geom = feat["geometry"]
            polys = [geom["coordinates"]] if geom["type"] == "Polygon" else (geom["coordinates"] if geom["type"] == "MultiPolygon" else [])
            for poly in polys:
                ring = poly[0]
                xs = [pt[0] for pt in ring]; ys = [pt[1] for pt in ring]
                if max(xs) < xlim[0] - 2 or min(xs) > xlim[1] + 2 or max(ys) < ylim[0] - 2 or min(ys) > ylim[1] + 2:
                    continue
                ax.add_patch(Polygon(
                    ring, closed=True,
                    facecolor="#F7F6F2" if is_tgt else "#E8E8E5",
                    edgecolor="#4A5568" if is_tgt else "#9AA5B1",
                    linewidth=0.65 if is_tgt else 0.4,
                    zorder=1
                ))
    if draw_states and os.path.exists(s_path):
        sgj = json.load(open(s_path, encoding="utf-8"))
        segs = []
        for feat in sgj["features"]:
            geom = feat["geometry"]
            lines = [geom["coordinates"]] if geom["type"] == "LineString" else (geom["coordinates"] if geom["type"] == "MultiLineString" else [])
            for line in lines:
                xs = [pt[0] for pt in line]; ys = [pt[1] for pt in line]
                if max(xs) < xlim[0] or min(xs) > xlim[1] or max(ys) < ylim[0] or min(ys) > ylim[1]:
                    continue
                segs.append(line)
        if segs:
            ax.add_collection(LineCollection(segs, colors="#8D99AE", linewidths=0.55, linestyles="--", alpha=0.75, zorder=2))
    ax.set_xlim(xlim); ax.set_ylim(ylim)

def make_fig1(attrs):
    gb = [r for r in attrs if r["region"] == "GB"]
    aus = [r for r in attrs if r["region"] == "AUS"]

    fig = plt.figure(figsize=(11.2, 8.4))
    # Symmetrical 2x2 grid: left column [0.07..0.46], right column [0.56..0.95] (exact alignment top & bottom!)
    ax1 = fig.add_axes([0.07, 0.56, 0.355, 0.38])
    cax1 = fig.add_axes([0.435, 0.56, 0.014, 0.38])
    ax2 = fig.add_axes([0.56, 0.56, 0.355, 0.38])
    cax2 = fig.add_axes([0.925, 0.56, 0.014, 0.38])
    ax3 = fig.add_axes([0.07, 0.08, 0.38, 0.38])
    ax4 = fig.add_axes([0.56, 0.08, 0.38, 0.38])

    # (a) GB spatial map (true geographic aspect ratio, box locked to exact grid alignment)
    gb_lon = np.array([float(r["lon"]) for r in gb])
    gb_lat = np.array([float(r["lat"]) for r in gb])
    gb_ai = np.array([float(r["long_term_AI"]) for r in gb])
    draw_basemap(ax1, {"GBR"}, xlim=(-10.8, 4.4), ylim=(49.6, 59.2), draw_states=False)
    sc1 = ax1.scatter(gb_lon, gb_lat, c=gb_ai, cmap="YlGnBu_r", s=17, vmin=0.12, vmax=0.95, alpha=0.90, edgecolors="#222222", linewidths=0.18, zorder=4)
    cb1 = plt.colorbar(sc1, cax=cax1)
    cb1.set_label(r"Long-term $AI = \overline{PET}/\overline{P}$", fontsize=8.4)
    ax1.set_aspect(1.0 / math.cos(math.radians(54.0)), adjustable="datalim")
    ax1.grid(True, ls=":", color="#CCCCCC", alpha=0.7)
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.set_title("(a) Great Britain Benchmark (CAMELS-GB, n = 644)")

    # (b) AUS spatial map with Mainland SE-Australia drought focus box (excluding Tasmania, lat > -39.2)
    aus_lon = np.array([float(r["lon"]) for r in aus])
    aus_lat = np.array([float(r["lat"]) for r in aus])
    aus_ai = np.array([float(r["long_term_AI"]) for r in aus])
    draw_basemap(ax2, {"AUS"}, xlim=(111.5, 155.5), ylim=(-44.8, -9.8), draw_states=True)
    sc2 = ax2.scatter(aus_lon, aus_lat, c=np.clip(aus_ai, 0.3, 4.0), cmap="inferno_r", s=17, vmin=0.3, vmax=4.0, alpha=0.90, edgecolors="#222222", linewidths=0.18, zorder=4)
    cb2 = plt.colorbar(sc2, cax=cax2)
    cb2.set_label(r"Long-term $AI$ (capped at 4.0)", fontsize=8.4)
    rect = Rectangle((138.0, -39.2), 16.0, 7.2, lw=1.3, ls="--", edgecolor=C_BLUE, facecolor="none", zorder=5)
    ax2.add_patch(rect)
    ax2.text(146.0, -30.8, "Mainland SE AUS\n(Vic / MDB)", color=C_BLUE, fontsize=7.3, ha="center", fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0), zorder=6)
    ax2.axhline(-25.0, color=C_GREEN, ls=":", lw=1.1, zorder=3)
    ax2.text(123.0, -23.5, "Northern Tropical (lat > -25°)", color=C_GREEN, fontsize=7.2, fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0), zorder=6)
    ax2.set_aspect(1.0 / math.cos(math.radians(-28.0)), adjustable="datalim")
    ax2.grid(True, ls=":", color="#CCCCCC", alpha=0.7)
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.set_title("(b) Australia Benchmark (CAMELS-AUS, n = 558)")

    # (c) Budyko space with family of Fu curves
    gb_er = 1.0 - np.array([float(r["long_term_RR"]) for r in gb])
    aus_er = 1.0 - np.array([float(r["long_term_RR"]) for r in aus])
    ai_grid = np.linspace(0.05, 7.2, 400)
    ax3.fill_between(ai_grid, np.minimum(ai_grid, 1.0), 1.05, color="#F0F0F0", alpha=0.7)
    ax3.plot(ai_grid, np.minimum(ai_grid, 1.0), "k--", lw=1.2, label="Energy / Water limits")
    for wv, ls in [(2.0, ":"), (3.09, "-"), (5.0, "-.")]:
        ax3.plot(ai_grid, fu_er(ai_grid, wv), color=C_DARK if wv == 3.09 else "#888888", lw=1.7 if wv == 3.09 else 1.0, ls=ls,
                 label=rf"Fu $\omega={wv}$" + (" (global fit)" if wv == 3.09 else ""))
    ax3.scatter(gb_ai, gb_er, s=14, alpha=0.55, color=C_BLUE, edgecolors="none", label="GB (energy-limited)")
    ax3.scatter(aus_ai, aus_er, s=14, alpha=0.55, color=C_ORANGE, edgecolors="none", label="AUS (water-limited)")
    ax3.axvline(1.0, color="#666666", ls=":", lw=0.9)
    ax3.set_xlim(0, 7.2); ax3.set_ylim(0, 1.05)
    ax3.set_xlabel(r"Long-term Aridity Index ($AI = \overline{PET}/\overline{P}$)")
    ax3.set_ylabel(r"Evaporative Ratio ($1 - \overline{Q}/\overline{P}$)")
    ax3.set_title("(c) Cross-Hemisphere Budyko Space (1,202 Catchments)")
    ax3.legend(loc="lower right", frameon=True, framealpha=0.92, ncol=2, fontsize=7.2)

    # (d) Climate-Storage regime scatter + KDE contours
    gb_bfi = np.array([float(r["bfi"]) for r in gb if r["bfi"] not in ("", "nan")])
    gb_aib = np.array([float(r["long_term_AI"]) for r in gb if r["bfi"] not in ("", "nan")])
    aus_bfi = np.array([float(r["bfi"]) for r in aus if r["bfi"] not in ("", "nan")])
    aus_aib = np.array([float(r["long_term_AI"]) for r in aus if r["bfi"] not in ("", "nan")])

    for x_v, y_v, col, lab in [(gb_aib, gb_bfi, C_BLUE, "GB (med AI=0.53, BFI=0.47)"), (aus_aib, aus_bfi, C_ORANGE, "AUS (med AI=1.37, BFI=0.22)")]:
        ax4.scatter(x_v, y_v, s=12, alpha=0.38, color=col, edgecolors="none", label=lab)
        lx = np.log10(x_v)
        xg, yg = np.mgrid[lx.min()-0.1:lx.max()+0.1:80j, 0.0:1.0:80j]
        kde = stats.gaussian_kde(np.vstack([lx, y_v]))
        z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
        ax4.contour(10 ** xg, yg, z, levels=4, colors=[col], linewidths=1.1, alpha=0.85)
    ax4.set_xscale("log")
    ax4.axvline(1.0, color="#666666", ls=":", lw=1.0)
    ax4.set_ylim(0, 1.02)
    ax4.set_xlabel(r"Long-term Aridity Index ($AI$, log scale)")
    ax4.set_ylabel("Baseflow Index (BFI)")
    ax4.set_title("(d) Kernel Density Contours of Climate–Storage Regimes")
    ax4.legend(loc="upper right", frameon=True, framealpha=0.92)

    fig.savefig(os.path.join(FIG_DIR, "Fig1.png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_fig2(attrs, diag, rho_map):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # (a) Violin + Boxplot of catchment-level lag-1 memory rho1 (GB vs AUS vs Near-Natural)
    gb_r = [v for k, v in rho_map.items() if k.startswith("GB_")]
    aus_r = [v for k, v in rho_map.items() if k.startswith("AUS_")]
    nat_ids = {r["basin_id"] for r in attrs if r["is_near_natural"] == "1"}
    nat_r = [v for k, v in rho_map.items() if k in nat_ids]

    data_v = [gb_r, aus_r, nat_r]
    cols_v = [C_BLUE, C_ORANGE, C_GREEN]
    pos = [1, 2, 3]
    parts = ax1.violinplot(data_v, positions=pos, widths=0.65, showmeans=False, showmedians=False, showextrema=False)
    for body, col in zip(parts["bodies"], cols_v):
        body.set_facecolor(col); body.set_alpha(0.32); body.set_edgecolor(col); body.set_linewidth(1.0)
    bp = ax1.boxplot(data_v, positions=pos, widths=0.18, patch_artist=True, showfliers=False,
                     medianprops=dict(color="white", lw=1.8), whiskerprops=dict(color=C_DARK, lw=1.1), capprops=dict(color=C_DARK, lw=1.1))
    for patch, col in zip(bp["boxes"], cols_v):
        patch.set_facecolor(col); patch.set_alpha(0.85); patch.set_edgecolor(C_DARK)

    ax1.axhline(0.0, color="#555555", ls="--", lw=0.9)
    iccs = [58.1, 57.4, 58.6]
    meds = [np.median(gb_r), np.median(aus_r), np.median(nat_r)]
    for i, (p, m, icc) in enumerate(zip(pos, meds, iccs)):
        ax1.text(p, 0.68, f"Med = {m:+.3f}\nICC = {icc:.1f}%", ha="center", va="top", fontsize=7.6,
                 bbox=dict(facecolor="#F8F9FA", edgecolor="#CCCCCC", pad=2.2))
    ax1.set_xticks(pos)
    ax1.set_xticklabels(["Great Britain\n(n = 644)", "Australia\n(n = 558)", "Near-Natural\n(n = 869)"])
    ax1.set_ylim(-0.55, 0.74)
    ax1.set_ylabel(r"Catchment Lag-1 Residual Autocorrelation ($\rho_1$)")
    ax1.set_title("(a) Distribution of Within-Basin Memory & Between-Basin ICC")

    # (b) Dumbbell / Decomposition Chart: Theorem 1 vs MC vs Cross-Fitted Climate Null & 95% CI
    p0 = json.load(open(os.path.join(RES_DIR, "crossfit_null_and_transfer.json"), encoding="utf-8"))["ClimatePreserving_Null_and_BasinBootstrap"]
    keys = ["ALL_1202", "GB_644", "AUS_558", "NEAR_NATURAL_869", "AUS_JulJun_WaterYear_Sensitivity"]
    ylabs = ["All Catchments\n(n = 1,202)", "Great Britain\n(n = 644)", "Australia\n(Apr–Mar, 558)", "Near-Natural\n(n = 869)", "Australia\n(Jul–Jun sens.)"]
    y = np.arange(len(keys))

    for i, k in enumerate(keys):
        if k == "AUS_JulJun_WaterYear_Sensitivity":
            eu = diag[k]["test_B_euler_null_dq_mm"]
            mc = eu
            obs = eu + diag[k]["test_B_excess_physical_dq_mm"]
            ant = diag[k]["test_A_wet_minus_dry_dq_mm"]
            ant_ci = None
            clim_null_a = None
        else:
            tb = diag[k]["test_B_euler_theorem_vs_mc_null"]
            eu = tb["euler_closed_form_null_dq_mm"]
            mc = tb["mc_null_mean_dq_mm"]
            obs = tb["obs_drying_minus_wetting_dq_mm"]
            ant = p0[k]["TestA_climate_residualized_mm"]
            ant_ci = p0[k]["TestA_climate_residualized_boot95CI"]
            clim_null_a = p0[k]["TestA_climate_preserving_null_mm"]

        ax2.plot([eu, obs], [i, i], color=C_ORANGE, lw=3.2, alpha=0.85, zorder=2,
                 label="Excess transition asymmetry (Test B)" if i == 0 else None)
        ax2.scatter(mc, i, s=66, color="#999999", marker="o", zorder=3,
                    label="200-iter. MC permutation null" if i == 0 else None)
        ax2.scatter(eu, i, s=36, color=C_DARK, marker="D", zorder=4,
                    label=r"Theorem 1: $-4\,\mathrm{Cov}(\Delta Q, F_P(P))$" if i == 0 else None)
        ax2.scatter(obs, i, s=66, color=C_ORANGE, marker="s", edgecolors=C_DARK, linewidths=0.5, zorder=4,
                    label="Observed drying − wetting (Test B)" if i == 0 else None)
        if clim_null_a is not None:
            ax2.scatter(clim_null_a, i - 0.21, s=44, facecolors="white", edgecolors=C_PURPLE, linewidths=1.4, marker="o", zorder=5,
                        label="Cross-fitted climate null (Test A)" if i == 0 else None)
        if ant_ci is not None:
            ax2.errorbar(ant, i - 0.21, xerr=[[ant - ant_ci[0]], [ant_ci[1] - ant]], fmt="^", color=C_BLUE,
                         ecolor=C_BLUE, elinewidth=1.2, capsize=2.5, ms=6.0, zorder=5,
                         label="Cross-fitted Test A resid. (95% CI)" if i == 0 else None)
        else:
            ax2.scatter(ant, i - 0.21, s=50, color=C_BLUE, marker="^", zorder=5)
        ax2.text((eu + obs) / 2, i + 0.25, f"+{obs - eu:.1f}", ha="center", va="center", fontsize=7.2, color="#A63603", fontweight="bold")

    ax2.axvline(0.0, color="#666666", ls=":", lw=0.9)
    ax2.set_yticks(y); ax2.set_yticklabels(ylabs); ax2.invert_yaxis()
    ax2.set_xlim(-7, 63); ax2.set_ylim(5.65, -0.58)
    ax2.set_xlabel(r"Runoff Anomaly Difference ($\mathrm{mm\ yr}^{-1}$)")
    ax2.set_title("(b) Analytical Null, Cross-Fitted Climate Null & Carry-Over")
    ax2.legend(loc="lower center", ncol=2, frameon=True, framealpha=0.96, fontsize=6.4, handlelength=1.6, columnspacing=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "Fig2.png"))
    plt.close(fig)


def make_fig3():
    d1 = np.load(os.path.join(RES_DIR, "s1_test_predictions.npz"))
    d3 = compute_s3_curves_if_needed()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # (a) S1 Hexbin Parity Plot (Observed vs Dynamic HGBR, n=10,432)
    hb = ax1.hexbin(d1["y_te"], d1["pred_te"], gridsize=38, cmap="Blues", mincnt=1, bins="log")
    cb = plt.colorbar(hb, ax=ax1, fraction=0.045, pad=0.03)
    cb.set_label(r"$\log_{10}(\mathrm{Count})$")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="1:1 Parity")
    ax1.text(0.04, 0.94,
             "S1 Gauged Temporal (n = 10,432)\n"
             "M4 Dyn. HGBR: RMSE = 0.0768, R² = 0.920\n"
             "M5 PhysRes:   RMSE = 0.0785, R² = 0.916\n"
             "M2 Dyn. ARX:  RMSE = 0.0887, R² = 0.893\n"
             "M1 Basin Fu:  RMSE = 0.1004, R² = 0.863",
             transform=ax1.transAxes, va="top", fontsize=7.4, family="monospace",
             bbox=dict(facecolor="white", edgecolor="#CCCCCC", alpha=0.93, pad=3.5))
    ax1.set_xlim(0, 1.0); ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("Observed Annual Runoff Ratio ($Q/P$)")
    ax1.set_ylabel("Predicted Annual Runoff Ratio (M4 HGBR)")
    ax1.set_title("(a) In-Support S1 Forecasting Parity Density")

    # (b) S3 GB->AUS Extrapolation Curves along Aridity Index (Observed vs Direct ML vs Fu vs PhysRes)
    ai = d3["ai_aus"]
    bins = np.array([0.3, 0.6, 0.9, 1.2, 1.6, 2.2, 3.0, 4.2, 6.5])
    bc, o_med, o_q25, o_q75, f_med, d_med, p_med = [], [], [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (ai >= bins[i]) & (ai < bins[i + 1])
        if np.sum(m) < 20:
            continue
        bc.append(float(np.median(ai[m])))
        o_med.append(float(np.median(d3["y_aus"][m])))
        o_q25.append(float(np.percentile(d3["y_aus"][m], 25)))
        o_q75.append(float(np.percentile(d3["y_aus"][m], 75)))
        f_med.append(float(np.median(d3["fu_aus"][m])))
        d_med.append(float(np.median(d3["p_direct"][m])))
        p_med.append(float(np.median(d3["p_physres"][m])))

    ax2.axvspan(0.3, 0.936, color="#E6F2FA", alpha=0.75, label="GB Training Support ($AI \\leq 0.94$)")
    ax2.fill_between(bc, o_q25, o_q75, color="#BDBDBD", alpha=0.35, label="Observed AUS IQR")
    ax2.plot(bc, o_med, "ko-", lw=2.0, ms=4.5, label="Observed AUS Median")
    ax2.plot(bc, d_med, "s--", color=C_ORANGE, lw=1.9, ms=4.5, label="Direct HGBR (R² = −1.12, Cov = 26%)")
    ax2.plot(bc, f_med, "^:", color=C_BLUE, lw=1.6, ms=4.5, label="Source Physical Fu (R² = +0.60)")
    ax2.plot(bc, p_med, "D-", color=C_GREEN, lw=2.1, ms=4.8, label="PhysRes HGBR (R² = +0.52, Cov = 86%)")
    ax2.axvline(0.936, color=C_BLUE, ls="--", lw=1.0)
    ax2.set_xscale("log")
    ax2.set_xticks([0.3, 0.5, 1.0, 2.0, 4.0, 6.0])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlim(0.3, 6.2); ax2.set_ylim(-0.02, 0.85)
    ax2.set_xlabel(r"Australian Annual Aridity Index ($AI_t$, log scale)")
    ax2.set_ylabel("Median Predicted / Observed Runoff Ratio ($Q/P$)")
    ax2.set_title("(b) Cross-Continent Extrapolation (GB → AUS) Along Aridity")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "Fig3.png"))
    plt.close(fig)


def make_fig4(m3_base, m3_up):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # (a) S2 PUB Calibration-Sharpness Trade-off Plot (GB Coverage vs AUS Coverage)
    s2r = m3_base["S2_PUB_Spatial_CV"]["by_region"]
    s2u = m3_up["S2_PUB_Upgraded"]
    pts = [
        ("Split-CP", s2r["GB"]["Split_CP"]["Coverage"]*100, s2r["AUS"]["Split_CP"]["Coverage"]*100, m3_base["S2_PUB_Spatial_CV"]["Split_CP"]["Winkler_Score"], C_GRAY, "o"),
        ("Local-CP", s2r["GB"]["Local_Adaptive_CP"]["Coverage"]*100, s2r["AUS"]["Local_Adaptive_CP"]["Coverage"]*100, m3_base["S2_PUB_Spatial_CV"]["Local_Adaptive_CP"]["Winkler_Score"], C_SKY, "^"),
        ("Global CQR", s2r["GB"]["CQR"]["Coverage"]*100, s2r["AUS"]["CQR"]["Coverage"]*100, m3_base["S2_PUB_Spatial_CV"]["CQR"]["Winkler_Score"], C_ORANGE, "s"),
        ("Mondrian CQR", s2r["GB"]["Mondrian_CQR"]["Coverage"]*100, s2r["AUS"]["Mondrian_CQR"]["Coverage"]*100, m3_base["S2_PUB_Spatial_CV"]["Mondrian_CQR"]["Winkler_Score"], C_GREEN, "D"),
        ("Mondrian\nLogit-CQR", s2u["Mondrian_Logit_CQR_GB"]["Coverage"]*100, s2u["Mondrian_Logit_CQR_AUS"]["Coverage"]*100, s2u["Mondrian_Logit_CQR"]["Winkler_Score"], C_BLUE, "*"),
    ]
    ax1.axvline(90.0, color="#666666", ls="--", lw=0.9)
    ax1.axhline(90.0, color="#666666", ls="--", lw=0.9)
    ax1.scatter([90.0], [90.0], s=140, facecolors="none", edgecolors="black", lw=1.4, zorder=2)
    ax1.annotate("Nominal 90%\ntarget", xy=(90.0, 90.0), xytext=(88.85, 89.25), fontsize=7.2, color="#333333",
                 ha="center", arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ann_cfg = {
        "Split-CP": ((89.65, 93.35), "left"),
        "Local-CP": ((88.45, 93.72), "center"),
        "Global CQR": ((87.32, 92.22), "left"),
        "Mondrian CQR": ((91.48, 90.26), "left"),
        "Mondrian\nLogit-CQR": ((91.05, 88.85), "center"),
    }
    for name, gx, ay, ws, col, mk in pts:
        ax1.scatter(gx, ay, s=165 if mk == "*" else 95, color=col, marker=mk, edgecolors=C_DARK, linewidths=0.7, zorder=4)
        (tx, ty), ha = ann_cfg[name]
        lbl = f"{name}\n(WS={ws:.3f})"
        ax1.annotate(lbl, xy=(gx, ay), xytext=(tx, ty), ha=ha, va="center", fontsize=7.2,
                     color=C_DARK, fontweight="bold" if "Mondrian" in name else "normal",
                     arrowprops=dict(arrowstyle="-", color="#999999", lw=0.65, shrinkA=2, shrinkB=5))

    # Arrow from Global CQR -> Mondrian CQR
    ax1.annotate("", xy=(91.2, 89.8), xytext=(88.4, 92.8),
                 arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.5, ls="-"))
    ax1.text(89.9, 91.55, "Mondrian\nre-balance", fontsize=7.4, color=C_GREEN, fontweight="bold", ha="center")
    ax1.set_xlim(87.2, 92.6); ax1.set_ylim(88.4, 94.15)
    ax1.set_xlabel("Great Britain 90% Coverage (%)")
    ax1.set_ylabel("Australia 90% Coverage (%)")
    ax1.set_title("(a) S2 PUB: Joint Regional Calibration & Winkler Score")
    ax1.grid(True, ls=":", alpha=0.45)

    # (b) Heteroscedastic 90% Envelope Along Aridity Index in S1 Test Set
    d1 = np.load(os.path.join(RES_DIR, "s1_test_predictions.npz"))
    ai = d1["ai_te"]
    bins = np.geomspace(0.15, 5.5, 16)
    bc, y_m, sp_lo, sp_hi, cq_lo, cq_hi = [], [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (ai >= bins[i]) & (ai < bins[i + 1])
        if np.sum(m) < 25:
            continue
        bc.append(float(np.median(ai[m])))
        y_m.append(float(np.median(d1["y_te"][m])))
        sp_lo.append(float(np.median(np.clip(d1["sp_lo"][m], 0, 1))))
        sp_hi.append(float(np.median(np.clip(d1["sp_hi"][m], 0, 1))))
        cq_lo.append(float(np.median(np.clip(d1["cqr_lo"][m], 0, 1))))
        cq_hi.append(float(np.median(np.clip(d1["cqr_hi"][m], 0, 1))))

    # Subsample background observed points for visual texture
    rng = np.random.default_rng(7)
    idx_sub = rng.choice(len(ai), size=1200, replace=False)
    ax2.scatter(ai[idx_sub], d1["y_te"][idx_sub], s=4, alpha=0.14, color="#555555", edgecolors="none", label="Observed test basin-years")
    ax2.fill_between(bc, sp_lo, sp_hi, color="#BBBBBB", alpha=0.35, label="Split-CP 90% envelope (homoscedastic)")
    ax2.fill_between(bc, cq_lo, cq_hi, color=C_ORANGE, alpha=0.32, label="CQR 90% envelope (heteroscedastic)")
    ax2.plot(bc, cq_lo, "-", color=C_ORANGE, lw=1.2)
    ax2.plot(bc, cq_hi, "-", color=C_ORANGE, lw=1.2)
    ax2.plot(bc, y_m, "k-", lw=1.6, label="Observed median $Q/P$")
    ax2.set_xscale("log")
    ax2.set_xticks([0.2, 0.5, 1.0, 2.0, 4.0])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlim(0.15, 5.5); ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel(r"Annual Aridity Index ($AI_t$, log scale)")
    ax2.set_ylabel("Annual Runoff Ratio ($Q/P$)")
    ax2.set_title("(b) S1 Adaptive 90% Conformal Envelope Across Aridity")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "Fig4.png"))
    plt.close(fig)


def make_fig5(m3_up):
    s4 = m3_up["S4_Upgraded_ACI_and_Spatial"]
    tr = s4["Yearly_Trace_1997_2021"]
    yrs = np.array([x["year"] for x in tr])
    cov_st = np.array([x["cov_static_cqr"] * 100 for x in tr])
    cov_aci = np.array([x["cov_aci"] * 100 for x in tr])
    fu_bias = np.array([x["fu_bias"] for x in tr])
    ml_bias = np.array([x["ml_bias"] for x in tr])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # (a) Time-series ribbon: Coverage of Static CQR vs Online ACI with shaded recovery gain
    ax1.axvspan(1997, 2009.5, color="#FDE8D7", alpha=0.55, label="Millennium Drought (1997–2009)")
    ax1.fill_between(yrs, cov_st, cov_aci, where=(cov_aci >= cov_st), color=C_GREEN, alpha=0.25, label="ACI recovery gain")
    ax1.plot(yrs, cov_st, "o--", color=C_GRAY, ms=3.8, lw=1.4, label="Static CQR coverage")
    ax1.plot(yrs, cov_aci, "s-", color=C_ORANGE, ms=4.2, lw=2.0, label="Online ACI-CQR coverage")
    ax1.axhline(90.0, color="black", ls=":", lw=1.1, label="Nominal 90%")
    ax1.set_xlim(1996.5, 2021.5); ax1.set_ylim(58, 102)
    ax1.set_xlabel("Australian Water Year")
    ax1.set_ylabel("Annual Empirical Coverage (%)")
    ax1.set_title("(a) Sequential Online ACI Tracking Through Drought & Post-Drought")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.94, fontsize=7.2)

    # (b) Paired Slope / Dumbbell Plot: Post-Drought (2010-2021) Spatial Contrast (SE AUS vs North Tropical)
    sp = s4["Spatial_PostDrought_2010_2021"]
    se = sp["SE_Australia_Victoria_MDB"]
    no = sp["Northern_Tropical_AUS"]

    # Top half: Coverage shift Static CQR -> Online ACI; Bottom inset/twin: Pre-1997 Fu Bias vs PhysRes Bias
    ax2.axvline(90.0, color="black", ls=":", lw=1.1)
    y_pos = [0, 1]
    labels = [f"Mainland SE AUS\n(Vic / MDB, n = {se['n']:,})\nPre-97 Fu Bias: {se['Pre97_Fu_Bias']:+.4f}",
              f"Northern Tropical\n(lat > -25°, n = {no['n']:,})\nPre-97 Fu Bias: {no['Pre97_Fu_Bias']:+.4f}"]
    st_vals = [se["Static_CQR"]["Coverage"] * 100, no["Static_CQR"]["Coverage"] * 100]
    ac_vals = [se["Online_ACI"]["Coverage"] * 100, no["Online_ACI"]["Coverage"] * 100]

    for i in range(2):
        ax2.annotate("", xy=(ac_vals[i], y_pos[i]), xytext=(st_vals[i], y_pos[i]),
                     arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=2.6))
        ax2.scatter(st_vals[i], y_pos[i], s=95, color=C_BLUE, marker="o", zorder=4, label="Static CQR (2010–2021)" if i == 0 else None)
        ax2.scatter(ac_vals[i], y_pos[i], s=110, color=C_ORANGE, marker="D", zorder=5, label="Online ACI-CQR (2010–2021)" if i == 0 else None)
        ax2.text(st_vals[i], y_pos[i] - 0.16, f"{st_vals[i]:.1f}%", ha="center", va="bottom", fontsize=8.0, color=C_BLUE, fontweight="bold")
        ax2.text(ac_vals[i], y_pos[i] - 0.16, f"{ac_vals[i]:.1f}%", ha="center", va="bottom", fontsize=8.0, color=C_ORANGE, fontweight="bold")
        ax2.text((st_vals[i] + ac_vals[i]) / 2, y_pos[i] + 0.14, f"+{ac_vals[i] - st_vals[i]:.1f} pp", ha="center", va="top", fontsize=7.8, color=C_GREEN, fontweight="bold")

    ax2.set_yticks(y_pos); ax2.set_yticklabels(labels); ax2.invert_yaxis()
    ax2.set_xlim(74, 93); ax2.set_ylim(1.55, -0.55)
    ax2.set_xlabel("Post-Drought Empirical 90% Coverage (%, 2010–2021)")
    ax2.set_title("(b) Spatial Contrast in Post-Drought Non-Recovery & ACI Gain")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.94, fontsize=7.6)
    ax2.grid(True, axis="x", ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "Fig5.png"))
    plt.close(fig)


def make_fig6(attrs, diag, rho_map, m3_up):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # (a) Cleveland Lollipop Plot of S2 PUB Permutation Feature Importance
    imp = m3_up["S2_PUB_Upgraded"]["Top10_PUB_Feature_Importance_DeltaRMSE"][:8]
    pretty_map = {
        "AI_t": r"$AI_t$", "AI_{t-1}": r"$AI_{t-1}$", "P_t": r"$P_t$",
        "P_seasonality": r"$P_{\mathrm{seas}}$", "PET_t": r"$PET_t$",
        "PET_{t-1}": r"$PET_{t-1}$", "log10(Area)": r"$\log_{10}(\mathrm{Area})$",
        "Elev": "Elevation",
    }
    fnames = [pretty_map.get(x["feature"], x["feature"]) for x in imp]
    vals = [x["delta_rmse"] for x in imp]
    y = np.arange(len(fnames))
    cols = [C_BLUE if i == 0 else (C_ORANGE if i == 1 else "#4B8BBE") for i in range(len(fnames))]

    ax1.hlines(y, 0, vals, color=cols, lw=2.2, alpha=0.85)
    ax1.scatter(vals, y, s=62, color=cols, edgecolors=C_DARK, linewidths=0.6, zorder=4)
    for i, v in enumerate(vals):
        ax1.text(v + 0.003, i, f"+{v:.4f}", va="center", fontsize=7.8, fontweight="bold" if i < 2 else "normal")
    ax1.set_yticks(y); ax1.set_yticklabels(fnames); ax1.invert_yaxis()
    ax1.set_xlim(0, 0.185)
    ax1.set_xlabel(r"Out-of-Sample Permutation Importance ($\Delta$RMSE)")
    ax1.set_title(r"(a) S2 PUB Predictors ($AI_{t-1}$ Ranks #2 Without Flow Data)")
    ax1.grid(True, axis="x", ls=":", alpha=0.45)

    # (b) Binned BFI vs Catchment Memory rho1 Curves (with 95% CI Ribbons) for GB vs AUS
    for reg, col, lab in [("GB", C_BLUE, "Great Britain (partial r = −0.27)"), ("AUS", C_ORANGE, "Australia (partial r = −0.19)")]:
        xs, ys = [], []
        for r in attrs:
            if r["region"] == reg and r["basin_id"] in rho_map and r["bfi"] not in ("", "nan"):
                xs.append(float(r["bfi"]))
                ys.append(float(rho_map[r["basin_id"]]))
        xs, ys = np.array(xs), np.array(ys)
        ax2.scatter(xs, ys, s=9, alpha=0.18, color=col, edgecolors="none")
        # 6 quantile bins
        qs = np.quantile(xs, np.linspace(0, 1, 7))
        bx, bm, blo, bhi = [], [], [], []
        for i in range(len(qs) - 1):
            m = (xs >= qs[i]) & (xs <= qs[i + 1])
            if np.sum(m) < 10:
                continue
            bx.append(float(np.median(xs[m])))
            mu = float(np.mean(ys[m]))
            se = float(np.std(ys[m], ddof=1) / math.sqrt(np.sum(m)))
            bm.append(mu); blo.append(mu - 1.96 * se); bhi.append(mu + 1.96 * se)
        ax2.fill_between(bx, blo, bhi, color=col, alpha=0.22)
        ax2.plot(bx, bm, "o-", color=col, lw=2.0, ms=4.8, label=lab)

    ax2.axhline(0.0, color="#555555", ls="--", lw=0.9)
    ax2.text(0.03, 0.05,
             "AUS additional controls:\n"
             r"  $\mathrm{CV}_Q$ vs $\rho_1$: $+0.14$ ($p = 6.5\times 10^{-4}$)" + "\n"
             r"  Forest $\mid AI$: $+0.18$ ($p = 2.6\times 10^{-5}$)",
             transform=ax2.transAxes, fontsize=7.3,
             bbox=dict(facecolor="#F8F9FA", edgecolor="#CCCCCC", pad=3.2))
    ax2.set_xlim(0.0, 0.96); ax2.set_ylim(-0.35, 0.55)
    ax2.set_xlabel("Baseflow Index (BFI)")
    ax2.set_ylabel(r"Within-Basin Lag-1 Memory ($\rho_1$)")
    ax2.set_title("(b) Resolving the Baseflow–Memory Paradox Across Regimes")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.6)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "Fig6.png"))
    plt.close(fig)


def main():
    attrs = list(csv.DictReader(open(os.path.join(DATA_DIR, "gb_aus_basin_attributes.csv"), encoding="utf-8")))
    diag = json.load(open(os.path.join(RES_DIR, "diagnostics_upgraded.json"), encoding="utf-8"))
    rho_map = json.load(open(os.path.join(RES_DIR, "basin_rho1_map.json"), encoding="utf-8"))
    m3_base = json.load(open(os.path.join(RES_DIR, "modeling_uq_summary.json"), encoding="utf-8"))
    m3_up = json.load(open(os.path.join(RES_DIR, "modeling_upgraded.json"), encoding="utf-8"))

    make_fig1(attrs)
    make_fig2(attrs, diag, rho_map)
    make_fig3()
    make_fig4(m3_base, m3_up)
    make_fig5(m3_up)
    make_fig6(attrs, diag, rho_map, m3_up)
    print("Generated all 6 top-journal diverse figures (Fig1.png - Fig6.png) in:", FIG_DIR)


if __name__ == "__main__":
    main()
