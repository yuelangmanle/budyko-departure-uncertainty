#!/usr/bin/env python3
"""
Generate 6 publication-ready 300 DPI figures integrating spatial catchment maps,
closed-form Euler theorem verification, physics-anchored residual ML rescue,
Logit-manifold Mondrian CQR, online ACI drought trajectories, and feature importance.
"""
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.2,
    "axes.labelsize": 9.2,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_GB = "#1f77b4"
C_AUS = "#d95f02"
C_GREEN = "#2ca02c"
C_GRAY = "#7f7f7f"
C_DARK = "#222222"


def fu_er(ai, omega):
    ai = np.maximum(np.asarray(ai, dtype=float), 1e-9)
    return 1.0 + ai - np.power(1.0 + np.power(ai, omega), 1.0 / omega)


def load_attrs():
    with open(os.path.join(DATA_DIR, "gb_aus_basin_attributes.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_fig1(attrs):
    gb = [r for r in attrs if r["region"] == "GB"]
    aus = [r for r in attrs if r["region"] == "AUS"]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # (a) GB spatial map colored by AI
    gb_lon = np.array([float(r["lon"]) for r in gb])
    gb_lat = np.array([float(r["lat"]) for r in gb])
    gb_ai = np.array([float(r["long_term_AI"]) for r in gb])
    sc1 = ax1.scatter(gb_lon, gb_lat, c=gb_ai, cmap="viridis_r", s=16, vmin=0.1, vmax=1.0, alpha=0.85, edgecolors="none")
    cb1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.03)
    cb1.set_label("Long-term AI (PET/P)")
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.set_title("(a) CAMELS-GB Spatial Distribution (n = 644)")
    ax1.grid(True, ls=":", alpha=0.35)

    # (b) AUS spatial map colored by log10(AI)
    aus_lon = np.array([float(r["lon"]) for r in aus])
    aus_lat = np.array([float(r["lat"]) for r in aus])
    aus_ai = np.array([float(r["long_term_AI"]) for r in aus])
    sc2 = ax2.scatter(aus_lon, aus_lat, c=np.clip(aus_ai, 0.3, 4.0), cmap="YlOrRd", s=16, vmin=0.3, vmax=4.0, alpha=0.85, edgecolors="none")
    cb2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.03)
    cb2.set_label("Long-term AI (clipped at 4.0)")
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.set_title("(b) CAMELS-AUS Spatial Distribution (n = 558)")
    ax2.grid(True, ls=":", alpha=0.35)

    # (c) Budyko space
    gb_er = 1.0 - np.array([float(r["long_term_RR"]) for r in gb])
    aus_er = 1.0 - np.array([float(r["long_term_RR"]) for r in aus])
    ai_grid = np.linspace(0.05, 7.2, 400)
    ax3.plot(ai_grid, np.minimum(ai_grid, 1.0), "k--", lw=1.2, label="Energy / Water limits")
    ax3.plot(ai_grid, fu_er(ai_grid, 3.09), color=C_DARK, lw=1.6, label=r"Global Fu ($\omega = 3.09$)")
    ax3.scatter(gb_ai, gb_er, s=12, alpha=0.5, color=C_GB, edgecolors="none", label="GB (energy-limited)")
    ax3.scatter(aus_ai, aus_er, s=12, alpha=0.5, color=C_AUS, edgecolors="none", label="AUS (water-limited)")
    ax3.axvline(1.0, color="#666666", ls=":", lw=1.0)
    ax3.set_xlim(0, 7.2); ax3.set_ylim(0, 1.05)
    ax3.set_xlabel(r"Long-term Aridity Index ($AI = \overline{PET}/\overline{P}$)")
    ax3.set_ylabel(r"Evaporative Ratio ($1 - \overline{Q}/\overline{P}$)")
    ax3.set_title("(c) Cross-Hemisphere Budyko Space (1,202 Catchments)")
    ax3.legend(loc="lower right", frameon=True, framealpha=0.92)

    # (d) Aridity vs BFI
    gb_bfi = np.array([float(r["bfi"]) for r in gb if r["bfi"] not in ("", "nan")])
    gb_aib = np.array([float(r["long_term_AI"]) for r in gb if r["bfi"] not in ("", "nan")])
    aus_bfi = np.array([float(r["bfi"]) for r in aus if r["bfi"] not in ("", "nan")])
    aus_aib = np.array([float(r["long_term_AI"]) for r in aus if r["bfi"] not in ("", "nan")])
    ax4.scatter(gb_aib, gb_bfi, s=12, alpha=0.5, color=C_GB, edgecolors="none", label="GB (med AI=0.53, BFI=0.47)")
    ax4.scatter(aus_aib, aus_bfi, s=12, alpha=0.5, color=C_AUS, edgecolors="none", label="AUS (med AI=1.37, BFI=0.22)")
    ax4.set_xscale("log")
    ax4.axvline(1.0, color="#666666", ls=":", lw=1.0)
    ax4.set_ylim(0, 1.02)
    ax4.set_xlabel(r"Long-term Aridity Index ($AI$, log scale)")
    ax4.set_ylabel("Baseflow Index (BFI)")
    ax4.set_title("(d) Complementary Climate–Storage Regimes")
    ax4.legend(loc="upper right", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_study_area_spatial_budyko.png"))
    plt.close(fig)


def make_fig2(diag):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3))
    groups = ["All\n(n=1,202)", "GB\n(n=644)", "AUS\n(n=558)", "Near-Natural\n(n=869)"]
    keys = ["ALL_1202", "GB_644", "AUS_558", "NEAR_NATURAL_869"]
    between = [diag[k]["variance_decomposition"]["between_basin_share_ICC"] * 100 for k in keys]
    within = [diag[k]["variance_decomposition"]["within_basin_share"] * 100 for k in keys]
    x = np.arange(len(groups))

    ax1.bar(x, between, width=0.52, color="#4c72b0", label="Between-basin static structure (ICC)")
    ax1.bar(x, within, width=0.52, bottom=between, color="#dd8452", label="Within-basin inter-annual variability")
    for i, (b, w) in enumerate(zip(between, within)):
        ax1.text(i, b / 2, f"{b:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.3)
        ax1.text(i, b + w / 2, f"{w:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.3)
    ax1.set_xticks(x); ax1.set_xticklabels(groups); ax1.set_ylim(0, 114)
    ax1.set_ylabel("Share of Budyko Departure Variance (%)")
    ax1.set_title("(a) Orthogonal Variance Decomposition")
    ax1.legend(loc="upper center", frameon=True, framealpha=0.95)

    # (b) Closed-form Euler null vs MC null vs excess physical vs zero-shared-Pt
    mc_null = [diag[k]["test_B_euler_theorem_vs_mc_null"]["mc_null_mean_dq_mm"] for k in keys]
    eu_null = [diag[k]["test_B_euler_theorem_vs_mc_null"]["euler_closed_form_null_dq_mm"] for k in keys]
    net_dq = [diag[k]["test_B_euler_theorem_vs_mc_null"]["excess_physical_asymmetry_dq_mm"] for k in keys]
    ant_dq = [diag[k]["test_A_antecedent_only_zero_shared_Pt"]["wet_minus_dry_dq_mm"] for k in keys]

    w = 0.24
    ax2.bar(x - w, mc_null, width=w, color="#bdbdbd", label=r"MC permutation null (algebraic $P_t$)")
    ax2.plot(x - w, eu_null, "kD", ms=5.5, label=r"Closed-form $-4\,\mathrm{Cov}(\Delta Q, F_P(P))$")
    ax2.bar(x, net_dq, width=w, color=C_AUS, label="Excess physical transition asymmetry")
    ax2.bar(x + w, ant_dq, width=w, color=C_GB, label=r"Zero-shared-$P_t$ antecedent effect ($P_{t-1}$)")
    for i in range(len(keys)):
        ax2.text(x[i] - w, mc_null[i] + 1.3, f"{mc_null[i]:.1f}", ha="center", va="bottom", fontsize=7.0)
        ax2.text(x[i], net_dq[i] + 1.3, f"{net_dq[i]:.1f}", ha="center", va="bottom", fontsize=7.0, color="#a63603")
        ax2.text(x[i] + w, ant_dq[i] + 1.3, f"{ant_dq[i]:.1f}", ha="center", va="bottom", fontsize=7.0, color="#08519c")

    ax2.set_xticks(x); ax2.set_xticklabels(["All", "GB", "AUS", "Near-Nat"])
    ax2.set_ylim(0, 56)
    ax2.set_ylabel("Runoff Anomaly Difference (mm/yr)")
    ax2.set_title("(b) Analytical Null Theorem vs. Physical Carry-Over")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_euler_theorem_and_drivers.png"))
    plt.close(fig)


def make_fig3(m3_base, m3_up):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3))

    s1 = m3_base["S1_Gauged_Temporal"]
    s2 = m3_base["S2_PUB_Spatial_CV"]
    labels1 = [
        "S1: Global Fu", "S1: Basin Fu", "S1: Dyn. ARX", "S1: Dyn. HGBR", "S1: PhysRes HGBR",
        "S2 PUB: Global Fu", "S2 PUB: HGBR", "S2 PUB: PhysRes"
    ]
    rmses1 = [
        s1["baselines"]["M0_Global_Fu"]["RMSE"],
        s1["baselines"]["M1_Basin_Fu"]["RMSE"],
        s1["baselines"]["M2_Dynamic_Budyko_ARX"]["RMSE"],
        s1["M4_HGBR_Dynamic_UQ"]["point"]["RMSE"],
        m3_up["S1_Upgraded"]["M5_PhysRes_HGBR"]["point"]["RMSE"],
        s2["M0_Global_Fu"]["RMSE"],
        s2["M3_HGBR_PUB_point"]["RMSE"],
        m3_up["S2_PUB_Upgraded"]["M5_PhysRes_HGBR_point"]["RMSE"],
    ]
    colors1 = ["#9ecae1", "#6baed6", "#4292c6", "#084594", "#2ca02c", "#fdae6b", "#d94801", "#2ca02c"]
    y = np.arange(len(labels1))
    ax1.barh(y, rmses1, color=colors1, height=0.62)
    for i, v in enumerate(rmses1):
        ax1.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=7.8)
    ax1.set_yticks(y); ax1.set_yticklabels(labels1); ax1.invert_yaxis()
    ax1.set_xlim(0, 0.145)
    ax1.set_xlabel("Out-of-Sample Runoff Ratio RMSE")
    ax1.set_title("(a) In-Support S1 Forecasting & S2 Ungauged PUB")

    # (b) S3 Cross-Continent Rescue: Physical Fu vs Pure ML vs Physics-Anchored Residual HGBR
    s3b = m3_base["S3_Cross_Continent_OOD"]
    s3u = m3_up["S3_Cross_Continent_PhysRes"]
    cats = ["GB → AUS\n(Out-of-Support)", "AUS → GB\n(Partial Overlap)"]
    fu_v = [s3b["GB_to_AUS"]["M0_Source_Fu"]["RMSE"], s3b["AUS_to_GB"]["M0_Source_Fu"]["RMSE"]]
    ml_v = [s3b["GB_to_AUS"]["M3_HGBR_Transfer_UQ"]["point"]["RMSE"], s3b["AUS_to_GB"]["M3_HGBR_Transfer_UQ"]["point"]["RMSE"]]
    pr_v = [s3u["GB_to_AUS"]["M5_PhysRes_HGBR"]["point"]["RMSE"], s3u["AUS_to_GB"]["M5_PhysRes_HGBR"]["point"]["RMSE"]]

    x = np.arange(len(cats))
    w = 0.25
    ax2.bar(x - w, fu_v, width=w, color=C_GB, label="Source Physical Fu")
    ax2.bar(x, ml_v, width=w, color=C_AUS, label="Pure HGBR (Direct)")
    ax2.bar(x + w, pr_v, width=w, color=C_GREEN, label="Physics-Anchored Residual HGBR")
    for i in range(len(cats)):
        ax2.text(x[i] - w, fu_v[i] + 0.006, f"{fu_v[i]:.3f}", ha="center", va="bottom", fontsize=7.6)
        ax2.text(x[i], ml_v[i] + 0.006, f"{ml_v[i]:.3f}", ha="center", va="bottom", fontsize=7.6)
        ax2.text(x[i] + w, pr_v[i] + 0.006, f"{pr_v[i]:.3f}", ha="center", va="bottom", fontsize=7.6, color="#006d2c", fontweight="bold")
    ax2.annotate("Split-CP Cov:\n23.1% → 89.0%\nR²: -1.67 → +0.52",
                 xy=(0 + w, pr_v[0] + 0.025), xytext=(0.35, 0.23),
                 arrowprops=dict(arrowstyle="->", color="#006d2c", lw=1.2),
                 fontsize=7.8, color="#006d2c", bbox=dict(facecolor="#edf8e9", edgecolor="#2ca02c", pad=3))
    ax2.set_xticks(x); ax2.set_xticklabels(cats); ax2.set_ylim(0, 0.35)
    ax2.set_ylabel("Out-of-Sample Runoff Ratio RMSE")
    ax2.set_title("(b) Cross-Continent Transfer: Rescuing Pure ML Collapse")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_point_and_physres_rescue.png"))
    plt.close(fig)


def make_fig4(m3_base, m3_up):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3))

    s2r = m3_base["S2_PUB_Spatial_CV"]["by_region"]
    s2u = m3_up["S2_PUB_Upgraded"]
    mlabs = ["Split-CP", "Global CQR", "Mondrian\nCQR", "Mondrian\nLogit-CQR"]
    gb_cov = [
        s2r["GB"]["Split_CP"]["Coverage"] * 100,
        s2r["GB"]["CQR"]["Coverage"] * 100,
        s2r["GB"]["Mondrian_CQR"]["Coverage"] * 100,
        s2u["Mondrian_Logit_CQR_GB"]["Coverage"] * 100,
    ]
    aus_cov = [
        s2r["AUS"]["Split_CP"]["Coverage"] * 100,
        s2r["AUS"]["CQR"]["Coverage"] * 100,
        s2r["AUS"]["Mondrian_CQR"]["Coverage"] * 100,
        s2u["Mondrian_Logit_CQR_AUS"]["Coverage"] * 100,
    ]
    gb_ws = [
        s2r["GB"]["Split_CP"]["Winkler_Score"],
        s2r["GB"]["CQR"]["Winkler_Score"],
        s2r["GB"]["Mondrian_CQR"]["Winkler_Score"],
        s2u["Mondrian_Logit_CQR_GB"]["Winkler_Score"],
    ]
    aus_ws = [
        s2r["AUS"]["Split_CP"]["Winkler_Score"],
        s2r["AUS"]["CQR"]["Winkler_Score"],
        s2r["AUS"]["Mondrian_CQR"]["Winkler_Score"],
        s2u["Mondrian_Logit_CQR_AUS"]["Winkler_Score"],
    ]

    x = np.arange(len(mlabs))
    w = 0.35
    ax1.bar(x - w / 2, gb_cov, width=w, color=C_GB, label="GB Coverage (%)")
    ax1.bar(x + w / 2, aus_cov, width=w, color=C_AUS, label="AUS Coverage (%)")
    ax1.axhline(90.0, color="black", ls="--", lw=1.1, label="Nominal 90%")
    box = dict(facecolor="white", alpha=0.88, edgecolor="none", pad=0.6)
    for i in range(len(mlabs)):
        ax1.text(x[i] - w / 2, gb_cov[i] + 0.5, f"{gb_cov[i]:.1f}%\n(WS={gb_ws[i]:.2f})", ha="center", va="bottom", fontsize=6.9, color="#08519c", bbox=box)
        ax1.text(x[i] + w / 2, aus_cov[i] + 0.5, f"{aus_cov[i]:.1f}%\n(WS={aus_ws[i]:.2f})", ha="center", va="bottom", fontsize=6.9, color="#a63603", bbox=box)
    ax1.set_xticks(x); ax1.set_xticklabels(mlabs); ax1.set_ylim(82, 99)
    ax1.set_ylabel("Empirical 90% Coverage (%)")
    ax1.set_title("(a) S2 PUB: Mondrian & Logit-Manifold CQR Calibration")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95)

    # (b) Heteroscedastic width adaptation along AI in S1
    d = np.load(os.path.join(RES_DIR, "s1_test_predictions.npz"))
    ai = d["ai_te"]
    w_cqr = np.clip(d["cqr_hi"], 0, 1) - np.clip(d["cqr_lo"], 0, 1)
    w_sp = np.clip(d["sp_hi"], 0, 1) - np.clip(d["sp_lo"], 0, 1)
    bins = [0.1, 0.4, 0.7, 1.0, 1.5, 2.5, 8.0]
    blabs = ["0.1–0.4", "0.4–0.7", "0.7–1.0", "1.0–1.5", "1.5–2.5", ">2.5"]
    cqr_m, sp_m = [], []
    for i in range(len(bins) - 1):
        m = (ai >= bins[i]) & (ai < bins[i + 1])
        cqr_m.append(float(np.mean(w_cqr[m])))
        sp_m.append(float(np.mean(w_sp[m])))
    xb = np.arange(len(blabs))
    ax2.plot(xb, sp_m, "o--", color=C_GRAY, lw=1.6, label="Split-CP Mean Width")
    ax2.plot(xb, cqr_m, "s-", color=C_AUS, lw=2.0, label="CQR Mean Width (Heteroscedastic)")
    ax2.set_xticks(xb); ax2.set_xticklabels(blabs); ax2.set_ylim(0.05, 0.28)
    ax2.set_xlabel(r"Annual Aridity Index ($AI_t$) Regime Bin")
    ax2.set_ylabel("Mean 90% Interval Width (W)")
    ax2.set_title("(b) S1 Heteroscedastic Sharpness Across Aridity Regimes")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_conformal_logit_mondrian.png"))
    plt.close(fig)


def make_fig5(m3_up):
    s4 = m3_up["S4_Upgraded_ACI_and_Spatial"]
    tr = s4["Yearly_Trace_1997_2021"]
    yrs = [x["year"] for x in tr]
    cov_st = [x["cov_static_cqr"] * 100 for x in tr]
    cov_aci = [x["cov_aci"] * 100 for x in tr]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax1.axvspan(1997, 2009.5, color="#fee6ce", alpha=0.55, label="Millennium Drought (1997–2009)")
    ax1.plot(yrs, cov_st, "o--", color=C_GRAY, ms=4, lw=1.5, label="Static CQR Coverage")
    ax1.plot(yrs, cov_aci, "s-", color=C_AUS, ms=4.2, lw=1.9, label="Online ACI-CQR Coverage")
    ax1.axhline(90.0, color="black", ls=":", lw=1.1, label="Nominal 90%")
    ax1.set_xlim(1996.5, 2021.5); ax1.set_ylim(58, 102)
    ax1.set_xlabel("Australian Water Year")
    ax1.set_ylabel("Annual Empirical Coverage (%)")
    ax1.set_title("(a) Online ACI Tracking Through Drought & Post-Drought")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.6)

    # (b) Spatial heterogeneity of 2010-2021 post-drought non-recovery
    sp = s4["Spatial_PostDrought_2010_2021"]
    regs = ["SE Australia\n(Vic / MDB, n=2,875)", "Northern Tropical\n(lat > -25°, n=887)"]
    st_c = [sp["SE_Australia_Victoria_MDB"]["Static_CQR"]["Coverage"] * 100, sp["Northern_Tropical_AUS"]["Static_CQR"]["Coverage"] * 100]
    ac_c = [sp["SE_Australia_Victoria_MDB"]["Online_ACI"]["Coverage"] * 100, sp["Northern_Tropical_AUS"]["Online_ACI"]["Coverage"] * 100]
    fb = [sp["SE_Australia_Victoria_MDB"]["Pre97_Fu_Bias"], sp["Northern_Tropical_AUS"]["Pre97_Fu_Bias"]]

    x = np.arange(len(regs))
    w = 0.30
    ax2.bar(x - w / 2, st_c, width=w, color=C_GB, label="Static CQR (2010–2021)")
    ax2.bar(x + w / 2, ac_c, width=w, color=C_AUS, label="Online ACI-CQR (2010–2021)")
    ax2.axhline(90.0, color="black", ls="--", lw=1.1, label="Nominal 90%")
    box = dict(facecolor="white", alpha=0.88, edgecolor="none", pad=0.6)
    for i in range(len(regs)):
        ax2.text(x[i] - w / 2, st_c[i] + 0.6, f"{st_c[i]:.1f}%", ha="center", va="bottom", fontsize=7.6, bbox=box)
        ax2.text(x[i] + w / 2, ac_c[i] + 0.6, f"{ac_c[i]:.1f}%", ha="center", va="bottom", fontsize=7.6, bbox=box)
        ax2.text(x[i], 68.5, f"Pre-1997 Fu Bias:\n{fb[i]:+.4f}", ha="center", va="center", fontsize=7.6,
                 bbox=dict(facecolor="#f7f7f7", edgecolor="#cccccc", pad=2.5))
    ax2.set_xticks(x); ax2.set_xticklabels(regs); ax2.set_ylim(64, 99)
    ax2.set_ylabel("Post-Drought Empirical Coverage (%)")
    ax2.set_title("(b) Spatial Contrast in Post-Drought Non-Recovery")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=7.6)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_drought_spatial_and_aci.png"))
    plt.close(fig)


def make_fig6(diag, m3_up):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.1))

    # (a) Top-7 PUB Permutation Feature Importance (showing AI_{t-1} is #2!)
    imp = m3_up["S2_PUB_Upgraded"]["Top10_PUB_Feature_Importance_DeltaRMSE"][:7]
    fnames = [x["feature"] for x in imp]
    vals = [x["delta_rmse"] for x in imp]
    y = np.arange(len(fnames))
    cols = [C_GB if i == 0 else (C_AUS if i == 1 else "#6baed6") for i in range(len(fnames))]
    ax1.barh(y, vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax1.text(v + 0.002, i, f"+{v:.4f}", va="center", fontsize=7.8)
    ax1.set_yticks(y); ax1.set_yticklabels(fnames); ax1.invert_yaxis()
    ax1.set_xlim(0, 0.185)
    ax1.set_xlabel(r"Out-of-Sample Permutation Importance ($\Delta$RMSE)")
    ax1.set_title(r"(a) S2 PUB Predictors ($AI_{t-1}$ Ranks #2 Without Flow History)")

    # (b) Physical drivers of within-basin rho1 controlling for AI (resolving BFI paradox)
    sub_labels = ["GB (n=644)", "AUS (n=558)", "Near-Natural (n=869)"]
    sub_keys = ["GB_644", "AUS_558", "NEAR_NATURAL_869"]
    bfi_p = [diag[k]["physical_drivers_and_bfi_resolution"]["partial_BFI_vs_rho1_given_AI"]["rho"] for k in sub_keys]
    for_p = [diag[k]["physical_drivers_and_bfi_resolution"]["partial_Forest_vs_rho1_given_AI"]["rho"] for k in sub_keys]
    qcv_r = [diag[k]["physical_drivers_and_bfi_resolution"]["raw_Qcv_vs_rho1"]["rho"] for k in sub_keys]

    x = np.arange(len(sub_labels))
    w = 0.25
    ax2.axhline(0, color="black", lw=0.9)
    ax2.bar(x - w, bfi_p, width=w, color=C_GB, label=r"Partial $\rho_s(\mathrm{BFI}, \rho_1 \mid AI)$")
    ax2.bar(x, for_p, width=w, color=C_GREEN, label=r"Partial $\rho_s(\mathrm{Forest}, \rho_1 \mid AI)$")
    ax2.bar(x + w, qcv_r, width=w, color=C_AUS, label=r"Spearman $\rho_s(\mathrm{CV}_Q, \rho_1)$")
    for i in range(len(sub_labels)):
        for off, val in [(-w, bfi_p[i]), (0, for_p[i]), (w, qcv_r[i])]:
            va = "bottom" if val >= 0 else "top"
            dy = 0.015 if val >= 0 else -0.015
            ax2.text(x[i] + off, val + dy, f"{val:+.2f}", ha="center", va=va, fontsize=7.0)
    ax2.set_xticks(x); ax2.set_xticklabels(sub_labels); ax2.set_ylim(-0.52, 0.48)
    ax2.set_ylabel(r"Correlation with Catchment Memory ($\rho_1$)")
    ax2.set_title("(b) Resolving Physical Drivers of Inter-Annual Memory")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_feature_importance_and_bfi.png"))
    plt.close(fig)


def main():
    attrs = load_attrs()
    diag = json.load(open(os.path.join(RES_DIR, "diagnostics_upgraded.json"), encoding="utf-8"))
    m3_base = json.load(open(os.path.join(RES_DIR, "modeling_uq_summary.json"), encoding="utf-8"))
    m3_up = json.load(open(os.path.join(RES_DIR, "modeling_upgraded.json"), encoding="utf-8"))

    make_fig1(attrs)
    make_fig2(diag)
    make_fig3(m3_base, m3_up)
    make_fig4(m3_base, m3_up)
    make_fig5(m3_up)
    make_fig6(diag, m3_up)
    print("Generated all 6 upgraded 300 DPI figures in:", FIG_DIR)


if __name__ == "__main__":
    main()
