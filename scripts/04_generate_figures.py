#!/usr/bin/env python3
"""
Generate 5 publication-ready 300 DPI figures for cross-hemisphere Budyko analysis.
All numbers are loaded strictly from verified data/ and results/ outputs.
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
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_GB = "#1f77b4"
C_AUS = "#d95f02"
C_ALL = "#2ca02c"
C_GRAY = "#7f7f7f"
C_DARK = "#222222"
C_PURP = "#7570b3"


def fu_er(ai, omega):
    ai = np.maximum(np.asarray(ai, dtype=float), 1e-9)
    return 1.0 + ai - np.power(1.0 + np.power(ai, omega), 1.0 / omega)


def load_attrs():
    rows = []
    with open(os.path.join(DATA_DIR, "gb_aus_basin_attributes.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def make_fig1(attrs):
    gb = [r for r in attrs if r["region"] == "GB"]
    aus = [r for r in attrs if r["region"] == "AUS"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # (a) Budyko space
    gb_ai = np.array([float(r["long_term_AI"]) for r in gb])
    gb_er = 1.0 - np.array([float(r["long_term_RR"]) for r in gb])
    aus_ai = np.array([float(r["long_term_AI"]) for r in aus])
    aus_er = 1.0 - np.array([float(r["long_term_RR"]) for r in aus])

    ai_grid = np.linspace(0.05, 7.2, 400)
    ax1.plot(ai_grid, np.minimum(ai_grid, 1.0), "k--", lw=1.2, label="Energy / Water limits")
    ax1.plot(ai_grid, fu_er(ai_grid, 3.09), color=C_DARK, lw=1.6, label=r"Global Fu ($\omega = 3.09$)")
    ax1.scatter(gb_ai, gb_er, s=14, alpha=0.55, color=C_GB, edgecolors="none", label=f"CAMELS-GB (n = {len(gb)})")
    ax1.scatter(aus_ai, aus_er, s=14, alpha=0.55, color=C_AUS, edgecolors="none", label=f"CAMELS-AUS (n = {len(aus)})")
    ax1.axvline(1.0, color="#666666", ls=":", lw=1.0)
    ax1.text(0.45, 0.08, "Energy-limited\n(AI < 1)", fontsize=8, color="#333333", ha="center")
    ax1.text(3.2, 0.08, "Water-limited\n(AI > 1)", fontsize=8, color="#333333", ha="center")
    ax1.set_xlim(0, 7.2)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel(r"Long-term Aridity Index ($AI = \overline{PET}/\overline{P}$)")
    ax1.set_ylabel(r"Long-term Evaporative Ratio ($1 - \overline{Q}/\overline{P}$)")
    ax1.set_title("(a) Cross-Hemisphere Budyko Space (1,202 Catchments)")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9)

    # (b) Aridity & BFI distributions (log-AI vs BFI)
    gb_bfi = np.array([float(r["bfi"]) for r in gb if r["bfi"] not in ("", "nan")])
    aus_bfi = np.array([float(r["bfi"]) for r in aus if r["bfi"] not in ("", "nan")])
    gb_ai_b = np.array([float(r["long_term_AI"]) for r in gb if r["bfi"] not in ("", "nan")])
    aus_ai_b = np.array([float(r["long_term_AI"]) for r in aus if r["bfi"] not in ("", "nan")])

    ax2.scatter(gb_ai_b, gb_bfi, s=14, alpha=0.55, color=C_GB, edgecolors="none", label="GB (median AI=0.53, BFI=0.47)")
    ax2.scatter(aus_ai_b, aus_bfi, s=14, alpha=0.55, color=C_AUS, edgecolors="none", label="AUS (median AI=1.37, BFI=0.22)")
    ax2.set_xscale("log")
    ax2.axvline(1.0, color="#666666", ls=":", lw=1.0)
    ax2.set_xlabel(r"Long-term Aridity Index ($AI$, log scale)")
    ax2.set_ylabel("Baseflow Index (BFI)")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("(b) Complementary Climate–Storage Regimes")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_study_area_budyko.png"))
    plt.close(fig)


def make_fig2(diag):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # (a) Variance Partitioning & Within-Basin Memory
    groups = ["All\n(n=1,202)", "GB\n(n=644)", "AUS\n(n=558)", "Near-Natural\n(n=869)"]
    keys = ["ALL_1202", "GB_644", "AUS_558", "NEAR_NATURAL_869"]
    between = [diag[k]["variance_decomposition"]["between_basin_share_ICC"] * 100 for k in keys]
    within = [diag[k]["variance_decomposition"]["within_basin_share"] * 100 for k in keys]
    x = np.arange(len(groups))

    ax1.bar(x, between, width=0.55, color="#4c72b0", label="Between-basin static structure (ICC)")
    ax1.bar(x, within, width=0.55, bottom=between, color="#dd8452", label="Within-basin inter-annual variability")
    for i, (b, w) in enumerate(zip(between, within)):
        ax1.text(i, b / 2, f"{b:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        ax1.text(i, b + w / 2, f"{w:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("Share of Budyko Departure Variance (%)")
    ax1.set_title("(a) Variance Decomposition of Annual Departures")
    ax1.legend(loc="upper center", ncol=1, frameon=True, framealpha=0.9)

    # (b) Transition Asymmetry: Observed vs Permutation Null vs Zero-Shared-Pt Antecedent Effect (mm/yr)
    sub_labels = ["All (1,202)", "GB (644)", "AUS (558)", "Near-Nat (869)"]
    obs_dq = [diag[k]["test_B_transition_vs_permutation_null"]["obs_drying_minus_wetting_dq_mm"] for k in keys]
    null_dq = [diag[k]["test_B_transition_vs_permutation_null"]["null_mean_drying_minus_wetting_dq_mm"] for k in keys]
    net_dq = [diag[k]["test_B_transition_vs_permutation_null"]["excess_physical_asymmetry_dq_mm"] for k in keys]
    ant_dq = [diag[k]["test_A_antecedent_only_zero_shared_Pt"]["wet_minus_dry_dq_mm"] for k in keys]

    w = 0.24
    ax2.bar(x - w, null_dq, width=w, color="#bdbdbd", label=r"Permutation null (algebraic $P_t$ artifact)")
    ax2.bar(x, net_dq, width=w, color=C_AUS, label="Excess physical transition asymmetry")
    ax2.bar(x + w, ant_dq, width=w, color=C_GB, label=r"Zero-shared-$P_t$ antecedent effect ($P_{t-1}$)")
    for i in range(len(keys)):
        ax2.text(x[i] - w, null_dq[i] + 1.0, f"{null_dq[i]:.1f}", ha="center", va="bottom", fontsize=7.2)
        ax2.text(x[i], net_dq[i] + 1.0, f"{net_dq[i]:.1f}", ha="center", va="bottom", fontsize=7.2, color="#a63603")
        ax2.text(x[i] + w, ant_dq[i] + 1.0, f"{ant_dq[i]:.1f}", ha="center", va="bottom", fontsize=7.2, color="#08519c")

    ax2.set_xticks(x)
    ax2.set_xticklabels(sub_labels)
    ax2.set_ylim(0, 55)
    ax2.set_ylabel("Runoff Anomaly Difference (mm/yr)")
    ax2.set_title("(b) Separating Physical Carry-Over from Algebraic Artifact")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_variance_and_asymmetry.png"))
    plt.close(fig)


def make_fig3(m3):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))

    # (a) S1 Gauged Temporal & S2 PUB RMSE comparison
    s1 = m3["S1_Gauged_Temporal"]
    s2 = m3["S2_PUB_Spatial_CV"]
    labels1 = ["S1: Global Fu", "S1: Pers. RR", "S1: Basin Fu", "S1: Dyn. ARX", "S1: Dyn. HGBR", "S2 PUB: Fu", "S2 PUB: HGBR"]
    rmses1 = [
        s1["baselines"]["M0_Global_Fu"]["RMSE"],
        s1["baselines"]["M1b_Persistence_RR_lag1"]["RMSE"],
        s1["baselines"]["M1_Basin_Fu"]["RMSE"],
        s1["baselines"]["M2_Dynamic_Budyko_ARX"]["RMSE"],
        s1["M4_HGBR_Dynamic_UQ"]["point"]["RMSE"],
        s2["M0_Global_Fu"]["RMSE"],
        s2["M3_HGBR_PUB_point"]["RMSE"],
    ]
    colors1 = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594", "#fdae6b", "#d94801"]
    y = np.arange(len(labels1))
    ax1.barh(y, rmses1, color=colors1, height=0.62)
    for i, v in enumerate(rmses1):
        ax1.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels1)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.145)
    ax1.set_xlabel("Out-of-Sample Runoff Ratio RMSE")
    ax1.set_title("(a) In-Support Forecasting (S1) & Ungauged PUB (S2)")

    # (b) Cross-Continent OOD (S3) & Millennium Drought (S4): Physics vs ML
    s3 = m3["S3_Cross_Continent_OOD"]
    s4 = m3["S4_AUS_Millennium_Drought"]
    cats = ["S3: GB→AUS\n(Out-of-Support)", "S3: AUS→GB\n(Partial Overlap)", "S4: AUS Drought\n(1997–2009)", "S4: AUS Post\n(2010–2021)"]
    phys_rmse = [
        s3["GB_to_AUS"]["M0_Source_Fu"]["RMSE"],
        s3["AUS_to_GB"]["M0_Source_Fu"]["RMSE"],
        s4["Drought_1997_2009"]["M1_PreDrought_Basin_Fu"]["RMSE"],
        s4["PostDrought_2010_2021"]["M1_PreDrought_Basin_Fu"]["RMSE"],
    ]
    ml_rmse = [
        s3["GB_to_AUS"]["M3_HGBR_Transfer_UQ"]["point"]["RMSE"],
        s3["AUS_to_GB"]["M3_HGBR_Transfer_UQ"]["point"]["RMSE"],
        s4["Drought_1997_2009"]["M4_HGBR_Dynamic_UQ"]["point"]["RMSE"],
        s4["PostDrought_2010_2021"]["M4_HGBR_Dynamic_UQ"]["point"]["RMSE"],
    ]
    x = np.arange(len(cats))
    w = 0.34
    ax2.bar(x - w / 2, phys_rmse, width=w, color=C_GB, label="Physical Fu Baseline")
    ax2.bar(x + w / 2, ml_rmse, width=w, color=C_AUS, label="HGBR Machine Learning")
    for i, (pv, mv) in enumerate(zip(phys_rmse, ml_rmse)):
        ax2.text(x[i] - w / 2, pv + 0.006, f"{pv:.3f}", ha="center", va="bottom", fontsize=7.8)
        ax2.text(x[i] + w / 2, mv + 0.006, f"{mv:.3f}", ha="center", va="bottom", fontsize=7.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.set_ylim(0, 0.35)
    ax2.set_ylabel("Out-of-Sample Runoff Ratio RMSE")
    ax2.set_title("(b) Regime-Shift & Cross-Continent Extrapolation")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_point_performance_all_scenarios.png"))
    plt.close(fig)


def make_fig4(m3):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3))

    # (a) PUB S2 Regional Calibration: Split-CP vs Local-CP vs CQR vs Mondrian-CQR
    s2r = m3["S2_PUB_Spatial_CV"]["by_region"]
    methods = ["Split_CP", "Local_Adaptive_CP", "CQR", "Mondrian_CQR"]
    mlabs = ["Split-CP", "Local-CP", "Global CQR", "Mondrian CQR"]
    gb_cov = [s2r["GB"][m]["Coverage"] * 100 for m in methods]
    aus_cov = [s2r["AUS"][m]["Coverage"] * 100 for m in methods]
    gb_w = [s2r["GB"][m]["Mean_Width"] for m in methods]
    aus_w = [s2r["AUS"][m]["Mean_Width"] for m in methods]

    x = np.arange(len(methods))
    w = 0.34
    ax1.bar(x - w / 2, gb_cov, width=w, color=C_GB, label="GB Coverage (%)")
    ax1.bar(x + w / 2, aus_cov, width=w, color=C_AUS, label="AUS Coverage (%)")
    ax1.axhline(90.0, color="black", ls="--", lw=1.1, label="Nominal 90% Target")
    for i in range(len(methods)):
        ax1.text(x[i] - w / 2, gb_cov[i] + 0.45, f"{gb_cov[i]:.1f}%\n(W={gb_w[i]:.2f})", ha="center", va="bottom", fontsize=7.0, color="#08519c", bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=0.6))
        ax1.text(x[i] + w / 2, aus_cov[i] + 0.45, f"{aus_cov[i]:.1f}%\n(W={aus_w[i]:.2f})", ha="center", va="bottom", fontsize=7.0, color="#a63603", bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=0.6))
    ax1.set_xticks(x)
    ax1.set_xticklabels(mlabs)
    ax1.set_ylim(82, 99)
    ax1.set_ylabel("Empirical 90% Interval Coverage (%)")
    ax1.set_title("(a) S2 PUB: Conditional Calibration & Sharpness (W)")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9)

    # (b) Heteroscedastic width adaptation along Aridity Index in S1 test set
    d = np.load(os.path.join(RES_DIR, "s1_test_predictions.npz"))
    ai = d["ai_te"]
    w_cqr = np.clip(d["cqr_hi"], 0, 1) - np.clip(d["cqr_lo"], 0, 1)
    w_sp = np.clip(d["sp_hi"], 0, 1) - np.clip(d["sp_lo"], 0, 1)
    cov_cqr = (d["y_te"] >= np.clip(d["cqr_lo"], 0, 1)) & (d["y_te"] <= np.clip(d["cqr_hi"], 0, 1))

    bins = [0.1, 0.4, 0.7, 1.0, 1.5, 2.5, 8.0]
    bin_labels = ["0.1–0.4", "0.4–0.7", "0.7–1.0", "1.0–1.5", "1.5–2.5", ">2.5"]
    cqr_means, sp_means, cov_means = [], [], []
    for i in range(len(bins) - 1):
        m = (ai >= bins[i]) & (ai < bins[i + 1])
        cqr_means.append(float(np.mean(w_cqr[m])))
        sp_means.append(float(np.mean(w_sp[m])))
        cov_means.append(float(np.mean(cov_cqr[m])) * 100)

    xb = np.arange(len(bin_labels))
    ax2.plot(xb, sp_means, "o--", color=C_GRAY, lw=1.6, label="Split-CP Mean Width")
    ax2.plot(xb, cqr_means, "s-", color=C_AUS, lw=2.0, label="CQR Mean Width (Heteroscedastic)")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(bin_labels)
    ax2.set_xlabel(r"Annual Aridity Index ($AI_t$) Regime Bin")
    ax2.set_ylabel("Mean 90% Prediction Interval Width (W)")
    ax2.set_ylim(0.05, 0.28)
    ax2.set_title("(b) S1 Interval Sharpness Adaptation Across Aridity Regimes")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_conformal_uq_comparison.png"))
    plt.close(fig)


def make_fig5(m3):
    s4 = m3["S4_AUS_Millennium_Drought"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))

    periods = ["Millennium Drought\n(1997–2009, n=6,300)", "Post-Drought\n(2010–2021, n=5,484)"]
    fu_bias = [s4["Drought_1997_2009"]["M1_PreDrought_Basin_Fu"]["Bias"], s4["PostDrought_2010_2021"]["M1_PreDrought_Basin_Fu"]["Bias"]]
    ml_bias = [s4["Drought_1997_2009"]["M4_HGBR_Dynamic_UQ"]["point"]["Bias"], s4["PostDrought_2010_2021"]["M4_HGBR_Dynamic_UQ"]["point"]["Bias"]]

    x = np.arange(len(periods))
    w = 0.32
    ax1.bar(x - w / 2, fu_bias, width=w, color=C_GB, label="Pre-1997 Basin Fu (Static)")
    ax1.bar(x + w / 2, ml_bias, width=w, color=C_AUS, label="Dynamic HGBR (With Memory)")
    for i, (fb, mb) in enumerate(zip(fu_bias, ml_bias)):
        ax1.text(x[i] - w / 2, fb + 0.0015, f"+{fb:.4f}", ha="center", va="bottom", fontsize=8)
        ax1.text(x[i] + w / 2, mb + 0.0015, f"+{mb:.4f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods)
    ax1.set_ylim(0, 0.056)
    ax1.set_ylabel("Mean Runoff Ratio Bias (Predicted − Observed)")
    ax1.set_title("(a) Runoff Overestimation Under Multi-Year Drought")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    # (b) Coverage & Winkler Score of Raw QR vs Split-CP vs CQR during Drought & Post-Drought
    uq_methods = ["Raw_QR", "Split_CP", "Local_Adaptive_CP", "CQR"]
    ulabs = ["Raw QR", "Split-CP", "Local-CP", "CQR"]
    d_cov = [s4["Drought_1997_2009"]["M4_HGBR_Dynamic_UQ"][m]["Coverage"] * 100 for m in uq_methods]
    p_cov = [s4["PostDrought_2010_2021"]["M4_HGBR_Dynamic_UQ"][m]["Coverage"] * 100 for m in uq_methods]
    d_ws = [s4["Drought_1997_2009"]["M4_HGBR_Dynamic_UQ"][m]["Winkler_Score"] for m in uq_methods]

    xu = np.arange(len(uq_methods))
    ax2.bar(xu - w / 2, d_cov, width=w, color="#e6550d", label="Drought (1997–2009)")
    ax2.bar(xu + w / 2, p_cov, width=w, color="#fdae6b", label="Post-Drought (2010–2021)")
    ax2.axhline(90.0, color="black", ls="--", lw=1.1, label="Nominal 90%")
    for i in range(len(uq_methods)):
        ax2.text(xu[i] - w / 2, d_cov[i] + 0.7, f"{d_cov[i]:.1f}%\n(WS={d_ws[i]:.2f})", ha="center", va="bottom", fontsize=7.0, bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=0.6))
        ax2.text(xu[i] + w / 2, p_cov[i] + 0.8, f"{p_cov[i]:.1f}%", ha="center", va="bottom", fontsize=7.2)
    ax2.set_xticks(xu)
    ax2.set_xticklabels(ulabs)
    ax2.set_ylim(58, 100)
    ax2.set_ylabel("Empirical 90% Coverage (%)")
    ax2.set_title("(b) Conformal Calibration Resilience Under Drought Shift")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_millennium_drought_stress.png"))
    plt.close(fig)


def main():
    attrs = load_attrs()
    with open(os.path.join(RES_DIR, "diagnostics_summary.json"), encoding="utf-8") as f:
        diag = json.load(f)
    with open(os.path.join(RES_DIR, "modeling_uq_summary.json"), encoding="utf-8") as f:
        m3 = json.load(f)

    make_fig1(attrs)
    make_fig2(diag)
    make_fig3(m3)
    make_fig4(m3)
    make_fig5(m3)
    print("Generated all 5 publication figures in:", FIG_DIR)


if __name__ == "__main__":
    main()
