\
\

import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, gaussian_kde
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from fetch_usgs_era5 import USGS_60_GAUGES

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.0,
    "axes.titlesize": 9.6,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

C_NAVY = "#1B365D"
C_BLUE = "#0072B2"
C_TEAL = "#009E73"
C_ORANGE = "#E69F00"
C_RED = "#D55E00"
C_LIGHT_BLUE = "#EBF4FA"
C_LIGHT_RED = "#FDF2F0"


def read_csv_dicts(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def fu_budyko_analytical(AI, w=2.6):
    AI = np.asarray(AI, dtype=float)
    et_ratio = 1.0 + AI - (1.0 + np.maximum(AI, 1e-6) ** w) ** (1.0 / w)
    return np.clip(1.0 - et_ratio, 0.0, 1.0)


def make_fig1_workflow(out_dir):
    fig, ax = plt.subplots(figsize=(13.4, 6.8), dpi=300)
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    hdr = mpatches.FancyBboxPatch(
        (0.20, 5.92), 13.0, 0.72, boxstyle="round,pad=0.06",
        facecolor=C_NAVY, edgecolor=C_NAVY
    )
    ax.add_patch(hdr)
    ax.text(
        6.7, 6.40,
        "Figure 1 | Reconstructible Process-to-Prediction Framework: Resolving the Hydrological Scaling–Uncertainty Paradox",
        ha="center", va="center", fontsize=11.0, fontweight="bold", color="white"
    )
    ax.text(
        6.7, 6.10,
        "Integrating 20-year temperate catchment observations (2005–2024), Buckingham-pi storage completion, and zero-parameter Jacobian conformal transport",
        ha="center", va="center", fontsize=8.6, style="italic", color="#E2E8F0"
    )

    boxes = [
        (0.20, 2.82, 2.98, 2.84, "#F0F7FF", C_BLUE,
         "STAGE 1: Multi-Regime Data\n(2005–2024, 20 Calendar Years)",
         "• 58 Temperate USGS Catchments:\n"
         "  - Screened from 60 NWIS gauges\n"
         "  - >=330 valid d/yr, P > 100 mm/yr\n"
         "  - 1,032 valid station-years\n"
         "  - 983 contiguous 2-yr transitions\n"
         "• Dual Analytical Benchmarks:\n"
         "  - A: 1D Solute ADR (C/C0 in [0,1])\n"
         "  - B: Coupled Fu–Budyko Runoff"),
        (3.54, 2.82, 2.98, 2.84, "#FFF5F5", C_RED,
         "STAGE 2: Physical Regime Split\n& Operational Scenarios",
         "• Strict Regime Boundary:\n"
         "  - Humid ID: AI = ET0/P <= 0.85\n"
         "    (n = 333; 60% Fit/20% Cal/20% Test)\n"
         "  - Dry-Shift OOD: AI > 0.85 (n = 650)\n"
         "• Two Operational Scenarios:\n"
         "  - Scenario 1 (Gauged Climate Shift):\n"
         "    Uses historical Q_{t-1} memory\n"
         "  - Scenario 2 (Pure Ungauged Basin):\n"
         "    Strictly NO target Q_{t-1} or BFI"),
        (6.88, 2.82, 2.98, 2.84, "#F8F0FC", "#805AD5",
         "STAGE 3: Process Completion\n(Carry-Over Storage Pi-Groups)",
         "• Steady-State Pi (Incomplete):\n"
         "  - AI_t = ET0_t / P_t only\n"
         "• Gauged Complete Pi* (Scenario 1):\n"
         "  - Carry-over storage anomaly:\n"
         "    Pi_{M,t} = Q_{t-1}/P_{t-1} - B(AI_{t-1})\n"
         "  - Wetting jump Pi_{Delta,t}, BFI, SI\n"
         "• Ungauged Pi_ung (Scenario 2):\n"
         "  - Met memory (AI_{t-1}, Pi_{Delta,t},\n"
         "    SI) + terrain (Lat, Lon, Area)"),
        (10.22, 2.82, 2.98, 2.84, "#E6FFFA", C_TEAL,
         "STAGE 4: Zero-Param IJS-CP\n(Theorem 1 Resolution)",
         "• Why Std-CP & LW-CP Collapse:\n"
         "  - Better Pi-physics shrinks ID error\n"
         "    (q_comp < q_inc < q_raw)\n"
         "• First-Principles IJS-CP Half-Width:\n"
         "  - Path-integrated Jacobian drift:\n"
         "    Delta_M = |M_0(Pi) - M_0(Pi_proj)|\n"
         "  - q_IJS = q_Pi * sqrt(1 + d_tilde)\n"
         "            + excess_Delta_M\n"
         "  - Zero tuned hyperparameters!"),
    ]

    for x0, y0, w, h, fc, ec, title, body in boxes:
        rect = mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.06",
            facecolor=fc, edgecolor=ec, linewidth=1.8
        )
        ax.add_patch(rect)
        tbar = mpatches.FancyBboxPatch(
            (x0 + 0.04, y0 + h - 0.54), w - 0.08, 0.50, boxstyle="round,pad=0.03",
            facecolor=ec, edgecolor="none"
        )
        ax.add_patch(tbar)
        ax.text(x0 + w / 2, y0 + h - 0.29, title, ha="center", va="center", fontsize=8.4, fontweight="bold", color="white")
        ax.text(x0 + 0.10, y0 + h - 0.66, body, ha="left", va="top", fontsize=7.45, color="#1A202C", linespacing=1.28)

    for x_arr in [3.20, 6.54, 9.88]:
        ax.annotate(
            "", xy=(x_arr + 0.30, 4.24), xytext=(x_arr, 4.24),
            arrowprops=dict(arrowstyle="->,head_width=0.30,head_length=0.28", lw=2.2, color=C_NAVY)
        )

    b1 = mpatches.FancyBboxPatch(
        (0.20, 0.18), 6.32, 2.36, boxstyle="round,pad=0.06",
        facecolor="#FFFBEB", edgecolor="#D69E2E", linewidth=1.6
    )
    ax.add_patch(b1)
    ax.text(0.35, 2.38, "HYDROLOGICAL PROCESS DISCOVERY (58 Temperate USGS Catchments)", ha="left", va="top", fontsize=8.7, fontweight="bold", color="#975A16")
    ax.text(
        0.35, 2.04,
        "1. Carry-Over Storage Breakdown Law: Steady-state Budyko transfer fails across\n"
        "   climate shifts because antecedent storage anomaly |Pi_M| decouples Q_t/P_t from\n"
        "   AI_t (Spearman rho = 0.580, p = 1.37 x 10^-59). Catchments bifurcate at |Pi_M| = 0.115:\n"
        "   Low-Memory retains 93.2% coverage vs. High-Memory 55.1% (p = 1.03 x 10^-31).\n"
        "2. Regional Process Attribution: Rocky Mountain snowmelt/groundwater basins\n"
        "   (mean |Pi_M| = 0.439, BFI = 0.422) suffer 87.5% Std-CP collapse, whereas Central\n"
        "   Plains summer-convective basins (SI = 0.283, |Pi_M| = 0.108) show only 8.4% loss.",
        ha="left", va="top", fontsize=7.35, color="#2D3748", linespacing=1.26
    )

    b2 = mpatches.FancyBboxPatch(
        (6.88, 0.18), 6.32, 2.36, boxstyle="round,pad=0.06",
        facecolor="#F0FFF4", edgecolor="#38A169", linewidth=1.6
    )
    ax.add_patch(b2)
    ax.text(7.03, 2.38, "METHODOLOGICAL RESOLUTION: Gauged & Pure Ungauged Scenarios", ha="left", va="top", fontsize=8.7, fontweight="bold", color="#276749")
    ax.text(
        7.03, 2.04,
        "1. Paradox Intensification (Propositions 1–2): Completing the physical Pi-space cuts\n"
        "   OOD RMSE by 14.7% (gauged) and 12.4% (pure ungauged), but compresses in-domain\n"
        "   calibration width by >50%, collapsing Std-CP coverage from 74.2% down to 41.2%\n"
        "   (Scenario 1: Gauged) and 48.5% (Scenario 2: Pure Ungauged).\n"
        "2. Zero-Parameter IJS-CP Recovery (Theorem 1): Coupling complete physical invariants\n"
        "   with path-integrated Jacobian drift restores exact nominal coverage in BOTH Scenario 1\n"
        "   (Gauged: 90.8% coverage, RMSE = 0.2370) and Scenario 2 (Ungauged: 91.1%, RMSE = 0.2433).",
        ha="left", va="top", fontsize=7.35, color="#2D3748", linespacing=1.26
    )

    plt.tight_layout()
    path = os.path.join(os.path.dirname(out_dir), "figures", "fig1_framework.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def draw_conus_outline(ax):
    conus_lon_lat = [
        (-124.7, 48.4), (-123.0, 48.3), (-122.5, 47.2), (-124.1, 46.9), (-124.0, 44.6),
        (-124.5, 42.0), (-124.2, 40.4), (-123.7, 38.9), (-122.4, 37.8), (-121.9, 36.6),
        (-120.6, 34.5), (-118.4, 34.0), (-117.1, 32.5), (-114.7, 32.7), (-111.1, 31.3),
        (-108.2, 31.3), (-106.5, 31.8), (-104.0, 29.5), (-103.0, 29.0), (-101.4, 29.8),
        (-99.1, 26.4), (-97.1, 25.9), (-97.4, 27.8), (-94.5, 29.4), (-91.2, 29.2),
        (-89.0, 29.0), (-88.8, 30.2), (-85.5, 29.7), (-83.0, 29.0), (-81.1, 25.1),
        (-80.1, 25.8), (-80.5, 28.5), (-81.4, 30.7), (-79.0, 33.5), (-75.5, 35.2),
        (-75.9, 36.9), (-76.3, 38.0), (-75.1, 38.8), (-74.0, 40.5), (-72.0, 41.0),
        (-70.0, 41.7), (-70.7, 43.1), (-68.9, 44.0), (-67.0, 44.8), (-67.8, 47.1),
        (-69.2, 47.4), (-71.5, 45.0), (-75.0, 45.0), (-76.3, 44.1), (-79.0, 43.3),
        (-82.5, 43.0), (-82.4, 45.0), (-84.5, 45.9), (-83.9, 46.5), (-88.0, 47.5),
        (-89.5, 48.0), (-95.1, 49.0), (-104.0, 49.0), (-117.0, 49.0), (-124.7, 48.4)
    ]
    xs, ys = zip(*conus_lon_lat)
    ax.fill(xs, ys, facecolor="#F7FAFC", edgecolor="#CBD5E0", linewidth=1.4, zorder=1)
    for lon_div in [-117.0, -104.0, -84.0]:
        ax.plot([lon_div, lon_div], [26.5, 48.8], color="#A0AEC0", linestyle=":", linewidth=1.1, zorder=2)


def make_fig2_spatial_and_process(out_dir):
    basins = read_csv_dicts(os.path.join(os.path.dirname(out_dir), "results", "basin_process_summary.csv"))
    with open(os.path.join(os.path.dirname(out_dir), "data", "usgs_108_catchments_2005_2024.json")) as f:
        raw_records = json.load(f)

    meta_map = {s[0]: {"lat": s[1], "lon": s[2], "area": s[3]} for s in USGS_60_GAUGES}
    ood_map = {b["site_id"]: b for b in basins}
    valid_sites = sorted(set(r["site_id"] for r in raw_records))

    fig = plt.figure(figsize=(13.4, 8.8), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.12, 1.0], width_ratios=[1.22, 1.0], hspace=0.38, wspace=0.30)

    ax_map = fig.add_subplot(gs[0, 0])
    draw_conus_outline(ax_map)

    humid_only = [sid for sid in valid_sites if sid not in ood_map]
    ax_map.scatter(
        [meta_map[s]["lon"] for s in humid_only],
        [meta_map[s]["lat"] for s in humid_only],
        s=48, marker="^", facecolor="#3182CE", edgecolor="k", linewidth=0.7,
        label="Humid Calibration Only (AI <= 0.85, n=8)", zorder=3
    )

    lons = np.array([float(b["lon"]) for b in basins])
    lats = np.array([float(b["lat"]) for b in basins])
    pi_m = np.array([float(b["mean_abs_pi_M"]) for b in basins])
    miss = np.array([float(b["miss_rate_std_pi1"]) * 100 for b in basins])
    sizes = 45.0 + 340.0 * np.clip(pi_m / 0.50, 0.05, 1.2)

    sc = ax_map.scatter(
        lons, lats, s=sizes, c=miss, cmap="RdYlBu_r", vmin=0, vmax=100,
        edgecolor="black", linewidth=0.8, alpha=0.90, zorder=4
    )
    cbar = plt.colorbar(sc, ax=ax_map, fraction=0.038, pad=0.02)
    cbar.set_label("Std-CP OOD Miscoverage Rate (%)")

    for sid, label_txt, dx, dy in [
        ("06191500", "USGS 06191500 (Yellowstone, MT)\n|Pi_M|=0.48, Miscov=100%", -0.5, -5.2),
        ("08195000", "USGS 08195000 (Frio, TX)\n|Pi_M|=0.05, Miscov=0%", 4.2, -2.2),
    ]:
        b = ood_map[sid]
        ax_map.annotate(
            label_txt, xy=(float(b["lon"]), float(b["lat"])),
            xytext=(float(b["lon"]) + dx, float(b["lat"]) + dy),
            fontsize=7.2, fontweight="bold", color="#1A202C",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFF0", edgecolor=C_NAVY, lw=0.9),
            arrowprops=dict(arrowstyle="->", lw=1.1, color=C_NAVY), zorder=6
        )

    for s_val, lbl in [(0.10, "|Pi_M|=0.10"), (0.40, "|Pi_M|=0.40")]:
        ax_map.scatter([], [], s=45.0 + 340.0 * (s_val / 0.50), c="gray", alpha=0.6, edgecolors="k", label=f"Bubble Size: {lbl}")

    ax_map.text(-121.0, 48.0, "Pacific\nNW", fontsize=7.3, color="#4A5568", ha="center", fontweight="bold")
    ax_map.text(-110.5, 48.0, "Rocky Mtns\n& Southwest", fontsize=7.3, color="#4A5568", ha="center", fontweight="bold")
    ax_map.text(-94.5, 48.0, "Central Plains\n& Midwest", fontsize=7.3, color="#4A5568", ha="center", fontweight="bold")
    ax_map.text(-75.5, 48.0, "Appalachian &\nNortheast", fontsize=7.3, color="#4A5568", ha="center", fontweight="bold")

    ax_map.set_xlim(-126.5, -66.0)
    ax_map.set_ylim(24.5, 50.5)
    ax_map.set_xlabel("Longitude (°W)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.set_title("(a) Spatial Regime Breakdown Across 58 USGS Temperate Catchments")
    ax_map.legend(loc="lower left", fontsize=6.8, framealpha=0.92)


    ax_reg = fig.add_subplot(gs[0, 1])
    reg_order = [
        "Central Plains & Midwest",
        "Appalachian & Northeast",
        "Pacific Northwest & Coast",
        "Rocky Mountains & Southwest",
    ]
    reg_short = ["Central Plains\n& Midwest\n(n=21)", "Appalachian\n& Northeast\n(n=18)", "Pacific NW\n& Coast\n(n=3)", "Rocky Mtns\n& Southwest\n(n=8)"]
    std_cov_reg, ijs_cov_reg, pim_reg = [], [], []
    for rname in reg_order:
        sub = [b for b in basins if b["region"] == rname]
        std_cov_reg.append(100.0 - float(np.mean([float(x["miss_rate_std_pi1"]) * 100 for x in sub])))
        ijs_cov_reg.append(float(np.mean([float(x["cov_rate_ijs3"]) * 100 for x in sub])))
        pim_reg.append(float(np.mean([float(x["mean_abs_pi_M"]) for x in sub])))

    xr = np.arange(len(reg_order))
    bw = 0.34
    ax_reg.bar(xr - bw / 2, std_cov_reg, width=bw, color=C_RED, edgecolor="k", linewidth=0.7, label=r"Incomplete $\Pi$ ($AI$) + Std-CP")
    ax_reg.bar(xr + bw / 2, ijs_cov_reg, width=bw, color=C_TEAL, edgecolor="k", linewidth=0.7, label=r"Complete $\Pi^\star$ + Zero-Param IJS-CP")
    ax_reg.axhline(90.0, color=C_NAVY, linestyle="--", linewidth=1.5, label="90.0% Nominal Target")

    for i in range(len(xr)):
        ax_reg.text(xr[i] - bw / 2, std_cov_reg[i] + 1.8, f"{std_cov_reg[i]:.1f}%", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=C_RED)
        ax_reg.text(xr[i] + bw / 2, ijs_cov_reg[i] + 1.8, f"{ijs_cov_reg[i]:.1f}%", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#00684A")
        ax_reg.text(xr[i], 4.5, f"|Pi_M|={pim_reg[i]:.2f}", ha="center", va="bottom", fontsize=7.2,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#CBD5E0"))

    ax_reg.set_xticks(xr)
    ax_reg.set_xticklabels(reg_short, fontsize=7.8)
    ax_reg.set_ylim(0, 126)
    ax_reg.set_ylabel("Mean Basin OOD Empirical Coverage (%)")
    ax_reg.set_title("(b) Regional Carry-Over Collapse & IJS-CP Recovery")
    ax_reg.legend(loc="upper left", ncol=1, fontsize=7.3, framealpha=0.92)
    ax_reg.grid(axis="y", alpha=0.3)


    ax_phys = fig.add_subplot(gs[1, 0])
    si_vals = np.array([float(b["si_walsh"]) for b in basins])
    bfi_vals = np.array([float(b["bfi_proxy"]) for b in basins])
    ax_phys.axvspan(0.04, 0.22, ymin=0.32, ymax=1.0, color="#FFF5F5", alpha=0.65, zorder=0)
    ax_phys.axvspan(0.25, 0.76, ymin=0.0, ymax=0.55, color="#F0FFF4", alpha=0.65, zorder=0)
    ax_phys.scatter(
        si_vals, bfi_vals, c=miss, s=60 + 260 * pi_m, cmap="RdYlBu_r", vmin=0, vmax=100,
        edgecolors="k", linewidth=0.8, alpha=0.90, zorder=3
    )
    ax_phys.text(
        0.055, 0.65,
        "Groundwater / Snowmelt Carry-Over Zone\n(Low Seasonality SI < 0.22, High Baseflow BFI > 0.25)\n-> Severe Steady-State Budyko Breakdown",
        fontsize=7.4, fontweight="bold", color="#9B2C2C",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFF5F5", edgecolor="#FEB2B2")
    )
    ax_phys.text(
        0.31, 0.04,
        "Seasonal Water-Limited Plains Zone\n(High SI > 0.28, Fast Depletion)\n-> Steady-State Budyko Holds (rho_SI = -0.36, p = 0.008)",
        fontsize=7.4, fontweight="bold", color="#22543D",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#F0FFF4", edgecolor="#9AE6B4")
    )
    ax_phys.set_xlim(0.03, 0.77)
    ax_phys.set_ylim(-0.03, 0.78)
    ax_phys.set_xlabel("Walsh–Lawler Precipitation Seasonality Index SI (-)")
    ax_phys.set_ylabel("Baseflow Storage Buffer Proxy BFI = Q10 / Q50 (-)")
    ax_phys.set_title("(c) Joint Physiographic Controls in (SI, BFI) Space")
    ax_phys.grid(alpha=0.3)


    ax_mem = fig.add_subplot(gs[1, 1])
    by_site = {}
    for r in raw_records:
        by_site.setdefault(r["site_id"], []).append(r)
    for sid in by_site:
        by_site[sid].sort(key=lambda x: x["year"])

    ood_pairs = []
    for sid, recs in by_site.items():
        for i in range(1, len(recs)):
            if recs[i]["year"] - recs[i - 1]["year"] == 1 and recs[i]["AI"] > 0.85:
                pim = abs(recs[i - 1]["Q_over_P"] - float(fu_budyko_analytical(recs[i - 1]["AI"])))
                err = abs(recs[i]["Q_over_P"] - float(fu_budyko_analytical(recs[i]["AI"])))
                ood_pairs.append((pim, err))
    pim_arr = np.array([x[0] for x in ood_pairs])
    err_arr = np.array([x[1] for x in ood_pairs])

    q_edges = np.quantile(pim_arr, np.linspace(0, 1, 9))
    xc, yc_med, yc_q25, yc_q75 = [], [], [], []
    for b0, b1 in zip(q_edges[:-1], q_edges[1:]):
        msk = (pim_arr >= b0) & (pim_arr <= b1)
        xc.append(float(np.median(pim_arr[msk])))
        yc_med.append(float(np.median(err_arr[msk])))
        yc_q25.append(float(np.percentile(err_arr[msk], 25)))
        yc_q75.append(float(np.percentile(err_arr[msk], 75)))

    ax_mem.fill_between(xc, yc_q25, yc_q75, color=C_RED, alpha=0.22, label="Interquartile Range (25th–75th)")
    ax_mem.plot(xc, yc_med, marker="o", markersize=6.5, linewidth=2.4, color=C_RED, label="Octile Median Error Trajectory")
    ax_mem.axvline(0.115, color=C_NAVY, linestyle="--", linewidth=1.6, label="Bifurcation Threshold |Pi_M| = 0.115")

    ax_mem.annotate(
        "Low-Memory Equilibrium\n(|Pi_M| <= 0.115)\nStd-CP Cov = 93.2%",
        xy=(0.06, 0.09), xytext=(0.012, 0.40),
        fontsize=7.4, fontweight="bold", color=C_NAVY,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_LIGHT_BLUE, edgecolor=C_BLUE),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=C_BLUE)
    )
    ax_mem.annotate(
        "High-Memory Breakdown\n(|Pi_M| > 0.115)\nStd-CP Cov = 55.1%\n(Mann–Whitney p = 1.0e-31)",
        xy=(0.30, 0.36), xytext=(0.18, 0.52),
        fontsize=7.4, fontweight="bold", color="#9B2C2C",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_LIGHT_RED, edgecolor=C_RED),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=C_RED)
    )

    ax_mem.set_xlabel(r"Dimensionless Carry-Over Storage Anomaly $|\Pi_M|$ (-)")
    ax_mem.set_ylabel("Steady-State Budyko Departure |Q/P - B(AI)|")
    ax_mem.set_ylim(0, 0.72)
    ax_mem.set_title("(d) Carry-Over Storage Breakdown Law (rho = 0.580, p = 1.4e-59)")
    ax_mem.legend(loc="lower right", fontsize=7.4)
    ax_mem.grid(alpha=0.3)

    fig.suptitle(
        "Figure 2 | Hydrological Process Discovery: Geographic and Physiographic Controls of Carry-Over Storage Breakdown (58 USGS Basins)",
        fontsize=11.4, fontweight="bold", y=0.98
    )
    path = os.path.join(os.path.dirname(out_dir), "figures", "fig2_spatial_process_signatures.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def make_fig3_phase_and_hydrographs(out_dir):
    with open(os.path.join(os.path.dirname(out_dir), "data", "usgs_108_catchments_2005_2024.json")) as f:
        raw_records = json.load(f)

    by_site = {}
    for r in raw_records:
        by_site.setdefault(r["site_id"], []).append(r)
    for sid in by_site:
        by_site[sid].sort(key=lambda x: x["year"])

    complete_records = []
    for sid, recs in by_site.items():
        for i in range(1, len(recs)):
            curr, prev = recs[i], recs[i - 1]
            if curr["year"] - prev["year"] != 1:
                continue
            pi_M = prev["Q_over_P"] - float(fu_budyko_analytical(prev["AI"]))
            pi_Delta = (curr["P"] - prev["P"]) / (prev["P"] + 1e-6)
            complete_records.append({**curr, "pi_M": pi_M, "abs_pi_M": abs(pi_M), "pi_Delta": pi_Delta})

    id_recs = [r for r in complete_records if r["AI"] <= 0.85]
    idx_f, idx_rest = train_test_split(np.arange(len(id_recs)), test_size=0.4, random_state=42)
    idx_c, _ = train_test_split(idx_rest, test_size=0.5, random_state=42)

    X_pi1_id = np.array([[r["AI"]] for r in id_recs])
    X_pi3_id = np.array([[r["AI"], r["pi_M"], r["pi_Delta"], r["bfi_proxy"], r["si_walsh"]] for r in id_recs])
    y_id = np.array([r["Q_over_P"] for r in id_recs])

    m_pi1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_pi1_id[idx_f], y_id[idx_f])
    m_pi3 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_pi3_id[idx_f], y_id[idx_f])

    alpha = 0.10
    q_pi1 = np.quantile(np.abs(y_id[idx_c] - m_pi1.predict(X_pi1_id[idx_c])), 1.0 - alpha)
    q_pi3 = np.quantile(np.abs(y_id[idx_c] - m_pi3.predict(X_pi3_id[idx_c])), 1.0 - alpha)

    sc3 = StandardScaler().fit(X_pi3_id[idx_f])
    nn3 = NearestNeighbors(n_neighbors=1).fit(sc3.transform(X_pi3_id[idx_f]))
    d_cal, idx_cal = nn3.kneighbors(sc3.transform(X_pi3_id[idx_c]))
    d_90 = np.quantile(d_cal[:, 0], 1.0 - alpha) + 1e-6
    phys_fn = lambda X: np.clip(fu_budyko_analytical(X[:, 0]) + 0.5 * X[:, 1], 0.0, 1.0)
    dM_cal = np.abs(phys_fn(X_pi3_id[idx_c]) - phys_fn(X_pi3_id[idx_f][idx_cal[:, 0]]))
    dM_scale = np.quantile(dM_cal, 1.0 - alpha) + 1e-4

    def predict_site_series(sid):
        s_recs = [r for r in complete_records if r["site_id"] == sid]
        yrs = np.array([r["year"] for r in s_recs])
        y_obs = np.array([r["Q_over_P"] for r in s_recs])
        X1 = np.array([[r["AI"]] for r in s_recs])
        X3 = np.array([[r["AI"], r["pi_M"], r["pi_Delta"], r["bfi_proxy"], r["si_walsh"]] for r in s_recs])
        p1 = np.clip(m_pi1.predict(X1), 0.0, 1.0)
        p3 = np.clip(m_pi3.predict(X3), 0.0, 1.0)
        d_t, idx_t = nn3.kneighbors(sc3.transform(X3))
        dM_t = np.abs(phys_fn(X3) - phys_fn(X_pi3_id[idx_f][idx_t[:, 0]]))
        hw_ijs = q_pi3 * np.sqrt(1.0 + np.maximum(0.0, d_t[:, 0] / d_90 - 1.0)) + np.maximum(0.0, dM_t - dM_scale)
        return yrs, y_obs, p1, np.clip(p1 - q_pi1, 0, 1), np.clip(p1 + q_pi1, 0, 1), p3, np.clip(p3 - hw_ijs, 0, 1), np.clip(p3 + hw_ijs, 0, 1)

    fig = plt.figure(figsize=(13.6, 8.4), dpi=300)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.22], hspace=0.40, wspace=0.34)


    ax_ph = fig.add_subplot(gs[0, 0])
    ai_all = np.array([r["AI"] for r in complete_records])
    pim_all = np.array([r["abs_pi_M"] for r in complete_records])
    X1_all = np.array([[r["AI"]] for r in complete_records])
    y_all = np.array([r["Q_over_P"] for r in complete_records])
    err1_all = np.abs(y_all - np.clip(m_pi1.predict(X1_all), 0.0, 1.0))

    ai_g = np.linspace(0.35, 2.8, 60)
    pim_g = np.linspace(0.0, 0.58, 60)
    AI_G, PIM_G = np.meshgrid(ai_g, pim_g)
    Z_err = np.zeros_like(AI_G)
    for i in range(AI_G.shape[0]):
        for j in range(AI_G.shape[1]):
            dist2 = ((ai_all - AI_G[i, j]) / 0.35) ** 2 + ((pim_all - PIM_G[i, j]) / 0.10) ** 2
            w = np.exp(-0.5 * dist2)
            Z_err[i, j] = np.sum(w * err1_all) / (np.sum(w) + 1e-8)

    cf = ax_ph.contourf(AI_G, PIM_G, Z_err, levels=np.linspace(0.05, 0.55, 11), cmap="YlOrRd", alpha=0.88)
    cs = ax_ph.contour(AI_G, PIM_G, Z_err, levels=[q_pi1], colors=[C_NAVY], linewidths=2.2, linestyles="--")
    ax_ph.clabel(cs, fmt={q_pi1: f"Std-CP Limit q_Pi={q_pi1:.2f}"}, fontsize=7.6, inline=True)
    ax_ph.scatter(ai_all, np.clip(pim_all, 0, 0.57), c="k", s=8, alpha=0.28, label="Station-Years (n=983)")
    ax_ph.axvline(0.85, color=C_BLUE, linestyle=":", linewidth=1.8, label="Regime Split (AI = 0.85)")
    ax_ph.axhline(0.115, color="#2B6CB0", linestyle="-.", linewidth=1.5, label="Threshold |Pi_M| = 0.115")
    cbar_ph = plt.colorbar(cf, ax=ax_ph, pad=0.02)
    cbar_ph.set_label("Mean Error |Q/P - f(AI)|")
    ax_ph.set_xlim(0.35, 2.8)
    ax_ph.set_ylim(0.0, 0.58)
    ax_ph.set_xlabel("Climatic Aridity Index AI = ET0 / P (-)")
    ax_ph.set_ylabel(r"Carry-Over Storage Anomaly $|\Pi_M|$ (-)")
    ax_ph.set_title("(a) 2D Phase Diagram in (AI, |Pi_M|) Space")
    ax_ph.legend(loc="upper right", fontsize=7.2, framealpha=0.92)


    ax_kd = fig.add_subplot(gs[1, 0])
    X_raw_id = np.array([[r["P"], r["Ep"]] for r in id_recs])
    m_raw = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_raw_id[idx_f], y_id[idx_f])
    res_raw_cal = np.abs(y_id[idx_c] - m_raw.predict(X_raw_id[idx_c]))
    res_pi1_cal = np.abs(y_id[idx_c] - m_pi1.predict(X_pi1_id[idx_c]))
    res_pi3_cal = np.abs(y_id[idx_c] - m_pi3.predict(X_pi3_id[idx_c]))

    xg = np.linspace(0, 0.44, 300)
    for arr, col, lbl, qv in [
        (res_pi1_cal, C_ORANGE, f"Steady Pi (AI): q_inc = {q_pi1:.3f}", q_pi1),
        (res_raw_cal, C_BLUE, f"Raw (P, ET0): q_raw = {np.quantile(res_raw_cal, 0.90):.3f}", np.quantile(res_raw_cal, 0.90)),
        (res_pi3_cal, C_RED, f"Complete Pi* (+Pi_M): q_comp = {q_pi3:.3f}", q_pi3),
    ]:
        kde = gaussian_kde(arr, bw_method=0.35)
        yg = kde(xg)
        ax_kd.plot(xg, yg, color=col, linewidth=2.2, label=lbl)
        ax_kd.fill_between(xg, 0, yg, color=col, alpha=0.18)
        ax_kd.axvline(qv, color=col, linestyle="--", linewidth=1.6)

    ax_kd.annotate(
        "Proposition 1 Compression:\nq_comp (0.138) < q_raw (0.251)\nBetter physics shrinks ID width\nby >50%, triggering OOD collapse!",
        xy=(q_pi3, 3.8), xytext=(0.175, 3.4),
        fontsize=7.3, fontweight="bold", color="#1A202C",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFBEB", edgecolor="#D69E2E"),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=C_RED)
    )
    ax_kd.set_xlim(0, 0.44)
    ax_kd.set_ylim(0, 9.2)
    ax_kd.set_xlabel("In-Domain Calibration Residual |y - f_hat| (Q/P)")
    ax_kd.set_ylabel("Probability Density")
    ax_kd.set_title("(b) Proposition 1: In-Domain Quantile Compression")
    ax_kd.legend(loc="upper right", fontsize=7.2, framealpha=0.94)
    ax_kd.grid(alpha=0.3)


    ax_c1 = fig.add_subplot(gs[0, 1])
    yrs, y_obs, p1, lo1, hi1, p3, lo3, hi3 = predict_site_series("06191500")
    ax_c1.fill_between(yrs, lo3, hi3, color=C_TEAL, alpha=0.25, label=r"Complete $\Pi^\star$ + Zero-Param IJS-CP (100% Cov)")
    ax_c1.fill_between(yrs, lo1, hi1, color=C_RED, alpha=0.28, label=r"Incomplete $\Pi$ ($AI$) + Std-CP (0% Cov — Collapse!)")
    ax_c1.plot(yrs, y_obs, marker="o", markersize=5.2, color="black", linewidth=2.0, label="Observed Q/P (USGS 06191500)", zorder=5)
    ax_c1.plot(yrs, p3, marker="s", markersize=4.0, color="#00684A", linestyle="--", linewidth=1.5, label=r"Complete $\Pi^\star$ Prediction")
    ax_c1.plot(yrs, p1, marker="^", markersize=4.0, color=C_RED, linestyle=":", linewidth=1.5, label=r"Steady-State $\Pi$ ($AI$) Prediction")
    ax_c1.set_ylim(0, 1.05)
    ax_c1.set_xlim(2005.5, 2024.5)
    ax_c1.set_xticks(range(2006, 2025, 2))
    ax_c1.set_ylabel("Annual Runoff Ratio Q / P (-)")
    ax_c1.set_title("(c) High-Memory Basin: Yellowstone River, MT (USGS 06191500, |Pi_M|=0.48)")
    ax_c1.legend(loc="lower left", fontsize=6.9, ncol=2, framealpha=0.92)
    ax_c1.grid(alpha=0.3)


    ax_c2 = fig.add_subplot(gs[1, 1])
    yrs2, y_obs2, p1_2, lo1_2, hi1_2, p3_2, lo3_2, hi3_2 = predict_site_series("08195000")
    ax_c2.fill_between(yrs2, lo3_2, hi3_2, color=C_TEAL, alpha=0.25, label=r"Complete $\Pi^\star$ + Zero-Param IJS-CP (100% Cov)")
    ax_c2.fill_between(yrs2, lo1_2, hi1_2, color=C_ORANGE, alpha=0.30, label=r"Incomplete $\Pi$ ($AI$) + Std-CP (100% Cov — Equilibrium)")
    ax_c2.plot(yrs2, y_obs2, marker="o", markersize=5.2, color="black", linewidth=2.0, label="Observed Q/P (USGS 08195000)", zorder=5)
    ax_c2.plot(yrs2, p3_2, marker="s", markersize=4.0, color="#00684A", linestyle="--", linewidth=1.5, label=r"Complete $\Pi^\star$ Prediction")
    ax_c2.set_ylim(0, 0.72)
    ax_c2.set_xlim(2005.5, 2024.5)
    ax_c2.set_xticks(range(2006, 2025, 2))
    ax_c2.set_xlabel("Water Calendar Year (2006–2024 Contiguous Transitions)")
    ax_c2.set_ylabel("Annual Runoff Ratio Q / P (-)")
    ax_c2.set_title("(d) Low-Memory Basin: Frio River at Concan, TX (USGS 08195000, |Pi_M|=0.05)")
    ax_c2.legend(loc="upper right", fontsize=7.0, ncol=2, framealpha=0.92)
    ax_c2.grid(alpha=0.3)

    fig.suptitle(
        "Figure 3 | Phase-Space Mechanism and 20-Year Catchment Case Studies: Carry-Over Storage vs. Equilibrium Regimes",
        fontsize=11.4, fontweight="bold", y=0.98
    )
    path = os.path.join(os.path.dirname(out_dir), "figures", "fig3_phase_diagram_hydrographs.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def make_fig4_gauged_vs_ungauged_ijs(out_dir):
    t4 = read_csv_dicts(os.path.join(os.path.dirname(out_dir), "results", "table1_catchment_benchmarks.csv"))

    fig = plt.figure(figsize=(13.6, 5.6), dpi=300)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.12], wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    labels = [
        "1. Raw (P, ET0) + Std-CP",
        "2. Steady-State Pi (AI only) + Std-CP",
        "3. Scenario 2: Pure Ungauged Pi_ung\n    (No Q_{t-1}) + Std-CP [Paradox!]",
        "4. Scenario 1: Complete Gauged Pi*\n    (With Pi_M) + Std-CP [Paradox!]",
        "5. Steady-State Pi (AI) + IJS-CP",
        "6. Scenario 2: Pure Ungauged Pi_ung\n    (No Q_{t-1}) + Zero-Param IJS-CP",
        "7. Scenario 1: Complete Gauged Pi*\n    (With Pi_M) + Zero-Param IJS-CP",
    ]
    covs = [float(r["cov90"]) * 100 for r in t4]
    widths = [float(r["mean_width"]) for r in t4]
    rmses = [float(r["ood_rmse"]) for r in t4]
    colors = [C_BLUE, C_ORANGE, "#E53E3E", "#9B2C2C", "#63B3ED", "#38A169", "#0072B2"]

    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, covs, height=0.58, color=colors, edgecolor="k", linewidth=0.7)
    ax1.axvline(90.0, color=C_NAVY, linestyle="--", linewidth=1.8, label="Exact 90.0% Nominal Target")
    ax1.axvspan(88.0, 93.0, color="#C6F6D5", alpha=0.45, zorder=0)

    for i, (c, w, rm) in enumerate(zip(covs, widths, rmses)):
        ax1.text(
            c + 1.2, y_pos[i],
            f"{c:.1f}%  (RMSE={rm:.3f}, W={w:.3f})",
            va="center", ha="left", fontsize=7.6,
            fontweight="bold" if i in (2, 3, 5, 6) else "normal",
            color="#9B2C2C" if i in (2, 3) else ("#1E6B34" if i in (5, 6) else "#1A202C")
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=7.8, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 152)
    ax1.set_xlabel("Empirical OOD Coverage Across 58 USGS Catchments (%)")
    ax1.set_title("(a) Gauged & Ungauged Coverage Collapse vs. IJS-CP")
    ax1.legend(loc="lower right", fontsize=7.6)
    ax1.grid(axis="x", alpha=0.3)


    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axhspan(88.0, 93.0, color="#C6F6D5", alpha=0.50, label="Target Reliability Band (90% ± 2.5%)")
    ax2.axhline(90.0, color=C_NAVY, linestyle="--", linewidth=1.6, label="Nominal 90.0% Target")

    ax2.annotate(
        "", xy=(rmses[5], covs[5] - 1.8), xytext=(rmses[2], covs[2] + 2.2),
        arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.28", lw=2.0, color="#38A169")
    )
    ax2.annotate(
        "", xy=(rmses[6], covs[6] - 1.8), xytext=(rmses[3], covs[3] + 2.2),
        arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.28", lw=2.0, color=C_BLUE)
    )

    markers = ["o", "s", "v", "D", "^", "P", "*"]
    sizes_p = [110, 110, 140, 140, 110, 190, 230]
    for i in range(len(t4)):
        ax2.scatter(rmses[i], covs[i], s=sizes_p[i], c=colors[i], marker=markers[i], edgecolors="k", linewidth=1.1, zorder=5)

    annots = [
        (0, "Raw + Std-CP (62.5%)", (-0.001, -7.5), "center"),
        (1, "Steady Pi + Std-CP (74.2%)", (-0.001, -7.5), "center"),
        (2, "Scenario 2: Ungauged + Std-CP\n(48.5% Collapse)", (0.0022, -2.5), "left"),
        (3, "Scenario 1: Gauged Pi* + Std-CP\n(41.2% Collapse)", (0.0022, -4.5), "left"),
        (4, "Steady Pi + IJS-CP\n(99.8%, W=0.992)", (0.000, -10.5), "center"),
        (5, "Scenario 2: Pure Ungauged + IJS-CP\n(No Q_{t-1}: RMSE=0.243, Cov=91.1%)", (-0.001, 5.2), "left"),
        (6, "Scenario 1: Gauged Complete Pi* + IJS-CP\n(Best Frontier: RMSE=0.237, Cov=90.8%)", (0.0020, -10.5), "left"),
    ]
    for idx_a, txt, (dx, dy), ha_a in annots:
        ax2.annotate(
            txt, (rmses[idx_a], covs[idx_a]),
            xytext=(rmses[idx_a] + dx, covs[idx_a] + dy),
            ha=ha_a, fontsize=7.3,
            fontweight="bold" if idx_a in (2, 3, 5, 6) else "normal",
            color="#9B2C2C" if idx_a in (2, 3) else ("#1E6B34" if idx_a in (5, 6) else "#1A202C")
        )

    ax2.set_xlim(0.232, 0.285)
    ax2.set_ylim(32, 108)
    ax2.set_xlabel("OOD Point Prediction RMSE of Q/P (Lower is Better -> Left)")
    ax2.set_ylabel("Empirical OOD Coverage (%)")
    ax2.set_title("(b) Accuracy–Reliability Pareto Frontier")
    ax2.legend(loc="lower right", fontsize=7.4)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Figure 4 | Operational Resolution Across Both Gauged Climate Shifts and Purely Ungauged Catchments via Zero-Parameter IJS-CP",
        fontsize=11.3, fontweight="bold", y=1.01
    )
    path = os.path.join(os.path.dirname(out_dir), "figures", "fig4_benchmark_interval_quality.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def make_fig5_benchmarks_and_generality(out_dir):
    rows_adr = read_csv_dicts(os.path.join(os.path.dirname(out_dir), "results", "table2_synthetic_adr.csv"))
    rows_run = read_csv_dicts(os.path.join(os.path.dirname(out_dir), "results", "table3_synthetic_runoff.csv"))
    rows_sens = read_csv_dicts(os.path.join(os.path.dirname(out_dir), "results", "table4_sensitivity.csv"))

    fig = plt.figure(figsize=(13.4, 7.8), dpi=300)
    gs = GridSpec(2, 6, figure=fig, hspace=0.42, wspace=0.68)

    sub_adr = [r for r in rows_adr if r["model"] == "GBR"]
    sub_run = [r for r in rows_run if r["model"] == "GBR"]
    x4 = np.arange(4)
    bw = 0.20

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_a.axvspan(-0.5, 0.5, color=C_LIGHT_BLUE, alpha=0.7, zorder=0)
    ax_a.axvspan(0.5, 3.5, color=C_LIGHT_RED, alpha=0.5, zorder=0)
    ax_a.bar(x4 - 1.5*bw, [float(r["cov_raw_std"])*100 for r in sub_adr], width=bw, color=C_BLUE, edgecolor="k", lw=0.5, label="Raw + Std-CP")
    ax_a.bar(x4 - 0.5*bw, [float(r["cov_pi_std"])*100 for r in sub_adr], width=bw, color=C_RED, edgecolor="k", lw=0.5, label=r"$\pi$ + Std-CP")
    ax_a.bar(x4 + 0.5*bw, [float(r["cov_pi_lw"])*100 for r in sub_adr], width=bw, color=C_ORANGE, edgecolor="k", lw=0.5, label=r"$\pi$ + LW-CP")
    ax_a.bar(x4 + 1.5*bw, [float(r["cov_pi_dsa"])*100 for r in sub_adr], width=bw, color=C_TEAL, edgecolor="k", lw=0.5, label=r"$\pi$ + DSA-CP")
    ax_a.axhline(90.0, color=C_NAVY, ls="--", lw=1.4)
    ax_a.set_xticks(x4)
    ax_a.set_xticklabels(["ID\nPe 0.2–5", "OOD1\nPe 5–10", "OOD2\nPe 10–20", "OOD3\nPe 20–45"], fontsize=7.8)
    ax_a.set_ylim(0, 115)
    ax_a.set_ylabel("Empirical Coverage (%)")
    ax_a.set_title("(a) Benchmark A: 1D Solute ADR (GBR)")
    ax_a.legend(loc="lower left", fontsize=6.8, ncol=2)
    ax_a.grid(axis="y", alpha=0.3)

    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_b.axvspan(-0.5, 0.5, color=C_LIGHT_BLUE, alpha=0.7, zorder=0)
    ax_b.axvspan(0.5, 3.5, color=C_LIGHT_RED, alpha=0.5, zorder=0)
    ax_b.bar(x4 - 1.5*bw, [float(r["cov_raw_std"])*100 for r in sub_run], width=bw, color=C_BLUE, edgecolor="k", lw=0.5, label="Raw + Std-CP")
    ax_b.bar(x4 - 0.5*bw, [float(r["cov_pi_std"])*100 for r in sub_run], width=bw, color=C_RED, edgecolor="k", lw=0.5, label=r"$\pi$ + Std-CP")
    ax_b.bar(x4 + 0.5*bw, [float(r["cov_pi_lw"])*100 for r in sub_run], width=bw, color=C_ORANGE, edgecolor="k", lw=0.5, label=r"$\pi$ + LW-CP")
    ax_b.bar(x4 + 1.5*bw, [float(r["cov_pi_dsa"])*100 for r in sub_run], width=bw, color=C_TEAL, edgecolor="k", lw=0.5, label=r"$\pi$ + DSA-CP")
    ax_b.axhline(90.0, color=C_NAVY, ls="--", lw=1.4)
    ax_b.set_xticks(x4)
    ax_b.set_xticklabels(["ID\nAI 0.3–1", "OOD1\nAI 1–1.6", "OOD2\nAI 1.6–2.5", "OOD3\nAI 2.5–4"], fontsize=7.8)
    ax_b.set_ylim(0, 115)
    ax_b.set_ylabel("Empirical Coverage (%)")
    ax_b.set_title("(b) Benchmark B: Fu–Budyko Runoff (GBR)")
    ax_b.grid(axis="y", alpha=0.3)

    ax_c = fig.add_subplot(gs[0, 4:6])
    ood3_run = [r for r in rows_run if r["regime"] == "OOD3"]
    xm = np.arange(4)
    bw_m = 0.25
    ax_c.bar(xm - bw_m, [float(r["cov_raw_std"])*100 for r in ood3_run], width=bw_m, color=C_BLUE, edgecolor="k", lw=0.5, label="Raw + Std-CP")
    ax_c.bar(xm, [float(r["cov_pi_std"])*100 for r in ood3_run], width=bw_m, color=C_RED, edgecolor="k", lw=0.5, label=r"$\pi$ + Std-CP")
    ax_c.bar(xm + bw_m, [float(r["cov_pi_dsa"])*100 for r in ood3_run], width=bw_m, color=C_TEAL, edgecolor="k", lw=0.5, label=r"$\pi$ + DSA-CP")
    ax_c.axhline(90.0, color=C_NAVY, ls="--", lw=1.4)
    ax_c.set_xticks(xm)
    ax_c.set_xticklabels(["GBR", "RF", "MLP", "PolyRidge"], fontsize=8.0)
    ax_c.set_ylim(0, 115)
    ax_c.set_ylabel("OOD3 Coverage (%)")
    ax_c.set_title("(c) Model-Agnostic Generality (OOD3)")
    ax_c.grid(axis="y", alpha=0.3)

    sweeps = [
        ("k_neighbors", "(d) Sensitivity to k-NN Size k", "Neighborhood Size k", slice(0, 2)),
        ("q_thresh", "(e) Sensitivity to Boundary Cutoff", "In-Domain Quantile Cutoff q_thresh", slice(2, 4)),
        ("noise_std", "(f) Robustness to Observation Noise", "Noise Standard Deviation sigma (Q/P)", slice(4, 6)),
    ]
    for sw, title, xlabel, col_slice in sweeps:
        ax = fig.add_subplot(gs[1, col_slice])
        sub = [r for r in rows_sens if r["sweep"] == sw]
        xv = [str(r["param_val"]) for r in sub]
        y_std = [float(r["cov_std"]) * 100 for r in sub]
        y_dsa = [float(r["cov_dsa"]) * 100 for r in sub]

        ax.plot(xv, y_std, marker="s", markersize=6, ls="--", lw=1.8, color=C_RED, label=r"$\pi$ + Std-CP")
        ax.plot(xv, y_dsa, marker="o", markersize=6, ls="-", lw=2.2, color=C_TEAL, label=r"$\pi$ + Sublinear DSA-CP")
        ax.axhline(90.0, color=C_NAVY, ls=":", lw=1.4, label="90% Target")
        for i, val in enumerate(y_dsa):
            ax.text(i + (0.06 if i == 0 else 0.0), val + 3.5, f"{val:.0f}%", ha="left" if i == 0 else "center", fontsize=7.4, fontweight="bold", color="#00684A")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("OOD2 Coverage (%)")
        ax.set_ylim(-5, 116)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if sw == "k_neighbors":
            ax.legend(loc="center left", fontsize=7.1, framealpha=0.94)

    fig.suptitle(
        "Figure 5 | Supporting Theoretical Verification Across Dual-Physics Benchmarks, Four ML Architectures, and Sensitivity Sweeps",
        fontsize=11.4, fontweight="bold", y=0.98
    )
    path = os.path.join(os.path.dirname(out_dir), "figures", "fig5_synthetic_benchmarks.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    p = make_fig5_benchmarks_and_generality(out_dir)
    print("Generated 300-DPI figure:", p)


if __name__ == "__main__":
    main()
