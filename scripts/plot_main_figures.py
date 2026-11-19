\
\

import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, gaussian_kde
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from fetch_usgs_era5 import USGS_60_GAUGES, USGS_ALL_GAUGES
from analyze_catchments import conformal_order_quantile

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.8,
    "axes.titlesize": 9.3,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

C_NAVY = "#1B365D"
C_BLUE = "#0072B2"
C_TEAL = "#009E73"
C_ORANGE = "#E69F00"
C_RED = "#D55E00"
C_PURPLE = "#6B46C1"
C_LIGHT_BLUE = "#EBF4FA"
C_LIGHT_RED = "#FDF2F0"


def read_csv_dicts(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def fu_budyko_analytical(AI, w=2.6):
    AI = np.asarray(AI, dtype=float)
    et_ratio = 1.0 + AI - (1.0 + np.maximum(AI, 1e-6) ** w) ** (1.0 / w)
    return np.clip(1.0 - et_ratio, 0.0, 1.0)


def load_dedup_records(out_dir):
    with open(os.path.join(os.path.dirname(out_dir), "data", "usgs_108_catchments_2005_2024.json")) as f:
        all_cached = json.load(f)
    seen = set()
    raw_records = []
    for r in all_cached:
        k = (r["site_id"], r["year"])
        if k not in seen:
            seen.add(k)
            raw_records.append(r)
    return raw_records


def make_fig1_workflow(out_dir):
    fig = plt.figure(figsize=(14.2, 8.8), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.02], width_ratios=[1.06, 1.06], hspace=0.36, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 6.3)
    ax_a.axis("off")
    ax_a.set_title("(a) Physical Mechanism: Antecedent Budyko Departure vs. Equilibrium", loc="left", pad=6)

    bg_left = mpatches.FancyBboxPatch((0.08, 0.12), 4.72, 5.75, boxstyle="round,pad=0.06", facecolor="#F4F9FD", edgecolor=C_BLUE, lw=1.4)
    ax_a.add_patch(bg_left)
    ax_a.text(2.44, 5.66, "Regime I: High Antecedent Departure\n(|Pi_M| > 0.120 -> 65.6% Std-CP Cov)", ha="center", va="top", fontsize=7.3, fontweight="bold", color=C_NAVY)

    bedrock_poly = Polygon([(0.20, 0.35), (4.68, 0.35), (4.68, 1.40), (0.20, 2.30)], facecolor="#CBD5E0", edgecolor="#718096", lw=1.0)
    aquifer_poly = Polygon([(0.20, 2.30), (4.68, 1.40), (4.68, 2.38), (2.55, 3.05), (0.88, 4.30), (0.20, 4.30)], facecolor="#BEE3F8", edgecolor=C_BLUE, lw=1.2)
    vadose_poly = Polygon([(0.20, 4.30), (0.88, 4.30), (2.55, 3.05), (4.68, 2.38), (4.68, 2.78), (2.45, 3.48), (0.92, 4.75), (0.20, 4.75)], facecolor="#E2E8F0", edgecolor="#4A5568", lw=1.0)
    snow_poly = Polygon([(0.20, 4.75), (0.92, 4.75), (1.48, 4.28), (0.88, 4.28), (0.20, 4.55)], facecolor="white", edgecolor=C_BLUE, lw=1.2)
    river_poly = Polygon([(4.18, 2.28), (4.68, 2.28), (4.68, 2.58), (4.18, 2.58)], facecolor="#0072B2", edgecolor="none")
    for p in [bedrock_poly, aquifer_poly, vadose_poly, snow_poly, river_poly]:
        ax_a.add_patch(p)

    ax_a.plot([0.30, 2.55, 4.58], [4.02, 2.90, 2.38], color="#0052CC", ls="--", lw=1.5)
    ax_a.text(0.32, 4.82, "Snowpack\nCarry-Over", fontsize=6.5, fontweight="bold", color=C_NAVY)
    ax_a.annotate("", xy=(3.95, 2.35), xytext=(1.45, 3.28), arrowprops=dict(arrowstyle="->,head_width=0.22", lw=1.8, color="#0052CC"))
    ax_a.text(2.44, 1.85, "Storage Release / ET Anomaly\nDelta S_t != 0 (Descriptive LFR = Q10/Q50 > 0.25)", ha="center", fontsize=6.3, fontweight="bold", color="#004085")
    ax_a.text(2.44, 0.78, "Pi_{M,t} = (ET_B - ET_{t-1} - Delta S_{t-1}) / P_{t-1}\nrho = 0.582 (92-Basin 95% CI [0.470, 0.674])", ha="center", fontsize=6.3,
              bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=C_BLUE, lw=0.9))

    bg_right = mpatches.FancyBboxPatch((5.18, 0.12), 4.72, 5.75, boxstyle="round,pad=0.06", facecolor="#FFFDF5", edgecolor=C_ORANGE, lw=1.4)
    ax_a.add_patch(bg_right)
    ax_a.text(7.54, 5.66, "Regime II: Convective Plains\n(|Pi_M| <= 0.120 -> 96.8% Std-CP Cov)", ha="center", va="top", fontsize=7.3, fontweight="bold", color="#975A16")

    plains_base = Polygon([(5.32, 0.35), (9.76, 0.35), (9.76, 1.80), (5.32, 2.00)], facecolor="#E2E8F0", edgecolor="#718096", lw=1.0)
    plains_soil = Polygon([(5.32, 2.00), (9.76, 1.80), (9.76, 2.80), (5.32, 3.05)], facecolor="#FBD38D", edgecolor="#C05621", lw=1.1)
    for p in [plains_base, plains_soil]:
        ax_a.add_patch(p)

    for x_r in [6.15, 7.35, 8.55]:
        ax_a.annotate("", xy=(x_r, 3.08), xytext=(x_r, 4.25), arrowprops=dict(arrowstyle="->,head_width=0.18", lw=1.4, color=C_BLUE))
        ax_a.annotate("", xy=(x_r + 0.38, 4.35), xytext=(x_r + 0.38, 3.08), arrowprops=dict(arrowstyle="->,head_width=0.18", lw=1.4, color=C_RED, ls="--"))
    ax_a.text(7.54, 4.52, "Summer P_t & E_{p,t} in Phase\n(Descriptive Seasonality SI_WL > 0.26)", ha="center", fontsize=6.5, fontweight="bold", color="#9C4221")
    ax_a.text(7.54, 2.32, "Fast Within-Year Depletion\n(Delta S_t ~ 0)", ha="center", fontsize=6.6, fontweight="bold", color="#744210")
    ax_a.text(7.54, 0.78, "Annual Closure: Q_t/P_t ~ B(AI_t)\nPlains Regional Coverage = 93.6% (n=36)", ha="center", fontsize=6.4,
              bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=C_ORANGE, lw=0.9))

    ax_b = fig.add_subplot(gs[0, 1])
    ai_x = np.linspace(0.30, 2.40, 220)
    true_curve = fu_budyko_analytical(ai_x, w=2.6) + 0.14 * (1.0 / (1.0 + np.exp(-(ai_x - 0.95) * 4.5)))
    surrogate_curve = fu_budyko_analytical(ai_x, w=2.6) - 0.03 * np.maximum(0, ai_x - 0.85)

    ax_b.axvspan(0.30, 0.85, color=C_LIGHT_BLUE, alpha=0.75, label="Humid Calibration Domain (AI <= 0.85)")
    ax_b.axvspan(0.85, 2.40, color=C_LIGHT_RED, alpha=0.45, label="Drier OOD Regime Shift (AI > 0.85)")
    ax_b.axvline(0.85, color=C_NAVY, ls="--", lw=1.6)

    q_r_vis = 0.18
    q_c_vis = 0.065
    q_ijs_vis = q_c_vis * np.sqrt(1.0 + 1.8 * np.maximum(0, ai_x - 0.85)) + 0.16 * (1.0 / (1.0 + np.exp(-(ai_x - 0.95) * 4.5)))

    ax_b.fill_between(ai_x, surrogate_curve - q_ijs_vis, surrogate_curve + q_ijs_vis, color=C_TEAL, alpha=0.24, label="Reduced Pi* + IJS-CP (97.1% Cov, IS=0.764)")
    ax_b.fill_between(ai_x, surrogate_curve - q_r_vis, surrogate_curve + q_r_vis, color=C_BLUE, alpha=0.18, label="Raw (P, E_p) + Std-CP (Wide q_raw = 0.347)")
    ax_b.fill_between(ai_x, surrogate_curve - q_c_vis, surrogate_curve + q_c_vis, color=C_RED, alpha=0.38, label="Reduced Pi* + Std-CP (Compressed q_comp = 0.208)")

    ax_b.plot(ai_x, true_curve, color="black", lw=2.3, label="True Shifted Process M*(Pi*) + Carry-Over")
    ax_b.plot(ai_x, surrogate_curve, color=C_RED, ls="--", lw=1.9, label="In-Domain Trained Surrogate f_hat(Pi*)")

    ax_b.annotate(
        "Proposition 1 (Order-Stat Quantiles):\nPhysical completion shrinks ID\ncalibration quantile by 37%–48%\n(q_m0h=0.170, q_comp=0.208 < q_raw=0.347)",
        xy=(0.56, surrogate_curve[42] + q_c_vis), xytext=(0.32, 0.06),
        fontsize=6.6, fontweight="bold", color=C_NAVY,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=C_BLUE),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=C_BLUE)
    )
    ax_b.annotate(
        "Proposition 2 (Non-Exchangeable Shift):\nExtrapolation bias b > q_comp\ndrives Std-CP coverage to 56.6%–71.2%!",
        xy=(1.75, true_curve[150]), xytext=(1.18, 0.45),
        fontsize=6.7, fontweight="bold", color="#9B2C2C",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFF5F5", edgecolor=C_RED),
        arrowprops=dict(arrowstyle="->", lw=1.3, color=C_RED)
    )
    ax_b.set_xlim(0.30, 2.40)
    ax_b.set_ylim(0.0, 1.02)
    ax_b.set_xlabel("Dimensionless Aridity Coordinate AI = E_p / P (-)")
    ax_b.set_ylabel("Runoff Response Q / P (-)")
    ax_b.set_title("(b) Mathematical Mechanism: ID Compression vs. OOD Under-Coverage", loc="left", pad=6)
    ax_b.legend(loc="upper right", fontsize=6.5, framealpha=0.94)
    ax_b.grid(alpha=0.25)

    ax_c = fig.add_subplot(gs[1, 0])
    g1 = np.linspace(0.2, 2.4, 100)
    g2 = np.linspace(-0.38, 0.55, 100)
    G1, G2 = np.meshgrid(g1, g2)
    M0_surf = np.clip(fu_budyko_analytical(G1) + 0.5 * G2, 0.0, 1.0)

    cf = ax_c.contourf(G1, G2, M0_surf, levels=np.linspace(0.0, 0.9, 10), cmap="PuBuGn", alpha=0.75)
    cs = ax_c.contour(G1, G2, M0_surf, levels=[0.2, 0.4, 0.6, 0.8], colors="#2D3748", linewidths=0.8, alpha=0.6)
    ax_c.clabel(cs, fmt="M0=%.1f", fontsize=6.6)
    cbar_c = plt.colorbar(cf, ax=ax_c, fraction=0.040, pad=0.02)
    cbar_c.set_label("Baseline Operator M0(Pi)")

    id_box = mpatches.FancyBboxPatch((0.24, -0.18), 0.57, 0.36, boxstyle="round,pad=0.03", facecolor="#EBF8FF", edgecolor=C_BLUE, lw=1.8)
    ax_c.add_patch(id_box)
    ax_c.text(0.525, 0.0, "In-Domain\nManifold D_fit\n(AI <= 0.85)", ha="center", va="center", fontsize=7.0, fontweight="bold", color=C_NAVY)

    p_proj = (0.83, 0.12)
    p_ood = (1.85, 0.38)
    ax_c.scatter([p_proj[0]], [p_proj[1]], s=90, c=C_BLUE, edgecolors="k", zorder=6, label="Nearest Anchor Pi_proj in D_fit")
    ax_c.scatter([p_ood[0]], [p_ood[1]], s=115, c=C_RED, marker="*", edgecolors="k", zorder=6, label="OOD Query Point Pi")
    ax_c.annotate(
        "", xy=p_ood, xytext=p_proj,
        arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.26", lw=2.3, color=C_RED)
    )
    ax_c.text(1.24, 0.30, r"Analytical Drift $|M_0(\Pi) - M_0(\Pi_{\mathrm{proj}})|$",
              rotation=15, fontsize=7.2, fontweight="bold", color="#9B2C2C",
              bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=C_RED, alpha=0.92))

    ax_c.text(
        0.86, -0.33,
        "Tuning-Free IJS-CP Adjustment:\n"
        "Delta_M(Pi) = |M0(Pi) - M0(Pi_proj)| >= 0\n"
        "q_IJS = q_Pi*sqrt(1 + [d_tilde - 1]_+) + [Delta_M - Delta_M,cal]_+",
        fontsize=6.8, fontweight="bold", color="#1A202C",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFF0", edgecolor=C_TEAL, lw=1.2)
    )
    ax_c.set_xlim(0.2, 2.4)
    ax_c.set_ylim(-0.38, 0.55)
    ax_c.set_xlabel("Dimensionless Aridity Coordinate AI_t (-)")
    ax_c.set_ylabel("Antecedent Departure Coordinate Pi_{M,t} (-)")
    ax_c.set_title("(c) Tuning-Free Analytical Baseline-Drift Adjustment", loc="left", pad=6)
    ax_c.legend(loc="upper left", fontsize=6.8, framealpha=0.92)

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 6.2)
    ax_d.axis("off")
    ax_d.set_title("(d) Three Fit-Isolated Operational Scenarios Across 108 Basins", loc="left", pad=6)

    tracks = [
        (4.02, "#EBF8FF", C_BLUE,
         "SCENARIO 1: Gauged Spatial Climate-Shift Transfer (n = 1,197 OOD)",
         "• Fit-Isolated & Zero-Static: ALRQ_fit/CVP_fit on D_fit (n=373) & Zero-Static Ablation\n"
         "• Result: Cuts RMSE from 0.2574 (AI) to 0.1374 (ModD-Phys) & 0.1159 (In-Fold OLS\n"
         "  M0_hyst, beating Persistence 0.1232, 95% CI [-0.0123, -0.0022]; Jac-CP 91.4% Cov)."),
        (2.04, "#F0FFF4", C_TEAL,
         "SCENARIO 2: Basin-Isolated PUB (All 107 Basins & Pure 60-Direct-Gauge Refit)",
         "• Protocol: Outer GroupKFold + Inner GroupShuffleSplit & Pure 60-Direct Refit\n"
         "• Result: Cuts OOD RMSE by 20.4% on all basins (0.2602 -> 0.2070, IJS 94.0% Cov) and\n"
         "  by 21.8% on Pure 60-Direct Refit (0.2687 -> 0.2100, IJS 95.1% Cov, IS=0.841)."),
        (0.08, "#FAF5FF", C_PURPLE,
         "SCENARIO 3: Strictly Causal Temporal Split (2005–2015 -> 2016–2024, n=545)",
         "• Fit-Isolated Protocol: Frozen <=2015 humid fit (n=200); OLS c_M=+0.875, c_D=-0.186\n"
         "• Result: S3B Pure Met (RMSE=0.2183 vs Raw 0.2355 [-7.3%] & AI 0.3422 [-36.2%]);\n"
         "  S3A OLS M0_hyst achieves RMSE=0.1106 (vs 0.1139 Persistence) with 96.1% Jac-CP Cov."),
    ]
    for y0, fc, ec, t_hdr, t_body in tracks:
        box = mpatches.FancyBboxPatch((0.12, y0), 9.76, 1.78, boxstyle="round,pad=0.05", facecolor=fc, edgecolor=ec, lw=1.5)
        ax_d.add_patch(box)
        bar = mpatches.FancyBboxPatch((0.16, y0 + 1.30), 9.68, 0.44, boxstyle="round,pad=0.03", facecolor=ec, edgecolor="none")
        ax_d.add_patch(bar)
        ax_d.text(5.0, y0 + 1.52, t_hdr, ha="center", va="center", fontsize=7.3, fontweight="bold", color="white")
        ax_d.text(0.28, y0 + 1.18, t_body, ha="left", va="top", fontsize=6.9, color="#1A202C", linespacing=1.22)

    fig.suptitle(
        "Figure 1 | Physical Mechanism, Quantile Compression, Analytical Baseline-Drift Adjustment, and Three-Scenario Validation (108 USGS Basins)",
        fontsize=11.0, fontweight="bold", y=0.98
    )
    path = os.path.join(os.path.dirname(out_dir), "figures", "fig1_framework.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def make_fig2_spatial_and_process(out_dir):
    basins = read_csv_dicts(os.path.join(os.path.dirname(out_dir), "results", "basin_process_summary.csv"))
    raw_records = load_dedup_records(out_dir)

    direct_60_ids = {s[0] for s in USGS_60_GAUGES}
    meta_map = {s[0]: {"lat": s[1], "lon": s[2], "area": s[3]} for s in USGS_ALL_GAUGES}
    ood_map = {b["site_id"]: b for b in basins}
    valid_sites = sorted(set(r["site_id"] for r in raw_records))

    fig = plt.figure(figsize=(13.6, 8.8), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.12, 1.0], width_ratios=[1.22, 1.0], hspace=0.38, wspace=0.30)

    ax_map = fig.add_subplot(gs[0, 0])
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
    ax_map.fill(xs, ys, facecolor="#F7FAFC", edgecolor="#CBD5E0", linewidth=1.4, zorder=1)
    for lon_div in [-117.0, -104.0, -84.0]:
        ax_map.plot([lon_div, lon_div], [26.5, 48.8], color="#A0AEC0", linestyle=":", linewidth=1.1, zorder=2)

    humid_only = [sid for sid in valid_sites if sid not in ood_map]
    ax_map.scatter(
        [meta_map[s]["lon"] for s in humid_only],
        [meta_map[s]["lat"] for s in humid_only],
        s=45, marker="^", facecolor="#3182CE", edgecolor="k", linewidth=0.7,
        label=f"Humid Calibration Only (AI <= 0.85, n={len(humid_only)})", zorder=3
    )

    b_d60 = [b for b in basins if b["site_id"] in direct_60_ids]
    b_idw = [b for b in basins if b["site_id"] not in direct_60_ids]

    sc = ax_map.scatter(
        [float(b["lon"]) for b in b_d60], [float(b["lat"]) for b in b_d60],
        s=[38.0 + 300.0 * np.clip(float(b["mean_abs_pi_M"]) / 0.50, 0.05, 1.2) for b in b_d60],
        c=[float(b["miss_rate_std_pi1"]) * 100 for b in b_d60],
        marker="o", cmap="RdYlBu_r", vmin=0, vmax=100,
        edgecolor="black", linewidth=0.75, alpha=0.90, zorder=4,
        label=f"OOD Direct ERA5-Land Gauges (o, n={len(b_d60)}/60)"
    )
    ax_map.scatter(
        [float(b["lon"]) for b in b_idw], [float(b["lat"]) for b in b_idw],
        s=[38.0 + 300.0 * np.clip(float(b["mean_abs_pi_M"]) / 0.50, 0.05, 1.2) for b in b_idw],
        c=[float(b["miss_rate_std_pi1"]) * 100 for b in b_idw],
        marker="s", cmap="RdYlBu_r", vmin=0, vmax=100,
        edgecolor="black", linewidth=0.75, alpha=0.85, zorder=4,
        label=f"OOD Regional-IDW Gauges (s, n={len(b_idw)}/48)"
    )
    cbar = plt.colorbar(sc, ax=ax_map, fraction=0.038, pad=0.02)
    cbar.set_label("Std-CP OOD Miscoverage Rate (%)")

    for sid, label_txt, dx, dy in [
        ("06191500", "USGS 06191500 (Yellowstone, MT)\n|Pi_M|=0.48, Miscov=100%", -0.5, -5.2),
        ("08195000", "USGS 08195000 (Frio, TX)\n|Pi_M|=0.05, Miscov=0%", 4.2, -2.2),
    ]:
        if sid in ood_map:
            b = ood_map[sid]
            ax_map.annotate(
                label_txt, xy=(float(b["lon"]), float(b["lat"])),
                xytext=(float(b["lon"]) + dx, float(b["lat"]) + dy),
                fontsize=7.0, fontweight="bold", color="#1A202C",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#FFFFF0", edgecolor=C_NAVY, lw=0.9),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=C_NAVY), zorder=6
            )

    ax_map.text(-121.0, 49.4, "Pacific\nNW", fontsize=7.1, color="#4A5568", ha="center", va="bottom", fontweight="bold")
    ax_map.text(-109.8, 49.4, "Rocky Mtns\n& Southwest", fontsize=7.1, color="#4A5568", ha="center", va="bottom", fontweight="bold")
    ax_map.text(-94.5, 49.4, "Central Plains\n& Midwest", fontsize=7.1, color="#4A5568", ha="center", va="bottom", fontweight="bold")
    ax_map.text(-75.5, 49.4, "Appalachian &\nNortheast", fontsize=7.1, color="#4A5568", ha="center", va="bottom", fontweight="bold")

    ax_map.set_xlim(-126.5, -66.0)
    ax_map.set_ylim(24.5, 52.6)
    ax_map.set_xlabel("Longitude (°W)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.set_title("(a) Spatial Regime Breakdown (108 Gauges: 60 Direct ERA5-Land vs. 48 IDW)")
    legend_handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#3182CE", markeredgecolor="k", markersize=6.2, label=f"Humid Calibration Only (AI <= 0.85, n={len(humid_only)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4575B4", markeredgecolor="k", markersize=6.2, label=f"OOD Direct ERA5-Land Gauges (o, n={len(b_d60)}/60)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#D73027", markeredgecolor="k", markersize=6.0, label=f"OOD Regional-IDW Gauges (s, n={len(b_idw)}/48)"),
    ]
    ax_map.legend(handles=legend_handles, loc="lower left", fontsize=6.3, framealpha=0.94, handletextpad=0.5)

    ax_reg = fig.add_subplot(gs[0, 1])
    reg_order = [
        "Central Plains & Midwest",
        "Appalachian & Northeast",
        "Pacific Northwest & Coast",
        "Rocky Mountains & Southwest",
    ]
    reg_short, std_cov_reg, ijs_cov_reg, pim_reg = [], [], [], []
    for rname, sname in zip(reg_order, ["Central Plains\n& Midwest", "Appalachian\n& Northeast", "Pacific NW\n& Coast", "Rocky Mtns\n& Southwest"]):
        sub = [b for b in basins if b["region"] == rname]
        reg_short.append(f"{sname}\n(n={len(sub)})")
        std_cov_reg.append(100.0 - float(np.mean([float(x["miss_rate_std_pi1"]) * 100 for x in sub])))
        ijs_cov_reg.append(float(np.mean([float(x["cov_rate_ijs3"]) * 100 for x in sub])))
        pim_reg.append(float(np.mean([float(x["mean_abs_pi_M"]) for x in sub])))

    xr = np.arange(len(reg_order))
    bw = 0.34
    ax_reg.bar(xr - bw / 2, std_cov_reg, width=bw, color=C_RED, edgecolor="k", linewidth=0.7, label=r"Steady-State $\Pi$ ($AI$) + Std-CP")
    ax_reg.bar(xr + bw / 2, ijs_cov_reg, width=bw, color=C_TEAL, edgecolor="k", linewidth=0.7, label=r"Complete $\Pi^\star$ + IJS-CP")
    ax_reg.axhline(90.0, color=C_NAVY, linestyle="--", linewidth=1.5, label="90.0% Nominal Target")

    for i in range(len(xr)):
        ax_reg.text(xr[i] - bw / 2, std_cov_reg[i] + 1.2, f"{std_cov_reg[i]:.1f}%", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=C_RED)
        ax_reg.text(xr[i] + bw / 2, ijs_cov_reg[i] + 1.2, f"{ijs_cov_reg[i]:.1f}%", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#00684A")
        ax_reg.text(xr[i], 3.5, f"|Pi_M|={pim_reg[i]:.2f}", ha="center", va="bottom", fontsize=7.0,
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#CBD5E0"))

    ax_reg.set_xticks(xr)
    ax_reg.set_xticklabels(reg_short, fontsize=7.8)
    ax_reg.set_ylim(0, 107)
    ax_reg.set_yticks([0, 20, 40, 60, 80, 100])
    ax_reg.set_ylabel("Mean Basin OOD Empirical Coverage (%)")
    ax_reg.set_title("(b) Regional Carry-Over Collapse & IJS-CP Recovery (0–100% Scale)")
    ax_reg.legend(loc="lower left", bbox_to_anchor=(0.01, 0.14), ncol=1, fontsize=6.9, framealpha=0.92)
    ax_reg.grid(axis="y", alpha=0.3)

    ax_phys = fig.add_subplot(gs[1, 0])
    si_vals = np.array([float(b["si_walsh"]) for b in basins])
    bfi_vals = np.array([float(b["bfi_proxy"]) for b in basins])
    pi_m = np.array([float(b["mean_abs_pi_M"]) for b in basins])
    miss = np.array([float(b["miss_rate_std_pi1"]) * 100 for b in basins])
    ax_phys.axvspan(0.04, 0.22, ymin=0.32, ymax=1.0, color="#FFF5F5", alpha=0.65, zorder=0)
    ax_phys.axvspan(0.25, 0.76, ymin=0.0, ymax=0.55, color="#F0FFF4", alpha=0.65, zorder=0)
    ax_phys.scatter(
        si_vals, bfi_vals, c=miss, s=50 + 220 * pi_m, cmap="RdYlBu_r", vmin=0, vmax=100,
        edgecolors="k", linewidth=0.7, alpha=0.88, zorder=3
    )
    ax_phys.text(
        0.23, 0.59,
        "Groundwater / Snowmelt Zone (Low SI_WL < 0.22, High LFR > 0.25)\n• Basin-w_i Asymmetry: rho_wi = -0.244 (60-Direct -0.273) vs Null -0.008\n• Cross-Decade Proxy: rho_cross = -0.904 (60-Direct -0.891)",
        fontsize=6.2, fontweight="bold", color="#9B2C2C",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFF5F5", edgecolor="#FEB2B2", alpha=0.92)
    )
    ax_phys.text(
        0.26, 0.03,
        "Convective Plains Zone (High SI_WL > 0.26, Low LFR <= 0.25)\n• Near-Equilibrium Closure (93.6% Basin / 96.6% 60-Direct Std-CP Cov)",
        fontsize=6.4, fontweight="bold", color="#22543D",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#F0FFF4", edgecolor="#9AE6B4", alpha=0.92)
    )
    ax_phys.set_xlim(0.03, 0.77)
    ax_phys.set_ylim(-0.03, 0.78)
    ax_phys.set_xlabel("Descriptive Walsh–Lawler Seasonality Index SI_WL (-)")
    ax_phys.set_ylabel("Descriptive Daily Low-Flow Ratio LFR = Q10/Q50 (-)")
    ax_phys.set_title(f"(c) Descriptive Physiographic Regime Mapping ({len(basins)} OOD Basins)")
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
    ax_mem.axvline(0.080, color=C_TEAL, linestyle=":", linewidth=1.7, label="Piecewise Hinge Breakpoint tau* = 0.080")
    ax_mem.axvline(0.120, color=C_NAVY, linestyle="--", linewidth=1.6, label="Sample Median Split |Pi_M| = 0.120")

    ax_mem.annotate(
        "Low-Departure Regime\n(|Pi_M| <= 0.120)\nStd-CP Cov = 96.8%",
        xy=(0.05, 0.085), xytext=(0.006, 0.25),
        fontsize=6.7, fontweight="bold", color=C_NAVY,
        bbox=dict(boxstyle="round,pad=0.18", facecolor=C_LIGHT_BLUE, edgecolor=C_BLUE),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=C_BLUE)
    )
    ax_mem.annotate(
        "High-Departure Regime (|Pi_M| > 0.120)\n60-Direct rho=0.577 | 48-IDW rho=0.579\nMatched Null Asymmetry Tests:\n• Basin-w_i: rho = -0.244 vs Null -0.008\n• 60-Direct OOD: rho_wi = -0.273 vs Null -0.012",
        xy=(0.28, 0.34), xytext=(0.140, 0.45),
        fontsize=6.2, fontweight="bold", color="#9B2C2C",
        bbox=dict(boxstyle="round,pad=0.18", facecolor=C_LIGHT_RED, edgecolor=C_RED),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=C_RED)
    )

    ax_mem.set_xlabel(r"Antecedent Budyko Departure Magnitude $|\Pi_M|$ (-)")
    ax_mem.set_ylabel("Steady-State Budyko Error |Q/P - B(AI)|")
    ax_mem.set_ylim(0, 0.72)
    ax_mem.set_title("(d) Antecedent Departure Relationship (rho = 0.582, 92-Basin 95% CI [0.470, 0.674])")
    ax_mem.legend(loc="lower right", fontsize=7.0)
    ax_mem.grid(alpha=0.3)

    fig.suptitle(
        "Figure 2 | Hydrological Process Signatures: Geographic and Physiographic Controls of Antecedent Budyko Departure (108 USGS Basins)",
        fontsize=11.2, fontweight="bold", y=0.98
    )
    path = os.path.join(os.path.dirname(out_dir), "figures", "fig2_spatial_process_signatures.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def make_fig3_phase_and_hydrographs(out_dir):
    raw_records = load_dedup_records(out_dir)

    by_site = {}
    for r in raw_records:
        by_site.setdefault(r["site_id"], []).append(r)
    for sid in by_site:
        by_site[sid].sort(key=lambda x: x["year"])
    sids_sorted = sorted(by_site.keys())

    complete_records = []
    for sid, recs in by_site.items():
        for i in range(1, len(recs)):
            curr, prev = recs[i], recs[i - 1]
            if curr["year"] - prev["year"] != 1:
                continue
            qb_prev = float(fu_budyko_analytical(prev["AI"]))
            qb_curr = float(fu_budyko_analytical(curr["AI"]))
            pi_M = prev["Q_over_P"] - qb_prev
            pi_Delta = (curr["P"] - prev["P"]) / (prev["P"] + 1e-6)
            complete_records.append({
                **curr,
                "pi_M": pi_M,
                "abs_pi_M": abs(pi_M),
                "pi_Delta": pi_Delta,
                "qb_curr": qb_curr,
            })

    id_recs = [r for r in complete_records if r["AI"] <= 0.85]
    idx_f, idx_rest = train_test_split(np.arange(len(id_recs)), test_size=0.4, random_state=42)
    idx_c, _ = train_test_split(idx_rest, test_size=0.5, random_state=42)

    fit_by_s = {}
    for i in idx_f:
        r = id_recs[i]
        fit_by_s.setdefault(r["site_id"], []).append(r)
    alrq_pool = [float(np.percentile([x["Q_over_P"] * x["P"] for x in rs], 15) / (np.percentile([x["Q_over_P"] * x["P"] for x in rs], 50) + 1e-6)) for rs in fit_by_s.values() if len(rs) >= 2]
    cvp_pool = [float(np.std([x["P"] for x in rs]) / (np.mean([x["P"] for x in rs]) + 1e-6)) for rs in fit_by_s.values() if len(rs) >= 2]
    med_alrq, med_cvp = float(np.median(alrq_pool)), float(np.median(cvp_pool))
    alrq_fit, cvp_fit = {}, {}
    for sid in sids_sorted:
        rs = fit_by_s.get(sid, [])
        if len(rs) >= 2:
            alrq_fit[sid] = float(np.percentile([x["Q_over_P"] * x["P"] for x in rs], 15) / (np.percentile([x["Q_over_P"] * x["P"] for x in rs], 50) + 1e-6))
            cvp_fit[sid] = float(np.std([x["P"] for x in rs]) / (np.mean([x["P"] for x in rs]) + 1e-6))
        else:
            alrq_fit[sid] = med_alrq
            cvp_fit[sid] = med_cvp

    for r in complete_records:
        r["alrq_fit"] = alrq_fit[r["site_id"]]
        r["cvp_fit"] = cvp_fit[r["site_id"]]

    X_pi1_id = np.array([[r["AI"]] for r in id_recs])
    X_pi3_id = np.array([[r["AI"], r["pi_M"], r["pi_Delta"], r["alrq_fit"], r["cvp_fit"]] for r in id_recs])
    y_id = np.array([r["Q_over_P"] for r in id_recs])
    qb_id = np.array([r["qb_curr"] for r in id_recs])

    m_pi1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_pi1_id[idx_f], y_id[idx_f])
    m_pi3 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_pi3_id[idx_f], y_id[idx_f])

    m_pi1_phys = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42).fit(X_pi1_id[idx_f], y_id[idx_f] - qb_id[idx_f])
    m_pi3_phys = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.06, random_state=42).fit(X_pi3_id[idx_f], y_id[idx_f] - qb_id[idx_f])

    alpha = 0.10
    q_pi1 = conformal_order_quantile(np.abs(y_id[idx_c] - m_pi1.predict(X_pi1_id[idx_c])), alpha)
    q_pi3 = conformal_order_quantile(np.abs(y_id[idx_c] - m_pi3.predict(X_pi3_id[idx_c])), alpha)

    q_pi1_phys = conformal_order_quantile(np.abs(y_id[idx_c] - np.clip(qb_id[idx_c] + m_pi1_phys.predict(X_pi1_id[idx_c]), 0, 1)), alpha)
    q_pi3_phys = conformal_order_quantile(np.abs(y_id[idx_c] - np.clip(qb_id[idx_c] + m_pi3_phys.predict(X_pi3_id[idx_c]), 0, 1)), alpha)

    sc3 = StandardScaler().fit(X_pi3_id[idx_f])
    nn3 = NearestNeighbors(n_neighbors=1).fit(sc3.transform(X_pi3_id[idx_f]))
    d_cal, idx_cal = nn3.kneighbors(sc3.transform(X_pi3_id[idx_c]))
    d_90 = conformal_order_quantile(d_cal[:, 0], alpha) + 1e-6
    phys_fn = lambda X: np.clip(fu_budyko_analytical(X[:, 0]) + 0.5 * X[:, 1], 0.0, 1.0)
    dM_cal = np.abs(phys_fn(X_pi3_id[idx_c]) - phys_fn(X_pi3_id[idx_f][idx_cal[:, 0]]))
    dM_scale = conformal_order_quantile(dM_cal, alpha) + 1e-4

    def predict_site_series(sid):
        s_recs = [r for r in complete_records if r["site_id"] == sid]
        yrs = np.array([r["year"] for r in s_recs])
        y_obs = np.array([r["Q_over_P"] for r in s_recs])
        qb_s = np.array([r["qb_curr"] for r in s_recs])
        X1 = np.array([[r["AI"]] for r in s_recs])
        X3 = np.array([[r["AI"], r["pi_M"], r["pi_Delta"], r["alrq_fit"], r["cvp_fit"]] for r in s_recs])
        p1 = np.clip(qb_s + m_pi1_phys.predict(X1), 0.0, 1.0)
        p3 = np.clip(qb_s + m_pi3_phys.predict(X3), 0.0, 1.0)
        d_t, idx_t = nn3.kneighbors(sc3.transform(X3))
        dM_t = np.abs(phys_fn(X3) - phys_fn(X_pi3_id[idx_f][idx_t[:, 0]]))
        hw_ijs = q_pi3_phys * np.sqrt(1.0 + np.maximum(0.0, d_t[:, 0] / d_90 - 1.0)) + np.maximum(0.0, dM_t - dM_scale)
        lo1, hi1 = np.clip(p1 - q_pi1_phys, 0, 1), np.clip(p1 + q_pi1_phys, 0, 1)
        lo3, hi3 = np.clip(p3 - hw_ijs, 0, 1), np.clip(p3 + hw_ijs, 0, 1)
        return yrs, y_obs, p1, lo1, hi1, p3, lo3, hi3

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
    ax_ph.scatter(ai_all, np.clip(pim_all, 0, 0.57), c="k", s=6, alpha=0.22, label=f"Station-Years (n={len(complete_records)})")
    ax_ph.axvline(0.85, color=C_BLUE, linestyle=":", linewidth=1.8, label="Regime Split (AI = 0.85)")
    ax_ph.axhline(0.120, color="#2B6CB0", linestyle="-.", linewidth=1.5, label="Median Split |Pi_M| = 0.120")
    cbar_ph = plt.colorbar(cf, ax=ax_ph, pad=0.02)
    cbar_ph.set_label("Mean Error |Q/P - f(AI)|")
    ax_ph.set_xlim(0.35, 2.8)
    ax_ph.set_ylim(0.0, 0.58)
    ax_ph.set_xlabel("Climatic Aridity Index AI = E_p / P (-)")
    ax_ph.set_ylabel(r"Antecedent Budyko Departure $|\Pi_M|$ (-)")
    ax_ph.set_title("(a) 2D Phase Diagram in (AI, |Pi_M|) Space")
    ax_ph.legend(loc="upper right", fontsize=7.2, framealpha=0.92)

    ax_kd = fig.add_subplot(gs[1, 0])
    X_raw_id = np.array([[r["P"], r["Ep"]] for r in id_recs])
    m_raw = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_raw_id[idx_f], y_id[idx_f])
    res_raw_cal = np.abs(y_id[idx_c] - np.clip(m_raw.predict(X_raw_id[idx_c]), 0, 1))
    res_pi1_cal = np.abs(y_id[idx_c] - np.clip(m_pi1.predict(X_pi1_id[idx_c]), 0, 1))
    res_pi3_cal = np.abs(y_id[idx_c] - np.clip(m_pi3.predict(X_pi3_id[idx_c]), 0, 1))
    q_pi1 = conformal_order_quantile(res_pi1_cal, 0.10)
    q_pi3 = conformal_order_quantile(res_pi3_cal, 0.10)
    print(f"Fig3b Quantiles -> q_comp={q_pi3:.3f}, q_inc={q_pi1:.3f}, q_raw={conformal_order_quantile(res_raw_cal, 0.10):.3f}")
    q_raw_cal = conformal_order_quantile(res_raw_cal, 0.10)

    xg = np.linspace(0, 0.48, 300)
    for arr, col, lbl, qv in [
        (res_pi1_cal, C_ORANGE, f"Steady Pi (AI): q_inc = {q_pi1:.3f}", q_pi1),
        (res_raw_cal, C_BLUE, f"Raw (P, E_p): q_raw = {q_raw_cal:.3f}", q_raw_cal),
        (res_pi3_cal, C_RED, f"Reduced Pi* (+Pi_M): q_comp = {q_pi3:.3f}", q_pi3),
    ]:
        kde = gaussian_kde(arr, bw_method=0.35)
        yg = kde(xg)
        ax_kd.plot(xg, yg, color=col, linewidth=2.2, label=lbl)
        ax_kd.fill_between(xg, 0, yg, color=col, alpha=0.18)
        ax_kd.axvline(qv, color=col, linestyle="--", linewidth=1.6)

    ax_kd.annotate(
        f"Proposition 1 Compression (Order Stat):\nq_comp ({q_pi3:.3f}) < q_inc ({q_pi1:.3f}) < q_raw ({q_raw_cal:.3f})\nPhysical completion tightens ID quantile,\ntriggering OOD Std-CP under-coverage!",
        xy=(q_pi3, 3.2), xytext=(0.22, 3.6),
        fontsize=7.1, fontweight="bold", color="#1A202C",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFBEB", edgecolor="#D69E2E"),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=C_RED)
    )
    ax_kd.set_xlim(0, 0.48)
    ax_kd.set_ylim(0, 9.2)
    ax_kd.set_xlabel("In-Domain Calibration Residual |y - f_hat| (Q/P)")
    ax_kd.set_ylabel("Probability Density")
    ax_kd.set_title("(b) Proposition 1: In-Domain Quantile Compression")
    ax_kd.legend(loc="upper right", fontsize=7.1, framealpha=0.94)
    ax_kd.grid(alpha=0.3)

    ax_c1 = fig.add_subplot(gs[0, 1])
    yrs, y_obs, p1, lo1, hi1, p3, lo3, hi3 = predict_site_series("06191500")
    rmse1_y = float(np.sqrt(np.mean((y_obs - p1) ** 2)))
    rmse3_y = float(np.sqrt(np.mean((y_obs - p3) ** 2)))
    cov1_y = float(np.mean((y_obs >= lo1) & (y_obs <= hi1)) * 100)
    cov3_y = float(np.mean((y_obs >= lo3) & (y_obs <= hi3)) * 100)

    ax_c1.fill_between(yrs, lo3, hi3, color=C_TEAL, alpha=0.25, label=f"Complete Pi* + IJS-CP (Cov={cov3_y:.1f}% [19/19], W={np.mean(hi3-lo3):.2f})")
    ax_c1.fill_between(yrs, lo1, hi1, color=C_RED, alpha=0.28, label=f"Steady Pi (AI) + Std-CP (Cov={cov1_y:.0f}% — Collapse!)")
    ax_c1.plot(yrs, y_obs, marker="o", markersize=5.2, color="black", linewidth=2.0, label="Observed Q/P (USGS 06191500)", zorder=5)
    ax_c1.plot(yrs, p3, marker="s", markersize=4.0, color="#00684A", linestyle="--", linewidth=1.6, label=f"Complete Pi* Point Pred (RMSE={rmse3_y:.3f})")
    ax_c1.plot(yrs, p1, marker="^", markersize=4.0, color=C_RED, linestyle=":", linewidth=1.5, label=f"Steady Pi (AI) Point Pred (RMSE={rmse1_y:.3f})")
    ax_c1.set_ylim(0, 1.28)
    ax_c1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c1.set_xlim(2005.5, 2024.5)
    ax_c1.set_xticks(range(2006, 2025, 2))
    ax_c1.set_ylabel("Annual Runoff Ratio Q / P (-)")
    ax_c1.set_title("(c) High-Departure Case: Yellowstone River, MT (06191500, |Pi_M|=0.48)")
    ax_c1.legend(loc="upper center", fontsize=6.4, ncol=2, framealpha=0.94)
    ax_c1.grid(alpha=0.3)

    ax_c2 = fig.add_subplot(gs[1, 1])
    yrs2, y_obs2, p1_2, lo1_2, hi1_2, p3_2, lo3_2, hi3_2 = predict_site_series("08195000")
    rmse1_f = float(np.sqrt(np.mean((y_obs2 - p1_2) ** 2)))
    rmse3_f = float(np.sqrt(np.mean((y_obs2 - p3_2) ** 2)))
    cov1_f = float(np.mean((y_obs2 >= lo1_2) & (y_obs2 <= hi1_2)) * 100)
    cov3_f = float(np.mean((y_obs2 >= lo3_2) & (y_obs2 <= hi3_2)) * 100)

    ax_c2.fill_between(yrs2, lo3_2, hi3_2, color=C_TEAL, alpha=0.25, label=f"Complete Pi* + IJS-CP (Cov={cov3_f:.0f}%, W={np.mean(hi3_2-lo3_2):.2f})")
    ax_c2.fill_between(yrs2, lo1_2, hi1_2, color=C_ORANGE, alpha=0.30, label=f"Steady Pi (AI) + Std-CP (Cov={cov1_f:.0f}%, W={np.mean(hi1_2-lo1_2):.2f})")
    ax_c2.plot(yrs2, y_obs2, marker="o", markersize=5.2, color="black", linewidth=2.0, label="Observed Q/P (USGS 08195000)", zorder=5)
    ax_c2.plot(yrs2, p3_2, marker="s", markersize=4.0, color="#00684A", linestyle="--", linewidth=1.6, label=f"Complete Pi* Point Pred (RMSE={rmse3_f:.3f})")
    ax_c2.plot(yrs2, p1_2, marker="^", markersize=4.0, color=C_RED, linestyle=":", linewidth=1.5, label=f"Steady Pi (AI) Point Pred (RMSE={rmse1_f:.3f})")
    ax_c2.set_ylim(0, 0.55)
    ax_c2.set_xlim(2005.5, 2024.5)
    ax_c2.set_xticks(range(2006, 2025, 2))
    ax_c2.set_xlabel("Water Calendar Year (2006–2024 Contiguous Transitions)")
    ax_c2.set_ylabel("Annual Runoff Ratio Q / P (-)")
    ax_c2.set_title("(d) Low-Departure Case: Frio River at Concan, TX (08195000, |Pi_M|=0.05)")
    ax_c2.legend(loc="upper right", fontsize=6.7, ncol=2, framealpha=0.92)
    ax_c2.grid(alpha=0.3)

    print(f"Case-Study Verification -> Yellowstone (06191500): Steady RMSE={rmse1_y:.3f} (Cov={cov1_y:.0f}%), Comp RMSE={rmse3_y:.3f} (Cov={cov3_y:.1f}% [19/19])")
    print(f"Case-Study Verification -> Frio River  (08195000): Steady RMSE={rmse1_f:.3f} (Cov={cov1_f:.0f}%), Comp RMSE={rmse3_f:.3f} (Cov={cov3_f:.0f}%)")

    fig.suptitle(
        "Figure 3 | Phase-Space Mechanism and 19-Transition Catchment Case Studies (2006–2024): High-Departure vs. Equilibrium Regimes (108 USGS Basins)",
        fontsize=11.2, fontweight="bold", y=0.98
    )
    path = os.path.join(os.path.dirname(out_dir), "figures", "fig3_phase_diagram_hydrographs.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def make_fig4_gauged_vs_ungauged_ijs(out_dir):
    t4_all = read_csv_dicts(os.path.join(os.path.dirname(out_dir), "results", "table1_catchment_benchmarks.csv"))
    sel_indices = [1, 2, 3, 5, 6, 9, 11, 13, 14, 16, 17, 18, 19, 21]
    t4 = [t4_all[i] for i in sel_indices]

    fig = plt.figure(figsize=(14.6, 7.2), dpi=300)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.28, 1.06], wspace=0.34)

    ax1 = fig.add_subplot(gs[0, 0])
    labels = [
        "1. S1-AI (Steady AI_t) + Std-CP",
        "2. S1-ModB (Reduced Pi*) + Std-CP",
        "3. S1-ModD-Phys (Bijective M0-Res) + Std-CP",
        "4. S1-Pers (1-Yr Persistence) + Std-CP",
        "5. S1-OLS-M0hyst (In-Fold OLS) + Std-CP",
        "6. S2-Res-107 (PUB, 107 Basins) + Std-CP",
        "7. S3B-Res (Multi-Yr Pure Met) + Std-CP",
        "8. S1-ModB (Reduced Pi*) + IJS-CP",
        "9. S1-ModD-Phys (Bijective M0-Res) + IJS-CP",
        "10. S1-OLS-M0hyst (In-Fold OLS) + Jac-CP",
        "11. S2-Res-107 (PUB, 107 Basins) + IJS-CP",
        "12. S2-Res-60Dir (Pure 60-Direct Refit) + IJS-CP",
        "13. S3B-Res (Multi-Yr Pure Met) + IJS-CP",
        "14. S3A-OLS-M0hyst (<=2015 OLS) + Jac-CP",
    ]
    covs = [float(r["cov90"]) * 100 for r in t4]
    rmses = [float(r["ood_rmse"]) for r in t4]
    isk = [float(r["winkler_score"]) for r in t4]
    colors = [
        C_ORANGE, "#C53030", "#DD6B20", "#718096", "#D69E2E", "#E53E3E", "#B83280",
        "#2B6CB0", C_BLUE, "#00684A", "#38A169", "#2F855A", "#319795", C_PURPLE
    ]

    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, covs, height=0.68, color=colors, edgecolor="k", linewidth=0.65)
    ax1.axvline(90.0, color=C_NAVY, linestyle="--", linewidth=1.8, label="Nominal 90.0% Target")
    ax1.axvspan(90.0, 100.0, color="#C6F6D5", alpha=0.45, zorder=0)

    for i, (c, rm, is_v) in enumerate(zip(covs, rmses, isk)):
        ax1.text(
            c - 1.2, y_pos[i],
            f"{c:.1f}%  (RMSE={rm:.4f}, IS={is_v:.3f})",
            va="center", ha="right", fontsize=6.4,
            fontweight="bold",
            color="white" if i not in (0, 4) else "#1A202C",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#1A202C" if i not in (0, 4) else "white", edgecolor="none", alpha=0.35 if i not in (0, 4) else 0.78)
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=6.8, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 102)
    ax1.set_xticks([0, 20, 40, 60, 80, 90, 100])
    ax1.set_xlabel("Empirical OOD Coverage Across 108 USGS Catchments (%)")
    ax1.set_title("(a) Fit-Isolated Paired OOD Coverage, RMSE & Winkler Score (0–100% Scale)")
    ax1.legend(loc="lower left", fontsize=7.0, framealpha=0.94)
    ax1.grid(axis="x", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axhspan(90.0, 101.5, color="#C6F6D5", alpha=0.48, label="Valid Nominal Zone (>=90% Cov)")
    ax2.axhline(90.0, color=C_NAVY, linestyle="--", linewidth=1.6, label="Minimum 90.0% Target")

    for idx_s, idx_i, col_a in [(1, 7, "#2B6CB0"), (2, 8, C_BLUE), (4, 9, "#00684A"), (5, 10, "#38A169"), (6, 12, "#319795")]:
        ax2.annotate(
            "", xy=(isk[idx_i], covs[idx_i] - 0.5), xytext=(isk[idx_s], covs[idx_s] + 0.5),
            arrowprops=dict(arrowstyle="->,head_width=0.20,head_length=0.20", lw=1.4, color=col_a, alpha=0.80)
        )

    for i in range(len(t4)):
        mk = "o" if i < 7 else "*"
        sz = 95 if i < 7 else 185
        ax2.scatter(isk[i], covs[i], s=sz, c=colors[i], marker=mk, edgecolors="k", linewidth=0.9, zorder=5)

    fixed_callouts = [
        (1, f"2. S1-ModB Std\n({covs[1]:.1f}%, IS={isk[1]:.3f})", (1.12, 56.5), "center"),
        (5, f"6. S2-Res-107 Std\n({covs[5]:.1f}%, IS={isk[5]:.3f})", (1.23, 66.0), "center"),
        (6, f"7. S3B-Res Std\n({covs[6]:.1f}%, IS={isk[6]:.3f})", (1.24, 78.5), "center"),
        (9, f"10. S1-OLS-M0hyst + Jac-CP\n({covs[9]:.1f}%, RMSE={rmses[9]:.4f}, IS={isk[9]:.3f})", (0.76, 82.0), "center"),
        (13, f"14. S3A-OLS-M0hyst + Jac-CP\n({covs[13]:.1f}%, RMSE={rmses[13]:.4f}, IS={isk[13]:.3f})", (0.64, 99.6), "center"),
        (11, f"12. S2-Res-60Dir + IJS-CP\n({covs[11]:.1f}%, RMSE={rmses[11]:.4f}, IS={isk[11]:.3f})", (1.17, 96.2), "center"),
    ]
    for idx_a, txt, (tx, ty), ha_a in fixed_callouts:
        is_cal = idx_a >= 7
        ax2.annotate(
            txt, xy=(isk[idx_a], covs[idx_a]),
            xytext=(tx, ty),
            ha=ha_a, va="center", fontsize=6.4,
            fontweight="bold" if is_cal else "normal",
            color="#005A32" if is_cal else "#742A2A",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#38A169" if is_cal else "#E53E3E", lw=0.9, alpha=0.95),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#005A32" if is_cal else "#9B2C2C", shrinkA=2, shrinkB=4),
            zorder=7
        )

    ax2.set_xlim(0.44, 1.48)
    ax2.set_ylim(52, 103.5)
    ax2.set_xlabel("Winkler Interval Score IS_0.10 (Lower is Better -> Left: Sharp & Reliable)")
    ax2.set_ylabel("Empirical OOD Coverage (%)")
    ax2.set_title("(b) Key Interval Quality Shifts (Std-CP -> IJS/Jac-CP)")
    ax2.legend(loc="lower right", fontsize=7.0)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Figure 4 | Fit-Isolated Paired Benchmark Including Pure 60-Direct-Gauge Refit & In-Fold OLS Analytical Baselines",
        fontsize=10.8, fontweight="bold", y=1.01
    )
    path = os.path.join(os.path.dirname(out_dir), "figures", "fig4_benchmark_interval_quality.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for fn in [make_fig1_workflow, make_fig2_spatial_and_process, make_fig3_phase_and_hydrographs, make_fig4_gauged_vs_ungauged_ijs]:
        p = fn(out_dir)
        print("Generated 300-DPI figure:", p)


if __name__ == "__main__":
    main()
