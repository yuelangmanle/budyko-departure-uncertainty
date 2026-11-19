#!/usr/bin/env python3
"""
Rigorous Hydrological Diagnostics across 1,202 GB + AUS Catchments
Explicitly resolves:
1. Variance partitioning: Between-basin static structure vs. Within-basin inter-annual memory.
2. Physical drivers of within-basin lag-1 memory (BFI, aridity, catchment area, climate regime).
3. Spurious-correlation-free wetting-drying asymmetry & antecedent carry-over tests:
   - Test A: Antecedent-only conditioning (uses ONLY t-1 climate/departure, zero shared P_t).
   - Test B: Wetting vs. drying transitions in mm/yr runoff anomaly & ER residual, benchmarked
             against a 200-iteration within-basin permutation null (isolating algebraic P_t effects).
4. Near-natural catchment sensitivity check (n = 869 near-natural vs. all 1,202).
"""
import csv
import json
import math
import os
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)


def fu_er(ai, omega):
    ai = np.maximum(np.asarray(ai, dtype=float), 1e-9)
    return 1.0 + ai - np.power(1.0 + np.power(ai, omega), 1.0 / omega)


def load_data():
    attrs = {}
    with open(os.path.join(DATA_DIR, "gb_aus_basin_attributes.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            attrs[r["basin_id"]] = r

    rows = []
    with open(os.path.join(DATA_DIR, "gb_aus_annual_water_years.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "basin_id": r["basin_id"],
                "region": r["region"],
                "wy": int(r["water_year"]),
                "P": float(r["P_mm"]),
                "PET": float(r["PET_mm"]),
                "Q": float(r["Q_mm"]),
                "AI": float(r["AI_t"]),
                "RR": float(r["RR_t"]),
                "ER": float(r["ER_t"]),
                "has_lag1": int(r["has_lag1"]),
                "P_lag1": float(r["P_lag1"]) if r["has_lag1"] == "1" else np.nan,
                "PET_lag1": float(r["PET_lag1"]) if r["has_lag1"] == "1" else np.nan,
                "Q_lag1": float(r["Q_lag1"]) if r["has_lag1"] == "1" else np.nan,
                "AI_lag1": float(r["AI_lag1"]) if r["has_lag1"] == "1" else np.nan,
                "RR_lag1": float(r["RR_lag1"]) if r["has_lag1"] == "1" else np.nan,
            })
    return attrs, rows


def analyze_subset(name, attrs, rows):
    # Fit global Fu omega on this subset's basin-years
    ai_all = np.array([r["AI"] for r in rows])
    er_all = np.array([r["ER"] for r in rows])
    res_w = minimize_scalar(
        lambda w: float(np.mean((fu_er(ai_all, w) - er_all) ** 2)),
        bounds=(1.1, 10.0),
        method="bounded",
    )
    omega_global = float(res_w.x)

    # Group by basin
    by_basin = defaultdict(list)
    for r in rows:
        by_basin[r["basin_id"]].append(r)

    # Compute global Budyko departure d_global = ER_t - fu(AI_t; omega_global)
    # and catchment-specific Fu residual r_basin = ER_t - fu(AI_t; omega_i)
    # and runoff anomaly in mm/yr: dQ_mm = Q_t - P_t * (1 - fu(AI_t; omega_i))
    d_global_all = []
    b_means = []
    within_d_all = []

    # Lag-1 paired arrays
    d_cur, d_prev = [], []
    wd_cur, wd_prev = [], []
    r_cur, r_prev = [], []
    dq_cur, dq_prev = [], []

    # Antecedent-only conditioning (ZERO shared P_t)
    # Condition on P_{t-1} < mean_P_i (antecedent dry) vs P_{t-1} > mean_P_i (antecedent wet)
    # and on d_{t-1} > 0 (antecedent deficit/high-ET) vs d_{t-1} < 0 (antecedent surplus/high-Q)
    dq_after_dry_P, dq_after_wet_P = [], []
    rr_res_after_dry_P, rr_res_after_wet_P = [], []
    dq_after_low_Q, dq_after_high_Q = [], []
    rr_res_after_low_Q, rr_res_after_high_Q = [], []

    # Transition arrays (for permutation null comparison)
    basin_series = []  # list of dicts per basin for permutation test

    # Catchment-level lag-1 stats
    basin_rho1 = []
    basin_rho1_sig = 0
    basin_phys = []  # (rho1, bfi, lt_AI, area_km2, forest_frac)

    for bid, blist in by_basin.items():
        blist.sort(key=lambda x: x["wy"])
        omega_i = float(attrs[bid]["fu_omega"])
        mean_P_i = float(attrs[bid]["mean_P_mm"])
        mean_AI_i = float(attrs[bid]["long_term_AI"])

        P_arr = np.array([x["P"] for x in blist])
        AI_arr = np.array([x["AI"] for x in blist])
        Q_arr = np.array([x["Q"] for x in blist])
        ER_arr = np.array([x["ER"] for x in blist])
        RR_arr = np.array([x["RR"] for x in blist])
        wys = np.array([x["wy"] for x in blist])

        dg = ER_arr - fu_er(AI_arr, omega_global)
        dg_mean = float(np.mean(dg))
        wd = dg - dg_mean

        er_hat_i = fu_er(AI_arr, omega_i)
        rr_hat_i = 1.0 - er_hat_i
        rr_res = RR_arr - rr_hat_i  # positive = more runoff than catchment's Fu curve
        dq_mm = Q_arr - P_arr * rr_hat_i  # in mm/yr, NOT divided by P_t!

        d_global_all.extend(dg.tolist())
        b_means.extend([dg_mean] * len(dg))
        within_d_all.extend(wd.tolist())

        basin_series.append({
            "P": P_arr, "AI": AI_arr, "rr_res": rr_res, "dq_mm": dq_mm, "wys": wys
        })

        # Consecutive lag-1 pairs
        b_rc, b_rp = [], []
        for k in range(1, len(blist)):
            if wys[k] - wys[k - 1] != 1:
                continue
            d_cur.append(dg[k]); d_prev.append(dg[k - 1])
            wd_cur.append(wd[k]); wd_prev.append(wd[k - 1])
            r_cur.append(rr_res[k]); r_prev.append(rr_res[k - 1])
            dq_cur.append(dq_mm[k]); dq_prev.append(dq_mm[k - 1])
            b_rc.append(rr_res[k]); b_rp.append(rr_res[k - 1])

            # Test A1: Antecedent P_{t-1} anomaly (zero P_t involvement)
            if P_arr[k - 1] < mean_P_i:
                dq_after_dry_P.append(dq_mm[k])
                rr_res_after_dry_P.append(rr_res[k])
            else:
                dq_after_wet_P.append(dq_mm[k])
                rr_res_after_wet_P.append(rr_res[k])

            # Test A2: Antecedent runoff residual r_{t-1} < 0 vs >= 0 (zero P_t involvement)
            if rr_res[k - 1] < 0:
                dq_after_low_Q.append(dq_mm[k])
                rr_res_after_low_Q.append(rr_res[k])
            else:
                dq_after_high_Q.append(dq_mm[k])
                rr_res_after_high_Q.append(rr_res[k])

        if len(b_rc) >= 15:
            r_val, p_val = stats.pearsonr(b_rp, b_rc)
            if math.isfinite(r_val):
                basin_rho1.append(float(r_val))
                if p_val < 0.05 and r_val > 0:
                    basin_rho1_sig += 1
                bfi = float(attrs[bid]["bfi"]) if attrs[bid]["bfi"] not in ("", "nan") else np.nan
                area = float(attrs[bid]["area_km2"]) if attrs[bid]["area_km2"] not in ("", "nan") else np.nan
                ff = float(attrs[bid]["forest_frac"]) if attrs[bid]["forest_frac"] not in ("", "nan") else np.nan
                basin_phys.append((float(r_val), bfi, mean_AI_i, area, ff))

    # Variance decomposition of global Budyko departure
    var_total = float(np.var(d_global_all, ddof=1))
    var_between = float(np.var(b_means, ddof=1))
    var_within = float(np.var(within_d_all, ddof=1))
    icc = var_between / var_total if var_total > 0 else np.nan

    # Correlations
    rho_pooled_dg = float(stats.pearsonr(d_prev, d_cur)[0])
    rho_within_wd = float(stats.pearsonr(wd_prev, wd_cur)[0])
    rho_within_rr_res = float(stats.pearsonr(r_prev, r_cur)[0])
    rho_within_dq_mm = float(stats.pearsonr(dq_prev, dq_cur)[0])

    # Physical drivers of catchment-level within-basin memory rho1
    phys_arr = np.array(basin_phys)
    def safe_spearman(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if np.sum(m) < 10:
            return {"rho": None, "p": None, "n": int(np.sum(m))}
        r, p = stats.spearmanr(x[m], y[m])
        return {"rho": round(float(r), 4), "p": float(f"{p:.2e}"), "n": int(np.sum(m))}

    drivers = {
        "BFI_vs_rho1": safe_spearman(phys_arr[:, 1], phys_arr[:, 0]),
        "AI_vs_rho1": safe_spearman(phys_arr[:, 2], phys_arr[:, 0]),
        "logArea_vs_rho1": safe_spearman(np.log10(np.maximum(phys_arr[:, 3], 1e-3)), phys_arr[:, 0]),
        "Forest_vs_rho1": safe_spearman(phys_arr[:, 4], phys_arr[:, 0]),
    }

    # Test A: Antecedent-only conditioning stats (zero shared P_t)
    t_dry_wet_dq, p_dry_wet_dq = stats.ttest_ind(dq_after_wet_P, dq_after_dry_P, equal_var=False)
    t_dry_wet_rr, p_dry_wet_rr = stats.ttest_ind(rr_res_after_wet_P, rr_res_after_dry_P, equal_var=False)
    t_q_dq, p_q_dq = stats.ttest_ind(dq_after_high_Q, dq_after_low_Q, equal_var=False)
    t_q_rr, p_q_rr = stats.ttest_ind(rr_res_after_high_Q, rr_res_after_low_Q, equal_var=False)

    # Test B: Wetting (P_t > P_{t-1}) vs Drying (P_t < P_{t-1}) transition asymmetry
    # Benchmarked against 200-iteration within-basin permutation null!
    def compute_transition_asymmetry(series_list, permute=False, rng=None):
        wet_rr, dry_rr = [], []
        wet_dq, dry_dq = [], []
        for s in series_list:
            n = len(s["P"])
            if permute:
                idx = rng.permutation(n)
                P = s["P"][idx]
                rr_res = s["rr_res"][idx]
                dq_mm = s["dq_mm"][idx]
                # keep same consecutive mask as original
                consec = (s["wys"][1:] - s["wys"][:-1]) == 1
            else:
                P = s["P"]
                rr_res = s["rr_res"]
                dq_mm = s["dq_mm"]
                consec = (s["wys"][1:] - s["wys"][:-1]) == 1
            dP = P[1:] - P[:-1]
            m_wet = consec & (dP > 0)
            m_dry = consec & (dP < 0)
            wet_rr.append(rr_res[1:][m_wet])
            dry_rr.append(rr_res[1:][m_dry])
            wet_dq.append(dq_mm[1:][m_wet])
            dry_dq.append(dq_mm[1:][m_dry])
        w_rr = np.concatenate(wet_rr); d_rr = np.concatenate(dry_rr)
        w_dq = np.concatenate(wet_dq); d_dq = np.concatenate(dry_dq)
        return (
            float(np.mean(d_rr) - np.mean(w_rr)),  # drying minus wetting in RR residual
            float(np.mean(d_dq) - np.mean(w_dq)),  # drying minus wetting in Q anomaly (mm/yr)
            float(np.mean(w_rr)), float(np.mean(d_rr)),
            float(np.mean(w_dq)), float(np.mean(d_dq)),
        )

    obs_diff_rr, obs_diff_dq, obs_w_rr, obs_d_rr, obs_w_dq, obs_d_dq = compute_transition_asymmetry(basin_series, False)
    rng = np.random.default_rng(42)
    null_rr, null_dq = [], []
    for _ in range(200):
        nr, nd, *_ = compute_transition_asymmetry(basin_series, True, rng)
        null_rr.append(nr)
        null_dq.append(nd)
    null_rr = np.array(null_rr)
    null_dq = np.array(null_dq)

    return {
        "subset": name,
        "n_basins": len(by_basin),
        "n_basin_years": len(rows),
        "n_lag1_pairs": len(r_cur),
        "global_fu_omega": round(omega_global, 4),
        "variance_decomposition": {
            "total_var": round(var_total, 5),
            "between_basin_share_ICC": round(icc, 4),
            "within_basin_share": round(var_within / var_total, 4),
        },
        "lag1_autocorrelation": {
            "pooled_global_departure_r": round(rho_pooled_dg, 4),
            "within_basin_demeaned_departure_r": round(rho_within_wd, 4),
            "within_basin_fu_rr_residual_r": round(rho_within_rr_res, 4),
            "within_basin_dq_mm_r": round(rho_within_dq_mm, 4),
            "basin_level_median_rho1": round(float(np.median(basin_rho1)), 4),
            "basin_level_IQR_rho1": [round(float(np.percentile(basin_rho1, 25)), 4), round(float(np.percentile(basin_rho1, 75)), 4)],
            "pct_basins_sig_positive_p05": round(100.0 * basin_rho1_sig / max(len(basin_rho1), 1), 1),
        },
        "physical_drivers_of_memory": drivers,
        "test_A_antecedent_only_zero_shared_Pt": {
            "after_wet_P_mean_rr_res": round(float(np.mean(rr_res_after_wet_P)), 5),
            "after_dry_P_mean_rr_res": round(float(np.mean(rr_res_after_dry_P)), 5),
            "wet_minus_dry_rr_res": round(float(np.mean(rr_res_after_wet_P) - np.mean(rr_res_after_dry_P)), 5),
            "p_val_rr_res": float(f"{p_dry_wet_rr:.2e}"),
            "after_wet_P_mean_dq_mm": round(float(np.mean(dq_after_wet_P)), 2),
            "after_dry_P_mean_dq_mm": round(float(np.mean(dq_after_dry_P)), 2),
            "wet_minus_dry_dq_mm": round(float(np.mean(dq_after_wet_P) - np.mean(dq_after_dry_P)), 2),
            "p_val_dq_mm": float(f"{p_dry_wet_dq:.2e}"),
            "after_high_Qres_minus_low_Qres_rr_res": round(float(np.mean(rr_res_after_high_Q) - np.mean(rr_res_after_low_Q)), 5),
            "after_high_Qres_minus_low_Qres_dq_mm": round(float(np.mean(dq_after_high_Q) - np.mean(dq_after_low_Q)), 2),
            "p_val_Qres_dq_mm": float(f"{p_q_dq:.2e}"),
        },
        "test_B_transition_vs_permutation_null": {
            "obs_drying_minus_wetting_rr_res": round(obs_diff_rr, 5),
            "null_mean_drying_minus_wetting_rr_res": round(float(np.mean(null_rr)), 5),
            "null_sd_rr_res": round(float(np.std(null_rr)), 5),
            "excess_physical_asymmetry_rr_res": round(obs_diff_rr - float(np.mean(null_rr)), 5),
            "z_score_rr_res": round((obs_diff_rr - float(np.mean(null_rr))) / max(float(np.std(null_rr)), 1e-9), 2),
            "obs_drying_minus_wetting_dq_mm": round(obs_diff_dq, 2),
            "null_mean_drying_minus_wetting_dq_mm": round(float(np.mean(null_dq)), 2),
            "excess_physical_asymmetry_dq_mm": round(obs_diff_dq - float(np.mean(null_dq)), 2),
            "z_score_dq_mm": round((obs_diff_dq - float(np.mean(null_dq))) / max(float(np.std(null_dq)), 1e-9), 2),
        },
    }


def main():
    attrs, rows = load_data()
    subsets = [
        ("ALL_1202", rows),
        ("GB_644", [r for r in rows if r["region"] == "GB"]),
        ("AUS_558", [r for r in rows if r["region"] == "AUS"]),
        ("NEAR_NATURAL_869", [r for r in rows if attrs[r["basin_id"]]["is_near_natural"] == "1"]),
    ]
    out = {}
    for name, sub in subsets:
        print(f"Analyzing {name} ({len(sub)} rows)...")
        out[name] = analyze_subset(name, attrs, sub)

    out_path = os.path.join(RES_DIR, "diagnostics_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n=== DIAGNOSTICS SUMMARY ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
