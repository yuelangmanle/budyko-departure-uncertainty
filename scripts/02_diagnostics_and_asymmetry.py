#!/usr/bin/env python3
"""
Rigorous Hydrological & Mathematical Diagnostics across 1,202 GB + AUS Catchments:
1. Orthogonal variance partitioning (Between-basin ICC vs. Within-basin variability).
2. Closed-form analytical identity for zero-memory transition asymmetry:
       Delta_null = E[dQ_2 | P_2 < P_1] - E[dQ_2 | P_2 > P_1] = -4 * Cov(dQ, F_P(P))
   verified directly against 200-iteration within-basin Monte Carlo permutation nulls.
3. Antecedent-only conditioning (zero shared P_t) across ALL, GB, AUS, and Near-Natural subsets.
4. Resolution of the BFI-vs-rho1 paradox via within-region partial Spearman correlations
   controlling for aridity (AI), plus linearized storage relaxation timescale tau = -1/ln(rho1).
5. Sensitivity check for Southern Hemisphere water-year definition in Australia (Apr-Mar vs. Jul-Jun).
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
AUS_RAW = os.environ.get("CAMELS_AUS_DIR", os.path.join(ROOT, "raw_data", "camels_aus_v2.03"))
os.makedirs(RES_DIR, exist_ok=True)


def fu_er(ai, omega):
    ai = np.maximum(np.asarray(ai, dtype=float), 1e-9)
    return 1.0 + ai - np.power(1.0 + np.power(ai, omega), 1.0 / omega)


def partial_spearman(x, y, z):
    """Partial Spearman rank correlation between x and y controlling for z."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.sum(m) < 15:
        return {"rho": None, "p": None, "n": int(np.sum(m))}
    rx = stats.rankdata(x[m])
    ry = stats.rankdata(y[m])
    rz = stats.rankdata(z[m])
    # Regress rx and ry on rz
    bx = np.polyfit(rz, rx, 1)
    by = np.polyfit(rz, ry, 1)
    ex = rx - np.polyval(bx, rz)
    ey = ry - np.polyval(by, rz)
    r, p = stats.pearsonr(ex, ey)
    return {"rho": round(float(r), 4), "p": float(f"{p:.2e}"), "n": int(np.sum(m))}


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
            })
    return attrs, rows


def analyze_subset(name, attrs, rows):
    ai_all = np.array([r["AI"] for r in rows])
    er_all = np.array([r["ER"] for r in rows])
    omega_global = float(minimize_scalar(
        lambda w: float(np.mean((fu_er(ai_all, w) - er_all) ** 2)),
        bounds=(1.1, 10.0), method="bounded"
    ).x)

    by_basin = defaultdict(list)
    for r in rows:
        by_basin[r["basin_id"]].append(r)

    d_global_all, b_means, within_d_all = [], [], []
    d_cur, d_prev, wd_cur, wd_prev, r_cur, r_prev, dq_cur, dq_prev = [], [], [], [], [], [], [], []
    dq_after_dry_P, dq_after_wet_P, rr_after_dry_P, rr_after_wet_P = [], [], [], []
    basin_series = []
    basin_rho1 = []
    basin_rho1_sig = 0
    basin_phys = []  # (rho1, bfi, lt_AI, area, forest, clay, sand, p_seas, q_cv)
    per_basin_euler_dq = []
    per_basin_euler_rr = []
    per_basin_weights = []
    basin_rho_map = {}

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

        rr_hat_i = 1.0 - fu_er(AI_arr, omega_i)
        rr_res = RR_arr - rr_hat_i
        dq_mm = Q_arr - P_arr * rr_hat_i

        d_global_all.extend(dg.tolist())
        b_means.extend([dg_mean] * len(dg))
        within_d_all.extend(wd.tolist())

        # Exact closed-form Euler identity per basin (finite-sample exact expectation under i.i.d. permutation):
        # For distinct ranks r_k in {1..n}, E[y_2 | P_2 < P_1] - E[y_2 | P_2 > P_1]
        # = (2 / (n*(n-1))) * sum_k y_k * ((n - r_k) - (r_k - 1)) = -4 * Cov_pop(y, F_emp(P)) * n/(n-1)
        n_b = len(P_arr)
        consec_count = int(np.sum((wys[1:] - wys[:-1]) == 1))
        if n_b >= 5 and consec_count > 0:
            ranks = stats.rankdata(P_arr, method="average")
            F_emp = (ranks - 0.5) / n_b
            # Exact finite-sample permutation expectation: -4 * mean((y - mean(y)) * (F_emp - 0.5)) * n/(n-1)
            euler_dq = float(-4.0 * np.mean((dq_mm - np.mean(dq_mm)) * (F_emp - 0.5)) * (n_b / (n_b - 1.0)))
            euler_rr = float(-4.0 * np.mean((rr_res - np.mean(rr_res)) * (F_emp - 0.5)) * (n_b / (n_b - 1.0)))
            per_basin_euler_dq.append(euler_dq)
            per_basin_euler_rr.append(euler_rr)
            per_basin_weights.append(consec_count)

        basin_series.append({"P": P_arr, "rr_res": rr_res, "dq_mm": dq_mm, "wys": wys})

        b_rc, b_rp = [], []
        for k in range(1, n_b):
            if wys[k] - wys[k - 1] != 1:
                continue
            d_cur.append(dg[k]); d_prev.append(dg[k - 1])
            wd_cur.append(wd[k]); wd_prev.append(wd[k - 1])
            r_cur.append(rr_res[k]); r_prev.append(rr_res[k - 1])
            dq_cur.append(dq_mm[k]); dq_prev.append(dq_mm[k - 1])
            b_rc.append(rr_res[k]); b_rp.append(rr_res[k - 1])

            if P_arr[k - 1] < mean_P_i:
                dq_after_dry_P.append(dq_mm[k]); rr_after_dry_P.append(rr_res[k])
            else:
                dq_after_wet_P.append(dq_mm[k]); rr_after_wet_P.append(rr_res[k])

        if len(b_rc) >= 15:
            r_val, p_val = stats.pearsonr(b_rp, b_rc)
            if math.isfinite(r_val):
                basin_rho1.append(float(r_val))
                basin_rho_map[bid] = float(r_val)
                if p_val < 0.05 and r_val > 0:
                    basin_rho1_sig += 1
                def ga(k):
                    v = attrs[bid].get(k, "")
                    return float(v) if v not in ("", "nan", "None") else np.nan
                q_cv = float(np.std(Q_arr) / max(np.mean(Q_arr), 1e-3))
                basin_phys.append((
                    float(r_val), ga("bfi"), mean_AI_i, ga("area_km2"),
                    ga("forest_frac"), ga("clay_pct"), ga("sand_pct"), ga("p_seasonality"), q_cv
                ))

    var_total = float(np.var(d_global_all, ddof=1))
    var_between = float(np.var(b_means, ddof=1))
    var_within = float(np.var(within_d_all, ddof=1))

    # Weighted theoretical Euler null across basins
    w_arr = np.array(per_basin_weights, dtype=float)
    w_arr /= np.sum(w_arr)
    euler_null_dq = float(np.sum(w_arr * np.array(per_basin_euler_dq)))
    euler_null_rr = float(np.sum(w_arr * np.array(per_basin_euler_rr)))

    # 200-iteration Monte Carlo permutation null
    def compute_trans(series_list, permute=False, rng=None):
        w_rr, d_rr, w_dq, d_dq = [], [], [], []
        for s in series_list:
            n = len(s["P"])
            if permute:
                idx = rng.permutation(n)
                P, rr_res, dq_mm = s["P"][idx], s["rr_res"][idx], s["dq_mm"][idx]
            else:
                P, rr_res, dq_mm = s["P"], s["rr_res"], s["dq_mm"]
            consec = (s["wys"][1:] - s["wys"][:-1]) == 1
            dP = P[1:] - P[:-1]
            m_w = consec & (dP > 0)
            m_d = consec & (dP < 0)
            w_rr.append(rr_res[1:][m_w]); d_rr.append(rr_res[1:][m_d])
            w_dq.append(dq_mm[1:][m_w]); d_dq.append(dq_mm[1:][m_d])
        return (
            float(np.mean(np.concatenate(d_rr)) - np.mean(np.concatenate(w_rr))),
            float(np.mean(np.concatenate(d_dq)) - np.mean(np.concatenate(w_dq))),
        )

    obs_rr, obs_dq = compute_trans(basin_series, False)
    rng = np.random.default_rng(42)
    mc_rr, mc_dq = [], []
    for _ in range(200):
        nr, nd = compute_trans(basin_series, True, rng)
        mc_rr.append(nr); mc_dq.append(nd)
    mc_rr, mc_dq = np.array(mc_rr), np.array(mc_dq)

    # Physical drivers & partial correlations controlling for AI (resolving BFI paradox)
    pa = np.array(basin_phys)
    def sp(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        r, p = stats.spearmanr(x[m], y[m])
        return {"rho": round(float(r), 4), "p": float(f"{p:.2e}"), "n": int(np.sum(m))}

    drivers = {
        "raw_BFI_vs_rho1": sp(pa[:, 1], pa[:, 0]),
        "partial_BFI_vs_rho1_given_AI": partial_spearman(pa[:, 1], pa[:, 0], pa[:, 2]),
        "partial_BFI_vs_rho1_given_Qcv": partial_spearman(pa[:, 1], pa[:, 0], pa[:, 8]),
        "raw_Qcv_vs_rho1": sp(pa[:, 8], pa[:, 0]),
        "raw_AI_vs_rho1": sp(pa[:, 2], pa[:, 0]),
        "raw_Forest_vs_rho1": sp(pa[:, 4], pa[:, 0]),
        "partial_Forest_vs_rho1_given_AI": partial_spearman(pa[:, 4], pa[:, 0], pa[:, 2]),
        "raw_Clay_vs_rho1": sp(pa[:, 5], pa[:, 0]),
    }

    # Linearized storage relaxation time tau = -1 / ln(rho1) for basins with significant positive memory
    pos_rhos = [r for r in basin_rho1 if r > 0.15]
    taus = [-1.0 / math.log(r) for r in pos_rhos]
    pooled_r = float(stats.pearsonr(r_prev, r_cur)[0])
    pooled_tau = round(-1.0 / math.log(max(pooled_r, 1e-6)), 3)

    t_dq, p_dq = stats.ttest_ind(dq_after_wet_P, dq_after_dry_P, equal_var=False)
    t_rr, p_rr = stats.ttest_ind(rr_after_wet_P, rr_after_dry_P, equal_var=False)

    return {
        "subset": name,
        "n_basins": len(by_basin),
        "n_basin_years": len(rows),
        "n_lag1_pairs": len(r_cur),
        "global_fu_omega": round(omega_global, 4),
        "variance_decomposition": {
            "between_basin_share_ICC": round(var_between / var_total, 4),
            "within_basin_share": round(var_within / var_total, 4),
        },
        "lag1_memory": {
            "pooled_global_departure_r": round(float(stats.pearsonr(d_prev, d_cur)[0]), 4),
            "within_basin_fu_rr_residual_r": round(pooled_r, 4),
            "within_basin_dq_mm_r": round(float(stats.pearsonr(dq_prev, dq_cur)[0]), 4),
            "pooled_relaxation_tau_years": pooled_tau,
            "basin_median_rho1": round(float(np.median(basin_rho1)), 4),
            "basin_IQR_rho1": [round(float(np.percentile(basin_rho1, 25)), 4), round(float(np.percentile(basin_rho1, 75)), 4)],
            "pct_basins_sig_p05": round(100.0 * basin_rho1_sig / max(len(basin_rho1), 1), 1),
            "median_tau_years_mem_basins": round(float(np.median(taus)), 3) if taus else None,
        },
        "physical_drivers_and_bfi_resolution": drivers,
        "test_A_antecedent_only_zero_shared_Pt": {
            "wet_minus_dry_rr_res": round(float(np.mean(rr_after_wet_P) - np.mean(rr_after_dry_P)), 5),
            "p_val_rr_res": float(f"{p_rr:.2e}"),
            "wet_minus_dry_dq_mm": round(float(np.mean(dq_after_wet_P) - np.mean(dq_after_dry_P)), 2),
            "p_val_dq_mm": float(f"{p_dq:.2e}"),
        },
        "test_B_euler_theorem_vs_mc_null": {
            "obs_drying_minus_wetting_dq_mm": round(obs_dq, 2),
            "mc_null_mean_dq_mm": round(float(np.mean(mc_dq)), 2),
            "euler_closed_form_null_dq_mm": round(euler_null_dq, 2),
            "euler_vs_mc_abs_error_mm": round(abs(euler_null_dq - float(np.mean(mc_dq))), 2),
            "excess_physical_asymmetry_dq_mm": round(obs_dq - euler_null_dq, 2),
            "z_score_dq_mm": round((obs_dq - float(np.mean(mc_dq))) / max(float(np.std(mc_dq)), 1e-9), 2),
            "obs_drying_minus_wetting_rr_res": round(obs_rr, 5),
            "mc_null_mean_rr_res": round(float(np.mean(mc_rr)), 5),
            "euler_closed_form_null_rr_res": round(euler_null_rr, 5),
            "excess_physical_asymmetry_rr_res": round(obs_rr - euler_null_rr, 5),
        },
    }, basin_rho_map


def check_aus_jul_jun_sensitivity(attrs):
    """Sensitivity check: re-aggregate AUS using Jul 1 - Jun 30 water year instead of Apr 1 - Mar 31."""
    print("Running AUS Jul-Jun water-year sensitivity check...")
    sf_path = os.path.join(AUS_RAW, "streamflow_mmd.csv")
    p_path = os.path.join(AUS_RAW, "hydrometeorology", "precipitation_AGCD.csv")
    pet_path = os.path.join(AUS_RAW, "hydrometeorology", "et_morton_wet_SILO.csv")

    def read_wide(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            rd = csv.reader(f)
            hdr = next(rd)
            sids = [c.strip() for c in hdr[3:]]
            data = {}
            for row in rd:
                y = int(row[0])
                if y < 1950:
                    continue
                vals = np.array([float(x) if x and x != "NaN" else np.nan for x in row[3:]], dtype=np.float32)
                vals[vals < 0] = np.nan
                data[(y, int(row[1]), int(row[2]))] = vals
        return sids, data

    sids, qd = read_wide(sf_path)
    _, pd = read_wide(p_path)
    _, petd = read_wide(pet_path)
    common = sorted(set(qd) & set(pd) & set(petd))
    nb = len(sids)

    wy_P = defaultdict(lambda: np.zeros(nb))
    wy_PET = defaultdict(lambda: np.zeros(nb))
    wy_Q = defaultdict(lambda: np.zeros(nb))
    wy_nv = defaultdict(lambda: np.zeros(nb, dtype=np.int32))
    wy_nt = defaultdict(int)

    for (y, m, d) in common:
        wy = y + 1 if m >= 7 else y  # Jul-Jun water year
        wy_nt[wy] += 1
        qv, pv, petv = qd[(y, m, d)], pd[(y, m, d)], petd[(y, m, d)]
        ok = np.isfinite(qv) & np.isfinite(pv) & np.isfinite(petv)
        wy_P[wy] += np.where(ok, pv, 0.0)
        wy_PET[wy] += np.where(ok, petv, 0.0)
        wy_Q[wy] += np.where(ok, qv, 0.0)
        wy_nv[wy] += ok.astype(np.int32)

    kept_aus_ids = {k.replace("AUS_", "") for k in attrs if k.startswith("AUS_")}
    rows = []
    for idx, sid in enumerate(sids):
        if sid not in kept_aus_ids:
            continue
        bid = f"AUS_{sid}"
        wmap = {}
        for wy, nt in wy_nt.items():
            if nt < 365:
                continue
            nv = int(wy_nv[wy][idx])
            if nv >= 355:
                sc = nt / nv
                P, PET, Q = float(wy_P[wy][idx] * sc), float(wy_PET[wy][idx] * sc), float(wy_Q[wy][idx] * sc)
                if P > 10 and PET > 10 and Q >= 0 and (Q / P) <= 1.2:
                    wmap[wy] = (P, PET, Q)
        for wy in sorted(wmap):
            P, PET, Q = wmap[wy]
            rows.append({
                "basin_id": bid, "region": "AUS", "wy": wy,
                "P": P, "PET": PET, "Q": Q, "AI": PET / P, "RR": Q / P, "ER": 1.0 - Q / P,
                "has_lag1": int((wy - 1) in wmap),
            })
    res, _ = analyze_subset("AUS_JulJun_Sensitivity", attrs, rows)
    return {
        "n_basin_years": res["n_basin_years"],
        "between_basin_ICC": res["variance_decomposition"]["between_basin_share_ICC"],
        "within_basin_fu_rr_r": res["lag1_memory"]["within_basin_fu_rr_residual_r"],
        "basin_median_rho1": res["lag1_memory"]["basin_median_rho1"],
        "test_A_wet_minus_dry_dq_mm": res["test_A_antecedent_only_zero_shared_Pt"]["wet_minus_dry_dq_mm"],
        "test_B_euler_null_dq_mm": res["test_B_euler_theorem_vs_mc_null"]["euler_closed_form_null_dq_mm"],
        "test_B_excess_physical_dq_mm": res["test_B_euler_theorem_vs_mc_null"]["excess_physical_asymmetry_dq_mm"],
    }


def main():
    attrs, rows = load_data()
    out = {}
    rho_map_all = {}
    for name, sub in [
        ("ALL_1202", rows),
        ("GB_644", [r for r in rows if r["region"] == "GB"]),
        ("AUS_558", [r for r in rows if r["region"] == "AUS"]),
        ("NEAR_NATURAL_869", [r for r in rows if attrs[r["basin_id"]]["is_near_natural"] == "1"]),
    ]:
        print(f"Analyzing {name}...")
        res, rmap = analyze_subset(name, attrs, sub)
        out[name] = res
        if name == "ALL_1202":
            rho_map_all = rmap

    out["AUS_JulJun_WaterYear_Sensitivity"] = check_aus_jul_jun_sensitivity(attrs)

    with open(os.path.join(RES_DIR, "diagnostics_upgraded.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RES_DIR, "basin_rho1_map.json"), "w", encoding="utf-8") as f:
        json.dump(rho_map_all, f)

    print("\n=== UPGRADED DIAGNOSTICS & EULER THEOREM VERIFICATION ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
