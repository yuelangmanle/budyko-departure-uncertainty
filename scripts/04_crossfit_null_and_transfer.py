#!/usr/bin/env python3
"""
Rigorous statistical & baseline verification:
  1. Single-source Table 1 & RR boundary integrity audit (basin counts vs water-year counts)
  2. Real lag-1 precipitation autocorrelation, strictly fold-isolated 5-fold contiguous
     time-block cross-fitting (fitting BOTH catchment Fu omega_i and memoryless climate
     regression g_i strictly inside training folds, for both quadratic and P*PET interaction
     specifications), 500-iteration catchment-clustered bootstrap 95% CIs, and Kendall
     finite-sample bias-corrected lag-1 residual autocorrelation
  3. Fair S3 (GB -> AUS) exact conformal evaluation on ONE unified deterministic split for
     M0 Source Physical Fu, M3 Direct HGBR, M5 PhysRes, and strictly source-calibrated M5-Damped
"""
import csv
import json
import math
import os
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
SEED = 42


def fu_er(ai, omega):
    ai = np.maximum(np.asarray(ai, dtype=float), 1e-9)
    return 1.0 + ai - np.power(1.0 + np.power(ai, omega), 1.0 / omega)


def fu_rr(ai, omega):
    return np.clip(1.0 - fu_er(ai, omega), 0.0, 1.0)


def fit_omega(ai, rr):
    res = minimize_scalar(lambda w: float(np.mean((fu_rr(ai, w) - rr) ** 2)), bounds=(1.1, 10.0), method="bounded")
    return float(res.x)


def conformal_quantile_exact(scores, alpha=0.10):
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k <= 0:
        return float(s[0])
    if k > n:
        return float(np.inf)
    return float(s[k - 1])


def winkler_score(y, lo, hi, alpha=0.10):
    w = hi - lo
    pen = (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    return float(np.mean(w + pen))


def eval_interval(y, lo, hi, alpha=0.10):
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.maximum(np.clip(hi, 0.0, 1.0), lo)
    return {
        "Coverage": round(float(np.mean((y >= lo) & (y <= hi))), 4),
        "Mean_Width": round(float(np.mean(hi - lo)), 4),
        "Winkler_Score": round(winkler_score(y, lo, hi, alpha), 4),
    }


def main():
    attrs = {r["basin_id"]: r for r in csv.DictReader(open(os.path.join(DATA_DIR, "gb_aus_basin_attributes.csv"), encoding="utf-8"))}
    by_b = defaultdict(list)
    n_gt1, n_lt1e3, n_all = 0, 0, 0
    for r in csv.DictReader(open(os.path.join(DATA_DIR, "gb_aus_annual_water_years.csv"), encoding="utf-8")):
        by_b[r["basin_id"]].append(r)
        n_all += 1
        rr_v = float(r["RR_t"])
        if rr_v > 1.0:
            n_gt1 += 1
        if rr_v < 1e-3:
            n_lt1e3 += 1

    # 1. Single-source Table 1 & RR Boundary Audit
    audit_t1 = {
        "RR_Boundary_Counts": {
            "total_water_years": n_all,
            "RR_gt_1_count": n_gt1,
            "RR_gt_1_pct": round(100.0 * n_gt1 / n_all, 2),
            "RR_lt_1e3_count": n_lt1e3,
            "RR_lt_1e3_pct": round(100.0 * n_lt1e3 / n_all, 2),
        }
    }
    for reg_key, filt in [("GB", lambda b: b.startswith("GB_")), ("AUS", lambda b: b.startswith("AUS_")), ("ALL", lambda b: True)]:
        bids = [b for b in by_b if filt(b)]
        lt_ais = np.array([float(attrs[b]["long_term_AI"]) for b in bids])
        wy_ais = np.array([float(r["AI_t"]) for b in bids for r in by_b[b]])
        n_basins_gt1 = int(np.sum(lt_ais > 1.0))
        n_wy_gt1 = int(np.sum(wy_ais > 1.0))
        med_lt_ai = float(np.median(lt_ais))
        if med_lt_ai > 1.0:
            assert n_basins_gt1 >= len(bids) / 2.0
        audit_t1[reg_key] = {
            "n_basins": len(bids),
            "n_basins_AI_gt_1": n_basins_gt1,
            "pct_basins_AI_gt_1": round(100.0 * n_basins_gt1 / len(bids), 1),
            "long_term_AI_median": round(med_lt_ai, 3),
            "n_water_years": len(wy_ais),
            "n_water_years_AI_gt_1": n_wy_gt1,
            "pct_water_years_AI_gt_1": round(100.0 * n_wy_gt1 / len(wy_ais), 1),
        }

    # 2. Strictly Fold-Isolated 5-Fold Time-Block Cross-Fitting (Fu omega_i + Quadratic & Interaction Climate Nulls)
    basin_stats = {}
    for bid, rows in by_b.items():
        rows.sort(key=lambda x: int(x["water_year"]))
        P = np.array([float(x["P_mm"]) for x in rows])
        PET = np.array([float(x["PET_mm"]) for x in rows])
        Q = np.array([float(x["Q_mm"]) for x in rows])
        AI = PET / np.maximum(P, 1e-6)
        RR = Q / np.maximum(P, 1e-6)
        wy = np.array([int(x["water_year"]) for x in rows])
        n_yr = len(rows)

        # Full-record Fu residual (for Kendall bias-corrected rho1 diagnostic)
        w_full = fit_omega(np.array([np.sum(PET) / np.sum(P)]), np.array([np.sum(Q) / np.sum(P)]))
        r_full = RR - fu_rr(AI, w_full)
        dq_full = Q - P * fu_rr(AI, w_full)

        # Strictly fold-isolated 5-fold contiguous time-block cross-fitting:
        # BOTH Fu omega_i AND g_i(P_t, P_t^2, PET_t[, P_t*PET_t]) are fitted ONLY on tr_mask!
        dq_iso = np.zeros(n_yr)
        dq_quad_null = np.zeros(n_yr)
        dq_quad_resid = np.zeros(n_yr)
        dq_int_null = np.zeros(n_yr)
        dq_int_resid = np.zeros(n_yr)
        folds = np.array_split(np.arange(n_yr), 5)
        for f_idx in folds:
            tr_mask = np.ones(n_yr, dtype=bool)
            tr_mask[f_idx] = False
            w_f = fit_omega(np.array([np.sum(PET[tr_mask]) / np.sum(P[tr_mask])]),
                            np.array([min(max(np.sum(Q[tr_mask]) / np.sum(P[tr_mask]), 0.002), 0.995)]))
            dq_tr = Q[tr_mask] - P[tr_mask] * fu_rr(AI[tr_mask], w_f)
            dq_te = Q[f_idx] - P[f_idx] * fu_rr(AI[f_idx], w_f)
            dq_iso[f_idx] = dq_te

            mu_p, sd_p = float(np.mean(P[tr_mask])), max(float(np.std(P[tr_mask])), 1e-6)
            mu_e, sd_e = float(np.mean(PET[tr_mask])), max(float(np.std(PET[tr_mask])), 1e-6)

            def X_quad(p_, e_):
                pz = (p_ - mu_p) / sd_p; ez = (e_ - mu_e) / sd_e
                return np.column_stack([np.ones_like(pz), pz, pz ** 2, ez])

            def X_inter(p_, e_):
                pz = (p_ - mu_p) / sd_p; ez = (e_ - mu_e) / sd_e
                return np.column_stack([np.ones_like(pz), pz, pz ** 2, ez, ez ** 2, pz * ez])

            bq, *_ = np.linalg.lstsq(X_quad(P[tr_mask], PET[tr_mask]), dq_tr, rcond=None)
            pq = X_quad(P[f_idx], PET[f_idx]) @ bq
            dq_quad_null[f_idx] = pq
            dq_quad_resid[f_idx] = dq_te - pq

            bi, *_ = np.linalg.lstsq(X_inter(P[tr_mask], PET[tr_mask]), dq_tr, rcond=None)
            pi_ = X_inter(P[f_idx], PET[f_idx]) @ bi
            dq_int_null[f_idx] = pi_
            dq_int_resid[f_idx] = dq_te - pi_

        p_mean = float(np.mean(P))
        p_t, p_tm1, r_t, r_tm1 = [], [], [], []
        dq_t, dq_qn_t, dq_qr_t, dq_in_t, dq_ir_t = [], [], [], [], []
        for k in range(1, n_yr):
            if wy[k] == wy[k - 1] + 1:
                p_t.append(P[k]); p_tm1.append(P[k - 1])
                r_t.append(r_full[k]); r_tm1.append(r_full[k - 1])
                dq_t.append(dq_iso[k])
                dq_qn_t.append(dq_quad_null[k]); dq_qr_t.append(dq_quad_resid[k])
                dq_in_t.append(dq_int_null[k]); dq_ir_t.append(dq_int_resid[k])
        p_t = np.array(p_t); p_tm1 = np.array(p_tm1)
        dq_t = np.array(dq_t); dq_qn_t = np.array(dq_qn_t); dq_qr_t = np.array(dq_qr_t)
        dq_in_t = np.array(dq_in_t); dq_ir_t = np.array(dq_ir_t)

        rho1_p = float(np.corrcoef(p_t, p_tm1)[0, 1]) if len(p_t) >= 15 else np.nan
        rho1_r = float(np.corrcoef(r_t, r_tm1)[0, 1]) if len(r_t) >= 15 else np.nan
        # Marriott-Pope / Kendall finite-sample bias correction: rho_corr = (N*rho + 1) / (N - 3)
        rho1_r_corr = float((n_yr * rho1_r + 1.0) / max(n_yr - 3, 1)) if np.isfinite(rho1_r) else np.nan

        wet_prev = p_tm1 > p_mean
        dry_prev = p_tm1 < p_mean
        drying = p_t < p_tm1
        wetting = p_t > p_tm1

        basin_stats[bid] = {
            "region": rows[0]["region"],
            "is_nn": attrs[bid]["is_near_natural"] == "1",
            "rho1_p": rho1_p, "rho1_r": rho1_r, "rho1_r_corr": rho1_r_corr,
            "ta_ws": float(np.sum(dq_t[wet_prev])), "ta_wn": int(np.sum(wet_prev)),
            "ta_ds": float(np.sum(dq_t[dry_prev])), "ta_dn": int(np.sum(dry_prev)),
            "ta_qn_ws": float(np.sum(dq_qn_t[wet_prev])), "ta_qn_ds": float(np.sum(dq_qn_t[dry_prev])),
            "ta_qr_ws": float(np.sum(dq_qr_t[wet_prev])), "ta_qr_ds": float(np.sum(dq_qr_t[dry_prev])),
            "ta_in_ws": float(np.sum(dq_in_t[wet_prev])), "ta_in_ds": float(np.sum(dq_in_t[dry_prev])),
            "ta_ir_ws": float(np.sum(dq_ir_t[wet_prev])), "ta_ir_ds": float(np.sum(dq_ir_t[dry_prev])),
            "tb_ds": float(np.sum(dq_t[drying])), "tb_dn": int(np.sum(drying)),
            "tb_ws": float(np.sum(dq_t[wetting])), "tb_wn": int(np.sum(wetting)),
            "tb_qn_ds": float(np.sum(dq_qn_t[drying])), "tb_qn_ws": float(np.sum(dq_qn_t[wetting])),
            "tb_qr_ds": float(np.sum(dq_qr_t[drying])), "tb_qr_ws": float(np.sum(dq_qr_t[wetting])),
            "tb_ir_ds": float(np.sum(dq_ir_t[drying])), "tb_ir_ws": float(np.sum(dq_ir_t[wetting])),
        }

    rng = np.random.default_rng(2026)
    n_boot = 500
    clim_diag = {}
    for name, pred in [
        ("ALL_1202", lambda s: True),
        ("GB_644", lambda s: s["region"] == "GB"),
        ("AUS_558", lambda s: s["region"] == "AUS"),
        ("NEAR_NATURAL_869", lambda s: s["is_nn"]),
    ]:
        sub = [s for s in basin_stats.values() if pred(s)]
        m = len(sub)
        r1p = np.array([s["rho1_p"] for s in sub if np.isfinite(s["rho1_p"])])
        r1r = np.array([s["rho1_r"] for s in sub if np.isfinite(s["rho1_r"])])
        r1rc = np.array([s["rho1_r_corr"] for s in sub if np.isfinite(s["rho1_r_corr"])])

        arr = lambda k: np.array([s[k] for s in sub])
        ta_ws, ta_wn, ta_ds, ta_dn = arr("ta_ws"), arr("ta_wn"), arr("ta_ds"), arr("ta_dn")
        ta_qn_ws, ta_qn_ds = arr("ta_qn_ws"), arr("ta_qn_ds")
        ta_qr_ws, ta_qr_ds = arr("ta_qr_ws"), arr("ta_qr_ds")
        ta_in_ws, ta_in_ds = arr("ta_in_ws"), arr("ta_in_ds")
        ta_ir_ws, ta_ir_ds = arr("ta_ir_ws"), arr("ta_ir_ds")
        tb_ds, tb_dn, tb_ws, tb_wn = arr("tb_ds"), arr("tb_dn"), arr("tb_ws"), arr("tb_wn")
        tb_qn_ds, tb_qn_ws = arr("tb_qn_ds"), arr("tb_qn_ws")
        tb_qr_ds, tb_qr_ws = arr("tb_qr_ds"), arr("tb_qr_ws")
        tb_ir_ds, tb_ir_ws = arr("tb_ir_ds"), arr("tb_ir_ws")

        dr = lambda s1, n1, s2, n2: float(np.sum(s1) / np.sum(n1) - np.sum(s2) / np.sum(n2))
        idx = rng.integers(0, m, size=(n_boot, m))
        b_qr = (ta_qr_ws[idx].sum(1) / ta_wn[idx].sum(1)) - (ta_qr_ds[idx].sum(1) / ta_dn[idx].sum(1))
        b_ir = (ta_ir_ws[idx].sum(1) / ta_wn[idx].sum(1)) - (ta_ir_ds[idx].sum(1) / ta_dn[idx].sum(1))
        b_tb = (tb_qr_ds[idx].sum(1) / tb_dn[idx].sum(1)) - (tb_qr_ws[idx].sum(1) / tb_wn[idx].sum(1))
        b_tb_i = (tb_ir_ds[idx].sum(1) / tb_dn[idx].sum(1)) - (tb_ir_ws[idx].sum(1) / tb_wn[idx].sum(1))

        clim_diag[name] = {
            "n_basins": m,
            "precip_lag1_median": round(float(np.median(r1p)), 4),
            "rho1_raw_median": round(float(np.median(r1r)), 4),
            "rho1_kendall_corrected_median": round(float(np.median(r1rc)), 4),
            "TestA_obs_mm": round(dr(ta_ws, ta_wn, ta_ds, ta_dn), 2),
            "TestA_climate_preserving_null_mm": round(dr(ta_qn_ws, ta_wn, ta_qn_ds, ta_dn), 2),
            "TestA_climate_residualized_mm": round(dr(ta_qr_ws, ta_wn, ta_qr_ds, ta_dn), 2),
            "TestA_climate_residualized_boot95CI": [round(float(np.percentile(b_qr, 2.5)), 2), round(float(np.percentile(b_qr, 97.5)), 2)],
            "TestA_interaction_null_mm": round(dr(ta_in_ws, ta_wn, ta_in_ds, ta_dn), 2),
            "TestA_interaction_residualized_mm": round(dr(ta_ir_ws, ta_wn, ta_ir_ds, ta_dn), 2),
            "TestA_interaction_boot95CI": [round(float(np.percentile(b_ir, 2.5)), 2), round(float(np.percentile(b_ir, 97.5)), 2)],
            "TestB_obs_mm": round(dr(tb_ds, tb_dn, tb_ws, tb_wn), 2),
            "TestB_climate_preserving_null_mm": round(dr(tb_qn_ds, tb_dn, tb_qn_ws, tb_wn), 2),
            "TestB_excess_beyond_climate_null_mm": round(dr(tb_qr_ds, tb_dn, tb_qr_ws, tb_wn), 2),
            "TestB_excess_boot95CI": [round(float(np.percentile(b_tb, 2.5)), 2), round(float(np.percentile(b_tb, 97.5)), 2)],
            "TestB_excess_interaction_mm": round(dr(tb_ir_ds, tb_dn, tb_ir_ws, tb_wn), 2),
            "TestB_excess_interaction_boot95CI": [round(float(np.percentile(b_tb_i, 2.5)), 2), round(float(np.percentile(b_tb_i, 97.5)), 2)],
        }

    # 3. Unified Deterministic S3 (GB -> AUS) Fair Comparison for ALL 4 models (M0, M3, M5, M5-Damped)
    gb_recs, aus_recs = [], []
    for bid, rows in by_b.items():
        a = attrs[bid]
        def fl(k, d=np.nan):
            v = a.get(k, "")
            return float(v) if v not in ("", "nan", "None") else d
        for r in rows:
            if r["has_lag1"] != "1":
                continue
            rec = {
                "bid": bid, "P": float(r["P_mm"]), "PET": float(r["PET_mm"]),
                "AI": float(r["AI_t"]), "RR": float(r["RR_t"]),
                "P1": float(r["P_lag1"]), "PET1": float(r["PET_lag1"]), "AI1": float(r["AI_lag1"]),
                "la": math.log10(max(fl("area_km2", 100.0), 0.1)), "el": fl("elev_mean_m"),
                "sl": fl("slope_proxy"), "ps": fl("p_seasonality"), "fo": fl("forest_frac"),
                "cl": fl("clay_pct"), "sa": fl("sand_pct"),
            }
            (gb_recs if r["region"] == "GB" else aus_recs).append(rec)

    gb_basins = sorted({r["bid"] for r in gb_recs})
    rng_s3 = np.random.default_rng(SEED)
    perm_idx = rng_s3.permutation(len(gb_basins))
    n_tr = int(0.75 * len(gb_basins))
    tr_set = {gb_basins[i] for i in perm_idx[:n_tr]}
    cal_set = {gb_basins[i] for i in perm_idx[n_tr:]}

    def to_xy(recs):
        X = np.array([[
            r["P"], r["PET"], r["AI"], math.log(max(r["AI"], 1e-4)),
            r["P1"], r["PET1"], r["AI1"], r["P"] / max(r["P1"], 1.0),
            r["la"], r["el"], r["sl"], r["ps"], r["fo"], r["cl"], r["sa"]
        ] for r in recs], dtype=float)
        y = np.array([r["RR"] for r in recs], dtype=float)
        ai = np.array([r["AI"] for r in recs], dtype=float)
        return X, y, ai

    X_tr, y_tr, ai_tr = to_xy([r for r in gb_recs if r["bid"] in tr_set])
    X_cal, y_cal, ai_cal = to_xy([r for r in gb_recs if r["bid"] in cal_set])
    X_te, y_te, ai_te = to_xy(aus_recs)

    def summarize_model(p_cal, p_te):
        p_cal = np.clip(p_cal, 0.0, 1.0)
        p_te = np.clip(p_te, 0.0, 1.0)
        q_sp = conformal_quantile_exact(np.abs(y_cal - p_cal), 0.10)
        rmse = float(np.sqrt(np.mean((p_te - y_te) ** 2)))
        r2 = float(1.0 - np.sum((p_te - y_te) ** 2) / np.sum((y_te - np.mean(y_te)) ** 2))
        bias = float(np.mean(p_te - y_te))
        return {
            "RMSE": round(rmse, 4), "R2": round(r2, 4), "Bias": round(bias, 4),
            "Split_CP": eval_interval(y_te, p_te - q_sp, p_te + q_sp, 0.10),
        }

    w_gb = fit_omega(ai_tr, y_tr)
    fu_tr, fu_cal, fu_te = fu_rr(ai_tr, w_gb), fu_rr(ai_cal, w_gb), fu_rr(ai_te, w_gb)

    m_dir = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=SEED).fit(X_tr, y_tr)
    m_res = HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.04, min_samples_leaf=40, l2_regularization=2.0, random_state=SEED).fit(X_tr, y_tr - fu_tr)

    ai_q95 = float(np.percentile(ai_tr, 95))
    ai_std = float(np.std(ai_tr))
    damp_w = lambda a: np.exp(-np.maximum(0.0, a - ai_q95) / max(ai_std, 1e-6))

    s3_unified = {
        "M0_Source_Fu": summarize_model(fu_cal, fu_te),
        "M3_Direct_HGBR": summarize_model(m_dir.predict(X_cal), m_dir.predict(X_te)),
        "M5_PhysRes_HGBR": summarize_model(fu_cal + m_res.predict(X_cal), fu_te + m_res.predict(X_te)),
        "M5_OOD_Damped_PhysRes": summarize_model(fu_cal + damp_w(ai_cal) * m_res.predict(X_cal), fu_te + damp_w(ai_te) * m_res.predict(X_te)),
    }

    out = {
        "Table1_SingleSource_Audit": audit_t1,
        "ClimatePreserving_Null_and_BasinBootstrap": clim_diag,
        "S3_GB_to_AUS_Unified_Split": s3_unified,
    }
    out_path = os.path.join(RES_DIR, "crossfit_null_and_transfer.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
