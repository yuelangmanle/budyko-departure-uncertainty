#!/usr/bin/env python3
"""
Zero-Leakage Modeling & Sharp Heteroscedastic Conformal Prediction
Across 1,202 Great Britain (CAMELS-GB) + Australia (CAMELS-AUS) Catchments.

Scenarios:
  S1: Gauged Temporal Forecast (per-basin chronological 60% train / 20% cal / 20% test)
  S2: Predictions in Ungauged Basins / PUB (5-fold basin-grouped CV, zero target history)
  S3: Cross-Continent Regime Shift (GB -> AUS and AUS -> GB)
  S4: Australian Millennium Drought Stress Test (Train <=1996 -> Test 1997-2009 Drought & 2010-2021 Recovery)

Conformal UQ Methods (nominal 90% coverage, alpha = 0.10):
  1. Split-CP (Global homoscedastic conformal)
  2. Locally Normalized CP (Scale-adaptive heteroscedastic conformal)
  3. Conformalized Quantile Regression (CQR, Romano et al. 2019)
  4. Region-Stratified (Mondrian) CQR (guaranteeing conditional coverage in both GB & AUS)
"""
import csv
import json
import math
import os
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)

ALPHA = 0.10  # 90% nominal coverage
SEED = 42


def fu_rr(ai, omega):
    ai = np.maximum(np.asarray(ai, dtype=float), 1e-9)
    er = 1.0 + ai - np.power(1.0 + np.power(ai, omega), 1.0 / omega)
    return np.clip(1.0 - er, 0.0, 1.0)


def fit_global_omega(ai, rr):
    res = minimize_scalar(
        lambda w: float(np.mean((fu_rr(ai, w) - rr) ** 2)),
        bounds=(1.1, 10.0),
        method="bounded",
    )
    return float(res.x)


def eval_point(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    bias = float(np.mean(err))
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4), "Bias": round(bias, 4)}


def eval_interval(y_true, lo, hi, alpha=ALPHA):
    y_true = np.asarray(y_true, dtype=float)
    lo = np.clip(np.asarray(lo, dtype=float), 0.0, 1.0)
    hi = np.clip(np.asarray(hi, dtype=float), 0.0, 1.0)
    hi = np.maximum(hi, lo)
    covered = (y_true >= lo) & (y_true <= hi)
    cov = float(np.mean(covered))
    width = float(np.mean(hi - lo))
    med_width = float(np.median(hi - lo))
    # Winkler / Interval Score (lower is better)
    winkler = (hi - lo) + (2.0 / alpha) * np.maximum(lo - y_true, 0.0) + (2.0 / alpha) * np.maximum(y_true - hi, 0.0)
    return {
        "Coverage": round(cov, 4),
        "Mean_Width": round(width, 4),
        "Median_Width": round(med_width, 4),
        "Winkler_Score": round(float(np.mean(winkler)), 4),
    }


def conformal_quantile(scores, alpha=ALPHA):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def load_unified():
    attrs = {}
    with open(os.path.join(DATA_DIR, "gb_aus_basin_attributes.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            attrs[r["basin_id"]] = r

    records = []
    with open(os.path.join(DATA_DIR, "gb_aus_annual_water_years.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["has_lag1"] != "1":
                continue
            bid = r["basin_id"]
            a = attrs[bid]
            def fl(k, default=np.nan):
                v = a.get(k, "")
                return float(v) if v not in ("", "nan", "None") else default

            records.append({
                "basin_id": bid,
                "region": r["region"],
                "wy": int(r["water_year"]),
                "P": float(r["P_mm"]),
                "PET": float(r["PET_mm"]),
                "Q": float(r["Q_mm"]),
                "AI": float(r["AI_t"]),
                "RR": float(r["RR_t"]),
                "P_lag1": float(r["P_lag1"]),
                "PET_lag1": float(r["PET_lag1"]),
                "AI_lag1": float(r["AI_lag1"]),
                "RR_lag1": float(r["RR_lag1"]),
                # Static physical attributes (strictly climate-free or long-term static physiography)
                "log_area": math.log10(max(fl("area_km2", 100.0), 0.1)),
                "elev": fl("elev_mean_m", np.nan),
                "slope": fl("slope_proxy", np.nan),
                "p_seas": fl("p_seasonality", np.nan),
                "forest": fl("forest_frac", np.nan),
                "clay": fl("clay_pct", np.nan),
                "sand": fl("sand_pct", np.nan),
                "bfi": fl("bfi", np.nan),
                "lat": fl("lat", np.nan),
                "is_aus": 1.0 if r["region"] == "AUS" else 0.0,
            })
    return attrs, records


def make_features(recs, mode="dynamic", basin_stats=None):
    """
    mode='static_pub': Uses ONLY current & lag-1 climate forcing + static physiographic attributes (NO BFI, NO RR_lag1, NO basin mean Q).
    mode='dynamic': Adds strictly training-only basin climatology + RR_lag1 (for gauged forecasting S1/S4).
    """
    X = []
    for r in recs:
        base = [
            r["P"], r["PET"], r["AI"], math.log(max(r["AI"], 1e-4)),
            r["P_lag1"], r["PET_lag1"], r["AI_lag1"],
            r["P"] / max(r["P_lag1"], 1.0),
            r["log_area"], r["elev"], r["slope"], r["p_seas"],
            r["forest"], r["clay"], r["sand"], r["is_aus"],
        ]
        if mode == "dynamic":
            bs = basin_stats[r["basin_id"]]
            base.extend([
                r["RR_lag1"],
                r["P"] / max(bs["mean_P"], 1.0),
                r["P_lag1"] / max(bs["mean_P"], 1.0),
                bs["omega_tr"],
                fu_rr(r["AI"], bs["omega_tr"]),
                r["RR_lag1"] - fu_rr(r["AI_lag1"], bs["omega_tr"]),
            ])
        X.append(base)
    return np.asarray(X, dtype=float)


def fit_and_eval_conformal_suite(X_tr, y_tr, X_cal, y_cal, X_te, y_te, reg_cal=None, reg_te=None):
    """Fit mean HGBR, local scale HGBR, and quantile HGBRs strictly on train; calibrate on cal; evaluate on test."""
    # 1. Point predictor
    m_mean = HistGradientBoostingRegressor(
        max_iter=250, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=SEED
    )
    m_mean.fit(X_tr, y_tr)
    pred_tr = np.clip(m_mean.predict(X_tr), 0.0, 1.0)
    pred_cal = np.clip(m_mean.predict(X_cal), 0.0, 1.0)
    pred_te = np.clip(m_mean.predict(X_te), 0.0, 1.0)

    pt_metrics = eval_point(y_te, pred_te)

    # 2. Split-CP (Global homoscedastic)
    scores_split = np.abs(y_cal - pred_cal)
    q_split = conformal_quantile(scores_split, ALPHA)
    iv_split = eval_interval(y_te, pred_te - q_split, pred_te + q_split)

    # 3. Locally Normalized CP (Heteroscedastic scale-adaptive)
    # Fit local scale model on internal 80/20 split of train to avoid in-sample overfitting of |resid|
    n_tr = len(y_tr)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_tr)
    cut = int(0.75 * n_tr)
    idx_a, idx_b = perm[:cut], perm[cut:]
    m_tmp = HistGradientBoostingRegressor(
        max_iter=200, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=SEED
    )
    m_tmp.fit(X_tr[idx_a], y_tr[idx_a])
    abs_res_b = np.abs(y_tr[idx_b] - np.clip(m_tmp.predict(X_tr[idx_b]), 0.0, 1.0))

    m_sig = HistGradientBoostingRegressor(
        max_iter=150, max_depth=4, learning_rate=0.05, min_samples_leaf=40, random_state=SEED
    )
    m_sig.fit(X_tr[idx_b], abs_res_b)
    eps = 0.015
    sig_cal = np.maximum(m_sig.predict(X_cal), 0.0) + eps
    sig_te = np.maximum(m_sig.predict(X_te), 0.0) + eps
    q_loc = conformal_quantile(np.abs(y_cal - pred_cal) / sig_cal, ALPHA)
    iv_loc = eval_interval(y_te, pred_te - q_loc * sig_te, pred_te + q_loc * sig_te)

    # 4. CQR (Conformalized Quantile Regression, Romano et al. 2019)
    m_lo = HistGradientBoostingRegressor(
        loss="quantile", quantile=ALPHA / 2.0,
        max_iter=250, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=SEED
    )
    m_hi = HistGradientBoostingRegressor(
        loss="quantile", quantile=1.0 - ALPHA / 2.0,
        max_iter=250, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=SEED
    )
    m_lo.fit(X_tr, y_tr)
    m_hi.fit(X_tr, y_tr)

    lo_cal = np.clip(m_lo.predict(X_cal), 0.0, 1.0)
    hi_cal = np.clip(m_hi.predict(X_cal), 0.0, 1.0)
    lo_te = np.clip(m_lo.predict(X_te), 0.0, 1.0)
    hi_te = np.clip(m_hi.predict(X_te), 0.0, 1.0)

    # Raw (uncalibrated) QR
    iv_qr_raw = eval_interval(y_te, lo_te, hi_te)

    # Global CQR
    scores_cqr = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    q_cqr = conformal_quantile(scores_cqr, ALPHA)
    cqr_lo = lo_te - q_cqr
    cqr_hi = hi_te + q_cqr
    iv_cqr = eval_interval(y_te, cqr_lo, cqr_hi)

    res = {
        "point": pt_metrics,
        "Split_CP": iv_split,
        "Local_Adaptive_CP": iv_loc,
        "Raw_QR": iv_qr_raw,
        "CQR": iv_cqr,
    }

    # Regional breakdown + Region-Stratified (Mondrian) CQR when both GB & AUS present
    if reg_cal is not None and reg_te is not None:
        reg_cal = np.asarray(reg_cal)
        reg_te = np.asarray(reg_te)
        mond_lo = np.copy(lo_te)
        mond_hi = np.copy(hi_te)
        by_reg = {}
        for rg in ["GB", "AUS"]:
            m_c = reg_cal == rg
            m_t = reg_te == rg
            if np.sum(m_c) > 20 and np.sum(m_t) > 20:
                q_rg = conformal_quantile(scores_cqr[m_c], ALPHA)
                mond_lo[m_t] = lo_te[m_t] - q_rg
                mond_hi[m_t] = hi_te[m_t] + q_rg
                by_reg[rg] = {
                    "point": eval_point(y_te[m_t], pred_te[m_t]),
                    "Split_CP": eval_interval(y_te[m_t], pred_te[m_t] - q_split, pred_te[m_t] + q_split),
                    "Local_Adaptive_CP": eval_interval(y_te[m_t], pred_te[m_t] - q_loc * sig_te[m_t], pred_te[m_t] + q_loc * sig_te[m_t]),
                    "CQR": eval_interval(y_te[m_t], cqr_lo[m_t], cqr_hi[m_t]),
                    "Mondrian_CQR": eval_interval(y_te[m_t], mond_lo[m_t], mond_hi[m_t]),
                }
        res["Mondrian_CQR"] = eval_interval(y_te, mond_lo, mond_hi)
        res["by_region"] = by_reg

    return res, pred_te, cqr_lo, cqr_hi, pred_te - q_split, pred_te + q_split


def run_s1_gauged_temporal(records):
    print("\n--- Scenario 1: Gauged Temporal Forecast (Chronological 60/20/20 per basin) ---")
    by_b = defaultdict(list)
    for r in records:
        by_b[r["basin_id"]].append(r)

    tr_recs, cal_recs, te_recs = [], [], []
    basin_stats = {}
    for bid, blist in by_b.items():
        blist.sort(key=lambda x: x["wy"])
        n = len(blist)
        i1 = int(0.60 * n)
        i2 = int(0.80 * n)
        tr_b = blist[:i1]
        cal_b = blist[i1:i2]
        te_b = blist[i2:]
        # Fit catchment Fu omega STRICTLY on training years only!
        mean_P = float(np.mean([x["P"] for x in tr_b]))
        mean_PET = float(np.mean([x["PET"] for x in tr_b]))
        mean_Q = float(np.mean([x["Q"] for x in tr_b]))
        lt_ai = mean_PET / mean_P
        lt_rr = min(max(mean_Q / mean_P, 0.002), 0.995)
        w_tr = fit_global_omega(np.array([lt_ai]), np.array([lt_rr]))
        basin_stats[bid] = {"mean_P": mean_P, "omega_tr": w_tr}
        tr_recs.extend(tr_b); cal_recs.extend(cal_b); te_recs.extend(te_b)

    y_tr = np.array([r["RR"] for r in tr_recs])
    y_cal = np.array([r["RR"] for r in cal_recs])
    y_te = np.array([r["RR"] for r in te_recs])
    reg_cal = [r["region"] for r in cal_recs]
    reg_te = [r["region"] for r in te_recs]

    # Baselines
    w_glob = fit_global_omega(np.array([r["AI"] for r in tr_recs]), y_tr)
    pred_m0 = fu_rr([r["AI"] for r in te_recs], w_glob)
    pred_m1 = np.array([fu_rr(r["AI"], basin_stats[r["basin_id"]]["omega_tr"]) for r in te_recs])
    pred_pers = np.array([r["RR_lag1"] for r in te_recs])

    # Analytical Dynamic Budyko ARX: RR_t = fu(AI_t; w_i) + b0 + b1*(RR_{t-1} - fu(AI_{t-1}; w_i)) + b2*log(P_{t-1}/mean_P_i)
    def build_arx_design(recs):
        f_cur = np.array([fu_rr(r["AI"], basin_stats[r["basin_id"]]["omega_tr"]) for r in recs])
        f_lag = np.array([fu_rr(r["AI_lag1"], basin_stats[r["basin_id"]]["omega_tr"]) for r in recs])
        res_lag = np.array([r["RR_lag1"] for r in recs]) - f_lag
        lp_lag = np.array([math.log(max(r["P_lag1"] / basin_stats[r["basin_id"]]["mean_P"], 0.05)) for r in recs])
        return f_cur, np.column_stack([np.ones(len(recs)), res_lag, lp_lag])

    f_tr, D_tr = build_arx_design(tr_recs)
    beta, *_ = np.linalg.lstsq(D_tr, y_tr - f_tr, rcond=None)
    f_te, D_te = build_arx_design(te_recs)
    pred_m2 = np.clip(f_te + D_te @ beta, 0.0, 1.0)

    # Full Dynamic HGBR + Conformal Suite
    X_tr = make_features(tr_recs, "dynamic", basin_stats)
    X_cal = make_features(cal_recs, "dynamic", basin_stats)
    X_te = make_features(te_recs, "dynamic", basin_stats)
    suite, pred_te, cqr_lo, cqr_hi, sp_lo, sp_hi = fit_and_eval_conformal_suite(
        X_tr, y_tr, X_cal, y_cal, X_te, y_te, reg_cal, reg_te
    )

    # Save test predictions for plotting Figure 4 & 5
    np.savez_compressed(
        os.path.join(RES_DIR, "s1_test_predictions.npz"),
        y_te=y_te,
        pred_te=pred_te,
        pred_m0=pred_m0,
        pred_m1=pred_m1,
        pred_m2=pred_m2,
        cqr_lo=cqr_lo,
        cqr_hi=cqr_hi,
        sp_lo=sp_lo,
        sp_hi=sp_hi,
        ai_te=np.array([r["AI"] for r in te_recs]),
        is_aus=np.array([1 if r["region"] == "AUS" else 0 for r in te_recs]),
    )

    return {
        "n_train": len(tr_recs),
        "n_cal": len(cal_recs),
        "n_test": len(te_recs),
        "baselines": {
            "M0_Global_Fu": eval_point(y_te, pred_m0),
            "M1_Basin_Fu": eval_point(y_te, pred_m1),
            "M1b_Persistence_RR_lag1": eval_point(y_te, pred_pers),
            "M2_Dynamic_Budyko_ARX": eval_point(y_te, pred_m2),
            "ARX_coeffs_const_resLag_logPlag": [round(float(x), 4) for x in beta],
        },
        "M4_HGBR_Dynamic_UQ": suite,
    }


def run_s2_pub_spatial_cv(records):
    print("\n--- Scenario 2: Ungauged Basins / PUB (5-Fold Basin-Grouped CV, Zero Target History) ---")
    basins = sorted(set(r["basin_id"] for r in records))
    rng = np.random.default_rng(SEED)
    perm_basins = np.array(basins)[rng.permutation(len(basins))]
    kf = KFold(n_splits=5, shuffle=False)

    by_b = defaultdict(list)
    for r in records:
        by_b[r["basin_id"]].append(r)

    all_y, all_m0, all_pred = [], [], []
    all_sp_lo, all_sp_hi = [], []
    all_loc_lo, all_loc_hi = [], []
    all_cqr_lo, all_cqr_hi = [], []
    all_mond_lo, all_mond_hi = [], []
    all_reg = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(perm_basins)):
        # Split tr_idx basins into 75% train basins / 25% calibration basins (ZERO basin overlap!)
        cut = int(0.75 * len(tr_idx))
        tr_b = perm_basins[tr_idx[:cut]]
        cal_b = perm_basins[tr_idx[cut:]]
        te_b = perm_basins[te_idx]

        tr_recs = [r for b in tr_b for r in by_b[b]]
        cal_recs = [r for b in cal_b for r in by_b[b]]
        te_recs = [r for b in te_b for r in by_b[b]]

        X_tr = make_features(tr_recs, "static_pub")
        X_cal = make_features(cal_recs, "static_pub")
        X_te = make_features(te_recs, "static_pub")
        y_tr = np.array([r["RR"] for r in tr_recs])
        y_cal = np.array([r["RR"] for r in cal_recs])
        y_te = np.array([r["RR"] for r in te_recs])
        reg_cal = np.array([r["region"] for r in cal_recs])
        reg_te = np.array([r["region"] for r in te_recs])

        w_g = fit_global_omega(np.array([r["AI"] for r in tr_recs]), y_tr)
        all_m0.append(fu_rr([r["AI"] for r in te_recs], w_g))

        # Fit mean + local scale + CQR
        m_mean = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=SEED + fold)
        m_mean.fit(X_tr, y_tr)
        p_cal = np.clip(m_mean.predict(X_cal), 0.0, 1.0)
        p_te = np.clip(m_mean.predict(X_te), 0.0, 1.0)

        q_sp = conformal_quantile(np.abs(y_cal - p_cal), ALPHA)
        all_sp_lo.append(p_te - q_sp); all_sp_hi.append(p_te + q_sp)

        # Local adaptive
        p_tr_perm = rng.permutation(len(y_tr))
        c2 = int(0.75 * len(y_tr))
        m_tmp = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=SEED + fold)
        m_tmp.fit(X_tr[p_tr_perm[:c2]], y_tr[p_tr_perm[:c2]])
        res_b = np.abs(y_tr[p_tr_perm[c2:]] - np.clip(m_tmp.predict(X_tr[p_tr_perm[c2:]]), 0.0, 1.0))
        m_sig = HistGradientBoostingRegressor(max_iter=150, max_depth=4, learning_rate=0.05, min_samples_leaf=40, random_state=SEED + fold)
        m_sig.fit(X_tr[p_tr_perm[c2:]], res_b)
        eps = 0.015
        sc_cal = np.maximum(m_sig.predict(X_cal), 0.0) + eps
        sc_te = np.maximum(m_sig.predict(X_te), 0.0) + eps
        q_loc = conformal_quantile(np.abs(y_cal - p_cal) / sc_cal, ALPHA)
        all_loc_lo.append(p_te - q_loc * sc_te); all_loc_hi.append(p_te + q_loc * sc_te)

        # CQR + Mondrian CQR
        m_lo = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2.0, max_iter=250, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=SEED + fold)
        m_hi = HistGradientBoostingRegressor(loss="quantile", quantile=1.0 - ALPHA / 2.0, max_iter=250, max_depth=6, learning_rate=0.05, min_samples_leaf=30, random_state=SEED + fold)
        m_lo.fit(X_tr, y_tr); m_hi.fit(X_tr, y_tr)
        lo_c, hi_c = np.clip(m_lo.predict(X_cal), 0.0, 1.0), np.clip(m_hi.predict(X_cal), 0.0, 1.0)
        lo_t, hi_t = np.clip(m_lo.predict(X_te), 0.0, 1.0), np.clip(m_hi.predict(X_te), 0.0, 1.0)
        s_cqr = np.maximum(lo_c - y_cal, y_cal - hi_c)
        q_cqr = conformal_quantile(s_cqr, ALPHA)
        all_cqr_lo.append(lo_t - q_cqr); all_cqr_hi.append(hi_t + q_cqr)

        m_lo_m, m_hi_m = np.copy(lo_t), np.copy(hi_t)
        for rg in ["GB", "AUS"]:
            q_rg = conformal_quantile(s_cqr[reg_cal == rg], ALPHA)
            m_lo_m[reg_te == rg] = lo_t[reg_te == rg] - q_rg
            m_hi_m[reg_te == rg] = hi_t[reg_te == rg] + q_rg
        all_mond_lo.append(m_lo_m); all_mond_hi.append(m_hi_m)

        all_y.append(y_te); all_pred.append(p_te); all_reg.append(reg_te)

    y_all = np.concatenate(all_y)
    m0_all = np.concatenate(all_m0)
    p_all = np.concatenate(all_pred)
    sp_lo, sp_hi = np.concatenate(all_sp_lo), np.concatenate(all_sp_hi)
    loc_lo, loc_hi = np.concatenate(all_loc_lo), np.concatenate(all_loc_hi)
    cqr_lo, cqr_hi = np.concatenate(all_cqr_lo), np.concatenate(all_cqr_hi)
    mond_lo, mond_hi = np.concatenate(all_mond_lo), np.concatenate(all_mond_hi)
    reg_all = np.concatenate(all_reg)

    by_reg = {}
    for rg in ["GB", "AUS"]:
        m = reg_all == rg
        by_reg[rg] = {
            "M0_Global_Fu": eval_point(y_all[m], m0_all[m]),
            "M3_HGBR_PUB_point": eval_point(y_all[m], p_all[m]),
            "Split_CP": eval_interval(y_all[m], sp_lo[m], sp_hi[m]),
            "Local_Adaptive_CP": eval_interval(y_all[m], loc_lo[m], loc_hi[m]),
            "CQR": eval_interval(y_all[m], cqr_lo[m], cqr_hi[m]),
            "Mondrian_CQR": eval_interval(y_all[m], mond_lo[m], mond_hi[m]),
        }

    return {
        "n_test_total": int(len(y_all)),
        "M0_Global_Fu": eval_point(y_all, m0_all),
        "M3_HGBR_PUB_point": eval_point(y_all, p_all),
        "Split_CP": eval_interval(y_all, sp_lo, sp_hi),
        "Local_Adaptive_CP": eval_interval(y_all, loc_lo, loc_hi),
        "CQR": eval_interval(y_all, cqr_lo, cqr_hi),
        "Mondrian_CQR": eval_interval(y_all, mond_lo, mond_hi),
        "by_region": by_reg,
    }


def run_s3_cross_continent(records):
    print("\n--- Scenario 3: Cross-Continent OOD Transfer (GB <-> AUS) ---")
    gb = [r for r in records if r["region"] == "GB"]
    aus = [r for r in records if r["region"] == "AUS"]
    out = {}
    rng = np.random.default_rng(SEED)

    for label, src, tgt in [("GB_to_AUS", gb, aus), ("AUS_to_GB", aus, gb)]:
        src_basins = sorted(set(r["basin_id"] for r in src))
        perm = rng.permutation(len(src_basins))
        cut = int(0.75 * len(src_basins))
        tr_set = set(src_basins[i] for i in perm[:cut])
        tr_recs = [r for r in src if r["basin_id"] in tr_set]
        cal_recs = [r for r in src if r["basin_id"] not in tr_set]
        te_recs = tgt

        X_tr = make_features(tr_recs, "static_pub")[:, :-1]  # drop region indicator for cross-continent!
        X_cal = make_features(cal_recs, "static_pub")[:, :-1]
        X_te = make_features(te_recs, "static_pub")[:, :-1]
        y_tr = np.array([r["RR"] for r in tr_recs])
        y_cal = np.array([r["RR"] for r in cal_recs])
        y_te = np.array([r["RR"] for r in te_recs])

        w_g = fit_global_omega(np.array([r["AI"] for r in tr_recs]), y_tr)
        m0_res = eval_point(y_te, fu_rr([r["AI"] for r in te_recs], w_g))
        suite, *_ = fit_and_eval_conformal_suite(X_tr, y_tr, X_cal, y_cal, X_te, y_te)
        out[label] = {"M0_Source_Fu": m0_res, "M3_HGBR_Transfer_UQ": suite}
    return out


def run_s4_millennium_drought(records):
    print("\n--- Scenario 4: Australian Millennium Drought Stress Test (Train <=1996 -> 1997-2009 & 2010-2021) ---")
    aus = [r for r in records if r["region"] == "AUS"]
    by_b = defaultdict(list)
    for r in aus:
        by_b[r["basin_id"]].append(r)

    tr_recs, cal_recs, drought_recs, post_recs = [], [], [], []
    basin_stats = {}
    for bid, blist in by_b.items():
        pre = [x for x in blist if x["wy"] <= 1996]
        if len(pre) < 10:
            continue
        pre.sort(key=lambda x: x["wy"])
        cut = int(0.75 * len(pre))
        tr_b = pre[:cut]
        cal_b = pre[cut:]
        mean_P = float(np.mean([x["P"] for x in tr_b]))
        mean_PET = float(np.mean([x["PET"] for x in tr_b]))
        mean_Q = float(np.mean([x["Q"] for x in tr_b]))
        w_tr = fit_global_omega(np.array([mean_PET / mean_P]), np.array([min(max(mean_Q / mean_P, 0.002), 0.995)]))
        basin_stats[bid] = {"mean_P": mean_P, "omega_tr": w_tr}
        tr_recs.extend(tr_b); cal_recs.extend(cal_b)
        drought_recs.extend([x for x in blist if 1997 <= x["wy"] <= 2009])
        post_recs.extend([x for x in blist if x["wy"] >= 2010])

    X_tr = make_features(tr_recs, "dynamic", basin_stats)
    X_cal = make_features(cal_recs, "dynamic", basin_stats)
    y_tr = np.array([r["RR"] for r in tr_recs])
    y_cal = np.array([r["RR"] for r in cal_recs])

    out = {"n_train_pre1997": len(tr_recs), "n_cal_pre1997": len(cal_recs)}
    for label, te_recs in [("Drought_1997_2009", drought_recs), ("PostDrought_2010_2021", post_recs)]:
        X_te = make_features(te_recs, "dynamic", basin_stats)
        y_te = np.array([r["RR"] for r in te_recs])
        m1_pred = np.array([fu_rr(r["AI"], basin_stats[r["basin_id"]]["omega_tr"]) for r in te_recs])
        suite, *_ = fit_and_eval_conformal_suite(X_tr, y_tr, X_cal, y_cal, X_te, y_te)
        out[label] = {
            "n_test": len(te_recs),
            "M1_PreDrought_Basin_Fu": eval_point(y_te, m1_pred),
            "M4_HGBR_Dynamic_UQ": suite,
        }
    return out


def main():
    attrs, records = load_unified()
    print(f"Loaded {len(records)} consecutive lag-1 records across {len(set(r['basin_id'] for r in records))} basins.")

    results = {
        "S1_Gauged_Temporal": run_s1_gauged_temporal(records),
        "S2_PUB_Spatial_CV": run_s2_pub_spatial_cv(records),
        "S3_Cross_Continent_OOD": run_s3_cross_continent(records),
        "S4_AUS_Millennium_Drought": run_s4_millennium_drought(records),
    }

    out_path = os.path.join(RES_DIR, "modeling_uq_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n=== MODELING & CONFORMAL UQ SUMMARY ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
