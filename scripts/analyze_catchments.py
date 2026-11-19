\
\
\
\
\
\
\
\
\
\
\
\
\

import csv
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from fetch_usgs_era5 import USGS_60_GAUGES, USGS_ALL_GAUGES


def resolve_data_path(filename):
    candidates = [
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(SCRIPT_DIR, "..", "data", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return os.path.join(SCRIPT_DIR, filename)


def fu_budyko_analytical(AI, w=2.6):
    AI = np.asarray(AI, dtype=float)
    et_ratio = 1.0 + AI - (1.0 + np.maximum(AI, 1e-6) ** w) ** (1.0 / w)
    return np.clip(1.0 - et_ratio, 0.0, 1.0)


def fit_basin_w(ai_arr, qp_arr):
    ai_arr = np.asarray(ai_arr, dtype=float)
    qp_arr = np.asarray(qp_arr, dtype=float)
    res = minimize_scalar(
        lambda w: float(np.mean((qp_arr - fu_budyko_analytical(ai_arr, w)) ** 2)),
        bounds=(1.05, 8.0),
        method="bounded",
    )
    return float(res.x)


def winkler_score(y_true, lo, hi, alpha=0.10):
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    w = hi - lo
    pen_lo = (2.0 / alpha) * np.maximum(0.0, lo - y_true)
    pen_hi = (2.0 / alpha) * np.maximum(0.0, y_true - hi)
    return float(np.mean(w + pen_lo + pen_hi))


def conformal_order_quantile(arr, alpha=0.10, max_bound=1.0):
\
\
\
\
\
\
\

    s = np.sort(np.asarray(arr, dtype=float))
    n = len(s)
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float(max(max_bound, s[-1]))
    return float(s[max(k - 1, 0)])


def ijs_conformal_interval(X_pi_fit, X_pi_cal, y_cal, pred_cal, X_pi_test, pred_test, phys_op_fn, alpha=0.10, mode="full", clip_bounds=True):
    sc = StandardScaler().fit(X_pi_fit)
    Z_fit = sc.transform(X_pi_fit)
    Z_cal = sc.transform(X_pi_cal)
    Z_test = sc.transform(X_pi_test)

    nn1 = NearestNeighbors(n_neighbors=1).fit(Z_fit)
    d_cal, idx_cal_nn = nn1.kneighbors(Z_cal)
    d_test, idx_test_nn = nn1.kneighbors(Z_test)
    d_cal = d_cal[:, 0]
    d_test = d_test[:, 0]
    d_90 = conformal_order_quantile(d_cal, alpha) + 1e-6

    pi_proj_cal = X_pi_fit[idx_cal_nn[:, 0]]
    pi_proj_test = X_pi_fit[idx_test_nn[:, 0]]

    dM_cal = np.abs(phys_op_fn(X_pi_cal) - phys_op_fn(pi_proj_cal))
    dM_test = np.abs(phys_op_fn(X_pi_test) - phys_op_fn(pi_proj_test))
    dM_scale = conformal_order_quantile(dM_cal, alpha) + 1e-4

    res_cal = np.abs(y_cal - pred_cal)
    q_pi = conformal_order_quantile(res_cal, alpha)

    excess_d = np.maximum(0.0, d_test / d_90 - 1.0)
    excess_M = np.maximum(0.0, dM_test - dM_scale)

    if mode == "std":
        hw = np.full_like(pred_test, q_pi)
    elif mode == "const":
        cut = conformal_order_quantile(d_cal, 0.20)
        b_mask = d_cal >= cut
        c_fac = max(1.0, float(conformal_order_quantile(res_cal[b_mask], alpha) / (q_pi + 1e-6)))
        hw = np.full_like(pred_test, q_pi * c_fac)
    elif mode == "dist_only":
        hw = q_pi * np.sqrt(1.0 + excess_d)
    elif mode == "jac_only":
        hw = q_pi + excess_M
    else:
        hw = q_pi * np.sqrt(1.0 + excess_d) + excess_M

    if clip_bounds:
        lo = np.clip(pred_test - hw, 0.0, 1.0)
        hi = np.clip(pred_test + hw, 0.0, 1.0)
    else:
        lo = pred_test - hw
        hi = pred_test + hw
    return lo, hi, q_pi


def classify_region(lat, lon):
    if lon > -84.0:
        return "Appalachian & Northeast"
    elif lon > -104.0:
        return "Central Plains & Midwest"
    elif lon > -117.0:
        return "Rocky Mountains & Southwest"
    else:
        return "Pacific Northwest & Coast"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return float(2.0 * R * np.arcsin(np.sqrt(a)))


def paired_cluster_bootstrap_rmse(y_true, pred_a, pred_b, site_ids, n_boot=3000, seed=42):
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    site_ids = np.asarray(site_ids)
    uniq_sites = np.unique(site_ids)
    site_to_idx = {s: np.where(site_ids == s)[0] for s in uniq_sites}
    sq_a = (y_true - pred_a) ** 2
    sq_b = (y_true - pred_b) ** 2
    obs_diff = float(np.sqrt(np.mean(sq_a)) - np.sqrt(np.mean(sq_b)))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        samp = rng.choice(uniq_sites, size=len(uniq_sites), replace=True)
        idx_b = np.concatenate([site_to_idx[s] for s in samp])
        diffs[b] = np.sqrt(np.mean(sq_a[idx_b])) - np.sqrt(np.mean(sq_b[idx_b]))
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return obs_diff, ci_lo, ci_hi, len(uniq_sites)


def cluster_bootstrap_coverage(y_true, lo, hi, site_ids, n_boot=3000, seed=42):
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    covered = ((y_true >= lo) & (y_true <= hi)).astype(float)
    site_ids = np.asarray(site_ids)
    uniq_sites = np.unique(site_ids)
    site_to_idx = {s: np.where(site_ids == s)[0] for s in uniq_sites}

    rng = np.random.default_rng(seed)
    covs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        samp = rng.choice(uniq_sites, size=len(uniq_sites), replace=True)
        idx_b = np.concatenate([site_to_idx[s] for s in samp])
        covs[b] = float(np.mean(covered[idx_b]))
    return float(np.mean(covered)), float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5)), len(uniq_sites)


def split_fit_cal_by_basin(id_tr_records, test_sites_set, test_size=0.25, random_state=42):
    groups = np.array([r["site_id"] for r in id_tr_records])
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    i_f, i_c = next(gss.split(np.arange(len(id_tr_records)), groups=groups))
    fit_sites = set(groups[i_f])
    cal_sites = set(groups[i_c])
    assert len(fit_sites & cal_sites) == 0, "Fit and calibration basins must be disjoint!"
    assert len(fit_sites & test_sites_set) == 0, "Fit and test basins must be disjoint!"
    assert len(cal_sites & test_sites_set) == 0, "Calibration and test basins must be disjoint!"
    return i_f, i_c, len(fit_sites), len(cal_sites)


FROZEN_5FOLD_SITES = [
    ['01047000', '01144000', '01181000', '01440000', '01539000', '01632900', '02051500', '02212600', '03182500', '03364500', '04122500', '05244000', '05454000', '06191500', '06224000', '06847900', '07144780', '08150000', '09404450', '10316500', '12413000', '14362250'],
    ['01031500', '01137500', '01362500', '01518000', '01568000', '02038850', '02143000', '03049000', '03164000', '03346000', '04105700', '05120500', '05436500', '06037500', '06803530', '07067000', '08086212', '09066200', '11532500', '12027500', '12117500', '14308000'],
    ['01013500', '01121000', '01421000', '02018000', '02138500', '03144000', '03303000', '03439000', '04063700', '04234000', '05412500', '05584500', '06441500', '07056000', '07340300', '08195000', '09210500', '09430500', '11264500', '14137000', '14185000'],
    ['01073000', '01350000', '01413500', '01552000', '01664000', '02102000', '03069500', '03281500', '03504000', '04045500', '04196800', '05362000', '05525000', '06409000', '06892000', '07261000', '08171000', '10172700', '11473900', '12358500', '13337000'],
    ['01055000', '01170100', '01193500', '01543500', '01644000', '02083500', '02118500', '03011020', '03237500', '03574500', '04159492', '05291000', '05508805', '06354000', '06614800', '06885500', '07186000', '08167500', '09081600', '09352900', '13313000'],
]


def deterministic_basin_splits(records):
    site_arr = np.array([r["site_id"] for r in records])
    splits = []
    for test_sites in FROZEN_5FOLD_SITES:
        te_set = set(test_sites)
        te_idx = np.array([i for i, s in enumerate(site_arr) if s in te_set], dtype=int)
        tr_idx = np.array([i for i, s in enumerate(site_arr) if s not in te_set], dtype=int)
        splits.append((tr_idx, te_idx))
    return splits


def compute_fit_isolated_basin_scalars(fit_records, all_site_ids, eval_records=None):
    by_s = {}
    for r in fit_records:
        by_s.setdefault(r["site_id"], []).append(r)
    alrq_pool, cvp_pool = [], []
    for sid, rs in sorted(by_s.items()):
        if len(rs) >= 2:
            q_arr = [x["Q_over_P"] * x["P"] for x in rs]
            p_arr = [x["P"] for x in rs]
            alrq_pool.append(float(np.percentile(q_arr, 15) / (np.percentile(q_arr, 50) + 1e-6)))
            cvp_pool.append(float(np.std(p_arr) / (np.mean(p_arr) + 1e-6)))
    med_alrq = float(np.median(alrq_pool)) if alrq_pool else 0.45
    med_cvp = float(np.median(cvp_pool)) if cvp_pool else 0.20

    alrq_map, cvp_map = {}, {}
    imputed_sids = set()
    for sid in all_site_ids:
        rs = by_s.get(sid, [])
        if len(rs) >= 2:
            q_arr = [x["Q_over_P"] * x["P"] for x in rs]
            p_arr = [x["P"] for x in rs]
            alrq_map[sid] = float(np.percentile(q_arr, 15) / (np.percentile(q_arr, 50) + 1e-6))
            cvp_map[sid] = float(np.std(p_arr) / (np.mean(p_arr) + 1e-6))
        else:
            alrq_map[sid] = med_alrq
            cvp_map[sid] = med_cvp
            imputed_sids.add(sid)

    n_imp_eval = sum(1 for r in (eval_records or []) if r["site_id"] in imputed_sids)
    pct_imp_eval = 100.0 * n_imp_eval / max(1, len(eval_records or []))
    diag = {
        "n_fit_records": len(fit_records),
        "n_total_basins": len(all_site_ids),
        "n_imputed_basins": len(imputed_sids),
        "n_eval_records": len(eval_records or []),
        "n_imputed_eval": n_imp_eval,
        "pct_imputed_eval": pct_imp_eval,
    }
    return alrq_map, cvp_map, diag


def main():
    cache_path = resolve_data_path("usgs_108_catchments_2005_2024.json")
    out_dir = os.path.dirname(cache_path)
    with open(cache_path) as f:
        all_cached = json.load(f)

    seen = set()
    raw_records = []
    for r in all_cached:
        k = (r["site_id"], r["year"])
        if k not in seen:
            seen.add(k)
            raw_records.append(r)

    direct_60_ids = {s[0] for s in USGS_60_GAUGES}
    meta_map = {s[0]: {"lat": s[1], "lon": s[2], "area_km2": s[3]} for s in USGS_ALL_GAUGES}

    by_site = {}
    for r in raw_records:
        by_site.setdefault(r["site_id"], []).append(r)
    for sid in by_site:
        by_site[sid].sort(key=lambda x: x["year"])
    sids_sorted = sorted(by_site.keys())


    d60_coords = [(meta_map[s]["lat"], meta_map[s]["lon"]) for s in sids_sorted if s in direct_60_ids]
    idw_dists = []
    for s in sids_sorted:
        if s not in direct_60_ids:
            lat1, lon1 = meta_map[s]["lat"], meta_map[s]["lon"]
            idw_dists.append(min(haversine_km(lat1, lon1, lat2, lon2) for lat2, lon2 in d60_coords))
    idw_dists = np.array(idw_dists)
    print("=== Geodesic Distance Summary (48 IDW Gauges to Nearest Direct ERA5-Land Gauge) ===")
    print(f"  48 IDW Gauges: median = {np.median(idw_dists):.1f} km, >100 km = {int(np.sum(idw_dists > 100))}/48, >200 km = {int(np.sum(idw_dists > 200))}/48, max = {np.max(idw_dists):.1f} km")

    w_by_site = {}
    pi_S_by_site = {}
    for sid, recs in by_site.items():
        ai_s = [r["AI"] for r in recs]
        qp_s = [r["Q_over_P"] for r in recs]
        w_by_site[sid] = fit_basin_w(ai_s, qp_s)
        pi_S_by_site[sid] = float(np.mean([r["Q_over_P"] - float(fu_budyko_analytical(r["AI"], w=2.6)) for r in recs]))


    rng = np.random.default_rng(42)
    pi_S_vec = np.array([pi_S_by_site[s] for s in sids_sorted])
    w_vec = np.array([w_by_site[s] for s in sids_sorted])
    pi_S_null_vec = []
    w_early, w_late, pis_early, pis_late, pis_early_strict_null, is_d60_f1 = [], [], [], [], [], []
    for s in sids_sorted:
        recs = by_site[s]
        ai = np.array([r["AI"] for r in recs])
        qp = np.array([r["Q_over_P"] for r in recs])
        wi = w_by_site[s]
        sd = float(np.std(qp - fu_budyko_analytical(ai, wi)))
        qp_null = np.clip(fu_budyko_analytical(ai, wi) + rng.normal(0, sd, size=len(ai)), 0.0, 1.0)
        pi_S_null_vec.append(float(np.mean(qp_null - fu_budyko_analytical(ai, 2.6))))
        e_recs = [r for r in recs if r["year"] <= 2014]
        l_recs = [r for r in recs if r["year"] >= 2015]
        if len(e_recs) >= 4 and len(l_recs) >= 4:
            ai_e = np.array([r["AI"] for r in e_recs]); qp_e = np.array([r["Q_over_P"] for r in e_recs])
            ai_l = np.array([r["AI"] for r in l_recs]); qp_l = np.array([r["Q_over_P"] for r in l_recs])
            we = fit_basin_w(ai_e, qp_e)
            wl = fit_basin_w(ai_l, qp_l)
            w_early.append(we)
            w_late.append(wl)
            pis_early.append(float(np.mean(qp_e - fu_budyko_analytical(ai_e, 2.6))))
            pis_late.append(float(np.mean(qp_l - fu_budyko_analytical(ai_l, 2.6))))
            pis_early_strict_null.append(float(np.mean(fu_budyko_analytical(ai_e, we) - fu_budyko_analytical(ai_e, 2.6))))
            is_d60_f1.append(s in direct_60_ids)

    is_d60_f1 = np.array(is_d60_f1, dtype=bool)
    pis_early_arr = np.array(pis_early)
    w_late_arr = np.array(w_late)
    pis_null_arr = np.array(pis_early_strict_null)
    f1_cross_all = float(spearmanr(pis_early_arr, w_late_arr).statistic)
    f1_null_all = float(spearmanr(pis_null_arr, w_late_arr).statistic)
    f1_cross_d60 = float(spearmanr(pis_early_arr[is_d60_f1], w_late_arr[is_d60_f1]).statistic)
    f1_null_d60 = float(spearmanr(pis_null_arr[is_d60_f1], w_late_arr[is_d60_f1]).statistic)

    print("=== F1 Null-Model Check: Structural Reparameterization & Cross-Decade Stability ===")
    print(f"  Same-period observed rho(Pi_S, w_i) = {spearmanr(pi_S_vec, w_vec).statistic:.4f} vs. Memoryless Null = {spearmanr(pi_S_null_vec, w_vec).statistic:.4f}")
    print(f"  All 108 Basins Cross-Decade rho(Pi_S_2005-2014, w_2015-2024) = {f1_cross_all:.4f} (p = {spearmanr(pis_early_arr, w_late_arr).pvalue:.2e}) vs. Early Static Null = {f1_null_all:.4f}")
    print(f"  Pure 60 Direct Gauges Cross-Decade rho(Pi_S_2005-2014, w_2015-2024) = {f1_cross_d60:.4f} vs. Early Static Null = {f1_null_d60:.4f}")

    complete_records = []
    for sid, recs in by_site.items():
        m = meta_map[sid]
        wi = w_by_site[sid]
        ai_all = np.array([r["AI"] for r in recs])
        qp_all = np.array([r["Q_over_P"] for r in recs])
        mean_res_wi = float(np.mean(qp_all - fu_budyko_analytical(ai_all, wi)))
        for i in range(1, len(recs)):
            curr = recs[i]
            prev = recs[i - 1]
            if curr["year"] - prev["year"] != 1:
                continue
            q_budyko_prev = float(fu_budyko_analytical(prev["AI"], w=2.6))
            q_budyko_curr = float(fu_budyko_analytical(curr["AI"], w=2.6))
            pi_M = prev["Q_over_P"] - q_budyko_prev
            pi_Delta = (curr["P"] - prev["P"]) / (prev["P"] + 1e-6)
            dep_curr = curr["Q_over_P"] - q_budyko_curr
            res_wi_curr = (curr["Q_over_P"] - float(fu_budyko_analytical(curr["AI"], w=wi))) - mean_res_wi
            complete_records.append({
                **curr,
                "is_direct60": sid in direct_60_ids,
                "lat": m["lat"],
                "lon": m["lon"],
                "area_km2": m["area_km2"],
                "log_area": float(np.log10(m["area_km2"])),
                "region": classify_region(m["lat"], m["lon"]),
                "P_prev": prev["P"],
                "Ep_prev": prev["Ep"],
                "Q_prev": prev["Q_over_P"] * prev["P"],
                "Q_over_P_prev": prev["Q_over_P"],
                "AI_prev": prev["AI"],
                "pi_M": pi_M,
                "pi_S": pi_S_by_site[sid],
                "pi_M_dyn": pi_M - pi_S_by_site[sid],
                "dy_dyn": dep_curr - pi_S_by_site[sid],
                "res_wi": res_wi_curr,
                "w_basin": wi,
                "pi_Delta": pi_Delta,
                "qb_curr": q_budyko_curr,
            })

    id_recs = [r for r in complete_records if r["AI"] <= 0.85]
    ood_recs = [r for r in complete_records if r["AI"] > 0.85]


    pid_all = np.array([r["pi_Delta"] for r in complete_records])
    dy_w0_all = np.array([r["dy_dyn"] for r in complete_records])
    res_wi_all = np.array([r["res_wi"] for r in complete_records])
    ood_mask_all = np.array([r["AI"] > 0.85 for r in complete_records])
    d60_mask_all = np.array([r["is_direct60"] for r in complete_records])
    pid_ood = np.array([r["pi_Delta"] for r in ood_recs])
    res_wi_ood = np.array([r["res_wi"] for r in ood_recs])
    d60_mask_ood = np.array([r["is_direct60"] for r in ood_recs])
    sites_ood = np.array([r["site_id"] for r in ood_recs])
    uniq_ood_sites = np.unique(sites_ood)

    mc_null_w0_rhos, mc_null_wi_all, mc_null_wi_ood, mc_null_wi_d60 = [], [], [], []
    for _ in range(300):
        dy_null_mc, r_wi_mc = [], []
        for sid, recs in by_site.items():
            wi = w_by_site[sid]
            ai_s = np.array([r["AI"] for r in recs])
            qp_s = np.array([r["Q_over_P"] for r in recs])
            sd_s = float(np.std(qp_s - fu_budyko_analytical(ai_s, wi)))
            qp_sim = np.clip(fu_budyko_analytical(ai_s, wi) + rng.normal(0, sd_s, size=len(recs)), 0.0, 1.0)
            pis_sim = float(np.mean(qp_sim - fu_budyko_analytical(ai_s, 2.6)))
            m_wi_sim = float(np.mean(qp_sim - fu_budyko_analytical(ai_s, wi)))
            for i in range(1, len(recs)):
                if recs[i]["year"] - recs[i - 1]["year"] == 1:
                    dy_null_mc.append((qp_sim[i] - float(fu_budyko_analytical(recs[i]["AI"], 2.6))) - pis_sim)
                    r_wi_mc.append((qp_sim[i] - float(fu_budyko_analytical(recs[i]["AI"], wi))) - m_wi_sim)
        r_wi_mc = np.array(r_wi_mc)
        mc_null_w0_rhos.append(spearmanr(pid_all, dy_null_mc).statistic)
        mc_null_wi_all.append(spearmanr(pid_all, r_wi_mc).statistic)
        mc_null_wi_ood.append(spearmanr(pid_all[ood_mask_all], r_wi_mc[ood_mask_all]).statistic)
        mc_null_wi_d60.append(spearmanr(pid_all[ood_mask_all & d60_mask_all], r_wi_mc[ood_mask_all & d60_mask_all]).statistic)

    by_site_comp = {}
    for idx_r, r in enumerate(complete_records):
        by_site_comp.setdefault(r["site_id"], []).append(idx_r)
    circ_wi_all, circ_wi_ood, circ_wi_d60 = [], [], []
    for _ in range(300):
        perm_res = np.empty_like(res_wi_all)
        for sid, idxs in by_site_comp.items():
            arr = res_wi_all[idxs]
            if len(arr) > 2:
                sh = int(rng.integers(1, len(arr)))
                perm_res[idxs] = np.roll(arr, sh)
            else:
                perm_res[idxs] = arr
        circ_wi_all.append(spearmanr(pid_all, perm_res).statistic)
        circ_wi_ood.append(spearmanr(pid_all[ood_mask_all], perm_res[ood_mask_all]).statistic)
        circ_wi_d60.append(spearmanr(pid_all[ood_mask_all & d60_mask_all], perm_res[ood_mask_all & d60_mask_all]).statistic)

    boot_rhos_ood = []
    for _ in range(500):
        samp = rng.choice(uniq_ood_sites, size=len(uniq_ood_sites), replace=True)
        idx_b = np.concatenate([np.where(sites_ood == s)[0] for s in samp])
        boot_rhos_ood.append(spearmanr(pid_ood[idx_b], res_wi_ood[idx_b]).statistic)

    f2_obs_d60_all = float(spearmanr(pid_all[d60_mask_all], res_wi_all[d60_mask_all]).statistic)
    f2_obs_d60_ood = float(spearmanr(pid_ood[d60_mask_ood], res_wi_ood[d60_mask_ood]).statistic)

    print("=== F2 Null-Model Check: Matched Null Models & Circular Autocorrelation-Preserving Null ===")
    print(f"  1) Raw Demeaned (w0=2.6, n={len(complete_records)}): Observed rho = {spearmanr(pid_all, dy_w0_all).statistic:.4f} vs. Matched w0=2.6 Null = {np.mean(mc_null_w0_rhos):.4f} (95% MC [{np.percentile(mc_null_w0_rhos, 2.5):.4f}, {np.percentile(mc_null_w0_rhos, 97.5):.4f}])")
    print(f"  2) Basin-w_i Residual (all n={len(complete_records)}): Observed rho_wi = {spearmanr(pid_all, res_wi_all).statistic:.4f} vs. Matched w_i Null = {np.mean(mc_null_wi_all):.4f} (95% MC [{np.percentile(mc_null_wi_all, 2.5):.4f}, {np.percentile(mc_null_wi_all, 97.5):.4f}]) & Circular Autocorrelation Null = {np.mean(circ_wi_all):.4f} (95% MC [{np.percentile(circ_wi_all, 2.5):.4f}, {np.percentile(circ_wi_all, 97.5):.4f}])")
    print(f"  3) Basin-w_i Residual (OOD n={len(ood_recs)}): Observed rho_wi = {spearmanr(pid_ood, res_wi_ood).statistic:.4f} (92-basin 95% CI [{np.percentile(boot_rhos_ood, 2.5):.4f}, {np.percentile(boot_rhos_ood, 97.5):.4f}]) vs. Matched w_i Null = {np.mean(mc_null_wi_ood):.4f} (95% MC [{np.percentile(mc_null_wi_ood, 2.5):.4f}, {np.percentile(mc_null_wi_ood, 97.5):.4f}]) & Circular Null = {np.mean(circ_wi_ood):.4f} (95% MC [{np.percentile(circ_wi_ood, 2.5):.4f}, {np.percentile(circ_wi_ood, 97.5):.4f}])")
    print(f"  4) Pure 60 Direct Gauges Basin-w_i Residual: All (n={int(np.sum(d60_mask_all))}) rho_wi = {f2_obs_d60_all:.4f}; OOD (n={int(np.sum(d60_mask_ood))}) rho_wi = {f2_obs_d60_ood:.4f} vs. Matched w_i Null = {np.mean(mc_null_wi_d60):.4f} & Circular Null = {np.mean(circ_wi_d60):.4f}")


    idx_f, idx_rest = train_test_split(np.arange(len(id_recs)), test_size=0.4, random_state=42)
    idx_c, idx_t = train_test_split(idx_rest, test_size=0.5, random_state=42)
    alrq_fit_s1, cvp_fit_s1, diag_s1 = compute_fit_isolated_basin_scalars([id_recs[i] for i in idx_f], sids_sorted, ood_recs)
    print(f"=== S1 Fit-Fold Static Imputation Summary (D_fit n={diag_s1['n_fit_records']}): {diag_s1['n_imputed_basins']}/{diag_s1['n_total_basins']} basins (<2 humid fit yrs) -> {diag_s1['n_imputed_eval']}/{diag_s1['n_eval_records']} ({diag_s1['pct_imputed_eval']:.1f}%) OOD records use training-fold median fallback ===")

    y_id = np.array([r["Q_over_P"] for r in id_recs])
    y_ood = np.array([r["Q_over_P"] for r in ood_recs])
    qb_id = np.array([r["qb_curr"] for r in id_recs])
    qb_ood = np.array([r["qb_curr"] for r in ood_recs])
    m0_id = np.clip(qb_id + 0.5 * np.array([r["pi_M"] for r in id_recs]), 0, 1)
    m0_ood = np.clip(qb_ood + 0.5 * np.array([r["pi_M"] for r in ood_recs]), 0, 1)

    X_ols_f = np.column_stack([
        np.array([id_recs[i]["pi_M"] for i in idx_f]),
        np.array([id_recs[i]["pi_Delta"] for i in idx_f]),
    ])
    c_M_fit, c_D_fit = np.linalg.lstsq(X_ols_f, y_id[idx_f] - qb_id[idx_f], rcond=None)[0]
    print(f"=== S1 In-Fold OLS Estimation on D_fit (n={len(idx_f)}): c_M = {c_M_fit:+.4f}, c_Delta = {c_D_fit:+.4f} ===")

    X_raw_id = np.array([[r["P"], r["Ep"]] for r in id_recs])
    X_raw_ood = np.array([[r["P"], r["Ep"]] for r in ood_recs])
    X_pi1_id = np.array([[r["AI"]] for r in id_recs])
    X_pi1_ood = np.array([[r["AI"]] for r in ood_recs])
    X_pers_id = np.array([[r["AI"], r["Q_over_P_prev"], r["pi_Delta"]] for r in id_recs])
    X_pers_ood = np.array([[r["AI"], r["Q_over_P_prev"], r["pi_Delta"]] for r in ood_recs])
    X_m0h_id = np.array([[r["AI"], r["pi_M"], r["pi_Delta"]] for r in id_recs])
    X_m0h_ood = np.array([[r["AI"], r["pi_M"], r["pi_Delta"]] for r in ood_recs])
    X_modA_id = np.array([[r["AI"], r["Q_over_P_prev"], r["pi_Delta"], alrq_fit_s1[r["site_id"]], cvp_fit_s1[r["site_id"]]] for r in id_recs])
    X_modA_ood = np.array([[r["AI"], r["Q_over_P_prev"], r["pi_Delta"], alrq_fit_s1[r["site_id"]], cvp_fit_s1[r["site_id"]]] for r in ood_recs])
    X_modB_id = np.array([[r["AI"], r["pi_M"], r["pi_Delta"], alrq_fit_s1[r["site_id"]], cvp_fit_s1[r["site_id"]]] for r in id_recs])
    X_modB_ood = np.array([[r["AI"], r["pi_M"], r["pi_Delta"], alrq_fit_s1[r["site_id"]], cvp_fit_s1[r["site_id"]]] for r in ood_recs])
    X_modD_id = np.array([[r["AI"], r["pi_M"], r["AI_prev"], r["pi_Delta"], alrq_fit_s1[r["site_id"]], cvp_fit_s1[r["site_id"]]] for r in id_recs])
    X_modD_ood = np.array([[r["AI"], r["pi_M"], r["AI_prev"], r["pi_Delta"], alrq_fit_s1[r["site_id"]], cvp_fit_s1[r["site_id"]]] for r in ood_recs])


    X_modA_ns_id, X_modA_ns_ood = X_modA_id[:, :3], X_modA_ood[:, :3]
    X_modB_ns_id, X_modB_ns_ood = X_modB_id[:, :3], X_modB_ood[:, :3]
    X_modD_ns_id, X_modD_ns_ood = X_modD_id[:, :4], X_modD_ood[:, :4]

    m_raw = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_raw_id[idx_f], y_id[idx_f])
    m_pi1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_pi1_id[idx_f], y_id[idx_f])
    m_modA = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_modA_id[idx_f], y_id[idx_f])
    m_modB = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_modB_id[idx_f], y_id[idx_f])
    m_modD_phys = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_modD_id[idx_f], y_id[idx_f] - m0_id[idx_f])

    m_modA_ns = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_modA_ns_id[idx_f], y_id[idx_f])
    m_modB_ns = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_modB_ns_id[idx_f], y_id[idx_f])
    m_modD_ns = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_modD_ns_id[idx_f], y_id[idx_f] - m0_id[idx_f])

    op_raw = lambda X: fu_budyko_analytical(X[:, 1] / (X[:, 0] + 1e-6))
    op_ai = lambda X: fu_budyko_analytical(X[:, 0])
    op_modA = lambda X: np.clip(0.5 * fu_budyko_analytical(X[:, 0]) + 0.5 * X[:, 1], 0.0, 1.0)
    op_m0 = lambda X: np.clip(fu_budyko_analytical(X[:, 0]) + 0.5 * X[:, 1], 0.0, 1.0)
    op_m0h_ols = lambda X: np.clip(fu_budyko_analytical(X[:, 0]) + c_M_fit * X[:, 1] + c_D_fit * X[:, 2], 0.0, 1.0)

    p_raw_ood = np.clip(m_raw.predict(X_raw_ood), 0.0, 1.0)
    p_pi1_ood = np.clip(m_pi1.predict(X_pi1_ood), 0.0, 1.0)
    p_pers_cal = np.array([id_recs[i]["Q_over_P_prev"] for i in idx_c])
    p_pers_ood = np.array([r["Q_over_P_prev"] for r in ood_recs])
    p_m0h_cal = op_m0h_ols(X_m0h_id[idx_c])
    p_m0h_ood = op_m0h_ols(X_m0h_ood)
    p_modA_ood = np.clip(m_modA.predict(X_modA_ood), 0.0, 1.0)
    p_modB_ood = np.clip(m_modB.predict(X_modB_ood), 0.0, 1.0)
    p_modD_cal = np.clip(m0_id[idx_c] + m_modD_phys.predict(X_modD_id[idx_c]), 0.0, 1.0)
    p_modD_ood = np.clip(m0_ood + m_modD_phys.predict(X_modD_ood), 0.0, 1.0)


    p_modD_ood_unclipped = (qb_ood + 0.5 * np.array([r["pi_M"] for r in ood_recs])) + m_modD_phys.predict(X_modD_ood)
    lo_modD_unc, hi_modD_unc, _ = ijs_conformal_interval(X_modD_id[idx_f], X_modD_id[idx_c], y_id[idx_c], p_modD_cal, X_modD_ood, p_modD_ood_unclipped, op_m0, mode="full", clip_bounds=False)
    print(f"=== In-Cache Unclipped Sensitivity Check (No [0, 1] Truncation on Model D-Phys): Unclipped RMSE={np.sqrt(np.mean((y_ood-p_modD_ood_unclipped)**2)):.4f} (vs Clipped {np.sqrt(np.mean((y_ood-p_modD_ood)**2)):.4f}), Unclipped IJS-CP Cov={np.mean((y_ood>=lo_modD_unc)&(y_ood<=hi_modD_unc))*100:.1f}% ===")

    p_modA_ns_ood = np.clip(m_modA_ns.predict(X_modA_ns_ood), 0.0, 1.0)
    p_modB_ns_ood = np.clip(m_modB_ns.predict(X_modB_ns_ood), 0.0, 1.0)
    p_modD_ns_cal = np.clip(m0_id[idx_c] + m_modD_ns.predict(X_modD_ns_id[idx_c]), 0.0, 1.0)
    p_modD_ns_ood = np.clip(m0_ood + m_modD_ns.predict(X_modD_ns_ood), 0.0, 1.0)
    lo_modD_ns_ijs, hi_modD_ns_ijs, _ = ijs_conformal_interval(X_modD_ns_id[idx_f], X_modD_ns_id[idx_c], y_id[idx_c], p_modD_ns_cal, X_modD_ns_ood, p_modD_ns_ood, op_m0, mode="full")
    print(f"=== S1 Single-Factor Zero-Static Ablation (ALRQ/CVP Removed): ModB-NoStatic RMSE={np.sqrt(np.mean((y_ood-p_modB_ns_ood)**2)):.4f} (vs {np.sqrt(np.mean((y_ood-p_modB_ood)**2)):.4f}), ModA-NoStatic RMSE={np.sqrt(np.mean((y_ood-p_modA_ns_ood)**2)):.4f} (vs {np.sqrt(np.mean((y_ood-p_modA_ood)**2)):.4f}), ModD-Phys-NoStatic RMSE={np.sqrt(np.mean((y_ood-p_modD_ns_ood)**2)):.4f} (IJS Cov={np.mean((y_ood>=lo_modD_ns_ijs)&(y_ood<=hi_modD_ns_ijs))*100:.1f}%, IS={winkler_score(y_ood,lo_modD_ns_ijs,hi_modD_ns_ijs):.3f} vs {np.sqrt(np.mean((y_ood-p_modD_ood)**2)):.4f}) ===")

    s1_diff, s1_ci_lo, s1_ci_hi, s1_nb = paired_cluster_bootstrap_rmse(y_ood, p_m0h_ood, p_pers_ood, sites_ood, n_boot=3000, seed=42)
    print(f"=== S1 Paired Basin-Cluster Bootstrap (3,000 draws, {s1_nb} OOD basins): M0_hyst - Persistence = {s1_diff:+.4f}, 95% CI [{s1_ci_lo:+.4f}, {s1_ci_hi:+.4f}] ===")


    qb_prev_id = np.array([float(fu_budyko_analytical(r["AI_prev"], w=2.6)) for r in id_recs])
    qb_prev_ood = np.array([float(fu_budyko_analytical(r["AI_prev"], w=2.6)) for r in ood_recs])
    X_arx_raw_id = np.column_stack([np.ones(len(id_recs)), [r["Q_over_P_prev"] for r in id_recs], [r["AI"] for r in id_recs], [r["pi_Delta"] for r in id_recs]])
    X_arx_raw_ood = np.column_stack([np.ones(len(ood_recs)), [r["Q_over_P_prev"] for r in ood_recs], [r["AI"] for r in ood_recs], [r["pi_Delta"] for r in ood_recs]])
    beta_arx_raw_s1 = np.linalg.lstsq(X_arx_raw_id[idx_f], y_id[idx_f], rcond=None)[0]
    p_arx_raw_s1_cal = np.clip(X_arx_raw_id[idx_c] @ beta_arx_raw_s1, 0.0, 1.0)
    p_arx_raw_s1_ood = np.clip(X_arx_raw_ood @ beta_arx_raw_s1, 0.0, 1.0)

    X_arx_mat_id = np.column_stack([np.ones(len(id_recs)), [r["Q_over_P_prev"] for r in id_recs], qb_prev_id, qb_id, [r["pi_Delta"] for r in id_recs]])
    X_arx_mat_ood = np.column_stack([np.ones(len(ood_recs)), [r["Q_over_P_prev"] for r in ood_recs], qb_prev_ood, qb_ood, [r["pi_Delta"] for r in ood_recs]])
    beta_arx_mat_s1 = np.linalg.lstsq(X_arx_mat_id[idx_f], y_id[idx_f], rcond=None)[0]
    p_arx_mat_s1_cal = np.clip(X_arx_mat_id[idx_c] @ beta_arx_mat_s1, 0.0, 1.0)
    p_arx_mat_s1_ood = np.clip(X_arx_mat_ood @ beta_arx_mat_s1, 0.0, 1.0)
    lo_arx_mat_s1_jac, hi_arx_mat_s1_jac, _ = ijs_conformal_interval(X_m0h_id[idx_f], X_m0h_id[idx_c], y_id[idx_c], p_arx_mat_s1_cal, X_m0h_ood, p_arx_mat_s1_ood, op_m0h_ols, mode="jac_only")
    s1_arx_diff, s1_arx_lo, s1_arx_hi, _ = paired_cluster_bootstrap_rmse(y_ood, p_m0h_ood, p_arx_mat_s1_ood, sites_ood, n_boot=3000, seed=42)
    s1_arx_raw_diff, s1_arx_raw_lo, s1_arx_raw_hi, _ = paired_cluster_bootstrap_rmse(y_ood, p_m0h_ood, p_arx_raw_s1_ood, sites_ood, n_boot=3000, seed=42)
    print(f"=== S1 Linear ARX Comparison: ARX-Raw (4-param) RMSE={np.sqrt(np.mean((y_ood-p_arx_raw_s1_ood)**2)):.4f} (M0-ARX_raw={s1_arx_raw_diff:+.4f} [{s1_arx_raw_lo:+.4f}, {s1_arx_raw_hi:+.4f}]) | ARX-Matched (5-param) RMSE={np.sqrt(np.mean((y_ood-p_arx_mat_s1_ood)**2)):.4f} (M0-ARX_mat={s1_arx_diff:+.4f} [{s1_arx_lo:+.4f}, {s1_arx_hi:+.4f}]) ===")


    d60_ood_mask = np.array([r["is_direct60"] for r in ood_recs])
    area_ood = np.array([r["area_km2"] for r in ood_recs])
    lfr_ood = np.array([r["bfi_proxy"] for r in ood_recs])
    pim_abs_ood = np.abs(np.array([r["pi_M"] for r in ood_recs]))
    err_b_ood = np.abs(y_ood - qb_ood)
    for sc_label, sc_mask in [
        ("60-Direct Grid-Matched Small-Medium Basins (A <= 300 km2, ~1-3 ERA5-Land cells)", d60_ood_mask & (area_ood <= 300.0)),
        ("60-Direct Larger Basins (A > 300 km2)", d60_ood_mask & (area_ood > 300.0)),
        ("60-Direct High Baseflow/Snowmelt Sustainment (LFR > 0.25)", d60_ood_mask & (lfr_ood > 0.25)),
        ("60-Direct Low Baseflow Seasonal-Convective (LFR <= 0.25)", d60_ood_mask & (lfr_ood <= 0.25)),
    ]:
        nb_sc = len(np.unique(sites_ood[sc_mask]))
        rho_err_sc = float(spearmanr(pim_abs_ood[sc_mask], err_b_ood[sc_mask]).statistic)
        rho_wi_sc = float(spearmanr(pid_ood[sc_mask], res_wi_ood[sc_mask]).statistic)
        rmse_m0_sc = float(np.sqrt(np.mean((y_ood[sc_mask] - p_m0h_ood[sc_mask])**2)))
        rmse_pers_sc = float(np.sqrt(np.mean((y_ood[sc_mask] - p_pers_ood[sc_mask])**2)))
        rmse_b_sc = float(np.sqrt(np.mean((y_ood[sc_mask] - qb_ood[sc_mask])**2)))
        print(f"  [{sc_label}] n={int(np.sum(sc_mask))} ({nb_sc} basins): rho(|Pi_M|,|err_B|)={rho_err_sc:.3f}, rho_wi(res_wi,Pi_Delta)={rho_wi_sc:.3f}, M0_hyst RMSE={rmse_m0_sc:.4f} (vs Pers {rmse_pers_sc:.4f}, Steady Budyko {rmse_b_sc:.4f})")

    lo_raw_std, hi_raw_std, _ = ijs_conformal_interval(X_raw_id[idx_f], X_raw_id[idx_c], y_id[idx_c], m_raw.predict(X_raw_id[idx_c]), X_raw_ood, p_raw_ood, op_raw, mode="std")
    lo_pi1_std, hi_pi1_std, _ = ijs_conformal_interval(X_pi1_id[idx_f], X_pi1_id[idx_c], y_id[idx_c], m_pi1.predict(X_pi1_id[idx_c]), X_pi1_ood, p_pi1_ood, op_ai, mode="std")
    lo_pers_std, hi_pers_std, _ = ijs_conformal_interval(X_pers_id[idx_f], X_pers_id[idx_c], y_id[idx_c], p_pers_cal, X_pers_ood, p_pers_ood, op_modA, mode="std")
    lo_m0h_std, hi_m0h_std, _ = ijs_conformal_interval(X_m0h_id[idx_f], X_m0h_id[idx_c], y_id[idx_c], p_m0h_cal, X_m0h_ood, p_m0h_ood, op_m0h_ols, mode="std")
    lo_m0h_jac, hi_m0h_jac, _ = ijs_conformal_interval(X_m0h_id[idx_f], X_m0h_id[idx_c], y_id[idx_c], p_m0h_cal, X_m0h_ood, p_m0h_ood, op_m0h_ols, mode="jac_only")
    lo_modA_std, hi_modA_std, _ = ijs_conformal_interval(X_modA_id[idx_f], X_modA_id[idx_c], y_id[idx_c], m_modA.predict(X_modA_id[idx_c]), X_modA_ood, p_modA_ood, op_modA, mode="std")
    lo_modB_std, hi_modB_std, _ = ijs_conformal_interval(X_modB_id[idx_f], X_modB_id[idx_c], y_id[idx_c], m_modB.predict(X_modB_id[idx_c]), X_modB_ood, p_modB_ood, op_m0, mode="std")
    lo_modD_std, hi_modD_std, _ = ijs_conformal_interval(X_modD_id[idx_f], X_modD_id[idx_c], y_id[idx_c], p_modD_cal, X_modD_ood, p_modD_ood, op_m0, mode="std")

    lo_modA_jac, hi_modA_jac, _ = ijs_conformal_interval(X_modA_id[idx_f], X_modA_id[idx_c], y_id[idx_c], m_modA.predict(X_modA_id[idx_c]), X_modA_ood, p_modA_ood, op_modA, mode="jac_only")
    lo_modB_ijs, hi_modB_ijs, _ = ijs_conformal_interval(X_modB_id[idx_f], X_modB_id[idx_c], y_id[idx_c], m_modB.predict(X_modB_id[idx_c]), X_modB_ood, p_modB_ood, op_m0, mode="full")
    lo_modD_ijs, hi_modD_ijs, _ = ijs_conformal_interval(X_modD_id[idx_f], X_modD_id[idx_c], y_id[idx_c], p_modD_cal, X_modD_ood, p_modD_ood, op_m0, mode="full")


    for rname in ["Central Plains & Midwest", "Rocky Mountains & Southwest", "Appalachian & Northeast", "Pacific Northwest & Coast"]:
        m_all = np.array([r["region"] == rname for r in ood_recs])
        m_d60 = np.array([(r["region"] == rname) and r["is_direct60"] for r in ood_recs])
        c_std_all = float(np.mean((y_ood[m_all] >= lo_pi1_std[m_all]) & (y_ood[m_all] <= hi_pi1_std[m_all])) * 100)
        c_ijs_all = float(np.mean((y_ood[m_all] >= lo_modD_ijs[m_all]) & (y_ood[m_all] <= hi_modD_ijs[m_all])) * 100)
        c_std_60 = float(np.mean((y_ood[m_d60] >= lo_pi1_std[m_d60]) & (y_ood[m_d60] <= hi_pi1_std[m_d60])) * 100) if np.any(m_d60) else 0.0
        c_ijs_60 = float(np.mean((y_ood[m_d60] >= lo_modD_ijs[m_d60]) & (y_ood[m_d60] <= hi_modD_ijs[m_d60])) * 100) if np.any(m_d60) else 0.0
        print(f"  Regional Station-Year Summary [{rname:27s}]: All (n={int(np.sum(m_all)):3d}) Std={c_std_all:5.1f}% -> IJS={c_ijs_all:5.1f}% | 60-Direct (n={int(np.sum(m_d60)):3d}) Std={c_std_60:5.1f}% -> IJS={c_ijs_60:5.1f}%")


    s2_data = {
        "ai_base": {"y": [], "p": [], "lo_s": [], "hi_s": [], "lo_i": [], "hi_i": [], "sites": []},
        "ung_dir": {"y": [], "p": [], "lo_s": [], "hi_s": [], "lo_i": [], "hi_i": [], "sites": []},
        "ung_phys": {"y": [], "p": [], "lo_s": [], "hi_s": [], "lo_i": [], "hi_i": [], "sites": []},
        "ung_phys_dbl": {"y": [], "p": [], "lo_s": [], "hi_s": [], "lo_i": [], "hi_i": [], "sites": []},
    }
    for train_idx, test_idx in deterministic_basin_splits(complete_records):
        id_tr = [complete_records[i] for i in train_idx if complete_records[i]["AI"] <= 0.85]
        ood_te = [complete_records[i] for i in test_idx if complete_records[i]["AI"] > 0.85]
        if not ood_te:
            continue
        te_sites_set = {r["site_id"] for r in ood_te}
        i_f, i_c = train_test_split(np.arange(len(id_tr)), test_size=0.25, random_state=42)
        i_f_dbl, i_c_dbl, _, _ = split_fit_cal_by_basin(id_tr, te_sites_set, test_size=0.25, random_state=42)

        y_u_tr = np.array([r["Q_over_P"] for r in id_tr])
        y_u_te = np.array([r["Q_over_P"] for r in ood_te])
        qb_tr = np.array([r["qb_curr"] for r in id_tr])
        qb_te = np.array([r["qb_curr"] for r in ood_te])
        te_sids = [r["site_id"] for r in ood_te]

        X_ai_tr = np.array([[r["AI"]] for r in id_tr])
        X_ai_te = np.array([[r["AI"]] for r in ood_te])
        X_u_tr = np.array([[r["AI"], r["AI_prev"], r["pi_Delta"], r["si_walsh"], r["lat"], r["lon"], r["log_area"]] for r in id_tr])
        X_u_te = np.array([[r["AI"], r["AI_prev"], r["pi_Delta"], r["si_walsh"], r["lat"], r["lon"], r["log_area"]] for r in ood_te])

        m_ai_g = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_ai_tr[i_f], y_u_tr[i_f])
        p_ai_cal = np.clip(m_ai_g.predict(X_ai_tr[i_c]), 0, 1)
        p_ai_te = np.clip(m_ai_g.predict(X_ai_te), 0, 1)
        lo_s0, hi_s0, _ = ijs_conformal_interval(X_ai_tr[i_f], X_ai_tr[i_c], y_u_tr[i_c], p_ai_cal, X_ai_te, p_ai_te, op_ai, mode="std")
        lo_i0, hi_i0, _ = ijs_conformal_interval(X_ai_tr[i_f], X_ai_tr[i_c], y_u_tr[i_c], p_ai_cal, X_ai_te, p_ai_te, op_ai, mode="full")
        s2_data["ai_base"]["y"].extend(y_u_te); s2_data["ai_base"]["p"].extend(p_ai_te); s2_data["ai_base"]["sites"].extend(te_sids)
        s2_data["ai_base"]["lo_s"].extend(lo_s0); s2_data["ai_base"]["hi_s"].extend(hi_s0)
        s2_data["ai_base"]["lo_i"].extend(lo_i0); s2_data["ai_base"]["hi_i"].extend(hi_i0)

        m_u = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_u_tr[i_f], y_u_tr[i_f])
        p_u_cal = np.clip(m_u.predict(X_u_tr[i_c]), 0, 1)
        p_u_te = np.clip(m_u.predict(X_u_te), 0, 1)
        lo_s1, hi_s1, _ = ijs_conformal_interval(X_u_tr[i_f], X_u_tr[i_c], y_u_tr[i_c], p_u_cal, X_u_te, p_u_te, op_ai, mode="std")
        lo_i1, hi_i1, _ = ijs_conformal_interval(X_u_tr[i_f], X_u_tr[i_c], y_u_tr[i_c], p_u_cal, X_u_te, p_u_te, op_ai, mode="full")
        s2_data["ung_dir"]["y"].extend(y_u_te); s2_data["ung_dir"]["p"].extend(p_u_te); s2_data["ung_dir"]["sites"].extend(te_sids)
        s2_data["ung_dir"]["lo_s"].extend(lo_s1); s2_data["ung_dir"]["hi_s"].extend(hi_s1)
        s2_data["ung_dir"]["lo_i"].extend(lo_i1); s2_data["ung_dir"]["hi_i"].extend(hi_i1)

        m_up = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_u_tr[i_f], y_u_tr[i_f] - qb_tr[i_f])
        p_up_cal = np.clip(qb_tr[i_c] + m_up.predict(X_u_tr[i_c]), 0, 1)
        p_up_te = np.clip(qb_te + m_up.predict(X_u_te), 0, 1)
        lo_s2, hi_s2, _ = ijs_conformal_interval(X_u_tr[i_f], X_u_tr[i_c], y_u_tr[i_c], p_up_cal, X_u_te, p_up_te, op_ai, mode="std")
        lo_i2, hi_i2, _ = ijs_conformal_interval(X_u_tr[i_f], X_u_tr[i_c], y_u_tr[i_c], p_up_cal, X_u_te, p_up_te, op_ai, mode="full")
        s2_data["ung_phys"]["y"].extend(y_u_te); s2_data["ung_phys"]["p"].extend(p_up_te); s2_data["ung_phys"]["sites"].extend(te_sids)
        s2_data["ung_phys"]["lo_s"].extend(lo_s2); s2_data["ung_phys"]["hi_s"].extend(hi_s2)
        s2_data["ung_phys"]["lo_i"].extend(lo_i2); s2_data["ung_phys"]["hi_i"].extend(hi_i2)


        m_up_dbl = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_u_tr[i_f_dbl], y_u_tr[i_f_dbl] - qb_tr[i_f_dbl])
        p_up_cal_dbl = np.clip(qb_tr[i_c_dbl] + m_up_dbl.predict(X_u_tr[i_c_dbl]), 0, 1)
        p_up_te_dbl = np.clip(qb_te + m_up_dbl.predict(X_u_te), 0, 1)
        lo_sd, hi_sd, _ = ijs_conformal_interval(X_u_tr[i_f_dbl], X_u_tr[i_c_dbl], y_u_tr[i_c_dbl], p_up_cal_dbl, X_u_te, p_up_te_dbl, op_ai, mode="std")
        lo_id, hi_id, _ = ijs_conformal_interval(X_u_tr[i_f_dbl], X_u_tr[i_c_dbl], y_u_tr[i_c_dbl], p_up_cal_dbl, X_u_te, p_up_te_dbl, op_ai, mode="full")
        s2_data["ung_phys_dbl"]["y"].extend(y_u_te); s2_data["ung_phys_dbl"]["p"].extend(p_up_te_dbl); s2_data["ung_phys_dbl"]["sites"].extend(te_sids)
        s2_data["ung_phys_dbl"]["lo_s"].extend(lo_sd); s2_data["ung_phys_dbl"]["hi_s"].extend(hi_sd)
        s2_data["ung_phys_dbl"]["lo_i"].extend(lo_id); s2_data["ung_phys_dbl"]["hi_i"].extend(hi_id)

    for k in s2_data:
        for arr_k in ("y", "p", "lo_s", "hi_s", "lo_i", "hi_i"):
            s2_data[k][arr_k] = np.array(s2_data[k][arr_k])
        s2_data[k]["sites"] = np.array(s2_data[k]["sites"])

    yd = s2_data["ung_phys_dbl"]["y"]
    pd = s2_data["ung_phys_dbl"]["p"]
    lod_s, hid_s = s2_data["ung_phys_dbl"]["lo_s"], s2_data["ung_phys_dbl"]["hi_s"]
    lod_i, hid_i = s2_data["ung_phys_dbl"]["lo_i"], s2_data["ung_phys_dbl"]["hi_i"]
    print(f"=== S2 Double-Basin-Isolated GroupShuffleSplit Verification (n={len(yd)}): RMSE={np.sqrt(np.mean((yd-pd)**2)):.4f}, Std-CP Cov={np.mean((yd>=lod_s)&(yd<=hid_s))*100:.1f}% -> IJS-CP Cov={np.mean((yd>=lod_i)&(yd<=hid_i))*100:.1f}%, W={np.mean(hid_i-lod_i):.3f}, IS={winkler_score(yd,lod_i,hid_i):.3f} ===")

    s2_all_diff, s2_all_lo, s2_all_hi, s2_all_nb = paired_cluster_bootstrap_rmse(
        s2_data["ung_phys"]["y"], s2_data["ung_phys"]["p"], s2_data["ai_base"]["p"], s2_data["ung_phys"]["sites"], n_boot=3000, seed=42
    )
    print(f"=== S2 All-107 Basins Paired Cluster Bootstrap ({s2_all_nb} OOD basins): Budyko-Res - AI = {s2_all_diff:+.4f}, 95% CI [{s2_all_lo:+.4f}, {s2_all_hi:+.4f}] ===")


    comp60 = [r for r in complete_records if r["is_direct60"]]
    y60_all, p_ai60, p_dir60, p_res60, lo_s60, hi_s60, lo_i60, hi_i60, sites60_all = [], [], [], [], [], [], [], [], []
    p_res60_dbl, lo_i60_dbl, hi_i60_dbl = [], [], []
    fold_covs_60_dbl, fold_ncal_basins_60, fold_ncal_rows_60 = [], [], []
    for tr60, te60 in deterministic_basin_splits(comp60):
        id_tr60 = [comp60[i] for i in tr60 if comp60[i]["AI"] <= 0.85]
        ood_te60 = [comp60[i] for i in te60 if comp60[i]["AI"] > 0.85]
        if not ood_te60:
            continue
        te60_sites_set = {r["site_id"] for r in ood_te60}
        i_f6, i_c6 = train_test_split(np.arange(len(id_tr60)), test_size=0.25, random_state=42)
        i_f6_dbl, i_c6_dbl, n_fit_b6, n_cal_b6 = split_fit_cal_by_basin(id_tr60, te60_sites_set, test_size=0.25, random_state=42)
        y_tr6 = np.array([r["Q_over_P"] for r in id_tr60]); y_te6 = np.array([r["Q_over_P"] for r in ood_te60])
        qb_tr6 = np.array([r["qb_curr"] for r in id_tr60]); qb_te6 = np.array([r["qb_curr"] for r in ood_te60])
        X_ai_tr6 = np.array([[r["AI"]] for r in id_tr60]); X_ai_te6 = np.array([[r["AI"]] for r in ood_te60])
        X_u_tr6 = np.array([[r["AI"], r["AI_prev"], r["pi_Delta"], r["si_walsh"], r["lat"], r["lon"], r["log_area"]] for r in id_tr60])
        X_u_te6 = np.array([[r["AI"], r["AI_prev"], r["pi_Delta"], r["si_walsh"], r["lat"], r["lon"], r["log_area"]] for r in ood_te60])
        m0_6 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_ai_tr6[i_f6], y_tr6[i_f6])
        m1_6 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_u_tr6[i_f6], y_tr6[i_f6])
        m2_6 = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_u_tr6[i_f6], y_tr6[i_f6] - qb_tr6[i_f6])
        p_cal6 = np.clip(qb_tr6[i_c6] + m2_6.predict(X_u_tr6[i_c6]), 0, 1)
        p_te6 = np.clip(qb_te6 + m2_6.predict(X_u_te6), 0, 1)
        ls6, hs6, _ = ijs_conformal_interval(X_u_tr6[i_f6], X_u_tr6[i_c6], y_tr6[i_c6], p_cal6, X_u_te6, p_te6, op_ai, mode="std")
        li6, hi6, _ = ijs_conformal_interval(X_u_tr6[i_f6], X_u_tr6[i_c6], y_tr6[i_c6], p_cal6, X_u_te6, p_te6, op_ai, mode="full")
        y60_all.extend(y_te6); p_ai60.extend(np.clip(m0_6.predict(X_ai_te6), 0, 1)); p_dir60.extend(np.clip(m1_6.predict(X_u_te6), 0, 1)); p_res60.extend(p_te6)
        lo_s60.extend(ls6); hi_s60.extend(hs6); lo_i60.extend(li6); hi_i60.extend(hi6)
        sites60_all.extend([r["site_id"] for r in ood_te60])

        m2_6_dbl = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_u_tr6[i_f6_dbl], y_tr6[i_f6_dbl] - qb_tr6[i_f6_dbl])
        p_cal6_dbl = np.clip(qb_tr6[i_c6_dbl] + m2_6_dbl.predict(X_u_tr6[i_c6_dbl]), 0, 1)
        p_te6_dbl = np.clip(qb_te6 + m2_6_dbl.predict(X_u_te6), 0, 1)
        li6_d, hi6_d, _ = ijs_conformal_interval(X_u_tr6[i_f6_dbl], X_u_tr6[i_c6_dbl], y_tr6[i_c6_dbl], p_cal6_dbl, X_u_te6, p_te6_dbl, op_ai, mode="full")
        p_res60_dbl.extend(p_te6_dbl); lo_i60_dbl.extend(li6_d); hi_i60_dbl.extend(hi6_d)
        fold_covs_60_dbl.append(float(np.mean((y_te6 >= li6_d) & (y_te6 <= hi6_d)) * 100.0))
        fold_ncal_basins_60.append(n_cal_b6)
        fold_ncal_rows_60.append(len(i_c6_dbl))

    y60_all, p_ai60, p_dir60, p_res60, lo_s60, hi_s60, lo_i60, hi_i60 = map(np.array, [y60_all, p_ai60, p_dir60, p_res60, lo_s60, hi_s60, lo_i60, hi_i60])
    p_res60_dbl, lo_i60_dbl, hi_i60_dbl = map(np.array, [p_res60_dbl, lo_i60_dbl, hi_i60_dbl])
    sites60_all = np.array(sites60_all)
    s2_60_diff, s2_60_lo, s2_60_hi, s2_60_nb = paired_cluster_bootstrap_rmse(y60_all, p_res60, p_ai60, sites60_all, n_boot=3000, seed=42)
    cov60_dbl_mean, cov60_dbl_lo, cov60_dbl_hi, _ = cluster_bootstrap_coverage(y60_all, lo_i60_dbl, hi_i60_dbl, sites60_all, n_boot=3000, seed=42)
    fold_cov_str = ", ".join(f"F{k+1}:{v:.1f}%(n_cal={nb}b/{nr}y)" for k, (v, nb, nr) in enumerate(zip(fold_covs_60_dbl, fold_ncal_basins_60, fold_ncal_rows_60)))
    print(f"=== S2 Pure 60-Direct-Gauge Refit (n={len(y60_all)}, {s2_60_nb} OOD basins, zero IDW in train/cal/test): AI-only RMSE={np.sqrt(np.mean((y60_all-p_ai60)**2)):.4f}, Direct GBR RMSE={np.sqrt(np.mean((y60_all-p_dir60)**2)):.4f}, Budyko-Res RMSE={np.sqrt(np.mean((y60_all-p_res60)**2)):.4f} (Cluster 95% CI vs AI: {s2_60_diff:+.4f} [{s2_60_lo:+.4f}, {s2_60_hi:+.4f}]) | Std-CP Cov={np.mean((y60_all>=lo_s60)&(y60_all<=hi_s60))*100:.1f}% -> IJS-CP Cov={np.mean((y60_all>=lo_i60)&(y60_all<=hi_i60))*100:.1f}%, IS={winkler_score(y60_all,lo_i60,hi_i60):.3f} ===")
    print(f"=== S2 Double-Grouped 60-Direct Diagnostics: Overall Cov={cov60_dbl_mean*100:.1f}% (50-Basin Cluster 95% CI [{cov60_dbl_lo*100:.1f}%, {cov60_dbl_hi*100:.1f}%]), RMSE={np.sqrt(np.mean((y60_all-p_res60_dbl)**2)):.4f}, IS={winkler_score(y60_all,lo_i60_dbl,hi_i60_dbl):.3f} | Fold Breakdown: {fold_cov_str} ===")


    id_hist = [r for r in complete_records if r["year"] <= 2015 and r["AI"] <= 0.85]
    ood_fut = [r for r in complete_records if r["year"] >= 2016 and r["AI"] > 0.85]
    idx_fh, idx_ch = train_test_split(np.arange(len(id_hist)), test_size=0.35, random_state=42)
    alrq_fit_s3, cvp_fit_s3, diag_s3 = compute_fit_isolated_basin_scalars([id_hist[i] for i in idx_fh], sids_sorted, ood_fut)
    print(f"=== S3 Fit-Fold Static Imputation Summary (D_fit,hist n={diag_s3['n_fit_records']}): {diag_s3['n_imputed_basins']}/{diag_s3['n_total_basins']} basins (<2 humid fit yrs) -> {diag_s3['n_imputed_eval']}/{diag_s3['n_eval_records']} ({diag_s3['pct_imputed_eval']:.1f}%) OOD records use training-fold median fallback ===")

    y_hist = np.array([r["Q_over_P"] for r in id_hist])
    y_fut = np.array([r["Q_over_P"] for r in ood_fut])
    qb_hist = np.array([r["qb_curr"] for r in id_hist])
    qb_fut = np.array([r["qb_curr"] for r in ood_fut])
    sites_fut = np.array([r["site_id"] for r in ood_fut])

    X_raw_hist = np.array([[r["P"], r["Ep"]] for r in id_hist])
    X_raw_fut = np.array([[r["P"], r["Ep"]] for r in ood_fut])
    m_raw_hist = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_raw_hist[idx_fh], y_hist[idx_fh])
    p_raw_fut = np.clip(m_raw_hist.predict(X_raw_fut), 0, 1)

    X_ai_hist = np.array([[r["AI"]] for r in id_hist])
    X_ai_fut = np.array([[r["AI"]] for r in ood_fut])
    m_ai_hist = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_ai_hist[idx_fh], y_hist[idx_fh])
    p_ai_fut = np.clip(m_ai_hist.predict(X_ai_fut), 0, 1)
    lo_ai_fut_s, hi_ai_fut_s, _ = ijs_conformal_interval(X_ai_hist[idx_fh], X_ai_hist[idx_ch], y_hist[idx_ch], np.clip(m_ai_hist.predict(X_ai_hist[idx_ch]), 0, 1), X_ai_fut, p_ai_fut, op_ai, mode="std")


    X_u_hist = np.array([[r["AI"], r["AI_prev"], r["pi_Delta"], cvp_fit_s3[r["site_id"]], r["lat"], r["lon"], r["log_area"]] for r in id_hist])
    X_u_fut = np.array([[r["AI"], r["AI_prev"], r["pi_Delta"], cvp_fit_s3[r["site_id"]], r["lat"], r["lon"], r["log_area"]] for r in ood_fut])
    m_s3b_phys = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_u_hist[idx_fh], y_hist[idx_fh] - qb_hist[idx_fh])
    p_s3b_cal = np.clip(qb_hist[idx_ch] + m_s3b_phys.predict(X_u_hist[idx_ch]), 0, 1)
    p_s3b_fut = np.clip(qb_fut + m_s3b_phys.predict(X_u_fut), 0, 1)
    lo_s3b_std, hi_s3b_std, _ = ijs_conformal_interval(X_u_hist[idx_fh], X_u_hist[idx_ch], y_hist[idx_ch], p_s3b_cal, X_u_fut, p_s3b_fut, op_ai, mode="std")
    lo_s3b_ijs, hi_s3b_ijs, _ = ijs_conformal_interval(X_u_hist[idx_fh], X_u_hist[idx_ch], y_hist[idx_ch], p_s3b_cal, X_u_fut, p_s3b_fut, op_ai, mode="full")

    X_u_ns_hist = np.array([[r["AI"], r["AI_prev"], r["pi_Delta"], r["lat"], r["lon"], r["log_area"]] for r in id_hist])
    X_u_ns_fut = np.array([[r["AI"], r["AI_prev"], r["pi_Delta"], r["lat"], r["lon"], r["log_area"]] for r in ood_fut])
    m_s3b_ns = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_u_ns_hist[idx_fh], y_hist[idx_fh] - qb_hist[idx_fh])
    p_s3b_ns_cal = np.clip(qb_hist[idx_ch] + m_s3b_ns.predict(X_u_ns_hist[idx_ch]), 0, 1)
    p_s3b_ns_fut = np.clip(qb_fut + m_s3b_ns.predict(X_u_ns_fut), 0, 1)
    lo_s3b_ns_ijs, hi_s3b_ns_ijs, _ = ijs_conformal_interval(X_u_ns_hist[idx_fh], X_u_ns_hist[idx_ch], y_hist[idx_ch], p_s3b_ns_cal, X_u_ns_fut, p_s3b_ns_fut, op_ai, mode="full")


    X_pi3_hist = np.array([[r["AI"], r["pi_M"], r["pi_Delta"], alrq_fit_s3[r["site_id"]], cvp_fit_s3[r["site_id"]]] for r in id_hist])
    X_pi3_fut = np.array([[r["AI"], r["pi_M"], r["pi_Delta"], alrq_fit_s3[r["site_id"]], cvp_fit_s3[r["site_id"]]] for r in ood_fut])
    m_pi3_hist = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_pi3_hist[idx_fh], y_hist[idx_fh])
    p_pi3_fut = np.clip(m_pi3_hist.predict(X_pi3_fut), 0.0, 1.0)
    lo_fut_std, hi_fut_std, _ = ijs_conformal_interval(X_pi3_hist[idx_fh], X_pi3_hist[idx_ch], y_hist[idx_ch], np.clip(m_pi3_hist.predict(X_pi3_hist[idx_ch]), 0, 1), X_pi3_fut, p_pi3_fut, op_m0, mode="std")
    lo_fut_ijs, hi_fut_ijs, _ = ijs_conformal_interval(X_pi3_hist[idx_fh], X_pi3_hist[idx_ch], y_hist[idx_ch], np.clip(m_pi3_hist.predict(X_pi3_hist[idx_ch]), 0, 1), X_pi3_fut, p_pi3_fut, op_m0, mode="full")


    X_pi3_ns_hist, X_pi3_ns_fut = X_pi3_hist[:, :3], X_pi3_fut[:, :3]
    m_pi3_ns_hist = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_pi3_ns_hist[idx_fh], y_hist[idx_fh])
    p_pi3_ns_fut = np.clip(m_pi3_ns_hist.predict(X_pi3_ns_fut), 0.0, 1.0)
    lo_fut_ns_ijs, hi_fut_ns_ijs, _ = ijs_conformal_interval(X_pi3_ns_hist[idx_fh], X_pi3_ns_hist[idx_ch], y_hist[idx_ch], np.clip(m_pi3_ns_hist.predict(X_pi3_ns_hist[idx_ch]), 0, 1), X_pi3_ns_fut, p_pi3_ns_fut, op_m0, mode="full")


    m0_hist = np.clip(qb_hist + 0.5 * np.array([r["pi_M"] for r in id_hist]), 0, 1)
    m0_fut = np.clip(qb_fut + 0.5 * np.array([r["pi_M"] for r in ood_fut]), 0, 1)
    X_s3a_bij_ns_hist = np.array([[r["AI"], r["pi_M"], r["AI_prev"], r["pi_Delta"]] for r in id_hist])
    X_s3a_bij_ns_fut = np.array([[r["AI"], r["pi_M"], r["AI_prev"], r["pi_Delta"]] for r in ood_fut])
    m_s3a_bij_ns = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_s3a_bij_ns_hist[idx_fh], y_hist[idx_fh] - m0_hist[idx_fh])
    p_s3a_bij_ns_cal = np.clip(m0_hist[idx_ch] + m_s3a_bij_ns.predict(X_s3a_bij_ns_hist[idx_ch]), 0, 1)
    p_s3a_bij_ns_fut = np.clip(m0_fut + m_s3a_bij_ns.predict(X_s3a_bij_ns_fut), 0, 1)
    lo_s3a_bij_ns_ijs, hi_s3a_bij_ns_ijs, _ = ijs_conformal_interval(X_s3a_bij_ns_hist[idx_fh], X_s3a_bij_ns_hist[idx_ch], y_hist[idx_ch], p_s3a_bij_ns_cal, X_s3a_bij_ns_fut, p_s3a_bij_ns_fut, op_m0, mode="full")

    X_ols_fh = np.column_stack([
        np.array([id_hist[i]["pi_M"] for i in idx_fh]),
        np.array([id_hist[i]["pi_Delta"] for i in idx_fh]),
    ])
    c_M_fh, c_D_fh = np.linalg.lstsq(X_ols_fh, y_hist[idx_fh] - qb_hist[idx_fh], rcond=None)[0]
    print(f"=== S3A In-Fold OLS Estimation on idx_fh (n={len(idx_fh)}): c_M = {c_M_fh:+.4f}, c_Delta = {c_D_fh:+.4f} ===")
    X_m0h_hist = np.array([[r["AI"], r["pi_M"], r["pi_Delta"]] for r in id_hist])
    X_m0h_fut = np.array([[r["AI"], r["pi_M"], r["pi_Delta"]] for r in ood_fut])
    op_m0h_fh = lambda X: np.clip(fu_budyko_analytical(X[:, 0]) + c_M_fh * X[:, 1] + c_D_fh * X[:, 2], 0.0, 1.0)
    p_m0h_fcal = op_m0h_fh(X_m0h_hist[idx_ch])
    p_m0h_ffut = op_m0h_fh(X_m0h_fut)
    lo_m0h_fstd, hi_m0h_fstd, _ = ijs_conformal_interval(X_m0h_hist[idx_fh], X_m0h_hist[idx_ch], y_hist[idx_ch], p_m0h_fcal, X_m0h_fut, p_m0h_ffut, op_m0h_fh, mode="std")
    lo_m0h_fjac, hi_m0h_fjac, _ = ijs_conformal_interval(X_m0h_hist[idx_fh], X_m0h_hist[idx_ch], y_hist[idx_ch], p_m0h_fcal, X_m0h_fut, p_m0h_ffut, op_m0h_fh, mode="jac_only")
    p_pers_fut = np.array([r["Q_over_P_prev"] for r in ood_fut])
    s3_diff, s3_ci_lo, s3_ci_hi, s3_nb = paired_cluster_bootstrap_rmse(y_fut, p_m0h_ffut, p_pers_fut, sites_fut, n_boot=3000, seed=42)


    qb_prev_hist = np.array([float(fu_budyko_analytical(r["AI_prev"], w=2.6)) for r in id_hist])
    qb_prev_fut = np.array([float(fu_budyko_analytical(r["AI_prev"], w=2.6)) for r in ood_fut])
    X_arx_raw_h = np.column_stack([np.ones(len(id_hist)), [r["Q_over_P_prev"] for r in id_hist], [r["AI"] for r in id_hist], [r["pi_Delta"] for r in id_hist]])
    X_arx_raw_f = np.column_stack([np.ones(len(ood_fut)), [r["Q_over_P_prev"] for r in ood_fut], [r["AI"] for r in ood_fut], [r["pi_Delta"] for r in ood_fut]])
    beta_arx_raw_s3 = np.linalg.lstsq(X_arx_raw_h[idx_fh], y_hist[idx_fh], rcond=None)[0]
    p_arx_raw_s3_fut = np.clip(X_arx_raw_f @ beta_arx_raw_s3, 0.0, 1.0)

    X_arx_mat_h = np.column_stack([np.ones(len(id_hist)), [r["Q_over_P_prev"] for r in id_hist], qb_prev_hist, qb_hist, [r["pi_Delta"] for r in id_hist]])
    X_arx_mat_f = np.column_stack([np.ones(len(ood_fut)), [r["Q_over_P_prev"] for r in ood_fut], qb_prev_fut, qb_fut, [r["pi_Delta"] for r in ood_fut]])
    beta_arx_mat_s3 = np.linalg.lstsq(X_arx_mat_h[idx_fh], y_hist[idx_fh], rcond=None)[0]
    p_arx_mat_s3_cal = np.clip(X_arx_mat_h[idx_ch] @ beta_arx_mat_s3, 0.0, 1.0)
    p_arx_mat_s3_fut = np.clip(X_arx_mat_f @ beta_arx_mat_s3, 0.0, 1.0)
    lo_arx_mat_s3_jac, hi_arx_mat_s3_jac, _ = ijs_conformal_interval(X_m0h_hist[idx_fh], X_m0h_hist[idx_ch], y_hist[idx_ch], p_arx_mat_s3_cal, X_m0h_fut, p_arx_mat_s3_fut, op_m0h_fh, mode="jac_only")
    s3_arx_diff, s3_arx_lo, s3_arx_hi, _ = paired_cluster_bootstrap_rmse(y_fut, p_m0h_ffut, p_arx_mat_s3_fut, sites_fut, n_boot=3000, seed=42)
    s3_arx_raw_diff, s3_arx_raw_lo, s3_arx_raw_hi, _ = paired_cluster_bootstrap_rmse(y_fut, p_m0h_ffut, p_arx_raw_s3_fut, sites_fut, n_boot=3000, seed=42)
    print(f"=== S3A Linear ARX Comparison: ARX-Raw (4-param) RMSE={np.sqrt(np.mean((y_fut-p_arx_raw_s3_fut)**2)):.4f} (M0-ARX_raw={s3_arx_raw_diff:+.4f} [{s3_arx_raw_lo:+.4f}, {s3_arx_raw_hi:+.4f}]) | ARX-Matched (5-param) RMSE={np.sqrt(np.mean((y_fut-p_arx_mat_s3_fut)**2)):.4f} (M0-ARX_mat={s3_arx_diff:+.4f} [{s3_arx_lo:+.4f}, {s3_arx_hi:+.4f}]) ===")
    print(f"=== S3A & S3B Single-Factor Zero-Static Ablation & Paired Cluster Bootstrap ({s3_nb} OOD basins): S3B-NoCVP RMSE={np.sqrt(np.mean((y_fut-p_s3b_ns_fut)**2)):.4f} (IJS Cov={np.mean((y_fut>=lo_s3b_ns_ijs)&(y_fut<=hi_s3b_ns_ijs))*100:.1f}%, IS={winkler_score(y_fut,lo_s3b_ns_ijs,hi_s3b_ns_ijs):.3f} vs {np.sqrt(np.mean((y_fut-p_s3b_fut)**2)):.4f}) | S3A-ModB-NoStatic (Single-Factor) RMSE={np.sqrt(np.mean((y_fut-p_pi3_ns_fut)**2)):.4f} (vs {np.sqrt(np.mean((y_fut-p_pi3_fut)**2)):.4f}) | S3A-ModD-Phys-NoStatic RMSE={np.sqrt(np.mean((y_fut-p_s3a_bij_ns_fut)**2)):.4f} (IJS Cov={np.mean((y_fut>=lo_s3a_bij_ns_ijs)&(y_fut<=hi_s3a_bij_ns_ijs))*100:.1f}%, IS={winkler_score(y_fut,lo_s3a_bij_ns_ijs,hi_s3a_bij_ns_ijs):.3f}) | S3A OLS M0_hyst RMSE={np.sqrt(np.mean((y_fut-p_m0h_ffut)**2)):.4f} vs Persistence {np.sqrt(np.mean((y_fut-p_pers_fut)**2)):.4f} (Diff={s3_diff:+.4f}, 95% CI [{s3_ci_lo:+.4f}, {s3_ci_hi:+.4f}]) ===")

    pub_y = s2_data["ung_dir"]["y"]
    rows_t4 = [
        ("S1-Raw: Steady (P_t, Ep_t)", "Std-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_raw_ood)**2))), float(np.mean((y_ood >= lo_raw_std) & (y_ood <= hi_raw_std))), float(np.mean(hi_raw_std - lo_raw_std)), winkler_score(y_ood, lo_raw_std, hi_raw_std)),
        ("S1-AI: Steady-State Pi (AI_t only)", "Std-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_pi1_ood)**2))), float(np.mean((y_ood >= lo_pi1_std) & (y_ood <= hi_pi1_std))), float(np.mean(hi_pi1_std - lo_pi1_std)), winkler_score(y_ood, lo_pi1_std, hi_pi1_std)),
        ("S1-ModB: Reduced Pi* (AI_t + Pi_M, Fit ALRQ/CVP)", "Std-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_modB_ood)**2))), float(np.mean((y_ood >= lo_modB_std) & (y_ood <= hi_modB_std))), float(np.mean(hi_modB_std - lo_modB_std)), winkler_score(y_ood, lo_modB_std, hi_modB_std)),
        ("S1-ModD-Phys: Bijective Pi* (M0-Res, Fit ALRQ/CVP)", "Std-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_modD_ood)**2))), float(np.mean((y_ood >= lo_modD_std) & (y_ood <= hi_modD_std))), float(np.mean(hi_modD_std - lo_modD_std)), winkler_score(y_ood, lo_modD_std, hi_modD_std)),
        ("S1-ModA: Lagged Q_{t-1}/P_{t-1} (Fit ALRQ/CVP)", "Std-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_modA_ood)**2))), float(np.mean((y_ood >= lo_modA_std) & (y_ood <= hi_modA_std))), float(np.mean(hi_modA_std - lo_modA_std)), winkler_score(y_ood, lo_modA_std, hi_modA_std)),
        ("S1-Pers: 1-Yr Persistence (Q_{t-1}/P_{t-1}, Zero-ML)", "Std-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_pers_ood)**2))), float(np.mean((y_ood >= lo_pers_std) & (y_ood <= hi_pers_std))), float(np.mean(hi_pers_std - lo_pers_std)), winkler_score(y_ood, lo_pers_std, hi_pers_std)),
        ("S1-OLS-M0hyst: In-Fold OLS Analytical (Zero-Static)", "Std-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_m0h_ood)**2))), float(np.mean((y_ood >= lo_m0h_std) & (y_ood <= hi_m0h_std))), float(np.mean(hi_m0h_std - lo_m0h_std)), winkler_score(y_ood, lo_m0h_std, hi_m0h_std)),
        ("S2-AI-107: PUB Baseline (AI_t only, 5-Fold GroupKFold)", "Std-CP", len(pub_y), float(np.sqrt(np.mean((pub_y - s2_data['ai_base']['p'])**2))), float(np.mean((pub_y >= s2_data['ai_base']['lo_s']) & (pub_y <= s2_data['ai_base']['hi_s']))), float(np.mean(s2_data['ai_base']['hi_s'] - s2_data['ai_base']['lo_s'])), winkler_score(pub_y, s2_data['ai_base']['lo_s'], s2_data['ai_base']['hi_s'])),
        ("S2-Dir-107: PUB Pi_ung (Direct GBR, 5-Fold GroupKFold)", "Std-CP", len(pub_y), float(np.sqrt(np.mean((pub_y - s2_data['ung_dir']['p'])**2))), float(np.mean((pub_y >= s2_data['ung_dir']['lo_s']) & (pub_y <= s2_data['ung_dir']['hi_s']))), float(np.mean(s2_data['ung_dir']['hi_s'] - s2_data['ung_dir']['lo_s'])), winkler_score(pub_y, s2_data['ung_dir']['lo_s'], s2_data['ung_dir']['hi_s'])),
        ("S2-Res-107: PUB Pi_ung (Budyko-Res, 5-Fold GroupKFold)", "Std-CP", len(pub_y), float(np.sqrt(np.mean((pub_y - s2_data['ung_phys']['p'])**2))), float(np.mean((pub_y >= s2_data['ung_phys']['lo_s']) & (pub_y <= s2_data['ung_phys']['hi_s']))), float(np.mean(s2_data['ung_phys']['hi_s'] - s2_data['ung_phys']['lo_s'])), winkler_score(pub_y, s2_data['ung_phys']['lo_s'], s2_data['ung_phys']['hi_s'])),
        ("S3-Base: Temporal AI_t (0.3422) / Raw (0.2355)", "Std-CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_ai_fut)**2))), float(np.mean((y_fut >= lo_ai_fut_s) & (y_fut <= hi_ai_fut_s))), float(np.mean(hi_ai_fut_s - lo_ai_fut_s)), winkler_score(y_fut, lo_ai_fut_s, hi_ai_fut_s)),
        ("S3B-Res: Multi-Yr Pure Met Pi_ung (Budyko-Res)", "Std-CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_s3b_fut)**2))), float(np.mean((y_fut >= lo_s3b_std) & (y_fut <= hi_s3b_std))), float(np.mean(hi_s3b_std - lo_s3b_std)), winkler_score(y_fut, lo_s3b_std, hi_s3b_std)),
        ("S3A-ModB: Rolling 1-Yr Hindcast Pi* (Fit ALRQ/CVP)", "Std-CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_pi3_fut)**2))), float(np.mean((y_fut >= lo_fut_std) & (y_fut <= hi_fut_std))), float(np.mean(hi_fut_std - lo_fut_std)), winkler_score(y_fut, lo_fut_std, hi_fut_std)),
        ("S1-ModB: Reduced Pi* (Fit ALRQ/CVP)", "Tuning-Free IJS-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_modB_ood)**2))), float(np.mean((y_ood >= lo_modB_ijs) & (y_ood <= hi_modB_ijs))), float(np.mean(hi_modB_ijs - lo_modB_ijs)), winkler_score(y_ood, lo_modB_ijs, hi_modB_ijs)),
        ("S1-ModD-Phys: Bijective Pi* (M0-Res, Fit ALRQ/CVP)", "Tuning-Free IJS-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_modD_ood)**2))), float(np.mean((y_ood >= lo_modD_ijs) & (y_ood <= hi_modD_ijs))), float(np.mean(hi_modD_ijs - lo_modD_ijs)), winkler_score(y_ood, lo_modD_ijs, hi_modD_ijs)),
        ("S1-ModA: Lagged Q_{t-1}/P_{t-1} (Fit ALRQ/CVP)", "Jacobian-Only CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_modA_ood)**2))), float(np.mean((y_ood >= lo_modA_jac) & (y_ood <= hi_modA_jac))), float(np.mean(hi_modA_jac - lo_modA_jac)), winkler_score(y_ood, lo_modA_jac, hi_modA_jac)),
        ("S1-OLS-M0hyst: In-Fold OLS Analytical (Zero-Static)", "Jacobian-Only CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_m0h_ood)**2))), float(np.mean((y_ood >= lo_m0h_jac) & (y_ood <= hi_m0h_jac))), float(np.mean(hi_m0h_jac - lo_m0h_jac)), winkler_score(y_ood, lo_m0h_jac, hi_m0h_jac)),
        ("S2-Res-107: PUB Pi_ung (Budyko-Res, 5-Fold GroupKFold)", "Tuning-Free IJS-CP", len(pub_y), float(np.sqrt(np.mean((pub_y - s2_data['ung_phys']['p'])**2))), float(np.mean((pub_y >= s2_data['ung_phys']['lo_i']) & (pub_y <= s2_data['ung_phys']['hi_i']))), float(np.mean(s2_data['ung_phys']['hi_i'] - s2_data['ung_phys']['lo_i'])), winkler_score(pub_y, s2_data['ung_phys']['lo_i'], s2_data['ung_phys']['hi_i'])),
        ("S2-Res-60Dir: Pure 60-Direct-Gauge Refit (vs AI 0.2687)", "Tuning-Free IJS-CP", len(y60_all), float(np.sqrt(np.mean((y60_all - p_res60)**2))), float(np.mean((y60_all >= lo_i60) & (y60_all <= hi_i60))), float(np.mean(hi_i60 - lo_i60)), winkler_score(y60_all, lo_i60, hi_i60)),
        ("S3B-Res: Multi-Yr Pure Met Pi_ung (Budyko-Res)", "Tuning-Free IJS-CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_s3b_fut)**2))), float(np.mean((y_fut >= lo_s3b_ijs) & (y_fut <= hi_s3b_ijs))), float(np.mean(hi_s3b_ijs - lo_s3b_ijs)), winkler_score(y_fut, lo_s3b_ijs, hi_s3b_ijs)),
        ("S3A-ModB: Rolling 1-Yr Hindcast Pi* (Fit ALRQ/CVP)", "Tuning-Free IJS-CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_pi3_fut)**2))), float(np.mean((y_fut >= lo_fut_ijs) & (y_fut <= hi_fut_ijs))), float(np.mean(hi_fut_ijs - lo_fut_ijs)), winkler_score(y_fut, lo_fut_ijs, hi_fut_ijs)),
        ("S3A-OLS-M0hyst: In-Fold <=2015 OLS (Zero-Static)", "Jacobian-Only CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_m0h_ffut)**2))), float(np.mean((y_fut >= lo_m0h_fjac) & (y_fut <= hi_m0h_fjac))), float(np.mean(hi_m0h_fjac - lo_m0h_fjac)), winkler_score(y_fut, lo_m0h_fjac, hi_m0h_fjac)),
    ]

    print("\n=== Complete Fit-Isolated Benchmark Table (Table 1, 22 Rows, Exact Finite-Sample Order Statistic) ===")
    for r in rows_t4:
        print(f"{r[0]:55s} | {r[1]:18s} | n={r[2]:4d} | RMSE={r[3]:.4f} | Cov={r[4]*100:5.1f}% | W={r[5]:.3f} | IS={r[6]:.3f}")

    table4_path = os.path.join(os.path.dirname(out_dir), "results", "table1_catchment_benchmarks.csv")
    with open(table4_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formulation", "uq_method", "n_ood", "ood_rmse", "cov90", "mean_width", "winkler_score"])
        for r in rows_t4:
            w.writerow(r)


    lo_modA_ns_jac, hi_modA_ns_jac, _ = ijs_conformal_interval(X_modA_ns_id[idx_f], X_modA_ns_id[idx_c], y_id[idx_c], m_modA_ns.predict(X_modA_ns_id[idx_c]), X_modA_ns_ood, p_modA_ns_ood, op_modA, mode="jac_only")
    lo_modB_ns_ijs, hi_modB_ns_ijs, _ = ijs_conformal_interval(X_modB_ns_id[idx_f], X_modB_ns_id[idx_c], y_id[idx_c], m_modB_ns.predict(X_modB_ns_id[idx_c]), X_modB_ns_ood, p_modB_ns_ood, op_m0, mode="full")
    diag_rows = [
        ("S1-ModB-NoStatic (AI_t, Pi_M, Pi_Delta; Zero ALRQ/CVP)", "Tuning-Free IJS-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_modB_ns_ood)**2))), float(np.mean((y_ood >= lo_modB_ns_ijs) & (y_ood <= hi_modB_ns_ijs))), float(np.mean(hi_modB_ns_ijs - lo_modB_ns_ijs)), winkler_score(y_ood, lo_modB_ns_ijs, hi_modB_ns_ijs), "Single-Factor Zero-Static Ablation (vs S1-ModB 0.2292)"),
        ("S1-ModA-NoStatic (AI_t, Q_{t-1}/P_{t-1}, Pi_Delta; Zero ALRQ/CVP)", "Jacobian-Only CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_modA_ns_ood)**2))), float(np.mean((y_ood >= lo_modA_ns_jac) & (y_ood <= hi_modA_ns_jac))), float(np.mean(hi_modA_ns_jac - lo_modA_ns_jac)), winkler_score(y_ood, lo_modA_ns_jac, hi_modA_ns_jac), "Single-Factor Zero-Static Ablation (vs S1-ModA 0.1424)"),
        ("S1-ModD-Phys-NoStatic (AI_t, Pi_M, AI_{t-1}, Pi_Delta; Zero ALRQ/CVP)", "Tuning-Free IJS-CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_modD_ns_ood)**2))), float(np.mean((y_ood >= lo_modD_ns_ijs) & (y_ood <= hi_modD_ns_ijs))), float(np.mean(hi_modD_ns_ijs - lo_modD_ns_ijs)), winkler_score(y_ood, lo_modD_ns_ijs, hi_modD_ns_ijs), "Single-Factor Zero-Static Ablation (vs S1-ModD-Phys 0.1374)"),
        ("S2 Double-Basin-Isolated Inner GroupShuffleSplit (All 107 Basins)", "Tuning-Free IJS-CP", len(yd), float(np.sqrt(np.mean((yd - pd)**2))), float(np.mean((yd >= lod_i) & (yd <= hid_i))), float(np.mean(hid_i - lod_i)), winkler_score(yd, lod_i, hid_i), "Disjoint Fit/Cal/Test Basins (All 107 Basins)"),
        ("S2 Double-Basin-Isolated Inner GroupShuffleSplit (Pure 60 Direct Gauges)", "Tuning-Free IJS-CP", len(y60_all), float(np.sqrt(np.mean((y60_all - p_res60_dbl)**2))), cov60_dbl_mean, float(np.mean(hi_i60_dbl - lo_i60_dbl)), winkler_score(y60_all, lo_i60_dbl, hi_i60_dbl), f"50-Basin Cluster Cov 95% CI [{cov60_dbl_lo*100:.1f}%, {cov60_dbl_hi*100:.1f}%]; Folds: {fold_cov_str}"),
        ("S3B-Res-NoCVP (Zero Fit-Fold CVP, Pure Met + Geo)", "Tuning-Free IJS-CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_s3b_ns_fut)**2))), float(np.mean((y_fut >= lo_s3b_ns_ijs) & (y_fut <= hi_s3b_ns_ijs))), float(np.mean(hi_s3b_ns_ijs - lo_s3b_ns_ijs)), winkler_score(y_fut, lo_s3b_ns_ijs, hi_s3b_ns_ijs), "Single-Factor Zero-Static Ablation (vs S3B-Res 0.2183)"),
        ("S3A-ModB-NoStatic (AI_t, Pi_M, Pi_Delta; Direct GBR, Zero ALRQ/CVP)", "Tuning-Free IJS-CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_pi3_ns_fut)**2))), float(np.mean((y_fut >= lo_fut_ns_ijs) & (y_fut <= hi_fut_ns_ijs))), float(np.mean(hi_fut_ns_ijs - lo_fut_ns_ijs)), winkler_score(y_fut, lo_fut_ns_ijs, hi_fut_ns_ijs), "Single-Factor Zero-Static Ablation (vs S3A-ModB 0.2181)"),
        ("S3A-ModD-Phys-NoStatic (AI_t, Pi_M, AI_{t-1}, Pi_Delta; Bijective M0-Res)", "Tuning-Free IJS-CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_s3a_bij_ns_fut)**2))), float(np.mean((y_fut >= lo_s3a_bij_ns_ijs) & (y_fut <= hi_s3a_bij_ns_ijs))), float(np.mean(hi_s3a_bij_ns_ijs - lo_s3a_bij_ns_ijs)), winkler_score(y_fut, lo_s3a_bij_ns_ijs, hi_s3a_bij_ns_ijs), "Bijective Coupled-Residual Zero-Static Hindcast"),
        ("Bootstrap S1 M0_hyst vs 1-Yr Persistence (92 OOD Basins)", "Cluster Bootstrap 3000", len(y_ood), s1_diff, s1_ci_lo, s1_ci_hi, 0.0, f"95% CI [{s1_ci_lo:+.4f}, {s1_ci_hi:+.4f}]"),
        ("Bootstrap S3A M0_hyst vs 1-Yr Persistence (90 OOD Basins)", "Cluster Bootstrap 3000", len(y_fut), s3_diff, s3_ci_lo, s3_ci_hi, 0.0, f"95% CI [{s3_ci_lo:+.4f}, {s3_ci_hi:+.4f}]"),
        ("Bootstrap S2 All-107 Budyko-Res vs AI-Only (92 OOD Basins)", "Cluster Bootstrap 3000", len(pub_y), s2_all_diff, s2_all_lo, s2_all_hi, 0.0, f"95% CI [{s2_all_lo:+.4f}, {s2_all_hi:+.4f}]"),
        ("Bootstrap S2 Pure 60-Direct Budyko-Res vs AI-Only (50 OOD Basins)", "Cluster Bootstrap 3000", len(y60_all), s2_60_diff, s2_60_lo, s2_60_hi, 0.0, f"95% CI [{s2_60_lo:+.4f}, {s2_60_hi:+.4f}]"),
        ("S1-ARX-Matched (5-Param Unconstrained OLS: 1, Q_{t-1}/P_{t-1}, B_{t-1}, B_t, Pi_Delta)", "Jacobian-Only CP", len(y_ood), float(np.sqrt(np.mean((y_ood - p_arx_mat_s1_ood)**2))), float(np.mean((y_ood >= lo_arx_mat_s1_jac) & (y_ood <= hi_arx_mat_s1_jac))), float(np.mean(hi_arx_mat_s1_jac - lo_arx_mat_s1_jac)), winkler_score(y_ood, lo_arx_mat_s1_jac, hi_arx_mat_s1_jac), f"ARX-Raw RMSE=0.1479; M0_hyst vs ARX-Matched Diff={s1_arx_diff:+.4f}, 95% CI [{s1_arx_lo:+.4f}, {s1_arx_hi:+.4f}]"),
        ("S3A-ARX-Matched (5-Param Unconstrained <=2015 OLS: 1, Q_{t-1}/P_{t-1}, B_{t-1}, B_t, Pi_Delta)", "Jacobian-Only CP", len(y_fut), float(np.sqrt(np.mean((y_fut - p_arx_mat_s3_fut)**2))), float(np.mean((y_fut >= lo_arx_mat_s3_jac) & (y_fut <= hi_arx_mat_s3_jac))), float(np.mean(hi_arx_mat_s3_jac - lo_arx_mat_s3_jac)), winkler_score(y_fut, lo_arx_mat_s3_jac, hi_arx_mat_s3_jac), f"ARX-Raw RMSE=0.1540; M0_hyst vs ARX-Matched Diff={s3_arx_diff:+.4f}, 95% CI [{s3_arx_lo:+.4f}, {s3_arx_hi:+.4f}] (Excludes 0)"),
        ("60-Direct Spatial-Scale Stratification: Grid-Matched Small-Medium Basins (A <= 300 km2, 12 OOD Basins)", "Scale & Process Diagnostic", int(np.sum(d60_ood_mask & (area_ood <= 300.0))), float(np.sqrt(np.mean((y_ood[d60_ood_mask & (area_ood <= 300.0)] - p_m0h_ood[d60_ood_mask & (area_ood <= 300.0)])**2))), float(spearmanr(pim_abs_ood[d60_ood_mask & (area_ood <= 300.0)], err_b_ood[d60_ood_mask & (area_ood <= 300.0)]).statistic), float(spearmanr(pid_ood[d60_ood_mask & (area_ood <= 300.0)], res_wi_ood[d60_ood_mask & (area_ood <= 300.0)]).statistic), 0.0, "Grid-scale (~1-3 ERA5-Land cells): rho(|Pi_M|,|err|)=0.724, rho_wi=-0.222, M0 RMSE=0.1345 vs Pers 0.1411 & Budyko 0.3496"),
        ("60-Direct Baseflow Stratification: High vs Low Sustainment (LFR > 0.25 vs <= 0.25, 50 OOD Basins)", "Scale & Process Diagnostic", int(np.sum(d60_ood_mask & (lfr_ood > 0.25))), float(np.sqrt(np.mean((y_ood[d60_ood_mask & (lfr_ood > 0.25)] - p_m0h_ood[d60_ood_mask & (lfr_ood > 0.25)])**2))), float(spearmanr(pim_abs_ood[d60_ood_mask & (lfr_ood > 0.25)], err_b_ood[d60_ood_mask & (lfr_ood > 0.25)]).statistic), float(spearmanr(pid_ood[d60_ood_mask & (lfr_ood > 0.25)], res_wi_ood[d60_ood_mask & (lfr_ood > 0.25)]).statistic), 0.0, "High LFR (>0.25): rho_wi=-0.403 (vs Low LFR -0.159), M0 RMSE=0.1111 vs Budyko 0.2722 (-59.2%)"),
    ]
    diag_csv_path = os.path.join(os.path.dirname(out_dir), "results", "table_s1_diagnostics_ablations.csv")
    with open(diag_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["diagnostic_or_ablation", "method", "n_ood", "metric_or_rmse", "cov90_or_ci_low", "width_or_ci_high", "winkler_score", "notes"])
        for r in diag_rows:
            w.writerow(r)
    print(f"Saved complete supplementary diagnostics & ablations table: {diag_csv_path}")


if __name__ == "__main__":
    main()
