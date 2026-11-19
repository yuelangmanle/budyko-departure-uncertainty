\
\
\
\

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from analyze_catchments import conformal_order_quantile


def adr_outlet_concentration(Pe, Da):
    Pe = np.asarray(Pe, dtype=float)
    Da = np.asarray(Da, dtype=float)
    q = np.sqrt(1.0 + 4.0 * Da / Pe)
    num = 4.0 * q * np.exp(0.5 * Pe * (1.0 - q))
    den = (1.0 + q) ** 2 - ((1.0 - q) ** 2) * np.exp(-Pe * q)
    return np.clip(num / den, 0.0, 1.0)


def hydrological_runoff_ratio(P, Ep, I, Ks, w=2.6):
    AI = Ep / P
    et_ratio = 1.0 + AI - (1.0 + AI ** w) ** (1.0 / w)
    q_budyko = np.maximum(1.0 - et_ratio, 0.0)
    excess_ratio = np.maximum(I / Ks - 1.0, 0.0)
    r_excess = 0.35 * (1.0 - np.exp(-excess_ratio)) * (1.0 - q_budyko)
    return np.clip(q_budyko + r_excess, 0.0, 1.0)


def calibrate_dsa_beta_sublinear(X_pi_fit, y_fit, model_fn, shift_col=0, q_thresh=0.82, alpha=0.10):
    cutoff = np.quantile(X_pi_fit[:, shift_col], q_thresh)
    mask_val = X_pi_fit[:, shift_col] > cutoff
    X_sub, y_sub = X_pi_fit[~mask_val], y_fit[~mask_val]
    X_val, y_val = X_pi_fit[mask_val], y_fit[mask_val]

    idx_f, idx_c = train_test_split(np.arange(len(y_sub)), test_size=0.25, random_state=42)
    m_sub = model_fn()
    m_sub.fit(X_sub[idx_f], y_sub[idx_f])
    q_sub = conformal_order_quantile(np.abs(y_sub[idx_c] - m_sub.predict(X_sub[idx_c])), alpha)

    sc = StandardScaler().fit(X_sub[idx_f])
    nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(X_sub[idx_f]))
    d_c = np.mean(nn.kneighbors(sc.transform(X_sub[idx_c]))[0], axis=1)
    d_scale = conformal_order_quantile(d_c, alpha) + 1e-6

    d_val = np.mean(nn.kneighbors(sc.transform(X_val))[0], axis=1)
    shift_val = np.power(np.maximum(0.0, d_val / d_scale - 1.0), 0.75)
    err_val = np.abs(y_val - m_sub.predict(X_val))

    for beta in np.linspace(0.0, 6.0, 601):
        if np.mean(err_val <= q_sub * (1.0 + beta * shift_val)) >= 1.0 - alpha:
            return float(beta)
    return 6.0


def evaluate_benchmark(name, X_raw_tr, X_pi_tr, y_tr, ood_splits, shift_col=0, alpha=0.10):
    idx = np.arange(len(y_tr))
    idx_fit, idx_rest = train_test_split(idx, test_size=0.4, random_state=42)
    idx_cal, idx_id = train_test_split(idx_rest, test_size=0.5, random_state=42)

    all_splits = [("ID", X_raw_tr[idx_id], X_pi_tr[idx_id], y_tr[idx_id])] + ood_splits
    model_factories = {
        "GBR": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42),
        "RF": lambda: RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=1),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)),
        "PolyRidge": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), Ridge(alpha=1.0)),
    }

    rows = []
    for m_name, m_fn in model_factories.items():
        m_raw = m_fn().fit(X_raw_tr[idx_fit], y_tr[idx_fit])
        m_pi = m_fn().fit(X_pi_tr[idx_fit], y_tr[idx_fit])

        res_raw_cal = np.abs(y_tr[idx_cal] - m_raw.predict(X_raw_tr[idx_cal]))
        res_pi_cal = np.abs(y_tr[idx_cal] - m_pi.predict(X_pi_tr[idx_cal]))

        q_raw = conformal_order_quantile(res_raw_cal, alpha)
        q_pi = conformal_order_quantile(res_pi_cal, alpha)

        lw_scale_model = RandomForestRegressor(n_estimators=60, max_depth=5, random_state=42, n_jobs=1)
        lw_scale_model.fit(X_pi_tr[idx_cal], res_pi_cal + 1e-4)
        q_lw = conformal_order_quantile(res_pi_cal / np.maximum(lw_scale_model.predict(X_pi_tr[idx_cal]), 1e-4), alpha)

        beta_opt = calibrate_dsa_beta_sublinear(X_pi_tr[idx_fit], y_tr[idx_fit], m_fn, shift_col=shift_col, alpha=alpha)
        sc = StandardScaler().fit(X_pi_tr[idx_fit])
        nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(X_pi_tr[idx_fit]))
        d_cal = np.mean(nn.kneighbors(sc.transform(X_pi_tr[idx_cal]))[0], axis=1)
        d_scale = conformal_order_quantile(d_cal, alpha) + 1e-6

        for reg_name, X_r, X_p, y_true in all_splits:
            pred_r = np.clip(m_raw.predict(X_r), 0.0, 1.0)
            pred_p = np.clip(m_pi.predict(X_p), 0.0, 1.0)

            lo_r, hi_r = np.clip(pred_r - q_raw, 0.0, 1.0), np.clip(pred_r + q_raw, 0.0, 1.0)
            lo_p, hi_p = np.clip(pred_p - q_pi, 0.0, 1.0), np.clip(pred_p + q_pi, 0.0, 1.0)

            sig_lw = np.maximum(lw_scale_model.predict(X_p), 1e-4)
            hw_lw = q_lw * sig_lw
            lo_lw, hi_lw = np.clip(pred_p - hw_lw, 0.0, 1.0), np.clip(pred_p + hw_lw, 0.0, 1.0)

            d_test = np.mean(nn.kneighbors(sc.transform(X_p))[0], axis=1)
            shift_ratio = np.power(np.maximum(0.0, d_test / d_scale - 1.0), 0.75)
            hw_dsa = q_pi * (1.0 + beta_opt * shift_ratio)
            lo_d, hi_d = np.clip(pred_p - hw_dsa, 0.0, 1.0), np.clip(pred_p + hw_dsa, 0.0, 1.0)

            rows.append({
                "benchmark": name,
                "model": m_name,
                "regime": reg_name,
                "beta_opt": beta_opt,
                "rmse_raw": float(np.sqrt(np.mean((y_true - pred_r) ** 2))),
                "rmse_pi": float(np.sqrt(np.mean((y_true - pred_p) ** 2))),
                "cov_raw_std": float(np.mean((y_true >= lo_r) & (y_true <= hi_r))),
                "cov_pi_std": float(np.mean((y_true >= lo_p) & (y_true <= hi_p))),
                "cov_pi_lw": float(np.mean((y_true >= lo_lw) & (y_true <= hi_lw))),
                "cov_pi_dsa": float(np.mean((y_true >= lo_d) & (y_true <= hi_d))),
                "width_raw_std": float(np.mean(hi_r - lo_r)),
                "width_pi_std": float(np.mean(hi_p - lo_p)),
                "width_pi_lw": float(np.mean(hi_lw - lo_lw)),
                "width_pi_dsa": float(np.mean(hi_d - lo_d)),
            })
    return rows


def main():
    rng = np.random.default_rng(2026)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    n_adr = 1600
    v_tr, L_tr = rng.uniform(0.1, 1.0, n_adr), rng.uniform(1.0, 5.0, n_adr)
    Pe_tr, Da_tr = rng.uniform(0.2, 5.0, n_adr), rng.uniform(0.1, 3.0, n_adr)
    D_tr, k_tr = v_tr * L_tr / Pe_tr, Da_tr * v_tr / L_tr
    y_adr = adr_outlet_concentration(Pe_tr, Da_tr) + rng.normal(0.0, 0.005, n_adr)
    X_adr_raw = np.column_stack([v_tr, L_tr, D_tr, k_tr])
    X_adr_pi = np.column_stack([Pe_tr, Da_tr])

    adr_ood = []
    for pmin, pmax, lbl in [(5.0, 10.0, "OOD1"), (10.0, 20.0, "OOD2"), (20.0, 45.0, "OOD3")]:
        n_o = 400
        v_o, L_o = rng.uniform(0.5, 2.0, n_o), rng.uniform(1.0, 5.0, n_o)
        Pe_o, Da_o = rng.uniform(pmin, pmax, n_o), rng.uniform(0.1, 3.0, n_o)
        D_o, k_o = v_o * L_o / Pe_o, Da_o * v_o / L_o
        y_o = adr_outlet_concentration(Pe_o, Da_o) + rng.normal(0.0, 0.005, n_o)
        adr_ood.append((lbl, np.column_stack([v_o, L_o, D_o, k_o]), np.column_stack([Pe_o, Da_o]), y_o))

    n_run = 1800
    P_tr, AI_tr = rng.uniform(800.0, 2000.0, n_run), rng.uniform(0.3, 1.0, n_run)
    Ep_tr, Ks_tr = AI_tr * P_tr, rng.uniform(5.0, 25.0, n_run)
    IK_tr = rng.uniform(0.4, 1.5, n_run)
    I_tr = IK_tr * Ks_tr
    y_run = hydrological_runoff_ratio(P_tr, Ep_tr, I_tr, Ks_tr) + rng.normal(0.0, 0.008, n_run)
    X_run_raw = np.column_stack([P_tr, Ep_tr, I_tr, Ks_tr])
    X_run_pi = np.column_stack([AI_tr, IK_tr])

    run_ood = []
    for amin, amax, ikmin, ikmax, lbl in [
        (1.0, 1.6, 0.4, 2.0, "OOD1"),
        (1.6, 2.5, 0.5, 2.5, "OOD2"),
        (2.5, 4.0, 0.5, 3.0, "OOD3"),
    ]:
        n_o = 500
        P_o, AI_o = rng.uniform(300.0, 1200.0, n_o), rng.uniform(amin, amax, n_o)
        Ep_o, Ks_o = AI_o * P_o, rng.uniform(5.0, 25.0, n_o)
        IK_o = rng.uniform(ikmin, ikmax, n_o)
        I_o = IK_o * Ks_o
        y_o = hydrological_runoff_ratio(P_o, Ep_o, I_o, Ks_o) + rng.normal(0.0, 0.008, n_o)
        run_ood.append((lbl, np.column_stack([P_o, Ep_o, I_o, Ks_o]), np.column_stack([AI_o, IK_o]), y_o))

    rows_adr = evaluate_benchmark("1D_ADR", X_adr_raw, X_adr_pi, y_adr, adr_ood, shift_col=0)
    rows_run = evaluate_benchmark("Fu_Budyko_Runoff", X_run_raw, X_run_pi, y_run, run_ood, shift_col=0)

    for fname, rows in [("table2_synthetic_adr.csv", rows_adr), ("table3_synthetic_runoff.csv", rows_run)]:
        path = os.path.join(out_dir, fname)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
