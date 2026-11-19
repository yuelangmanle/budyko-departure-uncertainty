\
\
\
\

import csv
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from run_full_experiments import hydrological_runoff_ratio, calibrate_dsa_beta_sublinear
from analyze_catchments import conformal_order_quantile


def eval_runoff_dsa(k_nn=5, q_thresh=0.82, noise_std=0.008, seed=2026):
    rng = np.random.default_rng(seed)
    n_run = 1800
    P_tr, AI_tr = rng.uniform(800.0, 2000.0, n_run), rng.uniform(0.3, 1.0, n_run)
    Ep_tr, Ks_tr = AI_tr * P_tr, rng.uniform(5.0, 25.0, n_run)
    IK_tr = rng.uniform(0.4, 1.5, n_run)
    I_tr = IK_tr * Ks_tr
    y_tr = np.clip(hydrological_runoff_ratio(P_tr, Ep_tr, I_tr, Ks_tr) + rng.normal(0.0, noise_std, n_run), 0.0, 1.0)
    X_pi_tr = np.column_stack([AI_tr, IK_tr])

    idx_fit, idx_rest = train_test_split(np.arange(n_run), test_size=0.4, random_state=42)
    idx_cal, _ = train_test_split(idx_rest, test_size=0.5, random_state=42)

    m_fn = lambda: GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
    m_pi = m_fn().fit(X_pi_tr[idx_fit], y_tr[idx_fit])
    alpha = 0.10
    q_pi = conformal_order_quantile(np.abs(y_tr[idx_cal] - m_pi.predict(X_pi_tr[idx_cal])), alpha)

    beta_opt = calibrate_dsa_beta_sublinear(X_pi_tr[idx_fit], y_tr[idx_fit], m_fn, shift_col=0, q_thresh=q_thresh, alpha=alpha)
    sc = StandardScaler().fit(X_pi_tr[idx_fit])
    nn = NearestNeighbors(n_neighbors=k_nn).fit(sc.transform(X_pi_tr[idx_fit]))
    d_cal = np.mean(nn.kneighbors(sc.transform(X_pi_tr[idx_cal]))[0], axis=1)
    d_scale = conformal_order_quantile(d_cal, alpha) + 1e-6

    n_o = 500
    P_o, AI_o = rng.uniform(300.0, 1200.0, n_o), rng.uniform(1.6, 2.5, n_o)
    Ep_o, Ks_o = AI_o * P_o, rng.uniform(5.0, 25.0, n_o)
    IK_o = rng.uniform(0.5, 2.5, n_o)
    I_o = IK_o * Ks_o
    X_p = np.column_stack([AI_o, IK_o])
    y_o = np.clip(hydrological_runoff_ratio(P_o, Ep_o, I_o, Ks_o) + rng.normal(0.0, noise_std, n_o), 0.0, 1.0)

    pred_p = np.clip(m_pi.predict(X_p), 0.0, 1.0)
    d_test = np.mean(nn.kneighbors(sc.transform(X_p))[0], axis=1)
    shift = np.power(np.maximum(0.0, d_test / d_scale - 1.0), 0.75)
    hw = q_pi * (1.0 + beta_opt * shift)
    lo_s, hi_s = np.clip(pred_p - q_pi, 0.0, 1.0), np.clip(pred_p + q_pi, 0.0, 1.0)
    lo_d, hi_d = np.clip(pred_p - hw, 0.0, 1.0), np.clip(pred_p + hw, 0.0, 1.0)

    return {
        "cov_std": float(np.mean((y_o >= lo_s) & (y_o <= hi_s))),
        "cov_dsa": float(np.mean((y_o >= lo_d) & (y_o <= hi_d))),
        "width_dsa": float(np.mean(hi_d - lo_d)),
        "beta_opt": beta_opt,
    }


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    rows = []

    for k in [3, 5, 10, 20]:
        res = eval_runoff_dsa(k_nn=k)
        rows.append({"sweep": "k_neighbors", "param_val": k, **res})
    for qt in [0.75, 0.80, 0.85, 0.90]:
        res = eval_runoff_dsa(q_thresh=qt)
        rows.append({"sweep": "q_thresh", "param_val": qt, **res})
    for ns in [0.002, 0.008, 0.020, 0.040]:
        res = eval_runoff_dsa(noise_std=ns)
        rows.append({"sweep": "noise_std", "param_val": ns, **res})

    csv_path = os.path.join(os.path.dirname(out_dir), "results", "table4_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Sensitivity analysis complete.")


if __name__ == "__main__":
    main()
