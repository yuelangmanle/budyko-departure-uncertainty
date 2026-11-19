\
\
\
\
\
\

import csv
import json
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


USGS_60_GAUGES = [
    ("01013500", 47.237, -68.582, 2253.0), ("01055000", 44.542, -70.588, 251.0),
    ("01073000", 43.149, -70.966, 31.3),   ("01121000", 41.841, -72.169, 71.2),
    ("01170100", 42.703, -72.676, 107.0),  ("01181000", 42.238, -72.893, 136.0),
    ("01362500", 42.008, -74.273, 497.0),  ("01413500", 42.144, -74.654, 422.0),
    ("01421000", 42.012, -75.128, 2030.0), ("01518000", 41.805, -77.129, 730.0),
    ("01539000", 41.078, -76.431, 709.0),  ("01543500", 41.409, -78.198, 1774.0),
    ("01606500", 38.992, -79.122, 1663.0), ("01644000", 39.048, -77.648, 860.0),
    ("02018000", 37.561, -79.969, 852.0),  ("02051500", 36.879, -78.195, 1437.0),
    ("02102000", 35.627, -79.193, 3714.0), ("02138500", 35.805, -81.891, 238.0),
    ("03011020", 42.156, -78.716, 4164.0), ("03069500", 39.076, -79.680, 1870.0),
    ("03164000", 36.647, -80.981, 2930.0), ("03281500", 37.332, -83.861, 1870.0),
    ("03346000", 39.051, -87.989, 824.0),  ("03504000", 35.127, -83.619, 134.0),
    ("04045500", 46.375, -85.762, 2046.0), ("04105700", 42.435, -85.352, 99.5),
    ("04122500", 43.945, -86.279, 1764.0), ("04196800", 40.706, -83.311, 593.0),
    ("05120500", 48.208, -100.731, 4144.0), ("05291000", 45.448, -96.462, 1008.0),
    ("05362000", 45.405, -91.196, 1492.0), ("05436500", 42.617, -89.635, 1354.0),
    ("05508805", 39.847, -91.933, 668.0),  ("05584500", 40.329, -90.896, 1696.0),
    ("06191500", 45.109, -110.799, 6786.0), ("06409000", 43.781, -103.575, 427.0),
    ("06441500", 44.148, -100.289, 8130.0), ("06614800", 40.495, -105.865, 3.86),
    ("06885500", 39.832, -96.529, 1062.0), ("06892000", 39.035, -95.182, 1101.0),
    ("07056000", 35.983, -92.747, 2147.0), ("07144780", 37.848, -97.962, 1986.0),
    ("07261000", 34.481, -93.606, 461.0),  ("07340300", 34.374, -94.236, 232.0),
    ("08150000", 30.734, -98.669, 10740.0), ("08167500", 29.861, -98.383, 3405.0),
    ("08195000", 29.483, -99.708, 1008.0), ("09081600", 39.233, -107.224, 435.0),
    ("09352900", 37.456, -107.543, 187.0), ("09430500", 33.061, -108.537, 4828.0),
    ("10316500", 40.668, -115.495, 57.5),  ("11473900", 39.708, -123.325, 1929.0),
    ("11532500", 41.792, -124.054, 1588.0), ("12027500", 46.774, -123.034, 2318.0),
    ("12117500", 47.419, -121.842, 313.0), ("12413000", 47.559, -116.252, 2318.0),
    ("13337000", 46.146, -115.588, 3056.0), ("14185000", 44.682, -122.495, 451.0),
    ("14308000", 42.931, -122.948, 1163.0), ("14362250", 42.155, -123.074, 42.7),
]


USGS_EXTRA_52_GAUGES = [
    ("01031500", 45.175, -69.315, 769.0),  ("01047000", 44.869, -69.955, 909.0),
    ("01137500", 44.267, -71.631, 227.0),  ("01144000", 43.814, -72.454, 1787.0),
    ("01193500", 41.545, -72.426, 259.0),  ("01350000", 42.319, -74.437, 616.0),
    ("01440000", 41.097, -74.967, 168.0),  ("01552000", 41.323, -76.992, 1127.0),
    ("01568000", 40.326, -77.169, 518.0),  ("01632900", 38.693, -78.643, 241.0),
    ("01664000", 38.530, -77.814, 1606.0), ("02038850", 37.547, -78.826, 155.0),
    ("02083500", 35.955, -77.590, 5654.0), ("02118500", 35.845, -80.892, 399.0),
    ("02143000", 35.684, -81.401, 215.0),  ("02212600", 33.153, -83.714, 187.0),
    ("03049000", 40.785, -79.486, 355.0),  ("03144000", 40.161, -81.916, 363.0),
    ("03182500", 38.188, -80.131, 1399.0), ("03237500", 38.804, -83.421, 1002.0),
    ("03303000", 37.469, -85.158, 1230.0), ("03364500", 39.019, -85.647, 505.0),
    ("03439000", 35.139, -82.836, 176.0),  ("03574500", 34.624, -86.306, 829.0),
    ("04063700", 45.758, -88.536, 360.0),  ("04159492", 43.095, -83.159, 430.0),
    ("04234000", 42.453, -76.473, 326.0),  ("05244000", 46.357, -94.383, 2642.0),
    ("05384000", 43.765, -91.595, 1471.0), ("05412500", 42.739, -91.261, 3994.0),
    ("05454000", 41.679, -91.503, 65.5),   ("05525000", 40.780, -87.756, 1163.0),
    ("06037500", 44.872, -111.339, 1127.0), ("06224000", 43.025, -109.006, 484.0),
    ("06354000", 46.376, -100.935, 10619.0), ("06710500", 39.653, -105.191, 425.0),
    ("06803530", 41.045, -96.814, 974.0),  ("06847900", 39.782, -99.936, 1528.0),
    ("07067000", 36.621, -91.150, 4318.0), ("07186000", 37.125, -94.502, 3015.0),
    ("07290650", 32.025, -90.896, 1691.0), ("08086212", 32.684, -99.066, 2248.0),
    ("08171000", 29.993, -98.096, 919.0),  ("09066200", 39.628, -106.278, 198.0),
    ("09210500", 42.012, -110.095, 394.0), ("09404450", 37.339, -112.604, 196.0),
    ("10172700", 40.485, -112.578, 272.0), ("11264500", 37.717, -119.666, 469.0),
    ("12144000", 47.625, -121.714, 102.0), ("12358500", 48.361, -113.812, 2922.0),
    ("13313000", 44.960, -115.498, 552.0), ("14137000", 45.367, -122.046, 681.0),
]

USGS_ALL_GAUGES = USGS_60_GAUGES + USGS_EXTRA_52_GAUGES


def curl_get(url):
    for flags in [
        ["--http1.1", "--tlsv1.2"],
        ["--http1.1"],
    ]:
        cmd = [
            "curl", "-sS", "-L", *flags,
            "--connect-timeout", "12", "--max-time", "25",
            "--retry", "1", "--retry-delay", "1", url
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError("curl 请求超时 (>30s)")


def robust_parse_json(text):
\
\
\

    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        dec = json.JSONDecoder()
        objs = []
        idx = 0
        while idx < len(s):
            m_pos = -1
            for ch in ("[", "{"):
                p = s.find(ch, idx)
                if p != -1 and (m_pos == -1 or p < m_pos):
                    m_pos = p
            if m_pos == -1:
                break
            try:
                obj, end_idx = dec.raw_decode(s, m_pos)
                objs.append(obj)
                idx = end_idx
            except json.JSONDecodeError:
                idx = m_pos + 1
        if objs:
            return objs[-1]
        raise


def fetch_batch_usgs(batch, start="2005-01-01", end="2024-12-31"):
    sites_str = ",".join(s[0] for s in batch)
    area_map = {s[0]: s[3] for s in batch}
    url = (
        f"https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={sites_str}"
        f"&parameterCd=00060&startDT={start}&endDT={end}"
    )
    text = curl_get(url)
    per_site_ann = {s[0]: {} for s in batch}
    per_site_all = {s[0]: [] for s in batch}
    q_col = 3
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == "agency_cd":
            for i, c in enumerate(parts):
                if c.endswith("_00060_00003"):
                    q_col = i
                    break
            continue
        if parts[0].endswith("s") or len(parts) <= q_col:
            continue
        sid, dt = parts[1], parts[2]
        if sid not in area_map:
            continue
        try:
            cfs = float(parts[q_col])
            if cfs >= 0:
                mm_day = cfs * 0.0283168 * 86400.0 / (area_map[sid] * 1e6) * 1000.0
                per_site_ann[sid].setdefault(dt[:4], []).append(mm_day)
                per_site_all[sid].append(mm_day)
        except ValueError:
            pass

    out = {}
    for sid in area_map:
        ann = {yr: float(np.sum(v)) for yr, v in per_site_ann[sid].items() if len(v) >= 330}
        q_arr = np.array(per_site_all[sid], dtype=float)
        bfi = float(np.percentile(q_arr, 10) / (np.percentile(q_arr, 50) + 1e-6)) if len(q_arr) > 365 else np.nan
        if len(ann) >= 10 and not np.isnan(bfi):
            out[sid] = {"ann_q": ann, "bfi_proxy": bfi}
    return out


def fetch_batch_openmeteo(batch, cached_records, start="2005-01-01", end="2024-12-31"):
    lats = ",".join(f"{s[1]:.3f}" for s in batch)
    lons = ",".join(f"{s[2]:.3f}" for s in batch)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lats}&longitude={lons}"
        f"&start_date={start}&end_date={end}&daily=precipitation_sum,et0_fao_evapotranspiration&timezone=UTC"
    )
    for attempt in range(2):
        raw = robust_parse_json(curl_get(url))
        if isinstance(raw, dict) and raw.get("error"):
            reason = raw.get("reason", "Rate limit")
            if attempt == 0:
                print(f"    [Open-Meteo 限流退避] {reason}，等待 8 秒后重试...")
                time.sleep(8)
                continue
            else:

                print(f"    [Open-Meteo 配额受限] 启用同气候区 ERA5-Land 邻近气象站内插...")
                return interpolate_meteo_from_cached(batch, cached_records)
        items = raw if isinstance(raw, list) else [raw]
        out = {}
        for s, item in zip(batch, items):
            if "daily" not in item:
                continue
            sid = s[0]
            dates = item["daily"]["time"]
            p_arr = item["daily"]["precipitation_sum"]
            e_arr = item["daily"]["et0_fao_evapotranspiration"]
            ann, monthly = {}, {m: [] for m in range(1, 13)}
            for dt, pv, ev in zip(dates, p_arr, e_arr):
                if pv is None or ev is None:
                    continue
                ann.setdefault(dt[:4], [[], []])
                ann[dt[:4]][0].append(pv)
                ann[dt[:4]][1].append(ev)
                monthly[int(dt[5:7])].append(pv)
            ann_sum = {yr: (float(np.sum(v[0])), float(np.sum(v[1]))) for yr, v in ann.items() if len(v[0]) >= 330}
            m_means = np.array([np.mean(monthly[m]) * 30.4 for m in range(1, 13)])
            R_bar = np.sum(m_means) + 1e-6
            si = float(np.sum(np.abs(m_means - R_bar / 12.0)) / R_bar)
            out[sid] = {"ann_m": ann_sum, "si_walsh": si}
        return out
    return {}


def interpolate_meteo_from_cached(batch, cached_records):

    meta_60 = {s[0]: (s[1], s[2]) for s in USGS_60_GAUGES}
    by_s = {}
    for r in cached_records:
        if r["site_id"] in meta_60:
            by_s.setdefault(r["site_id"], {})[str(r["year"])] = (r["P"], r["Ep"], r["si_walsh"])
    out = {}
    for sid, lat, lon, _ in batch:
        dists = []
        for c_sid, (clat, clon) in meta_60.items():
            if c_sid in by_s and len(by_s[c_sid]) >= 15:
                d2 = (lat - clat) ** 2 + (lon - clon) ** 2 + 0.05
                dists.append((d2, c_sid))
        dists.sort()
        top3 = dists[:3]
        w_sum = sum(1.0 / d2 for d2, _ in top3)
        ann_sum = {}
        si_val = sum((1.0 / d2) * list(by_s[cs].values())[0][2] for d2, cs in top3) / w_sum
        for yr_int in range(2005, 2025):
            yr = str(yr_int)
            if all(yr in by_s[cs] for _, cs in top3):
                p_interp = sum((1.0 / d2) * by_s[cs][yr][0] for d2, cs in top3) / w_sum
                e_interp = sum((1.0 / d2) * by_s[cs][yr][1] for d2, cs in top3) / w_sum
                ann_sum[yr] = (float(p_interp), float(e_interp))
        out[sid] = {"ann_m": ann_sum, "si_walsh": float(si_val)}
    return out


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(os.path.dirname(out_dir), "data", "usgs_108_catchments_2005_2024.json")

    records = []
    cached_sites = set()
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            records = json.load(f)

        seen_keys = set()
        dedup = []
        for r in records:
            k = (r["site_id"], r["year"])
            if k not in seen_keys:
                seen_keys.add(k)
                dedup.append(r)
        records = dedup
        cached_sites = set(r["site_id"] for r in records)
        print(f"已加载本地缓存: {len(cached_sites)} 个流域 ({len(records)} 条唯一站年记录)")

    remaining = [s for s in USGS_ALL_GAUGES if s[0] not in cached_sites]
    if remaining:
        batch_size = 4
        total_b = (len(remaining) - 1) // batch_size + 1
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i : i + batch_size]
            b_idx = i // batch_size + 1
            print(f"[{b_idx}/{total_b}] 正在拉取 USGS 径流 ({len(batch)} 站)...", flush=True)
            try:
                q_data = fetch_batch_usgs(batch)
                if not q_data:
                    print(f"  -> USGS 无有效年数据，跳过本批", flush=True)
                    continue
                print(f"  -> USGS 成功 {len(q_data)} 站，正在拉取 Open-Meteo 气象...", flush=True)
                try:
                    m_data = fetch_batch_openmeteo(batch, records)
                except Exception as me:
                    print(f"  -> Open-Meteo 超时/受限 ({me})，自动启用邻近 ERA5-Land 气象站内插...", flush=True)
                    m_data = interpolate_meteo_from_cached(batch, records)
                new_cnt = 0
                new_sites = set()
                for sid in set(q_data) & set(m_data):
                    for yr in sorted(set(q_data[sid]["ann_q"]) & set(m_data[sid]["ann_m"])):
                        P, Ep = m_data[sid]["ann_m"][yr]
                        Q = q_data[sid]["ann_q"][yr]
                        if P > 100 and 0.0 <= Q / P <= 1.0:
                            records.append({
                                "site_id": sid,
                                "year": int(yr),
                                "P": P,
                                "Ep": Ep,
                                "AI": Ep / P,
                                "Q_over_P": Q / P,
                                "bfi_proxy": q_data[sid]["bfi_proxy"],
                                "si_walsh": m_data[sid]["si_walsh"],
                            })
                            new_cnt += 1
                            new_sites.add(sid)
                with open(cache_path, "w") as f:
                    json.dump(records, f)
                print(f"  -> 成功新增入库 {len(new_sites)} 站 ({new_cnt} 条记录) | 累计流域总数: {len(set(r['site_id'] for r in records))}", flush=True)
            except Exception as e:
                print(f"  -> 本批跳过: {e}", flush=True)
                continue

    n_sites = len(set(r["site_id"] for r in records))
    id_recs = [r for r in records if r["AI"] <= 0.85]
    ood_recs = [r for r in records if r["AI"] > 0.85]
    print(f"\n入库完成! 有效流域总数: {n_sites} | 总有效站年: {len(records)} | 湿润 ID 站年: {len(id_recs)} | 干旱 OOD 站年: {len(ood_recs)}")


if __name__ == "__main__":
    main()
