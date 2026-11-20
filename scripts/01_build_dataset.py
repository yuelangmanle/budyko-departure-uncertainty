#!/usr/bin/env python3
"""
Milestone 1: Build unified, strictly quality-controlled annual water-year dataset
combining CAMELS-GB v1 (UK, energy-limited) and CAMELS-AUS v2.03 (Australia, water-limited).
"""
import csv
import io
import json
import math
import os
import zipfile
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar

GB_ZIP = "./data/raw/camels_gb_v1/CAMELS_GB_v1.zip"
AUS_DIR = "./data/raw/camels_aus_v2.03"
OUT_DIR = "./data"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_VALID_DAYS_PER_WY = 355
MIN_COMPLETE_WYS = 20
MAX_FRAC_SNOW = 0.15


def safe_float(val, default=np.nan):
    if val is None:
        return default
    s = str(val).strip()
    if not s or s.lower() in ("nan", "na", "null", "none", "n/a"):
        return default
    try:
        x = float(s)
        return x if math.isfinite(x) else default
    except ValueError:
        return default


def fu_evap_ratio(ai, omega):
    # Fu (1981): E/P = 1 + AI - (1 + AI^omega)^(1/omega)
    ai = np.asarray(ai, dtype=float)
    return 1.0 + ai - np.power(1.0 + np.power(np.maximum(ai, 1e-9), omega), 1.0 / omega)


def fit_fu_omega(ai_val, er_val):
    if not (math.isfinite(ai_val) and math.isfinite(er_val)) or ai_val <= 0:
        return np.nan
    er_clipped = min(max(er_val, 0.01), min(0.999, ai_val * 0.999))
    res = minimize_scalar(
        lambda w: (float(fu_evap_ratio(ai_val, w)) - er_clipped) ** 2,
        bounds=(1.01, 15.0),
        method="bounded",
    )
    return float(res.x)


def load_gb():
    print("[GB] Loading CAMELS-GB v1 from ZIP...")
    with zipfile.ZipFile(GB_ZIP) as z:
        def read_attr(substr):
            m = [n for n in z.namelist() if substr in n and n.endswith(".csv")][0]
            with z.open(m) as f:
                rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="ignore")))
            return {r["gauge_id"].strip(): r for r in rows}

        clim = read_attr("climatic_attributes")
        topo = read_attr("topographic_attributes")
        hydr = read_attr("hydrologic_attributes")
        hum = read_attr("humaninfluence_attributes")
        land = read_attr("landcover_attributes")
        soil = read_attr("soil_attributes")

        ts_files = sorted(n for n in z.namelist() if "timeseries" in n.lower() and n.endswith(".csv"))
        annual_rows = {}  # gid -> dict[wy -> (P, PET, Q, n_days)]

        for n in ts_files:
            base = os.path.basename(n)
            # CAMELS_GB_hydromet_timeseries_<gid>_19701001-20150930.csv
            parts = base.split("_")
            gid = parts[4]
            wy_acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0])  # P, PET, Q, n_valid, n_total
            with z.open(n) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="ignore"))
                for row in reader:
                    d = row["date"]
                    y, m = int(d[:4]), int(d[5:7])
                    wy = y + 1 if m >= 10 else y
                    wy_acc[wy][4] += 1
                    p = safe_float(row["precipitation"])
                    pet = safe_float(row["pet"])
                    q = safe_float(row["discharge_spec"])
                    if math.isfinite(p) and math.isfinite(pet) and math.isfinite(q) and p >= 0 and pet >= 0 and q >= 0:
                        wy_acc[wy][0] += p
                        wy_acc[wy][1] += pet
                        wy_acc[wy][2] += q
                        wy_acc[wy][3] += 1

            good = {}
            for wy, (sp, spet, sq, nv, nt) in wy_acc.items():
                if nv >= MIN_VALID_DAYS_PER_WY and nt >= 365:
                    scale = nt / nv
                    good[wy] = (sp * scale, spet * scale, sq * scale, nv)
            annual_rows[gid] = good

    attrs = {}
    for gid in annual_rows:
        c = clim.get(gid, {})
        t = topo.get(gid, {})
        h = hydr.get(gid, {})
        hu = hum.get(gid, {})
        lc = land.get(gid, {})
        so = soil.get(gid, {})

        dwood = safe_float(lc.get("dwood_perc"), 0.0)
        ewood = safe_float(lc.get("ewood_perc"), 0.0)
        forest_frac = (dwood + ewood) / 100.0 if (math.isfinite(dwood) and math.isfinite(ewood)) else np.nan
        bench = str(hu.get("benchmark_catch", "")).strip().upper()
        res_cap = safe_float(hu.get("reservoir_cap"), np.nan)
        is_nat = 1 if (bench in ("Y", "1", "TRUE") or (math.isfinite(res_cap) and res_cap == 0.0)) else 0

        attrs[gid] = {
            "basin_id": f"GB_{gid}",
            "raw_id": gid,
            "region": "GB",
            "lat": safe_float(t.get("gauge_lat")),
            "lon": safe_float(t.get("gauge_lon")),
            "area_km2": safe_float(t.get("area")),
            "elev_mean_m": safe_float(t.get("elev_mean")),
            "slope_proxy": (safe_float(t.get("dpsbar")) / 10.0) if not math.isnan(safe_float(t.get("dpsbar"))) else np.nan,  # converted from m/km to % to match CAMELS-AUS mean_slope_pct
            "bfi": safe_float(h.get("baseflow_index")),
            "frac_snow": safe_float(c.get("frac_snow"), 0.0),
            "p_seasonality": safe_float(c.get("p_seasonality")),
            "forest_frac": forest_frac,
            "clay_pct": safe_float(so.get("clay_perc")),
            "sand_pct": safe_float(so.get("sand_perc")),
            "is_near_natural": is_nat,
        }
    print(f"[GB] Loaded {len(annual_rows)} raw basins.")
    return annual_rows, attrs


def load_aus():
    print("[AUS] Loading CAMELS-AUS v2.03 (streamflow + AGCD P + SILO Morton Wet PET)...")
    sf_path = os.path.join(AUS_DIR, "streamflow_mmd.csv")
    p_path = os.path.join(AUS_DIR, "hydrometeorology", "precipitation_AGCD.csv")
    pet_path = os.path.join(AUS_DIR, "hydrometeorology", "et_morton_wet_SILO.csv")
    mt_path = os.path.join(AUS_DIR, "CAMELS_AUS_Attributes&Indices_MasterTable.csv")

    # Read MasterTable attributes
    attrs = {}
    with open(mt_path, "r", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            sid = r["station_id"].strip()
            river_di = safe_float(r.get("river_di"), np.nan)
            impound = safe_float(r.get("impound_fac"), np.nan)
            is_nat = 1 if (math.isfinite(river_di) and river_di < 0.2 and (not math.isfinite(impound) or impound < 0.1)) else 0
            attrs[sid] = {
                "basin_id": f"AUS_{sid}",
                "raw_id": sid,
                "region": "AUS",
                "lat": safe_float(r.get("lat_outlet")),
                "lon": safe_float(r.get("long_outlet")),
                "area_km2": safe_float(r.get("catchment_area")),
                "elev_mean_m": safe_float(r.get("elev_mean")),
                "slope_proxy": safe_float(r.get("mean_slope_pct")),
                "bfi": safe_float(r.get("sig_mag_BFI")),
                "frac_snow": safe_float(r.get("frac_snow"), 0.0),
                "p_seasonality": safe_float(r.get("p_seasonality")),
                "forest_frac": safe_float(r.get("prop_forested")),
                "clay_pct": safe_float(r.get("claya")),
                "sand_pct": safe_float(r.get("sanda")),
                "is_near_natural": is_nat,
            }

    # Stream aligned daily rows across Q, P, PET starting from 1950-01-01 (start of streamflow_mmd.csv)
    # First load P and PET into dicts keyed by (y, m, d) for y >= 1950
    def read_wide_csv(path, min_year=1950):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            hdr = next(reader)
            sids = [c.strip() for c in hdr[3:]]
            data = {}
            for row in reader:
                y = int(row[0])
                if y < min_year:
                    continue
                m, d = int(row[1]), int(row[2])
                vals = np.array([float(x) if x and x != "NaN" else np.nan for x in row[3:]], dtype=np.float32)
                vals[vals < 0] = np.nan
                data[(y, m, d)] = vals
        return sids, data

    print("  Reading streamflow_mmd.csv...")
    q_sids, q_data = read_wide_csv(sf_path, 1950)
    print(f"  Loaded Q: {len(q_sids)} basins, {len(q_data)} days")
    print("  Reading precipitation_AGCD.csv...")
    p_sids, p_data = read_wide_csv(p_path, 1950)
    print(f"  Loaded P: {len(p_sids)} basins, {len(p_data)} days")
    print("  Reading et_morton_wet_SILO.csv...")
    pet_sids, pet_data = read_wide_csv(pet_path, 1950)
    print(f"  Loaded PET: {len(pet_sids)} basins, {len(pet_data)} days")

    assert q_sids == p_sids == pet_sids, "Station ID order mismatch across AUS files!"
    n_basins = len(q_sids)
    common_dates = sorted(set(q_data.keys()) & set(p_data.keys()) & set(pet_data.keys()))
    print(f"  Common daily overlap: {len(common_dates)} days ({common_dates[0]} to {common_dates[-1]})")

    # Accumulate by Southern Hemisphere water year (Apr 1 - Mar 31): wy = y + 1 if m >= 4 else y
    wy_P = defaultdict(lambda: np.zeros(n_basins, dtype=np.float64))
    wy_PET = defaultdict(lambda: np.zeros(n_basins, dtype=np.float64))
    wy_Q = defaultdict(lambda: np.zeros(n_basins, dtype=np.float64))
    wy_nv = defaultdict(lambda: np.zeros(n_basins, dtype=np.int32))
    wy_nt = defaultdict(int)

    for dt in common_dates:
        y, m, d = dt
        wy = y + 1 if m >= 4 else y
        wy_nt[wy] += 1
        qv = q_data[dt]
        pv = p_data[dt]
        petv = pet_data[dt]
        valid = np.isfinite(qv) & np.isfinite(pv) & np.isfinite(petv)
        wy_P[wy] += np.where(valid, pv, 0.0)
        wy_PET[wy] += np.where(valid, petv, 0.0)
        wy_Q[wy] += np.where(valid, qv, 0.0)
        wy_nv[wy] += valid.astype(np.int32)

    annual_rows = {sid: {} for sid in q_sids}
    for wy, nt in wy_nt.items():
        if nt < 365:
            continue
        nv_arr = wy_nv[wy]
        p_arr = wy_P[wy]
        pet_arr = wy_PET[wy]
        q_arr = wy_Q[wy]
        for idx, sid in enumerate(q_sids):
            nv = int(nv_arr[idx])
            if nv >= MIN_VALID_DAYS_PER_WY:
                scale = nt / nv
                annual_rows[sid][wy] = (
                    float(p_arr[idx] * scale),
                    float(pet_arr[idx] * scale),
                    float(q_arr[idx] * scale),
                    nv,
                )

    print(f"[AUS] Aggregated {len(annual_rows)} raw basins.")
    return annual_rows, attrs


def main():
    gb_ann, gb_attr = load_gb()
    aus_ann, aus_attr = load_aus()

    all_sources = [("GB", gb_ann, gb_attr), ("AUS", aus_ann, aus_attr)]
    kept_basins = []
    annual_records = []
    filter_log = defaultdict(lambda: defaultdict(int))

    for region, ann_dict, attr_dict in all_sources:
        for raw_id, wy_map in sorted(ann_dict.items()):
            filter_log[region]["raw_total"] += 1
            if raw_id not in attr_dict:
                filter_log[region]["drop_no_attr"] += 1
                continue
            a = attr_dict[raw_id]
            fs = a["frac_snow"] if math.isfinite(a["frac_snow"]) else 0.0
            if fs >= MAX_FRAC_SNOW:
                filter_log[region]["drop_snow"] += 1
                continue

            # Keep only physically valid positive P, PET water years (Q >= 0)
            valid_wys = {
                wy: vals for wy, vals in wy_map.items()
                if vals[0] > 10.0 and vals[1] > 10.0 and vals[2] >= 0.0
            }
            if len(valid_wys) < MIN_COMPLETE_WYS:
                filter_log[region]["drop_short_record"] += 1
                continue

            p_vals = [v[0] for v in valid_wys.values()]
            pet_vals = [v[1] for v in valid_wys.values()]
            q_vals = [v[2] for v in valid_wys.values()]
            mean_P = float(np.mean(p_vals))
            mean_PET = float(np.mean(pet_vals))
            mean_Q = float(np.mean(q_vals))
            lt_RR = mean_Q / mean_P
            lt_AI = mean_PET / mean_P

            # Long-term water balance closure & non-zero runoff
            if lt_RR > 1.0 or lt_RR <= 0.001:
                filter_log[region]["drop_water_balance"] += 1
                continue

            # Also filter extreme single-year unphysical anomalies (Q_t / P_t > 1.5 indicates gauge/forcing error)
            # Keep only years with RR_t <= 1.2 (allow slight storage release >1.0 in wet-to-dry years, e.g. Chalk groundwater)
            clean_wys = {
                wy: vals for wy, vals in valid_wys.items()
                if (vals[2] / vals[0]) <= 1.2
            }
            if len(clean_wys) < MIN_COMPLETE_WYS:
                filter_log[region]["drop_after_wy_clean"] += 1
                continue

            filter_log[region]["kept"] += 1
            omega_i = fit_fu_omega(lt_AI, 1.0 - lt_RR)
            rec_attr = dict(a)
            rec_attr.update({
                "mean_P_mm": round(mean_P, 2),
                "mean_PET_mm": round(mean_PET, 2),
                "mean_Q_mm": round(mean_Q, 2),
                "long_term_AI": round(lt_AI, 4),
                "long_term_RR": round(lt_RR, 4),
                "fu_omega": round(omega_i, 4),
                "n_complete_wys": len(clean_wys),
                "wy_start": min(clean_wys.keys()),
                "wy_end": max(clean_wys.keys()),
            })
            kept_basins.append(rec_attr)

            sorted_wys = sorted(clean_wys.keys())
            for wy in sorted_wys:
                p, pet, q, nv = clean_wys[wy]
                ai_t = pet / p
                rr_t = q / p
                er_t = 1.0 - rr_t
                has_lag1 = (wy - 1) in clean_wys
                if has_lag1:
                    p1, pet1, q1, _ = clean_wys[wy - 1]
                    ai1 = pet1 / p1
                    rr1 = q1 / p1
                else:
                    p1 = pet1 = q1 = ai1 = rr1 = np.nan

                annual_records.append({
                    "basin_id": rec_attr["basin_id"],
                    "region": region,
                    "water_year": wy,
                    "P_mm": round(p, 3),
                    "PET_mm": round(pet, 3),
                    "Q_mm": round(q, 3),
                    "AI_t": round(ai_t, 5),
                    "RR_t": round(rr_t, 5),
                    "ER_t": round(er_t, 5),
                    "n_valid_days": nv,
                    "has_lag1": int(has_lag1),
                    "P_lag1": round(p1, 3) if has_lag1 else "",
                    "PET_lag1": round(pet1, 3) if has_lag1 else "",
                    "Q_lag1": round(q1, 3) if has_lag1 else "",
                    "AI_lag1": round(ai1, 5) if has_lag1 else "",
                    "RR_lag1": round(rr1, 5) if has_lag1 else "",
                })

    # Save CSVs
    attr_out = os.path.join(OUT_DIR, "gb_aus_basin_attributes.csv")
    with open(attr_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(kept_basins[0].keys()))
        w.writeheader()
        w.writerows(kept_basins)

    ann_out = os.path.join(OUT_DIR, "gb_aus_annual_water_years.csv")
    with open(ann_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(annual_records[0].keys()))
        w.writeheader()
        w.writerows(annual_records)

    # Compute summary stats
    summary = {"filter_log": {k: dict(v) for k, v in filter_log.items()}, "regions": {}}
    for reg in ["GB", "AUS", "ALL"]:
        bs = [b for b in kept_basins if reg == "ALL" or b["region"] == reg]
        rs = [r for r in annual_records if reg == "ALL" or r["region"] == reg]
        ais = [b["long_term_AI"] for b in bs]
        rrs = [b["long_term_RR"] for b in bs]
        areas = [b["area_km2"] for b in bs if math.isfinite(b["area_km2"])]
        nat = sum(b["is_near_natural"] for b in bs)
        lag1 = sum(r["has_lag1"] for r in rs)
        summary["regions"][reg] = {
            "n_basins": len(bs),
            "n_near_natural": nat,
            "n_basin_years": len(rs),
            "n_consecutive_lag1_pairs": lag1,
            "AI_min": round(min(ais), 3),
            "AI_median": round(float(np.median(ais)), 3),
            "AI_max": round(max(ais), 3),
            "AI_gt_1_count": sum(x > 1.0 for x in ais),
            "RR_min": round(min(rrs), 3),
            "RR_median": round(float(np.median(rrs)), 3),
            "RR_max": round(max(rrs), 3),
            "area_km2_median": round(float(np.median(areas)), 1),
            "wy_min": min(r["water_year"] for r in rs),
            "wy_max": max(r["water_year"] for r in rs),
        }

    sum_out = os.path.join(OUT_DIR, "dataset_summary.json")
    with open(sum_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== UNIFIED DATASET SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
