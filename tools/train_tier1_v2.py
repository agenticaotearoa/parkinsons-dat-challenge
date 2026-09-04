#!/usr/bin/env python3
"""Tier-1 v2 — features on the CANONICAL GRID (76,92,60)@(3,2.5,3)mm RAS.

v1 failed (OOF 0.707 > 0.688 const) because native geometries mix
(128x128x45 and 256^3 etc.). v2 uses data/canon/* (anatomy-aligned, verified
by group-diff EDA: pathologic-cooler peak at z~18, y~68, x~59 i.e. RIGHT
anterior striatum, neg~80 vs pos~35).

Features (all anatomy-stable now):
- striatal ROI (bilateral boxes centered on the empirical peak, mirrored on
  the L side), reference region (posterior band + whole-brain norm)
- SBR-style ratios per side, putamen/caudate split along axis1 within ROI
- L/R asymmetry (axis0 halves of the ROI pair)
- intensity percentiles inside ROI + whole brain, texture on the ROI slice
- uptake gradient along P-A axis (peak position/skew)
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path("/Users/agenta/.openclaw-autoclaw/workspace/parkinsons")
CANON = ROOT / "data/canon"
LABELS = ROOT / "data/downloads/train_labels.csv"
ASSETS = ROOT / "submission_src/assets"
GRID = (76, 92, 60)
EPS = 1e-6

# empirical peak (from canonicalize_eda.py): right anterior striatum
PEAK_R = (18, 68, 59)  # z(L-R axis idx), y(P-A), x(I-S)
# mirror to left side: axis0 size 76 -> L index = 75 - 18 = 57
PEAK_L = (GRID[0] - 1 - PEAK_R[0], PEAK_R[1], PEAK_R[2])


def roi_box(center, half=(5, 8, 8)):
    z, y, x = center
    hz, hy, hx = half
    return (slice(max(0, z - hz), z + hz + 1),
            slice(max(0, y - hy), y + hy + 1),
            slice(max(0, x - hx), x + hx + 1))


ROI_R = roi_box(PEAK_R)
ROI_L = roi_box(PEAK_L)


def features(vol: np.ndarray) -> dict:
    f = {}
    brain = vol > np.percentile(vol, 45)
    wm = np.percentile(vol[brain], 90) if brain.any() else 1.0
    nz = vol / (wm + EPS)

    f["brain_mean"] = float(nz[brain].mean())
    f["brain_p99"] = float(np.percentile(nz, 99))

    for side, box in (("r", ROI_R), ("l", ROI_L)):
        roi = nz[box]
        f[f"roi_mean_{side}"] = float(roi.mean())
        f[f"roi_p90_{side}"] = float(np.percentile(roi, 90))
        f[f"roi_hotfrac_{side}"] = float((roi > 1.2).mean())
        # putamen/caudate proxy: split ROI along P-A (axis1) at its middle
        mid = roi.shape[1] // 2
        f[f"put_mean_{side}"] = float(roi[:, :mid].mean())
        f[f"cau_mean_{side}"] = float(roi[:, mid:].mean())

    # reference: posterior band (occipital/cortical) on axis1
    post = nz[:, : int(GRID[1] * 0.35), :]
    f["ref_mean"] = float(post.mean())
    f["ref_p90"] = float(np.percentile(post, 90))

    # SBR ratios
    for side in ("r", "l"):
        f[f"sbr_{side}"] = f[f"roi_mean_{side}"] / (f["ref_mean"] + EPS)
        f[f"put_sbr_{side}"] = f[f"put_mean_{side}"] / (f["ref_mean"] + EPS)
        f[f"cau_sbr_{side}"] = f[f"cau_mean_{side}"] / (f["ref_mean"] + EPS)

    # asymmetries
    f["lr_roi_asym"] = abs(f["roi_mean_r"] - f["roi_mean_l"]) / (f["roi_mean_r"] + f["roi_mean_l"] + EPS)
    f["lr_put_asym"] = abs(f["put_mean_r"] - f["put_mean_l"]) / (f["put_mean_r"] + f["put_mean_l"] + EPS)
    f["put_cau_ratio_r"] = f["put_mean_r"] / (f["cau_mean_r"] + EPS)
    f["put_cau_ratio_l"] = f["put_mean_l"] / (f["cau_mean_l"] + EPS)

    # ant/post uptake gradient on axis1 (whole brain)
    pa = nz.mean(axis=(0, 2))
    f["pa_peak_pos"] = float(np.argmax(pa) / GRID[1])
    f["pa_skew"] = float(((np.arange(GRID[1]) - np.argmax(pa)) ** 3 * pa).sum() / (pa.sum() * GRID[1] ** 3))

    # texture: GLCM on the ROI mid slice (right)
    try:
        from skimage.feature import graycomatrix, graycoprops
        zc = (ROI_R[0].start + ROI_R[0].stop) // 2
        sl = nz[zc, ROI_R[1], ROI_R[2]]
        q = (np.clip(sl, 0, 2.0) / 2.0 * 31).astype(np.uint8)
        glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4], levels=32, symmetric=True, normed=True)
        f["glcm_contrast"] = float(graycoprops(glcm, "contrast").mean())
        f["glcm_homog"] = float(graycoprops(glcm, "homogeneity").mean())
    except Exception:
        pass
    return f


def main():
    t0 = time.time()
    df = pd.read_csv(LABELS)
    y = df.is_pathologic.values.astype(int)
    uids = df.uid.tolist()
    print(f"features on canonical grid, {len(uids)} scans...")
    rows = []
    for i, u in enumerate(uids):
        vol = nib.load(str(CANON / f"{u}.nii.gz")).get_fdata(dtype=np.float32)
        rows.append(features(vol))
        if (i + 1) % 300 == 0:
            print(f"  {i+1} ({time.time()-t0:.0f}s)", flush=True)
    X = pd.DataFrame(rows)
    order = list(X.columns)
    Xv = X.values
    print("matrix", Xv.shape, f"{time.time()-t0:.0f}s")

    # univariate signal audit
    from sklearn.metrics import roc_auc_score
    aucs = []
    for j, name in enumerate(order):
        try:
            a = roc_auc_score(y, Xv[:, j])
            aucs.append((abs(a - 0.5), name, a))
        except Exception:
            pass
    aucs.sort(reverse=True)
    print("TOP FEATURES BY |AUC-0.5|:")
    for _, n, a in aucs[:8]:
        print(f"  {n:22s} AUC={a:.3f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    oof = np.zeros(len(y))
    for tr, va in skf.split(Xv, y):
        clf = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                            reg_lambda=2.0, eval_metric="logloss", n_jobs=4, random_state=77, verbosity=0)
        clf.fit(Xv[tr], y[tr])
        oof[va] = clf.predict_proba(Xv[va])[:, 1]

    def ll(yv, pv):
        p = np.clip(pv, 1e-7, 1 - 1e-7)
        return float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))

    raw = ll(y, oof)
    base = ll(y, np.full(len(y), y.mean()))
    print(f"XGB OOF LOG LOSS: {raw:.4f} (const {base:.4f})")

    lgb_oof = np.zeros(len(y))
    for tr, va in skf.split(Xv, y):
        m = lgb.LGBMClassifier(n_estimators=600, num_leaves=24, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                               random_state=77, verbose=-1, n_jobs=4)
        m.fit(Xv[tr], y[tr])
        lgb_oof[va] = m.predict_proba(Xv[va])[:, 1]
    ll_lgb = ll(y, lgb_oof)
    ll_blend = ll(y, 0.5 * oof + 0.5 * lgb_oof)
    print(f"LGBM OOF: {ll_lgb:.4f} | blend: {ll_blend:.4f}")

    import joblib
    ASSETS.mkdir(parents=True, exist_ok=True)
    for old in ASSETS.glob("*.pkl"):
        old.unlink()
    blend_w = 0.5 if ll_blend < min(raw, ll_lgb) else 0.0
    if blend_w > 0:
        cal = CalibratedClassifierCV(XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                                   reg_lambda=2.0, eval_metric="logloss", n_jobs=4,
                                                   random_state=77, verbosity=0), method="isotonic", cv=5)
        cal.fit(Xv, y)
        joblib.dump(cal, ASSETS / "sbr_model_xgb.pkl")
        lgbf = lgb.LGBMClassifier(n_estimators=600, num_leaves=24, learning_rate=0.04, subsample=0.8,
                                  colsample_bytree=0.8, min_child_samples=20, random_state=77, verbose=-1, n_jobs=4)
        lgbf.fit(Xv, y)
        joblib.dump(lgbf, ASSETS / "sbr_model_lgb.pkl")
        models = ["sbr_model_xgb.pkl", "sbr_model_lgb.pkl"]
    else:
        best = ("xgb", raw) if raw <= ll_lgb else ("lgb", ll_lgb)
        if best[0] == "xgb":
            cal = CalibratedClassifierCV(XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                                       subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                                       reg_lambda=2.0, eval_metric="logloss", n_jobs=4,
                                                       random_state=77, verbosity=0), method="isotonic", cv=5)
            cal.fit(Xv, y)
            joblib.dump(cal, ASSETS / "sbr_model.pkl")
        else:
            lgbf = lgb.LGBMClassifier(n_estimators=600, num_leaves=24, learning_rate=0.04, subsample=0.8,
                                      colsample_bytree=0.8, min_child_samples=20, random_state=77, verbose=-1, n_jobs=4)
            lgbf.fit(Xv, y)
            joblib.dump(lgbf, ASSETS / "sbr_model.pkl")
        models = ["sbr_model.pkl"]

    def box_list(b):
        return [[s.start, s.stop] for s in b]

    spec = {"feature_order": order, "models": models, "blend_weight": blend_w,
            "oof_logloss": {"xgb": raw, "lgbm": ll_lgb, "blend": ll_blend},
            "grid": list(GRID), "roi_r": box_list(ROI_R), "roi_l": box_list(ROI_L),
            "trained": time.strftime("%Y-%m-%d %H:%M"), "n_train": len(uids),
            "note": "canonical-grid v2; inference must canonicalize+resample identically"}
    (ASSETS / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    print("SAVED", sorted(p.name for p in ASSETS.iterdir()))
    print(f"DONE {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
