#!/usr/bin/env python3
"""Tier-1 model training — SBR-style features -> gradient boosting.

Data: parkinsons/data/train/{uid}.nii.gz + train_labels.csv (1362 scans).
Local training only (RAM doctrine: sequential; this is the only heavy job).
Competition data NEVER leaves this machine.

Feature design (registration-free, orientation-aware):
- Volumes are (R,A,S): axis0=L-R(128), axis1=P-A(128), axis2=I-S(45).
- Striatal zone: anterior band of axis1 (A-S uptake concentrates anterior).
  Reference (occipital/cortical): posterior band of axis1.
- Robust normalization per volume via percentiles.
- Features: uptake ratios (anterior vs posterior), hot-band fractions per ROI,
  L/R asymmetry inside the striatal box (axis 0), putamen/caudate sub-boxes
  (ventral/dorsal split on axis2), intensity percentiles, texture (GLCM on the
  mid striatal slice), volume stats.

Output: assets/sbr_model.pkl + assets/feature_spec.json inside submission_src/
  (exact contract expected by submission_src/main.py tier-1 branch).
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path("/Users/agenta/.openclaw-autoclaw/workspace/parkinsons")
TRAIN_DIR = ROOT / "data/train"
LABELS = ROOT / "data/downloads/train_labels.csv"
ASSETS = ROOT / "submission_src/assets"

EPS = 1e-6


def load_vol(uid: str) -> np.ndarray:
    return nib.load(str(TRAIN_DIR / f"{uid}.nii.gz")).get_fdata(dtype=np.float32)


def robust_norm(vol: np.ndarray) -> tuple:
    p = np.percentile(vol, [50, 75, 90, 97, 99])
    span = max(float(p[4] - p[0]), EPS)
    return np.clip((vol - p[0]) / span, 0.0, 1.5), p


def features(vol: np.ndarray) -> dict:
    norm, pct = robust_norm(vol)
    nz, ny, nx = vol.shape  # (I-S=45, P-A=128, L-R=128) as (z, y, x)
    f = {}

    # --- ROI bands on the P-A axis (axis1) ---
    ant = norm[:, int(ny * 0.45):, :]          # anterior 55% = striatal zone
    post = norm[:, : int(ny * 0.40), :]        # posterior 40% = occipital/cortical ref
    hot_a, hot_p = ant >= 0.9, post >= 0.9
    mid_a = (ant >= 0.55) & (ant < 0.9)

    f["ant_hot_frac"] = float(hot_a.mean())
    f["ant_mid_frac"] = float(mid_a.mean())
    f["post_hot_frac"] = float(hot_p.mean())
    f["ant_post_ratio"] = float(hot_a.sum() / (hot_p.sum() + EPS))
    f["ant_mean"] = float(ant.mean())
    f["post_mean"] = float(post.mean())
    f["ant_p99"] = float(np.percentile(ant, 99))
    f["post_p99"] = float(np.percentile(post, 99))

    # --- L/R asymmetry inside striatal box (axis0 split) ---
    xs = nx // 2
    for name, sl in (("ventral", slice(0, nz // 2)), ("dorsal", slice(nz // 2, None))):
        hot_l = float(hot_a[sl, :, :xs].sum())
        hot_r = float(hot_a[sl, :, xs:].sum())
        f[f"lr_asym_{name}"] = abs(hot_l - hot_r) / (hot_l + hot_r + 1.0)
        f[f"hot_vol_{name}"] = (hot_l + hot_r) / (ant[sl].size + EPS)

    # --- putamen vs caudate proxy: inferior vs superior striatal sub-bands ---
    f["hot_ventral_dorsal_ratio"] = f["hot_vol_ventral"] / (f["hot_vol_dorsal"] + EPS)

    # --- global intensity / texture ---
    f["int_p50"], f["int_p97"] = float(pct[0]), float(pct[3])
    f["vol_nonzero_frac"] = float((vol > 0).mean())
    f["norm_mean"], f["norm_std"] = float(norm.mean()), float(norm.std())

    try:
        from skimage.feature import graycomatrix, graycoprops
        zs = nz // 2
        sl = norm[zs, int(ny * 0.45): int(ny * 0.85), :]
        q = (np.clip(sl, 0, 1.0) * 31).astype(np.uint8)
        glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4], levels=32, symmetric=True, normed=True)
        f["glcm_contrast"] = float(graycoprops(glcm, "contrast").mean())
        f["glcm_homog"] = float(graycoprops(glcm, "homogeneity").mean())
        f["glcm_energy"] = float(graycoprops(glcm, "energy").mean())
    except Exception:
        pass

    # --- gradient of uptake along P-A (pathologic = flatter anterior peak) ---
    pa_profile = norm.mean(axis=(0, 2))
    f["pa_grad_peak_pos"] = float(np.argmax(pa_profile) / ny)
    f["pa_grad_skew"] = float(((np.arange(ny) - np.argmax(pa_profile)) ** 3 * pa_profile).sum() / (pa_profile.sum() * ny ** 3))
    return f


def main():
    t0 = time.time()
    df = pd.read_csv(LABELS)
    uids = df.uid.tolist()
    y = df.is_pathologic.values.astype(int)
    missing = [u for u in uids if not (TRAIN_DIR / f"{u}.nii.gz").exists()]
    if missing:
        print(f"MISSING {len(missing)} scans — aborting (first: {missing[:3]})")
        sys.exit(1)

    print(f"Extracting features for {len(uids)} scans (sequential, ~1s each)...")
    rows = []
    for i, u in enumerate(uids):
        rows.append(features(load_vol(u)))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(uids)} ({el:.0f}s)", flush=True)

    X = pd.DataFrame(rows)
    order = list(X.columns)
    Xv = X.values
    print(f"Feature matrix: {Xv.shape}, {time.time()-t0:.0f}s")

    # --- CV ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    oof = np.zeros(len(y))
    for tr, va in skf.split(Xv, y):
        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
            eval_metric="logloss", n_jobs=4, random_state=77, verbosity=0,
        )
        clf.fit(Xv[tr], y[tr])
        oof[va] = clf.predict_proba(Xv[va])[:, 1]

    def ll(yv, pv):
        p = np.clip(pv, 1e-7, 1 - 1e-7)
        return float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))

    raw = ll(y, oof)
    base = ll(y, np.full(len(y), 0.5485))
    print(f"OOF LOG LOSS: {raw:.4f} (constant baseline {base:.4f})")

    # --- final model: calibrated, full data ---
    base_clf = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
        eval_metric="logloss", n_jobs=4, random_state=77, verbosity=0,
    )
    cal = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    cal.fit(Xv, y)

    # LGBM blend candidate
    lgb_oof = np.zeros(len(y))
    for tr, va in skf.split(Xv, y):
        m = lgb.LGBMClassifier(n_estimators=500, num_leaves=24, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                               random_state=77, verbose=-1, n_jobs=4)
        m.fit(Xv[tr], y[tr])
        lgb_oof[va] = m.predict_proba(Xv[va])[:, 1]
    ll_lgb = ll(y, lgb_oof)
    ll_blend = ll(y, 0.5 * oof + 0.5 * lgb_oof)
    print(f"LGBM OOF: {ll_lgb:.4f} | XGB+LGBM blend: {ll_blend:.4f} | raw XGB: {raw:.4f}")

    # Keep whichever is better (blend => average of the two saved models handled
    # at inference by saving both; spec carries a blend weight)
    ASSETS.mkdir(parents=True, exist_ok=True)
    import joblib
    blend_w = 0.5 if ll_blend < min(raw, ll_lgb) else 1.0
    if blend_w < 1.0:
        joblib.dump(cal, ASSETS / "sbr_model_xgb.pkl")
        lgb_full = lgb.LGBMClassifier(n_estimators=500, num_leaves=24, learning_rate=0.04,
                                      subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                                      random_state=77, verbose=-1, n_jobs=4)
        lgb_full.fit(Xv, y)
        joblib.dump(lgb_full, ASSETS / "sbr_model_lgb.pkl")
    else:
        better = cal if raw <= ll_lgb else lgb_full
        joblib.dump(better, ASSETS / "sbr_model.pkl")

    spec = {
        "feature_order": order,
        "blend_weight": blend_w,
        "models": ["sbr_model_xgb.pkl", "sbr_model_lgb.pkl"] if blend_w < 1.0 else ["sbr_model.pkl"],
        "oof_logloss": {"xgb": raw, "lgbm": ll_lgb, "blend": ll_blend},
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "n_train": len(uids),
    }
    (ASSETS / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    print("SAVED assets:", *sorted(p.name for p in ASSETS.iterdir()))
    print("DONE", f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
