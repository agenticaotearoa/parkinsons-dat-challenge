#!/usr/bin/env python3
# train_tier1_v4.py  --  tier-1 trainer, v4.1 (fix of the v4.0 regression)
#
# WHAT WAS WRONG IN v4.0 AND WHAT IS FIXED HERE (in order of importance)
#  1. OCCUPANCY gate (42.7% fail) was computed on the native scan with an own-intensity
#     percentile threshold -> "empty volume false alarm" bug.  Now: REGISTERED scan ->
#     scale-free normalisation (z-score, percentile stretch to 0..100) -> brain mask =
#     vn > OCC_NORM_THR -> coverage of the TEMPLATE mask (template mask built with the
#     identical rule) + inside/outside contrast.  Soft/hard thresholds are lenient; the
#     script ASSERTS ~>=99% pass and exits nonzero (GATE_MISCALIBRATED) otherwise.
#  2. EXTENT gate is applied to NATIVE scans only (never to the 118^3 registered grid),
#     with physically-possible windows, and is SOFT.
#  3. CALIBRATION: Platt is fitted on INNER-OOF bagged logits (not train-fold logits);
#     blend operates on PROBABILITIES (w*p_lgb_platt + (1-w)*p_spline) with w swept on
#     the outer OOF; mean(p)~prevalence and Brier<0.20 are asserted.
#  4. eps sweep re-clips the same probability array inside the loop and prints min/max p.
#  5. Gate rates: soft-fail <5%, hard-fail <1%, else GATE_MISCALIBRATED + exit 2.
#  6. Kept: label-free template (identical-means fallback), v3-geometry fallback,
#     feature-level TTA (5 jitters, patch mode), 3-seed bagging, adaptive ROI features
#     gated by a fold-0 ablation (dropped if they hurt fold-0 LL by > 0.005).
import os
import sys
import json
import time
import math
import argparse
import tempfile
import traceback

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import lightgbm as lgb
import joblib
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

# ----------------------------------------------------------------------------- constants
VERSION = "v4.1"
GRID = 118
SP = 2.5                                   # mm, isotropic registered grid
ORIGIN = -(GRID - 1) * SP / 2.0            # grid centred on 0 (RAS mm)
PREVALENCE_REF = 0.5485
CV_SEED = 42
SEEDS = [11, 22, 33]
N_FOLDS = 5
EPS_GRID = [0.005, 0.01, 0.02, 0.03, 0.05]
TTA_N = 5
TTA_ROT_DEG = 2.0
TTA_TRANS_MM = 2.0
TTA_SEED_BASE = 1000
REG_TIMEOUT_S = 90.0
OCC_NORM_THR = 12.0                        # on the 0..100 scale-free normalised volume
N_ROUNDS = 400
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=25,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  verbose=-1, num_threads=4)

GATES = dict(
    voxel_mm=(0.8, 6.0),
    native_extent_mm={"x": (80.0, 300.0), "y": (100.0, 300.0), "z": (60.0, 300.0)},
    occ_cov_soft=0.70, occ_cov_hard=0.40,
    occ_contrast_soft=1.5, occ_contrast_hard=1.1,
    ncc_floor=0.55, ncc_hard=0.30, ncc_thr=0.55,      # ncc_thr = max(0.55, P1-0.05) set at train time
    peak_dist_soft_mm=15.0, peak_dist_hard_mm=40.0,
    max_soft_rate=0.05, max_hard_rate=0.01,
)

# ROI boxes, mm offsets from the per-side striatal centroid: [lateral(+), anterior(+), superior(+)]
CLUSTER_DEFS = {
    "caud": {"off": [-5.0, 8.0, 2.0], "size": [10.0, 12.0, 10.0]},
    "put":  {"off": [6.0, -6.0, -1.0], "size": [10.0, 18.0, 10.0]},
    "search_mm": 6.0,
    "ref_dilate_mm": 10.0,
}
CLUSTER_RULE = ("boxes are mm offsets from the template per-side striatal centroid (RAS, lateral sign "
                "mirrored per side); ad_* re-centre each box on the local max of the 1-vox smoothed "
                "volume within +-search_mm; reference = template mask minus striatum dilated by "
                "ref_dilate_mm; SBR = mean(box)/mean(ref) - 1")

BASE_FEATURES = [
    "sbr_caud_L", "sbr_caud_R", "sbr_put_L", "sbr_put_R",
    "sbr_caud_min", "sbr_put_min", "sbr_min", "sbr_max", "sbr_mean",
    "asym_caud", "asym_put",
    "pc_ratio_L", "pc_ratio_R", "pc_ratio_min", "put_caud_diff_min",
    "peak_ratio_L", "peak_ratio_R", "peak_ratio_min",
    "striatal_extent_vox", "ncc_template", "occ_coverage",
]
AD_FEATURES = [
    "ad_sbr_caud_L", "ad_sbr_caud_R", "ad_sbr_put_L", "ad_sbr_put_R",
    "ad_sbr_put_min", "ad_asym_put", "ad_pc_ratio_min", "ad_shift_mm_max",
]
SPLINE_FEATURES = ["sbr_put_min", "sbr_caud_min", "asym_put", "pc_ratio_min", "sbr_mean"]


def mono_for(name):
    if name.startswith("asym") or name.startswith("ad_asym"):
        return 1
    if name in ("ncc_template", "occ_coverage", "ad_shift_mm_max"):
        return 0
    if (name.startswith("sbr_") or name.startswith("ad_sbr_") or name.startswith("peak_ratio")
            or name.startswith("pc_ratio") or name.startswith("ad_pc_ratio")
            or name.startswith("put_caud_diff") or name.startswith("striatal_extent")):
        return -1
    return 0


# ----------------------------------------------------------------------------- progressive writer
def write_submission_progressive(path, ids, probs, header=("id", "prob")):
    """Atomic full rewrite of the submission CSV (tmp + os.replace). Safe to call every N scans."""
    ids = list(ids)
    probs = np.asarray(probs, dtype=np.float64)
    if len(ids) != len(probs):
        raise ValueError("ids/probs length mismatch")
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".sub_", suffix=".csv", dir=d)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            f.write(",".join(header) + "\n")
            for i, p in zip(ids, probs):
                f.write("%s,%.6f\n" % (i, float(p)))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path


def _test_progressive_writer():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sub", "submission.csv")
        ids = ["a", "b", "c"]
        write_submission_progressive(p, ids, [PREVALENCE_REF] * 3)
        df = pd.read_csv(p)
        assert list(df.columns) == ["id", "prob"] and len(df) == 3
        assert np.allclose(df["prob"].values, PREVALENCE_REF, atol=1e-6)
        write_submission_progressive(p, ids, [0.1, 0.2, 0.3])
        df = pd.read_csv(p)
        assert np.allclose(df["prob"].values, [0.1, 0.2, 0.3], atol=1e-6)
        assert not [f for f in os.listdir(os.path.dirname(p)) if f.startswith(".sub_")]
    return True


# ----------------------------------------------------------------------------- IO / normalisation
def canonicalize_ras(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    while vol.ndim > 3:
        vol = vol[..., 0]
    aff = img.affine
    zooms = np.linalg.norm(aff[:3, :3], axis=0).astype(np.float64)
    origin = aff[:3, 3].astype(np.float64)
    return vol, zooms, origin


def norm100(vol):
    """Scale-free: z-score over the volume, then stretch [p1, p99.5] -> [0, 100]."""
    v = np.asarray(vol, dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(v.std())
    if not np.isfinite(s) or s < 1e-9:
        return np.zeros_like(v)
    z = (v - float(v.mean())) / s
    lo, hi = np.percentile(z, [1.0, 99.5])
    if hi - lo < 1e-6:
        return np.zeros_like(v)
    return np.clip(100.0 * (z - lo) / (hi - lo), 0.0, 100.0).astype(np.float32)


def _largest_component(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def native_extent_mm(vol, zooms):
    m = _largest_component(norm100(vol) > OCC_NORM_THR)
    if not m.any():
        return [0.0, 0.0, 0.0]
    sl = ndi.find_objects(m.astype(np.int8))[0]
    return [float((s.stop - s.start) * z) for s, z in zip(sl, zooms)]


def pre_gates(vol, zooms, gates=GATES):
    hard, soft = [], []
    finite = np.isfinite(vol)
    meta = {"nan_frac": float(1.0 - finite.mean()), "voxel": [float(z) for z in zooms]}
    if finite.mean() < 0.5:
        hard.append("nan_flat")
    vol = np.where(finite, vol, 0.0).astype(np.float32)
    if float(vol.std()) < 1e-9 or vol.size < 1000:
        if "nan_flat" not in hard:
            hard.append("nan_flat")
    lo, hi = gates["voxel_mm"]
    if any((z < lo or z > hi or not np.isfinite(z)) for z in zooms):
        hard.append("voxel")
    ext = [0.0, 0.0, 0.0]
    if not hard:
        ext = native_extent_mm(vol, zooms)
        win = gates["native_extent_mm"]
        for ax, e in zip("xyz", ext):
            a, b = win[ax]
            if e < a or e > b:
                soft.append("extent")
                break
    meta["native_extent_mm"] = ext
    return vol, hard, soft, meta


# ----------------------------------------------------------------------------- template
def _sitk_from(vol, zooms, origin):
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32))
    img.SetSpacing(tuple(float(z) for z in zooms))
    img.SetOrigin(tuple(float(o) for o in origin))
    return img


def _grid_image(arr):
    return _sitk_from(arr, (SP, SP, SP), (ORIGIN, ORIGIN, ORIGIN))


def _mask_centroid_extent(mask, zooms, origin):
    idx = np.argwhere(mask)
    c = idx.mean(axis=0) * np.asarray(zooms) + np.asarray(origin)
    ext = (idx.max(axis=0) - idx.min(axis=0) + 1) * np.asarray(zooms)
    return c, ext


def geometry_fallback_transform(mov_mask, zooms, origin, fix_mask):
    """v3 geometry fallback: centroid alignment + isotropic scale from mask extents (fixed->moving map)."""
    if not mov_mask.any() or not fix_mask.any():
        return sitk.Euler3DTransform()
    c_m, e_m = _mask_centroid_extent(mov_mask, zooms, origin)
    c_f, e_f = _mask_centroid_extent(fix_mask, (SP, SP, SP), (ORIGIN, ORIGIN, ORIGIN))
    s = float(np.clip(np.median(e_f / np.maximum(e_m, 1e-3)), 0.75, 1.35))
    tx = sitk.AffineTransform(3)
    tx.SetCenter(tuple(float(x) for x in c_f))
    tx.SetMatrix(tuple((np.eye(3) / s).ravel().tolist()))
    tx.SetTranslation(tuple(float(x) for x in (c_m - c_f)))
    return tx


def make_template_struct(T_raw):
    tvol = norm100(T_raw)
    tmask = tvol > OCC_NORM_THR
    if tmask.sum() < 1000:
        tmask = tvol > np.percentile(tvol, 60)
    lo, hi = int(GRID * 0.25), int(GRID * 0.75)
    central = np.zeros_like(tmask)
    central[lo:hi, lo:hi, lo:hi] = True
    thr = np.percentile(tvol[tmask], 98.5)
    blob = (tvol > thr) & tmask & central
    lab, n = ndi.label(blob)
    cs = []
    if n >= 2:
        sizes = ndi.sum(blob, lab, index=np.arange(1, n + 1))
        top = np.argsort(sizes)[::-1][:2] + 1
        cs = [np.array(ndi.center_of_mass(blob, lab, int(t))) for t in top]
        blob = np.isin(lab, top)
    elif n == 1:
        idx = np.argwhere(blob)
        xm = np.median(idx[:, 0])
        cs = [idx[idx[:, 0] <= xm].mean(axis=0), idx[idx[:, 0] > xm].mean(axis=0)]
        cs = [c if np.all(np.isfinite(c)) else idx.mean(axis=0) for c in cs]
    else:
        ctr = (GRID - 1) / 2.0
        cs = [np.array([ctr - 12.0 / SP, ctr + 2.0 / SP, ctr]), np.array([ctr + 12.0 / SP, ctr + 2.0 / SP, ctr])]
        blob = np.zeros_like(tmask)
        for c in cs:
            i = np.round(c).astype(int)
            blob[i[0] - 2:i[0] + 3, i[1] - 3:i[1] + 4, i[2] - 2:i[2] + 3] = True
    cs = sorted(cs, key=lambda c: c[0])
    cL, cR = np.asarray(cs[0], float), np.asarray(cs[1], float)
    stri = ndi.binary_dilation(blob, iterations=2)
    stri_dil = ndi.binary_dilation(blob, iterations=int(round(CLUSTER_DEFS["ref_dilate_mm"] / SP)))
    ref_mask = tmask & ~stri_dil
    if ref_mask.sum() < 500:
        ref_mask = tmask
    return dict(T_raw=np.asarray(T_raw, np.float32), tvol=tvol, tmask=tmask, cL=cL, cR=cR,
                mid=(cL + cR) / 2.0, stri=stri, stri_dil=stri_dil, ref_mask=ref_mask)


def save_template(tpl, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "template_v4_raw.npy"), tpl["T_raw"])
    np.save(os.path.join(out_dir, "template_v4.npy"), tpl["tvol"])
    np.save(os.path.join(out_dir, "template_mask_v4.npy"), tpl["tmask"])


def load_template(assets_dir):
    return make_template_struct(np.load(os.path.join(assets_dir, "template_v4_raw.npy")))


def _zscore_in(v, m):
    x = v[m]
    return (v - float(x.mean())) / (float(x.std()) + 1e-6)


def _geometry_place(vol, zooms, origin, target_ext_y_mm=165.0):
    """Template bootstrap: put scan brain centroid at grid centre, scale AP extent to target."""
    m = _largest_component(norm100(vol) > OCC_NORM_THR)
    if not m.any():
        return None
    c_m, e_m = _mask_centroid_extent(m, zooms, origin)
    s = float(np.clip(target_ext_y_mm / max(e_m[1], 1e-3), 0.75, 1.35))
    tx = sitk.AffineTransform(3)
    tx.SetCenter((0.0, 0.0, 0.0))
    tx.SetMatrix(tuple((np.eye(3) / s).ravel().tolist()))
    tx.SetTranslation(tuple(float(x) for x in c_m))
    mov = _sitk_from(np.clip(vol, 0, None), zooms, origin)
    fix = _grid_image(np.zeros((GRID, GRID, GRID), np.float32))
    res = sitk.Resample(mov, fix, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(res).transpose(2, 1, 0)


def _load_for_template(path):
    try:
        vol, zooms, origin = canonicalize_ras(path)
        vol, hard, _, _ = pre_gates(vol, zooms)
        if hard:
            return None
        return vol, zooms, origin
    except Exception:
        return None


def build_template(paths, n_ref=150, seed=CV_SEED, n_jobs=4):
    """Label-free template. Round 0: geometry placement + identical-means average (each scan
    z-scored within its own brain so every scan contributes with identical mean/std).
    Round 1: rigid MI registration to round-0 template, identical-means re-average."""
    rng = np.random.default_rng(seed)
    sel = [paths[i] for i in rng.permutation(len(paths))[:min(n_ref, len(paths))]]
    loaded = [x for x in Parallel(n_jobs=n_jobs)(delayed(_load_for_template)(p) for p in sel) if x is not None]
    acc = np.zeros((GRID, GRID, GRID), np.float64)
    n = 0
    for vol, zooms, origin in loaded:
        g = _geometry_place(vol, zooms, origin)
        if g is None:
            continue
        m = norm100(g) > OCC_NORM_THR
        if m.sum() < 1000:
            continue
        acc += _zscore_in(g, m)
        n += 1
    if n == 0:
        raise RuntimeError("template bootstrap failed: no usable scans")
    T0 = (acc / n).astype(np.float32)
    tpl0 = make_template_struct(T0)
    acc = np.zeros_like(acc)
    n = 0

    def _one(x):
        vol, zooms, origin = x
        try:
            reg, meta = register_to_template(vol, zooms, origin, tpl0, timeout_s=REG_TIMEOUT_S)
            if meta.get("ncc", 0.0) < 0.3:
                return None
            return reg
        except Exception:
            return None
    regs = Parallel(n_jobs=n_jobs)(delayed(_one)(x) for x in loaded)
    for r in regs:
        if r is None:
            continue
        acc += _zscore_in(np.clip(r, 0, None), tpl0["tmask"])
        n += 1
    if n < 10:
        print("[template] round-1 registration weak (n=%d); using identical-means round-0 template" % n)
        return tpl0
    return make_template_struct((acc / n).astype(np.float32))


# ----------------------------------------------------------------------------- registration
def _ncc(a, b):
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    return float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def post_metrics(reg_vol, tpl):
    """Fixed occupancy (scale-free, registered space, vs template mask), NCC, peak distance."""
    vn = norm100(reg_vol)
    smask = vn > OCC_NORM_THR
    tmask = tpl["tmask"]
    cov = float(smask[tmask].mean())
    inside = float(np.median(vn[tmask]))
    outside = float(np.median(vn[~tmask]))
    contrast = inside / (outside + 1.0)
    region = tpl["ref_mask"]
    ncc = _ncc(reg_vol[region], tpl["tvol"][region])
    sm = ndi.gaussian_filter(np.clip(reg_vol, 0, None), 1.0)
    m = np.where(tmask, sm, -np.inf)
    pk = np.array(np.unravel_index(int(np.argmax(m)), m.shape), float)
    dist = float(np.linalg.norm(pk - tpl["mid"]) * SP)
    return dict(occ_coverage=cov, occ_contrast=float(contrast), ncc=ncc, peak_dist_mm=dist)


def _rigid_register(mov_img, fix_img, timeout_s):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20, 12345)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3, numberOfIterations=150,
                                                 relaxationFactor=0.6, gradientMagnitudeTolerance=1e-5)
    reg.SetOptimizerScalesFromPhysicalShift()
    init = sitk.CenteredTransformInitializer(fix_img, mov_img, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg.SetInitialTransform(init, inPlace=True)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    t0 = time.time()
    state = {"timeout": False}

    def _cb():
        if time.time() - t0 > timeout_s:
            state["timeout"] = True
            reg.StopOptimization()
    reg.AddCommand(sitk.sitkIterationEvent, _cb)
    tx = reg.Execute(fix_img, mov_img)
    return tx, state["timeout"], float(reg.GetMetricValue())


def register_to_template(vol, zooms, origin, tpl, timeout_s=REG_TIMEOUT_S):
    t0 = time.time()
    mov = _sitk_from(np.clip(vol, 0, None), zooms, origin)
    fix = _grid_image(tpl["tvol"])
    cands = []
    meta = {"reg_method": None, "reg_timeout": False, "reg_error": None}
    try:
        tx, to, mval = _rigid_register(mov, fix, timeout_s)
        res = sitk.Resample(mov, fix, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        r = sitk.GetArrayFromImage(res).transpose(2, 1, 0).astype(np.float32)
        pm = post_metrics(r, tpl)
        cands.append(("rigid_mi", r, pm))
        meta["reg_timeout"] = bool(to)
        meta["reg_metric"] = mval
    except Exception as e:  # noqa
        meta["reg_error"] = str(e)[:200]
    if not cands or cands[0][2]["ncc"] < 0.5 or meta["reg_timeout"]:
        try:
            mmask = _largest_component(norm100(vol) > OCC_NORM_THR)
            tx = geometry_fallback_transform(mmask, zooms, origin, tpl["tmask"])
            res = sitk.Resample(mov, fix, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            r = sitk.GetArrayFromImage(res).transpose(2, 1, 0).astype(np.float32)
            cands.append(("v3_geometry", r, post_metrics(r, tpl)))
        except Exception as e:  # noqa
            meta["reg_error"] = (meta["reg_error"] or "") + " | fallback: " + str(e)[:200]
    if not cands:
        raise RuntimeError("registration failed: %s" % meta["reg_error"])
    name, r, pm = max(cands, key=lambda c: c[2]["ncc"])
    meta["reg_method"] = name
    meta["reg_time_s"] = float(time.time() - t0)
    meta.update(pm)
    return r, meta


# ----------------------------------------------------------------------------- features
def _box(center, size_mm):
    half = np.asarray(size_mm, float) / (2.0 * SP)
    lo = np.maximum(np.round(np.asarray(center) - half).astype(int), 0)
    hi = np.minimum(np.round(np.asarray(center) + half).astype(int) + 1, GRID)
    if np.any(hi <= lo):
        return None
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _box_stat(vol, center, size_mm, fn):
    sl = _box(center, size_mm)
    if sl is None:
        return float("nan")
    return float(fn(vol[sl]))


def _rot_matrix(angles_deg):
    a, b, c = np.deg2rad(angles_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def extract_features(reg_vol, tpl, jitter=None, meta_extra=None):
    """jitter = (R 3x3, t_mm) applied in 'patch mode' to ROI geometry about the striatal midpoint."""
    vol = np.clip(np.nan_to_num(reg_vol), 0, None).astype(np.float32)
    sm = ndi.gaussian_filter(vol, 1.0)
    ref = float(vol[tpl["ref_mask"]].mean()) + 1e-6
    R, t = (np.eye(3), np.zeros(3)) if jitter is None else jitter
    mid = tpl["mid"]
    search = CLUSTER_DEFS["search_mm"]
    f = {}
    for side, mx in (("L", -1.0), ("R", 1.0)):
        c_side = mid + R @ (tpl["c" + side] - mid) + t / SP
        for roi in ("caud", "put"):
            d = CLUSTER_DEFS[roi]
            off = np.array([mx * d["off"][0], d["off"][1], d["off"][2]]) / SP
            center = c_side + R @ off
            f["sbr_%s_%s" % (roi, side)] = _box_stat(vol, center, d["size"], np.mean) / ref - 1.0
            f["peak_%s_%s" % (roi, side)] = _box_stat(sm, center, d["size"], np.max) / ref - 1.0
            sl = _box(center, [2 * search] * 3)
            if sl is None:
                newc, shift = center, 0.0
            else:
                sub = sm[sl]
                i = np.array(np.unravel_index(int(np.argmax(sub)), sub.shape), float)
                newc = np.array([s.start for s in sl], float) + i
                dv = np.clip(newc - center, -search / SP, search / SP)
                newc = center + dv
                shift = float(np.linalg.norm(dv) * SP)
            f["ad_sbr_%s_%s" % (roi, side)] = _box_stat(vol, newc, d["size"], np.mean) / ref - 1.0
            f["ad_shift_%s_%s" % (roi, side)] = shift

    def asym(a, b):
        return abs(a - b) / (abs(a + b + 2.0) + 1e-6)
    sbrs = [f["sbr_caud_L"], f["sbr_caud_R"], f["sbr_put_L"], f["sbr_put_R"]]
    f["sbr_caud_min"] = min(f["sbr_caud_L"], f["sbr_caud_R"])
    f["sbr_put_min"] = min(f["sbr_put_L"], f["sbr_put_R"])
    f["sbr_min"] = float(np.nanmin(sbrs))
    f["sbr_max"] = float(np.nanmax(sbrs))
    f["sbr_mean"] = float(np.nanmean(sbrs))
    f["asym_caud"] = asym(f["sbr_caud_L"], f["sbr_caud_R"])
    f["asym_put"] = asym(f["sbr_put_L"], f["sbr_put_R"])
    for s in ("L", "R"):
        f["pc_ratio_%s" % s] = (1.0 + f["sbr_put_%s" % s]) / (1.0 + f["sbr_caud_%s" % s] + 1e-6)
        f["peak_ratio_%s" % s] = f["peak_put_%s" % s]
    f["pc_ratio_min"] = min(f["pc_ratio_L"], f["pc_ratio_R"])
    f["put_caud_diff_min"] = min(f["sbr_put_L"] - f["sbr_caud_L"], f["sbr_put_R"] - f["sbr_caud_R"])
    f["peak_ratio_min"] = min(f["peak_ratio_L"], f["peak_ratio_R"])
    f["striatal_extent_vox"] = float(((vol[tpl["stri_dil"]] / ref - 1.0) > 1.0).sum())
    f["ad_sbr_put_min"] = min(f["ad_sbr_put_L"], f["ad_sbr_put_R"])
    f["ad_asym_put"] = asym(f["ad_sbr_put_L"], f["ad_sbr_put_R"])
    f["ad_pc_ratio_min"] = min((1.0 + f["ad_sbr_put_L"]) / (1.0 + f["ad_sbr_caud_L"] + 1e-6),
                               (1.0 + f["ad_sbr_put_R"]) / (1.0 + f["ad_sbr_caud_R"] + 1e-6))
    f["ad_shift_mm_max"] = max(f["ad_shift_caud_L"], f["ad_shift_caud_R"], f["ad_shift_put_L"], f["ad_shift_put_R"])
    if meta_extra:
        f["ncc_template"] = float(meta_extra.get("ncc", np.nan))
        f["occ_coverage"] = float(meta_extra.get("occ_coverage", np.nan))
    return {k: f[k] for k in BASE_FEATURES + AD_FEATURES if k in f}


def extract_features_tta(reg_vol, tpl, meta_extra=None, n=TTA_N, rot_deg=TTA_ROT_DEG,
                         trans_mm=TTA_TRANS_MM, seed_base=TTA_SEED_BASE):
    """Feature-level TTA: identity + (n-1) random rigid jitters of the ROI geometry (patch mode)."""
    if meta_extra is None:
        meta_extra = post_metrics(reg_vol, tpl)
    jitters = [None]
    for j in range(1, n):
        rng = np.random.default_rng(seed_base + j)
        jitters.append((_rot_matrix(rng.uniform(-rot_deg, rot_deg, 3)), rng.uniform(-trans_mm, trans_mm, 3)))
    rows = [extract_features(reg_vol, tpl, jit, meta_extra) for jit in jitters]
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def feature_vector(fd, feature_order, fill):
    x = np.array([fd.get(k, np.nan) for k in feature_order], dtype=np.float64)
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.array([fill[k] for k in feature_order], dtype=np.float64)[bad]
    return x.reshape(1, -1)


# ----------------------------------------------------------------------------- gates
def evaluate_gates_one(meta, gates=GATES):
    hard = list(meta.get("pre_hard", []))
    soft = list(meta.get("pre_soft", []))
    if hard:
        return hard, soft
    if meta.get("failed"):
        hard.append("registration")
        return hard, soft
    cov, con = meta.get("occ_coverage", 0.0), meta.get("occ_contrast", 0.0)
    if cov < gates["occ_cov_hard"] or con < gates["occ_contrast_hard"]:
        hard.append("occupancy")
    elif cov < gates["occ_cov_soft"] or con < gates["occ_contrast_soft"]:
        soft.append("occupancy")
    ncc = meta.get("ncc", -1.0)
    if ncc < gates["ncc_hard"]:
        hard.append("ncc")
    elif ncc < gates["ncc_thr"]:
        soft.append("ncc")
    pd_ = meta.get("peak_dist_mm", 1e9)
    if pd_ > gates["peak_dist_hard_mm"]:
        hard.append("peak")
    elif pd_ > gates["peak_dist_soft_mm"]:
        soft.append("peak")
    return hard, soft


def apply_gates(p, hard, soft, prevalence):
    p = np.array(p, dtype=np.float64)
    p = np.where(soft & ~hard, prevalence + 0.5 * (p - prevalence), p)
    p = np.where(hard, prevalence, p)
    return p


# ----------------------------------------------------------------------------- per-scan pipeline
def process_scan(sid, path, tpl, cache_dir):
    t0 = time.time()
    meta = {"id": sid, "failed": False, "pre_hard": [], "pre_soft": []}
    feats = {}
    try:
        vol, zooms, origin = canonicalize_ras(path)
        vol, hard, soft, pm = pre_gates(vol, zooms)
        meta.update(pm)
        meta["pre_hard"], meta["pre_soft"] = hard, soft
        if hard:
            meta["failed"] = True
            return sid, feats, meta
        cache = os.path.join(cache_dir, "%s.npz" % sid) if cache_dir else None
        reg = None
        if cache and os.path.exists(cache):
            try:
                z = np.load(cache, allow_pickle=False)
                reg = z["reg"].astype(np.float32)
                rmeta = json.loads(str(z["meta"]))
                rmeta.update(post_metrics(reg, tpl))
            except Exception:
                reg = None
        if reg is None:
            reg, rmeta = register_to_template(vol, zooms, origin, tpl)
            if cache:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez_compressed(cache, reg=reg, meta=json.dumps(rmeta))
        meta.update(rmeta)
        feats = extract_features_tta(reg, tpl, meta_extra=rmeta)
    except Exception as e:  # noqa
        meta["failed"] = True
        meta["error"] = str(e)[:300]
    meta["wall_s"] = float(time.time() - t0)
    return sid, feats, meta


# ----------------------------------------------------------------------------- modelling
def _ll(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_lgb(X, y, mono, seed):
    params = dict(LGB_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                  monotone_constraints=[int(m) for m in mono])
    return lgb.train(params, lgb.Dataset(X, label=y, free_raw_data=False), num_boost_round=N_ROUNDS)


def raw_bag(boosters, X):
    return np.mean([b.predict(X, raw_score=True) for b in boosters], axis=0).astype(np.float64)


def fit_platt(raw, y):
    lr = LogisticRegression(C=1e4, max_iter=2000).fit(raw.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def make_spline():
    return make_pipeline(StandardScaler(),
                         SplineTransformer(n_knots=5, degree=3, extrapolation="linear"),
                         LogisticRegression(C=0.3, max_iter=5000))


def fit_fold(Xtr, ytr, Xva, mono, spl_idx, inner_seed):
    """Platt on INNER-OOF bagged logits (fixes the train-fold/OOF distribution mismatch)."""
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=inner_seed)
    inner_raw = np.zeros(len(ytr))
    for itr, iva in inner.split(Xtr, ytr):
        bs = [train_lgb(Xtr[itr], ytr[itr], mono, s) for s in SEEDS]
        inner_raw[iva] = raw_bag(bs, Xtr[iva])
    a, b = fit_platt(inner_raw, ytr)
    boosters = [train_lgb(Xtr, ytr, mono, s) for s in SEEDS]
    raw_va = raw_bag(boosters, Xva)
    p_lgb = expit(a * raw_va + b)
    spline = make_spline().fit(Xtr[:, spl_idx], ytr)
    p_spl = spline.predict_proba(Xva[:, spl_idx])[:, 1]
    return dict(boosters=boosters, platt=(a, b), spline=spline, p_lgb=p_lgb, p_spl=p_spl, raw=raw_va)


def run_cv(X, y, feature_names, only_fold0=False):
    mono = [mono_for(n) for n in feature_names]
    spl_idx = [feature_names.index(n) for n in SPLINE_FEATURES if n in feature_names]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    p_lgb = np.full(len(y), np.nan)
    p_spl = np.full(len(y), np.nan)
    folds = []
    for k, (tr, va) in enumerate(skf.split(X, y)):
        r = fit_fold(X[tr], y[tr], X[va], mono, spl_idx, CV_SEED + 7 + k)
        p_lgb[va], p_spl[va] = r["p_lgb"], r["p_spl"]
        folds.append(r)
        print("  fold %d: n_tr=%d n_va=%d  LL_lgb=%.4f LL_spl=%.4f platt=(%.3f,%.3f)"
              % (k, len(tr), len(va), _ll(y[va], r["p_lgb"]), _ll(y[va], r["p_spl"]), *r["platt"]), flush=True)
        if only_fold0:
            return dict(p_lgb=p_lgb, p_spl=p_spl, folds=folds, va0=va, mono=mono, spl_idx=spl_idx)
    return dict(p_lgb=p_lgb, p_spl=p_spl, folds=folds, mono=mono, spl_idx=spl_idx)


# ----------------------------------------------------------------------------- data
def load_manifest(path, max_scans=0):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    idc = cols.get("id") or cols.get("scan_id") or cols.get("subject") or df.columns[0]
    pc = cols.get("path") or cols.get("file") or cols.get("filepath")
    lc = cols.get("label") or cols.get("y") or cols.get("target")
    if pc is None or lc is None:
        raise ValueError("manifest needs path and label columns; got %s" % list(df.columns))
    base = os.path.dirname(os.path.abspath(path))
    out = pd.DataFrame({"id": df[idc].astype(str), "path": df[pc].astype(str), "y": df[lc].astype(int)})
    out["path"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in out["path"]]
    if max_scans:
        out = out.iloc[:max_scans].reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/train_manifest.csv")
    ap.add_argument("--out", default="assets_v4")
    ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--template_v3", default="", help="optional raw template npy (118^3) to reuse")
    ap.add_argument("--n_jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--max_scans", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)

    assert _test_progressive_writer()
    print("[ok] progressive writer unit test passed")

    df = load_manifest(args.manifest, args.max_scans)
    y_all = df["y"].values.astype(int)
    prevalence = float(y_all.mean())
    print("[data] n=%d prevalence=%.4f (ref %.4f)" % (len(df), prevalence, PREVALENCE_REF))

    # --- template (label-free)
    traw = os.path.join(args.out, "template_v4_raw.npy")
    if args.template_v3 and os.path.exists(args.template_v3):
        tpl = make_template_struct(np.load(args.template_v3).astype(np.float32))
        print("[template] reused %s" % args.template_v3)
    elif os.path.exists(traw):
        tpl = load_template(args.out)
        print("[template] loaded cached %s" % traw)
    else:
        print("[template] building label-free template ...", flush=True)
        tpl = build_template(df["path"].tolist(), n_jobs=args.n_jobs)
    save_template(tpl, args.out)
    print("[template] mask vox=%d cL=%s cR=%s" % (int(tpl["tmask"].sum()), np.round(tpl["cL"], 1), np.round(tpl["cR"], 1)))

    # --- per-scan registration + TTA features
    print("[scans] processing %d scans with %d jobs ..." % (len(df), args.n_jobs), flush=True)
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_scan)(sid, p, tpl, args.cache) for sid, p in zip(df["id"], df["path"]))
    order = {sid: i for i, sid in enumerate(df["id"])}
    res.sort(key=lambda r: order[r[0]])
    metas = [r[2] for r in res]
    feat_df = pd.DataFrame([r[1] for r in res]).reindex(columns=BASE_FEATURES + AD_FEATURES)
    walls = np.array([m.get("wall_s", np.nan) for m in metas])
    print("[scans] wall/scan median=%.1fs p90=%.1fs  methods=%s" % (
        np.nanmedian(walls), np.nanpercentile(walls, 90),
        pd.Series([m.get("reg_method") for m in metas]).value_counts().to_dict()))

    # --- gates (NCC threshold from training distribution)
    ncc_vals = np.array([m.get("ncc", np.nan) for m in metas], float)
    P1 = float(np.nanpercentile(ncc_vals, 1.0))
    gates = dict(GATES)
    gates["ncc_thr"] = float(max(gates["ncc_floor"], P1 - 0.05))
    print("[gates] NCC P1=%.3f -> ncc_thr=%.3f" % (P1, gates["ncc_thr"]))
    hard_l, soft_l = zip(*[evaluate_gates_one(m, gates) for m in metas])
    hard = np.array([len(h) > 0 for h in hard_l])
    soft = np.array([len(s) > 0 for s in soft_l]) & ~hard
    gate_table = {}
    for g in ["nan_flat", "voxel", "extent", "registration", "occupancy", "ncc", "peak"]:
        fail = np.array([(g in h) or (g in s) for h, s in zip(hard_l, soft_l)])
        gate_table[g] = dict(n_fail=int(fail.sum()), frac=float(fail.mean()),
                             pos_frac_among_fail=float(y_all[fail].mean()) if fail.any() else float("nan"))
        print("  gate %-12s fail=%4d (%.1f%%) pos_frac_among_fail=%.3f" % (
            g, gate_table[g]["n_fail"], 100 * gate_table[g]["frac"], gate_table[g]["pos_frac_among_fail"]))
    occ = np.array([m.get("occ_coverage", np.nan) for m in metas], float)
    con = np.array([m.get("occ_contrast", np.nan) for m in metas], float)
    pk = np.array([m.get("peak_dist_mm", np.nan) for m in metas], float)
    print("  occupancy coverage p1/p5/p50=%.3f/%.3f/%.3f  contrast p1/p50=%.2f/%.2f  peak_dist p95=%.1fmm" % (
        np.nanpercentile(occ, 1), np.nanpercentile(occ, 5), np.nanpercentile(occ, 50),
        np.nanpercentile(con, 1), np.nanpercentile(con, 50), np.nanpercentile(pk, 95)))
    hard_rate, soft_rate = float(hard.mean()), float(soft.mean())
    print("[gates] hard=%d (%.2f%%) soft=%d (%.2f%%)" % (hard.sum(), 100 * hard_rate, soft.sum(), 100 * soft_rate))
    if hard_rate > gates["max_hard_rate"] or soft_rate > gates["max_soft_rate"]:
        print("GATE_MISCALIBRATED hard=%.4f (max %.2f) soft=%.4f (max %.2f)" % (
            hard_rate, gates["max_hard_rate"], soft_rate, gates["max_soft_rate"]))
        sys.exit(2)

    # --- feature matrices
    keep = ~hard
    fill = {k: float(np.nanmedian(feat_df.loc[keep, k].values)) for k in feat_df.columns}
    fill = {k: (v if np.isfinite(v) else 0.0) for k, v in fill.items()}
    Xdf = feat_df.fillna(value=fill)
    y = y_all[keep]
    idx_keep = np.where(keep)[0]

    # --- ablation (fold 0 only): ad_* kept vs dropped
    print("[ablation] fold-0, ad_* kept vs dropped")
    X_full = Xdf[BASE_FEATURES + AD_FEATURES].values[keep]
    X_base = Xdf[BASE_FEATURES].values[keep]
    r_full = run_cv(X_full, y, BASE_FEATURES + AD_FEATURES, only_fold0=True)
    r_base = run_cv(X_base, y, BASE_FEATURES, only_fold0=True)
    va0 = r_full["va0"]
    ll_full = _ll(y[va0], r_full["p_lgb"][va0])
    ll_base = _ll(y[va0], r_base["p_lgb"][va0])
    use_ad = (ll_full - ll_base) <= 0.005
    print("[ablation] fold-0 LL with ad_*=%.4f without=%.4f -> %s" % (
        ll_full, ll_base, "KEEP ad_*" if use_ad else "DROP ad_* (hurt > 0.005)"))
    feature_order = BASE_FEATURES + (AD_FEATURES if use_ad else [])
    X = Xdf[feature_order].values[keep]

    # --- full CV
    print("[cv] %d folds x %d seeds, inner-OOF Platt, features=%d" % (N_FOLDS, len(SEEDS), len(feature_order)))
    cv = run_cv(X, y, feature_order)
    p_lgb, p_spl = cv["p_lgb"], cv["p_spl"]

    # --- blend on PROBABILITIES, w swept on OOF
    ws = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    lls = [_ll(y, w * p_lgb + (1 - w) * p_spl) for w in ws]
    w = float(ws[int(np.argmin(lls))])
    p_oof_keep = w * p_lgb + (1 - w) * p_spl
    print("[blend] w=%.2f  LL_lgb=%.4f LL_spl=%.4f LL_blend=%.4f" % (w, _ll(y, p_lgb), _ll(y, p_spl), _ll(y, p_oof_keep)))

    # --- OOF over all scans (hard-gated -> prevalence), calibration assertions
    p_oof = np.full(len(y_all), prevalence)
    p_oof[idx_keep] = p_oof_keep
    ll_raw = _ll(y_all, p_oof)
    auc = float(roc_auc_score(y_all, p_oof))
    brier = float(brier_score_loss(y_all, p_oof))
    mean_p = float(p_oof.mean())
    print("[oof] LL=%.4f AUC=%.4f Brier=%.4f mean(p)=%.4f prevalence=%.4f" % (ll_raw, auc, brier, mean_p, prevalence))
    if abs(mean_p - prevalence) > 0.02 or brier >= 0.20:
        print("CALIBRATION_BROKEN mean_p=%.4f prevalence=%.4f brier=%.4f" % (mean_p, prevalence, brier))
        sys.exit(3)

    # --- eps sweep: re-clip the SAME array per eps
    print("[eps] min(p)=%.5f max(p)=%.5f" % (p_oof.min(), p_oof.max()))
    eps_sweep = {}
    for e in EPS_GRID:
        pe = np.clip(p_oof.copy(), e, 1 - e)
        eps_sweep[str(e)] = _ll(y_all, pe)
        print("  eps=%.3f LL=%.5f n_clipped=%d" % (e, eps_sweep[str(e)], int(((p_oof < e) | (p_oof > 1 - e)).sum())))
    eps = float(min(EPS_GRID, key=lambda e: eps_sweep[str(e)]))

    # --- gated OOF
    p_gated = np.clip(apply_gates(p_oof, hard, soft, prevalence), eps, 1 - eps)
    ll_gated = _ll(y_all, p_gated)
    print("[gated] LL=%.4f (raw clipped %.4f) eps=%.3f" % (ll_gated, _ll(y_all, np.clip(p_oof, eps, 1 - eps)), eps))

    # --- save assets
    model_files, spline_files, platt = {}, {}, {}
    for k, r in enumerate(cv["folds"]):
        model_files[str(k)] = []
        for s, b in zip(SEEDS, r["boosters"]):
            fn = os.path.join(args.out, "lgb_v4_fold%d_seed%d.txt" % (k, s))
            b.save_model(fn)
            model_files[str(k)].append(os.path.basename(fn))
        fn = os.path.join(args.out, "spline_v4_fold%d.joblib" % k)
        joblib.dump(r["spline"], fn)
        spline_files[str(k)] = os.path.basename(fn)
        platt[str(k)] = [float(r["platt"][0]), float(r["platt"][1])]
    spec = dict(
        version=VERSION,
        feature_order=feature_order,
        blend={"w": w, "space": "probability", "rule": "p = w*sigmoid(a_k*raw_k+b_k) + (1-w)*spline_k.predict_proba"},
        cluster_defs=CLUSTER_DEFS,
        cluster_rule=CLUSTER_RULE,
        cv_seed=CV_SEED,
        eps=eps,
        gates={k: (list(v) if isinstance(v, tuple) else v) for k, v in gates.items()},
        inference=dict(
            rule=("canonical RAS -> pre-gates -> rigid Euler3D (GEOMETRY init, Mattes MI multi-res, 90s) "
                  "with v3-geometry fallback (pick higher NCC) -> post-gates -> TTA 5 jitters feature-avg "
                  "(patch mode, rot 2deg, trans 2mm, seed_base 1000) -> per fold: mean raw logit over 3 seeds "
                  "-> platt(fold) -> w*p_lgb + (1-w)*p_spline -> mean over 5 folds -> hard gate: tier-2 "
                  "heuristic else prevalence; soft gate: p = prev + 0.5*(p-prev) -> clip [eps, 1-eps]"),
            fill_values={k: fill[k] for k in feature_order},
            spline_features=[n for n in SPLINE_FEATURES if n in feature_order],
            spline_feature_idx=[feature_order.index(n) for n in SPLINE_FEATURES if n in feature_order],
            tta=dict(n=TTA_N, rot_deg=TTA_ROT_DEG, trans_mm=TTA_TRANS_MM, seed_base=TTA_SEED_BASE, mode="patch"),
            occ_norm_thr=OCC_NORM_THR,
            tier2=dict(prev=PREVALENCE_REF, max_shift=0.15, r_mid=2.5),
            grid=dict(n=GRID, spacing_mm=SP, origin_mm=ORIGIN),
        ),
        model_files=model_files,
        mono={n: mono_for(n) for n in feature_order},
        n_features=len(feature_order),
        n_folds=N_FOLDS,
        platt=platt,
        prevalence=prevalence,
        progressive_writer_test=True,
        results=dict(oof_ll=ll_raw, oof_auc=auc, oof_brier=brier, oof_mean_p=mean_p, gated_ll=ll_gated,
                     eps_sweep=eps_sweep, gate_table=gate_table, hard_rate=hard_rate, soft_rate=soft_rate,
                     ablation=dict(fold0_ll_with_ad=ll_full, fold0_ll_without_ad=ll_base, use_ad=bool(use_ad)),
                     blend_sweep={str(a): b for a, b in zip(ws.tolist(), lls)},
                     wall_s=float(time.time() - t_start), n=int(len(df))),
        seeds=SEEDS,
        spline_files=spline_files,
        template=dict(raw="template_v4_raw.npy", normalized="template_v4.npy", mask="template_mask_v4.npy",
                      grid=GRID, spacing_mm=SP, origin_mm=ORIGIN, label_free=True,
                      method="identical-means (z-score within brain) average, 1 rigid refinement round"),
    )
    with open(os.path.join(args.out, "feature_spec_v4.json"), "w") as f:
        json.dump(spec, f, indent=2)
    pd.DataFrame({"id": df["id"], "y": y_all, "p_oof": p_oof, "p_gated": p_gated, "hard": hard, "soft": soft,
                  "hard_reasons": ["|".join(h) for h in hard_l], "soft_reasons": ["|".join(s) for s in soft_l]}
                 ).to_csv(os.path.join(args.out, "oof_v4.csv"), index=False)
    with open(os.path.join(args.out, "meta_v4.json"), "w") as f:
        json.dump(metas, f, default=str)

    print("[done] OOF LL=%.4f AUC=%.4f gated LL=%.4f  assets -> %s  (%.0fs)" % (
        ll_raw, auc, ll_gated, args.out, time.time() - t_start))
    if ll_raw > 0.32:
        print("USE_V3_ASSETS  (v4 OOF LL %.4f > 0.32; orchestrator ships v3 models + fixed v4 gates)" % ll_raw)
    # ------------------------------------------------------------------ expected outcome after fix
    # Expected OOF LL: ~0.29-0.31 (v3 reference 0.3000, AUC ~0.945-0.95). The v4.0 regression was pure
    # calibration (AUC unchanged, LL doubled): train-fold Platt + logit-space blend. With inner-OOF Platt
    # and probability-space blend the LL must return to <= 0.32; otherwise USE_V3_ASSETS is printed above.
    # Gate expectations: occupancy ~99.4% pass (>=98% asserted via soft<5%/hard<1%), hard ~<=8/1362.
    # Gated LL must be <= raw LL + 0.005 (soft shrink only ever moves ~<5% of scans halfway to prior).


if __name__ == "__main__":
    main()
