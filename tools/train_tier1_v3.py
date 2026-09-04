#!/usr/bin/env python3
"""
train_tier1_v3.py — Parkinson's DaT-SPECT tier-1 model, v3.

Run:  python tools/train_tier1_v3.py

Pipeline (all on the COMMON registered grid in data/reg/, produced upstream):
  pass 0 : mean images of ~100 neg / ~100 pos registered scans -> data-driven template
           (brain mask, striatal blobs, caudate/putamen/posterior-putamen split, 11 mL large VOIs,
            generous striatal boxes, primary whole-brain reference, secondary occipital reference)
           + alignment verification (group-difference peak location).
  pass 1 : per-scan features (SBRs w/ two references, side-sorted worse/better, ratios, asymmetry,
           hot-spot SBR, blob shape at 50/70 %, acquisition covariates, dedup fingerprint).
  groups : duplicate detection = identical native header signature AND
           (feature-vector Pearson r > 0.97 OR residual image-fingerprint r > 0.98) -> union-find.
  CV     : StratifiedGroupKFold(5); monotone LightGBM + spline logistic regression;
           nested inner OOF per outer fold for Platt calibration; blend weight & eps swept on OOF.
  save   : submission_src/assets/{lgbm_model.txt, logistic_coeffs.npz, feature_spec.json, rois_v3.npz}
"""
from __future__ import annotations

import os
import sys
import json
import time
import math
import hashlib
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi
from scipy.interpolate import BSpline
from skimage.filters import threshold_otsu
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True, linewidth=180)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)

# ----------------------------------------------------------------------------- paths / constants
PARK = "/Users/agenta/.openclaw-autoclaw/workspace/parkinsons"
REG_DIR = os.path.join(PARK, "data", "reg")
REF_PATH = os.path.join(REG_DIR, "_reference.nii.gz")
NATIVE_DIR = os.path.join(PARK, "data", "train")
LABELS = os.path.join(PARK, "data", "downloads", "train_labels.csv")
ASSETS = os.path.join(PARK, "submission_src", "assets")
DIAG_DIR = os.path.join(PARK, "data", "diag_v3")

VERSION = "tier1_v3"
SEED = 20250916
N_JOBS = 4
N_FOLDS = 5
N_GROUPDIFF = 100
EPS_GRID = [0.005, 0.01, 0.02, 0.03, 0.05]
BLEND_GRID = [round(0.1 * i, 1) for i in range(11)]
C_GRID = [0.03, 0.1, 0.3, 1.0]

# feature definitions --------------------------------------------------------
SBR_FEATS = [
    "put_sbr_worse", "put_sbr_better",
    "cau_sbr_worse", "cau_sbr_better",
    "putfull_sbr_worse", "putfull_sbr_better",
    "str_sbr_worse", "str_sbr_better",
    "hot_sbr_worse", "hot_sbr_better",
    "put_sbr_occ_worse", "cau_sbr_occ_worse", "str_sbr_occ_worse",
]
NEG_MONO = SBR_FEATS + ["put_cau_ratio_worse", "put_cau_ratio_better"]
POS_MONO = ["cau_minus_put_worse", "asym_put", "asym_cau", "asym_str", "asym_put_u"]
SHAPE_FEATS = [
    "vol50_min", "vol50_max", "ap50_min", "ap50_max", "eig50_min", "eig50_max",
    "vol70_min", "vol70_max", "ap70_min", "ap70_max", "eig70_min", "eig70_max",
]
COV_FEATS = ["ref_ratio", "brain_vol_ml", "log_total_counts",
             "vox_x", "vox_y", "vox_z", "mat_x", "mat_y", "mat_z"]
FEATURES = NEG_MONO + POS_MONO + SHAPE_FEATS + COV_FEATS
MONO = [(-1 if f in NEG_MONO else (1 if f in POS_MONO else 0)) for f in FEATURES]
SPLINE_FEATS = ["put_sbr_worse", "put_sbr_better", "cau_sbr_worse", "cau_sbr_better",
                "str_sbr_worse", "hot_sbr_worse"]
SPL_IDX = [FEATURES.index(f) for f in SPLINE_FEATS]
LIN_IDX = [i for i, f in enumerate(FEATURES) if f not in SPLINE_FEATS]

LGB_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.03,
    num_leaves=6, max_depth=3, min_child_samples=30,
    reg_lambda=10.0, reg_alpha=0.0, min_split_gain=0.0,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    monotone_constraints_method="intermediate",
    n_jobs=N_JOBS, verbose=-1,
)

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def reg_path(uid):
    return os.path.join(REG_DIR, f"{uid}.nii.gz")


# ----------------------------------------------------------------------------- basic helpers
def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=np.float64), -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def ll(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc_safe(y, x):
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    if m.sum() < 10 or len(np.unique(y[m])) < 2 or np.nanstd(x[m]) == 0:
        return np.nan
    return float(roc_auc_score(y[m], x[m]))


def load_vol(path):
    img = nib.load(path)
    v = np.asarray(img.dataobj, dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v[v < 0] = 0.0
    return v, img


def brain_mask(vol, sigma=1.0):
    """Otsu (air vs tissue) on smoothed volume, largest component, holes filled."""
    sm = ndi.gaussian_filter(vol, sigma)
    vals = sm[sm > 0]
    if vals.size < 1000:
        return np.zeros(vol.shape, bool)
    cap = np.percentile(vals, 99.0)
    v2 = vals[vals < cap]
    try:
        thr = float(threshold_otsu(v2))
    except Exception:
        thr = 0.25 * cap
    m = sm > thr
    m = ndi.binary_opening(m, iterations=1)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(m)


def keep_major_components(mask, frac=0.2):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.asarray(ndi.sum(mask, lab, index=range(1, n + 1)))
    keep = np.where(sizes >= frac * sizes.max())[0] + 1
    return np.isin(lab, keep)


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = np.asarray(ndi.sum(mask, lab, index=range(1, n + 1)))
    return lab == (1 + int(np.argmax(sizes)))


def world_coords(shape, affine):
    ii, jj, kk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    ijk = np.stack([ii, jj, kk], 0).reshape(3, -1).astype(np.float64)
    xyz = affine[:3, :3] @ ijk + affine[:3, 3:4]
    return [xyz[a].reshape(shape).astype(np.float32) for a in range(3)]


def block_mean(vol, f):
    if f <= 1:
        return vol.astype(np.float32)
    a, b, c = [(s // f) * f for s in vol.shape]
    v = vol[:a, :b, :c]
    return v.reshape(a // f, f, b // f, f, c // f, f).mean(axis=(1, 3, 5)).astype(np.float32)


def sort2(a, b):
    arr = np.array([a, b], dtype=np.float64)
    if np.all(~np.isfinite(arr)):
        return np.nan, np.nan
    return float(np.nanmin(arr)), float(np.nanmax(arr))


# ----------------------------------------------------------------------------- template
def build_template(ref_img, mean_neg, mean_pos, n_neg, n_pos):
    shape = tuple(mean_neg.shape)
    aff = np.asarray(ref_img.affine, dtype=np.float64)
    spacing = np.sqrt((aff[:3, :3] ** 2).sum(0))  # mm per voxel along i,j,k
    voxvol = float(np.prod(spacing))
    ap_axis = int(np.argmax(np.abs(aff[1, :3])))
    WX, WY, WZ = world_coords(shape, aff)

    mean_all = (mean_neg * n_neg + mean_pos * n_pos) / max(1, n_neg + n_pos)
    brain = brain_mask(mean_all, 1.0)
    if brain.sum() < 1000:
        raise RuntimeError("template brain mask empty")
    brain_ml = brain.sum() * voxvol / 1000.0

    # striatum from mean healthy image
    smn = ndi.gaussian_filter(mean_neg, 0.5)
    bvals = smn[brain]
    bg = float(np.median(bvals))
    peak = float(np.percentile(bvals, 99.9))
    thr50 = bg + 0.5 * (peak - bg)
    str_both = brain & (smn > thr50)
    mid_x = float(WX[brain].mean())
    halves = [WX < mid_x, WX >= mid_x]
    side_masks = [keep_major_components(str_both & h) for h in halves]
    for s in (0, 1):
        if side_masks[s].sum() < 5:
            raise RuntimeError(f"striatal blob on side {s} empty (n={side_masks[s].sum()})")
    str_both = side_masks[0] | side_masks[1]
    z_str = float(WZ[str_both].mean())

    struct = ndi.generate_binary_structure(3, 1)
    dil_str = ndi.binary_dilation(str_both, structure=struct, iterations=4)
    ref_wb = ndi.binary_erosion(brain, structure=struct, iterations=2) & ~dil_str

    ymin = float(WY[brain].min())
    yext = float(WY[brain].max() - ymin)
    br1 = ndi.binary_erosion(brain, structure=struct, iterations=1) & ~dil_str
    occ_frac, occ_z = 0.22, 20.0
    occ = br1 & (WY <= ymin + occ_frac * yext) & (np.abs(WZ - z_str) <= occ_z)
    if occ.sum() * voxvol < 5000:
        occ_frac, occ_z = 0.30, 30.0
        occ = br1 & (WY <= ymin + occ_frac * yext) & (np.abs(WZ - z_str) <= occ_z)

    n_lv = int(round(11000.0 / voxvol))
    n_hot = max(4, int(round(1500.0 / voxvol)))

    sides = []
    save_masks = {}
    for s in (0, 1):
        m = side_masks[s]
        P = np.stack([WX[m], WY[m], WZ[m]], 1).astype(np.float64)
        c = P.mean(0)
        Pc = P - c
        evals, evecs = np.linalg.eigh(Pc.T @ Pc / len(P))
        e1 = evecs[:, -1]
        if e1[1] < 0:
            e1 = -e1  # point anterior (+y)
        t = Pc @ e1
        cau = t >= np.quantile(t, 0.60)
        put = t <= np.quantile(t, 0.55)
        pput = put.copy()
        tp = t[put]
        pput[put] = tp <= np.quantile(tp, 0.40)
        idx_side = np.flatnonzero(m.ravel())
        idx_cau, idx_put, idx_pput = idx_side[cau], idx_side[put], idx_side[pput]

        ijk = np.argwhere(m)
        lo = np.maximum(ijk.min(0) - 3, 0)
        hi = np.minimum(ijk.max(0) + 4, np.array(shape))
        slices = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
        box = np.zeros(shape, bool)
        box[slices] = True
        box &= brain & halves[s]
        box_sub = box[slices]
        bidx = np.flatnonzero(box.ravel())
        vals = smn.ravel()[bidx]
        order = np.argsort(vals)[::-1][:min(n_lv, len(bidx))]
        idx_lv = np.sort(bidx[order])

        sides.append(dict(
            idx_cau=idx_cau, idx_put=idx_put, idx_pput=idx_pput, idx_lv=idx_lv,
            slices=slices, box_sub=box_sub,
            wx=WX[slices].copy(), wy=WY[slices].copy(), wz=WZ[slices].copy(),
            n_vox=int(m.sum()), centroid=c.tolist(), axis=e1.tolist(),
        ))
        for nm, idx in (("cau", idx_cau), ("put", idx_put), ("pput", idx_pput), ("lv", idx_lv)):
            mm = np.zeros(shape, bool)
            mm.ravel()[idx] = True
            save_masks[f"side{s}_{nm}"] = mm
        save_masks[f"side{s}_box"] = box
        save_masks[f"side{s}_str"] = m

    save_masks.update(brain=brain, ref_wb=ref_wb, ref_occ=occ, str_both=str_both, dil_str=dil_str)
    fp_factor = max(1, min(shape) // 12)

    T = dict(
        shape=shape, affine=aff, spacing=spacing, voxvol=voxvol, spacing_ap=float(spacing[ap_axis]),
        idx_ref_wb=np.flatnonzero(ref_wb.ravel()), idx_ref_occ=np.flatnonzero(occ.ravel()),
        sides=sides, n_hot=n_hot, n_lv=n_lv, fp_factor=fp_factor,
        meta=dict(
            brain_ml=brain_ml, bg=bg, peak=peak, contrast_peak_over_bg=peak / max(bg, 1e-9),
            thr50=thr50, mid_x=mid_x, z_str=z_str, occ_frac=occ_frac, occ_z=occ_z,
            ref_wb_ml=float(ref_wb.sum() * voxvol / 1000), ref_occ_ml=float(occ.sum() * voxvol / 1000),
            str_ml=[float(side_masks[s].sum() * voxvol / 1000) for s in (0, 1)],
            cau_ml=[float(len(sides[s]["idx_cau"]) * voxvol / 1000) for s in (0, 1)],
            put_ml=[float(len(sides[s]["idx_put"]) * voxvol / 1000) for s in (0, 1)],
            pput_ml=[float(len(sides[s]["idx_pput"]) * voxvol / 1000) for s in (0, 1)],
            lv_ml=[float(len(sides[s]["idx_lv"]) * voxvol / 1000) for s in (0, 1)],
            box_ml=[float(sides[s]["box_sub"].sum() * voxvol / 1000) for s in (0, 1)],
            n_hot=n_hot, n_lv=n_lv, fp_factor=fp_factor,
        ),
    )
    return T, save_masks


# ----------------------------------------------------------------------------- feature extraction
def blob_shape(sub, S, thr, T):
    m = S["box_sub"] & (sub > thr)
    if int(m.sum()) < 3:
        return np.nan, np.nan, np.nan
    m = largest_component(m)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, np.nan
    X, Y, Z = S["wx"][m], S["wy"][m], S["wz"][m]
    vol_ml = n * T["voxvol"] / 1000.0
    ap = float(Y.max() - Y.min()) + T["spacing_ap"]
    P = np.stack([X, Y, Z], 1).astype(np.float64)
    C = np.cov(P, rowvar=False) + np.diag(np.square(T["spacing"]) / 12.0)
    ev = np.linalg.eigvalsh(C)
    eig = float(np.sqrt(ev[-1] / max(ev[0], 1e-9)))
    return float(vol_ml), ap, eig


def extract_one(uid, T, native_path):
    out = {"uid": uid}
    p = reg_path(uid)
    if not os.path.exists(p):
        out["error"] = "missing"
        return out
    try:
        vol, img = load_vol(p)
    except Exception as e:  # noqa
        out["error"] = f"load:{e}"
        return out
    if tuple(vol.shape) != tuple(T["shape"]):
        out["error"] = f"shape{vol.shape}"
        return out
    out["affine_ok"] = bool(np.allclose(img.affine, T["affine"], atol=1e-2))
    flat = vol.ravel()
    ref_wb = float(flat[T["idx_ref_wb"]].mean())
    if not np.isfinite(ref_wb) or ref_wb <= 1e-9:
        out["error"] = "ref_wb<=0"
        return out
    ref_occ = float(flat[T["idx_ref_occ"]].mean()) if T["idx_ref_occ"].size else np.nan
    if not (np.isfinite(ref_occ) and ref_occ > 1e-9):
        ref_occ = np.nan

    def sbr(idx, ref):
        return float(flat[idx].mean() / ref - 1.0) if np.isfinite(ref) else np.nan

    n_hot = T["n_hot"]
    sides = []
    for s in (0, 1):
        S = T["sides"][s]
        sub = vol[S["slices"]]
        bvals = sub[S["box_sub"]]
        if bvals.size > n_hot:
            hot = float(np.partition(bvals, -n_hot)[-n_hot:].mean())
        else:
            hot = float(bvals.mean()) if bvals.size else np.nan
        d = dict(
            cau=sbr(S["idx_cau"], ref_wb), put=sbr(S["idx_put"], ref_wb), pput=sbr(S["idx_pput"], ref_wb),
            lv=sbr(S["idx_lv"], ref_wb), hot=hot / ref_wb - 1.0,
            cau_occ=sbr(S["idx_cau"], ref_occ), pput_occ=sbr(S["idx_pput"], ref_occ), lv_occ=sbr(S["idx_lv"], ref_occ),
        )
        for frac, tag in ((0.5, "50"), (0.7, "70")):
            thr = ref_wb + frac * (hot - ref_wb)
            v, ap, eig = blob_shape(sub, S, thr, T)
            d[f"vol{tag}"], d[f"ap{tag}"], d[f"eig{tag}"] = v, ap, eig
        sides.append(d)

    a, b = sides
    w = 0 if (np.nan_to_num(a["pput"], nan=1e9) <= np.nan_to_num(b["pput"], nan=1e9)) else 1
    W, B = sides[w], sides[1 - w]
    f = out
    f["put_sbr_worse"], f["put_sbr_better"] = W["pput"], B["pput"]
    f["cau_sbr_worse"], f["cau_sbr_better"] = sort2(a["cau"], b["cau"])
    f["putfull_sbr_worse"], f["putfull_sbr_better"] = sort2(a["put"], b["put"])
    f["str_sbr_worse"], f["str_sbr_better"] = sort2(a["lv"], b["lv"])
    f["hot_sbr_worse"], f["hot_sbr_better"] = sort2(a["hot"], b["hot"])
    f["put_sbr_occ_worse"] = min(a["pput_occ"], b["pput_occ"]) if np.isfinite(ref_occ) else np.nan
    f["cau_sbr_occ_worse"] = min(a["cau_occ"], b["cau_occ"]) if np.isfinite(ref_occ) else np.nan
    f["str_sbr_occ_worse"] = min(a["lv_occ"], b["lv_occ"]) if np.isfinite(ref_occ) else np.nan
    # uptake ratios (1+SBR) are strictly positive -> stable ratios
    f["put_cau_ratio_worse"] = (1.0 + W["pput"]) / max(1.0 + W["cau"], 1e-3)
    f["put_cau_ratio_better"] = (1.0 + B["pput"]) / max(1.0 + B["cau"], 1e-3)
    f["cau_minus_put_worse"] = W["cau"] - W["pput"]

    def asym(x0, x1, floor=0.05):
        x0, x1 = max(x0, floor), max(x1, floor)
        return abs(x0 - x1) / (x0 + x1)

    f["asym_put"] = asym(a["pput"], b["pput"])
    f["asym_cau"] = asym(a["cau"], b["cau"])
    f["asym_str"] = asym(a["lv"], b["lv"])
    u0, u1 = 1.0 + a["pput"], 1.0 + b["pput"]
    f["asym_put_u"] = abs(u0 - u1) / max(u0 + u1, 1e-6)
    for tag in ("50", "70"):
        for q in ("vol", "ap", "eig"):
            f[f"{q}{tag}_min"], f[f"{q}{tag}_max"] = sort2(a[f"{q}{tag}"], b[f"{q}{tag}"])
    f["ref_ratio"] = ref_occ / ref_wb if np.isfinite(ref_occ) else np.nan
    bm = brain_mask(vol, 1.0)
    f["brain_vol_ml"] = float(bm.sum() * T["voxvol"] / 1000.0)
    f["log_total_counts"] = float(np.log1p(float(vol.sum(dtype=np.float64))))

    vx = vy = vz = np.nan
    mx = my = mz = np.nan
    sig = "NOSIG"
    if native_path:
        try:
            nimg = nib.load(native_path)
            zo = nimg.header.get_zooms()[:3]
            sh = nimg.shape[:3]
            vx, vy, vz = (float(z) for z in zo)
            mx, my, mz = (int(s) for s in sh)
            key = (str(np.round(np.asarray(nimg.affine), 2).tolist()) + str(tuple(sh)) +
                   str(nimg.header.get_data_dtype()) + str(tuple(np.round(zo, 3))))
            sig = hashlib.md5(key.encode()).hexdigest()[:12]
        except Exception:  # noqa
            pass
    f["vox_x"], f["vox_y"], f["vox_z"] = vx, vy, vz
    f["mat_x"], f["mat_y"], f["mat_z"] = mx, my, mz
    f["_sig"] = sig
    f["_fp"] = block_mean(vol / ref_wb, T["fp_factor"]).ravel().astype(np.float32)
    return f


# ----------------------------------------------------------------------------- duplicate groups
def find_groups(sigs, F, FP):
    n = len(sigs)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    med = np.nanmedian(F, axis=0)
    med[~np.isfinite(med)] = 0.0
    Fi = np.where(np.isfinite(F), F, med)
    sd = Fi.std(0)
    sd[sd < 1e-12] = 1.0
    Fz = (Fi - Fi.mean(0)) / sd
    Fz = Fz - Fz.mean(1, keepdims=True)
    nz = np.linalg.norm(Fz, axis=1, keepdims=True)
    nz[nz < 1e-12] = 1.0
    Fz = Fz / nz
    FPr = FP - FP.mean(0, keepdims=True)
    FPr = FPr - FPr.mean(1, keepdims=True)
    npf = np.linalg.norm(FPr, axis=1, keepdims=True)
    npf[npf < 1e-12] = 1.0
    FPr = FPr / npf

    buckets = defaultdict(list)
    for i, s in enumerate(sigs):
        buckets[s].append(i)
    n_feat = n_fp = 0
    for s, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        idxs = np.asarray(idxs)
        Cf = Fz[idxs] @ Fz[idxs].T
        Cp = FPr[idxs] @ FPr[idxs].T
        iu = np.triu_indices(len(idxs), 1)
        lf = Cf[iu] > 0.97
        lp = Cp[iu] > 0.98
        n_feat += int(lf.sum())
        n_fp += int(lp.sum())
        link = lf | lp
        for a_, b_ in zip(iu[0][link], iu[1][link]):
            union(int(idxs[a_]), int(idxs[b_]))
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    info = dict(n_signatures=len(buckets), largest_signature_bucket=max(len(v) for v in buckets.values()),
                pairs_feature_corr=n_feat, pairs_fingerprint_corr=n_fp)
    return groups, info


# ----------------------------------------------------------------------------- spline logistic regression
class SplineLR:
    def __init__(self, spline_idx, linear_idx, degree=3, n_knots=5):
        self.spline_idx = list(spline_idx)
        self.linear_idx = list(linear_idx)
        self.degree = degree
        self.n_knots = n_knots

    def _impute(self, X):
        return np.where(np.isfinite(X), X, self.median_)

    def _raw_design(self, Xi):
        cols = []
        for j, t in zip(self.spline_idx, self.knots_):
            x = Xi[:, j]
            if t is None:
                cols.append(x[:, None])
                continue
            lo, hi = t[self.degree], t[-self.degree - 1]
            xc = np.clip(x, lo, hi - 1e-9 * max(hi - lo, 1e-9) - 1e-12)
            Bm = BSpline.design_matrix(xc, t, self.degree).toarray()
            cols.append(Bm[:, :-1])  # drop last basis (sklearn include_bias=False analogue)
        cols.append(Xi[:, self.linear_idx])
        return np.hstack(cols)

    def prepare(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.median_ = np.nanmedian(X, axis=0)
        self.median_[~np.isfinite(self.median_)] = 0.0
        Xi = self._impute(X)
        self.knots_ = []
        for j in self.spline_idx:
            q = np.unique(np.quantile(Xi[:, j], np.linspace(0, 1, self.n_knots)))
            if len(q) < 2:
                self.knots_.append(None)
            else:
                self.knots_.append(np.r_[[q[0]] * self.degree, q, [q[-1]] * self.degree].astype(np.float64))
        D = self._raw_design(Xi)
        self.mu_ = D.mean(0)
        self.sd_ = D.std(0)
        self.sd_[self.sd_ < 1e-12] = 1.0
        return self

    def design(self, X):
        Xi = self._impute(np.asarray(X, dtype=np.float64))
        return (self._raw_design(Xi) - self.mu_) / self.sd_

    def fit(self, X, y, C):
        self.C = C
        self.lr_ = LogisticRegression(C=C, max_iter=5000).fit(self.design(X), y)
        return self

    def logit(self, X):
        return self.design(X) @ self.lr_.coef_[0] + self.lr_.intercept_[0]

    def export(self):
        maxlen = max([len(t) if t is not None else 0 for t in self.knots_] + [1])
        knots = np.full((len(self.spline_idx), maxlen), np.nan)
        klen = np.zeros(len(self.spline_idx), dtype=np.int64)
        for i, t in enumerate(self.knots_):
            if t is not None:
                knots[i, :len(t)] = t
                klen[i] = len(t)
        return dict(
            median=self.median_, spline_idx=np.array(self.spline_idx, dtype=np.int64),
            linear_idx=np.array(self.linear_idx, dtype=np.int64), degree=np.int64(self.degree),
            knots=knots, knots_len=klen, design_mu=self.mu_, design_sd=self.sd_,
            coef=self.lr_.coef_[0], intercept=np.float64(self.lr_.intercept_[0]), C=np.float64(self.C),
            feature_order=np.array(FEATURES),
        )


# ----------------------------------------------------------------------------- models / calibration
def fit_lgb(Xtr, ytr, seed):
    m = lgb.LGBMClassifier(**LGB_PARAMS, monotone_constraints=MONO, random_state=seed)
    m.fit(Xtr, ytr)
    return m


def lgb_logit(m, X):
    return np.asarray(m.booster_.predict(X, raw_score=True), dtype=np.float64)


def platt_fit(z, y):
    lr = LogisticRegression(C=1e6, max_iter=10000).fit(np.asarray(z).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(z, a, b):
    return sigmoid(a * np.asarray(z, dtype=np.float64) + b)


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(DIAG_DIR, exist_ok=True)
    log(f"=== {VERSION} ===  PARK={PARK}")

    labels = pd.read_csv(LABELS)
    labels["uid"] = labels["uid"].astype(str)
    labels["is_pathologic"] = labels["is_pathologic"].astype(int)
    uids = labels["uid"].tolist()
    lab_map = dict(zip(labels["uid"], labels["is_pathologic"]))
    prevalence = float(labels["is_pathologic"].mean())
    avail = np.array([os.path.exists(reg_path(u)) for u in uids])
    log(f"labels={len(uids)} prevalence={prevalence:.4f} registered_available={int(avail.sum())} "
        f"missing_registration={int((~avail).sum())}")
    if avail.sum() < 50:
        print("FATAL: fewer than 50 registered scans available")
        sys.exit(1)

    # native file index (headers only used)
    native_map = {}
    if os.path.isdir(NATIVE_DIR):
        stems = {}
        for root, _, files in os.walk(NATIVE_DIR):
            for fn in files:
                if fn.endswith(".nii.gz"):
                    stems[fn[:-7]] = os.path.join(root, fn)
                elif fn.endswith(".nii"):
                    stems[fn[:-4]] = os.path.join(root, fn)
        for u in uids:
            if u in stems:
                native_map[u] = stems[u]
        if len(native_map) < len(uids):
            for u in uids:
                if u in native_map:
                    continue
                for st, pth in stems.items():
                    if u in st:
                        native_map[u] = pth
                        break
    log(f"native files matched: {len(native_map)}/{len(uids)} in {NATIVE_DIR}")

    if not os.path.exists(REF_PATH):
        print(f"FATAL: reference not found {REF_PATH}")
        sys.exit(1)
    ref_vol, ref_img = load_vol(REF_PATH)
    ref_shape = tuple(ref_vol.shape)
    ref_brain = brain_mask(ref_vol, 1.0)
    log(f"reference grid shape={ref_shape} zooms={tuple(np.round(ref_img.header.get_zooms()[:3], 3))} "
        f"ref_brain_vox={int(ref_brain.sum())}")

    # ------------------------------------------------------------------ pass 0: group means
    rng = np.random.default_rng(SEED)
    neg_av = [u for u, a in zip(uids, avail) if a and lab_map[u] == 0]
    pos_av = [u for u, a in zip(uids, avail) if a and lab_map[u] == 1]
    rng.shuffle(neg_av)
    rng.shuffle(pos_av)
    sums = {0: np.zeros(ref_shape, np.float64), 1: np.zeros(ref_shape, np.float64)}
    counts = {0: 0, 1: 0}
    for cls, sel in ((0, neg_av[:N_GROUPDIFF]), (1, pos_av[:N_GROUPDIFF])):
        for u in sel:
            try:
                v, _ = load_vol(reg_path(u))
            except Exception:
                continue
            if tuple(v.shape) != ref_shape:
                continue
            s = float(v[ref_brain].mean()) if ref_brain.any() else float(v.mean())
            if not np.isfinite(s) or s <= 0:
                continue
            sums[cls] += v / s
            counts[cls] += 1
    n_neg, n_pos = counts[0], counts[1]
    log(f"pass0: mean images from n_neg={n_neg} n_pos={n_pos}")
    if n_neg < 10 or n_pos < 10:
        print("FATAL: not enough scans for template/mean images")
        sys.exit(1)
    mean_neg = (sums[0] / n_neg).astype(np.float32)
    mean_pos = (sums[1] / n_pos).astype(np.float32)

    T, masks = build_template(ref_img, mean_neg, mean_pos, n_neg, n_pos)
    M = T["meta"]
    log("TEMPLATE: brain_ml=%.0f bg=%.4f peak=%.4f contrast=%.2f mid_x=%.1f z_str=%.1f" %
        (M["brain_ml"], M["bg"], M["peak"], M["contrast_peak_over_bg"], M["mid_x"], M["z_str"]))
    log("TEMPLATE: str_ml=%s cau_ml=%s put_ml=%s pput_ml=%s lv_ml=%s box_ml=%s" %
        tuple(np.round(M[k], 2).tolist() for k in ("str_ml", "cau_ml", "put_ml", "pput_ml", "lv_ml", "box_ml")))
    log("TEMPLATE: ref_wb_ml=%.0f ref_occ_ml=%.0f (occ_frac=%.2f occ_z=%.0f) n_hot=%d n_lv=%d fp_factor=%d" %
        (M["ref_wb_ml"], M["ref_occ_ml"], M["occ_frac"], M["occ_z"], M["n_hot"], M["n_lv"], M["fp_factor"]))
    if not (600 < M["brain_ml"] < 2500):
        log("WARNING: template brain volume outside 600-2500 mL — check brain mask")

    # alignment check A: group-difference peak
    diff = ndi.gaussian_filter(mean_neg - mean_pos, 1.0)
    pk = np.unravel_index(int(np.argmax(diff)), diff.shape)
    dt = ndi.distance_transform_edt(~masks["str_both"], sampling=T["spacing"])
    pk_world = (T["affine"][:3, :3] @ np.array(pk, dtype=float) + T["affine"][:3, 3]).round(1).tolist()
    peak_in_brain = bool(masks["brain"][pk])
    peak_in_dil_str = bool(masks["dil_str"][pk])
    peak_dist_mm = float(dt[pk])
    log(f"ALIGN_CHECK group-diff peak ijk={tuple(int(i) for i in pk)} world={pk_world} "
        f"in_brain={peak_in_brain} in_dilated_striatum={peak_in_dil_str} dist_to_striatum_mm={peak_dist_mm:.1f} "
        f"peak_diff={float(diff[pk]):.4f}")
    for nm, arr in (("mean_neg", mean_neg), ("mean_pos", mean_pos), ("diff_neg_minus_pos", diff)):
        nib.save(nib.Nifti1Image(arr.astype(np.float32), T["affine"]), os.path.join(DIAG_DIR, f"{nm}.nii.gz"))
    roi_vis = np.zeros(ref_shape, np.uint8)
    roi_vis[masks["ref_wb"]] = 1
    roi_vis[masks["ref_occ"]] = 2
    for s in (0, 1):
        roi_vis[masks[f"side{s}_lv"]] = 3 + s
        roi_vis[masks[f"side{s}_cau"]] = 5 + s
        roi_vis[masks[f"side{s}_put"]] = 7 + s
        roi_vis[masks[f"side{s}_pput"]] = 9 + s
    nib.save(nib.Nifti1Image(roi_vis, T["affine"]), os.path.join(DIAG_DIR, "rois_vis.nii.gz"))

    # ------------------------------------------------------------------ pass 1: features
    todo = [u for u, a in zip(uids, avail) if a]
    results = []
    with Parallel(n_jobs=N_JOBS, batch_size=8) as par:
        for start in range(0, len(todo), 100):
            chunk = todo[start:start + 100]
            results.extend(par(delayed(extract_one)(u, T, native_map.get(u)) for u in chunk))
            log(f"features: {min(start + 100, len(todo))}/{len(todo)} scans")
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    err_counts = Counter(r["error"].split(":")[0] for r in errs)
    n_aff_bad = sum(1 for r in ok if not r.get("affine_ok", True))
    log(f"processed={len(ok)} errors={len(errs)} {dict(err_counts)} affine_mismatch={n_aff_bad}")
    if len(ok) < 50:
        print("FATAL: too few feature rows")
        sys.exit(1)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in ok])
    df["y"] = df["uid"].map(lab_map).astype(int)
    sigs = [r["_sig"] for r in ok]
    FP = np.stack([r["_fp"] for r in ok]).astype(np.float64)
    X = df[FEATURES].values.astype(np.float64)
    y = df["y"].values.astype(int)
    n = len(df)

    # ------------------------------------------------------------------ duplicate groups
    groups, ginfo = find_groups(sigs, X, FP)
    df["group"] = groups
    gs = np.bincount(groups)
    n_groups = int(len(gs))
    log(f"GROUPS: n_groups={n_groups} of n={n}; sizes: {dict(Counter(gs.tolist()))}; {ginfo}")
    df["sig"] = sigs
    del FP

    # ------------------------------------------------------------------ diagnostics: univariate
    rows = []
    for f, mono in zip(FEATURES, MONO):
        x = df[f].values.astype(np.float64)
        rows.append(dict(feature=f, mono=mono, auc=auc_safe(y, x), missing_pct=100 * np.mean(~np.isfinite(x)),
                         mean=np.nanmean(x), sd=np.nanstd(x), p05=np.nanpercentile(x, 5), p95=np.nanpercentile(x, 95)))
    ftab = pd.DataFrame(rows)
    sbr_aucs = ftab.set_index("feature").loc[SBR_FEATS, "auc"]
    max_dev = float(np.nanmax(np.abs(sbr_aucs.values - 0.5)))
    alignment_ok = peak_in_brain and (max_dev >= 0.05)
    print("\n================ FEATURE TABLE (AUC of raw value vs is_pathologic; SBR expect < 0.5) ================")
    print(ftab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSBR features: max |AUC-0.5| = {max_dev:.3f}; group-diff peak in brain = {peak_in_brain}")
    if alignment_ok:
        print("ALIGNMENT_OK")
    else:
        print("ALIGNMENT_SUSPECT  <-- SBR AUCs ~0.5 for all features and/or group-diff peak outside brain mask")
    const_ll = ll(y, np.full(n, y.mean()))
    print(f"constant-prevalence baseline log loss on processed set: {const_ll:.4f} (prev={y.mean():.4f})")

    # ------------------------------------------------------------------ CV
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(X, y, groups))
    oof_lgb = np.zeros(n)
    oof_lr = np.zeros(n)
    inner_store = []
    bestCs = []
    fold_rows = []
    for k, (tr, te) in enumerate(folds):
        Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
        inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 11 * (k + 1))
        in_lgb = np.zeros(len(tr))
        in_lr = {C: np.zeros(len(tr)) for C in C_GRID}
        for itr, ite in inner.split(Xtr, ytr, gtr):
            m = fit_lgb(Xtr[itr], ytr[itr], SEED + k)
            in_lgb[ite] = lgb_logit(m, Xtr[ite])
            prep = SplineLR(SPL_IDX, LIN_IDX).prepare(Xtr[itr])
            Dtr, Dte = prep.design(Xtr[itr]), prep.design(Xtr[ite])
            for C in C_GRID:
                lr = LogisticRegression(C=C, max_iter=5000).fit(Dtr, ytr[itr])
                in_lr[C][ite] = Dte @ lr.coef_[0] + lr.intercept_[0]
        C_ll = {C: ll(ytr, sigmoid(in_lr[C])) for C in C_GRID}
        bestC = min(C_ll, key=C_ll.get)
        bestCs.append(bestC)
        m = fit_lgb(Xtr, ytr, SEED + k)
        oof_lgb[te] = lgb_logit(m, X[te])
        lrm = SplineLR(SPL_IDX, LIN_IDX).prepare(Xtr).fit(Xtr, ytr, bestC)
        oof_lr[te] = lrm.logit(X[te])
        inner_store.append((in_lgb, in_lr[bestC], ytr, te))
        fold_rows.append(dict(fold=k, n_test=len(te), prev_test=float(y[te].mean()), bestC=bestC,
                              inner_ll_lgb=ll(ytr, sigmoid(in_lgb)), inner_ll_lr=C_ll[bestC],
                              test_ll_lgb_raw=ll(y[te], sigmoid(oof_lgb[te])), test_ll_lr_raw=ll(y[te], sigmoid(oof_lr[te])),
                              test_auc_lgb=auc_safe(y[te], oof_lgb[te]), test_auc_lr=auc_safe(y[te], oof_lr[te])))
        log(f"fold {k}: n_te={len(te)} bestC={bestC} ll_lgb={fold_rows[-1]['test_ll_lgb_raw']:.4f} "
            f"ll_lr={fold_rows[-1]['test_ll_lr_raw']:.4f}")
    print("\n================ FOLD TABLE ================")
    print(pd.DataFrame(fold_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # blend sweep with nested Platt
    blend_rows = []
    cal_by_w = {}
    for w in BLEND_GRID:
        cal = np.zeros(n)
        platts = []
        for (in_lgb, in_lr_, ytr, te) in inner_store:
            a_, b_ = platt_fit(w * in_lgb + (1 - w) * in_lr_, ytr)
            platts.append((a_, b_))
            cal[te] = platt_apply(w * oof_lgb[te] + (1 - w) * oof_lr[te], a_, b_)
        zb = w * oof_lgb + (1 - w) * oof_lr
        cal_by_w[w] = cal
        blend_rows.append(dict(w_lgbm=w, ll_raw=ll(y, sigmoid(zb)), ll_platt_nested=ll(y, cal),
                               auc=auc_safe(y, zb), platt_a_mean=np.mean([p[0] for p in platts]),
                               platt_b_mean=np.mean([p[1] for p in platts])))
    btab = pd.DataFrame(blend_rows)
    best_w = float(btab.loc[btab["ll_platt_nested"].idxmin(), "w_lgbm"])
    print("\n================ BLEND SWEEP (w = weight on LightGBM logit) ================")
    print(btab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"best blend weight (LightGBM) = {best_w}")

    cal = cal_by_w[best_w]
    zb = best_w * oof_lgb + (1 - best_w) * oof_lr
    eps_rows = []
    for eps in EPS_GRID:
        eps_rows.append(dict(eps=eps, ll_platt_clipped=ll(y, np.clip(cal, eps, 1 - eps)),
                             ll_raw_clipped=ll(y, np.clip(sigmoid(zb), eps, 1 - eps)),
                             n_clipped=int(((cal < eps) | (cal > 1 - eps)).sum())))
    etab = pd.DataFrame(eps_rows)
    best_eps = float(etab.loc[etab["ll_platt_clipped"].idxmin(), "eps"])
    print("\n================ EPS SWEEP (calibrated OOF, blend w=%.1f) ================" % best_w)
    print(etab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    p_final = np.clip(cal, best_eps, 1 - best_eps)
    oof_ll_final = ll(y, p_final)
    oof_ll_raw = ll(y, sigmoid(zb))
    print(f"best eps = {best_eps}  ->  OOF log loss (nested Platt + clip) = {oof_ll_final:.4f}   "
          f"[raw blend {oof_ll_raw:.4f}, constant {const_ll:.4f}]  AUC={auc_safe(y, zb):.4f}")

    # worst 5 %
    loss_i = -(y * np.log(p_final) + (1 - y) * np.log(1 - p_final))
    n_worst = int(math.ceil(0.05 * n))
    order = np.argsort(loss_i)[::-1]
    worst = order[:n_worst]
    wtab = pd.DataFrame(dict(uid=df["uid"].values[worst], y=y[worst], p=p_final[worst], loss=loss_i[worst],
                             put_sbr_worse=X[worst, FEATURES.index("put_sbr_worse")],
                             cau_sbr_worse=X[worst, FEATURES.index("cau_sbr_worse")],
                             str_sbr_worse=X[worst, FEATURES.index("str_sbr_worse")],
                             asym_put=X[worst, FEATURES.index("asym_put")],
                             brain_vol_ml=X[worst, FEATURES.index("brain_vol_ml")],
                             group=groups[worst], sig=df["sig"].values[worst]))
    print(f"\n================ WORST 5% ({n_worst} scans) ================")
    print(wtab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    share = float(loss_i[worst].sum() / loss_i.sum())
    print(f"worst 5% carry {100 * share:.1f}% of total OOF loss; among them label=1: {int(y[worst].sum())}, "
          f"label=0: {int(n_worst - y[worst].sum())}; mean loss worst={loss_i[worst].mean():.3f} rest={loss_i[order[n_worst:]].mean():.3f}")
    print("loss by decile of put_sbr_worse:")
    dec = pd.qcut(pd.Series(X[:, FEATURES.index("put_sbr_worse")]).rank(method="first"), 10, labels=False)
    dtab = pd.DataFrame(dict(decile=dec, loss=loss_i, y=y, p=p_final)).groupby("decile").agg(
        n=("y", "size"), prev=("y", "mean"), mean_p=("p", "mean"), mean_loss=("loss", "mean"))
    print(dtab.to_string(float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------------------ final refit
    final_lgb = fit_lgb(X, y, SEED)
    final_C = Counter(bestCs).most_common(1)[0][0]
    final_lr = SplineLR(SPL_IDX, LIN_IDX).prepare(X).fit(X, y, final_C)
    platt_a, platt_b = platt_fit(zb, y)
    log(f"final: LGBM refit; LR C={final_C}; Platt a={platt_a:.4f} b={platt_b:.4f}")
    imp = pd.DataFrame(dict(feature=FEATURES, gain=final_lgb.booster_.feature_importance("gain"),
                            split=final_lgb.booster_.feature_importance("split"))).sort_values("gain", ascending=False)
    print("\n================ LGBM IMPORTANCE (top 20, gain) ================")
    print(imp.head(20).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    lr_coef = pd.Series(final_lr.lr_.coef_[0])
    print(f"LR: {len(lr_coef)} design columns, |coef| max={lr_coef.abs().max():.3f}, intercept={final_lr.lr_.intercept_[0]:.3f}")

    # ------------------------------------------------------------------ save assets
    lgb_path = os.path.join(ASSETS, "lgbm_model.txt")
    final_lgb.booster_.save_model(lgb_path)
    np.savez(os.path.join(ASSETS, "logistic_coeffs.npz"), **final_lr.export())
    np.savez_compressed(
        os.path.join(ASSETS, "rois_v3.npz"),
        affine=T["affine"], shape=np.array(T["shape"]), spacing=T["spacing"],
        n_hot=np.int64(T["n_hot"]), n_lv=np.int64(T["n_lv"]), fp_factor=np.int64(T["fp_factor"]),
        side0_slices=np.array([[s.start, s.stop] for s in T["sides"][0]["slices"]]),
        side1_slices=np.array([[s.start, s.stop] for s in T["sides"][1]["slices"]]),
        **{k: v.astype(np.uint8) for k, v in masks.items()},
    )
    spec = dict(
        version=VERSION, created=time.strftime("%Y-%m-%d %H:%M:%S"),
        feature_order=FEATURES, monotone_constraints=MONO, spline_features=SPLINE_FEATS,
        n_labels=len(uids), n_registered_available=int(avail.sum()), n_processed=int(n), n_errors=len(errs),
        error_counts=dict(err_counts), affine_mismatch=int(n_aff_bad),
        prevalence_train=prevalence, prevalence_processed=float(y.mean()), tier3_fallback_constant=prevalence,
        groups=dict(n_groups=n_groups, size_hist={str(k): int(v) for k, v in Counter(gs.tolist()).items()}, **ginfo),
        cv=dict(n_folds=N_FOLDS, seed=SEED, fold_table=fold_rows, bestC_per_fold=bestCs, final_C=final_C),
        oof_logloss=oof_ll_final, oof_logloss_raw_blend=oof_ll_raw, oof_logloss_constant=const_ll,
        oof_auc=auc_safe(y, zb), oof_logloss_lgbm_only=float(btab.loc[btab.w_lgbm == 1.0, "ll_platt_nested"].iloc[0]),
        oof_logloss_lr_only=float(btab.loc[btab.w_lgbm == 0.0, "ll_platt_nested"].iloc[0]),
        blend_weight_lgbm=best_w, blend_table=blend_rows, eps=best_eps, eps_table=eps_rows,
        platt=dict(a=platt_a, b=platt_b, note="p = sigmoid(a*z + b), z = w*z_lgbm + (1-w)*z_lr (raw logits)"),
        lgbm_params={k: v for k, v in LGB_PARAMS.items() if k != "n_jobs"},
        grid=dict(shape=list(T["shape"]), affine=T["affine"].tolist(), spacing=T["spacing"].tolist(),
                  voxel_volume_mm3=T["voxvol"], reference=REF_PATH),
        roi=dict(meta=M, sides=[dict(n_vox=S["n_vox"], centroid=S["centroid"], axis=S["axis"]) for S in T["sides"]],
                 pass0_n_neg=n_neg, pass0_n_pos=n_pos),
        alignment=dict(ok=bool(alignment_ok), peak_ijk=[int(i) for i in pk], peak_world=pk_world,
                       peak_in_brain=peak_in_brain, peak_in_dilated_striatum=peak_in_dil_str,
                       peak_dist_to_striatum_mm=peak_dist_mm, sbr_max_abs_auc_dev=max_dev,
                       sbr_aucs={k: (None if not np.isfinite(v) else float(v)) for k, v in sbr_aucs.items()}),
        feature_table=[{k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r.items()} for r in rows],
        assets=["lgbm_model.txt", "logistic_coeffs.npz", "rois_v3.npz", "feature_spec.json"],
        notes=[
            "Inference: register scan to _reference.nii.gz exactly as training (rigid MI, SimpleITK), load rois_v3.npz,",
            "  ref_wb = mean(vol[ref_wb]); ref_occ = mean(vol[ref_occ]); SBR = mean(vol[roi])/ref - 1;",
            "  hot = mean of top n_hot voxels inside side box; shapes at thr = ref_wb + f*(hot-ref_wb), f in {0.5,0.7},",
            "  largest component inside box: volume mL, AP extent (world y range + spacing), sqrt(eig_max/eig_min) of coords cov (+spacing^2/12);",
            "  worse side = lower posterior-putamen SBR; other features side-sorted min/max; put_cau_ratio uses (1+SBR)/(1+SBR);",
            "  asym_* on SBR floored at 0.05; asym_put_u on uptake (1+SBR); brain_vol_ml via Otsu brain mask on the registered scan;",
            "  log_total_counts = log1p(sum(registered vol)); vox_*/mat_* from native header (NaN if unavailable).",
            "LR design: for spline features, cubic clamped B-spline basis with quantile knots (scipy BSpline.design_matrix on x clipped",
            "  to [knots[degree], knots[-degree-1]]), last basis column dropped; if knots_len==0 the feature is linear; other features linear;",
            "  NaN -> median; design standardized by design_mu/design_sd; logit = design @ coef + intercept.",
            "Final probability: p = clip(sigmoid(a*(w*z_lgbm+(1-w)*z_lr)+b), eps, 1-eps). Tier-3 fallback = tier3_fallback_constant.",
            "Never cerebellum reference. Registration NOT redone here. Registered inputs must match grid shape/affine.",
        ],
    )
    with open(os.path.join(ASSETS, "feature_spec.json"), "w") as fh:
        json.dump(spec, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    df_out = df.copy()
    df_out["z_lgb"], df_out["z_lr"], df_out["p_cal"], df_out["loss"] = oof_lgb, oof_lr, p_final, loss_i
    df_out.to_csv(os.path.join(DIAG_DIR, "features_oof_v3.csv"), index=False)
    ftab.to_csv(os.path.join(DIAG_DIR, "feature_table_v3.csv"), index=False)

    print("\n================ SUMMARY ================")
    print(f"processed {n}/{len(uids)} scans ({int((~avail).sum())} registrations not yet available, {len(errs)} errors)")
    print(f"groups={n_groups}  blend_w_lgbm={best_w}  eps={best_eps}  Platt a={platt_a:.3f} b={platt_b:.3f}  final LR C={final_C}")
    print(f"OOF log loss: final={oof_ll_final:.4f}  lgbm-only={spec['oof_logloss_lgbm_only']:.4f}  "
          f"lr-only={spec['oof_logloss_lr_only']:.4f}  raw-blend={oof_ll_raw:.4f}  constant={const_ll:.4f}")
    print("ALIGNMENT_OK" if alignment_ok else "ALIGNMENT_SUSPECT")
    print(f"assets written to {ASSETS}; diagnostics in {DIAG_DIR}")
    log("done")


if __name__ == "__main__":
    main()
