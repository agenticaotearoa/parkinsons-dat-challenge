"""DaT Parkinson's Prediction Challenge — submission scaffold.

Contract (from competition runtime examples/template/main.py):
  - DATA_ROOT=/code_execution/data, NIFTI_DIR=<root>/niftis, submission_format.csv at root
  - Must write ./submission.csv at repo root of execution dir
  - Predictions: probability in [0,1] that each DaT-SPECT scan is pathologic (is_pathologic)

Strategy tiers (auto-selected by asset availability):
  1. MODELS/sbr_model.pkl  -> trained model over SBR-style features (feature_order from feature_spec.json)
  2. No model asset        -> registration-free intensity-based SBR proxy heuristic
  3. Absolute fallback     -> prior 0.5485 (train prevalence; constant valid submission)

Implementation notes (from expert review of an earlier revision):
  - Axis order / orientation: volumes are loaded via nib.as_closest_canonical (RAS: axis0=L-R,
    axis1=A-P, axis2=S-I); the L-R midline is the brain center-of-mass along axis 0, never a
    hard-coded shape[-1] split. 4D exports are squeezed to 3D; NaNs are scrubbed.
  - SBR proxy: features are anchored to a brain-mask reference (nonspecific uptake, not the
    self-referential p99 tail) and a fixed physical hot-spot volume (~2 mL), not a percentile
    tail — so the score reflects true striatal contrast and does not collapse to a constant.
  - Resolution harmonization: volumes are resampled to ~3 mm isotropic when coarse/fine enough
    to matter, and total voxel count is capped, so mixed native geometries (e.g. 256^3) behave.

All imports below are verified present in the official runtime lock (runtime/uv.lock):
nibabel, numpy, pandas, scipy, scikit-learn, joblib, xgboost, lightgbm. skimage is NOT
guaranteed, so Otsu's threshold is implemented with numpy here.
"""

import json
import os
import sys
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import BSpline
from scipy.special import expit

# SimpleITK / lightgbm are confirmed in the official runtime lock (runtime/uv.lock). They are
# wrapped in try/except so a missing dependency degrades tier-1 -> tier-2 instead of crashing.
try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

# DATA_ROOT is overridable via env so a local driver can point at smoke data; the
# in-competition default (no env) is /code_execution/data.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/code_execution/data"))
NIFTI_DIR = DATA_ROOT / "niftis"
SUBMISSION_FORMAT_PATH = DATA_ROOT / "submission_format.csv"
# cwd-relative per the runtime contract (entrypoint cds to /code_execution and expects
# submission.csv there); overridable via env for the local driver.
WRITE_SUBMISSION = Path(os.environ.get("WRITE_SUBMISSION", "submission.csv"))

# Model assets are bundled inside submission.zip (no network at execution time).
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LGBM_MODEL_PATH = ASSETS_DIR / "lgbm_model.txt"
LOGISTIC_PATH = ASSETS_DIR / "logistic_coeffs.npz"
ROIS_PATH = ASSETS_DIR / "rois_v3.npz"
FEATURE_SPEC_PATH = ASSETS_DIR / "feature_spec.json"
REFERENCE_PATH = ASSETS_DIR / "reference.nii.gz"

# Train-set prevalence (from the data; per expert review the fallback must be the
# prevalence, not 0.35). Also used as the heuristic's prior.
PREVALENCE = 0.5485
NEUTRAL_P = PREVALENCE
HEURISTIC_PRIOR = PREVALENCE

# Heuristic constants (literature/EANM-anchored, per review). Deliberately NOT tuned to
# the tiny 20-scan smoke set. mid sits on the worse-side SBR scale; the review flags mid
# and the QC window as "guess until you see data" and says to set mid at the antimode of
# the observed sbr_worse distribution. With a whole-head foreground mask the boundary is
# ~3.0 and brain_ml spans ~2100-4400 mL, so the QC window is set to the whole-head scale
# (the review's [800, 2200] was computed for a brain-only mask and would reject every scan).
MID_SBR = 3.0
SLOPE = 3.0
CLIP_LO = 0.15
CLIP_HI = 0.85
# Whole-head volume QC gate (mL). If a scan's segmentation falls outside this it is treated
# as a computation failure and the prior is returned instead of an ungrounded guess.
BRAIN_ML_MIN = 1500.0
BRAIN_ML_MAX = 6000.0

# Resolution harmonization targets.
TARGET_MM = 3.0          # resample to ~3 mm isotropic when it helps
MAX_VOX = 2_000_000      # trigger resampling for large recons (e.g. 256^3)
HARD_VOX_CAP = 8_000_000 # emergency safety so no volume grows unbounded

# The keys _features() produces. feature_spec.json's feature_order MUST be a subset of these
# for a bundled tier-1 model to be usable; if it is not (a stale model asset), we fall back to
# the heuristic rather than emitting a silent constant submission.
SBR_FEATURES = frozenset({
    "sbr_worse", "sbr_better", "asym", "blob_ap_min_mm", "blob_vol_min_ml", "brain_ml", "ref",
})


def _log(msg: str) -> None:
    """Loud, persistent log line to stderr (teed to submission/log.txt)."""
    print(f"[scaffold] {msg}", file=sys.stderr)


def _otsu(arr: np.ndarray) -> float:
    """Otsu's between-class-variance threshold (skimage-free).

    Operates on a 1-D array of the values that matter (positive tissue); returns the
    intensity threshold that best separates the two modes (air-noise vs uptake).
    """
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0
    hist, edges = np.histogram(a, bins=256)
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total <= 0:
        return float(centers[0])
    w = np.cumsum(hist)
    mu = np.cumsum(hist * centers)
    mt = mu[-1]
    eps = 1e-12
    w0 = w
    w1 = total - w
    m0 = mu / (w0 + eps)
    m1 = (mt - mu) / (w1 + eps)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.nanargmax(between))])


def _downsample(data: np.ndarray, zooms: np.ndarray):
    """Resample to ~3 mm isotropic when the volume is fine-grained or very large.

    Never enlarges an already-coarse scan, and caps the voxel count so mixed native
    geometries (e.g. 256^3) remain tractable. Returns (data, zooms) with updated spacing.
    """
    if not (zooms.min() < 2.0 or data.size > MAX_VOX):
        return data, zooms
    # factor = zooms / target: this would ENLARGE an already-coarse axis (spacing>3mm), so
    # clamp at 1.0 to never grow a coarse scan; fine/large scans still normalize to 3 mm.
    factor = np.minimum(zooms / TARGET_MM, 1.0)
    if data.size > HARD_VOX_CAP:
        factor = factor * ((HARD_VOX_CAP / data.size) ** (1.0 / 3.0))
    factor = np.maximum(factor, 0.05)
    if np.allclose(factor, 1.0):
        return data, zooms
    data = ndimage.zoom(data, factor, order=1).astype(np.float32)
    return data, np.abs(zooms / factor)


def _load_volume(filepath: Path):
    """Load a nifti as a canonical-RAS float volume; return (data, zooms).

    - as_closest_canonical reorders to RAS: axis0=L-R, axis1=A-P, axis2=S-I.
    - NaN/inf are scrubbed (a NaN propagates through every downstream percentile).
    - 4D exports are squeezed to 3D (collapse extra frames by mean).
    - zooms are the (possibly resampled) mm spacing, for mL and mm features.
    """
    img = nib.as_closest_canonical(nib.load(str(filepath)))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    while data.ndim > 3:
        data = data.mean(axis=-1) if data.shape[-1] > 1 else data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected 3D volume, got {data.shape}")
    zooms = np.abs(np.asarray(img.header.get_zooms()[:3], dtype=np.float32))
    return _downsample(data, zooms)


def _brain_mask(vol: np.ndarray, zooms: np.ndarray) -> np.ndarray:
    """Segment the brain as the largest foreground component above an Otsu threshold."""
    sm = ndimage.gaussian_filter(vol, sigma=6.0 / zooms / 2.355)  # ~6 mm FWHM
    t = _otsu(sm[sm > 0])
    m = sm > t
    lab, n = ndimage.label(m)
    if n == 0:
        raise ValueError("empty brain mask")
    m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return ndimage.binary_fill_holes(m)


def _features(vol: np.ndarray, zooms: np.ndarray) -> dict:
    """Registration-free striatum-vs-brain SBR proxy features.

    Numerator: hottest ~2 mL per side (a fixed physical volume, so it is resolution
    invariant); Denominator: nonspecific brain uptake (the brain reference, NOT the
    p99 tail). If reference were self-referential, hot_frac would be pinned at ~0.01 and
    the heuristic would collapse to a constant ~0.55 probability.
    """
    brain = _brain_mask(vol, zooms)
    vox_ml = float(np.prod(zooms)) / 1000.0
    core = ndimage.binary_erosion(brain, iterations=int(round(8.0 / zooms.min())))
    com = ndimage.center_of_mass(brain)

    # Restrict the striatal search to a central box around the brain COM (registration-free
    # localisation) so scalp/parotid/nasal uptake cannot win the top-k on a cool striatum.
    box = np.zeros_like(brain)
    r = (np.array([40, 45, 30]) / zooms).astype(int)  # ± mm in L-R, A-P, S-I
    c = np.round(com).astype(int)
    sl = tuple(
        slice(max(0, c[i] - r[i]), min(brain.shape[i], c[i] + r[i])) for i in range(3)
    )
    box[sl] = True
    search = core & box

    # Reference = nonspecific brain: core voxels excluding the hottest 10% (crude striatum
    # exclusion), so a cold/pathologic striatum still reads as a contrast ratio.
    bv = vol[core]
    if bv.size == 0:
        raise ValueError("empty brain core")
    ref = float(np.mean(bv[bv <= np.percentile(bv, 90)]))

    sm = ndimage.gaussian_filter(vol, sigma=4.0 / zooms / 2.355)  # peak stability
    out = {}
    # The midline is the brain's L-R centre of mass (axis 0 in canonical RAS), not shape//2.
    for side, sel in (
        ("a", slice(None, int(com[0]))),
        ("b", slice(int(com[0]), None)),
    ):
        half = np.zeros_like(search)
        half[sel] = True
        vals = sm[search & half]
        if vals.size == 0:
            out[side] = 0.0
            continue
        k = max(8, int(2.0 / vox_ml))  # hottest ~2 mL per side
        top = np.partition(vals, -k)[-k:] if vals.size > k else vals
        out[side] = (float(top.mean()) - ref) / max(ref, 1e-6)

    sbr_worse, sbr_better = min(out.values()), max(out.values())

    # Blob shape: extent above 50% contrast on the better side (comma vs dot).
    thr = ref + 0.5 * (sbr_better * ref)
    blob = (sm > thr) & search
    lab, n = ndimage.label(blob)
    ap_extents, vols = [], []
    for i in range(1, n + 1):
        idx = np.argwhere(lab == i)
        if idx.shape[0] * vox_ml < 0.5:
            continue
        ap_extents.append((np.ptp(idx[:, 1]) + 1) * zooms[1])  # axis 1 = A-P
        vols.append(idx.shape[0] * vox_ml)

    return {
        "sbr_worse": sbr_worse,
        "sbr_better": sbr_better,
        "asym": abs(out["a"] - out["b"]) / (abs(out["a"]) + abs(out["b"]) + 1e-3),
        "blob_ap_min_mm": float(min(ap_extents)) if ap_extents else 0.0,
        "blob_vol_min_ml": float(min(vols)) if vols else 0.0,
        "brain_ml": float(brain.sum() * vox_ml),
        "ref": ref,
    }


def _heuristic_probability(
    feats: dict, prior: float = None, mid: float = None, slope: float = None
) -> float:
    """Map SBR-proxy features to a pathology probability.

    One dominant feature (worse-side SBR), a soft slope, an asymmetry tie-breaker that only
    fires once clearly abnormal, and a shape tie-breaker. QC failures return the prior rather
    than an ungrounded guess. Clipped to the loss-minimizing band for an unfitted classifier.

    prior/mid/slope default from module globals at CALL time (not import time) so they can be
    overridden for calibration without reloading the module.
    """
    if prior is None:
        prior = HEURISTIC_PRIOR
    if mid is None:
        mid = MID_SBR
    if slope is None:
        slope = SLOPE
    z = -slope * (feats["sbr_worse"] - mid)
    z += 2.0 * max(0.0, feats["asym"] - 0.12)  # only helps once clearly abnormal
    z += 0.04 * max(0.0, 25.0 - feats["blob_ap_min_mm"])  # "dot" instead of "comma"
    if feats["brain_ml"] < BRAIN_ML_MIN or feats["brain_ml"] > BRAIN_ML_MAX or feats["ref"] <= 0:
        return float(prior)
    p = expit(z + np.log(prior / (1.0 - prior)))
    return float(np.clip(p, CLIP_LO, CLIP_HI))


def _nib_to_sitk(img_nib):
    """nibabel -> SimpleITK, preserving TRUE physical geometry (mirror of register_study.py).

    SimpleITK is x-fastest; nibabel is z-fastest, so the array is transposed. Spacing/direction
    come from the RAS affine so the image is physically oriented identically to nibabel's view.
    """
    vol = np.asarray(img_nib.dataobj, dtype=np.float32)
    img = sitk.GetImageFromArray(vol.T)
    aff = img_nib.affine
    spacing = np.linalg.norm(aff[:3, :3], axis=0)
    direction = aff[:3, :3] / spacing
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(o) for o in aff[:3, 3]))
    img.SetDirection(tuple(float(d) for d in direction.T.flatten()))
    return img


def _register_t1(fixed, moving):
    """Rigid Mattes-MI registration exactly as register_study.py (CenteredTransformInitializer GEOMETRY)."""
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4,
                                               numberOfIterations=200, relaxationFactor=0.5)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(init, inPlace=False)
    tx = R.Execute(fixed, moving)
    return sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def _world_coords(shape, affine):
    """World coordinates (mm) of every voxel of a grid, per-axis arrays in nibabel order."""
    ii, jj, kk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    ijk = np.stack([ii, jj, kk], 0).reshape(3, -1).astype(np.float64)
    xyz = affine[:3, :3] @ ijk + affine[:3, 3:4]
    return [xyz[a].reshape(shape).astype(np.float32) for a in range(3)]


def _sort2(a, b):
    arr = np.array([a, b], dtype=np.float64)
    if np.all(~np.isfinite(arr)):
        return np.nan, np.nan
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def _largest_component(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.asarray(ndimage.sum(mask, lab, index=range(1, n + 1)))
    return lab == (1 + int(np.argmax(sizes)))


def _brain_mask_t1(vol, sigma=1.0):
    """Otsu brain mask identical to train_tier1_v3.brain_mask (p99 cap, opening, largest comp)."""
    sm = ndimage.gaussian_filter(vol, sigma)
    vals = sm[sm > 0]
    if vals.size < 1000:
        return np.zeros(vol.shape, bool)
    cap = float(np.percentile(vals, 99.0))
    v2 = vals[vals < cap]
    try:
        thr = _otsu(v2)  # numpy Otsu (skimage.filters.threshold_otsu equivalent)
    except Exception:
        thr = 0.25 * cap
    m = sm > thr
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = np.asarray(ndimage.sum(m, lab, index=range(1, n + 1)))
    m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m)


def _blob_shape(sub, S, thr, T):
    """Shape of the largest component above a threshold inside a side box (mirror training)."""
    m = S["box_sub"] & (sub > thr)
    if int(m.sum()) < 3:
        return np.nan, np.nan, np.nan
    m = _largest_component(m)
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


def _lr_design(X, lr):
    """Spline + linear design matrix, imputed and standardized (mirror train_tier1_v3.SplineLR.design)."""
    X = np.asarray(X, dtype=np.float64)
    median = lr["median"]
    Xi = np.where(np.isfinite(X), X, median)
    degree = int(lr["degree"])
    knots = lr["knots"]
    knots_len = lr["knots_len"]
    cols = []
    for ti, j in enumerate(np.asarray(lr["spline_idx"])):
        t = knots[ti]
        if knots_len[ti] == 0:
            cols.append(Xi[:, j][:, None])
            continue
        lo, hi = t[degree], t[-degree - 1]
        x = Xi[:, j]
        xc = np.clip(x, lo, hi - 1e-9 * max(hi - lo, 1e-9) - 1e-12)
        Bm = BSpline.design_matrix(xc, t, degree).toarray()
        cols.append(Bm[:, :-1])  # drop last basis (sklearn include_bias=False analogue)
    cols.append(Xi[:, np.asarray(lr["linear_idx"], dtype=int)])
    D = np.hstack(cols)
    return (D - lr["design_mu"]) / lr["design_sd"]


def _tier1_features(vol, native, T):
    """Per-scan features from a REGISTERED volume (mirror train_tier1_v3.extract_one).

    vol is the scan resampled to the reference grid in nibabel (x,y,z) order; native is the
    (vx,vy,vz,mx,my,mz) acquisition covariates from the RAW scan header.
    """
    flat = vol.ravel()
    ref_wb = float(flat[T["idx_ref_wb"]].mean())
    if not np.isfinite(ref_wb) or ref_wb <= 1e-9:
        raise ValueError("ref_wb<=0")
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
            cau=sbr(S["idx_cau"], ref_wb), put=sbr(S["idx_put"], ref_wb),
            pput=sbr(S["idx_pput"], ref_wb), lv=sbr(S["idx_lv"], ref_wb),
            hot=hot / ref_wb - 1.0,
            cau_occ=sbr(S["idx_cau"], ref_occ), pput_occ=sbr(S["idx_pput"], ref_occ),
            lv_occ=sbr(S["idx_lv"], ref_occ),
        )
        for frac, tag in ((0.5, "50"), (0.7, "70")):
            thr = ref_wb + frac * (hot - ref_wb)
            v, ap, eig = _blob_shape(sub, S, thr, T)
            d[f"vol{tag}"], d[f"ap{tag}"], d[f"eig{tag}"] = v, ap, eig
        sides.append(d)

    a, b = sides
    w = 0 if (np.nan_to_num(a["pput"], nan=1e9) <= np.nan_to_num(b["pput"], nan=1e9)) else 1
    W, B = sides[w], sides[1 - w]
    f = {
        "put_sbr_worse": W["pput"],
        "put_sbr_better": B["pput"],
    }
    f["cau_sbr_worse"], f["cau_sbr_better"] = _sort2(a["cau"], b["cau"])
    f["putfull_sbr_worse"], f["putfull_sbr_better"] = _sort2(a["put"], b["put"])
    f["str_sbr_worse"], f["str_sbr_better"] = _sort2(a["lv"], b["lv"])
    f["hot_sbr_worse"], f["hot_sbr_better"] = _sort2(a["hot"], b["hot"])
    f["put_sbr_occ_worse"] = min(a["pput_occ"], b["pput_occ"]) if np.isfinite(ref_occ) else np.nan
    f["cau_sbr_occ_worse"] = min(a["cau_occ"], b["cau_occ"]) if np.isfinite(ref_occ) else np.nan
    f["str_sbr_occ_worse"] = min(a["lv_occ"], b["lv_occ"]) if np.isfinite(ref_occ) else np.nan
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
            f[f"{q}{tag}_min"], f[f"{q}{tag}_max"] = _sort2(a[f"{q}{tag}"], b[f"{q}{tag}"])
    f["ref_ratio"] = ref_occ / ref_wb if np.isfinite(ref_occ) else np.nan
    bm = _brain_mask_t1(vol, 1.0)
    f["brain_vol_ml"] = float(bm.sum() * T["voxvol"] / 1000.0)
    f["log_total_counts"] = float(np.log1p(float(vol.sum(dtype=np.float64))))
    vox, mat = native
    f["vox_x"], f["vox_y"], f["vox_z"] = vox
    f["mat_x"], f["mat_y"], f["mat_z"] = mat
    return f


def _load_tier1():
    """Load all tier-1 assets + template structures; return a predict(filepath)->float callable."""
    if sitk is None or lgb is None:
        raise RuntimeError("SimpleITK or lightgbm unavailable for tier-1")
    booster = lgb.Booster(model_file=str(LGBM_MODEL_PATH))
    lr = dict(np.load(LOGISTIC_PATH))
    rv = np.load(ROIS_PATH)
    spec = json.loads(FEATURE_SPEC_PATH.read_text())

    # Register against the reference in SimpleITK's OWN world frame (what register_study.py
    # used as the fixed image = the in-memory ref_resampled it wrote to disk). Using
    # nib_to_sitk(nib.load(...)) instead would give a physically DIFFERENT grid (the RAS origin
    # is -11.5/-11.5, not +11.5/+11.5) and misalign the registration, collapsing the SBR
    # features and driving every prediction toward 1.0.
    fixed = sitk.ReadImage(str(REFERENCE_PATH))

    affine = np.asarray(rv["affine"], dtype=np.float64)
    shape = tuple(int(s) for s in rv["shape"])
    spacing = np.asarray(rv["spacing"], dtype=np.float64)
    voxvol = float(np.prod(spacing))
    ap_axis = int(np.argmax(np.abs(affine[1, :3])))
    spacing_ap = float(spacing[ap_axis])
    WX, WY, WZ = _world_coords(shape, affine)

    def side(i):
        sl = np.asarray(rv[f"side{i}_slices"])
        slices = tuple(slice(int(sl[a, 0]), int(sl[a, 1])) for a in range(3))
        box = np.asarray(rv[f"side{i}_box"], dtype=bool)
        return dict(
            slices=slices, box_sub=box[slices],
            idx_cau=np.flatnonzero(np.asarray(rv[f"side{i}_cau"]).ravel()),
            idx_put=np.flatnonzero(np.asarray(rv[f"side{i}_put"]).ravel()),
            idx_pput=np.flatnonzero(np.asarray(rv[f"side{i}_pput"]).ravel()),
            idx_lv=np.flatnonzero(np.asarray(rv[f"side{i}_lv"]).ravel()),
            wx=WX[slices], wy=WY[slices], wz=WZ[slices],
        )

    T = dict(
        shape=shape, affine=affine, spacing=spacing, voxvol=voxvol, spacing_ap=spacing_ap,
        idx_ref_wb=np.flatnonzero(np.asarray(rv["ref_wb"]).ravel()),
        idx_ref_occ=np.flatnonzero(np.asarray(rv["ref_occ"]).ravel()),
        n_hot=int(rv["n_hot"]),
        sides=[side(0), side(1)],
    )

    feature_order = list(spec["feature_order"])
    blend_w = float(spec["blend_weight_lgbm"])
    eps = float(spec["eps"])
    platt_a = float(spec["platt"]["a"])
    platt_b = float(spec["platt"]["b"])
    _log(f"tier-1 model loaded: {len(feature_order)} features, blend_w={blend_w}, eps={eps}")

    def predict_tier1(filepath: Path) -> float:
        raw = nib.load(str(filepath))
        zo = raw.header.get_zooms()[:3]
        sh = raw.shape[:3]
        native = (tuple(float(z) for z in zo), tuple(int(s) for s in sh))
        src = nib.as_closest_canonical(raw)
        moving = _nib_to_sitk(src)
        out = _register_t1(fixed, moving)
        arr = sitk.GetArrayFromImage(out)
        vol = np.asarray(arr.transpose(2, 1, 0), dtype=np.float32)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        vol[vol < 0] = 0.0
        if tuple(vol.shape) != T["shape"]:
            raise ValueError(f"registered shape {vol.shape} != expected {T['shape']}")
        feats = _tier1_features(vol, native, T)
        X = np.array([[feats[k] for k in feature_order]], dtype=np.float64)
        z_lgb = float(np.asarray(booster.predict(X, raw_score=True)).ravel()[0])
        D = _lr_design(X, lr)
        z_lr = float(np.asarray(D @ lr["coef"] + lr["intercept"]).ravel()[0])
        z = blend_w * z_lgb + (1.0 - blend_w) * z_lr
        p = float(expit(platt_a * z + platt_b))
        return float(np.clip(p, eps, 1.0 - eps))

    return predict_tier1


def _heuristic_predictor():
    def predict_heuristic(filepath: Path) -> float:
        return _heuristic_probability(_features(*_load_volume(filepath)))

    return predict_heuristic


def make_predictor(seed: int = 77):
    """Return an ordered fallback chain of predict(filepath)->float callables.

    Order is tier-1 -> tier-2:
      1. tier-1 LightGBM + spline logistic model (registration-based SBR features), if assets load;
      2. tier-2 registration-free heuristic proxy (the review-fixed _features/_heuristic_probability);
      and main() uses the prevalence constant (0.5485) if both raise.
    seed is retained for interface compatibility but everything is deterministic.
    """
    predictors = []
    tier1_ready = (LGBM_MODEL_PATH.exists() and LOGISTIC_PATH.exists() and ROIS_PATH.exists()
                   and FEATURE_SPEC_PATH.exists() and REFERENCE_PATH.exists())
    if tier1_ready:
        try:
            predictors.append(_load_tier1())
        except Exception:
            traceback.print_exc()
            _log("tier-1 model assets failed to load; falling back to the heuristic")
    # tier-2 heuristic is always available as the next rung.
    predictors.append(_heuristic_predictor())
    return predictors





def main(seed: int = 77):
    # dtype={0: str} keeps the uid index as strings (preserves leading zeros / type).
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0, dtype={0: str})

    # Build a filename->path map from both .nii.gz and .nii so a pattern mismatch does not
    # silently make every scan "missing".
    files = {}
    for p in list(NIFTI_DIR.rglob("*.nii.gz")) + list(NIFTI_DIR.rglob("*.nii")):
        files[p.name.split(".nii")[0]] = p
    _log(f"{len(files)} niftis found for {len(submission_format)} uids")

    try:
        predictors = make_predictor(seed=seed)
    except Exception:
        traceback.print_exc()
        predictors = []  # neutral prior, but loudly
    tier_names = ["tier-1", "tier-2"]

    preds = []
    n_missing = 0
    n_prior = 0
    n_tier1 = 0
    n_tier2 = 0
    tb_shown = 0
    for uid in submission_format.index:
        s = str(uid)
        filepath = files.get(s)
        if filepath is None:
            n_missing += 1
            _log(f"missing scan for uid={s}; using prior {NEUTRAL_P}")
            preds.append(NEUTRAL_P)
            continue
        p = None
        used = -1
        for ti, pred_fn in enumerate(predictors):
            try:
                cand = pred_fn(filepath)
                if not np.isfinite(cand):
                    raise ValueError("non-finite prediction")
                p = cand
                used = ti
                break
            except Exception:
                if tb_shown < 5:
                    tb_shown += 1
                    traceback.print_exc()
                _log(f"[{tier_names[ti] if ti < len(tier_names) else 'tier?'}] error for uid={s}")
        if p is None:
            n_prior += 1
            _log(f"all tiers failed for uid={s}; using prior {NEUTRAL_P}")
            p = NEUTRAL_P
        elif used == 0:
            n_tier1 += 1
        else:
            n_tier2 += 1
        preds.append(p)

    submission_format["is_pathologic"] = preds

    # Format guard: probabilities in [0,1] and no NaN (NaN would be an invalid submission).
    probs = submission_format["is_pathologic"].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all():
        raise ValueError("non-finite prediction in submission; aborting write")
    if not ((probs >= 0.0) & (probs <= 1.0)).all():
        raise ValueError("prediction outside [0,1] in submission; aborting write")

    submission_format.to_csv(WRITE_SUBMISSION)

    missing_rate = n_missing / max(len(submission_format), 1)
    if missing_rate > 0.05:
        _log(f"WARNING: {missing_rate:.3f} missing rate suggests filename mismatch; sample:")
        _log(f"        {[s for s in submission_format.index][:5]}")
    _log(
        f"tier-1={n_tier1}, tier-2={n_tier2}, missing={n_missing} (prior), all-tiers-failed={n_prior} "
        f"(prior); wrote {WRITE_SUBMISSION}"
    )


if __name__ == "__main__":
    main()
