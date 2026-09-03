"""DaT Parkinson's Prediction Challenge — submission scaffold.

Contract (from competition runtime examples/template/main.py):
  - DATA_ROOT=/code_execution/data, NIFTI_DIR=<root>/niftis, submission_format.csv at root
  - Must write ./submission.csv at repo root of execution dir
  - Predictions: probability in [0,1] that each DaT-SPECT scan is pathologic (is_pathologic)

Strategy tiers (auto-selected by asset availability):
  1. MODELS/sbr_model.pkl  -> trained sklearn/xgboost/lightgbm model over SBR-style features
  2. No model asset        -> intensity-based SBR proxy heuristic (baseline; replace after
                              training data is downloaded and a model is fitted locally)
  3. Absolute fallback     -> neutral 0.35 (constant valid submission, keeps pipeline green)

All imports below are verified present in the official runtime lock (runtime/uv.lock):
nibabel, numpy, pandas, scipy, scikit-learn, joblib, xgboost, lightgbm.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

DATA_ROOT = Path("/code_execution/data")
NIFTI_DIR = DATA_ROOT / "niftis"
SUBMISSION_FORMAT_PATH = DATA_ROOT / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")

# Model assets are bundled inside submission.zip (no network at execution time).
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "sbr_model.pkl"
FEATURE_SPEC_PATH = ASSETS_DIR / "feature_spec.json"

# Fallback constant when neither model nor heuristic signal is available.
NEUTRAL_P = 0.35


def _load_volume(filepath: Path) -> np.ndarray:
    """Load a nifti as a float volume; keep native resolution."""
    img = nib.load(str(filepath))
    data = img.get_fdata(dtype=np.float32)
    return np.asarray(data)


def _sbr_style_features(vol: np.ndarray) -> dict:
    """Compute striatum-vs-reference uptake features (DaT-SPECT standard approach).

    Approach (scanner/protocol agnostic, no template registration needed):
      - robust intensity normalization via percentiles
      - 'hot' reference = upper-quartile intensity mask (healthy putamen/caudate
        light up strongest; occipital/cortical background is cooler)
      - candidate striatal zones = mid-to-high uptake bands; pathology shows as
        reduced hot-band volume and stronger left/right asymmetry
    Returns a flat dict of scalar features. NOTE: this is a *baseline proxy* —
    real SBR uses template registration (ANTs is available in the runtime for
    the trained-model tier).
    """
    eps = 1e-6
    p = np.percentile(vol, [50, 75, 90, 97, 99])
    span = max(float(p[4] - p[0]), eps)
    norm = np.clip((vol - p[0]) / span, 0.0, 1.5)

    hot = norm >= 0.9          # strongest uptake
    mid = (norm >= 0.55) & (norm < 0.9)  # peristriatal band
    ref = norm >= 0.75         # reference uptake band

    feats = {
        "hot_frac": float(hot.mean()),
        "mid_frac": float(mid.mean()),
        "hot_to_ref": float(hot.sum() / (ref.sum() + eps)),
        "int_p50": float(p[0]),
        "int_p97": float(p[3]),
    }

    # Left/right asymmetry of the hot band over the mid-plane of the volume.
    # Count-based, not mean-based: localized striatal hot voxels are a tiny
    # fraction of each half, so half-volume means dilute the signal ~100x
    # (caught by the synthetic-blob test — see tools/local_validation.md).
    x = vol.shape[-1]
    hot_left = float(hot[..., : x // 2].sum())
    hot_right = float(hot[..., x // 2 :].sum())
    feats["lr_hot_asym"] = abs(hot_left - hot_right) / (hot_left + hot_right + 1.0)
    return feats


def _heuristic_probability(feats: dict) -> float:
    """Baseline mapping from SBR-proxy features to pathology probability.

    Lower striatal hot-band fraction and higher asymmetry -> more likely
    pathologic. Weights are placeholders until fitted on the training set
    (see README-PLAN.md tier 2); they are biased to be conservative.
    """
    score = (
        -2.0 * (feats["hot_frac"] - 0.02) * 10.0
        + 1.5 * feats["lr_hot_asym"]
        - 1.0 * (feats["hot_to_ref"] - 0.25)
    )
    return float(np.clip(1.0 / (1.0 + np.exp(-score)), 0.02, 0.98))


def make_predictor(seed: int = 77):
    """Return a predict(filepath) -> float callable based on available assets."""
    if MODEL_PATH.exists() and FEATURE_SPEC_PATH.exists():
        import joblib

        model = joblib.load(MODEL_PATH)
        spec = json.loads(FEATURE_SPEC_PATH.read_text())
        order = spec["feature_order"]

        def predict_trained(filepath: Path) -> float:
            feats = _sbr_style_features(_load_volume(filepath))
            x = np.array([[feats[k] for k in order]], dtype=np.float64)
            return float(np.clip(model.predict_proba(x)[0, 1], 0.0, 1.0))

        return predict_trained

    def predict_heuristic(filepath: Path) -> float:
        return _heuristic_probability(_sbr_style_features(_load_volume(filepath)))

    return predict_heuristic


def main(seed: int = 77):
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0)

    try:
        predict = make_predictor(seed=seed)
    except Exception:
        predict = None  # fall through to neutral constant

    for uid in submission_format.index:
        filepath = NIFTI_DIR / f"{uid}.nii.gz"
        if not filepath.exists():
            raise FileNotFoundError(f"missing scan for uid={uid}: {filepath}")
        if predict is not None:
            try:
                pred = predict(filepath)
            except Exception:
                pred = NEUTRAL_P
        else:
            pred = NEUTRAL_P
        submission_format.loc[uid, "is_pathologic"] = pred

    submission_format.to_csv(WRITE_SUBMISSION_PATH)


if __name__ == "__main__":
    main()
