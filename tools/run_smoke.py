"""Local smoke driver for fixed submission_src/main.py.

Runs the full main() path (reads submission_format.csv, predicts every uid, writes a
submission.csv) against the data-demo/ smoke set, and FORCES the heuristic tier by pointing
the model-asset constants at nonexistent files: the bundled assets/sbr_model.pkl is a stale
28-feature ROI model whose feature_order does not match the SBR feature set, so main() would
otherwise auto-fall-back to the heuristic anyway (see main.make_predictor). Forcing keeps this
a clean, explicit test of the heuristic tier independent of that auto-detection.

Usage:
  DATA_ROOT=/path/to/data-demo python run_smoke.py [mid] [slope]
    mid    heuristic sbr_worse midpoint (default: main.MID_SBR)
    slope  heuristic logistic slope   (default: main.SLOPE)

Prints log loss vs test_labels.csv and CSV-structure checks.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]  # parkinsons/
sys.path.insert(0, str(HERE / "submission_src"))
import main  # noqa: E402

DATA = HERE / "data-demo"
NIFTIS = DATA / "niftis"

# Force the heuristic tier (no model assets reachable).
main.MODEL_PATH = Path("/nonexistent/sbr_model.pkl")
main.FEATURE_SPEC_PATH = Path("/nonexistent/feature_spec.json")
main.DATA_ROOT = DATA
main.NIFTI_DIR = NIFTIS
main.SUBMISSION_FORMAT_PATH = DATA / "submission_format.csv"
main.WRITE_SUBMISSION = DATA / "_smoke_out" / "submission.csv"

if len(sys.argv) > 1:
    main.MID_SBR = float(sys.argv[1])
if len(sys.argv) > 2:
    main.SLOPE = float(sys.argv[2])

main.WRITE_SUBMISSION.parent.mkdir(exist_ok=True)

main.main()

out = pd.read_csv(main.WRITE_SUBMISSION, index_col=0, dtype={0: str})
fmt = pd.read_csv(DATA / "submission_format.csv", index_col=0, dtype={0: str})
labels = pd.read_csv(DATA / "test_labels.csv", index_col=0, dtype={0: str})

# ---- structure checks -----------------------------------------------------
assert list(out.index) == list(fmt.index), "uid index mismatch vs submission_format.csv"
assert list(out.columns) == list(fmt.columns), "columns mismatch vs submission_format.csv"
assert out.columns.tolist() == ["is_pathologic"], "expected exactly the is_pathologic column"
assert out["is_pathologic"].notna().all(), "NaN present in is_pathologic"
p = out["is_pathologic"].to_numpy(dtype=np.float64)
assert ((p >= 0.0) & (p <= 1.0)).all(), "probability outside [0,1]"
assert (out.index == out.index.astype(str)).all(), "uid index not strings"

# ---- log loss -------------------------------------------------------------
y = labels.loc[out.index, "is_pathologic"].to_numpy(dtype=np.float64)
eps = 1e-9
logloss = -np.mean(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))

# ---- comparisons ----------------------------------------------------------
c50 = -np.mean(y * np.log(0.5) + (1 - y) * np.log(0.5))
cv = -np.mean(y * np.log(main.PREVALENCE) + (1 - y) * np.log(1 - main.PREVALENCE))

print(f"\nlogloss(heuristic mid={main.MID_SBR}, slope={main.SLOPE}) = {logloss:.4f}")
print(f"  constant 0.5 baseline        = {c50:.4f}  (review's 0.6931 target)")
print(f"  constant prevalence {main.PREVALENCE} = {cv:.4f}")
print(f"  n={len(out)}, pos_rate={y.mean():.3f}")
print(f"  pred range=[{p.min():.4f},{p.max():.4f}] mean={p.mean():.4f}")
print("\nStructure OK: uid index + is_pathologic column match submission_format.csv; probs in [0,1]; no NaN.")
