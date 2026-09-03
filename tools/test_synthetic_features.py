"""Synthetic-volume regression test for the tier-2 feature/heuristic path.

Validates the MECHANISM (features compute, respond in the right direction)
without needing nibabel/pandas installed — those exist only inside the official
competition Docker runtime, so they are stubbed here. Full end-to-end validation
happens in the runtime via tools/local_validation.md.

Fixture: 64^3 tissue volume + two striatal 'hot' blobs.
  - symmetric case: both blobs at normal uptake
  - pathologic case: right-blob uptake collapses (unilateral reduction)
Expected: heuristic ranks the pathologic pattern meaningfully higher.
"""
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np

for name in ["nibabel", "scipy", "scipy.ndimage", "pandas"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["scipy"].ndimage = sys.modules["scipy.ndimage"]
sys.modules["pandas"].DataFrame = object

_spec = importlib.util.spec_from_file_location(
    "main", Path(__file__).resolve().parents[1] / "submission_src" / "main.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def make_volume(right_blob_delta: float, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vol = np.clip(0.5 + 0.12 * rng.randn(64, 64, 64), 0.1, None).astype(np.float32)
    vol[20:30, 28:34, 12:20] += 0.5  # left striatal blob (normal uptake)
    vol[20:30, 28:34, 44:52] += right_blob_delta
    return vol


def main() -> None:
    p_sym = m._heuristic_probability(m._sbr_style_features(make_volume(0.5)))
    p_path = m._heuristic_probability(m._sbr_style_features(make_volume(0.02)))
    print(f"symmetric-normal      p={p_sym:.4f}")
    print(f"unilateral-reduction  p={p_path:.4f}")
    assert 0.0 < p_sym < p_path < 1.0, "heuristic must rank unilateral reduction higher"
    # Separation is intentionally modest: tier-2 is a mechanism-validation
    # baseline with placeholder weights. Do NOT tune it against this synthetic
    # fixture — discriminative power comes from the tier-1 model fitted on real
    # training data. This test guards direction and range only.
    assert (p_path - p_sym) > 0.03, "asymmetry response lost (regression)"
    print("OK — tier-2 mechanism responds correctly: reduced unilateral uptake -> higher pathologic probability")


if __name__ == "__main__":
    main()
