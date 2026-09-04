"""Synthetic-volume regression test for the tier-2 feature/heuristic path.

Validates the MECHANISM (features compute, respond in the right direction)
without needing nibabel/pandas installed (those exist only inside the official
competition Docker runtime, so they are stubbed here — but this test only
exercises the numpy/scipy feature+heuristic path anyway).

Fixture: 64^3 tissue volume at 3 mm isotropic (FOV 192 mm) + two striatal 'hot'
blobs straddling the L-R midline.
  - symmetric case: both blobs at normal uptake     -> low pathologic probability
  - pathologic case: right-blob uptake collapses    -> higher pathologic probability
Expected: the heuristic ranks the pathologic pattern meaningfully higher.

NOTE (2026-09-04): the feature API changed per the expert review — `_sbr_style_features`
was replaced by `_features(vol, zooms)` and features are now brain-anchored (fixed-mL hot
spot vs a brain reference rather than self-referential percentiles). The fixture was redone
to describe a plausible DaT-like volume (air background + brain + striatal blobs) and the
assertions updated to the new ranking semantics. Mechanism-only: it guards direction & range,
not absolute values; do NOT tune it or the heuristic constants to this fixture.
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np

# The feature path now uses scipy.ndimage (brain mask) and scipy.special (expit), so scipy
# must be the real package. nibabel/pandas are only needed at import time and the local venv
# has them; this test never performs an I/O run and only imports the module for its functions.
_spec = importlib.util.spec_from_file_location(
    "main", Path(__file__).resolve().parents[1] / "submission_src" / "main.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

ZOOMS = np.array([3.0, 3.0, 3.0], dtype=np.float32)  # 3 mm isotropic


def make_volume(right_blob_peak: float, seed: int = 7) -> np.ndarray:
    """A plausible DaT-like volume: air background, a brain ellipsoid, two striatal blobs.

    `right_blob_peak` is the right-side (axis 0 high half) striatal intensity. Use
    2.2 for the symmetric-normal case and a lower value for unilateral reduction.
    """
    rng = np.random.RandomState(seed)
    nx = ny = nz = 64
    vol = rng.normal(0.0, 0.02, (nx, ny, nz)).astype(np.float32)
    vol = np.clip(vol, 0.0, None)  # 'air' around zero

    cx = cy = cz = 32
    yy, xx, zz = np.mgrid[0:nx, 0:ny, 0:nz]
    rr = ((xx - cx) / 27.0) ** 2 + ((yy - cy) / 25.0) ** 2 + ((zz - cz) / 23.0) ** 2
    brain = rr <= 1.0
    vol[brain] = np.clip(0.3 + 0.5 * rng.rand(np.count_nonzero(brain)), 0.1, None)
    vol[brain] += 0.05 * rng.randn(np.count_nonzero(brain))

    # Striatal blobs straddling the midline. mgrid returns (axis0=L-R, axis1=A-P, axis2=S-I)
    # as (yy, xx, zz). The two blobs MUST differ along axis 0 (L-R); both sit at the AP/SI
    # centre. (This axis handling doubles as a regression guard for the L-R split.)
    left = (np.abs(yy - (cx - 9)) <= 4) & (np.abs(xx - cy) <= 7) & (np.abs(zz - cz) <= 7)
    right = (np.abs(yy - (cx + 9)) <= 4) & (np.abs(xx - cy) <= 7) & (np.abs(zz - cz) <= 7)
    vol[left] = 2.2
    vol[right] = right_blob_peak
    return vol


def main() -> None:
    f_sym = m._features(make_volume(2.2), ZOOMS)
    f_path = m._features(make_volume(0.9), ZOOMS)
    p_sym = m._heuristic_probability(f_sym)
    p_path = m._heuristic_probability(f_path)

    print(f"symmetric-normal      sbr_worse={f_sym['sbr_worse']:.3f}  p={p_sym:.4f}")
    print(f"unilateral-reduction  sbr_worse={f_path['sbr_worse']:.3f}  p={p_path:.4f}")

    assert 0.0 <= p_sym <= 1.0 and 0.0 <= p_path <= 1.0, "probability out of [0,1]"
    assert f_path["sbr_worse"] < f_sym["sbr_worse"], "reduced uptake must lower worse-side SBR"
    assert p_sym < p_path, "heuristic must rank unilateral reduction higher"
    assert (p_path - p_sym) > 0.05, "asymmetry response lost (regression)"

    # The weaker blob should drive the worse-side SBR, and the two sides should be asymmetric
    # only when one collapses (sanity that the L-R split is real, not an axis bug).
    assert f_sym["asym"] < f_path["asym"], "symmetric case should be more symmetric"

    print("OK — tier-2 mechanism responds correctly: reduced unilateral uptake -> lower worse-side SBR -> higher pathologic probability")


if __name__ == "__main__":
    main()
