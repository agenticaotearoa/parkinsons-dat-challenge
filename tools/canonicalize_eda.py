#!/usr/bin/env python3
"""Canonicalize + resample all scans to a common grid, then EDA the signal.

- nib.as_closest_canonical: force RAS orientation regardless of native affine.
- resample to GRID (mm-consistent) with each scan's own affine via
  nibabel.processing.resample_from_to — anatomy-aligned across scans.
- Output: data/canon/{uid}.nii.gz + group-mean diff stats printed.
"""
import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.processing as nip
from pathlib import Path
import time

ROOT = Path("/Users/agenta/.openclaw-autoclaw/workspace/parkinsons")
TRAIN = ROOT / "data/train"
CANON = ROOT / "data/canon"
CANON.mkdir(parents=True, exist_ok=True)

# canonical grid: 3mm iso, ~76x92x60 covers the brain envelope; we fit later
GRID_SHAPE = (76, 92, 60)
ZOOMS = (3.0, 2.5, 3.0)


def main():
    t0 = time.time()
    df = pd.read_csv(ROOT / "data/downloads/train_labels.csv")
    shapes = {}
    for i, row in df.iterrows():
        uid = row.uid
        src = nib.load(str(TRAIN / f"{uid}.nii.gz"))
        shapes[tuple(src.shape)] = shapes.get(tuple(src.shape), 0) + 1
        canon = nib.as_closest_canonical(src)
        # affine target: identity-anchored grid centered on the data
        target_aff = np.diag(list(ZOOMS) + [1.0])
        target = nib.Nifti1Image(np.zeros(GRID_SHAPE, np.float32), target_aff)
        try:
            res = nip.resample_from_to(canon, (GRID_SHAPE, target_aff), order=1)
        except Exception as e:
            print("RESAMPLE_FAIL", uid, str(e)[:100])
            continue
        data = np.asarray(res.dataobj, dtype=np.float32)
        nib.save(nib.Nifti1Image(data, target_aff), str(CANON / f"{uid}.nii.gz"))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)

    print("native shapes:", sorted(shapes.items(), key=lambda kv: -kv[1])[:6])
    print(f"canonicalized {len(list(CANON.glob('*.nii.gz')))} scans in {time.time()-t0:.0f}s")

    # --- EDA on canonical grid ---
    y = df.is_pathologic.values
    rng = np.random.RandomState(0)
    idx_pos = rng.choice(np.where(y == 1)[0], 300, replace=False)
    idx_neg = rng.choice(np.where(y == 0)[0], 300, replace=False)
    uids = df.uid.values

    def stack_mean(idx):
        acc = np.zeros(GRID_SHAPE, np.float64)
        for i in idx:
            acc += nib.load(str(CANON / f"{uids[i]}.nii.gz")).get_fdata(dtype=np.float32)
        return acc / len(idx)

    m_pos = stack_mean(idx_pos)
    m_neg = stack_mean(idx_neg)
    d = m_neg - m_pos  # positive = pathologic COOLER there (expected striatum)
    print("diff: min=%.1f max=%.1f" % (d.min(), d.max()))
    flat = np.argsort(d.ravel())[-10:][::-1]
    for fi in flat[:5]:
        loc = np.unravel_index(fi, d.shape)
        print("pathologic-cooler at z=%d y=%d x=%d  neg=%.1f pos=%.1f" % (loc[0], loc[1], loc[2], m_neg[loc], m_pos[loc]))
    np.save(ROOT / "data/group_mean_pos.npy", m_pos)
    np.save(ROOT / "data/group_mean_neg.npy", m_neg)
    np.save(ROOT / "data/group_diff.npy", d)
    print("EDA saved.", f"{time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
