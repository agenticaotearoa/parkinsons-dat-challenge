#!/usr/bin/env python3
"""Validate _tier1_features against train_tier1_v3's saved OOF features (features_oof_v3.csv).

Loads already-registered train volumes (data/reg) and compares my feature extraction to the
training script's output for the same scans. Isolates feature extraction from registration.
"""
import sys
import numpy as np
import pandas as pd
import nibabel as nib
sys.path.insert(0, "submission_src")
import main

rv = np.load("submission_src/assets/rois_v3.npz")
affine = np.asarray(rv["affine"])
shape = tuple(int(s) for s in rv["shape"])
spacing = np.asarray(rv["spacing"])
voxvol = float(np.prod(spacing))
ap_axis = int(np.argmax(np.abs(affine[1, :3])))
spacing_ap = float(spacing[ap_axis])
WX, WY, WZ = main._world_coords(shape, affine)


def side(i):
    sl = np.asarray(rv[f"side{i}_slices"])
    slices = tuple(slice(int(sl[a, 0]), int(sl[a, 1])) for a in range(3))
    box = np.asarray(rv[f"side{i}_box"], dtype=bool)
    return dict(slices=slices, box_sub=box[slices],
                idx_cau=np.flatnonzero(np.asarray(rv[f"side{i}_cau"]).ravel()),
                idx_put=np.flatnonzero(np.asarray(rv[f"side{i}_put"]).ravel()),
                idx_pput=np.flatnonzero(np.asarray(rv[f"side{i}_pput"]).ravel()),
                idx_lv=np.flatnonzero(np.asarray(rv[f"side{i}_lv"]).ravel()),
                wx=WX[slices], wy=WY[slices], wz=WZ[slices])


T = dict(shape=shape, affine=affine, spacing=spacing, voxvol=voxvol, spacing_ap=spacing_ap,
         idx_ref_wb=np.flatnonzero(np.asarray(rv["ref_wb"]).ravel()),
         idx_ref_occ=np.flatnonzero(np.asarray(rv["ref_occ"]).ravel()),
         n_hot=int(rv["n_hot"]), sides=[side(0), side(1)])

oof = pd.read_csv("data/diag_v3/features_oof_v3.csv", dtype={"uid": str}).set_index("uid")
KEY_FEATS = ["put_sbr_worse", "cau_sbr_worse", "str_sbr_worse", "hot_sbr_worse",
             "put_cau_ratio_worse", "asym_put", "brain_vol_ml", "log_total_counts"]

for uid in ["01nouhtc", "0224wk0y", "049enulq"]:
    img = nib.load(f"data/reg/{uid}.nii.gz")
    vol = np.asarray(img.dataobj, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol[vol < 0] = 0.0
    nimg = nib.load(f"data/train/{uid}.nii.gz")
    zo = nimg.header.get_zooms()[:3]
    sh = nimg.shape[:3]
    native = (tuple(float(z) for z in zo), tuple(int(s) for s in sh))
    f = main._tier1_features(vol, native, T)
    row = oof.loc[uid]
    diffs = {}
    for k in KEY_FEATS:
        mine = f[k]
        theirs = float(row[k])
        diffs[k] = (round(mine, 4), round(theirs, 4), abs(mine - theirs))
    print(f"{uid}:")
    for k, (m, t, d) in diffs.items():
        print(f"   {k:22s} mine={m:8.4f} train={t:8.4f} |diff|={d:.4f}")
    # also verify vox/mat from native header
    print(f"   vox={tuple(round(x,3) for x in native[0])} mat={native[1]} (train vox={tuple(np.round(row[['vox_x','vox_y','vox_z']],3))} mat={tuple(int(x) for x in row[['mat_x','mat_y','mat_z']])})")
