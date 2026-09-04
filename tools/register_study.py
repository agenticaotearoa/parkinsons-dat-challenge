#!/usr/bin/env python3
"""Study-specific registration: rigid MI alignment of every scan to a
study reference, output on a common 2.5mm grid. (Fable review Q3/Q6:
same-modality registration; no MNI needed.)

Steps:
1. Canonicalize (RAS) each scan.
2. Reference = the scan with median brain-mask volume among normals-ish
   (avoid pathologic atrophy bias; any clean scan works for rigid MI).
3. SimpleITK rigid (Euler) + MI, multi-resolution; resample to reference grid.
4. Save data/reg/{uid}.nii.gz on the shared grid.
5. Verify: group-diff map peak must be INSIDE the brain and bilateral-
   symmetric-ish, not at a FOV edge.
"""
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path("/Users/agenta/.openclaw-autoclaw/workspace/parkinsons")
TRAIN = ROOT / "data/train"
REG = ROOT / "data/reg"
REG.mkdir(parents=True, exist_ok=True)


def nib_to_sitk(img_nib) -> sitk.Image:
    """Preserve TRUE physical geometry: origin, direction (from RAS affine), spacing."""
    vol = np.asarray(img_nib.dataobj, dtype=np.float32)
    img = sitk.GetImageFromArray(vol.T)  # sitk is x-fastest; nibabel is z-fastest
    aff = img_nib.affine
    spacing = np.linalg.norm(aff[:3, :3], axis=0)
    direction = aff[:3, :3] / spacing
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(o) for o in aff[:3, 3]))
    img.SetDirection(tuple(float(d) for d in direction.T.flatten()))
    return img


def main():
    t0 = time.time()
    df = pd.read_csv(ROOT / "data/downloads/train_labels.csv")
    uids = df.uid.values

    # reference selection: median brain size scan (proxy for clean acquisition)
    sizes = []
    for u in uids[:400]:
        v = nib.load(str(TRAIN / f"{u}.nii.gz")).get_fdata(dtype=np.float32)
        sizes.append((int((v > np.percentile(v, 50)).sum()), u))
    sizes.sort()
    ref_uid = sizes[len(sizes) // 2][1]
    print("reference scan:", ref_uid, flush=True)

    ref_nib = nib.as_closest_canonical(nib.load(str(TRAIN / f"{ref_uid}.nii.gz")))
    ref_img = nib_to_sitk(ref_nib)

    # resample reference to ~2.5mm iso grid in its own physical space
    old_spacing = np.array(ref_img.GetSpacing())
    new_size = [int(round(s * sp / 2.5)) for s, sp in zip(ref_img.GetSize(), old_spacing)]
    ref_resampled = sitk.Resample(ref_img, new_size, sitk.Transform(),
                                  sitk.sitkLinear, ref_img.GetOrigin(), (2.5, 2.5, 2.5),
                                  ref_img.GetDirection(), 0.0, sitk.sitkFloat32)
    sitk.WriteImage(ref_resampled, str(REG / "_reference.nii.gz"))
    print("reference grid:", ref_resampled.GetSize(),
          "origin:", np.round(ref_resampled.GetOrigin(), 1).tolist(), flush=True)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4,
                                               numberOfIterations=200,
                                               relaxationFactor=0.5)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])

    fails = 0
    empty = 0
    for i, u in enumerate(uids):
        try:
            src = nib.as_closest_canonical(nib.load(str(TRAIN / f"{u}.nii.gz")))
            img = nib_to_sitk(src)
            init = sitk.CenteredTransformInitializer(
                ref_resampled, img, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
            R.SetInitialTransform(init, inPlace=False)
            tx = R.Execute(ref_resampled, img)
            out = sitk.Resample(img, ref_resampled, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            arr = sitk.GetArrayFromImage(out)
            ref_arr = sitk.GetArrayFromImage(ref_resampled)
            occ = (arr[ref_arr > 20] > 10).mean() if (ref_arr > 20).any() else 0.0
            if occ < 0.05:
                empty += 1
                print("REG_EMPTY", u, round(float(occ), 3), flush=True)
            sitk.WriteImage(out, str(REG / f"{u}.nii.gz"))
        except Exception as e:
            fails += 1
            print("REG_FAIL", u, str(e)[:100], flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(uids)} ({time.time()-t0:.0f}s, fails={fails}, empty={empty})", flush=True)

    print(f"DONE {time.time()-t0:.0f}s fails={fails} empty={empty}")
    # quick verify: group diff on registered grid
    y = df.is_pathologic.values
    rng = np.random.RandomState(0)
    ip = rng.choice(np.where(y == 1)[0], 300, replace=False)
    ing = rng.choice(np.where(y == 0)[0], 300, replace=False)

    def mean_of(idx):
        acc = None
        for i in idx:
            v = sitk.GetArrayFromImage(sitk.ReadImage(str(REG / f"{uids[i]}.nii.gz")))
            acc = v.astype(np.float64) if acc is None else acc + v
        return acc / len(idx)

    mp = mean_of(ip)
    mn = mean_of(ing)
    d = mn - mp
    loc = np.unravel_index(np.argmax(d), d.shape)
    print("diff peak at (z,y,x)=%s d=%.1f (neg=%.1f pos=%.1f)" % (str(loc), d.max(), mn[loc], mp[loc]))
    # inside-brain check: reference brain mask bbox
    rv = sitk.GetArrayFromImage(ref_resampled)
    m = rv > 20
    zz, yy, xx = np.where(m)
    print("ref brain bbox z[%d..%d] y[%d..%d] x[%d..%d]" % (zz.min(), zz.max(), yy.min(), yy.max(), xx.min(), xx.max()))
    print("peak inside bbox:", bool(zz.min() <= loc[0] <= zz.max() and yy.min() <= loc[1] <= yy.max() and xx.min() <= loc[2] <= xx.max()))
    np.save(ROOT / "data/group_diff_reg.npy", d)


if __name__ == "__main__":
    main()
