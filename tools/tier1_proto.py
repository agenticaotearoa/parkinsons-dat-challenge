#!/usr/bin/env python3
"""Prototype: tier-1 inference registration + array-order conversion + timing.

Validates that registering a raw scan to the reference grid with SimpleITK matches the
nibabel-order array the tier-1 features expect (i.e. the reverse of register_study.nib_to_sitk).
"""
import time
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path

PARK = Path("/Users/agenta/.openclaw-autoclaw/workspace/parkinsons")
REF = PARK / "submission_src/assets/reference.nii.gz"
DEMO = PARK / "data-demo/niftis"


def nib_to_sitk(img_nib):
    vol = np.asarray(img_nib.dataobj, dtype=np.float32)
    img = sitk.GetImageFromArray(vol.T)
    aff = img_nib.affine
    spacing = np.linalg.norm(aff[:3, :3], axis=0)
    direction = aff[:3, :3] / spacing
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(o) for o in aff[:3, 3]))
    img.SetDirection(tuple(float(d) for d in direction.T.flatten()))
    return img


def register(fixed, moving):
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


fixed_img = nib_to_sitk(nib.load(str(REF)))
fixed_arr_ref = np.asarray(nib.load(str(REF)).dataobj, dtype=np.float32)
print("fixed grid size:", fixed_img.GetSize(), "spacing:", fixed_img.GetSpacing())

for uid in ["cwbfuk9e", "cw2lbm2q"]:
    t0 = time.time()
    src = nib.as_closest_canonical(nib.load(str(DEMO / f"{uid}.nii.gz")))
    moving = nib_to_sitk(src)
    out = register(fixed_img, moving)
    arr_sitk = sitk.GetArrayFromImage(out)
    vol_t = np.asarray(arr_sitk.transpose(2, 1, 0), dtype=np.float32)  # nibabel (x,y,z)
    print(f"{uid}: native={src.shape} reg_time={time.time()-t0:.2f}s out_size={out.GetSize()} arr_sitk={arr_sitk.shape} vol_t={vol_t.shape} finite={np.isfinite(vol_t).all()}")
    # cross-check: write out as a nibabel image with reference affine, then compare orders
    nib_out = nib.Nifti1Image(arr_sitk.transpose(2, 1, 0), np.asarray(nib.load(str(REF)).affine))
    arr_from_disk = np.asarray(nib_out.get_fdata(), dtype=np.float32)
    print(f"   vs disk-read max abs diff = {np.abs(vol_t - arr_from_disk).max():.4g}")
