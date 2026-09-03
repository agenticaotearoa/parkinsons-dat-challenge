# Runtime Notes — locked dependencies & packaging constraints

Everything here was read statically from `competition-sfmn-parkinsons-runtime/`
(changelog-dated 2026-07-07; `runtime/uv.lock`, `runtime/pyproject.toml`,
`runtime/Dockerfile`, `runtime/entrypoint.sh`, `justfile`). No Docker involved.

## Locked-in dependencies relevant to imaging ML

Environment: **Python 3.12** (`runtime/.python-version` = `3.12`;
`requires-python >=3.12,<3.13`), managed by **uv ≥ 0.11.14**, target platform
**linux/amd64** (justfile `platform_args = "--platform linux/amd64"`; base image
`nvidia/cuda:12.9.0-runtime-ubuntu22.04`, CUDA **12.9**).

Versions below are read directly from `runtime/uv.lock` and apply to the
linux/amd64 runtime target.

### Deep learning / imaging

| Package | Version | Notes |
|---|---|---|
| torch | **2.12.1+cu129** | from the explicit `pytorch-cu129` index; CUDA 12.9 build. Runs CPU-only when no GPU is visible (e.g. emulated on macOS) |
| torchvision | 0.27.1+cu129 | same cu129 index |
| monai | 1.6.0 | 3D medical DL framework |
| torchio | 1.2.1 | 3D loading / patch sampling / augmentation (pure Python) |
| antspyx | 0.6.3 | ANTs registration (template alignment). Wheel only for **x86_64** — the runtime target is amd64, so it IS available in-container |
| SimpleITK | 2.5.5 | registration, resampling, DICOM series I/O |
| nibabel | 5.4.2 | NifTI I/O |
| timm | 1.0.27 | vision model zoo |
| transformers | 4.57.6 | + accelerate 1.13.0, peft 0.19.1, huggingface-hub 0.36.2, safetensors 0.7.0 |
| pytorch-lightning | 2.6.1 | |
| tensorflow | 2.21.0 (and-cuda) | + tf-keras 2.21.0, keras 3.15.1 |
| einops | 0.8.2 | |

### Classical ML / numerics

| Package | Version | Notes |
|---|---|---|
| scikit-learn | **1.8.0** | joblib 1.5.3 (model serialization for tier-1 `sbr_model.pkl`) |
| xgboost | **3.3.0** | |
| lightgbm | **4.6.0** | |
| numpy | **2.2.6** | NumPy 2.x — mind ABI when shipping pickled models |
| pandas | **3.0.3** | ⚠️ pandas 3.x (copy-on-write etc. — behavior differs from pandas 1/2 code) |
| scipy | **1.15.3** | on the linux/**x86_64** runtime target; the lock also carries 1.17.1 for non-x86_64 marker sets only |
| scikit-image | 0.26.0 | GLCM/texture features |
| numba | 0.65.0 | |
| opencv-python-headless | 4.13.0.92 | |
| statsmodels | 0.14.6 | |
| polars | 1.40.1 | |
| loguru | 0.7.3 | the entrypoint/README examples log through loguru; `LOGURU_LEVEL=INFO` is set at run time |
| matplotlib / pillow | 3.11.0 / 12.2.0 | |
| diskcache | 5.6.3 | |

Lockfile discipline: `UV_FROZEN=1`, `UV_COMPILE_BYTECODE=1`, and (runtime stage)
`UV_NO_SYNC=1` are set in the image; pyproject sets `exclude-newer = "7 days"` and
`environments = ["sys_platform == 'linux'"]`. CI runs `just check-lock`
(`uv lock --check`). **No package installs are possible at submission run time**
(no internet) — everything above is preinstalled in the image and importable
(verified by `runtime/tests/test_packages.py`, which also asserts torch CUDA +
tensorflow GPU work when an NVIDIA device is present).

## Constraints on model-asset size and file placement inside submission.zip

- **No size limit is enforced anywhere in this repo** — neither `entrypoint.sh`
  nor the Dockerfile checks archive size. Practical limits come from the
  DrivenData upload page (not documented here — check before bundling big
  assets) and from runtime cost of unzipping/`joblib.load` on the judge machine.
- **Placement**: the entrypoint unzips the archive flat into `/code_execution/`
  and runs `python main.py` with cwd = `/code_execution`. Therefore:
  - `main.py` MUST be at the zip root (also grepped by the entrypoint and by
    `just check-submission`);
  - any bundled model assets should live under an `assets/` dir at the zip root —
    at run time they resolve to `/code_execution/assets/…`, which is exactly what
    `submission_src/main.py` expects via `Path(__file__).resolve().parent / "assets"`;
  - never write into `/code_execution/data` (read-only bind mount). The only
    writable artifacts are `submission.csv` (cwd) and the mounted
    `/code_execution/submission/` dir (zip in, `submission.csv` + `log.txt` out).
- **Permissions/ownership**: container runs as non-root `runtimeuser` (uid/gid
  1000); the repro-zipfile packer normalizes all file perms to 0644 (dirs 0755),
  so everything extracted is readable — no executable-bit tricks will survive.
- **Deterministic packing**: the official recipes (`just pack-submission`,
  `just pack-example`) use `uvx rpzip` (drivendataorg/repro-zipfile): ZIP_STORED
  (no compression), all timestamps 1980-01-01 00:00:00, sorted entries, explicit
  directory entries. Our `tools/pack_submission.sh` replicates this byte-layout
  (verified against the committed example `submission/submission.zip`).
- **Network**: submissions run with `--network none` (platform always blocks
  internet) → any model weights must be bundled in the zip; nothing may be
  fetched from HuggingFace etc. at run time.
- **Memory**: containers get `--shm-size 8g` and `--pid host`; image needs ~21 GB
  disk. No explicit RAM/CPU caps exist in this repo (platform-side limits are not
  published here).

## Packaging history of this zip layout (verification trail)

- `competition-sfmn-parkinsons-runtime/submission/submission.zip` (3130 bytes,
  committed) was produced by `just pack-example minimal` and contains:
  `main.py, model/, model/model.txt, src/, src/__init__.py, src/model.py` —
  all STORED, timestamps 1980-01-01, file attr `0x01A40000`, dir attr
  `0x41ED0010`, create_system 3. Our packer reproduces exactly this metadata
  profile (asserted programmatically on 2026-09-03).
