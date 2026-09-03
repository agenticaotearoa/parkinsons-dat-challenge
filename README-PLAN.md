# DaT Parkinson's Prediction Challenge — Pipeline Plan

**Card**: DrivenData competition 311 — €25,000 total (1st €12.5k / 2nd €7.5k / 3rd €5k)
**Deadline**: 2026-09-16, 11:59 p.m. UTC (13 days from 2026-09-03)
**Task**: Binary classification — for each DaT-SPECT scan (`{uid}.nii.gz`), predict probability of pathologic dopaminergic deficit (`is_pathologic`).
**Runtime**: official Docker GPU image (source of truth: `competition-sfmn-parkinsons-runtime/` in this folder). Submissions have **no network**; every dependency must be pre-installed in the image (verified locked: torch, monai, scikit-learn, xgboost, lightgbm, nibabel, scipy, ANTs, SimpleITK, timm, huggingface-hub).

## Architecture (three tiers, auto-selected at runtime)

| Tier | Trigger | What runs | Status |
|---|---|---|---|
| 1. Trained model | `assets/sbr_model.pkl` + `feature_spec.json` bundled in zip | SBR-style features → gradient-boosted classifier | Awaiting training data |
| 2. Heuristic | no model assets | intensity-band SBR proxy + L/R asymmetry → conservative probability | **implemented** in `submission_src/main.py` |
| 3. Neutral | any runtime failure | constant 0.35 (still a valid submission) | implemented |

## Roadmap to a competitive entry

1. **Unblock data (owner step)** — log into DrivenData (account required for downloads) and fetch:
   - `smoke_test_data.tar.gz` → put niftis into `data-demo/niftis/`, `submission_format.csv` into `data-demo/`
   - full training set → `parkinsons/data/train/`
   - Read the **evaluation metric** and the **external-data policy** on the competition page (a community question about PPMI/DUA-gated data eligibility is open — using gated external pretraining could forfeit prize eligibility; do not assume).
2. **EDA + baseline validation** — run tier-2 heuristic against smoke data locally:
   - `cd competition-sfmn-parkinsons-runtime && just pull` (downloads official image)
   - `.env`: `SUBMISSION_IMAGE=competitionsfmnparkinsonsprodacr.azurecr.io/competition-sfmn-parkinsons-runtime:gpu-latest`
   - pack our source into `submission/submission.zip` (same layout as `just pack-example minimal`), then `just test-submission` — must produce `submission.csv` without error.
3. **Fit tier-1 model** — features per scan: SBR-style striatum/reference ratios with proper template registration (ANTs is in the runtime; MNI-space striatal/occipital masks), left/right putamen & caudate separately, volume/texture stats (scikit-image GLCM). Fit XGBoost/LightGBM with grouped CV; calibrate probabilities.
4. **Stretch** — MONAI 3D CNN (resnet-style) on registered, skull-stripped volumes if data volume justifies it; blend with feature model.
5. **Ship** — final `submission.zip`, local docker green, then **owner uploads** on drivendata.org before 2026-09-16.

## Frontier review (2026-09-03, Claude Sonnet 4.5 via arena.ai free lane)

Reviewed the plan; verdicts after checking against competition rules:

1. **ADOPTED (modified):** ensemble a shallow 3D CNN with the feature model — but note our training compute is local (no CUDA on this Mac; MPS/CPU only), so it trains on downsampled volumes and only after the tier-1 feature model is validated. Not promoted above stretch tier.
2. **ADOPTED:** multi-atlas ROI definition — fuse 3-5 freely licensed public atlases (e.g., subcortical/striatal atlases) via ANTs instead of single-template registration; directly improves SBR feature quality. Atlas licenses to be recorded for the external-data audit trail.
3. **REJECTED as stated:** pseudo-labeling/pretraining on PPMI "derivatives" — processed PPMI files are participant-level data, which the host confirmed is banned for prize eligibility (forum 11472). Allowable residue: published aggregate statistics from papers may serve as sanity ranges only; anything further needs a host ruling before use.

## Owner dependencies (the only things I can't do)

1. DrivenData account login → data download (and final upload of submission.zip).
2. Confirmation of external-data eligibility if we want PPMI pretraining.
3. Payout identity (prize claims go through the account holder).

## Evidence

- Prize/deadline confirmed on drivendata.org competition page + kickoff mirror (searched 2026-09-03).
- Runtime contract taken from the official `drivendataorg/competition-sfmn-parkinsons-runtime` repo (cloned locally, template + lock inspected).
