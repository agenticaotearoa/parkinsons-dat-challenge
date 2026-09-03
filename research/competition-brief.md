# DaT Parkinson's Prediction Challenge — Competition Brief

Compiled 2026-09-03 from live-fetched primary sources (cache: `research/raw/`). Every claim cites its source. UNCONFIRMED = not verifiable without a DrivenData login.

**Card**: [DrivenData #311](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/) — €25,000 (1st €12,500 / 2nd €7,500 / 3rd €5,000). Ends 2026-09-16 11:59 pm UTC (`overview.txt`, drivendata competition page, kickoff mirror). ~837 teams joined, ~1 week remaining as of fetch (`leaderboard.html`).

## 1. Task & evaluation metric

- Binary classification: predict `is_pathologic` probability for each DaT-SPECT (dopamine transporter) scan, one exam at a time.
- **Metric: LOG LOSS** — "Predictions are scored with log loss" (`page989-code-format.txt` line 118). Random guess baseline = ln 2 ≈ 0.693.
- Submission = `submission.zip` code submission; `submission_format.csv` has exactly two columns: `uid,is_pathologic` (`page989-code-format.txt` lines 110-141).
- Final ranking: **best-scoring submission on the private leaderboard counts automatically** — no explicit final-selection step (forum 11493, staff answer).
- Test-set size and public/private split: **not disclosed** (forum 11493, staff).
- Per-sample independence during inference: no pseudo-labeling or any across-test-sample aggregation (official rules, `rules.txt` §98-99).

## 2. Submission mechanics & runtime constraints

- Runs in official Docker runtime, Python 3.12, **no network access inside the runtime** (`page989-code-format.txt` line 177). Every dependency must be pre-installed (locked in `runtime/uv.lock`; see `tools/runtime_notes.md` from our local clone).
- Smoke-test environment: replicates inference runtime on a small training-sample subset; **not counted for prizes** — used to test correctness/speed (`page989-code-format.txt` §157-169).
- Logs capped at 300 lines × 300 chars (`page989-code-format.txt` line 52). **Printing/logging any test-dataset content is grounds for disqualification** (line 178).
- Execution machines are shared; there is a time limit (exact value in runtime repo docs — see `tools/runtime_notes.md`); test locally first, add progress logs, cancel doomed jobs (line 193).
- Data access: use direct paths, not `glob()` (forum 11467, staff recommendation).
- Zip upload: no hard size cap; host increased upload time so >1 GB zips upload successfully (forum 11470, staff). Model weights ship inside `submission.zip` (forum 11480, staff — example exists).

## 3. Data description

- DaT-SPECT nifti volumes (`{uid}.nii.gz`) + `submission_format.csv`. Ground truth: pathologic vs normal dopaminergic scan (parkinsonian syndromes).
- Participants report the dataset is **very small for training neural nets from scratch** (forum 11490). Exact train/test counts: UNCONFIRMED (login-gated).

## 4. External data / pre-trained model policy (decision-critical)

Official rule (`rules.txt` §47-50, §270): external data and pre-trained models ARE permitted **provided you hold all rights/licenses**, and for **prize eligibility** data must be freely publicly available to all participants under a license permitting commercial use (no NC/CC-NC).

Host clarifications (staff = DrivenData):
- **PPMI is NOT prize-eligible.** Its DUA restricts use to scientific investigation/teaching/clinical-research planning; competition use doesn't qualify, and winning models must be openly commercially licensed (forum 11472, staff).
- **Permissively licensed weights ARE eligible** (MIT / Apache-2.0-style, derivative works usable commercially); license of specific weights is verified if your solution places (forum 11475, staff — ImageNet case). DINOv3 (custom license, sign-up gated): question asked, **no staff answer yet** (forum 11488) → avoid unless permissively licensed.
- Weights must be bundled in `submission.zip`; external data used only for training need not be included (forum 11480, staff).

## 5. ⚠️ Data-handling rule — directly constrains OUR workflow (agent + cloud AI)

Host quote (forum 11479, staff): *"Never share, copy, or publish the data… you may not use tools like Codex and ChatGPT that store or retain uploaded data, though you may download model weights and run models locally."*

**Implication**: the NIfTI files and labels must NEVER be uploaded to any cloud service that stores/retains them. Our pipeline is compliant as long as: cloud AI (this assistant) writes/tests CODE only; training, feature extraction and inference run **locally**; only aggregate, non-data artifacts (code, zips of weights) travel. No cloud-API training runs on competition data.

## 6. Prize eligibility (human requirements)

- Open to natural persons ≥18, legal residents of a non-sanctioned country (OFAC-excluded and Russia excluded), individuals only — no entities (`rules.txt` §74-79).
- Register via the website within the competition period; data access follows registration (`rules.txt` §81).
- Prizes paid by check/electronic transfer **to the individual**, ~30 days after verification; US residents receive IRS 1099; winner documentation incl. U.S. tax forms required within 15 days of prize notification (`rules.txt` §122-129).
- Winning code: winners grant the sponsor broad rights; solutions must be reproducible with open-source (commercial-use-permitting) software (`rules.txt` §46, §138-154).
- Team rules: one team per person, separate accounts per member, even prize split by default (`rules.txt` §92-95).

## 7. Forum signal (approaches & gotchas)

- Small data confirmed by multiple participants; transfer-learning / external-pretraining questions dominate (11488, 11490).
- Smoke-test "failed despite valid submission.csv" debugging thread exists — failure modes happen in packaging (11469).
- Team merges vs submission quota question raised (11494) — merge only before submitting.

## 8. Analogous DrivenData imaging competitions

- **Clog Loss (Alzheimer's, 3D image-stack classification)**: winners' code MIT-licensed; 3D deep learning + careful CV dominated (`repo-clog-loss-alzheimers-research.md`).
- **TissueNet cervical biopsies** (histology classification, ~large dataset): pretrained CNNs + ensembling (`repo-tissuenet-cervical-biopsies.md`).
- **VISIOMEL melanoma** (dermoscopic images): transfer learning from ImageNet-scale backbones (`repo-visiomel-melanoma.md`).
- **PREPARE ADRD** was report-based (qualitative) — less analogous (`repo-prepare-adrd.md`).

## 9. Implications for our approach (recommendations)

1. **Optimize log loss, not accuracy**: predict well-calibrated probabilities; use probability calibration (temperature scaling / isotonic on out-of-fold predictions) before finalizing.
2. **PPMI is off the table for prize eligibility** — build on competition data + permissively-licensed resources only; audit every pretrained weight's license (Apache-2.0/MIT OK) before including it in the zip.
3. **Architecture compliance**: competition data never leaves the local machine. Training/inference: local. I (assistant) write and review code, never ingest the imaging data.
4. **Small data ⇒ start with engineered features**: template-registered striatal uptake ratios (SBR-style, ANTs available in runtime), L/R putamen-caudate asymmetry, GLCM texture → XGBoost/LightGBM with grouped CV. Treat heavy 3D CNNs as a stretch tier only via permissively-licensed pretrained backbones, and only after the feature model is validated on smoke data.
5. **Ship complete**: weights inside `submission.zip` (zip may exceed 1 GB); verify locally in the official Docker image first (see `tools/local_validation.md`); add progress logging within the 300-line cap; direct-path data access; never log anything about test scans.
6. **Beat the random baseline first**: our heuristic scaffold targets log loss well below 0.693 on smoke data before any training happens — cheap signal that the packaging + inference path is sound.
7. **Register and submit early** (deadline Sep 16 UTC, ~2 weeks): earliest valid private-LB submission banks the "best private score counts" rule; iterate from there.
8. **Open questions to check with the owner's login**: exact metric restatement on the competition page, daily submission cap, execution time limit value, train data volume, and whether the competition Data itself (not PPMI) is MJFF-sourced with any special attribution rules (`rules.txt` §42-46 mention "provided to top-ranking participants").

## Sources (all fetched live 2026-09-03, cached in `research/raw/`)

- `overview.txt` / `rules.txt` — drivendata.org competition overview + official rules pages
- `page989-code-format.txt` — code submission format page (metric, mechanics, constraints)
- `leaderboard.html` / `leaderboard.txt` — leaderboard page (joined count)
- `sfmn.txt` / `hdh.txt` — competition + Health Data Hub (host) pages
- `forum-11462..11494.json` — community forum threads (staff answers marked in text)
- `winners-readme.md`, `repo-prepare-adrd.md`, `repo-clog-loss-alzheimers-research.md`, `repo-tissuenet-cervical-biopsies.md`, `repo-visiomel-melanoma.md` — analogous winners' code repos
- Local clone: `competition-sfmn-parkinsons-runtime/` (template, lockfile) + `parkinsons/tools/runtime_notes.md`
