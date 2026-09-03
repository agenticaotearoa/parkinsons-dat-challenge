# Local Validation Runbook — testing our submission in the official runtime image

This is the exact procedure for validating `parkinsons/submission_src/` inside the
official DrivenData runtime Docker image, before any platform submission.

Source of truth for everything below: `competition-sfmn-parkinsons-runtime/`
(README.md, justfile, runtime/entrypoint.sh, runtime/Dockerfile). That repo is
**read-only** for our workflow; the only files that land in it at test time are
throwaway build artifacts in its `submission/` dir (restored afterwards — step 7).

Paths below are relative to `parkinsons/` (this folder).

---

## 0. Prerequisites (owner machine)

- Docker Desktop running. At least **21 GB free disk** for the GPU runtime image.
- [`just`](https://github.com/casey/just) and [`uv`](https://docs.astral.sh/uv/) installed.
- GPU path (Linux only): NVIDIA drivers with CUDA 12 + NVIDIA Container Toolkit.
  **On this Mac mini (Apple Silicon) there is no GPU** — `just test-submission`
  will run the `linux/amd64` image under Docker emulation on CPU. Fine for smoke
  validation; slow for large data.
- Smoke-test data downloaded from the DrivenData competition data page
  (`smoke_test_data.tar.gz`) — see step 3.

## 1. Point `just` at the official image (`.env`)

```sh
cd competition-sfmn-parkinsons-runtime
cp .env.example .env
```

Then set in `.env` (the `justfile` does `set dotenv-load`):

```dotenv
SUBMISSION_IMAGE=competitionsfmnparkinsonsprodacr.azurecr.io/competition-sfmn-parkinsons-runtime:gpu-latest
```

(`just pull` fetches that image; without this line `just` would look for a local
`competition-sfmn-parkinsons-runtime:gpu-local` and `test-submission` would abort
with "No image found".)

Optional overrides the justfile honors: `CONTAINER_NAME`, `DATA_DIR`,
`BLOCK_INTERNET` (default `true` → container runs with `--network none`; leave it
that way — the platform blocks internet too), `GITHUB_ACTIONS_NO_TTY=true` if you
want non-interactive output.

## 2. Pull the official image

```sh
just pull        # = docker pull competitionsfmnparkinsonsprodacr.azurecr.io/competition-sfmn-parkinsons-runtime:gpu-latest
```

Nothing to compile afterwards — `just build` is NOT needed to test a submission
(your code is mounted at run time, not baked into the image).

## 3. Place the smoke-test data (currently `data-demo/` is EMPTY — only `.gitkeep`)

From the competition site → **data download page** → download
`smoke_test_data.tar.gz`, then unpack so that:

```
competition-sfmn-parkinsons-runtime/data-demo/
├── submission_format.csv      # at the ROOT of data-demo (columns: uid, is_pathologic)
└── niftis/
    └── <uid>.nii.gz           # one NifTI scan per uid in submission_format.csv
```

`just test-submission` bind-mounts `data-demo/` read-only at
`/code_execution/data`. To test against a different folder:
`DATA_DIR=/path/to/data just test-submission`.

## 4. Pack our submission

```sh
cd ..                                  # back to parkinsons/
tools/pack_submission.sh --check       # builds + validates
```

- Produces `submission/submission.zip` (currently: `main.py`, 5618 bytes).
- `--check` additionally verifies `main.py` is at zip root and runs
  `python3 -m py_compile` on the extracted file.
- Layout is byte-compatible with the official `uvx rpzip` packer (STORED
  entries, 1980-01-01 timestamps, 0644/0755 perms, explicit `assets/` dir
  entries, sorted). When a trained model lands in `submission_src/assets/`
  it is picked up automatically.

## 5. Sanity-check the zip with the repo's own validator (optional but cheap)

```sh
cd competition-sfmn-parkinsons-runtime
cp ../parkinsons/submission/submission.zip submission/submission.zip
just check-submission      # unzip -l + asserts main.py at zip root
```

## 6. Run it in the official image

```sh
just test-submission
```

What `just` actually runs (for reference):

```sh
docker run --platform linux/amd64 -it \
  [--gpus all  …only if nvidia-smi exists on host] \
  --network none \
  -e LOGURU_LEVEL=INFO -e IS_SMOKE_TEST="1" \
  --mount type=bind,source=<repo>/data-demo,target=/code_execution/data,readonly \
  --mount type=bind,source=<repo>/submission,target=/code_execution/submission \
  --shm-size 8g --pid host --name competition-sfmn-parkinsons-runtime --rm \
  <SUBMISSION_IMAGE_ID>
```

### Expected output

1. Container logs stream to the terminal (also written to
   `submission/log.txt` on the host). Key entrypoint lines, in order:
   - `INFO | Unpacking submission` — our zip is unzipped into `/code_execution/`
   - `ls -alh` showing `main.py` (+ `assets/` once tier-1 assets ship)
   - `INFO | Data directory contents:` listing `submission_format.csv` and `niftis/`
   - `INFO | Smoke test mode enabled (IS_SMOKE_TEST=1)`
   - `Running submission...` → our `main.py` executes (heuristic tier logs nothing itself)
   - `Exporting submission.csv result...` then `INFO | Script completed its run.`
2. Exit code `0`.
3. On the host, inside `competition-sfmn-parkinsons-runtime/submission/`:
   - `submission.csv`  (copied out of the container; one row per uid,
     `is_pathologic` probabilities in [0,1])
   - `log.txt`         (full run log)

Failure signature to watch for: `ERROR | Script did not produce a submission.csv
file in the main directory.` → exit code 1.

### Debugging

```sh
just interact-container   # bash shell inside the same runtime; inspect /code_execution manually
```

## 7. Clean up the runtime repo afterwards

`submission/submission.zip` is a **git-tracked** file in the runtime repo, and the
run creates untracked `submission/submission.csv` + `submission/log.txt`:

```sh
cd competition-sfmn-parkinsons-runtime
git checkout -- submission/submission.zip   # restore the committed minimal-example zip
rm -f submission/submission.csv submission/log.txt
```

## 8. After local validation is green

1. Make a **smoke test submission** on the DrivenData platform (upload
   `parkinsons/submission/submission.zip`, not the CSV) — smoke tests run on a
   small training-set slice and don't count for prizes.
2. Then submit the same zip as a real submission. Deadline 2026-09-16 23:59 UTC.

---

## Constraints discovered in the runtime (summary)

### `runtime/entrypoint.sh` (what runs inside the container)

- `cd /code_execution`; `zip -sf submission.zip` output is grepped for `main.py`
  — **`main.py` must be at the zip ROOT** or the run aborts.
- Zip is unzipped into `/code_execution/` (flat), then `python main.py` runs with
  cwd = `/code_execution` (`python` = the image venv at `/code_execution/.venv`).
- **`main.py` MUST create `submission.csv` in `/code_execution/` (cwd)**; the
  entrypoint copies it to the mounted `submission/` dir for retrieval.
- All stdout/stderr is teed to `submission/log.txt` (and `/tmp/log` for the
  platform's terminationMessagePath); the container exit code is the script's
  exit code (`${PIPESTATUS[0]}`).
- Environment inside: `IS_SMOKE_TEST=1` and `LOGURU_LEVEL=INFO` are set by
  `just test-submission`; runs as non-root user `runtimeuser` (uid/gid 1000).

### Docker flags / resources (`justfile test-submission`)

| Constraint | Value | Note |
|---|---|---|
| Platform | `--platform linux/amd64` | image is amd64-only; emulated on Apple Silicon |
| Network | `--network none` | `BLOCK_INTERNET` defaults `true`; **no pip installs, no model downloads at run time** — everything must be inside the zip |
| GPU | `--gpus all` only if host has `nvidia-smi` | macOS → CPU-only run |
| Shared memory | `--shm-size 8g` | headroom for PyTorch/MONAI DataLoader workers |
| PID namespace | `--pid host` | avoids PID-limit stalls in frameworks |
| Data mount | `/code_execution/data` **read-only** | never write into `data/` from main.py |
| Submission mount | `/code_execution/submission` rw | zip in / csv + log.txt out |
| Timeouts | **none** in entrypoint or justfile | platform smoke/full runs have their own time limits (not documented in this repo) |
| Env vars expected | `IS_SMOKE_TEST` (set by justfile), `LOGURU_LEVEL` | image also sets `UV_NO_SYNC=1`, `UV_FROZEN=1` — dependency env is frozen |

### Smoke-test data contract (`/code_execution/data`)

- `data/submission_format.csv` — one row per test uid (our `main.py` reads it
  with `index_col=0`).
- `data/niftis/<uid>.nii.gz` — a NifTI scan for **every** uid; our `main.py`
  raises `FileNotFoundError` if any is missing (by design — fail loudly locally).
