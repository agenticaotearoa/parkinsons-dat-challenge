#!/usr/bin/env bash
# pack_submission.sh — build parkinsons/submission/submission.zip from parkinsons/submission_src/
#
# Byte-layout parity with the OFFICIAL packer used by the runtime repo's
# `just pack-submission` / `just pack-example` recipes, which run:
#     cd <source_dir> && uvx rpzip -r <out.zip> ./*
# (`rpzip` = drivendataorg/repro-zipfile, verified against its cli/rpzip.py +
# repro_zipfile/__init__.py source and against the committed
# competition-sfmn-parkinsons-runtime/submission/submission.zip built from
# examples/minimal.)
#
# Exact semantics replicated here:
#   * arcnames are relative to the source dir, no "./" prefix; main.py at zip ROOT
#   * every directory gets an explicit 0-length entry with a trailing slash
#     (e.g. "assets/", "assets/models/")
#   * compression = ZIP_STORED (method 0, no compression)
#   * all timestamps forced to 1980-01-01 00:00:00 (deterministic archives)
#   * unix perms forced: files 0644, dirs 0755 (+ MS-DOS dir flag 0x10)
#   * entries sorted lexicographically by path components (Python sorted(Path))
#   * top-level dotfiles excluded (rpzip receives its inputs from the shell glob
#     "./*", which never matches them)
#   * DELIBERATE DEVIATION: nested dotfiles (e.g. .DS_Store) are excluded here too
#     — rpzip's inner `Path.glob('**/*')` would include them (pathlib glob matches
#     dotfiles), but shipping OS junk is never wanted. Also excluded:
#     __pycache__/ dirs and *.pyc/*.pyo bytecode (the official recipe would include
#     these if present; they are local build artifacts and must not ship).
#
# Idempotent: same source bytes -> byte-identical zip (atomic tmp + rename).
#
# Usage:
#   tools/pack_submission.sh            # pack submission_src/ -> submission/submission.zip
#   tools/pack_submission.sh --check    # + verify layout, py_compile extracted main.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"          # .../parkinsons
SRC_DIR="$ROOT_DIR/submission_src"
OUT_DIR="$ROOT_DIR/submission"
OUT_ZIP="$OUT_DIR/submission.zip"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: source dir not found: $SRC_DIR" >&2
    exit 1
fi
if [[ ! -f "$SRC_DIR/main.py" ]]; then
    echo "ERROR: $SRC_DIR/main.py missing — submission zip must have main.py at root." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

python3 - "$SRC_DIR" "$OUT_ZIP" <<'PYEOF'
import hashlib
import os
import stat
import sys
import zipfile
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])

EXCLUDED_DIR_NAMES = {"__pycache__"}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}

def is_hidden(rel: Path) -> bool:
    """rpzip parity: shell glob `./*` and Path.glob('**/*') never match dotfiles."""
    return any(part.startswith(".") for part in rel.parts)

# ---- collect entries exactly like: rpzip -r out.zip ./* -------------------
# in_paths = {Path(p) for p in globbed_args}; for each dir add path.glob('**/*')
# glob('**/*') yields all non-hidden children recursively (dirs AND files).
entries = {}  # rel Path -> abspath
for child in SRC.iterdir():          # ./* expansion (no dotfiles via glob, but be strict)
    rel = Path(child.name)
    if child.is_dir():
        entries.setdefault(rel, child)
        for sub in child.rglob("*"):
            entries.setdefault(Path(sub.relative_to(SRC)), sub)
    else:
        entries.setdefault(rel, child)

def keep(rel: Path, abspath: Path) -> bool:
    if is_hidden(rel):
        return False
    parts = rel.parts
    if any(p in EXCLUDED_DIR_NAMES for p in parts[:-1]):
        return False
    if abspath.is_dir():
        return parts[-1] not in EXCLUDED_DIR_NAMES
    if rel.suffix in EXCLUDED_FILE_SUFFIXES:
        return False
    return True

entries = {rel: abspath for rel, abspath in entries.items() if keep(rel, abspath)}
if not entries:
    print("ERROR: nothing to pack (source dir effectively empty)", file=sys.stderr)
    sys.exit(1)

# rpzip: for path in sorted(in_paths): zp.write(path)
# sorted(Path) compares tuples of string components; parent sorts before child.
ordered = sorted(entries, key=lambda p: tuple(p.parts))

# repro_zipfile normalization constants (see repro_zipfile/__init__.py)
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
FILE_ATTR = (0o644) << 16                       # -> 0x01A40000
DIR_ATTR = ((0o40000 | 0o755) << 16) | 0x10     # -> 0x41ED0010 (incl. MS-DOS dir flag)

tmp_out = OUT.with_suffix(".zip.tmp")
with zipfile.ZipFile(tmp_out, "w") as zf:
    for rel in ordered:
        abspath = entries[rel]
        arcname = (str(rel) + "/") if abspath.is_dir() else str(rel)
        zi = zipfile.ZipInfo(filename=arcname, date_time=FIXED_DATE_TIME)
        zi.create_system = 3  # unix, deterministic across pack hosts
        zi.external_attr = DIR_ATTR if abspath.is_dir() else FILE_ATTR
        zi.compress_type = zipfile.ZIP_STORED
        if abspath.is_dir():
            zf.writestr(zi, b"")
        else:
            with open(abspath, "rb") as f:
                zf.writestr(zi, f.read())

os.replace(tmp_out, OUT)  # atomic; safe to re-run

data = OUT.read_bytes()
size = len(data)
digest = hashlib.sha256(data).hexdigest()
n = len(zipfile.ZipFile(OUT).infolist())
print(f"Packed {n} entries -> {OUT}")
print(f"zip byte size : {size}")
print(f"sha256        : {digest}")
PYEOF

echo
echo "---- unzip -l $OUT_ZIP ----"
if command -v unzip >/dev/null 2>&1; then
    unzip -l "$OUT_ZIP"
else
    echo "(unzip not found; python listing above already printed entry table)"
fi

# Layout gate mirroring `just check-submission`: main.py must be at zip root.
if unzip -Z1 "$OUT_ZIP" 2>/dev/null | grep -F -x -q -- "main.py"; then
    echo "VALIDATION PASSED: Submission ZIP archive contains main.py."
else
    echo "ERROR: main.py not at zip root." >&2
    exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
    VERIFY_DIR="$ROOT_DIR/.openclaw/tmp/pack_verify"
    rm -rf "$VERIFY_DIR"
    mkdir -p "$VERIFY_DIR"
    unzip -q "$OUT_ZIP" -d "$VERIFY_DIR"
    python3 -m py_compile "$VERIFY_DIR/main.py"
    echo "py_compile PASSED for extracted main.py"
    rm -rf "$VERIFY_DIR"
fi
