# ---- DROP-IN PATCH for submission_src/main.py :: tier-1 branch (v4 assets) ----------------------
# Usage inside main.py:
#     from tier1_v4_infer import run_tier1_v4
#     ...
#     if tier == 1:
#         run_tier1_v4(scans, out_csv, assets_dir="assets_v4")     # scans = list of (scan_id, path)
# Save this block as submission_src/tier1_v4_infer.py (train_tier1_v4.py must be importable).
import os
import sys
import json
import time
import traceback

import numpy as np
import joblib
import lightgbm as lgb
from scipy.special import expit

from train_tier1_v4 import (canonicalize_ras, pre_gates, load_template, register_to_template,
                            evaluate_gates_one, extract_features_tta, feature_vector, norm100,
                            write_submission_progressive, GATES, PREVALENCE_REF)


def load_assets(assets_dir="assets_v4"):
    with open(os.path.join(assets_dir, "feature_spec_v4.json")) as f:
        spec = json.load(f)
    tpl = load_template(assets_dir)
    boosters = {int(k): [lgb.Booster(model_file=os.path.join(assets_dir, fn)) for fn in files]
                for k, files in spec["model_files"].items()}
    splines = {int(k): joblib.load(os.path.join(assets_dir, fn)) for k, fn in spec["spline_files"].items()}
    platt = {int(k): (float(v[0]), float(v[1])) for k, v in spec["platt"].items()}
    gates = dict(GATES)
    for k, v in spec["gates"].items():
        gates[k] = tuple(v) if isinstance(v, list) and len(v) == 2 and k in ("voxel_mm",) else v
    gates["native_extent_mm"] = {a: tuple(b) for a, b in gates["native_extent_mm"].items()}
    return dict(spec=spec, tpl=tpl, boosters=boosters, splines=splines, platt=platt, gates=gates,
                feature_order=spec["feature_order"], fill=spec["inference"]["fill_values"],
                spl_idx=spec["inference"]["spline_feature_idx"], w=float(spec["blend"]["w"]),
                eps=float(spec["eps"]), prev=PREVALENCE_REF, tta=spec["inference"]["tta"])


def predict_from_features(feat, A):
    """feat: (1, n_features) in spec order. Per fold: mean raw logit of 3 seeds -> platt(fold)
    -> blend with spline on probabilities; average probabilities over folds."""
    ps = []
    for k in sorted(A["boosters"]):
        raw = float(np.mean([b.predict(feat, raw_score=True)[0] for b in A["boosters"][k]]))
        a, b = A["platt"][k]
        p_lgb = float(expit(a * raw + b))
        p_spl = float(A["splines"][k].predict_proba(feat[:, A["spl_idx"]])[0, 1])
        ps.append(A["w"] * p_lgb + (1.0 - A["w"]) * p_spl)
    return float(np.mean(ps))


def tier2_heuristic(native_vol, A):
    """Crude native-space fallback for hard-gate failures: peak-to-brain ratio, heavily shrunk to prior.
    Returns None if nothing usable (caller then uses prevalence)."""
    try:
        vn = norm100(native_vol)
        m = vn > A["spec"]["inference"]["occ_norm_thr"]
        if m.sum() < 500:
            return None
        peak = float(np.percentile(vn[m], 99.9))
        bg = float(np.median(vn[m]))
        r = peak / (bg + 1e-6)
        t2 = A["spec"]["inference"]["tier2"]
        shift = float(np.clip(0.15 * (t2["r_mid"] - r), -t2["max_shift"], t2["max_shift"]))
        return float(np.clip(A["prev"] + shift, 0.05, 0.95))
    except Exception:
        return None


def predict_scan(path, A):
    t0 = time.time()
    log = {"path": path}
    vol, zooms, origin = canonicalize_ras(path)
    vol, hard, soft, pm = pre_gates(vol, zooms, A["gates"])
    meta = dict(pm, pre_hard=hard, pre_soft=soft, failed=False)
    if not hard:
        try:
            reg, rmeta = register_to_template(vol, zooms, origin, A["tpl"], timeout_s=90.0)
            meta.update(rmeta)
        except Exception as e:  # noqa
            meta["failed"] = True
            meta["error"] = str(e)[:200]
    hard, soft = evaluate_gates_one(meta, A["gates"])
    log.update(hard=hard, soft=soft, reg_method=meta.get("reg_method"), ncc=meta.get("ncc"),
               occ=meta.get("occ_coverage"), peak_dist=meta.get("peak_dist_mm"))
    if hard:
        p = tier2_heuristic(vol, A)
        p = A["prev"] if p is None else p
        log["mode"] = "hard->tier2" if p != A["prev"] else "hard->prevalence"
    else:
        tta = A["tta"]
        fd = extract_features_tta(reg, A["tpl"], meta_extra=meta, n=tta["n"], rot_deg=tta["rot_deg"],
                                  trans_mm=tta["trans_mm"], seed_base=tta["seed_base"])
        feat = feature_vector(fd, A["feature_order"], A["fill"])
        p = predict_from_features(feat, A)
        if soft:
            p = A["prev"] + 0.5 * (p - A["prev"])
        log["mode"] = "soft" if soft else "ok"
    p = float(np.clip(p, A["eps"], 1.0 - A["eps"]))
    log["p"] = p
    log["wall_s"] = time.time() - t0
    return p, log


def run_tier1_v4(scans, out_csv, assets_dir="assets_v4", rewrite_every=50, log_path=None):
    """scans: list of (scan_id, path). Writes all-prevalence first, rewrites every `rewrite_every`,
    survives any exception in the main loop (BaseException), logs per-scan wall time."""
    ids = [s[0] for s in scans]
    probs = np.full(len(ids), PREVALENCE_REF, dtype=np.float64)
    write_submission_progressive(out_csv, ids, probs)
    log_path = log_path or (os.path.splitext(out_csv)[0] + "_tier1_v4_log.jsonl")
    A = load_assets(assets_dir)
    t_all = time.time()
    n_done = 0
    try:
        with open(log_path, "a") as lf:
            for i, (sid, path) in enumerate(scans):
                try:
                    p, log = predict_scan(path, A)
                except BaseException as e:  # per-scan: never lose the row
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    p, log = PREVALENCE_REF, {"path": path, "mode": "exception", "error": traceback.format_exc()[-400:]}
                probs[i] = p
                n_done += 1
                log["id"] = sid
                lf.write(json.dumps(log, default=str) + "\n")
                lf.flush()
                print("[tier1_v4] %d/%d %s p=%.4f mode=%s %.1fs" % (
                    i + 1, len(scans), sid, p, log.get("mode"), log.get("wall_s", 0.0)), flush=True)
                if (i + 1) % rewrite_every == 0:
                    write_submission_progressive(out_csv, ids, probs)
    except BaseException as e:  # main loop: flush what we have, then re-raise only for interrupts
        write_submission_progressive(out_csv, ids, probs)
        print("[tier1_v4] main loop aborted after %d scans: %r" % (n_done, e), file=sys.stderr, flush=True)
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
    write_submission_progressive(out_csv, ids, probs)
    el = time.time() - t_all
    print("[tier1_v4] done %d/%d scans in %.0fs (%.2fs/scan, target <10s)" % (
        n_done, len(scans), el, el / max(1, n_done)), flush=True)
    return out_csv
