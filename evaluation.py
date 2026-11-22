"""
evaluation.py - event-level evaluation of vehicle counting.

Ground truth = list of manually annotated passage times (seconds) per video.
A detection matches a ground-truth event if |t_det - t_gt| <= tol (seconds); matching is
one-to-one and optimal (Hungarian on |dt|), so a detection can never be reused.

Reported: TP, FP, FN, precision, recall, F1, signed count bias (N_det - N_gt),
absolute count error |N_det - N_gt| and relative count error (%), with bootstrap 95% CIs
(resampling fixed-length time blocks to respect temporal correlation).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_events(path) -> np.ndarray:
    """CSV with a column 't_s' (seconds). Other columns (lane, class, note, annotator) ignored."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.sort(np.array([float(r["t_s"]) for r in rows]))


def match(gt: np.ndarray, det: np.ndarray, tol: float):
    gt = np.asarray(gt, float); det = np.asarray(det, float)
    if len(gt) == 0 or len(det) == 0:
        return [], list(range(len(gt))), list(range(len(det)))
    cost = np.abs(gt[:, None] - det[None, :])
    big = 1e6
    c = np.where(cost <= tol, cost, big)
    r, k = linear_sum_assignment(c)
    pairs = [(i, j) for i, j in zip(r, k) if c[i, j] < big]
    mg = {i for i, _ in pairs}; md = {j for _, j in pairs}
    fn = [i for i in range(len(gt)) if i not in mg]
    fp = [j for j in range(len(det)) if j not in md]
    return pairs, fn, fp


def metrics(gt, det, tol):
    pairs, fn, fp = match(gt, det, tol)
    tp = len(pairs)
    P = tp / (tp + len(fp)) if tp + len(fp) else 0.0
    R = tp / (tp + len(fn)) if tp + len(fn) else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    ngt, ndet = len(gt), len(det)
    dts = [det[j] - gt[i] for i, j in pairs]
    return {"N_gt": ngt, "N_det": ndet, "TP": tp, "FP": len(fp), "FN": len(fn),
            "precision": P, "recall": R, "F1": F,
            "count_bias": ndet - ngt, "abs_count_error": abs(ndet - ngt),
            "rel_count_error_pct": 100.0 * abs(ndet - ngt) / ngt if ngt else np.nan,
            "median_abs_dt_s": float(np.median(np.abs(dts))) if dts else np.nan,
            "fn_idx": fn, "fp_idx": fp, "pairs": pairs}


def bootstrap_ci(gt, det, tol, duration_s, block_s=30.0, B=2000, seed=0):
    """Moving-block bootstrap over time blocks. Returns 95% CIs for P, R, F1, bias, rel. error."""
    rng = np.random.default_rng(seed)
    nb = max(1, int(np.ceil(duration_s / block_s)))
    edges = np.arange(nb + 1) * block_s
    g_blk = [gt[(gt >= edges[b]) & (gt < edges[b + 1])] - edges[b] for b in range(nb)]
    d_blk = [det[(det >= edges[b]) & (det < edges[b + 1])] - edges[b] for b in range(nb)]
    out = {k: [] for k in ("precision", "recall", "F1", "count_bias", "rel_count_error_pct")}
    for _ in range(B):
        idx = rng.integers(0, nb, nb)
        g = np.concatenate([g_blk[b] + i * block_s for i, b in enumerate(idx)]) if nb else gt
        d = np.concatenate([d_blk[b] + i * block_s for i, b in enumerate(idx)]) if nb else det
        if len(g) == 0:
            continue
        m = metrics(g, d, tol)
        for k in out:
            out[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items() if v}


def inter_annotator(a, b, tol):
    """Agreement between two annotators: F1 of A vs B, timing offsets, and a consensus list
    (matched pairs averaged; unmatched events flagged for adjudication)."""
    m = metrics(a, b, tol)
    cons = [0.5 * (a[i] + b[j]) for i, j in m["pairs"]]
    disputed = [("A", float(a[i])) for i in m["fn_idx"]] + [("B", float(b[j])) for j in m["fp_idx"]]
    dts = np.array([b[j] - a[i] for i, j in m["pairs"]])
    return {"N_A": len(a), "N_B": len(b), "agreement_F1": m["F1"], "matched": m["TP"],
            "only_A": m["FN"], "only_B": m["FP"],
            "dt_mean_s": float(dts.mean()) if len(dts) else np.nan,
            "dt_std_s": float(dts.std(ddof=1)) if len(dts) > 1 else np.nan,
            "consensus_t_s": sorted(cons), "to_adjudicate": disputed}


def select_tolerance(gt_cal, ann2_cal=None, k=3.0, floor=0.2):
    """Tolerance chosen WITHOUT looking at detector output: k x std of inter-annotator timing
    offsets on the calibration subset (fallback: floor). Avoids tuning on the test set."""
    if ann2_cal is None:
        return floor
    r = inter_annotator(gt_cal, ann2_cal, tol=1.0)
    s = r["dt_std_s"]
    return max(floor, k * s) if np.isfinite(s) else floor


def write_events(path, times, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + (list(extra[0].keys()) if extra else []))
        for i, t in enumerate(times):
            w.writerow([f"{t:.3f}"] + (list(extra[i].values()) if extra else []))
