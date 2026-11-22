"""
tvic.py - Temporal Image-Intensity Vehicle Counting (core library)

Reference implementation used in the revised manuscript
"Temporal Image-Intensity Analysis for Vehicle Counting" (IEEE Access, resubmission).

Pipeline (one pass, streaming; no intermediate .npy is required):
    decode frame -> resize (documented interpolation / aspect policy) -> BGR->gray (uint8)
    -> mean intensity inside polygonal ROI  I(t)
    -> baseline B(t): global median (offline) | centred moving median | causal moving median
    -> D(t) = |I(t) - B(t)|
    -> scipy.signal.find_peaks(D, height=h, distance=d_frames)

All parameters are explicit, physical units are documented, and every run can be
serialised to JSON so that results are reproducible.
"""
from __future__ import annotations

import json
import os
import time
import tracemalloc
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

INTERP = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "cubic": cv2.INTER_CUBIC,
}

def safe_path(path) -> str:
    """Path that OpenCV can open on Windows. OpenCV fails on folders with accents or other non-ASCII
    characters (e.g. 'publicacoes' written with c-cedilla); the Windows short (8.3) name avoids them."""
    here = os.path.dirname(os.path.abspath(__file__))
    p = find_file(path, [here, os.getcwd()])
    if os.name == "nt" and (not p.isascii() or len(p) >= 250):
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(32768)
            if ctypes.windll.kernel32.GetShortPathNameW(p, buf, 32768):
                s = buf.value
                return s[4:] if s.startswith("\\\\?\\") and len(s) < 250 else s
        except Exception:
            pass
    return p


def find_file(path, search_dirs=()) -> str:
    """Return an existing path for `path`. Windows consoles sometimes hand over a dragged path with
    accents in another code page, so a path that does not exist is also tried re-encoded, in
    Unicode normal forms, and finally by its file name in the given folders."""
    import unicodedata
    p = str(path).strip().strip('"').strip()
    cands = [p, unicodedata.normalize("NFC", p), unicodedata.normalize("NFD", p)]
    for a, b in (("cp850", "cp1252"), ("cp1252", "cp850"), ("latin-1", "utf-8"), ("cp850", "utf-8")):
        try:
            cands.append(p.encode(a).decode(b))
        except Exception:
            pass
    if os.name == "nt":                       # paths over 260 characters need the \\?\ prefix
        cands += ["\\\\?\\" + os.path.abspath(c) for c in list(cands) if not c.startswith("\\\\?\\")]
    for c in cands:
        if os.path.exists(c):
            return c
    name = os.path.basename(p.replace(chr(92), "/"))
    for d in search_dirs:
        for c in (name, unicodedata.normalize("NFC", name)):
            q = os.path.join(d, c)
            if os.path.exists(q):
                return q
    return p


def open_video(path):
    return cv2.VideoCapture(safe_path(path))



# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    """Detection parameters. Units are given in the field names / comments."""
    height: float = 3.0              # peak height threshold; gray levels (0-255) if height_mode='abs'
    height_mode: str = "abs"         # 'abs' (gray levels) or 'kmad' (multiples of robust noise sigma of D)
    min_sep_s: float = 0.40          # minimum peak separation in SECONDS (converted with fps)
    baseline: str = "global_median"  # 'global_median' | 'moving_median' | 'causal_median'
    baseline_win_s: float = 60.0     # window of moving / causal median, seconds
    prominence: float | None = None  # optional, gray levels
    interp: str = "area"             # resize interpolation: area | linear | nearest | cubic
    aspect: str = "preserve"         # 'preserve' (letterbox-free, exact target keeps AR) | 'stretch'

    stabilize: object = False        # False | True (on the fly) | path to transforms .npy (estimate_motion)

    def distance_frames(self, fps: float) -> int:
        return max(1, int(round(self.min_sep_s * fps)))


@dataclass
class ROI:
    """Polygon in NORMALISED coordinates (x/W, y/H) of the native frame."""
    polygon: list  # [[x0,y0],[x1,y1],...] in [0,1]
    name: str = "roi"

    @staticmethod
    def load(path) -> "ROI":
        here = os.path.dirname(os.path.abspath(__file__))
        path = find_file(path, [os.path.join(here, "rois"), here, os.getcwd(), os.path.join(os.getcwd(), "rois")])
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return ROI(polygon=d["polygon_normalized"], name=d.get("name", "roi"))

    def save(self, path, native_wh=None):
        d = {"name": self.name, "polygon_normalized": self.polygon}
        if native_wh:
            W, H = native_wh
            d["native_resolution"] = [W, H]
            d["polygon_pixels_native"] = [[round(x * W, 1), round(y * H, 1)] for x, y in self.polygon]
        Path(path).write_text(json.dumps(d, indent=2))

    def mask(self, w: int, h: int) -> np.ndarray:
        """Rasterise the polygon at resolution (w,h). Rounding rule: vertex = round(x*w), round(y*h);
        pixel included if its centre lies inside/on the polygon (cv2.fillPoly)."""
        pts = np.array([[round(x * w), round(y * h)] for x, y in self.polygon], dtype=np.int32)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        return m.astype(bool)

    def perturbed(self, rng: np.random.Generator, sigma_norm: float) -> "ROI":
        p = np.asarray(self.polygon, float) + rng.normal(0, sigma_norm, (len(self.polygon), 2))
        return ROI(polygon=np.clip(p, 0, 1).tolist(), name=self.name + "_pert")


# --------------------------------------------------------------------------- #
# Camera-motion compensation ("anchor" registration, adapted by F. S. Rodrigues)
# --------------------------------------------------------------------------- #
STAB_WIDTH = 640   # motion is always estimated at this width, independently of the processing size


class Stabilizer:
    """Registers every frame to an anchor frame (frame 0).

    Anchor points: Shi-Tomasi corners of frame 0 outside the (dilated) ROI, so vehicles inside the
    ROI do not pull the estimate. Each frame: pyramidal Lucas-Kanade tracking of the anchor points
    from frame 0, then a RANSAC similarity transform (rotation, uniform scale, translation).
    If tracking fails, the previous transform is kept (never a jump back to identity).

    The transform is returned in NORMALISED coordinates (x/W, y/H), as a 3x3 matrix A that maps a
    point of the current frame to the anchor frame. Because it is estimated at a fixed width
    (STAB_WIDTH) it is the same for every processing resolution, so the resolution study is not
    confounded by the stabiliser. Instead of resampling the image, the ROI polygon is mapped into the
    current frame with A^-1 (no extra interpolation of the pixels that are averaged).
    """

    def __init__(self, first_bgr: np.ndarray, roi: "ROI | None" = None, width: int = STAB_WIDTH):
        H, W = first_bgr.shape[:2]
        self.aw = int(width); self.ah = max(1, int(round(width * H / W)))
        self.g0 = self._gray(first_bgr)
        fmask = np.full((self.ah, self.aw), 255, np.uint8)
        if roi is not None:
            m = roi.mask(self.aw, self.ah).astype(np.uint8)
            m = cv2.dilate(m, np.ones((15, 15), np.uint8))
            fmask[m > 0] = 0
        self.p0 = cv2.goodFeaturesToTrack(self.g0, maxCorners=400, qualityLevel=0.01,
                                          minDistance=max(2, min(self.aw, self.ah) // 30),
                                          blockSize=3, mask=fmask)
        self.S = np.diag([self.aw, self.ah, 1.0]); self.Si = np.linalg.inv(self.S)
        self.A = np.eye(3)
        self.n_fail = 0; self.n = 0; self.inliers = []

    def _gray(self, bgr):
        return cv2.cvtColor(cv2.resize(bgr, (self.aw, self.ah), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)

    def update(self, bgr: np.ndarray) -> np.ndarray:
        """Return A (3x3, normalised, current frame -> anchor) for this frame."""
        self.n += 1
        if self.p0 is None or len(self.p0) < 10:
            self.n_fail += 1; return self.A
        g = self._gray(bgr)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.g0, g, self.p0, None,
                                             winSize=(31, 31), maxLevel=4)
        ok = st is not None and p1 is not None
        idx = st.ravel() == 1 if ok else None
        if not ok or idx.sum() < 10:
            self.n_fail += 1; return self.A
        M, inl = cv2.estimateAffinePartial2D(p1[idx], self.p0[idx], method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0)
        if M is None or inl is None or inl.sum() < 10:
            self.n_fail += 1; return self.A
        self.inliers.append(int(inl.sum()))
        self.A = self.Si @ np.vstack([M, [0, 0, 1]]) @ self.S
        return self.A

    @staticmethod
    def roi_in_frame(roi: "ROI", A: np.ndarray) -> "ROI":
        """ROI drawn on the anchor frame, mapped into the current frame."""
        P = np.c_[np.asarray(roi.polygon, float), np.ones(len(roi.polygon))]
        Q = (np.linalg.inv(A) @ P.T).T
        return ROI(polygon=(Q[:, :2] / Q[:, 2:]).tolist(), name=roi.name)

    @staticmethod
    def warp_to_anchor(img: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Resample a frame (any size) into the anchor geometry; used only for display/annotation."""
        h, w = img.shape[:2]
        S = np.diag([w, h, 1.0])
        M = (S @ A @ np.linalg.inv(S))[:2]
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def estimate_motion(video, roi: "ROI | None" = None, out=None, max_frames=None):
    """Pre-compute the per-frame transforms of a video (n,3,3); optionally save to .npy.
    The same file is used by the detector, the baselines and the annotation tool."""
    cap = open_video(video); ok, fr = cap.read()
    if not ok:
        raise IOError(video)
    st = Stabilizer(fr, roi); As = [np.eye(3)]
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and len(As) >= max_frames):
            break
        As.append(st.update(fr).copy())
    cap.release()
    As = np.asarray(As)
    if out:
        np.save(out, As)
    return As, {"frames": len(As), "failed_frames": st.n_fail,
                "median_inliers": float(np.median(st.inliers)) if st.inliers else 0.0}


# --------------------------------------------------------------------------- #
# Video metadata
# --------------------------------------------------------------------------- #
def measured_fps(path, frames: int = 300):
    """Frame rate measured from the decoded timestamps of the first frames.
    The metadata of the camera recordings (.ts) report half the real rate (15 instead of ~30 fps),
    so the metadata value is only trusted when it agrees with the measurement within 5 %."""
    cap = open_video(path)
    meta = cap.get(cv2.CAP_PROP_FPS) or 0.0
    t = []
    for _ in range(frames):
        if not cap.grab():
            break
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    if len(t) > 10 and t[-1] > t[0]:
        est = (len(t) - 1) / ((t[-1] - t[0]) / 1000.0)
        # Override only the camera-stream case: metadata about half the real rate (.ts, 15 vs 30 fps).
        # Phone videos with a variable frame rate also differ in their first frames, but there the
        # metadata (average over the whole file) is the better value and is kept.
        if not meta or est > 1.3 * meta:
            return float(est), float(meta)
    return float(meta or 30.0), float(meta)


def video_info(path) -> dict:
    cap = open_video(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps_meta = cap.get(cv2.CAP_PROP_FPS)
    fps, _ = measured_fps(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps_meta and fps != fps_meta:                      # metadata wrong (.ts): rescale the count
        n = int(round(n / fps_meta * fps))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    cap.release()
    return {"path": str(path), "fps": fps, "fps_metadata": fps_meta, "frames": n, "width": W, "height": H,
            "duration_s": n / fps if fps else None, "codec_fourcc": codec,
            "file_bytes": Path(path).stat().st_size}


def target_size(native_wh, target_w, target_h=None, aspect="preserve"):
    """Return (w,h). With aspect='preserve' the height is derived from the native aspect ratio."""
    W, H = native_wh
    if aspect == "preserve" or target_h is None:
        return int(target_w), max(1, int(round(target_w * H / W)))
    return int(target_w), int(target_h)


# --------------------------------------------------------------------------- #
# Signal extraction (streaming)
# --------------------------------------------------------------------------- #
def extract_signal(video, roi: ROI, size=None, interp="area", max_frames=None,
                   store_npy: str | None = None, stabilize=False):
    """Single pass over the video. Returns I(t) (float64, gray levels) and timing breakdown.

    stabilize: False | True (estimate camera motion on the fly, timed as 'stab_s') |
               path of a .npy with pre-computed transforms (estimate_motion). With stabilisation the
               ROI (drawn on frame 0) is mapped into each frame, so it stays on the same road area.

    size: (w,h) processing resolution; None = native.
    store_npy: optional path; if given, the full uint8 grayscale stack (frames,h,w) is saved,
               reproducing the original offline workflow (file size = frames*w*h + 128 B header).
    """
    cap = open_video(video)
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w, h = size if size else (W, H)
    mask = roi.mask(w, h)
    npx = int(mask.sum())
    flag = INTERP[interp]
    vals, stack = [], []
    t_dec = t_res = t_gray = t_roi = t_stab = 0.0
    As = np.load(stabilize) if isinstance(stabilize, (str, Path)) else None
    stab, A_prev = None, None
    k = 0
    while True:
        t0 = time.perf_counter()
        ok, fr = cap.read()
        t1 = time.perf_counter(); t_dec += t1 - t0
        if not ok or (max_frames and k >= max_frames):
            break
        if stabilize is not False and stabilize is not None:
            if As is not None:
                A = As[min(k, len(As) - 1)]
            elif stab is None:
                stab = Stabilizer(fr, roi); A = np.eye(3)
            else:
                A = stab.update(fr)
            if A_prev is None or not np.allclose(A, A_prev, atol=1e-5):
                mask = Stabilizer.roi_in_frame(roi, A).mask(w, h); A_prev = A.copy()
            t1b = time.perf_counter(); t_stab += t1b - t1; t1 = t1b
        if (w, h) != (W, H):
            fr = cv2.resize(fr, (w, h), interpolation=flag)
        t2 = time.perf_counter(); t_res += t2 - t1
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        t3 = time.perf_counter(); t_gray += t3 - t2
        vals.append(float(g[mask].mean()) if mask.any() else np.nan)
        if store_npy:
            stack.append(g)
        t_roi += time.perf_counter() - t3
        k += 1
    cap.release()
    t_io = 0.0
    if store_npy:
        t0 = time.perf_counter()
        np.save(store_npy, np.stack(stack))
        t_io = time.perf_counter() - t0
    timing = {"decode_s": t_dec, "stab_s": t_stab, "resize_s": t_res, "gray_s": t_gray,
              "roi_mean_s": t_roi, "npy_write_s": t_io}
    vals = np.asarray(vals, float); bad = np.isnan(vals)
    if bad.any() and (~bad).any():      # ROI moved completely out of the image: bridge the gap
        vals[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), vals[~bad])
    meta = {"proc_w": w, "proc_h": h, "roi_pixels": npx, "roi_fraction": npx / (w * h), "frames": k,
            "roi_outside_frames": int(bad.sum())}
    return vals, timing, meta


def baseline(I: np.ndarray, fps: float, p: Params) -> np.ndarray:
    if p.baseline == "global_median":
        return np.full_like(I, np.median(I))
    win = max(3, int(round(p.baseline_win_s * fps)) | 1)
    if p.baseline == "moving_median":          # centred, non-causal
        return median_filter(I, size=win, mode="nearest")
    if p.baseline == "causal_median":          # uses only past samples -> online-compatible
        out = np.empty_like(I)
        for i in range(len(I)):
            out[i] = np.median(I[max(0, i - win + 1): i + 1])
        return out
    raise ValueError(p.baseline)


def robust_sigma(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def detect(I: np.ndarray, fps: float, p: Params):
    """Return peak frame indices, D(t) and the effective numeric height used."""
    B = baseline(I, fps, p)
    D = np.abs(I - B)
    h = p.height if p.height_mode == "abs" else p.height * robust_sigma(D)
    kw = dict(height=h, distance=p.distance_frames(fps))
    if p.prominence is not None:
        kw["prominence"] = p.prominence
    peaks, _ = find_peaks(D, **kw)
    return peaks, D, h


def run(video, roi: ROI, p: Params, target_w=None, target_h=None, fps=None, repeats=1,
        warmup=True, measure_memory=False, store_npy=None, max_frames=None):
    """End-to-end run (decode -> count). Returns dict with counts, peaks and timing statistics.

    Timing boundary: everything from opening the file to the list of peaks
    (decode + resize + grayscale + ROI mean + baseline + find_peaks). Disk write of the optional
    .npy is reported separately (npy_write_s)."""
    info = video_info(video)
    fps = fps or info["fps"]
    size = None
    if target_w:
        size = target_size((info["width"], info["height"]), target_w, target_h, p.aspect)
    if warmup:  # discard first (cold-cache) run
        extract_signal(video, roi, size, p.interp, max_frames=min(50, info["frames"] or 50),
                       stabilize=p.stabilize)
    totals, parts = [], []
    peak_mem = None
    for r in range(repeats):
        if measure_memory and r == 0:
            tracemalloc.start()
        t0 = time.perf_counter()
        I, timing, meta = extract_signal(video, roi, size, p.interp, max_frames=max_frames,
                                         store_npy=store_npy if r == 0 else None,
                                         stabilize=p.stabilize)
        t1 = time.perf_counter()
        peaks, D, h_eff = detect(I, fps, p)
        t2 = time.perf_counter()
        if measure_memory and r == 0:
            peak_mem = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        timing["detect_s"] = t2 - t1
        totals.append(t2 - t0 - timing["npy_write_s"])
        parts.append(timing)
    totals = np.asarray(totals)
    q1, q3 = np.percentile(totals, [25, 75])
    return {
        "video": str(video), "fps": fps, **meta, "params": asdict(p), "height_effective": h_eff,
        "distance_frames": p.distance_frames(fps), "count": int(len(peaks)),
        "peaks": peaks.tolist(), "peak_times_s": (peaks / fps).tolist(),
        "time_median_s": float(np.median(totals)), "time_iqr_s": float(q3 - q1),
        "time_mean_s": float(totals.mean()), "time_std_s": float(totals.std(ddof=1)) if len(totals) > 1 else 0.0,
        "time_all_s": totals.tolist(), "repeats": repeats,
        "throughput_fps": float(meta["frames"] / np.median(totals)),
        "timing_breakdown_median": {k: float(np.median([t[k] for t in parts])) for k in parts[0]},
        "peak_python_mem_bytes": peak_mem,
        "signal_I": I, "signal_D": D,
    }


def theoretical_npy_bytes(frames, w, h, dtype_bytes=1, header=128):
    return frames * w * h * dtype_bytes + header
