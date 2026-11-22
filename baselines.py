"""
baselines.py - same-data baselines.

B1  Frame differencing virtual zone : s(t) = mean |g_t - g_{t-1}| over the ROI pixels, peaks via find_peaks
                                       (ROI mapped into each frame with the camera-motion transforms, as for
                                       the proposed detector; frames are not resampled a second time)
B2  MOG2 virtual loop detector       : foreground fraction of the ROI (cv2.BackgroundSubtractorMOG2 with
                                       shadow detection; shadow pixels excluded), hysteresis on/off with
                                       off = on/3, minimum on-time 0.1 s, minimum separation between events.
                                       With camera-motion compensation the frames are warped to the anchor
                                       frame, because the background model is per pixel.
B3  Lightweight detector + tracker   : YOLOv8n (Ultralytics, COCO classes car/motorcycle/bus/truck) + ByteTrack
                                       on the CPU, input width 640 px, sampled at about 10 frames/s. One event
                                       per track whose box covers at least `cover` of the ROI area in some frame
                                       (event time = frame of maximum coverage), counting only boxes with
                                       confidence >= conf.

All baselines use the same ROI, ground truth and timing boundary (open file -> event list).
"""
from __future__ import annotations

import time
import tracemalloc

import cv2
import numpy as np
from scipy.signal import find_peaks

from tvic import ROI, INTERP, video_info, target_size, Stabilizer, open_video


def _reader(video, size, interp, stab=None, warp=False, max_frames=None, step=1):
    """Yield (k, frame_bgr_resized, A) ; A = 3x3 transform current->anchor (normalised) or None."""
    As = np.load(stab) if stab is not None and not isinstance(stab, np.ndarray) else stab
    cap = open_video(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w, h = size if size else (W, H)
    k = 0
    while True:
        if max_frames and k >= max_frames:
            break
        if step > 1 and k % step:
            if not cap.grab():
                break
            k += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        if (w, h) != (W, H):
            fr = cv2.resize(fr, (w, h), interpolation=INTERP[interp])
        A = As[min(k, len(As) - 1)] if As is not None else None
        if warp and A is not None:
            fr = Stabilizer.warp_to_anchor(fr, A)
        yield k, fr, A
        k += 1
    cap.release()


def _wrap(fn, video, roi, target_w, aspect, interp, measure_memory, **kw):
    info = video_info(video)
    size = target_size((info["width"], info["height"]), target_w, None, aspect) if target_w else \
        (info["width"], info["height"])
    if measure_memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    times, extra = fn(video, roi, size, info["fps"], interp, **kw)
    dt = time.perf_counter() - t0
    mem = tracemalloc.get_traced_memory()[1] if measure_memory else None
    if measure_memory:
        tracemalloc.stop()
    nfr = extra.pop("frames", None) or info["frames"]
    return {"count": len(times), "event_times_s": list(map(float, times)), "time_s": dt,
            "frames": nfr, "throughput_fps": nfr / dt if dt else None, "peak_python_mem_bytes": mem,
            **extra}


# --- B1 -------------------------------------------------------------------- #
def framediff_signal(video, roi, size, interp="area", stab=None, max_frames=None):
    w, h = size
    mask0 = roi.mask(w, h); mask = mask0; A_prev = None; prev = None; s = []
    for k, fr, A in _reader(video, size, interp, stab, warp=False, max_frames=max_frames):
        if A is not None and (A_prev is None or not np.allclose(A, A_prev, atol=1e-5)):
            mask = Stabilizer.roi_in_frame(roi, A).mask(w, h); A_prev = A.copy()
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        s.append(0.0 if prev is None or not mask.any() else float(np.abs(g - prev)[mask].mean()))
        prev = g
    return np.asarray(s)


def _framediff(video, roi, size, fps, interp, height=2.0, min_sep_s=0.4, stab=None, max_frames=None):
    s = framediff_signal(video, roi, size, interp, stab, max_frames)
    pk, _ = find_peaks(s, height=height, distance=max(1, int(round(min_sep_s * fps))))
    return pk / fps, {"method": "frame_difference", "height": height, "min_sep_s": min_sep_s,
                      "frames": len(s)}


def framediff(video, roi, target_w=None, aspect="preserve", interp="area", measure_memory=False, **kw):
    return _wrap(_framediff, video, roi, target_w, aspect, interp, measure_memory, **kw)


# --- B2 -------------------------------------------------------------------- #
def mog2_occupancy(video, roi, size, fps, interp="area", stab=None, max_frames=None, history_s=20.0,
                   var_threshold=16):
    m = roi.mask(*size)
    bg = cv2.createBackgroundSubtractorMOG2(history=int(history_s * fps), varThreshold=var_threshold,
                                            detectShadows=True)
    occ = []
    for k, fr, A in _reader(video, size, interp, stab, warp=True, max_frames=max_frames):
        fg = bg.apply(fr)
        occ.append(float((fg[m] == 255).mean()))   # 127 = shadow, excluded
    return np.asarray(occ)


def mog2_events(occ, fps, on=0.15, min_sep_s=0.4, min_on_s=0.10):
    off = on / 3.0
    events, state, start, last = [], False, 0, -1e9
    for i, o in enumerate(occ):
        if not state and o >= on:
            state, start = True, i
        elif state and o <= off:
            state = False
            if (i - start) / fps >= min_on_s and (start / fps - last) >= min_sep_s:
                events.append(start / fps); last = start / fps
    return np.asarray(events)


def _mog2(video, roi, size, fps, interp, on=0.15, min_sep_s=0.4, stab=None, max_frames=None, **kw):
    occ = mog2_occupancy(video, roi, size, fps, interp, stab, max_frames, **kw)
    return mog2_events(occ, fps, on, min_sep_s), {"method": "MOG2_virtual_loop", "on": on,
                                                  "off": on / 3.0, "frames": len(occ)}


def mog2(video, roi, target_w=None, aspect="preserve", interp="area", measure_memory=False, **kw):
    return _wrap(_mog2, video, roi, target_w, aspect, interp, measure_memory, **kw)


# --- B3 -------------------------------------------------------------------- #
def yolo_tracks(video, roi, fps, weights="yolov8n.pt", width=640, rate_hz=10.0, stab=None,
                max_frames=None, t_start=0.0, t_end=None, min_conf=0.1, classes=(2, 3, 5, 7)):
    """Run detector+tracker; return list of (t, track_id, conf, x1, y1, x2, y2) in the 'width' frame
    (frames warped to the anchor frame when stab is given) and the frame size."""
    from ultralytics import YOLO  # optional dependency: pip install ultralytics
    info = video_info(video)
    W, H = info["width"], info["height"]
    w, h = width, int(round(width * H / W))
    model = YOLO(weights)
    step = max(1, int(round(fps / rate_hz)))
    cap = open_video(video)
    As = np.load(stab) if stab is not None and not isinstance(stab, np.ndarray) else stab
    tracks, k = [], 0
    while True:
        if max_frames and k >= max_frames:
            break
        ok = cap.grab()
        if not ok:
            break
        t = k / fps
        if t_end is not None and t >= t_end:
            break
        if t >= t_start and k % step == 0:
            ok, fr = cap.retrieve()
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            if As is not None:
                fr = Stabilizer.warp_to_anchor(fr, As[min(k, len(As) - 1)])
            r = model.track(fr, persist=True, conf=min_conf, classes=list(classes), device="cpu",
                            imgsz=640, verbose=False, tracker="bytetrack.yaml")[0]
            if r.boxes.id is not None:
                for (x1, y1, x2, y2), tid, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy(),
                                                    r.boxes.conf.cpu().numpy()):
                    tracks.append((round(t, 4), int(tid), float(c), float(x1), float(y1), float(x2), float(y2)))
        k += 1
    cap.release()
    return tracks, (w, h), k


def yolo_events(tracks, roi, size, conf=0.25, cover=0.3):
    """One event per track: time of maximum ROI coverage, if that coverage >= cover (boxes with conf>=conf)."""
    w, h = size
    m = roi.mask(w, h).astype(np.int64)
    A = m.sum()
    cs = m.cumsum(0).cumsum(1)

    def frac(x1, y1, x2, y2):
        x1, y1 = max(0, int(x1)), max(0, int(y1)); x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2 <= x1 or y2 <= y1 or A == 0:
            return 0.0
        s = cs[y2, x2] - (cs[y1 - 1, x2] if y1 > 0 else 0) - (cs[y2, x1 - 1] if x1 > 0 else 0) + \
            (cs[y1 - 1, x1 - 1] if x1 > 0 and y1 > 0 else 0)
        return s / A
    best = {}
    for t, tid, c, x1, y1, x2, y2 in tracks:
        if c < conf:
            continue
        f = frac(x1, y1, x2, y2)
        if f >= cover and f > best.get(tid, (0.0, 0.0))[0]:
            best[tid] = (f, t)
    return np.array(sorted(v[1] for v in best.values()))


def _yolo(video, roi, size, fps, interp, weights="yolov8n.pt", conf=0.25, cover=0.3, rate_hz=10.0,
          stab=None, max_frames=None):
    tracks, sz, k = yolo_tracks(video, roi, fps, weights, 640, rate_hz, stab, max_frames)
    return yolo_events(tracks, roi, sz, conf, cover), {"method": f"YOLO({weights})+ByteTrack",
                                                       "conf": conf, "cover": cover, "rate_hz": rate_hz,
                                                       "frames": k}


def yolo(video, roi, target_w=None, aspect="preserve", interp="area", measure_memory=False, **kw):
    return _wrap(_yolo, video, roi, target_w, aspect, interp, measure_memory, **kw)
