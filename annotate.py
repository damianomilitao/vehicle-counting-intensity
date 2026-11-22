"""
annotate.py - manual event annotation and ROI definition (OpenCV GUI).

1) Define / save the ROI (polygon, normalised coordinates):
    python annotate.py roi  --video videoB.mp4 --out rois/videoB_roi.json
   Left click = add vertex, right click = undo, ENTER = save, ESC = abort.

2) Annotate vehicle passages (independently by each annotator, without seeing detector output):
    python annotate.py events --video videoB.mp4 --roi rois/videoB_roi.json \
                              --annotator A --out gt/videoB_A.csv
   Mark the frame in which the vehicle body is centred over the ROI (maximum ROI coverage);
   this is the instant that corresponds to the intensity peak (protocol, see README).
   Keys:  v = car   m = motorcycle   t = bus/truck   o = other (bicycle, pedestrian...)
          space = pause/play   a / d = 1 frame back / forward   A / D = 1 s back / forward
          + / - = playback speed   u = undo last mark   q = save and quit
   Optional --start/--end (seconds) restrict the annotated window; --stab uses the camera-motion
   transforms of estabilizar.py (display aligned to frame 0, same geometry as the detector). Marks are autosaved after
   every key press; re-running the same command continues from the last mark.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import OrderedDict
import faulthandler, sys, time

import sys
import cv2
import numpy as np

from tvic import ROI, Stabilizer, measured_fps, open_video, safe_path


def define_roi(video, out, name):
    cap = open_video(video); ok, fr = cap.read(); cap.release()
    if not ok:
        raise IOError(video)
    H, W = fr.shape[:2]
    pts = []

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL); cv2.setMouseCallback("ROI", cb)
    while True:
        im = fr.copy()
        if pts:
            cv2.polylines(im, [np.array(pts, np.int32)], len(pts) > 2, (0, 0, 255), 2)
        cv2.imshow("ROI", im)
        k = cv2.waitKey(20) & 0xFF
        if k == 13 and len(pts) >= 3:
            break
        if k == 27:
            return
    cv2.destroyAllWindows()
    roi = ROI([[x / W, y / H] for x, y in pts], name=name)
    roi.save(out, native_wh=(W, H))
    m = roi.mask(W, H)
    print(f"saved {out}: {len(pts)} vertices, {m.sum()} px ({100*m.mean():.2f}% of frame)")
    cv2.imwrite(str(Path(out).with_suffix(".png")), (m * 255).astype(np.uint8))
    ov = fr.copy(); ov[m > 0] = (0.5 * ov[m > 0] + [0, 0, 127]).astype(np.uint8)
    cv2.polylines(ov, [np.array(pts, np.int32)], True, (0, 0, 255), 3)
    cv2.imwrite(str(Path(out).with_name(Path(out).stem + "_sobre_video.jpg")), ov)


def _save(marks, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_s", "frame", "class", "annotator"])
        w.writeheader(); w.writerows(sorted(marks, key=lambda r: r["frame"]))


def annotate(video, roi_path, annotator, out, start_s=0.0, end_s=None, stab=None):
    roi = ROI.load(roi_path) if roi_path else None
    cap = open_video(video)
    if not cap.isOpened():
        print(f"\nERRO: nao consegui abrir o video:\n  {video}\n"
              "Confira se arrastou SO o video (e teclou Enter) antes de arrastar a ROI.")
        sys.exit(3)
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 30.0; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps, _ = measured_fps(video)                        # .ts metadata report half the real rate
    if fps != fps_meta:
        n = int(round(n / fps_meta * fps))
        print(f"fps corrigido: {fps_meta:g} (metadados) -> {fps:.2f} (medido)")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = min(1.0, 1280.0 / W)                       # display size (keeps memory/GPU low)
    dW, dH = int(W * sc), int(H * sc)
    poly = np.array([[x * dW, y * dH] for x, y in roi.polygon], np.int32) if roi else None
    As = np.load(stab) if stab else None             # same transforms as the detector (estabilizar.py)
    if As is not None:
        print(f"estabilizacao: {stab}")
    f0 = int(round(start_s * fps))
    f1 = min(n - 1, int(round(end_s * fps))) if end_s else n - 1
    marks = []
    if Path(out).exists():   # continue a previous session (autosaved)
        with open(out) as f:
            marks = [dict(r, frame=int(r["frame"]), t_s=float(r["t_s"])) for r in csv.DictReader(f)]
        print(f"continuando: {len(marks)} marcas ja existentes em {out}")
    i = max([f0] + [m["frame"] for m in marks]); play, speed = False, 1.0
    cls_keys = {ord("v"): "car", ord("m"): "motorcycle", ord("t"): "bus_truck", ord("o"): "other"}
    cv2.namedWindow("annotate", cv2.WINDOW_NORMAL); cv2.resizeWindow("annotate", dW, dH)
    buf, pos = OrderedDict(), -1                    # cache of the last ~10 s of frames; pos = last frame decoded

    def get(k):
        """Frame k = the k-th frame of a SEQUENTIAL decode, exactly as the detector counts frames.
        No seeking (cap.set is approximate for .ts streams and variable-frame-rate phone videos):
        forward jumps skip frames with grab(); a backward jump beyond the cache re-opens the video."""
        nonlocal cap, pos
        if k in buf:
            return buf[k]
        if k <= pos:                                   # behind the cache: read again from the start
            print(f"voltando para {k/fps:.1f} s: relendo o video desde o inicio (alguns segundos)...")
            cap.release(); cap = open_video(video); pos = -1
        keep = int(10 * fps)                           # last ~10 s kept: a/d/Shift+A are instant
        while pos < k:
            j = pos + 1
            if k - j > keep:                           # far ahead: skip without decoding to an image
                if not cap.grab():
                    print(f"aviso: o video acabou antes do quadro {k} ({k/fps:.2f}s)")
                    return None
                pos = j
                continue
            ok, im = cap.read()
            if not ok:
                print(f"aviso: nao consegui ler o quadro {j} ({j/fps:.2f}s)")
                return None
            pos = j
            im = cv2.resize(im, (dW, dH), interpolation=cv2.INTER_AREA) if sc < 1 else im
            if As is not None:
                im = Stabilizer.warp_to_anchor(im, As[min(j, len(As) - 1)])
            buf[j] = im
            while len(buf) > keep + 5:
                buf.popitem(last=False)
        return buf.get(k)

    fr0, bad = None, 0
    while True:
        im = get(i)
        if im is None:                               # unreadable frame: skip it (never quit); many in a row = real end
            bad += 1
            if bad > 2 * fps or i >= f1:
                f1 = max(f0, i - bad); i = f1; play = False; bad = 0
                if fr0 is None:
                    break
            else:
                i += 1
                continue
        else:
            fr0, bad = im, 0
        fr = fr0.copy()
        if poly is not None:
            cv2.polylines(fr, [poly], True, (0, 0, 255), 2)
        state = "FIM DO TRECHO - tecle q" if i >= f1 else ("" if play else "PAUSADO")
        txt = f"{annotator}{' [ESTAB]' if As is not None else ''}  t={i/fps:7.2f}s  ({f0/fps:.0f}-{f1/fps:.0f}s)  marcas={len(marks)}  x{speed:g}  {state}"
        cv2.putText(fr, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if marks and abs(marks[-1]["frame"] - i) < fps:      # flash the last mark
            cv2.putText(fr, f"MARCADO: {marks[-1]['class']}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("annotate", fr)
        k = cv2.waitKey(max(1, int(1000 / (fps * speed))) if play else 50) & 0xFF
        if cv2.getWindowProperty("annotate", cv2.WND_PROP_VISIBLE) < 1:   # window closed with X
            break
        if k == ord("q"):
            break
        elif k == ord(" "):
            play = not play
        elif k == ord("a"):
            i = max(f0, i - 1); play = False
        elif k == ord("d"):
            i = min(f1, i + 1); play = False
        elif k == ord("A"):
            i = max(f0, i - int(fps))
        elif k == ord("D"):
            i = min(f1, i + int(fps))
        elif k == ord("+"):
            speed = min(4, speed * 2)
        elif k == ord("-"):
            speed = max(0.125, speed / 2)
        elif k == ord("u") and marks:
            marks.pop(); _save(marks, out)
        elif k in cls_keys:
            same = [m for m in marks if abs(m["frame"] - i) <= 2]
            if same:                       # key pressed twice on the same frame: change the class only
                same[0]["class"] = cls_keys[k]
            else:
                marks.append({"frame": i, "t_s": round(i / fps, 3), "class": cls_keys[k],
                              "annotator": annotator})
            _save(marks, out)              # autosave after every mark
        if play:
            if i >= f1:
                play = False
            else:
                i += 1
    cap.release(); cv2.destroyAllWindows()
    _save(marks, out)
    print(f"salvo: {len(marks)} marcas em {out}")


if __name__ == "__main__":
    _log = open(Path(__file__).with_name("crash_log.txt"), "a")   # native crashes leave a trace here
    faulthandler.enable(_log)
    cv2.setNumThreads(1)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roi"); r.add_argument("--video", required=True); r.add_argument("--out", required=True)
    r.add_argument("--name", default="roi")
    e = sub.add_parser("events"); e.add_argument("--video", required=True); e.add_argument("--roi")
    e.add_argument("--annotator", required=True); e.add_argument("--out", required=True)
    e.add_argument("--start", type=float, default=0.0, help="inicio do trecho (s)")
    e.add_argument("--end", type=float, default=None, help="fim do trecho (s)")
    e.add_argument("--stab", default=None, help="transformacoes .npy de estabilizar.py")
    a = ap.parse_args()
    if a.cmd == "roi":
        define_roi(a.video, a.out, a.name)
    else:
        annotate(a.video, a.roi, a.annotator, a.out, a.start, a.end, a.stab)
