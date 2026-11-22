"""
estabilizar.py - pre-compute camera-motion transforms of a video (once per video).

    python estabilizar.py --video videos/videoB.mp4 --roi rois/videoB_roi.json --out stab/videoB_stab.npy

The .npy (frames x 3 x 3, normalised coordinates, current frame -> frame 0) is then used by
run_experiments.py ("stab" field of the video in config.json) and by annotate.py (--stab), so the
detector, the baselines and the annotators see exactly the same geometry. Prints the estimated
displacement of the ROI centre over the video (pixels of the native frame).
"""
import argparse
from pathlib import Path
import numpy as np
from tvic import ROI, estimate_motion, video_info

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--roi", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
roi = ROI.load(a.roi); info = video_info(a.video)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
As, st = estimate_motion(a.video, roi, out=a.out)
c = np.r_[np.mean(np.asarray(roi.polygon), 0), 1.0]
q = np.array([np.linalg.inv(A) @ c for A in As])[:, :2] - c[:2]
dx, dy = q[:, 0] * info["width"], q[:, 1] * info["height"]
print(f"{a.out}: {st['frames']} quadros, falhas de rastreamento: {st['failed_frames']}, "
      f"inliers (mediana): {st['median_inliers']:.0f}")
print(f"deslocamento do centro da ROI: max |dx| = {np.abs(dx).max():.1f} px, max |dy| = {np.abs(dy).max():.1f} px "
      f"(mediana {np.median(np.hypot(dx, dy)):.1f} px)")
out = 0
for A in As:
    P = np.array([np.linalg.inv(A) @ np.r_[v, 1.0] for v in roi.polygon])[:, :2]
    out += bool((P < 0).any() or (P > 1).any())
print(f"quadros em que a ROI sai (parcial ou totalmente) da imagem: {out} de {len(As)} ({100*out/len(As):.1f} %)")
if out:
    print("ATENCAO: escolha uma ROI numa parte da pista que fique visivel no video inteiro.")
