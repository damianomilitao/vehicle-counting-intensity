"""MOG2 occupancy of the ROI for every frame (input of the MOG2 virtual-loop baseline).
Frames of the smartphone recordings are warped to the anchor frame (camera-motion transforms).
Writes bsig/<name>_mog.npy. Usage: python mog2_signals.py [names...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import V, paths, E, full_w
from tvic import video_info, target_size
import baselines as BL


def job(n):
    vid, roi, st, win, role = paths(n)
    info = video_info(vid)
    fps = info['fps']
    size = target_size((info['width'], info['height']), full_w(n))
    mf = int(win[1] * fps) + 60 if win else None
    occ = BL.mog2_occupancy(vid, roi, size, fps, 'area', stab=st or None, max_frames=mf)
    os.makedirs(E + 'bsig', exist_ok=True)
    np.save(E + f'bsig/{n}_mog.npy', occ)
    return n, len(occ)


if __name__ == '__main__':
    for n in (sys.argv[1:] or list(V)):
        print(job(n), flush=True)
