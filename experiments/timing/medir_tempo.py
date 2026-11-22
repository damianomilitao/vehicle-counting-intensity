"""
medir_tempo.py - tempo de processamento e memoria de cada metodo (artigo IEEE Access, secao de custo).

Mede, em trechos de 30 s de dois videos (camera da Olaria a noite, 1920x1080 ~30 fps; Video B, 848x478
60 fps, com compensacao de movimento):
  - so a decodificacao;
  - metodo proposto (mediana global) em 6 resolucoes; mediana causal na resolucao maxima;
  - estimativa da compensacao de movimento (Video B);
  - diferenca de quadros, MOG2 e YOLOv8n+ByteTrack (640 px, ~10 quadros/s).
Cada configuracao roda num processo separado (para medir o pico de memoria dela), com 1 rodada de
aquecimento descartada e R repeticoes. OpenCV limitado a 1 thread (exceto YOLO, que usa o PyTorch padrao).
Resultado: tempo_resultados.json e tempo_resultados.csv nesta pasta.
"""
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, '..')   # experiments/ folder (videos/, rois/, stab/)
sys.path.insert(0, os.path.join(ROOT, '..'))
VID = {
    "olaria_noite": ("olaria_noite_20261116_1846.ts", "olaria.json", None, [1280, 640, 320, 160, 80, 40]),
    "B": ("B.mp4", "B.json", "B.npy", [848, 640, 320, 160, 80, 40]),
}
CLIP_S = float(os.environ.get("MEDIR_CLIP_S", "30"))


def preparar_videos():
    for name, (fn, _, _, _) in VID.items():
        if not os.path.exists(os.path.join(ROOT, "videos", fn)):
            sys.exit("video not found: " + os.path.join(ROOT, "videos", fn))


def hw():
    info = {"os": platform.platform(), "python": platform.python_version(), "machine": platform.machine()}
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        info["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        info["cpu"] = platform.processor()
    try:
        import psutil
        info["cores_physical"] = psutil.cpu_count(logical=False)
        info["cores_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        f = psutil.cpu_freq()
        info["cpu_freq_max_mhz"] = f.max if f else None
    except Exception:
        pass
    for mod in ("cv2", "numpy", "scipy", "torch", "ultralytics"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def peak_mem_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round((getattr(mi, "peak_wset", None) or mi.rss) / 2**20, 1)
    except Exception:
        return None


def rodar_uma(cfg):
    """Executa uma configuracao (dentro de um subprocesso) e devolve os tempos."""
    import tracemalloc
    import numpy as np
    import cv2
    import tvic
    import baselines as BL
    if cfg["method"] != "yolo":
        cv2.setNumThreads(1)
    fn, roi_f, stab_f, _ = VID[cfg["video"]]
    video = os.path.join(ROOT, "videos", fn)
    roi = tvic.ROI.load(os.path.join(ROOT, "rois", roi_f))
    stab = os.path.join(ROOT, "stab", stab_f) if stab_f else None
    info = tvic.video_info(video)
    fps = info["fps"]; mf = int(round(CLIP_S * fps))
    size = tvic.target_size((info["width"], info["height"]), cfg["w"]) if cfg.get("w") else None
    m = cfg["method"]

    def job():
        if m == "decode":
            cap = tvic.open_video(video); k = 0
            while k < mf and cap.read()[0]:
                k += 1
            cap.release(); return k
        if m in ("proposed_global", "proposed_causal"):
            I, _, meta = tvic.extract_signal(video, roi, size, "area", max_frames=mf, stabilize=stab or False)
            p = tvic.Params(height=6.0, min_sep_s=1.0,
                            baseline="global_median" if m == "proposed_global" else "causal_median",
                            baseline_win_s=30.0)
            tvic.detect(I, fps, p); return meta["frames"]
        if m == "stab_estimate":
            tvic.estimate_motion(video, roi, out=None, max_frames=mf); return mf
        if m == "framediff":
            r = BL.framediff(video, roi, target_w=cfg["w"], height=4.0, min_sep_s=1.0, stab=stab, max_frames=mf)
            return r["frames"]
        if m == "mog2":
            r = BL.mog2(video, roi, target_w=cfg["w"], on=0.15, min_sep_s=1.0, stab=stab, max_frames=mf)
            return r["frames"]
        if m == "yolo":
            tracks, sz, k = BL.yolo_tracks(video, roi, fps, os.environ.get("YOLO_WEIGHTS", "yolov8n.pt"), 640, 10.0, stab, mf)
            BL.yolo_events(tracks, roi, sz)
            return int(np.ceil(k / max(1, round(fps / 10.0))))
        raise ValueError(m)

    if m == "yolo":   # aquecimento curto (carregar o modelo)
        mf_full = mf; mf = int(2 * fps); job(); mf = mf_full
    else:
        job()                                   # aquecimento descartado
    times, frames = [], 0
    tracemalloc.start()
    for r in range(cfg["R"]):
        t0 = time.perf_counter(); frames = job(); times.append(time.perf_counter() - t0)
    tm_peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    times = np.array(times)
    return {**cfg, "fps_video": fps, "clip_s": CLIP_S, "frames_processed": int(frames),
            "proc_w": size[0] if size else info["width"], "proc_h": size[1] if size else info["height"],
            "time_all_s": times.round(4).tolist(), "time_median_s": float(np.median(times)),
            "time_iqr_s": float(np.subtract(*np.percentile(times, [75, 25]))),
            "time_mean_s": float(times.mean()), "time_std_s": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
            "throughput_fps": float(frames / np.median(times)),
            "realtime_factor": float(CLIP_S / np.median(times)),
            "tracemalloc_peak_mib": round(tm_peak, 1), "peak_working_set_mib": peak_mem_mb()}


def configs():
    out = []
    for v, (_, _, stab, widths) in VID.items():
        out.append({"video": v, "method": "decode", "w": None, "R": 5})
        for w in widths:
            out.append({"video": v, "method": "proposed_global", "w": w, "R": 5})
        out.append({"video": v, "method": "proposed_causal", "w": widths[0], "R": 3})
        if stab:
            out.append({"video": v, "method": "stab_estimate", "w": None, "R": 3})
        out.append({"video": v, "method": "framediff", "w": widths[0], "R": 5})
        out.append({"video": v, "method": "mog2", "w": widths[0], "R": 3})
        out.append({"video": v, "method": "yolo", "w": 640, "R": 3})
    return out


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        print("RESULT " + json.dumps(rodar_uma(json.loads(sys.argv[2]))), flush=True)
        sys.exit(0)
    preparar_videos()
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    outname = "tempo_resultados_yolo.json" if only == "yolo" else "tempo_resultados.json"
    res = {"hardware": hw(), "results": [], "errors": []}
    print(json.dumps(res["hardware"], indent=1), flush=True)
    cf = [c for c in configs() if only is None or c["method"] == only]; t_all = time.time()
    for i, c in enumerate(cf, 1):
        print(f"[{i}/{len(cf)}] {c['video']} {c['method']} w={c['w']} ...", end=" ", flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", json.dumps(c)],
                           capture_output=True, text=True, cwd=HERE)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            r = json.loads(line[-1][7:]); res["results"].append(r)
            print(f"{r['time_median_s']:.2f} s, {r['throughput_fps']:.0f} quadros/s, {r['peak_working_set_mib']} MiB", flush=True)
        else:
            res["errors"].append({"cfg": c, "stderr": p.stderr[-3000:]}); print("ERRO", flush=True)
        json.dump(res, open(os.path.join(HERE, outname), "w"), indent=1)
    keys = ["video", "method", "proc_w", "proc_h", "frames_processed", "time_median_s", "time_iqr_s",
            "time_mean_s", "time_std_s", "throughput_fps", "realtime_factor", "tracemalloc_peak_mib",
            "peak_working_set_mib", "R"]
    with open(os.path.join(HERE, outname.replace(".json", ".csv")), "w", newline="") as f:
        wr = csv.DictWriter(f, keys, extrasaction="ignore"); wr.writeheader(); wr.writerows(res["results"])
    print(f"\nPronto em {(time.time()-t_all)/60:.1f} min. Erros: {len(res['errors'])}")
