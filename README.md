# Temporal image-intensity vehicle counting: code, data and results

Code, annotations, extracted signals and results of the article

F. S. Rodrigues, J. S. Dominguez, R. Fabbri, J. T. de Assis and D. S. Militão,
"Temporal Image-Intensity Analysis for Low-Cost Vehicle Counting: Event-Level Evaluation and Sensitivity to Spatial Resolution," *IEEE Access* (under review).

The method follows the mean gray level of a polygonal region of interest (ROI) covering one lane,
computes its absolute deviation from a median baseline and counts vehicles as peaks with a threshold
in gray levels and a minimum separation in seconds. It is compared with frame differencing, an MOG2
virtual loop and YOLOv8n + ByteTrack on the same ROIs and against an adjudicated two-annotator ground truth.

## Layout

- `tvic.py`: the method: ROI handling, streaming signal extraction, baselines (global, centred moving and causal median), peak detection, camera-motion estimation.
- `baselines.py`: frame differencing, MOG2 virtual loop, YOLOv8n + ByteTrack (Ultralytics).
- `evaluation.py`: event lists and one-to-one matching.
- `annotate.py`, `estabilizar.py`, `*.bat`: annotation tool (ROI drawing, event marking) and camera-motion estimation, with Windows shortcuts.
- `experiments/`: every script and data file used for the results of the article (below).

### `experiments/`

- `common.py`, `core.py`: recordings, evaluation windows, calibration/test split, matching, detectors, calibration grids.
- `resample.py`, `fullextract.py`: ROI signals at the 99 + 3 processing sizes in one decoding pass (exact OpenCV `INTER_AREA`/`INTER_LINEAR` weights), frame differencing, Monte Carlo and sensitivity ROIs.
- `signals_streaming.py`: full-resolution signals with the streaming implementation, with and without camera-motion compensation.
- `mog2_signals.py`, `yolo_cal.py`, `yolo_run.py`: MOG2 occupancy and YOLOv8n + ByteTrack tracks (calibration and test segments).
- `analysis_main.py`: per-recording calibration, test results of all methods (tolerances 1.0, 1.5 and 2.0 s), bootstrap CIs.
- `analysis_extra.py`: transfer from Video A, rain/night transfer, camera motion on/off, resolution sweep, sensitivity, Monte Carlo, error list, detector vs. each annotator, occupancy.
- `agreement.py`, `dataset_table.py`, `paired_bootstrap.py`: inter-annotator agreement, data-set table and reference-count uncertainty, paired differences between methods.
- `fail_montage.py`, `fail_examples.py`: frames and signals used to classify the errors; example figure (pixelated).
- `dusk.py`: dusk recording of the Olaria camera (Supplementary Table S3c): signal extraction with decoded timestamps and the global, centred moving and causal median baselines.
- `make_tables.py`, `make_figures.py`: all tables (`tables/`, `si_tables/`) and figures (`figures/`).
- `timing/`: timing protocol (`medir_tempo.py`, Windows shortcuts).
- `rois/`: ROI polygons (normalized coordinates).
- `gt/`: event lists of each annotator (`*_A.csv`: F. S. Rodrigues; `*_B.csv`: second annotator), consensus lists and `adjudication_decisions.csv`.
- `stab/`: camera-motion transforms (current frame to anchor frame, normalized coordinates).
- `full/`, `sig/`, `bsig/`, `yolo/`, `dusk/`: extracted signals, MOG2 occupancy, YOLO tracks and the signals of the dusk recording.
- `results/`: all result files; `fail/errors_classified.csv`: every error with its cause.
- `videos/`: place for the recordings (see `videos/README.txt`).

## Recordings

- `A` (Video A): smartphone, site 1, day (transfer test only).
- `B` (Video B): smartphone, site 2, day, 60-frames/s file, evaluated 0–125 s.
- `olaria_dia` (Video C): Olaria camera, sunny day.
- `olaria_tarde_chuva` (Video D): Olaria camera, rain.
- `olaria_noite` (Video E): Olaria camera, night.
- `raul_tarde_chuva` (Video F): Raul Sertã camera, rain.
- `raul_noite` (Video G): Raul Sertã camera, night.
- `S1` (Video H): smartphone, site 1, rain, through window glass.
- `S2` (Video I): smartphone, site 1, rain.
- `olaria_transicao`: Olaria camera at dusk, 20 November 2026; two intervals annotated, before sunset (17:56–18:05) and after it (18:10–18:19) (supplementary test, not in the pooled results).

Calibration segment: first 40 % of each evaluation interval; test segment: remaining 60 %.

## Reproducing the results

Requirements: Python ≥ 3.10 with `opencv-python`, `numpy`, `scipy`, `pandas`, `matplotlib`
(`pip install -r requirements.txt`); `ultralytics` and `psutil` only for the YOLO baseline and the timing.

Without the videos (from the released signals, tracks and annotations):

```
python experiments/dataset_table.py
python experiments/agreement.py
python experiments/analysis_main.py
python experiments/analysis_extra.py
python experiments/paired_bootstrap.py
python experiments/dusk.py evaluate
python experiments/make_tables.py
python experiments/make_figures.py
```

With the videos in `experiments/videos/` (signal extraction, baselines and timing):

```
python experiments/fullextract.py
python experiments/signals_streaming.py
python experiments/mog2_signals.py
python experiments/yolo_cal.py B olaria_dia olaria_tarde_chuva olaria_noite raul_tarde_chuva raul_noite S1 S2
python experiments/yolo_run.py  B olaria_dia olaria_tarde_chuva olaria_noite raul_tarde_chuva raul_noite S1 S2
python experiments/timing/medir_tempo.py
```

`fail_examples.py` (error examples) also needs the videos.

## Data availability

The smartphone recordings (A, B, H, I) are deposited in the Zenodo archive of this repository with restricted access, because they show licence plates and passers-by; access is granted upon request for research use. The surveillance-camera
recordings (C–G and the dusk recording) were made from the public live streams of the Programa Nova Friburgo Cidade Inteligente
(https://nfci.pmnf.rj.gov.br/camerasaovivo.html). Their use is authorized by the Central de Monitoramento
(Prefeitura Municipal de Nova Friburgo, Ofício SESOP No. 000078/2026) exclusively for academic research,
and their redistribution to third parties without prior authorization is not allowed (LGPD, Law
13.709/2018). They are therefore not included; requests can be sent to the corresponding author
(dmilitao@iprj.uerj.br) and will be forwarded to the Central de Monitoramento. Their ROIs, annotations,
extracted signals, detections and results are included, so every number of the article can be recomputed
without them. Image credit: Central de Monitoramento – Programa Nova Friburgo Cidade Inteligente,
Prefeitura Municipal de Nova Friburgo.
