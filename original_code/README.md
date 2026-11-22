# Original script

`original_pipeline.py` is the processing script of the original submission, as listed in its Supporting Information (Listing 1). It is kept here unchanged for reference.

In the analyses of the original submission, frames were resized with the default bilinear interpolation of `cv2.resize`, the ROI was drawn once by the operator, and the minimum peak distance was 30 frames for both videos; the height threshold was 4 gray levels for Video A and 10.5 gray levels for Video B (the listing shows the default value 10). Resolutions were changed by editing `target_resolution`.

All results of the revised article were obtained with `tvic.py` and the scripts in `experiments/`, not with this script (see Supplementary Section S1 of the article).
