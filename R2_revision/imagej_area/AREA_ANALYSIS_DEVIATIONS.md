# Canopy-area analysis: deviations from the plan (AREA_ANALYSIS_PLAN.md 33da919 + Amendment 1 acec144)

Written 2026-11-13 07:42 UTC. **At this time no agreement statistic (bias, ratio, CCC, decomposition) had been computed or seen.** What was known: the geometry checks, the backdrop classification, and how often model areas are zero (below).

## D1. Model area of zero
**Found:** on some photos a LOTO model predicts no canopy inside the rectangle (model area M = 0), so log(M/I) and log(M/G) are undefined. The plan does not say how to handle this. In the primary set (144 photos with a scale), the counts are: U-Net 4, DeepLabv3 0, SegFormer 0, Mask R-CNN 2, simplified SOLOv2 8, YOLO11-seg 1, YOLO11-PGSS 8. Most are white-backdrop photos; all LOTO training images have a black backdrop.
**Rule:**
- Statistics in cm² (bias, 95% limits of agreement, MAE, MAPE, CCC, Pearson r, Deming): all photos, with M = 0.
- Statistics on the log scale (geometric-mean ratio, its 95% limits, the bootstrap CI, the decomposition): photos with M > 0 only. For the decomposition, log(G/I) is averaged over the same photos as log(M/I), so the identity log(M/I) = log(G/I) + log(M/G) holds within each model's set.
- The number of photos with M = 0 is reported for each model and stratum.

## D2. Backdrop stratum (operational definition; the plan says only "also by backdrop")
The date-based backdrop labels in the data files are known to be wrong for some sessions, so the backdrop is read from each image: the median grey value inside the rectangle, outside the ground-truth polygons (dilated by 15 px) and outside the ImageJ mask. The values are bimodal with an empty interval of 125–153, and the 12 images between 106 and 169 were checked by eye. Threshold 140: 88 white, 81 black.
