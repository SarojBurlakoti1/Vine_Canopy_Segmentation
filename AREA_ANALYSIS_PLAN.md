# Canopy-area comparison: analysis plan (addendum to ANALYSIS_PLAN.md)

**Status: written 10–11 November 2026, before any of the 169 images was measured.** The frozen LOTO plan (commit ecf84fd) does not cover the canopy-area comparison; this addendum specifies it. It is frozen by a public GitHub commit before the first of the 169 images is measured. Any later change is a deviation and will be reported.

Purpose: answer reviewer comment 3 (R2). "Recalculate the comparison using matched per-image scales and image regions before interpreting the bias in Table 6 and Figure 5."

## 1. Images and measurement

| Item | Specification |
|---|---|
| Images | The 169 test images of the plant-overlapping split (the R1 ImageJ set) |
| What is measured | The 640 × 640 dataset image of each photo (the same pixels the annotators and models used), with square pixels and an upright plant restored by `make_measurement_images.py`. The 134 `IMG_` images are resized from 640 × 640 to 853 × 640 (bicubic) and rotated 90° clockwise, giving 640 × 853. The 35 `TRT_` images are letterboxed exports and are used unchanged. Geometry was checked on 10 original photos: r = 0.985–0.996 with the upright original, aspect 0.7503 vs 0.7500. |
| Order | Random, fixed by seed 20261110 (file prefix 001–169) |
| Procedure | `Measure_Vine_Canopy.ijm` v1.1, the R1 method:<br>1. a 4 ft (121.92 cm) post-centre-to-post-centre line at the height of the middle of the plant, which sets the per-image scale;<br>2. one rectangle per image containing the whole plant; the image is cropped to this rectangle, so steps 3 and 4 see only the region (R1 also measured within a crop);<br>3. a manual colour threshold (HSB or RGB) set per image;<br>4. removal of non-target pixels by hand;<br>5. Fill Holes;<br>6. Analyze Particles with a minimum of 3 px. This is equivalent to R1's 100 px on photos about 6.3 times wider.<br>The scale line, the rectangle and the final mask are saved for every image. |
| Analyst | The corresponding author, a single analyst. The analyst does not see the ground-truth or model areas of these images before all measurements are finished and checked. |
| Practice | Before the 169, the macro is tried once on two validation-set images that are not among the 169 (`practice/P1.png`, an `IMG_` image; `P2.png`, a `TRT_` image). This only tests the macro; the practice results are not analysed. |
| Repeat | 20 images drawn at random (seed 20261111) are measured again on a different day, under neutral names (`repeat_20/R01–R20`), after all 169 are done |
| Checks | `read_imagej_outputs.py` verifies every row: mask size and binarity, no canopy outside the rectangle, recomputed pixel count and area, saved line and rectangle. Only rows that fail a check, or that the analyst flagged during measuring, may be re-measured. Each re-measurement is logged with its reason in `remeasure_log.csv`. |
| Exclusions | Images where no 4 ft reference can be drawn (note "NO SCALE"), and images whose ImageJ area is 0. They are counted and reported, not replaced. |

## 2. Areas compared (all in the measurement frame, the same rectangle, the same px/cm)

- **ImageJ area (I):** canopy pixels of the saved final mask / (px per cm)².
- **Ground-truth polygon area (G):** the union of all annotated polygons of the image (any species), rasterised on the 640 × 640 dataset grid from `Run/rebuilt/test_640/_annotations.coco.json`. For the 35 `TRT_` images, this file holds the export polygons already transformed into the letterboxed frame (x · 0.3125 + 80, y · 0.3125). Checked on 11 November 2026: for all 169 images, the polygon count and area match the Roboflow export used by LOTO. The result is mapped to the measurement frame with the same resize (nearest neighbour) and rotation as the image. Only pixels inside the rectangle are counted. Species labels do not affect G, and neither author-confirmed label correction (`annotation_corrections.json`, `post_evaluation_corrections.json`) concerns any of the 169 images.
- **Model area (M):** non-background pixels (any species) of the 512 × 512 predicted label map. Predictions were made on the Roboflow export images, which each predictor resized (stretched) to 512 × 512. The maps are therefore brought into the measurement frame by nearest-neighbour resampling, by image kind:
  - **`IMG_` images** (export 640 × 640, stretched): 512 → 640 × 640, then the same resize to 853 × 640 and 90° rotation as the image.
  - **`TRT_` images** (export 1536 × 2048, upright): 512 → 480 × 640, then padded with 80 px of background on each side, giving the 640 × 640 letterboxed frame. This is the geometry recorded by `letterbox_resize.py` (scale 0.3125, left pad 80 px), confirmed on 11 November 2026 by recreating every `TRT_` measurement image from its export image (r ≥ 0.993).

  Pixels are counted inside the rectangle. Predictions come from `step4_predict_imagej_images.py` (LOTO checkpoints, seed 42).
  - **35 tier-A images:** the model of the fold in which the image is a test image. These are unseen plants: 16 in fold trt20, 11 in trt50 and 8 in trt80.
  - **134 tier-C/D images:** these were not used by LOTO. The primary value is the mean of the three fold models' areas; each fold separately is a sensitivity analysis. All 134 belong to the dated `IMG_` series (July–September 2023 sessions). LOTO used no `IMG_` image: its 572 images are all `TRT_` or `Trt-` images. The result therefore describes transfer to other acquisition sessions, and because plant identity is unknown for these images, overlap with LOTO training plants cannot be excluded.

## 3. Analyses (each architecture; strata: all 169, tier A (35), tier C/D (134); also by backdrop)

1. **Reference definition:** G versus I (model-free; all 169 images).
2. **Segmentation error:** M versus G.
3. **End to end:** M versus I.

Agreement statistics, as implemented in `area_agreement_matched.py`:
- mean difference (bias) with 95% limits of agreement (cm²);
- geometric-mean ratio with 95% limits on the log scale (primary, because areas span more than an order of magnitude);
- MAE and MAPE;
- Lin's CCC with 95% CI;
- Pearson r;
- Deming regression (error-variance ratio 1).

95% CIs of the mean log ratios use a percentile bootstrap over images (B = 10,000, seed 42).

**Decomposition.** For each image, log(M/I) = log(G/I) + log(M/G), exactly. The share of the end-to-end discrepancy due to the reference definition is mean log(G/I) / mean log(M/I). It is reported only when the 95% CI of mean log(M/I) excludes 0; otherwise only the two components are reported.

**ImageJ repeatability (repeat_20):**
- per-image differences between the two measurements in px per cm (%) and in area (log ratio, Bland–Altman);
- the IoU of the two rectangles.

These are reported as the measurement uncertainty of the ImageJ reference (same analyst, different day).

## 4. Reporting

- Table 7 and Fig. 5 (Bland–Altman on the log scale) report the matched comparison.
- The R1 comparison (fixed constant 1,596.507 px = 121.92 cm against per-image-calibrated ImageJ on full-size photos with a fixed crop) moves to Table S5, marked superseded and not interpreted.
- The R1 ImageJ areas themselves are not reused, because their per-image scale and crop were not saved.

## 5. Files fixed at the time of freezing (SHA-256)

Practice run (11 November 2026): the analyst measured `practice/P1.png` and `P2.png` with macro v1.1, and both rows passed every check of `read_imagej_outputs.py`. None of the 169 images had been measured when this plan was frozen (no `images_to_measure_results/` folder existed).

| File | SHA-256 |
|---|---|
| `Measure_Vine_Canopy.ijm` (v1.1) | `5a266bf4937eaf2c9f5fa02deb30af6f24a774d3e0255d1b7ddc67c8a5c9c3b6` |
| `make_measurement_images.py` | `1740f19a3ecbac037d65302cab0a82e64a7db9dbc43a5bf1622447043742831d` |
| `read_imagej_outputs.py` | `a8d94eb86311035c1729f775d8752a67b32c3d3a81bd24abb5eca7c662da7881` |
| `measurement_manifest.csv` (lists the SHA-256 of each of the 169 images; all 169 verified on 11 November 2026) | `34dc5ea9313a5afc91b34ed5ff87bfec0bdc8471caae08a46596c9ce14786dcb` |
| `repeat_key.csv` | `91497a455fc8aab00ea11d6af63119041be98d6876a14d257d53fcaf9c32fb14` |
