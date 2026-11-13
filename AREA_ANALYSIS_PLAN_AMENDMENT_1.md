# Amendment 1 to the canopy-area analysis plan (AREA_ANALYSIS_PLAN.md, commit 33da919)

**Status: FINAL. Approved by the author on 12 November 2026 (Pacific time; 13 November 07:28 UTC). Frozen by public commit before the analysis is run.**

**When and why.** All 169 images were measured (11–12 Nov 2026). On 12 Nov 2026 a duplicate-photo search (`analysis/audit_outputs/DUPLICATE_PHOTOS_FINDINGS.md`) showed that 41 of the 134 `IMG_` images are identical photos to `TRT_` images used in LOTO, and that 5 photos occur twice among the 169. The 20 repeat images were then measured (12 Nov 2026) and checked with `read_imagej_outputs.py` (20 rows, 0 problems). **At the time of this amendment no ground-truth or model area had been computed for any of the 169 images, the repeat measurements had not been compared with the first measurements, and no agreement or repeatability statistic had been calculated.** The changes below were therefore made without knowledge of any result.

## Changes

1. **Unit of analysis: unique photos (164).** For each of the 5 photos measured twice, the `TRT_` version is used: it is a LOTO test image, and its upright frame is the higher-resolution source. Its `IMG_` twin is not analysed as a separate image. Pairs: IMG_1124 = TRT_50_HA, IMG_1532 = TRT_20_LH, IMG_1549 = TRT_50_LH, IMG_1141 = TRT_20_PQ, IMG_1316 = TRT_80_PQ.
2. **Model assignment by photo, not by file name.**
   - **Unseen in LOTO (71 photos):** the 35 `TRT_` images, plus the 36 remaining `IMG_` images whose identical `TRT_` copy is in LOTO. Each uses the LOTO model of the fold in which the photo (or its copy) is a test image, so that model never saw the photo.
   - **Not in LOTO (93 photos):** the `IMG_` images with no copy in LOTO. They use the mean of the three fold models, as pre-specified. Overlap with LOTO training plants (other photos of the same plant) cannot be excluded.
3. **Strata:** the two groups above replace "tier A (35)" and "tier C/D (134)". Overall results use all 164 photos.
4. **Pre-specified version kept:** the analysis exactly as frozen in commit 33da919 (169 images, the original assignment, all 20 repeat images) is reported in the supplement as a sensitivity analysis.
5. **Additional repeat (reported as unplanned):** the analyst's two measurements of each of the 5 duplicate photos are reported alongside the planned 20-image repeat, as same-photo repeats under different file names and geometry.
6. **Correction of a factual statement in the plan:** the 17 `IMG_xxxx-2-` files are 2024 photos (black backdrop), not July–September 2023. Two of them (IMG_1553-2- = TRT_50_PQ, IMG_1562-2- = TRT_80_HA) are among the 169; both are in group "unseen in LOTO".
7. **Repeat set: one photo occurs twice.** R02 (TRT_20_LH) and R03 (IMG_1532) are the same photo. Following change 1, R02 is used and R03 is left out, so the repeat analysis uses 19 unique photos.
8. **Repeat images without a scale.** Whether a complete 4 ft reference is visible is a property of the photo, so the status in `scale_review.csv` also applies to the photo's repeat. Four repeat images are NO SCALE photos: R01, R04, R07 and R10 (`repeat_scale_review.csv`; for R04 the analyst reported after measuring that the note "No scale" had been forgotten). Under the plan's exclusion rule they are left out of the px/cm and area comparisons and are counted. The rectangle IoU does not depend on the scale and uses every repeat image. The repeat comparisons therefore use 15 photos for px/cm and area and 19 for IoU (16 and 20 in the pre-specified version).
9. **Timing of the repeat (deviation).** The plan says "on a different day". The repeat was measured on 12 Nov 2026, 22:29–22:52 by the computer clock (Pacific time), 4.8–24.4 h after each image's first measurement (median 8.9 h). By that clock, 18 of the 20 first measurements were made on the same date. The manuscript reports the time gap in hours, not "a different day". The repeat is not redone.

Everything else in AREA_ANALYSIS_PLAN.md is unchanged: the areas, the statistics, the decomposition, the NO SCALE rule (20 of the 169 images; see `scale_review.csv`) and the repeatability statistics.

## Measurement files at the time of this amendment (SHA-256)

These files are fixed. Any later change to them is a deviation and will be reported.

| File (in `ImageJ_remeasurement/v2_dataset_images/`) | SHA-256 |
|---|---|
| `images_to_measure_results/measurements.csv` (169 rows) | `4e0a10f4bb0c2978914a1c3f4b45c7a689336c5af3a6c8ae33b4a25b1d790372` |
| `repeat_20_results/measurements.csv` (20 rows) | `448d13a2d83a178687b6d752831d44e0a2c30781ce96d67c93b4daf6b6ad0c5c` |
| `scale_review.csv` (23 rows) | `3ab1a4cf27853ccdd950c8299d8327f2eb6ff0ac67c3b4778b884b5016028f90` |
| `repeat_scale_review.csv` (5 rows) | `06d0f0d10f6e7cd8eceb0bd93f86b4fdf62f8e1db7054594bb48b7a22a93ae69` |
