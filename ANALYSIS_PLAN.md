# LOTO analysis plan (pre-specification)

**Status: FROZEN.** Prepared 9 November 2026; the timestamp of the public commit (or OSF upload) of this file is the freezing date. The values in bold were confirmed by the corresponding author. No test-fold result existed when this plan was frozen: only a 1-epoch technical test on fold test_trt20 had been run, which uses the training and validation images and never opens the test fold. Any later change is a deviation and will be reported in the manuscript.

Every setting below is already stated in the R2 manuscript draft (Sections 2.2, 2.5–2.7) or implemented in the scripts in this folder; this file collects them in one place.

## 1. Design

| Item | Specification |
|---|---|
| Design | Leave-one-treatment-out (LOTO), 3 folds (test treatment 20, 50, 80) |
| Images | 572 images with recorded treatment (tier A: 167 block-resolved; tier B: 405 treatment × species). Tiers C and D excluded. |
| Fold file | `analysis/audit_outputs/loto_fold_assignment.csv` (record its SHA-256 when freezing) |
| Validation set | Tier-A images of blocks 6–8 of the two training treatments (42 per fold). Shares plants with training (checkpoint selection only). The test fold is plant-disjoint from training and validation. |
| Data source | The Roboflow export used in R1 (`Vines_2025.v8-vineo.coco-segmentation`), one unaugmented file per image (pre-flight check P2 in step 0) |
| Annotation corrections | One author-confirmed correction (`annotation_corrections.json`): the duplicate Madame Galen polygon on one Parthenocissus image is left out. No other annotation is changed. |
| Preparation | `step0_build_loto_datasets.py`: R1 `augment_train.py` (3 augmented copies per training image, seed 42; training images only) and R1 `letterbox_resize.py` (640 × 640, grey 114), both unchanged |

## 2. Training

| Item | Specification |
|---|---|
| Models | The seven R1 architectures and training recipes, unchanged (Table 2; `patches/*.diff` lists every code change). One exception: the simplified SOLOv2 baseline applies R1's geometric training augmentations to image and masks together (R1: image only); stated in the manuscript. |
| Epochs and early stopping | As in R1 for all seven models: at most **100** epochs; training stops after **15** consecutive epochs without improvement of the selection score below (in R1 the rule was the same but each model used its own score) |
| Seeds | **Seed 42 only** (21 runs: 7 models × 3 folds). Variability is reported across the three folds. The plant-overlapping R1 models keep their three seeds. |
| Checkpoint selection | After every epoch: pooled 8-class macro pixel Dice on the validation images at 512 × 512 (TP/FP/FN summed over all validation pixels; Dice per class; mean over classes present in ground truth or prediction). Best epoch kept; ties go to the earlier epoch. Selected epochs reported in Table S9. |
| Output → label map | U-Net/DeepLabv3/SegFormer: arg-max. Mask R-CNN, YOLO11-seg, YOLO11-PGSS: instance masks with score ≥ **0.25** painted in ascending score order. Simplified SOLOv2: 16 × 16 class and objectness maps upsampled to 512; arg-max class where objectness > 0.5. |
| Ground truth | COCO polygons filled at 640 × 640 (letterboxed), nearest-neighbour resized to 512 × 512 (as in R1 `run_full_test_evaluation.py`) |

## 3. Endpoints (test folds)

| | Endpoint | Computation |
|---|---|---|
| Primary | Macro mIoU over 8 classes (background + 7 species) | Pooled pixel confusion matrix per test fold at 512 × 512; reported per fold and as mean ± SD (n − 1) across the 3 folds |
| Secondary | Macro precision, recall, F1; mIoU over the 7 species; per-class IoU (Table S3) | Same confusion matrices |
| Secondary | COCO mask AP, AP50, AP75 (instance-capable models) | pycocotools, all detections with score ≥ **0.05** for every model |
| Secondary | Matched comparison with the plant-overlapping models on the 35 tier-A images common to both test sets | Same metrics on the same images |

## 4. Pairwise comparison of architectures (Table S4)

| Item | Specification |
|---|---|
| Per-image quantities | Per-image mIoU (primary) and per-image macro F1, excluding classes absent from both ground truth and prediction |
| Effect | Paired per-image difference (row − column, percentage points), averaged over the three folds |
| Uncertainty | Cluster bootstrap, 10,000 resamples, clusters = treatment × species groups (7 per fold, 21 in total); 95% percentile CI |
| Test | Two-sided bootstrap p-values, Holm-adjusted across the 21 pairs, α = 0.05 |
| Equivalence | "Practically equivalent" only if the 90% bootstrap CI lies entirely within **±2 percentage points** (two one-sided tests, α = 0.05) |
| Software | `cluster_bootstrap_compare.py` |

## 5. Reporting rules

- All seven models and all three folds are reported, whatever the outcome; no fold or model is dropped after results are seen.
- A drop in performance relative to the plant-overlapping split is reported as observed; it is not attributed to leakage alone, because training-set size and test composition also differ (Section 4.1).
- Deviations from this plan are listed in the manuscript (Methods) with the reason.

## 6. Freezing (do this before step 1)

1. Confirm the bold values, delete "DRAFT" from the title, and add the date.
2. Record SHA-256 hashes: `shasum -a 256 ANALYSIS_PLAN.md ../../audit_outputs/loto_fold_assignment.csv`
3. Put a timestamped copy where it cannot be changed silently: commit it to the GitHub repository (the commit time is the timestamp) or upload it to the OSF project.
4. Only then start `run_step1_training.sh`.

## 7. Frozen files (SHA-256)

The data and code below are fixed by this plan. The three folds were built on 9 November 2026 (step0_report.json: all checks passed). Training hardware: RunPod, NVIDIA A40 48 GB; PyTorch 2.8.0 (CUDA 12.8), torchvision 0.23.0, Ultralytics 8.4.208, Transformers 5.22.2.

| File (relative to `analysis/author_scripts/loto_retraining/`) | SHA-256 |
|---|---|
| `../../audit_outputs/loto_fold_assignment.csv` | `03f4cc52bde842db4a1ad52ca659310e8546f2d26fbd32f3ade504c4e3b32c0b` |
| `annotation_corrections.json` | `e552ad7bd7b199f00fd66dd355597a04bc445680354d0983ce91f929ad342e7f` |
| `loto_common.py` | `ae2b12cdbadcfc91f12cfef976d493333f705e67f537e0d193b6fcd4f62b25c9` |
| `make_loto_training_scripts.py` | `55bc562687dddd9502f523c42dd314b5c8ee8d441ee40f9789f09d6636289645` |
| `run_step1_training.sh` | `b3425cbb7ec06c4337c73a3de6a9150d6639f0cefc1aaebe558f3a0896c68967` |
| `step0_build_loto_datasets.py` | `bcb1f632f1939632563f3894f6bac54059bdea7e5ed4e989d059d262cba6ceec` |
| `../unified_pixel_dice.py` | `0d0df0e8f65dbfc00fbcc62b24058156f251c039265932baba41748e0d88419f` |
| `models/DeepLabV3/train_deeplabv3_loto.py` | `3ec0d3758106fe132cdf7c8cf19d54d1d826db0011e32a8c3c50710441d05282` |
| `models/MaskRCNN/train_maskrcnn_loto.py` | `c69118bae5866e3440a01f502834450778a22db8268016833e213173b11d5424` |
| `models/PGSS_YOLO11/pgss_modules.py` | `ac4717f7cc8fc1ba526ba02c5b8eac006a66e287e64460511d0c946e67190c07` |
| `models/PGSS_YOLO11/train_pgss_loto.py` | `9ad899889a741fb9c4a1ba4e0d32253360c871e8d9ebb9cb8f8db147153bb9cc` |
| `models/SegFormer/train_segformer_loto.py` | `a5c478fec0cb3a5e956cb78f00b16c6743d65b53db2eff6f4424effc52308fa0` |
| `models/SoloV2/train_solov2_loto.py` | `042dc540e5512f06f0ea15fa42a7f567104ac2cacb445f9b81f91837ad664450` |
| `models/UNet/train_unet_loto.py` | `0ab6b0d307170997da5f295442bce8a9641ab13d483934af08b541555df3b114` |
| `models/YOLO11seg/train_yolo11_seg_loto.py` | `0d0ff1597a20b89b86f11a8f9b6da9ba3e9dc24aad6ec270896a6f4ef06c2cf5` |
