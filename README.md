# Vine Canopy Segmentation — Code & Results

Code, evaluation pipeline, and result artifacts for a comparison of seven segmentation
architectures on a 7-species ornamental vine canopy dataset (Akebia, Campsis, Hydrangea,
Lonicera, Madame Galen, Parthenocissus, Wisteria).

**Second revision (R2).** The image-level train/val/test split described below is the split of
the first revision. It is not plant-disjoint: test plants were also photographed for training on
other dates, and 36 of its 169 test images have an identical photograph in the training set. It
is kept only as a plant-overlapping reference. The primary, plant-disjoint evaluation
(leave-one-treatment-out, all seven models retrained with one pixel-level Dice criterion) and the
ImageJ comparison with matched scale and region are in [`R2_revision/`](R2_revision/README.md).
Analysis plans: `ANALYSIS_PLAN.md`, `AREA_ANALYSIS_PLAN.md` and `AREA_ANALYSIS_PLAN_AMENDMENT_1.md`.
Data: https://doi.org/10.17605/OSF.IO/YJHCK.

## Where to find each result of the paper

| Paper | Files |
|---|---|
| Tables 3 and 5, Tables S3 and S4, Fig. 3 (unseen plants, LOTO) | [`R2_revision/plant_disjoint_LOTO/`](R2_revision/plant_disjoint_LOTO) |
| Table 6, Fig. 5 (canopy area compared with ImageJ) | [`R2_revision/imagej_area/`](R2_revision/imagej_area) |
| Table 1, Table S5 (dataset, plant-identity tiers) | `split_assignment_final.csv`, `dataset_accounting.csv`, `R2_revision/plant_disjoint_LOTO/recover_identity_from_capture_sequence.py` |
| Table 4 (plant-overlapping split, three seeds) | training scripts below, `run_multiseed_evaluation.py`, `evaluation_multiseed.json` |
| Table 2 (GFLOPs, latency) | `benchmark_flops_latency.py` |
| Fig. 4 (qualitative comparison) | `make_fig4_qualitative.py` |
| Section 3.5 (failure cases) | `find_worst_cases.py`, `failure_analysis_summary.json`, `failure_analysis_worst10.csv` |
| Images and polygon annotations, identity manifest, LOTO folds, ImageJ outputs | OSF, https://doi.org/10.17605/OSF.IO/YJHCK |

The sections below describe the scripts of the first revision, which produced the plant-overlapping
results (Table 4, Fig. 4, Section 3.5). The R2 scripts reuse their training code (as patches) and
their data preparation.

Models: UNet, DeepLabV3, SegFormer, Mask R-CNN, SOLOv2 (simplified), YOLO11-seg,
YOLO11-PGSS (custom GLSA/AWFF/DEConv architecture).

**Note on scope**: this repository contains code and small structured result files only.
Trained model checkpoints, the raw/augmented image datasets, and generated figures are
excluded (see `.gitignore`) since they are large binary artifacts, not reviewable source
code — checkpoints and datasets should be distributed separately (e.g. cloud storage) if
needed for full reproduction.

## Repository contents

### Training scripts (one per model)
| File | Model |
|---|---|
| `UNet/train_unet.py` | UNet |
| `DeepLabV3/train_deeplabv3.py` | DeepLabV3 (torchvision, ResNet50 backbone) |
| `SegFormer/train_segformer.py` | SegFormer (HuggingFace, MiT-b2 backbone) |
| `MaskRCNN/train_maskrcnn.py` | Mask R-CNN (torchvision, ResNet50-FPN) |
| `SoloV2/train_solov2_clean.py` | SimplifiedSOLOv2 (ResNet50 backbone, custom head) |
| `YOLO11seg/train_yolo11_seg.py` | YOLO11l-seg (Ultralytics) |
| `PGSS_YOLO11/train_pgss.py` | YOLO11-PGSS (custom GLSA/AWFF/DEConv modules) |

`PGSS_YOLO11/pgss_modules.py` and `PGSS_YOLO11/yolo11s-pgss-seg.yaml` are included
alongside the PGSS training script because they are hard dependencies of it (the custom
attention/fusion/detail-conv module definitions and the model architecture YAML that
`train_pgss.py` builds from) — without them the training script cannot run.

In these first-revision scripts, UNet, DeepLabV3, SegFormer and Mask R-CNN select their best
checkpoint by pixel-level validation Dice. The two Ultralytics YOLO models use a custom
early-stopping callback (`make_dice_early_stopper` in each training script) that monitors an F1
score computed from the instance-matched mask precision and recall reported by Ultralytics, and
the simplified SOLOv2 uses Dice at its native 16 x 16 resolution; these criteria are not
equivalent to pixel-level Dice. The R2 retraining uses one shared pixel-level Dice criterion for
all seven models (`R2_revision/plant_disjoint_LOTO/unified_pixel_dice.py`).

### Dataset pipeline
| File | Purpose |
|---|---|
| `rebuild_splits.py` | Rebuilds the train/val/test split assignment of the first revision (plant-overlapping, see above) |
| `augment_train.py` | Polygon-aware augmentation, three copies per training image (`__a1/__a2/__a3`); used for the LOTO folds of R2 (the first-revision split was augmented in Roboflow) |
| `letterbox_resize.py` | Letterbox resize/pad pipeline used to prepare images for model input |
| `integrity_check.py` | Validates COCO annotation integrity (orphan annotations, out-of-bounds polygons) across split folders |
| `verify_dataset_loading.py` | Sanity-checks that each split loads correctly end-to-end |

### Evaluation scripts
Per-model evaluate scripts (original, validation-split-oriented methodology):
`UNet/unet_evaluate.py`, `DeepLabV3/deeplabv3_evaluate.py`, `SegFormer/segformer_evaluate.py`,
`MaskRCNN/maskrcnn_evaluate.py`, `YOLO11seg/yolo_evaluate.py`, `PGSS_YOLO11/pgss_evaluate.py`.

Test-set evaluation and analysis scripts (all seven models, test set of the first-revision split):
| File | Purpose |
|---|---|
| `run_full_test_evaluation.py` | Full test-set semantic (macro P/R/F1/mIoU, per-class IoU, confusion matrix) + COCO instance (AP/AP50/AP75/AR) metrics, single seed |
| `run_multiseed_evaluation.py` | Same, aggregated across 3 seeds (mean ± std, per-seed breakdown) |
| `compute_per_image_f1.py` | Per-image macro-F1 per model per seed (the same per-image F1 is used in R2 `step2_evaluate_test_folds.py`) |
| `benchmark_flops_latency.py` | Per-model GFLOPs and batch-1 inference latency, same GPU |
| `make_fig4_qualitative.py` | Qualitative single-image comparison figure generation |
| `find_worst_cases.py` | Worst-10-per-model failure case identification + overlay grid generation for the failure-analysis section |

### Data / metadata
- `split_assignment_final.csv` — the authoritative per-image split/tier/species assignment
  (1,129 retained images: 789 train / 171 val / 169 test) used by every training and
  evaluation script above.

### Result files
| File | Contents |
|---|---|
| `evaluation_multiseed.json` | Full per-seed + aggregated (mean/std) semantic and instance metrics, all 7 models, seeds 42/43/44 |
| `evaluation_multiseed_summary.csv` | Flat one-row-per-model summary of the above |
| `dataset_accounting.csv` | Per-species raw/augmented image counts, tier breakdown, year/backdrop breakdown |
| `failure_analysis_summary.json` | Categorized failure-mode taxonomy from manual inspection of each model's 10 worst test-set predictions (seed 42), including the identified visual cause of the Campsis/Madame Galen confusion |
| `failure_analysis_worst10.csv` | Flat ranking of the 70 worst-case (model, image) pairs by per-image mIoU |

## Reproducing results

1. Place the rebuilt dataset at `rebuilt/final/` (COCO format, for the 5 non-YOLO models)
   and `rebuilt/final_yolo/` (YOLO format, for YOLO11-seg / YOLO11-PGSS), matching the
   paths referenced in `split_assignment_final.csv` and the training scripts.
2. Run each `train_*.py` script to reproduce checkpoints (not included in this repo).
3. Point `run_multiseed_evaluation.py` / `run_full_test_evaluation.py` at the resulting
   checkpoint paths (see `CKPT_ROOT` near the top of each script) and run against
   `rebuilt/final/test` / `rebuilt/final_yolo/test`.
