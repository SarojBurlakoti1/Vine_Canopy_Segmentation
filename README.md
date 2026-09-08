# Vine Canopy Segmentation — Code & Results

Code, evaluation pipeline, and result artifacts for a comparison of seven segmentation
architectures on a 7-species ornamental vine canopy dataset (Akebia, Campsis, Hydrangea,
Lonicera, Madame Galen, Parthenocissus, Wisteria), evaluated on a held-out test set never
used in training or validation.

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

All five plain-PyTorch models (UNet, DeepLabV3, SegFormer, Mask R-CNN, SOLOv2) select
their best checkpoint by validation Dice/mIoU. The two Ultralytics YOLO models use a
custom Dice-monitored early-stopping callback (see the `make_dice_early_stopper` function
in each training script) so checkpoint selection is consistent across all seven models,
rather than relying on Ultralytics' own composite fitness metric.

### Dataset pipeline
| File | Purpose |
|---|---|
| `rebuild_splits.py` | Rebuilds the leakage-corrected train/val/test split assignment |
| `augment_train.py` | Generates the 3x on-disk training augmentation (`__a1/__a2/__a3` variants) |
| `letterbox_resize.py` | Letterbox resize/pad pipeline used to prepare images for model input |
| `integrity_check.py` | Validates COCO annotation integrity (orphan annotations, out-of-bounds polygons) across split folders |
| `verify_dataset_loading.py` | Sanity-checks that each split loads correctly end-to-end |

### Evaluation scripts
Per-model evaluate scripts (original, validation-split-oriented methodology):
`UNet/unet_evaluate.py`, `DeepLabV3/deeplabv3_evaluate.py`, `SegFormer/segformer_evaluate.py`,
`MaskRCNN/maskrcnn_evaluate.py`, `SoloV2/eval_test.py`, `YOLO11seg/yolo_evaluate.py`,
`PGSS_YOLO11/pgss_evaluate.py`.

Test-set evaluation and analysis scripts (all seven models, held-out test set only):
| File | Purpose |
|---|---|
| `run_full_test_evaluation.py` | Full test-set semantic (macro P/R/F1/mIoU, per-class IoU, confusion matrix) + COCO instance (AP/AP50/AP75/AR) metrics, single seed |
| `run_multiseed_evaluation.py` | Same, aggregated across 3 seeds (mean ± std, per-seed breakdown) |
| `compute_per_image_f1.py` | Per-image macro-F1 per model per seed (input to the pairwise significance testing) |
| `clustered_pairwise_stats.py` | Plant-clustered bootstrap pairwise model comparison (Holm-corrected p-values, effect sizes, 21 pairs x 3 seeds) |
| `benchmark_flops_latency.py` | Per-model GFLOPs and batch-1 inference latency, same GPU |
| `make_fig3_confusion_matrices.py` | Normalized per-species confusion matrix figure generation |
| `make_fig4_qualitative.py` | Qualitative single-image comparison figure generation |
| `find_worst_cases.py` | Worst-10-per-model failure case identification + overlay grid generation for the failure-analysis section |

### Data / metadata
- `split_assignment_final.csv` — the authoritative per-image split/tier/species assignment
  (1,129 retained images: 789 train / 171 val / 169 test) used by every training and
  evaluation script above.

  **Note**: `all_images_grouped.csv` was requested for this repo but does not exist
  anywhere in the environment this work was done in — only `split_assignment_final.csv`
  was available, and all pipeline/evaluation code here is built on that file. If
  `all_images_grouped.csv` exists elsewhere (e.g. an earlier working copy), it was not
  accessible when this repository was assembled and is not included.

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
