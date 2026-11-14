# Second revision (R2): plant-disjoint evaluation and matched ImageJ comparison

Code and derived results added for the second revision of "Multi-class segmentation of ornamental vine
canopies" (JAFR-D-26-02492). The analysis plans are in the repository root: `ANALYSIS_PLAN.md` (LOTO
retraining), `AREA_ANALYSIS_PLAN.md` (commit 33da919) and `AREA_ANALYSIS_PLAN_AMENDMENT_1.md` (commit acec144).
The images are on OSF (https://doi.org/10.17605/OSF.IO/YJHCK).

## plant_disjoint_LOTO/ (Sections 2.2, 2.6, 2.7, 3.1, 3.2; Tables 3, 5, S3, S4; Fig. 3)
| File | Purpose |
|---|---|
| `make_plant_disjoint_folds.py`, `loto_fold_assignment.csv` | three leave-one-treatment-out folds (572 tier A/B images) and the disjointness checks |
| `recover_identity_from_capture_sequence.py` | evidence that tier C test images show training plants (Table S5) |
| `step0_build_loto_datasets.py` | builds the fold datasets from the Roboflow export; offline polygon-aware augmentation |
| `make_loto_training_scripts.py`, `patches/`, `models/`, `run_step1_training.sh` | training scripts of the seven architectures with the common checkpoint criterion |
| `unified_pixel_dice.py` | the shared validation macro Dice (pixel counts pooled over all validation images, 512 × 512) |
| `step2_evaluate_test_folds.py` | semantic metrics and COCO instance metrics (score threshold 0.05) on the test folds |
| `step3_stats_and_figures.py`, `cluster_bootstrap_compare.py` | fold means/SD, cluster bootstrap with Holm adjustment and equivalence margin |
| `annotation_corrections.json`, `post_evaluation_corrections.json` | the two label corrections applied to the ground truth |
| `step4_predict_imagej_images.py` | LOTO-model predictions for the ImageJ-measured images |
| `figure_confusion_matrices.py`, `loto_figure3.py` | Fig. 3 |
| `results/` | per-image metrics, fold summaries and the tables (file names follow the working numbering) |

## imagej_area/ (Sections 2.3, 2.5, 3.4; Table 6; Fig. 5)
| File | Purpose |
|---|---|
| `make_measurement_images.py` | dataset images restored to square pixels and upright orientation, random order |
| `Measure_Vine_Canopy.ijm`, `measurement/PROTOCOL.md` | ImageJ macro and protocol; saves scale line, rectangle and mask of every image |
| `read_imagej_outputs.py` | reads the saved outputs and checks every mask against the recorded values |
| `area_analysis_matched.py` | ImageJ, ground-truth polygon and model areas in the same rectangle and scale; all agreement statistics, including 95% limits of agreement, Lin's CCC, Deming regression, backdrop strata and the sensitivity analyses |
| `area_table6_fig5_correlation.py` | Table 6 and Fig. 5 |
| `AREA_ANALYSIS_DEVIATIONS.md` | decisions not specified in the plan, recorded before any agreement statistic was computed |
| `measurement/` | measurement manifest, ImageJ measurements (169 images and the 20 repeats), scale review |
| `results/` | per-image areas, `agreement.csv` (all statistics, primary and as-planned analyses), `sensitivity_single_fold_models.csv`, repeatability, `run_info.json` (input hashes) |

## audit/
`split_assignment_final.csv` (image-to-identity manifest with tiers), `duplicate_photos.csv` (167 IMG_/TRT_
pairs of identical photographs), `dataset_accounting.csv`.

Paths to the data are set at the top of each script and must be adapted to the local copy.
