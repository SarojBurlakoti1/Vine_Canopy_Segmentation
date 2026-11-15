# Second revision (R2): plant-disjoint evaluation and matched ImageJ comparison

Code and derived results added for the second revision of "Multi-class segmentation of ornamental vine
canopies" (JAFR-D-26-02492). The analysis plans are in the repository root: `ANALYSIS_PLAN.md` (LOTO
retraining), `AREA_ANALYSIS_PLAN.md` (commit 33da919) and `AREA_ANALYSIS_PLAN_AMENDMENT_1.md` (commit acec144).
The images and annotations, the image-to-identity manifest, the LOTO fold assignments and the saved ImageJ
outputs are on OSF (https://doi.org/10.17605/OSF.IO/YJHCK).

## plant_disjoint_LOTO/ (Sections 2.2, 2.6, 2.7, 3.1, 3.2; Tables 3, 5, S3, S4; Fig. 3)
| File | Purpose |
|---|---|
| `make_plant_disjoint_folds.py`, `loto_fold_assignment.csv` | three leave-one-treatment-out folds (572 tier A/B images) and the disjointness checks |
| `recover_identity_from_capture_sequence.py` | evidence that tier C test images show training plants (Table S5) |
| `step0_build_loto_datasets.py` | builds the fold datasets from the Roboflow export; offline polygon-aware augmentation |
| `make_loto_training_scripts.py`, `patches/`, `models/`, `run_step1_training.sh` | training scripts of the seven architectures with the common checkpoint criterion (the first-revision training scripts in the repository root plus the changes in `patches/`) |
| `loto_common.py` | code shared by the seven LOTO training scripts: run settings, and ground truth and predictors computed as in the first-revision `run_full_test_evaluation.py` |
| `unified_pixel_dice.py` | the shared validation macro Dice (pixel counts pooled over all validation images, 512 × 512) |
| `step2_evaluate_test_folds.py` | semantic metrics and COCO instance metrics (score threshold 0.05) on the test folds |
| `step3_stats_and_figures.py`, `cluster_bootstrap_compare.py` | fold means/SD, cluster bootstrap with Holm adjustment and equivalence margin |
| `annotation_corrections.json`, `post_evaluation_corrections.json` | the two label corrections applied to the ground truth |
| `step4_predict_imagej_images.py` | LOTO-model predictions for the ImageJ-measured images |
| `figure_confusion_matrices.py` | Fig. 3 (the script also draws the first-revision matrices, which are not in the final paper) |
| `results/evaluation/` | per-image metrics, confusion matrices and fold summaries, semantic and COCO metrics (Tables 3 and 5) |
| `results/stats/` | `table3_*` = Table 3, `tableS3_*` = Table S3, `tableS4_*` = Table S4, `fig3_normalized_species_cm_all_folds.csv` = values of Fig. 3, `tableS9_selected_epochs.csv` = selected epoch of every model and fold (not a table of the paper) |
| `results/step0_report.json` | image counts and checks of the fold datasets |

## imagej_area/ (Sections 2.3, 2.5, 3.4; Table 6; Fig. 5)
| File | Purpose |
|---|---|
| `make_measurement_images.py` | dataset images restored to square pixels and upright orientation, random order |
| `Measure_Vine_Canopy.ijm`, `measurement/PROTOCOL.md` | ImageJ macro and protocol; saves scale line, rectangle and mask of every image |
| `read_imagej_outputs.py` | reads the saved outputs and checks every mask against the recorded values |
| `../area_agreement_matched.py` | agreement statistics named in the analysis plan (`agreement()`); imported by `area_analysis_matched.py`; `python area_agreement_matched.py --selftest` checks it on synthetic data |
| `area_analysis_matched.py` | ImageJ, ground-truth polygon and model areas in the same rectangle and scale; all agreement statistics, including 95% limits of agreement, Lin's CCC, Deming regression, backdrop strata and the sensitivity analyses |
| `area_table6_fig5_correlation.py` | Table 6 and Fig. 5 |
| `AREA_ANALYSIS_DEVIATIONS.md` | decisions not specified in the plan, recorded before any agreement statistic was computed |
| `measurement/` | measurement manifest, ImageJ measurements (169 images and the 20 repeats), scale review |
| `results/` | per-image areas, `agreement.csv` (all statistics, primary and as-planned analyses), `sensitivity_single_fold_models.csv`, repeatability, `run_info.json` (input hashes) |

## audit/
`duplicate_photos.csv`: the 167 IMG_/TRT_ pairs of identical photographs. The image-to-identity manifest with
tiers (`split_assignment_final.csv`) and `dataset_accounting.csv` are in the repository root.

## Software
LOTO models: Python 3.12.3, PyTorch 2.8.0 (CUDA 12.8), torchvision 0.23.0, Ultralytics 8.4.208 and Transformers
5.22.2 on an NVIDIA A40 GPU (48 GB). The scripts also use numpy, pandas, opencv-python, Pillow, pycocotools,
scikit-learn, matplotlib, seaborn and tqdm.

Paths to the data are set at the top of each script and must be adapted to the local copy.
