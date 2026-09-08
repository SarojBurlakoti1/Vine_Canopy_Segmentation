"""
Fig 3: normalized confusion matrices for all 7 models (seed 42, primary), full test set.

Matches this project's own established per-model confusion-matrix styling exactly
(see plot_normalized_confusion_matrix in train_solov2_clean.py / equivalents in the
other evaluate scripts): species-only submatrix (background row/col dropped), row
sums taken from the FULL 8-class matrix (including the background column) so pixels
a model calls "background" remain in the denominator as real errors, seaborn Blues
heatmap, 2-decimal annotations, "Ground Truth Class" / "Predicted Class" axis labels.

Uses the seed-42 confusion matrices already computed in evaluation_results.json.
Saves: fig3_confusion_matrices_combined.png (7-panel grid)
       fig3_confusion_matrix_<Model>.png (7 individual files)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RUN = "/root/vine-project/Run"
CLASS_NAMES = ["background", "Akebia", "Campsis", "Hydrangea", "Lonicera",
               "Madame Galen", "Parthenocissus", "Wisteria"]
SPECIES = CLASS_NAMES[1:]
SPECIES_IDX = list(range(1, 8))

results = json.load(open(f"{RUN}/evaluation_results.json"))
model_keys = ["UNet", "DeepLabV3", "SegFormer", "MaskRCNN", "SoloV2", "YOLO11-seg", "YOLO11-PGSS"]


def normalized_species_cm(cm):
    cm = np.array(cm, dtype=float)
    cm_species = cm[np.ix_(SPECIES_IDX, SPECIES_IDX)]
    row_sums = cm[SPECIES_IDX, :].sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm_species / row_sums


def plot_one(ax, cm_norm, title, cbar=True):
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=SPECIES, yticklabels=SPECIES, cbar=cbar, ax=ax,
                vmin=0, vmax=1, square=True, annot_kws={"size": 7})
    ax.set_ylabel("Ground Truth Class", fontsize=8)
    ax.set_xlabel("Predicted Class", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)


# ---- individual figures ----
for key in model_keys:
    cm = results[key]["confusion_matrix_rows_gt_cols_pred"]
    cm_norm = normalized_species_cm(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_one(ax, cm_norm, f"Normalized Confusion Matrix ({results[key]['model']}, seed 42, test set)")
    plt.tight_layout()
    safe_name = results[key]["model"].replace(" ", "_").replace("-", "")
    out = f"{RUN}/fig3_confusion_matrix_{safe_name}.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)

# ---- combined 2x4 grid (7 panels + 1 blank) ----
fig, axes = plt.subplots(2, 4, figsize=(26, 13))
axes = axes.flatten()
for i, key in enumerate(model_keys):
    cm = results[key]["confusion_matrix_rows_gt_cols_pred"]
    cm_norm = normalized_species_cm(cm)
    plot_one(axes[i], cm_norm, results[key]["model"], cbar=False)
axes[-1].axis("off")
fig.suptitle("Fig. 3 — Normalized confusion matrices, all seven models (seed 42, held-out test set, n=169)",
             fontsize=14, y=1.0)
plt.tight_layout()
out = f"{RUN}/fig3_confusion_matrices_combined.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
