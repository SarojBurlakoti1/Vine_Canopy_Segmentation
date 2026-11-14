#!/usr/bin/env python3
"""
figure_confusion_matrices.py
============================
Fig. 3 and Fig. S1 of the R2 manuscript, drawn in the layout of Fig. 3 of the first revision
(one normalized species confusion matrix per architecture, blue scale, two-decimal values,
panels SegFormer / DeepLabv3 / Mask R-CNN, YOLO11-seg / YOLO11-PGSS / simplified SOLOv2, U-Net),
at print size (190 mm wide, 300 dpi) and with one colour scale (0-1) shared by all panels.

Fig. 3  unseen plants: LOTO test folds pooled (572 images; ground truth with the two
        author-confirmed label corrections), from
        LOTO_eval_results/loto_stats_corrected/fig3_normalized_species_cm_all_folds.csv
Fig. S1 plant-overlapping split (169 test images), first-revision models, from the pixel
        confusion counts saved by the first-revision evaluation scripts
        (Run/<model>/<model>_results_v6/*_confusion_matrix_counts.npy; class 0 = background).
        These counts reproduce every value of the first-revision Fig. 3.

Each row is one ground-truth species: the fraction of that species' pixels predicted as each
species. Pixels predicted as background are not shown, so a row sums to less than 1.

Output: Fig3.png/.tif/.pdf and FigS1.png/.tif/.pdf in JAFR_Final_Revision/figures.
Usage:  /opt/anaconda3/bin/python3 -I figure_confusion_matrices.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REV = os.path.expanduser("~/Downloads/Revision")
OUT = os.path.join(REV, "JAFR_Final_Revision", "figures")
LOTO_CSV = os.path.join(REV, "LOTO_eval_results", "loto_stats_corrected", "fig3_normalized_species_cm_all_folds.csv")
R1 = {  # display name -> first-revision pixel confusion counts (8 x 8, background first)
    "SegFormer": "SegFormer/SegFormer_results_v6/SegFormer_confusion_matrix_counts.npy",
    "DeepLabv3": "DeepLabV3/DeepLabV3_results_v6/DeepLabV3_confusion_matrix_counts.npy",
    "Mask R-CNN": "MaskRCNN/MaskRCNN_results_v6/MaskRCNN_confusion_matrix_counts.npy",
    "YOLO11-seg": "YOLO11seg/YOLO11seg_results_v6/YOLO11seg_confusion_matrix_counts.npy",
    "YOLO11-PGSS": "PGSS_YOLO11/PGSS_YOLO11_results_v6/PGSS_confusion_matrix_counts.npy",
    "Simplified SOLOv2 baseline": "SoloV2/SOLOv2_results_v6/SOLOv2_confusion_matrix_counts.npy",
    "U-Net": "UNet/UNet_results_v6/UNet_confusion_matrix_counts.npy",
}
ORDER = list(R1)  # panel order of the first-revision figure
CSV_NAME = {"Simplified SOLOv2 baseline": "Simplified SOLOv2"}
SP = ["Akebia", "Campsis", "Hydrangea", "Lonicera", "Madame Galen", "Parthenocissus", "Wisteria"]
INK, INK2 = "#0b0b0b", "#3d3d3a"
CMAP = plt.get_cmap("Blues")
MM = 1 / 25.4

plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.linewidth": 0.4,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


def loto_matrices():
    df = pd.read_csv(LOTO_CSV)
    out = {}
    for m in ORDER:
        sub = df[df.model == CSV_NAME.get(m, m)].set_index("ground_truth").loc[SP, SP].values
        assert sub.shape == (7, 7) and (sub.sum(1) <= 1.0001).all(), m
        out[m] = sub
    return out


def r1_matrices():
    out = {}
    for m in ORDER:
        c = np.load(os.path.join(REV, "Run", R1[m])).astype(float)
        assert c.shape == (8, 8), m
        out[m] = (c / c.sum(1, keepdims=True))[1:, 1:]          # row-normalized incl. background
    # spot checks against the values printed in the first-revision Fig. 3
    assert round(out["Mask R-CNN"][1, 6], 2) == 0.24 and round(out["U-Net"][1, 4], 2) == 0.23
    assert round(out["Simplified SOLOv2 baseline"][2, 2], 2) == 0.16 and round(out["SegFormer"][0, 0], 2) == 0.92
    return out


def panel(ax, M, title, ylabels):
    ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest", extent=(-0.5, 6.5, 6.5, -0.5))
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(6.5, -0.5)
    for i in range(7):
        for j in range(7):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if v >= 0.6 else INK)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(SP, rotation=45, ha="right", rotation_mode="anchor", fontsize=6.2)
    ax.set_yticklabels(SP if ylabels else [], fontsize=6.2)
    ax.tick_params(length=1.5, width=0.4, pad=1.5)
    if not ylabels:
        ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(0.5, 6, 1), minor=True); ax.set_yticks(np.arange(0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.5); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=4)
    ax.set_xlabel("Predicted class", fontsize=6.8, labelpad=2)
    if ylabels:
        ax.set_ylabel("Ground-truth class", fontsize=6.8, labelpad=2)


def figure(mats, name):
    fig = plt.figure(figsize=(190 * MM, 216 * MM))
    gs = GridSpec(3, 3, figure=fig, left=0.115, right=0.995, top=0.975, bottom=0.088, wspace=0.10, hspace=0.62)
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)]
    for k, (m, (r, c)) in enumerate(zip(ORDER, slots)):
        ax = fig.add_subplot(gs[r, c])
        panel(ax, mats[m], f"({'abcdefg'[k]}) {m}", ylabels=(c == 0 or (r, c) == (2, 1)))
    # the slots beside panel (g) are left empty (author's request); every cell carries its value,
    # and the caption states the 0-1 scale and that background predictions are not shown
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (("png", {}), ("tif", {"pil_kwargs": {"compression": "tiff_lzw"}}), ("pdf", {})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, **kw)
    plt.close(fig)
    print("written", name)


if __name__ == "__main__":
    figure(loto_matrices(), "Fig3")
    figure(r1_matrices(), "FigS1")
