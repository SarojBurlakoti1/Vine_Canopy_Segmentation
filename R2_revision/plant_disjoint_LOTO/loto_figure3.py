#!/usr/bin/env python3
"""
loto_figure3.py
===============
Fig. 3 of the R2 manuscript at print size: species confusion matrices of the seven architectures on
unseen plants (LOTO test folds pooled, 572 images; ground truth with the two author-confirmed label
corrections). Re-draws step3_stats_and_figures.py's matrices (fig3_normalized_species_cm_all_folds.csv)
at 190 mm width, 300 dpi.

Each row is one ground-truth species; values are the percentage of that species' pixels predicted as
each species (pooled pixel counts). The remainder of each row (100 - row sum) was predicted as
background. Values that round to 0% are left blank.

Output: Fig3.png, Fig3.tif (300 dpi) and Fig3.pdf in JAFR_Final_Revision/figures.
Usage: /opt/anaconda3/bin/python3 -I loto_figure3.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REV = os.path.expanduser("~/Downloads/Revision")
CSV = os.path.join(REV, "LOTO_eval_results", "loto_stats_corrected", "fig3_normalized_species_cm_all_folds.csv")
OUT = os.path.join(REV, "JAFR_Final_Revision", "figures")
MODELS = ["U-Net", "DeepLabv3", "SegFormer", "Mask R-CNN", "Simplified SOLOv2", "YOLO11-seg", "YOLO11-PGSS"]
SP = ["Akebia", "Campsis", "Hydrangea", "Lonicera", "Madame Galen", "Parthenocissus", "Wisteria"]
AB = ["Ak", "Ca", "Hy", "Lo", "MG", "Pa", "Wi"]
INK, INK2 = "#0b0b0b", "#52514e"
# sequential: one hue (reference blue ramp), light -> dark
RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue", RAMP)
MM = 1 / 25.4

plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.linewidth": 0.5, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


def main():
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(2, 4, figsize=(190 * MM, 104 * MM))
    axes = axes.ravel()
    for k, m in enumerate(MODELS):
        ax = axes[k]
        sub = df[df.model == m].set_index("ground_truth").loc[SP, SP].values * 100
        assert sub.shape == (7, 7) and (sub.sum(axis=1) <= 100.01).all()
        ax.imshow(sub, cmap=CMAP, vmin=0, vmax=100)
        for i in range(7):
            for j in range(7):
                v = sub[i, j]
                if round(v) >= 1:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6,
                            color="white" if v >= 55 else INK)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(AB, fontsize=6); ax.set_yticklabels(AB if k % 4 == 0 else [], fontsize=6)
        ax.tick_params(length=0, pad=1.5)
        ax.set_title(f"{'abcdefg'[k]}  {m}", loc="left", fontsize=7.5, fontweight="bold", pad=3)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True); ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
    # last slot: colour scale and key
    ax = axes[7]; ax.axis("off")
    cax = fig.add_axes([0.80, 0.15, 0.15, 0.025])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, length=2, width=0.4)
    cb.set_label("% of ground-truth pixels", fontsize=6.5)
    key = "\n".join(f"{a}  {s}" for a, s in zip(AB, SP))
    ax.text(0.08, 0.98, key, transform=ax.transAxes, va="top", ha="left", fontsize=6.3, linespacing=1.35)
    fig.supxlabel("Predicted species", fontsize=7.5, y=0.01)
    fig.supylabel("Ground-truth species", fontsize=7.5, x=0.008)
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.07, top=0.95, wspace=0.12, hspace=0.25)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "Fig3.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "Fig3.tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(OUT, "Fig3.pdf"))
    print("written", OUT)


if __name__ == "__main__":
    main()
