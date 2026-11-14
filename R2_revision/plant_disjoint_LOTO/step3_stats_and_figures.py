#!/usr/bin/env python3
"""
step3_stats_and_figures.py
==========================
Step 3 of the LOTO analysis. It reads the outputs of step2_evaluate_test_folds.py (--split test)
and writes:

  * Table S4 inputs: pairwise model comparisons with cluster_bootstrap_compare.py, unchanged
    (10,000 cluster-bootstrap resamples, clusters = treatment x species, Holm across the
    21 pairs, equivalence margin +-2 points). It runs on per-image mIoU (primary) and on
    per-image macro F1 (ANALYSIS_PLAN.md, Section 4). For an image in annotation_corrections.json
    the cluster uses the author-confirmed species (the fold file still holds the old manifest
    species);
  * draft tables as CSV (Table 3: mean +- SD across the three folds, and per fold; Table S3:
    per-class IoU; Table S9: selected epoch and epochs run). Numbers are in percent;
  * Fig. 3: normalised confusion matrices, styled as R1 make_fig3_confusion_matrices.py
    (species rows/columns; row sums from the full 8-class matrix, so pixels predicted as
    background stay in the denominator). It is drawn from the confusion matrix summed over the
    three test folds (all 572 test images), plus one figure per fold.

Usage (needs numpy, pandas, matplotlib, seaborn):
  python step3_stats_and_figures.py --eval_root /workspace/loto_eval --out_root /workspace/loto_stats
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["UNet", "DeepLabV3", "SegFormer", "MaskRCNN", "SoloV2", "YOLO11seg", "PGSS_YOLO11"]
DISPLAY = {"UNet": "U-Net", "DeepLabV3": "DeepLabv3", "SegFormer": "SegFormer", "MaskRCNN": "Mask R-CNN",
           "SoloV2": "Simplified SOLOv2", "YOLO11seg": "YOLO11-seg", "PGSS_YOLO11": "YOLO11-PGSS"}
FOLDS = ["test_trt20", "test_trt50", "test_trt80"]
CLASS_NAMES = ["background", "Akebia", "Campsis", "Hydrangea", "Lonicera",
               "Madame Galen", "Parthenocissus", "Wisteria"]
SPECIES_IDX = list(range(1, 8))
BOOT_B = 10000
DELTA = 0.02          # pre-specified equivalence margin (2 percentage points)


def load_bootstrap_module():
    path = os.path.join(HERE, "..", "cluster_bootstrap_compare.py")
    spec = importlib.util.spec_from_file_location("cluster_bootstrap_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct(x):
    return 100.0 * x


def tables(eval_root, out):
    s = pd.read_csv(os.path.join(eval_root, "summary_per_fold.csv"))
    if (s.split != "test").any():
        raise SystemExit("summary_per_fold.csv is not from a test evaluation")
    if sorted(s.model.unique()) != sorted(MODELS) or sorted(s.fold.unique()) != FOLDS or len(s) != 21:
        raise SystemExit("summary_per_fold.csv does not hold all 7 models x 3 folds")
    s["model_name"] = s.model.map(DISPLAY)
    order = {m: i for i, m in enumerate(MODELS)}
    s = s.sort_values(["model", "fold"], key=lambda c: c.map(order) if c.name == "model" else c)

    main = ["mIoU_8class", "mIoU_7species", "macro_precision_8class", "macro_recall_8class",
            "macro_f1_8class", "coco_AP", "coco_AP50", "coco_AP75"]
    rows_ms, rows_fold = [], []
    for m in MODELS:
        g = s[s.model == m]
        r = {"model": DISPLAY[m], "n_test_images": int(g.n_images.sum())}
        for c in main:
            v = pd.to_numeric(g[c], errors="coerce").values
            if np.isnan(v).all():
                r[c] = "n/a"
                continue
            r[c] = f"{pct(v.mean()):.1f} ± {pct(v.std(ddof=1)):.1f}"
        rows_ms.append(r)
        for _, x in g.iterrows():
            rf = {"model": DISPLAY[m], "fold": x.fold, "n_images": int(x.n_images)}
            for c in main:
                v = pd.to_numeric(pd.Series([x[c]]), errors="coerce").iloc[0]
                rf[c] = "n/a" if pd.isna(v) else f"{pct(v):.1f}"
            rows_fold.append(rf)
    pd.DataFrame(rows_ms).to_csv(os.path.join(out, "table3_mean_sd_across_folds.csv"), index=False)
    pd.DataFrame(rows_fold).to_csv(os.path.join(out, "table3_per_fold.csv"), index=False)

    rows_s3 = []
    for m in MODELS:
        g = s[s.model == m]
        r = {"model": DISPLAY[m]}
        for c in CLASS_NAMES:
            v = g[f"IoU_{c}"].values.astype(float)
            r[c] = f"{pct(v.mean()):.1f} ± {pct(v.std(ddof=1)):.1f}"
        rows_s3.append(r)
    pd.DataFrame(rows_s3).to_csv(os.path.join(out, "tableS3_per_class_iou_mean_sd.csv"), index=False)

    s9 = s[["model_name", "fold", "selected_epoch", "epochs_run"]].rename(columns={"model_name": "model"})
    s9.to_csv(os.path.join(out, "tableS9_selected_epochs.csv"), index=False)
    return s


def species_corrections():
    """Author-confirmed species of corrected images (annotation_corrections.json). The species
    column of loto_fold_assignment.csv still holds the old manifest species for them, so their
    treatment x species cluster is set from the confirmed species here."""
    path = os.path.join(HERE, "annotation_corrections.json")
    if not os.path.isfile(path):
        return {}
    data = json.load(open(path))
    items = data.get("corrections", data) if isinstance(data, dict) else data
    return {c["file_name"]: c["kept_category"] for c in items if c.get("kept_category")}


def pairwise(eval_root, out):
    bc = load_bootstrap_module()
    fix = species_corrections()
    res = {}
    for score, fname in (("miou", "bootstrap_input_per_image_miou.csv"),
                         ("macro_f1", "bootstrap_input_per_image_f1.csv")):
        df = pd.read_csv(os.path.join(eval_root, fname))
        hit = df.image_id.isin(fix.keys())
        df.loc[hit, "cluster_id"] = [cid.split("|")[0] + "|" + fix[i]
                                     for cid, i in zip(df.loc[hit, "cluster_id"], df.loc[hit, "image_id"])]
        n_fixed = int(df.loc[hit, "image_id"].nunique())
        n_missing = int(df.score.isna().sum())
        df = df.dropna(subset=["score"])
        df["model"] = df.model.map(DISPLAY)
        t = bc.compare(df, B=BOOT_B, delta=DELTA)
        t["n_images_with_nan_score_dropped"] = n_missing
        t.to_csv(os.path.join(out, f"tableS4_pairwise_{score}.csv"), index=False)
        res[score] = {"n_pairs": len(t), "n_clusters": int(t.n_clusters.iloc[0]),
                      "n_images": int(t.n_images.iloc[0]), "n_nan_dropped": n_missing,
                      "n_images_cluster_from_corrected_species": n_fixed,
                      "n_significant_holm_0.05": int((t.p_holm < 0.05).sum()),
                      "n_equivalent_within_2_points": int(t.equivalent_within_delta.sum())}
    return res


def figures(eval_root, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    cms = json.load(open(os.path.join(eval_root, "confusion_matrices.json")))
    species = CLASS_NAMES[1:]

    def norm(cm):
        cm = np.array(cm, dtype=float)
        sub = cm[np.ix_(SPECIES_IDX, SPECIES_IDX)]
        rs = cm[SPECIES_IDX, :].sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return sub / rs

    def panel(ax, cmn, title, cbar=False):
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=species, yticklabels=species,
                    cbar=cbar, ax=ax, vmin=0, vmax=1, square=True, annot_kws={"size": 7})
        ax.set_ylabel("Ground Truth Class", fontsize=8)
        ax.set_xlabel("Predicted Class", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    sets = [("all_folds", cms["all_folds"], "LOTO test folds combined (n = 572 images)")]
    sets += [(f, cms["per_fold"][f], f"LOTO test fold {f.replace('test_trt', 'treatment ')}") for f in FOLDS]
    written = []
    for tag, mats, label in sets:
        fig, axes = plt.subplots(2, 4, figsize=(26, 13))
        axes = axes.flatten()
        for i, m in enumerate(MODELS):
            panel(axes[i], norm(mats[m]), DISPLAY[m])
        axes[-1].axis("off")
        fig.suptitle(f"Normalized confusion matrices, seven models — {label}", fontsize=14, y=1.0)
        plt.tight_layout()
        p = os.path.join(out, f"fig3_confusion_matrices_{tag}.png")
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    # the combined matrices as CSV (row-normalised species block), for the text
    rows = []
    for m in MODELS:
        cmn = norm(cms["all_folds"][m])
        for i, g in enumerate(species):
            rows.append({"model": DISPLAY[m], "ground_truth": g, **{p_: round(float(cmn[i, j]), 4)
                                                                     for j, p_ in enumerate(species)}})
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig3_normalized_species_cm_all_folds.csv"), index=False)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out_root", required=True)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.eval_root, "eval_manifest.json")))
    if man.get("smoke_test_NOT_A_RESULT") or man.get("partial_run") or man.get("failures"):
        raise SystemExit("eval_root is a smoke test, a partial run or has failures; not a result set")
    os.makedirs(a.out_root, exist_ok=True)
    tables(a.eval_root, a.out_root)
    stats = pairwise(a.eval_root, a.out_root)
    figs = figures(a.eval_root, a.out_root)
    json.dump({"eval_root": a.eval_root, "eval_manifest_started_utc": man.get("started_utc"),
               "bootstrap_B": BOOT_B, "equivalence_margin": DELTA, "pairwise": stats, "figures": figs},
              open(os.path.join(a.out_root, "step3_summary.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
