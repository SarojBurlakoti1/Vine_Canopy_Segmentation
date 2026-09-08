"""
Evaluate the trained PGSS-YOLO11s on the SAME clean split, using the SAME pixel-level
methodology and ground truth as SimplifiedSOLOv2 — so every output (confusion matrix,
per-class IoU, canopy validation, Wisteria comparison) is directly comparable.

Ground truth: clean-split COCO polygons rasterized to 512² (identical to SOLOv2).
Prediction:   YOLO instance masks rasterized to a per-pixel class map on the same 512² grid.
Operating point: conf=0.25 (YOLO's standard), IoU handled internally by the model.
"""
import os
import sys
import json
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from pycocotools.coco import COCO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# reuse SOLOv2's confusion-matrix plotter for identical styling
sys.path.insert(0, "/root/runpod/SoloV2")
from train_solov2_clean import plot_normalized_confusion_matrix, iou_from_confusion_matrix  # noqa: E402

import sys; sys.path.insert(0,"/root/runpod/PGSS_YOLO11"); import pgss_modules  # register GLSA/AWFF/DEConv
BASE = "/root/runpod/PGSS_YOLO11"
COCO_ROOT = "/root/runpod/SoloV2/Vines_2025-6_clean"          # same GT as SOLOv2
WEIGHTS = os.path.join(BASE, "runs", "pgss_vine_v6", "weights", "best.pt")
OUT = os.path.join(BASE, "results")
CANOPY_OUT = os.path.join(BASE, "canopy_area_validation")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CANOPY_OUT, exist_ok=True)

IMG = 512
CONF = 0.25
SPECIES = ["Akebia", "Campsis", "Hydrangea", "Lonicera", "Madame Galen", "Parthenocissus", "Wisteria"]
CLASS_NAMES = ["background"] + SPECIES          # index 0..7, matches SOLOv2
ID_TO_IDX = {i + 1: i + 1 for i in range(7)}    # COCO cat 1..7 -> class idx 1..7

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(WEIGHTS)


def gt_pixel_map(coco, image_id):
    info = coco.imgs[image_id]
    H, W = info["height"], info["width"]
    canvas = np.zeros((H, W), np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
        cls = ID_TO_IDX.get(ann["category_id"], 0)
        if cls == 0:
            continue
        seg = ann["segmentation"]
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                if pts.shape[0] >= 3:
                    cv2.fillPoly(canvas, [pts], int(cls))
    return cv2.resize(canvas, (IMG, IMG), interpolation=cv2.INTER_NEAREST)


def pred_pixel_map(image_path, conf=CONF):
    r = model.predict(image_path, imgsz=IMG, conf=conf, device=device, verbose=False)[0]
    H, W = r.orig_shape
    canvas = np.zeros((H, W), np.uint8)
    if r.masks is not None and len(r.masks) > 0:
        polys = r.masks.xy
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for idx in np.argsort(confs):            # paint highest-conf last (wins overlaps)
            poly = polys[idx]
            if poly.shape[0] >= 3:
                cv2.fillPoly(canvas, [poly.astype(np.int32)], int(clss[idx]) + 1)
    return cv2.resize(canvas, (IMG, IMG), interpolation=cv2.INTER_NEAREST)


# ============ 1. Confusion matrix + per-class metrics on VALID (matches SOLOv2) ============
def evaluate_split(split):
    coco = COCO(os.path.join(COCO_ROOT, split, "_annotations.coco.json"))
    img_dir = os.path.join(COCO_ROOT, split)
    all_true, all_pred = [], []
    for iid, info in coco.imgs.items():
        gt = gt_pixel_map(coco, iid).reshape(-1)
        pr = pred_pixel_map(os.path.join(img_dir, info["file_name"])).reshape(-1)
        all_true.append(gt)
        all_pred.append(pr)
    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    acc = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, average="weighted", zero_division=0)
    rec = recall_score(yt, yp, average="weighted", zero_division=0)
    f1 = f1_score(yt, yp, average="weighted", zero_division=0)
    cm = confusion_matrix(yt, yp, labels=list(range(8)))
    ious, miou = iou_from_confusion_matrix(cm)
    return acc, prec, rec, f1, cm, ious, miou


print("Evaluating validation split (pixel-level)...")
acc, prec, rec, f1, cm, ious, miou = evaluate_split("valid")
np.save(os.path.join(OUT, "confusion_matrix.npy"), cm)
metrics = {
    "accuracy": float(acc), "precision_weighted": float(prec),
    "recall_weighted": float(rec), "f1_weighted": float(f1), "miou": float(miou),
    "per_class_iou": {CLASS_NAMES[i]: float(ious[i]) for i in range(8)},
}
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
plot_normalized_confusion_matrix(cm, CLASS_NAMES, save_dir=OUT, model_label="PGSS-YOLO11s")
print(f"  acc={acc:.4f} f1={f1:.4f} mIoU(8)={miou:.4f}")
print("  per-class IoU:", {k: round(v, 3) for k, v in metrics["per_class_iou"].items()})


# ============ 2. Wisteria comparison on the SAME fixed test image ============
def sample_comparison():
    fixed = os.path.join(COCO_ROOT, "test",
                         "Trt-3-Wisteria-floribunda-22-_aug1_jpeg.rf.42660424bbf147fedbfc34d51112efc7.jpg")
    img_bgr = cv2.imread(fixed)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb.shape[:2]
    r = model.predict(fixed, imgsz=IMG, conf=CONF, device=device, verbose=False)[0]
    overlay = img_rgb.copy()
    boxes = []
    if r.masks is not None and len(r.masks) > 0:
        polys = r.masks.xy
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        for idx in np.argsort(confs):
            poly = polys[idx].astype(np.int32)
            if poly.shape[0] < 3:
                continue
            m = np.zeros((H0, W0), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            red = np.zeros_like(overlay); red[:, :, 0] = 255
            a = 0.55; mb = m.astype(bool)
            overlay[mb] = (overlay[mb] * (1 - a) + red[mb] * a).astype(np.uint8)
            x1, y1, x2, y2 = xyxy[idx].astype(int)
            boxes.append((x1, y1, x2, y2, SPECIES[clss[idx]], float(confs[idx])))
    for (x1, y1, x2, y2, cname, cf) in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)
        label = f"{cname} {cf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        tx1 = min(max(0, x1), W0 - tw - 10); ty1 = max(0, y1 - th - bl - 6)
        cv2.rectangle(overlay, (tx1, ty1), (tx1 + tw + 10, ty1 + th + bl + 6), (255, 0, 0), -1)
        cv2.putText(overlay, label, (tx1 + 5, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    fig, ax = plt.subplots(1, 2, figsize=(20, 11.5))
    ax[0].imshow(img_rgb); ax[0].set_title("Original RGB", fontsize=16); ax[0].axis("off")
    ax[1].imshow(overlay); ax[1].set_title("PGSS-YOLO11s Prediction", fontsize=16); ax[1].axis("off")
    fig.suptitle("Wisteria floribunda — instance segmentation + bounding boxes", fontsize=18)
    plt.tight_layout()
    p = os.path.join(OUT, "sample_prediction_comparison.png")
    plt.savefig(p, bbox_inches="tight", dpi=150); plt.close()
    print("Sample comparison saved:", p, f"({len(boxes)} instances)")


sample_comparison()


# ============ 3. Canopy area validation on TEST (matches SOLOv2 method) ============
def canopy_validation():
    coco = COCO(os.path.join(COCO_ROOT, "test", "_annotations.coco.json"))
    img_dir = os.path.join(COCO_ROOT, "test")
    cat = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    rows = []
    for iid, info in coco.imgs.items():
        gt_map = gt_pixel_map(coco, iid)
        pr_map = pred_pixel_map(os.path.join(img_dir, info["file_name"]))
        gt_fg = gt_map > 0
        pr_fg = pr_map > 0
        gt_px, pr_px = int(gt_fg.sum()), int(pr_fg.sum())
        gt_pct = 100.0 * gt_px / (IMG * IMG)
        pr_pct = 100.0 * pr_px / (IMG * IMG)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid))
        species_px = {}
        for a in anns:
            nm = cat[a["category_id"]]
            if nm == "objects":
                continue
            species_px[nm] = species_px.get(nm, 0) + 1
        dom = max(species_px, key=species_px.get) if species_px else "none"
        inter = int((gt_fg & pr_fg).sum()); union = int((gt_fg | pr_fg).sum())
        rows.append({
            "image": info["file_name"], "species": dom,
            "gt_instances": len([a for a in anns if cat[a["category_id"]] != "objects"]),
            "gt_canopy_%": round(gt_pct, 3), "pred_canopy_%": round(pr_pct, 3),
            "abs_error_pp": round(abs(pr_pct - gt_pct), 3),
            "signed_error_pp": round(pr_pct - gt_pct, 3),
            "area_ratio_pred_gt": round(pr_pct / gt_pct, 3) if gt_pct > 0 else np.nan,
            "canopy_iou": round(inter / union, 3) if union > 0 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("species").reset_index(drop=True)
    gt = df["gt_canopy_%"].to_numpy(); pr = df["pred_canopy_%"].to_numpy()
    mae = float(np.mean(np.abs(pr - gt))); bias = float(np.mean(pr - gt))
    rmse = float(np.sqrt(np.mean((pr - gt) ** 2))); r = float(np.corrcoef(gt, pr)[0, 1])
    ss_res = float(np.sum((gt - pr) ** 2)); ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    summary = pd.DataFrame({
        "metric": ["Test images", "Mean GT canopy %", "Mean predicted canopy %",
                   "MAE (percentage points)", "Bias (pred − GT, pp)", "RMSE (pp)",
                   "Pearson r (GT vs pred)", "R²", "Mean canopy IoU"],
        "value": [len(df), round(float(gt.mean()), 3), round(float(pr.mean()), 3),
                  round(mae, 3), round(bias, 3), round(rmse, 3), round(r, 4),
                  round(r2, 4), round(float(df["canopy_iou"].mean(skipna=True)), 3)],
    })
    per_species = df.groupby("species").apply(
        lambda g: pd.Series({
            "images": len(g),
            "mean_gt_%": round(g["gt_canopy_%"].mean(), 3),
            "mean_pred_%": round(g["pred_canopy_%"].mean(), 3),
            "MAE_pp": round(np.mean(np.abs(g["pred_canopy_%"] - g["gt_canopy_%"])), 3),
            "bias_pp": round(np.mean(g["pred_canopy_%"] - g["gt_canopy_%"]), 3),
            "mean_canopy_iou": round(g["canopy_iou"].mean(skipna=True), 3),
        }), include_groups=False).reset_index()

    xlsx = os.path.join(CANOPY_OUT, "canopy_area_validation_test.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        per_species.to_excel(xw, sheet_name="Per-species", index=False)
        df.to_excel(xw, sheet_name="Per-image", index=False)
        from openpyxl.styles import Font
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True)
            for col in ws.columns:
                w = max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2
                ws.column_dimensions[col[0].column_letter].width = min(w, 40)
    # scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    lim = max(gt.max(), pr.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#9aa292", lw=1, label="perfect agreement (y = x)")
    cmap = plt.get_cmap("tab10")
    for i, sp in enumerate(sorted(df["species"].unique())):
        s = df[df["species"] == sp]
        ax.scatter(s["gt_canopy_%"], s["pred_canopy_%"], s=28, alpha=0.75, color=cmap(i % 10), label=sp)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("Ground-truth canopy cover (%)"); ax.set_ylabel("Predicted canopy cover (%)")
    ax.set_title(f"Canopy area validation · PGSS-YOLO11s · test (n={len(df)})\n"
                 f"MAE {mae:.2f} pp · bias {bias:+.2f} pp · r {r:.3f}")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(CANOPY_OUT, "canopy_area_scatter_test.png"), dpi=150); plt.close()
    print(f"Canopy: MAE={mae:.2f} bias={bias:+.2f} r={r:.3f} R2={r2:.3f} -> {xlsx}")
    return summary, per_species


canopy_validation()
print("\nPGSS-YOLO11s evaluation complete. Outputs in", OUT, "and", CANOPY_OUT)
