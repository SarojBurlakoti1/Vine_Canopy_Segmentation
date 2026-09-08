"""
Evaluate Mask R-CNN with the SAME pixel-level methodology and matched output formats as the
other models: normalized confusion matrix, per-class IoU, held-out Wisteria comparison, and
canopy-area validation. Instance predictions are rasterized to a per-pixel class map on the
same 512² grid, using the same clean-split COCO ground truth.
"""
import os
import sys
import json
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision import transforms as T
from pycocotools.coco import COCO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_maskrcnn import build_model, DATA_ROOT, SAVE_DIR, IMG, CLASS_NAMES  # noqa: E402

SPECIES = CLASS_NAMES[1:]
CONF = 0.5
OUT = os.path.join(os.path.dirname(SAVE_DIR), "results")
CANOPY_OUT = os.path.join(os.path.dirname(SAVE_DIR), "canopy_area_validation")
os.makedirs(OUT, exist_ok=True); os.makedirs(CANOPY_OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_WIST = "Trt-3-Wisteria-floribunda-22-_aug1_jpeg.rf.42660424bbf147fedbfc34d51112efc7.jpg"
to_tensor = T.ToTensor()

model = build_model().to(device)
ckpt = torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
print(f"Loaded best_model.pth (epoch {ckpt['epoch']+1}, val loss {ckpt['val_loss']:.4f})")


def iou_from_cm(cm):
    tp = np.diag(cm).astype(np.float64); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    ious = tp / (tp + fp + fn + 1e-7)
    return ious, np.mean(ious)


def plot_cm(cm, save_dir, label="Mask R-CNN"):
    idx = [i for i, c in enumerate(CLASS_NAMES) if c != "background"]
    sp = cm[np.ix_(idx, idx)].astype("float")
    rs = cm[idx, :].sum(1, keepdims=True).astype("float"); rs[rs == 0] = 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(sp / rs, annot=True, fmt=".2f", cmap="Blues", xticklabels=SPECIES, yticklabels=SPECIES, cbar=True)
    plt.ylabel("Ground Truth Class"); plt.xlabel("Predicted Class")
    plt.title(f"Normalized Confusion Matrix ({label})")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0); plt.tight_layout()
    p = os.path.join(save_dir, "confusion_matrix.png"); plt.savefig(p, dpi=200); plt.close()
    print("Confusion matrix saved:", p); return p


def gt_pixel_map(coco, iid):
    from pycocotools import mask as cm
    info = coco.imgs[iid]; H, W = info["height"], info["width"]
    canvas = np.zeros((H, W), np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=iid)):
        cid = ann["category_id"]
        if cid == 0 or not isinstance(ann["segmentation"], list):
            continue
        for poly in ann["segmentation"]:
            pts = np.array(poly, np.float32).reshape(-1, 2).astype(np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(canvas, [pts], int(cid))
    return cv2.resize(canvas, (IMG, IMG), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def predict(image_path):
    """Return (H0,W0, list of (mask_full, label, score, box)) and a 512² class map."""
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    H0, W0 = img.shape[:2]
    x = to_tensor(cv2.resize(img, (IMG, IMG))).to(device)
    out = model([x])[0]
    class_map = np.zeros((IMG, IMG), np.uint8)
    dets = []
    order = torch.argsort(out["scores"])          # low->high, paint high last
    for i in order.tolist():
        score = float(out["scores"][i])
        if score < CONF:
            continue
        m = (out["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)   # 512²
        if m.sum() == 0:
            continue
        lbl = int(out["labels"][i])
        class_map[m > 0] = lbl
        dets.append((m, lbl, score, out["boxes"][i].cpu().numpy()))
    return img, H0, W0, dets, class_map


# ---- 1. Confusion matrix + metrics on VALID ----
print("Evaluating validation split (pixel-level)...")
coco = COCO(os.path.join(DATA_ROOT, "valid", "_annotations.coco.json"))
img_dir = os.path.join(DATA_ROOT, "valid")
all_t, all_p = [], []
for iid, info in coco.imgs.items():
    gt = gt_pixel_map(coco, iid).reshape(-1)
    _, _, _, _, pm = predict(os.path.join(img_dir, info["file_name"]))
    all_t.append(gt); all_p.append(pm.reshape(-1))
yt = np.concatenate(all_t); yp = np.concatenate(all_p)
acc = accuracy_score(yt, yp); f1 = f1_score(yt, yp, average="weighted", zero_division=0)
prec = precision_score(yt, yp, average="weighted", zero_division=0)
rec = recall_score(yt, yp, average="weighted", zero_division=0)
cm = confusion_matrix(yt, yp, labels=list(range(8)))
ious, miou = iou_from_cm(cm)
np.save(os.path.join(OUT, "confusion_matrix.npy"), cm)
metrics = {"accuracy": float(acc), "precision_weighted": float(prec), "recall_weighted": float(rec),
           "f1_weighted": float(f1), "miou": float(miou),
           "per_class_iou": {CLASS_NAMES[i]: float(ious[i]) for i in range(8)}}
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
plot_cm(cm, OUT)
print(f"  acc={acc:.4f} f1={f1:.4f} mIoU(8)={miou:.4f}")
print("  per-class IoU:", {k: round(v, 3) for k, v in metrics["per_class_iou"].items()})


# ---- 2. Wisteria comparison on the fixed held-out test image ----
def sample_comparison():
    fixed = os.path.join(DATA_ROOT, "test", FIXED_WIST)
    if not os.path.exists(fixed):
        print("Fixed image not found"); return
    img, H0, W0, dets, _ = predict(fixed)
    overlay = img.copy(); boxes = []
    for (m, lbl, score, box) in dets:
        mfull = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
        red = np.zeros_like(overlay); red[:, :, 0] = 255; a = 0.55
        overlay[mfull] = (overlay[mfull] * (1 - a) + red[mfull] * a).astype(np.uint8)
        sx, sy = W0 / IMG, H0 / IMG
        x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
        boxes.append((int(x1), int(y1), int(x2), int(y2), SPECIES[lbl - 1], score))
    for (x1, y1, x2, y2, cn, cf) in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)
        lab = f"{cn} {cf:.2f}"
        (tw, th), bl = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        tx = min(max(0, x1), W0 - tw - 10); ty = max(0, y1 - th - bl - 6)
        cv2.rectangle(overlay, (tx, ty), (tx + tw + 10, ty + th + bl + 6), (255, 0, 0), -1)
        cv2.putText(overlay, lab, (tx + 5, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    fig, ax = plt.subplots(1, 2, figsize=(20, 11.5))
    ax[0].imshow(img); ax[0].set_title("Original RGB", fontsize=16); ax[0].axis("off")
    ax[1].imshow(overlay); ax[1].set_title("Mask R-CNN Prediction", fontsize=16); ax[1].axis("off")
    fig.suptitle("Wisteria floribunda — instance segmentation + bounding boxes", fontsize=18)
    plt.tight_layout()
    p = os.path.join(OUT, "sample_prediction_comparison.png")
    plt.savefig(p, bbox_inches="tight", dpi=150); plt.close()
    print(f"Sample comparison saved: {p} ({len(boxes)} instances)")


sample_comparison()


# ---- 3. Canopy validation on TEST ----
def canopy():
    coco = COCO(os.path.join(DATA_ROOT, "test", "_annotations.coco.json"))
    cat = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    img_dir = os.path.join(DATA_ROOT, "test"); rows = []
    for iid, info in coco.imgs.items():
        gt = gt_pixel_map(coco, iid); _, _, _, _, pm = predict(os.path.join(img_dir, info["file_name"]))
        gt_fg = gt > 0; pr_fg = pm > 0
        gt_pct = 100.0 * gt_fg.sum() / (IMG * IMG); pr_pct = 100.0 * pr_fg.sum() / (IMG * IMG)
        cls_px = {}
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=iid)):
            nm = cat[ann["category_id"]]
            if nm != "objects":
                cls_px[nm] = cls_px.get(nm, 0) + 1
        dom = max(cls_px, key=cls_px.get) if cls_px else "none"
        inter = int((gt_fg & pr_fg).sum()); union = int((gt_fg | pr_fg).sum())
        rows.append({"image": info["file_name"], "species": dom,
                     "gt_canopy_%": round(gt_pct, 3), "pred_canopy_%": round(pr_pct, 3),
                     "abs_error_pp": round(abs(pr_pct - gt_pct), 3), "signed_error_pp": round(pr_pct - gt_pct, 3),
                     "area_ratio_pred_gt": round(pr_pct / gt_pct, 3) if gt_pct > 0 else np.nan,
                     "canopy_iou": round(inter / union, 3) if union > 0 else np.nan})
    df = pd.DataFrame(rows).sort_values("species").reset_index(drop=True)
    gt = df["gt_canopy_%"].to_numpy(); pr = df["pred_canopy_%"].to_numpy()
    mae = float(np.mean(np.abs(pr - gt))); bias = float(np.mean(pr - gt))
    rmse = float(np.sqrt(np.mean((pr - gt) ** 2))); r = float(np.corrcoef(gt, pr)[0, 1])
    ss_res = float(np.sum((gt - pr) ** 2)); ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    summary = pd.DataFrame({"metric": ["Test images", "Mean GT canopy %", "Mean predicted canopy %",
                                       "MAE (percentage points)", "Bias (pred − GT, pp)", "RMSE (pp)",
                                       "Pearson r (GT vs pred)", "R²", "Mean canopy IoU"],
                            "value": [len(df), round(float(gt.mean()), 3), round(float(pr.mean()), 3),
                                      round(mae, 3), round(bias, 3), round(rmse, 3), round(r, 4),
                                      round(r2, 4), round(float(df["canopy_iou"].mean(skipna=True)), 3)]})
    per_species = df.groupby("species").apply(
        lambda g: pd.Series({"images": len(g), "mean_gt_%": round(g["gt_canopy_%"].mean(), 3),
                             "mean_pred_%": round(g["pred_canopy_%"].mean(), 3),
                             "MAE_pp": round(np.mean(np.abs(g["pred_canopy_%"] - g["gt_canopy_%"])), 3),
                             "bias_pp": round(np.mean(g["pred_canopy_%"] - g["gt_canopy_%"]), 3),
                             "mean_canopy_iou": round(g["canopy_iou"].mean(skipna=True), 3)}),
        include_groups=False).reset_index()
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
    fig, ax = plt.subplots(figsize=(7, 7))
    lim = max(gt.max(), pr.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#9aa292", lw=1, label="perfect agreement (y = x)")
    cmap = plt.get_cmap("tab10")
    for i, sp in enumerate(sorted(df["species"].unique())):
        s = df[df["species"] == sp]
        ax.scatter(s["gt_canopy_%"], s["pred_canopy_%"], s=28, alpha=0.75, color=cmap(i % 10), label=sp)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("Ground-truth canopy cover (%)"); ax.set_ylabel("Predicted canopy cover (%)")
    ax.set_title(f"Canopy area validation · Mask R-CNN · test (n={len(df)})\nMAE {mae:.2f} pp · bias {bias:+.2f} pp · r {r:.3f}")
    ax.legend(fontsize=8, loc="upper left"); plt.tight_layout()
    plt.savefig(os.path.join(CANOPY_OUT, "canopy_area_scatter_test.png"), dpi=150); plt.close()
    print(f"Canopy: MAE={mae:.2f} bias={bias:+.2f} r={r:.3f} R2={r2:.3f} -> {xlsx}")


canopy()
print("\nMask R-CNN evaluation complete. Outputs in", OUT, "and", CANOPY_OUT)
