"""
Evaluate U-Net on the clean split with the same pixel-level methodology and matched
output formats as SOLOv2 / YOLO11-seg: normalized confusion matrix, per-class IoU,
held-out Wisteria comparison (semantic mask + derived box), and canopy-area validation.
U-Net is semantic (no instances / no mAP / no native boxes) — the box is derived from
the predicted Wisteria region's extent.
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
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_unet import UNet, VineDataset, DATA_ROOT, SAVE_DIR, IMG, CLASS_NAMES  # noqa: E402

SPECIES = CLASS_NAMES[1:]
OUT = os.path.join(os.path.dirname(SAVE_DIR), "results")
CANOPY_OUT = os.path.join(os.path.dirname(SAVE_DIR), "canopy_area_validation")
os.makedirs(OUT, exist_ok=True); os.makedirs(CANOPY_OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_WIST = "Trt-3-Wisteria-floribunda-22-_aug1_jpeg.rf.42660424bbf147fedbfc34d51112efc7.jpg"

model = UNet(3, len(CLASS_NAMES)).to(device)
ckpt = torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
print(f"Loaded best_model.pth (epoch {ckpt['epoch']+1}, val mIoU {ckpt['val_miou']:.4f})")


def iou_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    ious = tp / (tp + fp + fn + 1e-7)
    return ious, np.mean(ious)


def plot_cm(cm, save_dir, label="U-Net"):
    idx = [i for i, c in enumerate(CLASS_NAMES) if c != "background"]
    sp = cm[np.ix_(idx, idx)].astype("float")
    rs = cm[idx, :].sum(1, keepdims=True).astype("float"); rs[rs == 0] = 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(sp / rs, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=SPECIES, yticklabels=SPECIES, cbar=True)
    plt.ylabel("Ground Truth Class"); plt.xlabel("Predicted Class")
    plt.title(f"Normalized Confusion Matrix ({label})")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0); plt.tight_layout()
    p = os.path.join(save_dir, "confusion_matrix.png"); plt.savefig(p, dpi=200); plt.close()
    print("Confusion matrix saved:", p); return p


# ============ 1. Confusion matrix + metrics on VALID ============
print("Evaluating validation split...")
val_loader = DataLoader(VineDataset("valid"), batch_size=8, shuffle=False, num_workers=4)
all_t, all_p = [], []
with torch.no_grad():
    for images, masks in val_loader:
        pred = torch.argmax(model(images.to(device)), dim=1).cpu().numpy()
        all_p.append(pred.reshape(-1)); all_t.append(masks.numpy().reshape(-1))
yt = np.concatenate(all_t); yp = np.concatenate(all_p)
acc = accuracy_score(yt, yp)
f1 = f1_score(yt, yp, average="weighted", zero_division=0)
prec = precision_score(yt, yp, average="weighted", zero_division=0)
rec = recall_score(yt, yp, average="weighted", zero_division=0)
cm = confusion_matrix(yt, yp, labels=list(range(len(CLASS_NAMES))))
ious, miou = iou_from_cm(cm)
np.save(os.path.join(OUT, "confusion_matrix.npy"), cm)
metrics = {"accuracy": float(acc), "precision_weighted": float(prec), "recall_weighted": float(rec),
           "f1_weighted": float(f1), "miou": float(miou),
           "per_class_iou": {CLASS_NAMES[i]: float(ious[i]) for i in range(len(CLASS_NAMES))}}
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
plot_cm(cm, OUT)
print(f"  acc={acc:.4f} f1={f1:.4f} mIoU(8)={miou:.4f}")
print("  per-class IoU:", {k: round(v, 3) for k, v in metrics["per_class_iou"].items()})


# ============ 2. Wisteria comparison on the fixed held-out test image ============
def sample_comparison():
    hits = [os.path.join(DATA_ROOT, s, FIXED_WIST) for s in ("test", "valid", "train")
            if os.path.exists(os.path.join(DATA_ROOT, s, FIXED_WIST))]
    if not hits:
        print("Fixed Wisteria image not found."); return
    fixed = hits[0]
    print("Wisteria fixed image is in", fixed.split(os.sep)[-2], "split.")
    from torchvision import transforms as T
    img = cv2.cvtColor(cv2.imread(fixed), cv2.COLOR_BGR2RGB)
    H0, W0 = img.shape[:2]
    x = T.ToTensor()(cv2.resize(img, (IMG, IMG))).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    wist_idx = CLASS_NAMES.index("Wisteria")
    conf = float(prob[wist_idx][pred == wist_idx].mean().cpu()) if (pred == wist_idx).any() else 0.0
    pred_full = cv2.resize(pred, (W0, H0), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    mask = (pred_full == wist_idx)
    red = np.zeros_like(overlay); red[:, :, 0] = 255; a = 0.55
    overlay[mask] = (overlay[mask] * (1 - a) + red[mask] * a).astype(np.uint8)
    if mask.any():
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)
        lab = f"Wisteria {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        tx = min(max(0, x1), W0 - tw - 10); ty = max(0, y1 - th - bl - 6)
        cv2.rectangle(overlay, (tx, ty), (tx + tw + 10, ty + th + bl + 6), (255, 0, 0), -1)
        cv2.putText(overlay, lab, (tx + 5, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    fig, ax = plt.subplots(1, 2, figsize=(20, 11.5))
    ax[0].imshow(img); ax[0].set_title("Original RGB", fontsize=16); ax[0].axis("off")
    ax[1].imshow(overlay); ax[1].set_title("U-Net Prediction", fontsize=16); ax[1].axis("off")
    fig.suptitle("Wisteria floribunda — semantic segmentation (mask + derived box)", fontsize=18)
    plt.tight_layout()
    p = os.path.join(OUT, "sample_prediction_comparison.png")
    plt.savefig(p, bbox_inches="tight", dpi=150); plt.close()
    print(f"Sample comparison saved: {p} (Wisteria conf {conf:.2f})")


sample_comparison()


# ============ 3. Canopy validation on TEST ============
def canopy():
    coco = COCO(os.path.join(DATA_ROOT, "test", "_annotations.coco.json"))
    cat = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    from torchvision import transforms as T
    rows = []
    for iid, info in coco.imgs.items():
        ip = os.path.join(DATA_ROOT, "test", info["file_name"])
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        H, W = info["height"], info["width"]
        # GT semantic map at 512
        gt = np.zeros((H, W), np.uint8)
        cls_px = {}
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=iid)):
            if not isinstance(ann["segmentation"], list):
                continue
            from pycocotools import mask as cmask
            m = cmask.decode(cmask.frPyObjects(ann["segmentation"], H, W))
            if m.ndim == 3:
                m = np.any(m, axis=2)
            gt[m.astype(bool)] = ann["category_id"]
            nm = cat[ann["category_id"]]
            if nm != "objects":
                cls_px[nm] = cls_px.get(nm, 0) + int(m.sum())
        gt = cv2.resize(gt, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        x = T.ToTensor()(cv2.resize(img, (IMG, IMG))).unsqueeze(0).to(device)
        with torch.no_grad():
            pr = torch.argmax(model(x), dim=1)[0].cpu().numpy()
        gt_fg = gt > 0; pr_fg = pr > 0
        gt_pct = 100.0 * gt_fg.sum() / (IMG * IMG); pr_pct = 100.0 * pr_fg.sum() / (IMG * IMG)
        dom = max(cls_px, key=cls_px.get) if cls_px else "none"
        inter = int((gt_fg & pr_fg).sum()); union = int((gt_fg | pr_fg).sum())
        rows.append({"image": info["file_name"], "species": dom,
                     "gt_canopy_%": round(gt_pct, 3), "pred_canopy_%": round(pr_pct, 3),
                     "abs_error_pp": round(abs(pr_pct - gt_pct), 3),
                     "signed_error_pp": round(pr_pct - gt_pct, 3),
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
    ax.set_title(f"Canopy area validation · U-Net · test (n={len(df)})\n"
                 f"MAE {mae:.2f} pp · bias {bias:+.2f} pp · r {r:.3f}")
    ax.legend(fontsize=8, loc="upper left"); plt.tight_layout()
    plt.savefig(os.path.join(CANOPY_OUT, "canopy_area_scatter_test.png"), dpi=150); plt.close()
    print(f"Canopy: MAE={mae:.2f} bias={bias:+.2f} r={r:.3f} R2={r2:.3f} -> {xlsx}")


canopy()
print("\nU-Net evaluation complete. Outputs in", OUT, "and", CANOPY_OUT)
