"""
Fig 4: qualitative comparison on a single representative Wisteria test image, all 7
models, seed-42 (primary) checkpoints.

Representative image: the project's own previously-used fixed reference image
(Trt-3-Wisteria-floribunda-22-...42660424...) is from the pre-rebuild dataset export
and does not exist in split_assignment_final.csv or rebuilt/final/* at all -- its
same-plant counterpart in the rebuilt/leakage-corrected dataset landed in TRAIN, not
test, so it can't be reused for a legitimate held-out qualitative figure. New pick:
IMG_9797_aug1_jpg.rf.8a6247d6d92a4ec13af9778c91bf47b9.jpg -- single clean Wisteria
instance, only annotation in the image, from the current test set.

For each model: run inference at its own native settings (matching run_multiseed_evaluation.py
exactly), extract the predicted Wisteria region (semantic argmax for the 3 semantic-only
models; best-matching predicted instance of category Wisteria for the 4 instance models),
compute IoU against the single GT Wisteria polygon on THIS image specifically, and save
an overlay (green=GT-only, red=pred-only, yellow=overlap) plus report IoU as the title.

Outputs: fig4_qualitative_<Model>.png (8 individual: original+GT, then one per model)
         fig4_qualitative_combined.png (3x3 grid: original+GT, then 7 models)
         fig4_iou_summary.json / .csv
"""
import os
import sys
import json
import importlib.util

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

RUN = "/root/vine-project/Run"
CKPT_ROOT = "/workspace/vine-project-Run"
COCO_TEST_DIR = f"{RUN}/rebuilt/final/test"
COCO_TEST_JSON = f"{COCO_TEST_DIR}/_annotations.coco.json"
YOLO_TEST_IMG_DIR = f"{RUN}/rebuilt/final_yolo/test/images"
IMG = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_FILE = "IMG_9797_aug1_jpg.rf.8a6247d6d92a4ec13af9778c91bf47b9.jpg"
WISTERIA_CAT_ID = 7  # COCO category id (matches CLASS_NAMES[1:] = Akebia..Wisteria -> index 7 = Wisteria)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


coco = COCO(COCO_TEST_JSON)
target_iid = [iid for iid, info in coco.imgs.items() if info["file_name"] == TARGET_FILE][0]
info = coco.imgs[target_iid]
H0, W0 = info["height"], info["width"]
img_path = os.path.join(COCO_TEST_DIR, TARGET_FILE)
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# GT Wisteria mask, native resolution
gt_mask = np.zeros((H0, W0), np.uint8)
for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_iid)):
    if ann["category_id"] != WISTERIA_CAT_ID:
        continue
    for poly in ann["segmentation"]:
        pts = np.array(poly, np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(gt_mask, [pts], 1)
print(f"Target image: {TARGET_FILE} ({W0}x{H0}), GT Wisteria pixels: {gt_mask.sum()}")


def iou(pred_bool, gt_bool):
    inter = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def make_overlay(pred_bool, title):
    overlay = img_rgb.copy().astype(np.float32)
    gt_bool = gt_mask.astype(bool)
    tp = pred_bool & gt_bool
    fp = pred_bool & ~gt_bool
    fn = ~pred_bool & gt_bool
    color = np.zeros_like(overlay)
    color[tp] = [255, 255, 0]   # yellow = correct overlap
    color[fp] = [255, 0, 0]     # red = false positive (over-prediction)
    color[fn] = [0, 255, 0]     # green = false negative (missed GT)
    alpha = 0.55
    mask_any = tp | fp | fn
    overlay[mask_any] = overlay[mask_any] * (1 - alpha) + color[mask_any] * alpha
    return overlay.astype(np.uint8), title


results = {}
overlays = []  # (title, overlay_img)

# ---------------------------------------------------------------------------
# 3 semantic-only models
# ---------------------------------------------------------------------------
def sem_predict_wisteria(build_fn, ckpt_path, preprocess_extra=None):
    model = build_fn().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    to_tensor = T.ToTensor()
    img512 = cv2.resize(img_rgb, (IMG, IMG))
    x = to_tensor(img512).unsqueeze(0).to(device)
    if preprocess_extra is not None:
        x = preprocess_extra(x)
    with torch.no_grad():
        yield model, x


mod_unet = load_module("m_unet", f"{RUN}/UNet/train_unet.py")
model = mod_unet.UNet(3, 8).to(device)
ckpt = torch.load(f"{CKPT_ROOT}/UNet/checkpoints/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
x = T.ToTensor()(cv2.resize(img_rgb, (IMG, IMG))).unsqueeze(0).to(device)
with torch.no_grad():
    pred512 = torch.argmax(model(x), dim=1)[0].cpu().numpy().astype(np.uint8)
pred_native = cv2.resize(pred512, (W0, H0), interpolation=cv2.INTER_NEAREST)
pred_wist = pred_native == 7
results["UNet"] = iou(pred_wist, gt_mask.astype(bool))
overlays.append(make_overlay(pred_wist, f"UNet (IoU={results['UNet']:.3f})"))
print("UNet IoU:", results["UNet"])

mod_dl = load_module("m_dl", f"{RUN}/DeepLabV3/train_deeplabv3.py")
model = mod_dl.build_model().to(device)
ckpt = torch.load(f"{CKPT_ROOT}/DeepLabV3/checkpoints/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
x = T.ToTensor()(cv2.resize(img_rgb, (IMG, IMG))).unsqueeze(0).to(device)
with torch.no_grad():
    pred512 = torch.argmax(model(x)["out"], dim=1)[0].cpu().numpy().astype(np.uint8)
pred_native = cv2.resize(pred512, (W0, H0), interpolation=cv2.INTER_NEAREST)
pred_wist = pred_native == 7
results["DeepLabV3"] = iou(pred_wist, gt_mask.astype(bool))
overlays.append(make_overlay(pred_wist, f"DeepLabV3 (IoU={results['DeepLabV3']:.3f})"))
print("DeepLabV3 IoU:", results["DeepLabV3"])

mod_sf = load_module("m_sf", f"{RUN}/SegFormer/train_segformer.py")
model = mod_sf.build_model().to(device)
ckpt = torch.load(f"{CKPT_ROOT}/SegFormer/checkpoints/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
x = mod_sf.NORM(T.ToTensor()(cv2.resize(img_rgb, (IMG, IMG)))).unsqueeze(0).to(device)
with torch.no_grad():
    logits = mod_sf.forward_logits(model, x)
    pred512 = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
pred_native = cv2.resize(pred512, (W0, H0), interpolation=cv2.INTER_NEAREST)
pred_wist = pred_native == 7
results["SegFormer"] = iou(pred_wist, gt_mask.astype(bool))
overlays.append(make_overlay(pred_wist, f"SegFormer (IoU={results['SegFormer']:.3f})"))
print("SegFormer IoU:", results["SegFormer"])

# ---------------------------------------------------------------------------
# Mask R-CNN -- best-matching predicted Wisteria instance
# ---------------------------------------------------------------------------
mod_mr = load_module("m_mr", f"{RUN}/MaskRCNN/train_maskrcnn.py")
model = mod_mr.build_model().to(device)
ckpt = torch.load(f"{CKPT_ROOT}/MaskRCNN/checkpoints/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
x = T.ToTensor()(cv2.resize(img_rgb, (IMG, IMG))).to(device)
with torch.no_grad():
    out = model([x])[0]
best_iou, best_mask = 0.0, np.zeros((H0, W0), bool)
for i in range(len(out["scores"])):
    if int(out["labels"][i]) != 7 or float(out["scores"][i]) < 0.5:
        continue
    m512 = (out["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
    m_native = cv2.resize(m512, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
    cur_iou = iou(m_native, gt_mask.astype(bool))
    if cur_iou > best_iou:
        best_iou, best_mask = cur_iou, m_native
results["MaskRCNN"] = best_iou if best_mask.any() else 0.0
overlays.append(make_overlay(best_mask, f"Mask R-CNN (IoU={results['MaskRCNN']:.3f})"))
print("MaskRCNN IoU:", results["MaskRCNN"])

# ---------------------------------------------------------------------------
# SOLOv2 -- connected-component pseudo-instance, best match
# ---------------------------------------------------------------------------
mod_sv = load_module("m_sv", f"{RUN}/SoloV2/train_solov2_clean.py")
model = mod_sv.SimplifiedSOLOv2(num_classes=8).to(device)
ckpt = torch.load(f"{CKPT_ROOT}/SoloV2/checkpoints_clean/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
transform = T.Compose([T.ToPILImage(), T.Resize((IMG, IMG)), T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
x = transform(img_rgb).unsqueeze(0).to(device)
with torch.no_grad():
    category_pred, mask_pred = model(x)
    category_probs = torch.softmax(category_pred, dim=1)
    objectness = torch.sigmoid(mask_pred)
    category_probs_full = torch.nn.functional.interpolate(category_probs, size=(H0, W0), mode="bilinear", align_corners=False)[0]
    objectness_full = torch.nn.functional.interpolate(objectness, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
    class_conf_full, class_ids_full = category_probs_full.max(dim=0)
    fg_full = (class_ids_full > 0) & (objectness_full > 0.5)
    class_ids_np = class_ids_full.cpu().numpy()
    fg_np = fg_full.cpu().numpy().astype(np.uint8)
cls_mask = ((class_ids_np == 7) & (fg_np > 0)).astype(np.uint8)
best_iou, best_mask = 0.0, np.zeros((H0, W0), bool)
if cls_mask.sum() > 0:
    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < 150:
            continue
        comp_mask = labels_cc == lbl
        cur_iou = iou(comp_mask, gt_mask.astype(bool))
        if cur_iou > best_iou:
            best_iou, best_mask = cur_iou, comp_mask
results["SoloV2"] = best_iou if best_mask.any() else 0.0
overlays.append(make_overlay(best_mask, f"SOLOv2 (IoU={results['SoloV2']:.3f})"))
print("SoloV2 IoU:", results["SoloV2"])

# ---------------------------------------------------------------------------
# YOLO11-seg / YOLO11-PGSS -- best-matching predicted Wisteria instance
# ---------------------------------------------------------------------------
def yolo_best_wisteria(weights_path, extra_module_dir=None):
    if extra_module_dir:
        sys.path.insert(0, extra_module_dir)
        import pgss_modules  # noqa: F401
    from ultralytics import YOLO
    model = YOLO(weights_path)
    yolo_img_path = os.path.join(YOLO_TEST_IMG_DIR, TARGET_FILE)
    r = model.predict(yolo_img_path, imgsz=IMG, conf=0.25, device=(0 if torch.cuda.is_available() else "cpu"),
                       verbose=False)[0]
    best_iou, best_mask = 0.0, np.zeros((H0, W0), bool)
    if r.masks is not None and len(r.masks) > 0:
        polys = r.masks.xy
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for idx in range(len(clss)):
            if clss[idx] + 1 != 7:  # YOLO class 6 -> COCO cat id 7 (Wisteria)
                continue
            m = np.zeros((H0, W0), np.uint8)
            poly = polys[idx]
            if poly.shape[0] < 3:
                continue
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            m = m.astype(bool)
            cur_iou = iou(m, gt_mask.astype(bool))
            if cur_iou > best_iou:
                best_iou, best_mask = cur_iou, m
    if extra_module_dir:
        sys.path.pop(0)
    return best_iou, best_mask


best_iou, best_mask = yolo_best_wisteria(f"{CKPT_ROOT}/YOLO11seg/runs/vine_clean_v6/weights/best_dice.pt")
results["YOLO11-seg"] = best_iou if best_mask.any() else 0.0
overlays.append(make_overlay(best_mask, f"YOLO11-seg (IoU={results['YOLO11-seg']:.3f})"))
print("YOLO11-seg IoU:", results["YOLO11-seg"])

best_iou, best_mask = yolo_best_wisteria(f"{CKPT_ROOT}/PGSS_YOLO11/runs/pgss_vine_v6/weights/best_dice.pt",
                                          extra_module_dir=f"{RUN}/PGSS_YOLO11")
results["YOLO11-PGSS"] = best_iou if best_mask.any() else 0.0
overlays.append(make_overlay(best_mask, f"YOLO11-PGSS (IoU={results['YOLO11-PGSS']:.3f})"))
print("YOLO11-PGSS IoU:", results["YOLO11-PGSS"])

# ---------------------------------------------------------------------------
# Save individual + combined figures
# ---------------------------------------------------------------------------
for title, (overlay_img, full_title) in zip(results.keys(), overlays):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay_img)
    ax.set_title(full_title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    safe = title.replace(" ", "_")
    out = f"{RUN}/fig4_qualitative_{safe}.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)

# original + GT panel
fig, ax = plt.subplots(figsize=(7, 7))
gt_overlay = img_rgb.copy().astype(np.float32)
color = np.zeros_like(gt_overlay); color[gt_mask.astype(bool)] = [0, 255, 0]
gt_overlay[gt_mask.astype(bool)] = gt_overlay[gt_mask.astype(bool)] * 0.45 + color[gt_mask.astype(bool)] * 0.55
ax.imshow(gt_overlay.astype(np.uint8))
ax.set_title(f"Original + Ground Truth\n{TARGET_FILE}", fontsize=10)
ax.axis("off")
plt.tight_layout()
plt.savefig(f"{RUN}/fig4_qualitative_GroundTruth.png", dpi=200)
plt.close(fig)

# combined 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()
axes[0].imshow(gt_overlay.astype(np.uint8))
axes[0].set_title("Original + Ground Truth", fontsize=11)
axes[0].axis("off")
for i, (title, (overlay_img, full_title)) in enumerate(zip(results.keys(), overlays)):
    ax = axes[i + 1]
    ax.imshow(overlay_img)
    ax.set_title(full_title, fontsize=11)
    ax.axis("off")
axes[-1].axis("off")
fig.suptitle(f"Fig. 4 — Qualitative comparison, seed 42, test image: {TARGET_FILE}\n"
             "(yellow=correct overlap, red=false positive, green=missed ground truth)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{RUN}/fig4_qualitative_combined.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("wrote combined figure")

with open(f"{RUN}/fig4_iou_summary.json", "w") as f:
    json.dump({"target_image": TARGET_FILE, "image_size": [W0, H0],
               "gt_wisteria_pixels": int(gt_mask.sum()), "per_model_iou": results}, f, indent=2)
import csv
with open(f"{RUN}/fig4_iou_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "wisteria_iou_this_image"])
    for k, v in results.items():
        w.writerow([k, v])
print("\nFinal IoUs:", json.dumps(results, indent=2))
