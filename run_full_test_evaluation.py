"""
Full test-set evaluation for all seven trained vine-segmentation models.

Semantic metrics (all 7 models): macro-averaged precision/recall/F1/mIoU across the
8 classes (background + 7 species), plus per-class IoU for each species, computed from
a pixel-level confusion matrix on the untouched `rebuilt/final/test` (or `final_yolo/test`)
split. Ground truth is rasterized directly from the COCO polygon annotations at each
image's native resolution, then both GT and prediction are resized to the model's
512x512 working grid with nearest-neighbor interpolation before accumulating the
confusion matrix -- this matches the methodology already used by this project's own
per-model evaluate scripts (unet_evaluate.py, maskrcnn_evaluate.py, yolo_evaluate.py, etc.),
just re-pointed at the test split and switched from weighted- to macro-averaging.

Instance metrics (Mask R-CNN, SOLOv2, YOLO11-seg, PGSS-YOLO11): COCO-style mask AP /
AP50 / AP75 / AR via pycocotools, against the same test-split GT polygons.

Checkpoint selection: every PyTorch model (UNet, DeepLabV3, SegFormer, Mask R-CNN, SOLOv2)
saves best_model.pth by best validation Dice/mIoU. For the two Ultralytics YOLO models we
use best_dice.pt (not Ultralytics' own best.pt) so the selection criterion is the same
val-Dice-based rule across all seven models, per train_pgss.py's own stated rationale.

Run: python3 run_full_test_evaluation.py 2>&1 | tee evaluation_run.log
Output: evaluation_results.json (full detail) and evaluation_results.csv (flat summary table).
"""
import os
import sys
import json
import time
import importlib.util

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms as T
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

RUN = "/root/vine-project/Run"
CKPT_ROOT = "/workspace/vine-project-Run"
COCO_TEST_DIR = f"{RUN}/rebuilt/final/test"
COCO_TEST_JSON = f"{COCO_TEST_DIR}/_annotations.coco.json"
YOLO_TEST_IMG_DIR = f"{RUN}/rebuilt/final_yolo/test/images"

IMG = 512
CLASS_NAMES = ["background", "Akebia", "Campsis", "Hydrangea", "Lonicera",
               "Madame Galen", "Parthenocissus", "Wisteria"]
SPECIES = CLASS_NAMES[1:]
CAT_IDS = list(range(1, 8))  # COCO category ids 1..7, matches CLASS_NAMES[1:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = {}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared ground-truth / metric helpers
# ---------------------------------------------------------------------------
def gt_pixel_map_native(coco, iid):
    info = coco.imgs[iid]
    H, W = info["height"], info["width"]
    canvas = np.zeros((H, W), np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=iid)):
        cid = ann["category_id"]
        if cid == 0 or not isinstance(ann["segmentation"], list):
            continue
        for poly in ann["segmentation"]:
            pts = np.array(poly, np.float32).reshape(-1, 2).astype(np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(canvas, [pts], int(cid))
    return canvas


def resize_map(canvas, size=IMG):
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)


def macro_metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return precision, recall, f1, iou


def semantic_summary(cm, model_name, checkpoint_used, test_set, n_images, notes=None):
    precision, recall, f1, iou = macro_metrics_from_cm(cm)
    out = {
        "model": model_name,
        "checkpoint_used": checkpoint_used,
        "test_set": test_set,
        "n_test_images": n_images,
        "test_set_used_in_training_or_validation": False,
        "semantic_metrics": {
            "macro_precision_8class": float(np.mean(precision)),
            "macro_recall_8class": float(np.mean(recall)),
            "macro_f1_8class": float(np.mean(f1)),
            "mIoU_8class": float(np.mean(iou)),
            "per_class_precision": {CLASS_NAMES[i]: float(precision[i]) for i in range(8)},
            "per_class_recall": {CLASS_NAMES[i]: float(recall[i]) for i in range(8)},
            "per_class_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(8)},
            "per_class_iou_species": {SPECIES[i]: float(iou[i + 1]) for i in range(7)},
            "background_iou": float(iou[0]),
        },
        "confusion_matrix_rows_gt_cols_pred": cm.tolist(),
    }
    if notes:
        out["notes"] = notes
    return out


def coco_instance_eval(coco_gt, predictions, conf_thresh, iou_thresh_matching_note):
    result = {
        "confidence_threshold": conf_thresh,
        "iou_matching_thresholds_evaluated": "COCO standard: 0.50:0.05:0.95 (AP), 0.50 (AP50), 0.75 (AP75)",
        "note": iou_thresh_matching_note,
    }
    if len(predictions) == 0:
        result.update({"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR_max100": 0.0,
                        "warning": "No predictions above confidence threshold on the test set."})
        return result
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, iouType="segm")
    ev.params.catIds = CAT_IDS
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    result.update({
        "AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
        "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
        "AR_max1": float(s[6]), "AR_max10": float(s[7]), "AR_max100": float(s[8]),
        "AR_small": float(s[9]), "AR_medium": float(s[10]), "AR_large": float(s[11]),
    })
    return result


def mask_to_rle(mask_bool):
    rle = coco_mask.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def xyxy_to_xywh(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


coco_test = COCO(COCO_TEST_JSON)
_smoke_n = os.environ.get("SMOKE_N")
if _smoke_n:
    _keep = list(coco_test.imgs.keys())[: int(_smoke_n)]
    coco_test.imgs = {k: coco_test.imgs[k] for k in _keep}
    coco_test.dataset["images"] = [im for im in coco_test.dataset["images"] if im["id"] in _keep]
    coco_test.dataset["annotations"] = [a for a in coco_test.dataset["annotations"] if a["image_id"] in _keep]
    coco_test.createIndex()
    log(f"SMOKE TEST MODE: trimmed to {len(coco_test.imgs)} images")
N_TEST = len(coco_test.imgs)
log(f"Loaded COCO test set: {N_TEST} images, {len(coco_test.anns)} annotations")

# Pre-cache native-res GT pixel maps and resized-512 GT maps (reused by several models)
GT_NATIVE = {}
GT_512 = {}
for iid in coco_test.imgs:
    g = gt_pixel_map_native(coco_test, iid)
    GT_NATIVE[iid] = g
    GT_512[iid] = resize_map(g)


# ===========================================================================
# 1. UNet -- semantic only
# ===========================================================================
def eval_unet():
    log("=== UNet ===")
    mod = load_module("train_unet_mod", f"{RUN}/UNet/train_unet.py")
    model = mod.UNet(3, 8).to(device)
    ckpt_path = f"{CKPT_ROOT}/UNet/checkpoints/best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    to_tensor = T.ToTensor()

    all_t, all_p = [], []
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            img = Image.open(os.path.join(COCO_TEST_DIR, info["file_name"])).convert("RGB")
            img512 = img.resize((IMG, IMG), Image.BILINEAR)
            x = to_tensor(img512).unsqueeze(0).to(device)
            pred = torch.argmax(model(x), dim=1)[0].cpu().numpy().astype(np.uint8)
            all_t.append(GT_512[iid].reshape(-1))
            all_p.append(pred.reshape(-1))
    cm = sk_confusion_matrix(np.concatenate(all_t), np.concatenate(all_p), labels=list(range(8)))
    out = semantic_summary(cm, "UNet", f"{ckpt_path} (epoch {ckpt['epoch']+1}, "
                            f"val_miou={ckpt['val_miou']:.4f}, val_dice={ckpt['val_dice']:.4f})",
                            "rebuilt/final/test", N_TEST)
    out["instance_metrics"] = {
        "supported": False,
        "reason": ("UNet is a pure semantic-segmentation architecture: it outputs one "
                   "per-pixel class map with no separate instance identities, confidence "
                   "scores, or object proposals. COCO-style instance AP/AP50/AP75/AR is "
                   "not defined for this output and was not computed. This is an "
                   "architectural limitation, stated here explicitly rather than skipped silently."),
    }
    RESULTS["UNet"] = out
    log(f"UNet done: mIoU={out['semantic_metrics']['mIoU_8class']:.4f}")


# ===========================================================================
# 2. DeepLabV3 -- semantic only
# ===========================================================================
def eval_deeplabv3():
    log("=== DeepLabV3 ===")
    mod = load_module("train_deeplabv3_mod", f"{RUN}/DeepLabV3/train_deeplabv3.py")
    model = mod.build_model().to(device)
    ckpt_path = f"{CKPT_ROOT}/DeepLabV3/checkpoints/best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    to_tensor = T.ToTensor()

    all_t, all_p = [], []
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            img = Image.open(os.path.join(COCO_TEST_DIR, info["file_name"])).convert("RGB")
            img512 = img.resize((IMG, IMG), Image.BILINEAR)
            x = to_tensor(img512).unsqueeze(0).to(device)
            logits = model(x)["out"]
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            all_t.append(GT_512[iid].reshape(-1))
            all_p.append(pred.reshape(-1))
    cm = sk_confusion_matrix(np.concatenate(all_t), np.concatenate(all_p), labels=list(range(8)))
    out = semantic_summary(cm, "DeepLabV3", f"{ckpt_path} (epoch {ckpt['epoch']+1}, "
                            f"val_miou={ckpt['val_miou']:.4f}, val_dice={ckpt['val_dice']:.4f})",
                            "rebuilt/final/test", N_TEST)
    out["instance_metrics"] = {
        "supported": False,
        "reason": ("DeepLabV3 is a pure semantic-segmentation architecture: it outputs one "
                   "per-pixel class map with no separate instance identities, confidence "
                   "scores, or object proposals. COCO-style instance AP/AP50/AP75/AR is "
                   "not defined for this output and was not computed. This is an "
                   "architectural limitation, stated here explicitly rather than skipped silently."),
    }
    RESULTS["DeepLabV3"] = out
    log(f"DeepLabV3 done: mIoU={out['semantic_metrics']['mIoU_8class']:.4f}")


# ===========================================================================
# 3. SegFormer -- semantic only
# ===========================================================================
def eval_segformer():
    log("=== SegFormer ===")
    mod = load_module("train_segformer_mod", f"{RUN}/SegFormer/train_segformer.py")
    model = mod.build_model().to(device)
    ckpt_path = f"{CKPT_ROOT}/SegFormer/checkpoints/best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    to_tensor = T.ToTensor()
    norm = mod.NORM

    all_t, all_p = [], []
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            img = Image.open(os.path.join(COCO_TEST_DIR, info["file_name"])).convert("RGB")
            img512 = img.resize((IMG, IMG), Image.BILINEAR)
            x = norm(to_tensor(img512)).unsqueeze(0).to(device)
            logits = mod.forward_logits(model, x)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            all_t.append(GT_512[iid].reshape(-1))
            all_p.append(pred.reshape(-1))
    cm = sk_confusion_matrix(np.concatenate(all_t), np.concatenate(all_p), labels=list(range(8)))
    out = semantic_summary(cm, "SegFormer", f"{ckpt_path} (epoch {ckpt['epoch']+1}, "
                            f"val_miou={ckpt['val_miou']:.4f}, val_dice={ckpt['val_dice']:.4f})",
                            "rebuilt/final/test", N_TEST)
    out["instance_metrics"] = {
        "supported": False,
        "reason": ("SegFormer is a pure semantic-segmentation (transformer encoder + MLP "
                   "decode head) architecture: it outputs one per-pixel class map with no "
                   "separate instance identities, confidence scores, or object proposals. "
                   "COCO-style instance AP/AP50/AP75/AR is not defined for this output and "
                   "was not computed. This is an architectural limitation, stated here "
                   "explicitly rather than skipped silently."),
    }
    RESULTS["SegFormer"] = out
    log(f"SegFormer done: mIoU={out['semantic_metrics']['mIoU_8class']:.4f}")


# ===========================================================================
# 4. Mask R-CNN -- instance + semantic (rasterized)
# ===========================================================================
def eval_maskrcnn():
    log("=== Mask R-CNN ===")
    mod = load_module("train_maskrcnn_mod", f"{RUN}/MaskRCNN/train_maskrcnn.py")
    model = mod.build_model().to(device)
    ckpt_path = f"{CKPT_ROOT}/MaskRCNN/checkpoints/best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    to_tensor = T.ToTensor()
    CONF = 0.5

    all_t, all_p = [], []
    coco_predictions = []
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            path = os.path.join(COCO_TEST_DIR, info["file_name"])
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            H0, W0 = img.shape[:2]
            x = to_tensor(cv2.resize(img, (IMG, IMG))).to(device)
            out = model([x])[0]
            class_map = np.zeros((IMG, IMG), np.uint8)
            order = torch.argsort(out["scores"])
            sx, sy = W0 / IMG, H0 / IMG
            for i in order.tolist():
                score = float(out["scores"][i])
                if score < CONF:
                    continue
                m512 = (out["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
                if m512.sum() == 0:
                    continue
                lbl = int(out["labels"][i])
                class_map[m512 > 0] = lbl
                # native-res mask + box for COCO instance eval
                m_native = cv2.resize(m512, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
                if m_native.sum() == 0:
                    continue
                box = out["boxes"][i].cpu().numpy()
                x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
                coco_predictions.append({
                    "image_id": iid, "category_id": lbl, "segmentation": mask_to_rle(m_native),
                    "bbox": xyxy_to_xywh(x1, y1, x2, y2), "score": score,
                })
            all_t.append(GT_512[iid].reshape(-1))
            all_p.append(resize_map(class_map).reshape(-1))
    cm = sk_confusion_matrix(np.concatenate(all_t), np.concatenate(all_p), labels=list(range(8)))
    out = semantic_summary(cm, "Mask R-CNN", f"{ckpt_path} (epoch {ckpt['epoch']+1}, "
                            f"val_loss={ckpt['val_loss']:.4f}, val_dice={ckpt['val_dice']:.4f})",
                            "rebuilt/final/test", N_TEST,
                            notes="Semantic metrics computed by rasterizing instance predictions "
                                  f"(conf>={CONF}) to a per-pixel class map, highest-confidence "
                                  "instance painted last on overlaps.")
    out["instance_metrics"] = coco_instance_eval(
        coco_test, coco_predictions, conf_thresh=CONF,
        iou_thresh_matching_note="pycocotools COCOeval, iouType='segm', native per-instance predicted masks vs GT polygons.")
    RESULTS["MaskRCNN"] = out
    log(f"Mask R-CNN done: mIoU={out['semantic_metrics']['mIoU_8class']:.4f}, "
        f"AP={out['instance_metrics'].get('AP')}, n_preds={len(coco_predictions)}")


# ===========================================================================
# 5. SOLOv2 (SimplifiedSOLOv2) -- semantic + connected-component pseudo-instances
# ===========================================================================
def eval_solov2():
    log("=== SOLOv2 ===")
    mod = load_module("train_solov2_clean_mod", f"{RUN}/SoloV2/train_solov2_clean.py")
    model = mod.SimplifiedSOLOv2(num_classes=8).to(device)
    ckpt_path = f"{CKPT_ROOT}/SoloV2/checkpoints_clean/best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = T.Compose([
        T.ToPILImage(), T.Resize((IMG, IMG)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    CONF = 0.5
    MIN_AREA = 150

    all_t, all_p = [], []
    coco_predictions = []
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            path = os.path.join(COCO_TEST_DIR, info["file_name"])
            img_bgr = cv2.imread(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H0, W0 = img_rgb.shape[:2]
            x = transform(img_rgb).unsqueeze(0).to(device)
            category_pred, mask_pred = model(x)
            category_probs = torch.softmax(category_pred, dim=1)
            objectness = torch.sigmoid(mask_pred)

            # Native grid semantic map for the confusion matrix: plain argmax over the category
            # head, matching this repo's own evaluate_metrics() convention exactly (no
            # objectness gating there -- background is class 0 of the category head itself).
            # The category head has no decoder, so its output stays at the ResNet50 stride
            # (512/32 = 16x16) -- evaluate_metrics() downsamples GT to that coarse grid rather
            # than upsampling the prediction, and we match that exactly here.
            Hc, Wc = category_pred.shape[-2:]
            pred_coarse = torch.argmax(category_pred, dim=1)[0].cpu().numpy().astype(np.uint8)
            gt_coarse = cv2.resize(GT_512[iid], (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            all_t.append(gt_coarse.reshape(-1))
            all_p.append(pred_coarse.reshape(-1))

            # native-res upsample for connected-component pseudo-instances (COCO eval)
            category_probs_full = torch.nn.functional.interpolate(
                category_probs, size=(H0, W0), mode="bilinear", align_corners=False)[0]
            objectness_full = torch.nn.functional.interpolate(
                objectness, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
            class_conf_full, class_ids_full = category_probs_full.max(dim=0)
            fg_full = (class_ids_full > 0) & (objectness_full > CONF)
            class_ids_np = class_ids_full.cpu().numpy()
            class_conf_np = class_conf_full.cpu().numpy()
            fg_np = fg_full.cpu().numpy().astype(np.uint8)

            for cls_idx in np.unique(class_ids_np):
                if cls_idx == 0:
                    continue
                cls_mask = ((class_ids_np == cls_idx) & (fg_np > 0)).astype(np.uint8)
                if cls_mask.sum() == 0:
                    continue
                num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
                for lbl in range(1, num_labels):
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    if area < MIN_AREA:
                        continue
                    comp_mask = (labels_cc == lbl)
                    score = float(class_conf_np[comp_mask].mean())
                    ys, xs = np.where(comp_mask)
                    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                    coco_predictions.append({
                        "image_id": iid, "category_id": int(cls_idx),
                        "segmentation": mask_to_rle(comp_mask),
                        "bbox": xyxy_to_xywh(x1, y1, x2 + 1, y2 + 1), "score": score,
                    })
    cm = sk_confusion_matrix(np.concatenate(all_t), np.concatenate(all_p), labels=list(range(8)))
    out = semantic_summary(cm, "SOLOv2", f"{ckpt_path} (epoch {ckpt['epoch']+1}, "
                            f"val_loss={ckpt['val_loss']:.4f}, val_dice={ckpt['val_dice']:.4f})",
                            "rebuilt/final/test", N_TEST,
                            notes="This 'SimplifiedSOLOv2' head predicts one shared per-pixel "
                                  "category map plus a single class-agnostic objectness mask per "
                                  "image -- it does NOT predict per-instance mask proposals the way "
                                  "real SOLOv2 or Mask R-CNN do. Semantic metrics are computed on "
                                  "the category head's native 16x16 grid (ResNet50 stride 32, no "
                                  "decoder) with GT downsampled to match, exactly as this repo's "
                                  "own evaluate_metrics() does -- NOT at the 512x512 resolution "
                                  "used for the other six models, so SOLOv2's mIoU/precision/"
                                  "recall/F1 are not directly resolution-comparable to the rest. "
                                  "Instance metrics below are "
                                  "*derived* pseudo-instances (connected components of the "
                                  f"class map, area>={MIN_AREA}px, confidence=mean class prob over "
                                  "the component), matching the visualization convention already "
                                  "used in train_solov2_clean.py's plot_sample_prediction_comparison. "
                                  "Treat SOLOv2 instance AP as a weaker, derived signal, not a "
                                  "native per-instance prediction like Mask R-CNN/YOLO11.")
    out["instance_metrics"] = coco_instance_eval(
        coco_test, coco_predictions, conf_thresh=CONF,
        iou_thresh_matching_note="pycocotools COCOeval, iouType='segm', on connected-component "
                                  "pseudo-instances (see notes) vs GT polygons.")
    RESULTS["SoloV2"] = out
    log(f"SOLOv2 done: mIoU={out['semantic_metrics']['mIoU_8class']:.4f}, "
        f"AP={out['instance_metrics'].get('AP')}, n_preds={len(coco_predictions)}")


# ===========================================================================
# 6/7. YOLO11-seg and PGSS-YOLO11 -- native instance segmentation
# ===========================================================================
def eval_yolo_model(name, weights_path, extra_module_dir=None):
    log(f"=== {name} ===")
    if extra_module_dir:
        sys.path.insert(0, extra_module_dir)
        import pgss_modules  # noqa: F401  registers GLSA/AWFF/DEConv before loading the .pt
    from ultralytics import YOLO
    model = YOLO(weights_path)
    CONF = 0.25
    yolo_device = 0 if torch.cuda.is_available() else "cpu"

    all_t, all_p = [], []
    coco_predictions = []
    for iid, info in coco_test.imgs.items():
        fname = info["file_name"]
        img_path = os.path.join(YOLO_TEST_IMG_DIR, fname)
        r = model.predict(img_path, imgsz=IMG, conf=CONF, device=yolo_device, verbose=False)[0]
        H0, W0 = r.orig_shape
        canvas512 = np.zeros((IMG, IMG), np.uint8)
        if r.masks is not None and len(r.masks) > 0:
            polys = r.masks.xy
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            order = np.argsort(confs)
            canvas_native_paint = np.zeros((H0, W0), np.uint8)
            for idx in order:
                poly = polys[idx]
                if poly.shape[0] < 3:
                    continue
                cat_id = int(clss[idx]) + 1  # YOLO 0..6 -> COCO category id 1..7
                cv2.fillPoly(canvas_native_paint, [poly.astype(np.int32)], cat_id)
                m_native = np.zeros((H0, W0), np.uint8)
                cv2.fillPoly(m_native, [poly.astype(np.int32)], 1)
                m_native = m_native.astype(bool)
                if m_native.sum() == 0:
                    continue
                x1, y1, x2, y2 = xyxy[idx]
                coco_predictions.append({
                    "image_id": iid, "category_id": cat_id, "segmentation": mask_to_rle(m_native),
                    "bbox": xyxy_to_xywh(x1, y1, x2, y2), "score": float(confs[idx]),
                })
            canvas512 = resize_map(canvas_native_paint)
        all_t.append(GT_512[iid].reshape(-1))
        all_p.append(canvas512.reshape(-1))

    cm = sk_confusion_matrix(np.concatenate(all_t), np.concatenate(all_p), labels=list(range(8)))
    out = semantic_summary(cm, name, f"{weights_path} (val-Dice-selected checkpoint, "
                            "see dice_early_stop_log.json)", "rebuilt/final_yolo/test", N_TEST,
                            notes="Semantic metrics computed by rasterizing native YOLO instance "
                                  f"masks (conf>={CONF}) to a per-pixel class map; ground truth is "
                                  "the same COCO polygon annotations used for the other models "
                                  "(rebuilt/final/test and rebuilt/final_yolo/test contain the "
                                  "identical 169 test images).")
    out["instance_metrics"] = coco_instance_eval(
        coco_test, coco_predictions, conf_thresh=CONF,
        iou_thresh_matching_note="pycocotools COCOeval, iouType='segm', native per-instance masks "
                                  "(Ultralytics NMS, IoU=0.7) vs GT polygons.")
    RESULTS[name] = out
    log(f"{name} done: mIoU={out['semantic_metrics']['mIoU_8class']:.4f}, "
        f"AP={out['instance_metrics'].get('AP')}, n_preds={len(coco_predictions)}")
    if extra_module_dir:
        sys.path.pop(0)


# ===========================================================================
if __name__ == "__main__":
    t0 = time.time()
    steps = [
        ("UNet", eval_unet),
        ("DeepLabV3", eval_deeplabv3),
        ("SegFormer", eval_segformer),
        ("MaskRCNN", eval_maskrcnn),
        ("SoloV2", eval_solov2),
        ("YOLO11seg", lambda: eval_yolo_model(
            "YOLO11-seg", f"{CKPT_ROOT}/YOLO11seg/runs/vine_clean_v6/weights/best_dice.pt")),
        ("PGSS_YOLO11", lambda: eval_yolo_model(
            "YOLO11-PGSS", f"{CKPT_ROOT}/PGSS_YOLO11/runs/pgss_vine_v6/weights/best_dice.pt",
            extra_module_dir=f"{RUN}/PGSS_YOLO11")),
    ]
    for step_name, fn in steps:
        try:
            fn()
        except Exception as e:
            log(f"!!! {step_name} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            RESULTS[step_name] = {"model": step_name, "error": str(e)}
        # checkpoint progress after every model in case of a crash later on
        with open(f"{RUN}/evaluation_results.json", "w") as f:
            json.dump(RESULTS, f, indent=2)

    # ---- flat CSV summary ----
    import csv
    with open(f"{RUN}/evaluation_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "checkpoint_used", "test_set", "n_test_images",
                  "macro_precision", "macro_recall", "macro_f1", "mIoU_8class"]
        header += [f"IoU_{s}" for s in SPECIES]
        header += ["instance_AP", "instance_AP50", "instance_AP75", "instance_AR_max100",
                   "instance_conf_thresh", "instance_supported"]
        w.writerow(header)
        for name, r in RESULTS.items():
            if "error" in r:
                w.writerow([name, "ERROR: " + r["error"]] + [""] * (len(header) - 2))
                continue
            sm = r["semantic_metrics"]
            im = r.get("instance_metrics", {})
            row = [r["model"], r["checkpoint_used"], r["test_set"], r["n_test_images"],
                   sm["macro_precision_8class"], sm["macro_recall_8class"], sm["macro_f1_8class"],
                   sm["mIoU_8class"]]
            row += [sm["per_class_iou_species"][s] for s in SPECIES]
            row += [im.get("AP", ""), im.get("AP50", ""), im.get("AP75", ""), im.get("AR_max100", ""),
                    im.get("confidence_threshold", ""), im.get("supported", True)]
            w.writerow(row)

    log(f"ALL DONE in {time.time()-t0:.1f}s. Wrote evaluation_results.json and evaluation_results.csv")
