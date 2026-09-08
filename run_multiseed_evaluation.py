"""
Multi-seed (42, 43, 44) test-set evaluation for all seven trained vine-segmentation models.

Same methodology as run_full_test_evaluation.py (seed 42's original run), just parameterized
over checkpoint path per seed and looped 3x. Seed 42's per-model results are read back from
the already-computed evaluation_results.json (identical script/methodology) rather than
recomputed, to save ~4 minutes; seeds 43/44 are computed fresh here.

Checkpoint paths per seed:
  UNet/DeepLabV3/SegFormer/MaskRCNN: <CKPT_ROOT>/<Model>/checkpoints[_seed43|_seed44]/best_model.pth
  SOLOv2:                            <CKPT_ROOT>/SoloV2/checkpoints_clean[_seed43|_seed44]/best_model.pth
  YOLO11-seg/PGSS:                   <CKPT_ROOT>/<Model>/runs/<run_name>[_seed43|_seed44]/weights/best_dice.pt

Output:
  evaluation_results_seed43.json / _seed44.json  -- per-seed full detail (same schema as seed 42's)
  evaluation_multiseed.json  -- {model: {per_seed: {42:..,43:..,44:..}, aggregate: {mean,std,...}}}
  evaluation_multiseed_summary.csv  -- flat mean+-std table, one row per model
  evaluation_multiseed_per_seed.csv -- flat table, one row per (model, seed)
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
CAT_IDS = list(range(1, 8))
SEEDS = [42, 43, 44]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

GT_NATIVE = {}
GT_512 = {}
for iid in coco_test.imgs:
    g = gt_pixel_map_native(coco_test, iid)
    GT_NATIVE[iid] = g
    GT_512[iid] = resize_map(g)


def ckpt_suffix(seed):
    return "" if seed == 42 else f"_seed{seed}"


# ===========================================================================
def eval_unet(ckpt_path):
    mod = load_module("train_unet_mod", f"{RUN}/UNet/train_unet.py")
    model = mod.UNet(3, 8).to(device)
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
    out["instance_metrics"] = {"supported": False, "reason": "UNet is a pure semantic-segmentation "
                                "architecture with no separate instance identities/scores/proposals; "
                                "COCO instance AP/AR is not defined for it."}
    return out


def eval_deeplabv3(ckpt_path):
    mod = load_module("train_deeplabv3_mod", f"{RUN}/DeepLabV3/train_deeplabv3.py")
    model = mod.build_model().to(device)
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
    out["instance_metrics"] = {"supported": False, "reason": "DeepLabV3 is a pure semantic-segmentation "
                                "architecture with no separate instance identities/scores/proposals; "
                                "COCO instance AP/AR is not defined for it."}
    return out


def eval_segformer(ckpt_path):
    mod = load_module("train_segformer_mod", f"{RUN}/SegFormer/train_segformer.py")
    model = mod.build_model().to(device)
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
    out["instance_metrics"] = {"supported": False, "reason": "SegFormer is a pure semantic-segmentation "
                                "architecture with no separate instance identities/scores/proposals; "
                                "COCO instance AP/AR is not defined for it."}
    return out


def eval_maskrcnn(ckpt_path):
    mod = load_module("train_maskrcnn_mod", f"{RUN}/MaskRCNN/train_maskrcnn.py")
    model = mod.build_model().to(device)
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
                            "rebuilt/final/test", N_TEST)
    out["instance_metrics"] = coco_instance_eval(
        coco_test, coco_predictions, conf_thresh=CONF,
        iou_thresh_matching_note="pycocotools COCOeval, iouType='segm', native per-instance masks.")
    return out


def eval_solov2(ckpt_path):
    mod = load_module("train_solov2_clean_mod", f"{RUN}/SoloV2/train_solov2_clean.py")
    model = mod.SimplifiedSOLOv2(num_classes=8).to(device)
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

            Hc, Wc = category_pred.shape[-2:]
            pred_coarse = torch.argmax(category_pred, dim=1)[0].cpu().numpy().astype(np.uint8)
            gt_coarse = cv2.resize(GT_512[iid], (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            all_t.append(gt_coarse.reshape(-1))
            all_p.append(pred_coarse.reshape(-1))

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
                            notes="Semantic metrics on the native 16x16 category-head grid (no "
                                  "decoder); GT downsampled to match. Not resolution-comparable to "
                                  "the other six models' 512x512 metrics. Instance metrics are "
                                  "derived connected-component pseudo-instances, not native.")
    out["instance_metrics"] = coco_instance_eval(
        coco_test, coco_predictions, conf_thresh=CONF,
        iou_thresh_matching_note="pycocotools COCOeval, iouType='segm', connected-component pseudo-instances.")
    return out


def eval_yolo_model(name, weights_path, extra_module_dir=None):
    if extra_module_dir:
        sys.path.insert(0, extra_module_dir)
        import pgss_modules  # noqa: F401
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
                cat_id = int(clss[idx]) + 1
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
    out = semantic_summary(cm, name, f"{weights_path} (val-Dice-selected checkpoint)",
                            "rebuilt/final_yolo/test", N_TEST,
                            notes="Semantic metrics from rasterized native YOLO instance masks.")
    out["instance_metrics"] = coco_instance_eval(
        coco_test, coco_predictions, conf_thresh=CONF,
        iou_thresh_matching_note="pycocotools COCOeval, iouType='segm', native per-instance masks (NMS IoU=0.7).")
    if extra_module_dir:
        sys.path.pop(0)
    return out


# ===========================================================================
# Aggregation helpers
# ===========================================================================
def aggregate_model(per_seed_results):
    """per_seed_results: {seed: result_dict}. Returns aggregate dict with mean/std."""
    seeds_sorted = sorted(per_seed_results.keys())
    sm_keys = ["macro_precision_8class", "macro_recall_8class", "macro_f1_8class", "mIoU_8class"]
    agg = {"semantic_metrics": {}}

    for k in sm_keys:
        vals = np.array([per_seed_results[s]["semantic_metrics"][k] for s in seeds_sorted])
        agg["semantic_metrics"][k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                                       "per_seed": {int(s): float(v) for s, v in zip(seeds_sorted, vals)}}

    for cls in CLASS_NAMES:
        vals = np.array([per_seed_results[s]["semantic_metrics"]["per_class_iou_species"].get(cls,
                         per_seed_results[s]["semantic_metrics"]["background_iou"] if cls == "background" else None)
                         for s in seeds_sorted])
        key = "background_iou" if cls == "background" else f"IoU_{cls}"
        agg["semantic_metrics"][key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                                         "per_seed": {int(s): float(v) for s, v in zip(seeds_sorted, vals)}}

    # confusion matrices: elementwise mean across seeds (primary), plus identify median-mIoU seed
    cms = [np.array(per_seed_results[s]["confusion_matrix_rows_gt_cols_pred"]) for s in seeds_sorted]
    mean_cm = np.mean(cms, axis=0)
    mious = [per_seed_results[s]["semantic_metrics"]["mIoU_8class"] for s in seeds_sorted]
    median_seed = seeds_sorted[int(np.argsort(mious)[len(mious) // 2])]
    agg["confusion_matrix"] = {
        "method": "elementwise_mean_across_seeds",
        "why": ("Linear averaging of the three seeds' 8x8 confusion matrices is a well-defined, "
                "information-preserving aggregate (uses all three seeds, not just one) and is the "
                "standard way to report an 'average confusion matrix' across repeated runs. The "
                "median-mIoU seed is also reported below for reference/cross-check."),
        "mean_matrix_rows_gt_cols_pred": mean_cm.tolist(),
        "median_mIoU_seed": int(median_seed),
        "median_mIoU_seed_matrix_rows_gt_cols_pred": per_seed_results[median_seed]["confusion_matrix_rows_gt_cols_pred"],
    }

    # instance metrics (only if supported)
    im0 = per_seed_results[seeds_sorted[0]].get("instance_metrics", {})
    if im0.get("supported") is False:
        agg["instance_metrics"] = {"supported": False, "reason": im0.get("reason")}
    else:
        agg["instance_metrics"] = {"supported": True}
        for k in ["AP", "AP50", "AP75", "AR_max100"]:
            vals = np.array([per_seed_results[s]["instance_metrics"].get(k, np.nan) for s in seeds_sorted])
            agg["instance_metrics"][k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=0)),
                                           "per_seed": {int(s): float(v) for s, v in zip(seeds_sorted, vals)}}
        agg["instance_metrics"]["confidence_threshold"] = im0.get("confidence_threshold")

    agg["checkpoints_used"] = {int(s): per_seed_results[s]["checkpoint_used"] for s in seeds_sorted}
    agg["test_set"] = per_seed_results[seeds_sorted[0]]["test_set"]
    agg["n_test_images"] = per_seed_results[seeds_sorted[0]]["n_test_images"]
    return agg


# ===========================================================================
if __name__ == "__main__":
    t0 = time.time()
    ALL_SEED_RESULTS = {42: {}}

    # seed 42: reuse already-computed results (same script/methodology)
    with open(f"{RUN}/evaluation_results.json") as f:
        seed42 = json.load(f)
    name_map = {"UNet": "UNet", "DeepLabV3": "DeepLabV3", "SegFormer": "SegFormer",
                "MaskRCNN": "MaskRCNN", "SoloV2": "SoloV2", "YOLO11seg": "YOLO11-seg",
                "PGSS_YOLO11": "YOLO11-PGSS"}
    inv_name_map = {"UNet": "UNet", "DeepLabV3": "DeepLabV3", "SegFormer": "SegFormer",
                    "MaskRCNN": "MaskRCNN", "SoloV2": "SoloV2", "YOLO11-seg": "YOLO11seg",
                    "YOLO11-PGSS": "PGSS_YOLO11"}
    for display_name, val in seed42.items():
        step_key = inv_name_map.get(display_name, display_name)
        ALL_SEED_RESULTS[42][step_key] = val
    log(f"Loaded existing seed-42 results for {list(ALL_SEED_RESULTS[42].keys())}")

    steps = [
        ("UNet", lambda seed: eval_unet(f"{CKPT_ROOT}/UNet/checkpoints{ckpt_suffix(seed)}/best_model.pth")),
        ("DeepLabV3", lambda seed: eval_deeplabv3(f"{CKPT_ROOT}/DeepLabV3/checkpoints{ckpt_suffix(seed)}/best_model.pth")),
        ("SegFormer", lambda seed: eval_segformer(f"{CKPT_ROOT}/SegFormer/checkpoints{ckpt_suffix(seed)}/best_model.pth")),
        ("MaskRCNN", lambda seed: eval_maskrcnn(f"{CKPT_ROOT}/MaskRCNN/checkpoints{ckpt_suffix(seed)}/best_model.pth")),
        ("SoloV2", lambda seed: eval_solov2(f"{CKPT_ROOT}/SoloV2/checkpoints_clean{ckpt_suffix(seed)}/best_model.pth")),
        ("YOLO11seg", lambda seed: eval_yolo_model(
            "YOLO11-seg", f"{CKPT_ROOT}/YOLO11seg/runs/vine_clean_v6{ckpt_suffix(seed)}/weights/best_dice.pt")),
        ("PGSS_YOLO11", lambda seed: eval_yolo_model(
            "YOLO11-PGSS", f"{CKPT_ROOT}/PGSS_YOLO11/runs/pgss_vine_v6{ckpt_suffix(seed)}/weights/best_dice.pt",
            extra_module_dir=f"{RUN}/PGSS_YOLO11")),
    ]

    for seed in [43, 44]:
        ALL_SEED_RESULTS[seed] = {}
        for step_name, fn in steps:
            log(f"=== seed {seed}: {step_name} ===")
            try:
                out = fn(seed)
                ALL_SEED_RESULTS[seed][step_name] = out
                log(f"seed {seed} {step_name} done: mIoU={out['semantic_metrics']['mIoU_8class']:.4f}"
                    + (f", AP={out['instance_metrics'].get('AP')}" if out["instance_metrics"].get("supported", True) else ""))
            except Exception as e:
                log(f"!!! seed {seed} {step_name} FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                ALL_SEED_RESULTS[seed][step_name] = {"model": step_name, "error": str(e)}
        with open(f"{RUN}/evaluation_results_seed{seed}.json", "w") as f:
            json.dump(ALL_SEED_RESULTS[seed], f, indent=2)
        log(f"Wrote evaluation_results_seed{seed}.json")

    # ---- aggregate across seeds per model ----
    MULTISEED = {}
    for step_name, _ in steps:
        per_seed = {s: ALL_SEED_RESULTS[s][step_name] for s in SEEDS
                    if "error" not in ALL_SEED_RESULTS[s].get(step_name, {"error": 1})}
        if len(per_seed) < len(SEEDS):
            missing = [s for s in SEEDS if s not in per_seed]
            log(f"WARNING: {step_name} missing seeds {missing} (errored) -- aggregating over {list(per_seed.keys())} only")
        MULTISEED[step_name] = {
            "model": name_map[step_name],
            "per_seed_full_results": {int(s): per_seed[s] for s in per_seed},
            "aggregate": aggregate_model(per_seed) if per_seed else None,
        }

    with open(f"{RUN}/evaluation_multiseed.json", "w") as f:
        json.dump(MULTISEED, f, indent=2)
    log("Wrote evaluation_multiseed.json")

    # ---- flat summary CSV (mean +- std) ----
    import csv
    with open(f"{RUN}/evaluation_multiseed_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "n_seeds", "macro_precision_mean", "macro_precision_std",
                   "macro_recall_mean", "macro_recall_std", "macro_f1_mean", "macro_f1_std",
                   "mIoU_mean", "mIoU_std"]
        header += [f"IoU_{c}_mean" for c in CLASS_NAMES] + [f"IoU_{c}_std" for c in CLASS_NAMES]
        header += ["AP_mean", "AP_std", "AP50_mean", "AP50_std", "AP75_mean", "AP75_std",
                   "AR_max100_mean", "AR_max100_std", "instance_supported"]
        w.writerow(header)
        for step_name, _ in steps:
            entry = MULTISEED[step_name]
            agg = entry["aggregate"]
            if agg is None:
                w.writerow([entry["model"], 0] + [""] * (len(header) - 2))
                continue
            sm = agg["semantic_metrics"]
            row = [entry["model"], len(entry["per_seed_full_results"]),
                   sm["macro_precision_8class"]["mean"], sm["macro_precision_8class"]["std"],
                   sm["macro_recall_8class"]["mean"], sm["macro_recall_8class"]["std"],
                   sm["macro_f1_8class"]["mean"], sm["macro_f1_8class"]["std"],
                   sm["mIoU_8class"]["mean"], sm["mIoU_8class"]["std"]]
            for c in CLASS_NAMES:
                key = "background_iou" if c == "background" else f"IoU_{c}"
                row.append(sm[key]["mean"])
            for c in CLASS_NAMES:
                key = "background_iou" if c == "background" else f"IoU_{c}"
                row.append(sm[key]["std"])
            im = agg["instance_metrics"]
            if im.get("supported") is False:
                row += ["", "", "", "", "", "", "", "", False]
            else:
                row += [im["AP"]["mean"], im["AP"]["std"], im["AP50"]["mean"], im["AP50"]["std"],
                        im["AP75"]["mean"], im["AP75"]["std"], im["AR_max100"]["mean"], im["AR_max100"]["std"], True]
            w.writerow(row)
    log("Wrote evaluation_multiseed_summary.csv")

    # ---- flat per-seed CSV ----
    with open(f"{RUN}/evaluation_multiseed_per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "seed", "checkpoint_used", "macro_precision", "macro_recall", "macro_f1", "mIoU"]
        header += [f"IoU_{c}" for c in CLASS_NAMES]
        header += ["AP", "AP50", "AP75", "AR_max100", "instance_supported"]
        w.writerow(header)
        for step_name, _ in steps:
            entry = MULTISEED[step_name]
            for seed in SEEDS:
                r = entry["per_seed_full_results"].get(seed)
                if r is None:
                    w.writerow([entry["model"], seed, "ERROR/MISSING"] + [""] * (len(header) - 3))
                    continue
                sm = r["semantic_metrics"]
                row = [entry["model"], seed, r["checkpoint_used"], sm["macro_precision_8class"],
                       sm["macro_recall_8class"], sm["macro_f1_8class"], sm["mIoU_8class"]]
                row.append(sm["background_iou"])
                row += [sm["per_class_iou_species"][c] for c in SPECIES]
                im = r.get("instance_metrics", {})
                if im.get("supported") is False:
                    row += ["", "", "", "", False]
                else:
                    row += [im.get("AP"), im.get("AP50"), im.get("AP75"), im.get("AR_max100"), True]
                w.writerow(row)
    log("Wrote evaluation_multiseed_per_seed.csv")

    log(f"ALL DONE in {time.time()-t0:.1f}s")
