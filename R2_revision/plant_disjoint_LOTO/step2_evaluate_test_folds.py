#!/usr/bin/env python3
"""
step2_evaluate_test_folds.py
============================
Step 2 of the leave-one-treatment-out (LOTO) analysis: evaluate the selected checkpoints
(7 models x 3 folds, seed 42) on the test images of their own fold, as pre-specified in
ANALYSIS_PLAN.md, Section 3.

Per fold and model it computes:
  * the pooled 8 x 8 pixel confusion matrix on the 512 x 512 grid (rows = ground truth,
    columns = prediction);
  * from that matrix, exactly as R1 run_full_test_evaluation.py: macro precision, recall, F1 and
    mIoU over the 8 classes (background + 7 species) and per-class IoU; plus mIoU over the 7
    species (plan, Section 3);
  * per-image mIoU and per-image macro F1 (classes absent from both ground truth and prediction
    skipped, as in R1 compute_per_image_f1.py), with each image's treatment x species cluster.
    These are the inputs of cluster_bootstrap_compare.py (Table S4);
  * COCO mask AP, AP50 and AP75 (pycocotools, iouType "segm") for the instance-capable models,
    using every detection with score >= 0.05. For the simplified SOLOv2 baseline these are
    connected-component pseudo-instances, built as in R1.

Predictions and ground truth come from loto_common.py, the same code that selected the epochs.
So the test metrics use the same predictor, the same 0.25 instance score threshold and the same
ground-truth rasterisation as the selection score.

Checks before any metric is written:
  * the epoch stored in each checkpoint equals the selected epoch in selection_summary.json;
  * every evaluated image is assigned to this split of this fold in loto_fold_assignment.csv;
  * with `--split valid` the script re-scores the validation images. It compares the pooled Dice
    with the value recorded during training, which shows the right weights were loaded. Run this
    before the test evaluation.

Usage (on the GPU machine, with the training environment):
  # 1. pipeline test with the 1-epoch smoke checkpoints (validation images only; NOT a result)
  python step2_evaluate_test_folds.py --runs_root /workspace/loto_smoke --folds test_trt20 \
      --split valid --out_root /workspace/loto_eval_smoke --smoke
  # 2. re-score the validation images with the 21 selected checkpoints (must match training)
  python step2_evaluate_test_folds.py --split valid --out_root /workspace/loto_eval_validcheck
  # 3. test evaluation (the results)
  python step2_evaluate_test_folds.py --split test --out_root /workspace/loto_eval

This script only reads the fold data and the checkpoints. It writes only under --out_root.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # as R1: use the cached SegFormer backbone
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loto_common as LC  # noqa: E402  (also puts author_scripts/ on sys.path)
from unified_pixel_dice import PooledDice  # noqa: E402

MODELS = ["UNet", "DeepLabV3", "SegFormer", "MaskRCNN", "SoloV2", "YOLO11seg", "PGSS_YOLO11"]
FOLDS = ["test_trt20", "test_trt50", "test_trt80"]
TRAIN_SCRIPTS = {
    "UNet": "models/UNet/train_unet_loto.py",
    "DeepLabV3": "models/DeepLabV3/train_deeplabv3_loto.py",
    "SegFormer": "models/SegFormer/train_segformer_loto.py",
    "MaskRCNN": "models/MaskRCNN/train_maskrcnn_loto.py",
    "SoloV2": "models/SoloV2/train_solov2_loto.py",
}
YOLO_MODELS = ("YOLO11seg", "PGSS_YOLO11")
INSTANCE_MODELS = ("MaskRCNN", "SoloV2", "YOLO11seg", "PGSS_YOLO11")
PREDICTOR_KIND = {"UNet": "unet", "DeepLabV3": "deeplabv3", "SegFormer": "segformer",
                  "MaskRCNN": "maskrcnn", "SoloV2": "solov2", "YOLO11seg": "yolo", "PGSS_YOLO11": "yolo"}
COCO_SCORE_THR = 0.05     # plan, Section 3: all detections with score >= 0.05, every model
SOLO_MIN_AREA = 150       # R1 run_full_test_evaluation.py: minimum pseudo-instance area (pixels)
IMG = LC.IMG
K = LC.NUM_CLASSES
CLASS_NAMES = LC.CLASS_NAMES
SPECIES = CLASS_NAMES[1:]
CAT_IDS = list(range(1, 8))
SPLIT_LABEL = {"test": "test", "valid": "val"}   # split folder -> value in loto_fold_assignment.csv
FROZEN_PLAN_COMMIT = "ecf84fd346cecf73b5f17f84ba89fb42edab9807"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------
def cm_from_maps(gt, pred):
    """8 x 8 confusion matrix, rows = ground truth, columns = prediction."""
    g = gt.astype(np.int64).ravel()
    p = pred.astype(np.int64).ravel()
    return np.bincount(g * K + p, minlength=K * K).reshape(K, K)


def macro_metrics_from_cm(cm):
    """R1 run_full_test_evaluation.py macro_metrics_from_cm(), unchanged."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return precision, recall, f1, iou


def per_image_scores(cm):
    """Per-image mIoU and macro F1 over the classes present in ground truth or prediction.
    F1 follows R1 compute_per_image_f1.py macro_f1_from_cm(); IoU uses the same class set."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    ious, f1s = [], []
    for i in range(len(tp)):
        if tp[i] + fp[i] + fn[i] == 0:
            continue
        ious.append(tp[i] / (tp[i] + fp[i] + fn[i]))
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    if not ious:
        return float("nan"), float("nan"), 0
    return float(np.mean(ious)), float(np.mean(f1s)), len(ious)


def semantic_metrics(cm):
    precision, recall, f1, iou = macro_metrics_from_cm(cm)
    return {
        "mIoU_8class": float(np.mean(iou)),
        "mIoU_7species": float(np.mean(iou[1:])),
        "macro_precision_8class": float(np.mean(precision)),
        "macro_recall_8class": float(np.mean(recall)),
        "macro_f1_8class": float(np.mean(f1)),
        "per_class_iou": {CLASS_NAMES[i]: float(iou[i]) for i in range(K)},
        "per_class_precision": {CLASS_NAMES[i]: float(precision[i]) for i in range(K)},
        "per_class_recall": {CLASS_NAMES[i]: float(recall[i]) for i in range(K)},
        "per_class_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(K)},
    }


def coco_instance_eval(coco_gt, predictions, img_ids):
    """R1 coco_instance_eval(), restricted to the evaluated images."""
    from pycocotools.cocoeval import COCOeval
    res = {"score_threshold": COCO_SCORE_THR, "n_detections": len(predictions),
           "iou_thresholds": "COCO standard: 0.50:0.05:0.95 (AP), 0.50 (AP50), 0.75 (AP75)"}
    if not predictions:
        res.update({"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR_max100": 0.0,
                    "warning": "no detections at or above the score threshold"})
        return res
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, iouType="segm")
    ev.params.catIds = CAT_IDS
    ev.params.imgIds = list(img_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    res.update({"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
                "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
                "AR_max1": float(s[6]), "AR_max10": float(s[7]), "AR_max100": float(s[8])})
    return res


def mask_to_rle(mask_bool):
    from pycocotools import mask as coco_mask
    rle = coco_mask.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def xyxy_to_xywh(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


# ------------------------------------------------------------------------------------------------
# Instance predictions for COCO AP (R1 run_full_test_evaluation.py, score threshold 0.05)
# ------------------------------------------------------------------------------------------------
def maskrcnn_instances(net, path, iid, device):
    import torch
    from torchvision import transforms as T
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    H0, W0 = img.shape[:2]
    x = T.ToTensor()(cv2.resize(img, (IMG, IMG))).to(device)
    out = net([x])[0]
    sx, sy = W0 / IMG, H0 / IMG
    dets = []
    for i in torch.argsort(out["scores"], descending=True).tolist():
        score = float(out["scores"][i])
        if score < COCO_SCORE_THR:
            continue
        m512 = (out["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
        if m512.sum() == 0:
            continue
        m_native = cv2.resize(m512, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
        if m_native.sum() == 0:
            continue
        b = out["boxes"][i].cpu().numpy()
        dets.append({"image_id": iid, "category_id": int(out["labels"][i]),
                     "segmentation": mask_to_rle(m_native),
                     "bbox": xyxy_to_xywh(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), "score": score})
    return dets


def solo_instances(net, path, iid, device):
    import torch
    from torchvision import transforms as T
    tf = T.Compose([T.ToPILImage(), T.Resize((IMG, IMG)), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb.shape[:2]
    category_pred, mask_pred = net(tf(img_rgb).unsqueeze(0).to(device))
    probs = torch.nn.functional.interpolate(torch.softmax(category_pred, dim=1), size=(H0, W0),
                                            mode="bilinear", align_corners=False)[0]
    obj = torch.nn.functional.interpolate(torch.sigmoid(mask_pred), size=(H0, W0),
                                          mode="bilinear", align_corners=False)[0, 0]
    conf_full, ids_full = probs.max(dim=0)
    fg = ((ids_full > 0) & (obj > LC.SOLO_OBJ_THR)).cpu().numpy().astype(np.uint8)
    ids_np = ids_full.cpu().numpy()
    conf_np = conf_full.cpu().numpy()
    dets = []
    for cls_idx in np.unique(ids_np):
        if cls_idx == 0:
            continue
        cls_mask = ((ids_np == cls_idx) & (fg > 0)).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue
        n, labels_cc, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] < SOLO_MIN_AREA:
                continue
            comp = labels_cc == lbl
            score = float(conf_np[comp].mean())
            if score < COCO_SCORE_THR:
                continue
            ys, xs = np.where(comp)
            dets.append({"image_id": iid, "category_id": int(cls_idx), "segmentation": mask_to_rle(comp),
                         "bbox": xyxy_to_xywh(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                         "score": score})
    return dets


def yolo_instances(yolo, path, iid, device):
    r = yolo.predict(path, imgsz=IMG, conf=COCO_SCORE_THR, device=device, verbose=False)[0]
    H0, W0 = r.orig_shape
    dets = []
    if r.masks is None or len(r.masks) == 0:
        return dets
    polys = r.masks.xy
    clss = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    xyxy = r.boxes.xyxy.cpu().numpy()
    for idx in np.argsort(-confs):
        poly = polys[idx]
        if poly.shape[0] < 3:
            continue
        m = np.zeros((H0, W0), np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        m = m.astype(bool)
        if m.sum() == 0:
            continue
        x1, y1, x2, y2 = xyxy[idx]
        dets.append({"image_id": iid, "category_id": int(clss[idx]) + 1, "segmentation": mask_to_rle(m),
                     "bbox": xyxy_to_xywh(x1, y1, x2, y2), "score": float(confs[idx])})
    return dets


# ------------------------------------------------------------------------------------------------
# Data and models
# ------------------------------------------------------------------------------------------------
def apply_gt_corrections(coco, corrections):
    """Relabel the polygons of author-confirmed mislabelled images (post_evaluation_corrections.json).
    Only the ground truth used here changes; the fold data on disk are not modified."""
    applied = []
    name_to_id = {c["name"]: cid for cid, c in coco.cats.items()}
    by_name = {info["file_name"]: iid for iid, info in coco.imgs.items()}
    for c in corrections:
        iid = by_name.get(c["file_name"])
        if iid is None:
            continue
        src, dst = name_to_id[c["from_category"]], name_to_id[c["to_category"]]
        n = 0
        for ann in coco.dataset["annotations"]:
            if ann["image_id"] == iid and ann["category_id"] == src:
                ann["category_id"] = dst
                n += 1
        if n == 0:
            raise SystemExit(f"[step2] correction for {c['file_name']}: no {c['from_category']} polygon found")
        applied.append({"file_name": c["file_name"], "from": c["from_category"], "to": c["to_category"],
                        "n_polygons_relabelled": n})
    if applied:
        coco.createIndex()
    return applied


def load_split_items(folds_root, fold, split, max_images, corrections=None):
    from pycocotools.coco import COCO
    split_dir = os.path.join(folds_root, fold, "rebuilt", "final", split)
    coco = COCO(os.path.join(split_dir, "_annotations.coco.json"))
    applied = apply_gt_corrections(coco, corrections or [])
    for a in applied:
        log(f"GT correction ({fold}/{split}): {a}")
    ids = sorted(coco.imgs)                                   # same order as loto_common.load_split
    if max_images:
        ids = ids[:max_images]
    items = [(iid, coco.imgs[iid]["file_name"], os.path.join(split_dir, coco.imgs[iid]["file_name"]),
              LC.gt_512(coco, iid)) for iid in ids]
    return coco, items, applied


def check_assignment(items, assign, fold, split, species_fix=None):
    col = f"fold_{fold}"
    meta = {}
    for _, fname, path, _ in items:
        row = assign.get(fname)
        if row is None:
            raise SystemExit(f"[step2] {fname} ({fold}/{split}) is not in loto_fold_assignment.csv")
        if row[col] != SPLIT_LABEL[split]:
            raise SystemExit(f"[step2] {fname}: {col} = {row[col]!r}, expected {SPLIT_LABEL[split]!r}")
        if not os.path.isfile(path):
            raise SystemExit(f"[step2] image not found: {path}")
        species = (species_fix or {}).get(fname, row["species"])
        meta[fname] = {"species": species, "tier": row["tier"], "treatment": row["treatment"],
                       "block": row["block"], "group_id": row["group_id"],
                       "cluster_id": f"trt{row['treatment']}|{species}"}
    return meta


def import_training_module(model, fold, folds_root, seed, scratch):
    """Import a training script for its model definition. The scripts call
    loto_common.run_config() at import time, which writes loto_run_config.json into
    LOTO_OUT_ROOT/<fold>/seed<seed>/<model>; LOTO_OUT_ROOT is pointed at a scratch folder so the
    training outputs are never touched. Nothing is trained."""
    for k in ("LOTO_MAX_EPOCHS", "LOTO_PATIENCE"):
        os.environ.pop(k, None)
    os.environ.update(LOTO_FOLD=fold, LOTO_SEED=str(seed), LOTO_OUT_ROOT=scratch,
                      LOTO_DATA_ROOT=os.path.join(folds_root, fold, "rebuilt", "final"))
    path = os.path.join(HERE, TRAIN_SCRIPTS[model])
    spec = importlib.util.spec_from_file_location(f"loto_train_{model}_{fold}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(model, fold, run_dir, args, scratch, device):
    """Returns (net, predictor, ckpt_path, stored_epoch1, stored_val_dice)."""
    import torch
    if model in YOLO_MODELS:
        if model == "PGSS_YOLO11":
            sys.path.insert(0, os.path.join(HERE, "models", "PGSS_YOLO11"))
            import pgss_modules  # noqa: F401  registers the PGSS layers before loading the weights
        from ultralytics import YOLO
        ckpt_path = os.path.join(run_dir, "weights", "best_unified.pt")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch1 = int(ck["epoch"]) + 1
        del ck
        yolo_device = 0 if torch.cuda.is_available() else "cpu"
        return YOLO(ckpt_path), LC.make_predictor("yolo", yolo_device), ckpt_path, epoch1, None
    mod = import_training_module(model, fold, args.folds_root, args.seed, scratch)
    ckpt_path = os.path.join(run_dir, "best_model.pth")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if model == "UNet":
        net = mod.UNet(3, K)
    elif model == "SoloV2":
        net = mod.SimplifiedSOLOv2(num_classes=K)
    else:
        net = mod.build_model()
    net.load_state_dict(ck["model_state_dict"])            # strict: every weight must match
    net.to(device).eval()
    if model == "SegFormer":
        pred = LC.make_predictor("segformer", device, segformer_forward=mod.forward_logits,
                                 segformer_norm=mod.NORM)
    else:
        pred = LC.make_predictor(PREDICTOR_KIND[model], device)
    return net, pred, ckpt_path, int(ck["epoch"]) + 1, ck.get("val_pooled_dice_512")


# ------------------------------------------------------------------------------------------------
# One fold x model
# ------------------------------------------------------------------------------------------------
def evaluate_one(model, fold, coco, items, meta, args, scratch, device):
    import torch
    run_dir = os.path.join(args.runs_root, fold, f"seed{args.seed}", model)
    summ_path = os.path.join(run_dir, "selection_summary.json")
    if not os.path.isfile(summ_path):
        raise SystemExit(f"[step2] {summ_path} not found (training not finished?)")
    summ = json.load(open(summ_path))
    net, predict, ckpt_path, epoch1, stored_val = load_model(model, fold, run_dir, args, scratch, device)
    if os.path.basename(summ["checkpoint"]) != os.path.basename(ckpt_path):
        raise SystemExit(f"[step2] {model}/{fold}: summary names {summ['checkpoint']}, expected {ckpt_path}")
    if epoch1 != summ["selected_epoch"]:
        raise SystemExit(f"[step2] {model}/{fold}: checkpoint holds epoch {epoch1} but the selected "
                         f"epoch is {summ['selected_epoch']}; refusing to evaluate")
    ck_sha = sha256(ckpt_path)
    log(f"{model} | {fold} | {args.split}: checkpoint epoch {epoch1} (selected {summ['selected_epoch']} of "
        f"{summ['epochs_run']}), sha256 {ck_sha[:12]}..., {len(items)} images")

    t0 = time.time()
    preds, cms, rows = [], [], []
    pooled = PooledDice(K)
    detections = []
    instance = model in INSTANCE_MODELS and not args.no_coco
    yolo_device = 0 if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for iid, fname, path, gt in items:
            p = predict(net, path).astype(np.uint8)
            cm = cm_from_maps(gt, p)
            pooled.update(p.astype(np.int64), gt.astype(np.int64))
            miou, mf1, ncls = per_image_scores(cm)
            preds.append(p)
            cms.append(cm)
            rows.append({"fold": fold, "model": model, "image_id": fname, "coco_image_id": iid,
                         **meta[fname], "per_image_miou": miou, "per_image_macro_f1": mf1,
                         "n_classes_scored": ncls})
            if instance:
                if model == "MaskRCNN":
                    detections += maskrcnn_instances(net, path, iid, device)
                elif model == "SoloV2":
                    detections += solo_instances(net, path, iid, device)
                else:
                    detections += yolo_instances(net, path, iid, yolo_device)
    secs = time.time() - t0
    cm_total = np.sum(cms, axis=0)
    pooled_dice = pooled.macro()
    sem = semantic_metrics(cm_total)
    coco_res = coco_instance_eval(coco, detections, [it[0] for it in items]) if instance else \
        {"supported": False, "reason": "semantic-segmentation architecture (no instances or scores)"}

    out = {
        "model": model, "fold": fold, "split": args.split, "seed": args.seed,
        "n_images": len(items), "smoke_test_NOT_A_RESULT": bool(args.smoke),
        "checkpoint": ckpt_path, "checkpoint_sha256": ck_sha, "checkpoint_epoch": epoch1,
        "selected_epoch": summ["selected_epoch"], "epochs_run": summ["epochs_run"],
        "stopped_early": summ.get("stopped_early"),
        "val_pooled_dice_recorded_in_training": summ["val_pooled_macro_dice_512"],
        "val_pooled_dice_stored_in_checkpoint": stored_val,
        "pooled_macro_dice_present_classes": pooled_dice,   # the selection criterion, on this split
        "semantic_metrics": sem,
        "instance_metrics": coco_res,
        "confusion_matrix_rows_gt_cols_pred": cm_total.tolist(),
        "settings": {"grid": IMG, "instance_score_threshold_semantic": LC.SCORE_THR,
                     "solo_objectness_threshold": LC.SOLO_OBJ_THR,
                     "coco_score_threshold": COCO_SCORE_THR, "solo_min_area_px": SOLO_MIN_AREA},
        "seconds": round(secs, 1),
    }
    d = os.path.join(args.out_root, fold, model)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    with open(os.path.join(d, "per_image_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(d, "per_image_confusion.npz"),
                        image_id=np.array([r["image_id"] for r in rows]), cm=np.stack(cms).astype(np.int64))
    np.savez_compressed(os.path.join(d, "pred_maps_512.npz"),
                        image_id=np.array([r["image_id"] for r in rows]), pred=np.stack(preds))
    if instance:
        with gzip.open(os.path.join(d, "coco_detections.json.gz"), "wt") as f:
            json.dump(detections, f)
    log(f"{model} | {fold}: mIoU(8) {sem['mIoU_8class']:.4f} | macro F1 {sem['macro_f1_8class']:.4f} | "
        f"pooled Dice {pooled_dice:.4f}" + (f" | AP {coco_res.get('AP'):.4f}" if instance else "")
        + f" | {secs:.0f}s")
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, rows


# ------------------------------------------------------------------------------------------------
# Summaries
# ------------------------------------------------------------------------------------------------
SUMMARY_KEYS = ["mIoU_8class", "mIoU_7species", "macro_precision_8class", "macro_recall_8class",
                "macro_f1_8class"]


def flat_row(r):
    sm, im = r["semantic_metrics"], r["instance_metrics"]
    row = {"fold": r["fold"], "model": r["model"], "split": r["split"], "n_images": r["n_images"],
           "selected_epoch": r["selected_epoch"], "epochs_run": r["epochs_run"]}
    row.update({k: sm[k] for k in SUMMARY_KEYS})
    row.update({f"IoU_{c}": sm["per_class_iou"][c] for c in CLASS_NAMES})
    row["pooled_macro_dice_present_classes"] = r["pooled_macro_dice_present_classes"]
    for k in ("AP", "AP50", "AP75", "AR_max100", "n_detections"):
        row[f"coco_{k}"] = im.get(k, "")
    row["checkpoint_sha256"] = r["checkpoint_sha256"]
    return row


def write_summaries(results, all_rows, args):
    out = args.out_root
    flat = [flat_row(r) for r in results]
    with open(os.path.join(out, "summary_per_fold.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    # mean +- SD (n - 1) across folds, only for models evaluated on all three folds
    metric_cols = SUMMARY_KEYS + [f"IoU_{c}" for c in CLASS_NAMES] + ["coco_AP", "coco_AP50", "coco_AP75"]
    across = []
    for m in MODELS:
        fr = [x for x in flat if x["model"] == m]
        if sorted(x["fold"] for x in fr) != FOLDS:
            continue
        row = {"model": m, "n_folds": 3, "n_images_total": sum(x["n_images"] for x in fr)}
        for c in metric_cols:
            vals = [x[c] for x in fr]
            if any(v == "" for v in vals):
                continue
            v = np.array(vals, float)
            row[f"{c}_mean"] = float(v.mean())
            row[f"{c}_sd"] = float(v.std(ddof=1))
            for x in fr:
                row[f"{c}_{x['fold']}"] = x[c]
        across.append(row)
    if across:
        keys = []
        for r in across:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(out, "summary_across_folds.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(across)
    # confusion matrices (per fold and summed over folds) for Fig. 3
    cms = {"rows": "ground truth", "cols": "prediction", "classes": CLASS_NAMES, "per_fold": {}, "all_folds": {}}
    for r in results:
        cms["per_fold"].setdefault(r["fold"], {})[r["model"]] = r["confusion_matrix_rows_gt_cols_pred"]
    for m in MODELS:
        mats = [np.array(r["confusion_matrix_rows_gt_cols_pred"]) for r in results if r["model"] == m]
        if len(mats) == len(FOLDS):
            cms["all_folds"][m] = np.sum(mats, axis=0).tolist()
    json.dump(cms, open(os.path.join(out, "confusion_matrices.json"), "w"), indent=1)
    # per-image scores, long format, and the two inputs of cluster_bootstrap_compare.py
    if all_rows:
        with open(os.path.join(out, "per_image_metrics_all.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        for score, name in (("per_image_miou", "bootstrap_input_per_image_miou.csv"),
                            ("per_image_macro_f1", "bootstrap_input_per_image_f1.csv")):
            with open(os.path.join(out, name), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "cluster_id", "fold", "model", "score"])
                for r in all_rows:
                    w.writerow([r["image_id"], r["cluster_id"], r["fold"], r["model"], r[score]])
    # validation re-scoring check
    if args.split == "valid":
        rows, worst = [], 0.0
        for r in results:
            rec = r["val_pooled_dice_recorded_in_training"]
            diff = r["pooled_macro_dice_present_classes"] - rec
            worst = max(worst, abs(diff))
            status = "match" if abs(diff) < 1e-6 else ("near (GPU rounding)" if abs(diff) < 1e-3 else "MISMATCH")
            rows.append([r["fold"], r["model"], r["selected_epoch"], f"{rec:.6f}",
                         f"{r['pooled_macro_dice_present_classes']:.6f}", f"{diff:+.2e}", status])
        with open(os.path.join(out, "valid_rescoring_check.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "model", "selected_epoch", "val_dice_recorded_in_training",
                        "val_dice_rescored_now", "difference", "status"])
            w.writerows(rows)
        for row in rows:
            log("valid check:", " | ".join(str(x) for x in row))
        return worst
    return None


def versions():
    import importlib.metadata as md
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for p in ("torch", "torchvision", "ultralytics", "transformers", "pycocotools", "numpy",
              "opencv-python", "opencv-python-headless"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    try:
        import torch
        v["cuda"] = torch.version.cuda
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:  # pragma: no cover
        v["torch_error"] = str(e)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds_root", default="/workspace/Revision/LOTO_folds")
    ap.add_argument("--runs_root", default="/workspace/loto_runs")
    ap.add_argument("--fold_file", default=os.path.join(HERE, "..", "..", "audit_outputs",
                                                        "loto_fold_assignment.csv"))
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--split", choices=["test", "valid"], required=True)
    ap.add_argument("--folds", nargs="+", default=FOLDS, choices=FOLDS)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_images", type=int, default=0, help="only the first N images (pipeline tests)")
    ap.add_argument("--no_coco", action="store_true", help="skip COCO AP (faster pipeline tests)")
    ap.add_argument("--smoke", action="store_true", help="pipeline test with smoke checkpoints; NOT a result")
    ap.add_argument("--gt_corrections", default="",
                    help="post_evaluation_corrections.json: relabel author-confirmed mislabelled test "
                         "images (a deviation from the frozen plan; write to a separate --out_root)")
    args = ap.parse_args()

    args.out_root = os.path.abspath(args.out_root)
    args.runs_root = os.path.abspath(args.runs_root)
    if os.path.commonpath([args.out_root, args.runs_root]) == args.runs_root:
        raise SystemExit("[step2] --out_root must not be inside --runs_root (training outputs are read-only)")
    is_smoke_run = "smoke" in args.runs_root.lower()
    if is_smoke_run and not args.smoke:
        raise SystemExit("[step2] runs_root looks like the smoke-test runs; add --smoke")
    if args.smoke and "smoke" not in args.out_root.lower():
        raise SystemExit("[step2] with --smoke the --out_root name must contain 'smoke'")
    partial = args.max_images or args.no_coco or args.smoke or sorted(args.folds) != FOLDS \
        or sorted(args.models) != sorted(MODELS)
    if args.split == "test" and partial and not args.smoke:
        log("NOTE: partial test evaluation (subset of folds, models or images, or no COCO). "
            "Report only a complete run.")
    os.makedirs(args.out_root, exist_ok=True)
    if args.smoke:
        open(os.path.join(args.out_root, "NOT_A_RESULT.txt"), "w").write(
            "Pipeline test with 1-epoch smoke checkpoints. These numbers are not results and must "
            "never be reported.\n")
    scratch = os.path.join(args.out_root, "_import_scratch")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assign = {r["roboflow_filename"]: r for r in csv.DictReader(open(args.fold_file))}
    corrections = json.load(open(args.gt_corrections)) if args.gt_corrections else []
    species_fix = {c["file_name"]: c["to_category"] for c in corrections}
    manifest = {
        "script": os.path.abspath(__file__), "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "args": vars(args), "analysis_plan_commit": FROZEN_PLAN_COMMIT,
        "sha256": {p: sha256(os.path.join(HERE, p)) for p in
                   ["step2_evaluate_test_folds.py", "loto_common.py", "../unified_pixel_dice.py",
                    *TRAIN_SCRIPTS.values(), "models/PGSS_YOLO11/pgss_modules.py"]},
        "fold_file_sha256": sha256(args.fold_file), "versions": versions(),
        "smoke_test_NOT_A_RESULT": bool(args.smoke), "partial_run": bool(partial),
        "gt_corrections": corrections,
        "gt_corrections_sha256": sha256(args.gt_corrections) if args.gt_corrections else None,
        "gt_corrections_applied": [],
    }
    json.dump(manifest, open(os.path.join(args.out_root, "eval_manifest.json"), "w"), indent=2)

    results, all_rows, failures = [], [], []
    for fold in args.folds:
        coco, items, applied = load_split_items(args.folds_root, fold, args.split, args.max_images, corrections)
        manifest["gt_corrections_applied"] += [dict(a, fold=fold, split=args.split) for a in applied]
        meta = check_assignment(items, assign, fold, args.split, species_fix)
        log(f"=== {fold} / {args.split}: {len(items)} images, all assigned to this split of this fold")
        for model in args.models:
            try:
                out, rows = evaluate_one(model, fold, coco, items, meta, args, scratch, device)
                results.append(out)
                all_rows += rows
            except SystemExit:
                raise
            except Exception as e:  # record and continue with the other models
                import traceback
                traceback.print_exc()
                failures.append(f"{fold}/{model}: {type(e).__name__}: {e}")
                log(f"!!! {fold}/{model} FAILED: {e}")

    worst = write_summaries(results, all_rows, args) if results else None
    manifest.update(finished_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    n_evaluated=len(results), failures=failures)
    json.dump(manifest, open(os.path.join(args.out_root, "eval_manifest.json"), "w"), indent=2)
    if failures:
        log("FAILED:", *failures)
        sys.exit(1)
    if worst is not None and worst >= 1e-3:
        log(f"VALIDATION RE-SCORING MISMATCH (largest difference {worst:.2e}); do not run the test evaluation")
        sys.exit(2)
    log(f"ALL DONE: {len(results)} fold x model evaluations written to {args.out_root}")


if __name__ == "__main__":
    main()
