"""
Per-image macro-F1 (8-class) for all 7 models x 3 seeds on the 169-image test set.
Reuses the exact same model-loading / inference code as run_multiseed_evaluation.py,
but keeps one confusion matrix (and derived macro-F1) per image instead of accumulating
a single pooled confusion matrix, so per-image scores can be compared across models for
the clustered pairwise significance testing.

Output: per_image_f1.json  -- {model: {seed: {image_id: f1, ...}, ...}, ...}
        per_image_f1.csv   -- long format: model, seed, image_id, file_name, f1
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
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

RUN = "/root/vine-project/Run"
CKPT_ROOT = "/workspace/vine-project-Run"
COCO_TEST_DIR = f"{RUN}/rebuilt/final/test"
COCO_TEST_JSON = f"{COCO_TEST_DIR}/_annotations.coco.json"
YOLO_TEST_IMG_DIR = f"{RUN}/rebuilt/final_yolo/test/images"

IMG = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44]


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


def macro_f1_from_cm(cm):
    """Per-image macro-F1, skipping only classes absent from BOTH GT and prediction in
    this image (tp+fp+fn==0) -- matches this project's own established per-image
    mIoU/Dice convention (see compute_mIoU/compute_dice in train_unet.py etc, which do
    'nan if union==0 else ...' then nanmean). Without this, a per-image macro-F1 over all
    8 classes would be swamped by trivial F1=0 contributions from the 5-6 species that
    simply don't appear in any single photo (each image typically shows only background +
    1-2 species), which has nothing to do with model quality and would dominate/distort
    the metric used for the pairwise significance test.

    Classes that DO appear (in GT and/or prediction) use the standard zero_division=0
    convention per-class -- e.g. a class hallucinated by the model but absent from GT
    (tp=0, fn=0, fp>0) is included and correctly scored F1=0, it is not skipped."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    f1_per_class = []
    for i in range(len(tp)):
        if tp[i] + fp[i] + fn[i] == 0:
            continue  # class absent from both GT and prediction in this image -- skip
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)
    return float(np.mean(f1_per_class)) if f1_per_class else float("nan")


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
log(f"Loaded COCO test set: {N_TEST} images")

GT_512 = {}
for iid in coco_test.imgs:
    GT_512[iid] = resize_map(gt_pixel_map_native(coco_test, iid))


def ckpt_suffix(seed):
    return "" if seed == 42 else f"_seed{seed}"


def per_image_f1_from_preds(preds_512_by_iid, labels=range(8)):
    out = {}
    for iid, pred in preds_512_by_iid.items():
        cm = sk_confusion_matrix(GT_512[iid].reshape(-1), pred.reshape(-1), labels=list(labels))
        out[iid] = macro_f1_from_cm(cm)
    return out


# ===========================================================================
def eval_unet(ckpt_path):
    mod = load_module("train_unet_mod", f"{RUN}/UNet/train_unet.py")
    model = mod.UNet(3, 8).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()
    preds = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            img = Image.open(os.path.join(COCO_TEST_DIR, info["file_name"])).convert("RGB")
            x = to_tensor(img.resize((IMG, IMG), Image.BILINEAR)).unsqueeze(0).to(device)
            preds[iid] = torch.argmax(model(x), dim=1)[0].cpu().numpy().astype(np.uint8)
    return per_image_f1_from_preds(preds)


def eval_deeplabv3(ckpt_path):
    mod = load_module("train_deeplabv3_mod", f"{RUN}/DeepLabV3/train_deeplabv3.py")
    model = mod.build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()
    preds = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            img = Image.open(os.path.join(COCO_TEST_DIR, info["file_name"])).convert("RGB")
            x = to_tensor(img.resize((IMG, IMG), Image.BILINEAR)).unsqueeze(0).to(device)
            preds[iid] = torch.argmax(model(x)["out"], dim=1)[0].cpu().numpy().astype(np.uint8)
    return per_image_f1_from_preds(preds)


def eval_segformer(ckpt_path):
    mod = load_module("train_segformer_mod", f"{RUN}/SegFormer/train_segformer.py")
    model = mod.build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor(); norm = mod.NORM
    preds = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            img = Image.open(os.path.join(COCO_TEST_DIR, info["file_name"])).convert("RGB")
            x = norm(to_tensor(img.resize((IMG, IMG), Image.BILINEAR))).unsqueeze(0).to(device)
            logits = mod.forward_logits(model, x)
            preds[iid] = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return per_image_f1_from_preds(preds)


def eval_maskrcnn(ckpt_path):
    mod = load_module("train_maskrcnn_mod", f"{RUN}/MaskRCNN/train_maskrcnn.py")
    model = mod.build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()
    CONF = 0.5
    preds = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            path = os.path.join(COCO_TEST_DIR, info["file_name"])
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            x = to_tensor(cv2.resize(img, (IMG, IMG))).to(device)
            out = model([x])[0]
            class_map = np.zeros((IMG, IMG), np.uint8)
            order = torch.argsort(out["scores"])
            for i in order.tolist():
                if float(out["scores"][i]) < CONF:
                    continue
                m512 = (out["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
                if m512.sum() == 0:
                    continue
                class_map[m512 > 0] = int(out["labels"][i])
            preds[iid] = class_map
    return per_image_f1_from_preds(preds)


def eval_solov2(ckpt_path):
    mod = load_module("train_solov2_clean_mod", f"{RUN}/SoloV2/train_solov2_clean.py")
    model = mod.SimplifiedSOLOv2(num_classes=8).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    transform = T.Compose([
        T.ToPILImage(), T.Resize((IMG, IMG)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    out_f1 = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            path = os.path.join(COCO_TEST_DIR, info["file_name"])
            img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            x = transform(img_rgb).unsqueeze(0).to(device)
            category_pred, _ = model(x)
            Hc, Wc = category_pred.shape[-2:]
            pred_coarse = torch.argmax(category_pred, dim=1)[0].cpu().numpy().astype(np.uint8)
            gt_coarse = cv2.resize(GT_512[iid], (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            cm = sk_confusion_matrix(gt_coarse.reshape(-1), pred_coarse.reshape(-1), labels=list(range(8)))
            out_f1[iid] = macro_f1_from_cm(cm)
    return out_f1


def eval_yolo_model(weights_path, extra_module_dir=None):
    if extra_module_dir:
        sys.path.insert(0, extra_module_dir)
        import pgss_modules  # noqa: F401
    from ultralytics import YOLO
    model = YOLO(weights_path)
    CONF = 0.25
    yolo_device = 0 if torch.cuda.is_available() else "cpu"
    preds = {}
    for iid, info in coco_test.imgs.items():
        img_path = os.path.join(YOLO_TEST_IMG_DIR, info["file_name"])
        r = model.predict(img_path, imgsz=IMG, conf=CONF, device=yolo_device, verbose=False)[0]
        H0, W0 = r.orig_shape
        canvas512 = np.zeros((IMG, IMG), np.uint8)
        if r.masks is not None and len(r.masks) > 0:
            polys = r.masks.xy
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            order = np.argsort(confs)
            canvas_native = np.zeros((H0, W0), np.uint8)
            for idx in order:
                poly = polys[idx]
                if poly.shape[0] < 3:
                    continue
                cv2.fillPoly(canvas_native, [poly.astype(np.int32)], int(clss[idx]) + 1)
            canvas512 = resize_map(canvas_native)
        preds[iid] = canvas512
    if extra_module_dir:
        sys.path.pop(0)
    return per_image_f1_from_preds(preds)


# ===========================================================================
if __name__ == "__main__":
    t0 = time.time()
    steps = [
        ("UNet", lambda seed: eval_unet(f"{CKPT_ROOT}/UNet/checkpoints{ckpt_suffix(seed)}/best_model.pth")),
        ("DeepLabV3", lambda seed: eval_deeplabv3(f"{CKPT_ROOT}/DeepLabV3/checkpoints{ckpt_suffix(seed)}/best_model.pth")),
        ("SegFormer", lambda seed: eval_segformer(f"{CKPT_ROOT}/SegFormer/checkpoints{ckpt_suffix(seed)}/best_model.pth")),
        ("MaskRCNN", lambda seed: eval_maskrcnn(f"{CKPT_ROOT}/MaskRCNN/checkpoints{ckpt_suffix(seed)}/best_model.pth")),
        ("SoloV2", lambda seed: eval_solov2(f"{CKPT_ROOT}/SoloV2/checkpoints_clean{ckpt_suffix(seed)}/best_model.pth")),
        ("YOLO11seg", lambda seed: eval_yolo_model(
            f"{CKPT_ROOT}/YOLO11seg/runs/vine_clean_v6{ckpt_suffix(seed)}/weights/best_dice.pt")),
        ("PGSS_YOLO11", lambda seed: eval_yolo_model(
            f"{CKPT_ROOT}/PGSS_YOLO11/runs/pgss_vine_v6{ckpt_suffix(seed)}/weights/best_dice.pt",
            extra_module_dir=f"{RUN}/PGSS_YOLO11")),
    ]

    RESULTS = {name: {} for name, _ in steps}
    for name, fn in steps:
        for seed in SEEDS:
            log(f"=== {name} seed {seed} ===")
            try:
                f1_by_iid = fn(seed)
                RESULTS[name][seed] = f1_by_iid
                vals = list(f1_by_iid.values())
                log(f"{name} seed {seed}: mean per-image F1 = {np.mean(vals):.4f}")
            except Exception as e:
                log(f"!!! {name} seed {seed} FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                RESULTS[name][seed] = {"error": str(e)}
        with open(f"{RUN}/per_image_f1.json", "w") as f:
            json.dump(RESULTS, f, indent=2)

    # long-format CSV
    import csv
    with open(f"{RUN}/per_image_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "image_id", "file_name", "f1"])
        for name, _ in steps:
            for seed in SEEDS:
                d = RESULTS[name][seed]
                if "error" in d:
                    continue
                for iid, f1 in d.items():
                    fname = coco_test.imgs[iid]["file_name"]
                    w.writerow([name, seed, iid, fname, f1])
    log(f"ALL DONE in {time.time()-t0:.1f}s")
