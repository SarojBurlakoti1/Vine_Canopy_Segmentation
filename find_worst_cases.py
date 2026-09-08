"""
Find the 10 lowest per-image mIoU test images per model (seed 42), and generate
[Original | GT | Prediction] overlay grids for genuine visual failure-mode inspection
(not inferred from aggregate numbers).

Per-image mIoU: macro IoU over classes present in GT or prediction (skip only classes
absent from BOTH, same convention as the per-image F1 fix used for the pairwise stats --
consistent methodology, avoids the trivial-zero-from-absent-classes artifact).
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
from pycocotools.coco import COCO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "/root/vine-project/Run"
CKPT_ROOT = "/workspace/vine-project-Run"
COCO_TEST_DIR = f"{RUN}/rebuilt/final/test"
COCO_TEST_JSON = f"{COCO_TEST_DIR}/_annotations.coco.json"
YOLO_TEST_IMG_DIR = f"{RUN}/rebuilt/final_yolo/test/images"
IMG = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["background", "Akebia", "Campsis", "Hydrangea", "Lonicera",
               "Madame Galen", "Parthenocissus", "Wisteria"]
COLORS = np.array([
    [0, 0, 0],        # background -> no overlay
    [230, 25, 75],    # Akebia - red
    [60, 180, 75],    # Campsis - green
    [255, 225, 25],   # Hydrangea - yellow
    [0, 130, 200],    # Lonicera - blue
    [245, 130, 48],   # Madame Galen - orange
    [145, 30, 180],   # Parthenocissus - purple
    [70, 240, 240],   # Wisteria - cyan
], dtype=np.uint8)


def log(*a):
    print(*a, flush=True)


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


def macro_iou_and_cm(gt_map, pred_map):
    labels = list(range(8))
    from sklearn.metrics import confusion_matrix as skcm
    cm = skcm(gt_map.reshape(-1), pred_map.reshape(-1), labels=labels)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    ious = []
    for i in range(8):
        if tp[i] + fp[i] + fn[i] == 0:
            continue
        ious.append(tp[i] / (tp[i] + fp[i] + fn[i]))
    return (float(np.mean(ious)) if ious else float("nan")), cm


coco_test = COCO(COCO_TEST_JSON)
N_TEST = len(coco_test.imgs)
log(f"Loaded {N_TEST} test images")

# native-res GT maps + native RGB images, cached once
GT_NATIVE = {}
IMG_RGB_NATIVE = {}
for iid, info in coco_test.imgs.items():
    GT_NATIVE[iid] = gt_pixel_map_native(coco_test, iid)
    p = os.path.join(COCO_TEST_DIR, info["file_name"])
    IMG_RGB_NATIVE[iid] = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def colorize(class_map):
    return COLORS[class_map]


def overlay(img_rgb, class_map, alpha=0.55):
    color = colorize(class_map)
    fg = class_map > 0
    out = img_rgb.copy().astype(np.float32)
    out[fg] = out[fg] * (1 - alpha) + color[fg].astype(np.float32) * alpha
    return out.astype(np.uint8)


# ===========================================================================
def predict_unet(ckpt_path):
    mod = load_module("m_unet", f"{RUN}/UNet/train_unet.py")
    model = mod.UNet(3, 8).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()
    preds = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            H0, W0 = info["height"], info["width"]
            img = IMG_RGB_NATIVE[iid]
            x = to_tensor(cv2.resize(img, (IMG, IMG))).unsqueeze(0).to(device)
            p512 = torch.argmax(model(x), dim=1)[0].cpu().numpy().astype(np.uint8)
            preds[iid] = cv2.resize(p512, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return preds


def predict_deeplabv3(ckpt_path):
    mod = load_module("m_dl", f"{RUN}/DeepLabV3/train_deeplabv3.py")
    model = mod.build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()
    preds = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            H0, W0 = info["height"], info["width"]
            img = IMG_RGB_NATIVE[iid]
            x = to_tensor(cv2.resize(img, (IMG, IMG))).unsqueeze(0).to(device)
            p512 = torch.argmax(model(x)["out"], dim=1)[0].cpu().numpy().astype(np.uint8)
            preds[iid] = cv2.resize(p512, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return preds


def predict_segformer(ckpt_path):
    mod = load_module("m_sf", f"{RUN}/SegFormer/train_segformer.py")
    model = mod.build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor(); norm = mod.NORM
    preds = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            H0, W0 = info["height"], info["width"]
            img = IMG_RGB_NATIVE[iid]
            x = norm(to_tensor(cv2.resize(img, (IMG, IMG)))).unsqueeze(0).to(device)
            logits = mod.forward_logits(model, x)
            p512 = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            preds[iid] = cv2.resize(p512, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return preds


def predict_maskrcnn(ckpt_path):
    mod = load_module("m_mr", f"{RUN}/MaskRCNN/train_maskrcnn.py")
    model = mod.build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()
    CONF = 0.5
    preds = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            H0, W0 = info["height"], info["width"]
            img = IMG_RGB_NATIVE[iid]
            x = to_tensor(cv2.resize(img, (IMG, IMG))).to(device)
            out = model([x])[0]
            class_map = np.zeros((IMG, IMG), np.uint8)
            order = torch.argsort(out["scores"])
            for i in order.tolist():
                if float(out["scores"][i]) < CONF:
                    continue
                m512 = (out["masks"][i, 0] > 0.5).cpu().numpy().astype(np.uint8)
                class_map[m512 > 0] = int(out["labels"][i])
            preds[iid] = cv2.resize(class_map, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return preds


def predict_solov2(ckpt_path):
    mod = load_module("m_sv", f"{RUN}/SoloV2/train_solov2_clean.py")
    model = mod.SimplifiedSOLOv2(num_classes=8).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    transform = T.Compose([T.ToPILImage(), T.Resize((IMG, IMG)), T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    preds = {}
    with torch.no_grad():
        for iid, info in coco_test.imgs.items():
            H0, W0 = info["height"], info["width"]
            img = IMG_RGB_NATIVE[iid]
            x = transform(img).unsqueeze(0).to(device)
            category_pred, _ = model(x)
            pcoarse = torch.argmax(category_pred, dim=1)[0].cpu().numpy().astype(np.uint8)
            preds[iid] = cv2.resize(pcoarse, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return preds


def predict_yolo(weights_path, extra_module_dir=None):
    if extra_module_dir:
        sys.path.insert(0, extra_module_dir)
        import pgss_modules  # noqa: F401
    from ultralytics import YOLO
    model = YOLO(weights_path)
    yolo_device = 0 if torch.cuda.is_available() else "cpu"
    preds = {}
    for iid, info in coco_test.imgs.items():
        H0, W0 = info["height"], info["width"]
        img_path = os.path.join(YOLO_TEST_IMG_DIR, info["file_name"])
        r = model.predict(img_path, imgsz=IMG, conf=0.25, device=yolo_device, verbose=False)[0]
        canvas = np.zeros((H0, W0), np.uint8)
        if r.masks is not None and len(r.masks) > 0:
            polys = r.masks.xy
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            order = np.argsort(confs)
            for idx in order:
                poly = polys[idx]
                if poly.shape[0] < 3:
                    continue
                cv2.fillPoly(canvas, [poly.astype(np.int32)], int(clss[idx]) + 1)
        preds[iid] = canvas
    if extra_module_dir:
        sys.path.pop(0)
    return preds


MODELS = [
    ("UNet", lambda: predict_unet(f"{CKPT_ROOT}/UNet/checkpoints/best_model.pth")),
    ("DeepLabV3", lambda: predict_deeplabv3(f"{CKPT_ROOT}/DeepLabV3/checkpoints/best_model.pth")),
    ("SegFormer", lambda: predict_segformer(f"{CKPT_ROOT}/SegFormer/checkpoints/best_model.pth")),
    ("MaskRCNN", lambda: predict_maskrcnn(f"{CKPT_ROOT}/MaskRCNN/checkpoints/best_model.pth")),
    ("SoloV2", lambda: predict_solov2(f"{CKPT_ROOT}/SoloV2/checkpoints_clean/best_model.pth")),
    ("YOLO11seg", lambda: predict_yolo(f"{CKPT_ROOT}/YOLO11seg/runs/vine_clean_v6/weights/best_dice.pt")),
    ("PGSS_YOLO11", lambda: predict_yolo(f"{CKPT_ROOT}/PGSS_YOLO11/runs/pgss_vine_v6/weights/best_dice.pt",
                                          extra_module_dir=f"{RUN}/PGSS_YOLO11")),
]

ALL_WORST = {}
for name, fn in MODELS:
    log(f"=== {name} ===")
    preds = fn()
    per_image = []
    for iid in coco_test.imgs:
        miou, cm = macro_iou_and_cm(GT_NATIVE[iid], preds[iid])
        per_image.append((iid, miou))
    per_image.sort(key=lambda x: x[1])
    worst10 = per_image[:10]
    fname_by_iid = {iid: coco_test.imgs[iid]["file_name"] for iid, _ in worst10}
    ALL_WORST[name] = [{"image_id": iid, "file_name": fname_by_iid[iid], "miou": miou} for iid, miou in worst10]
    log(f"{name} worst mIoU: {[round(m,3) for _,m in worst10]}")

    # build a grid: rows=10 worst cases, cols=[original, GT, prediction]
    fig, axes = plt.subplots(10, 3, figsize=(12, 40))
    for row, (iid, miou) in enumerate(worst10):
        img = IMG_RGB_NATIVE[iid]
        gt = GT_NATIVE[iid]
        pred = preds[iid]
        axes[row, 0].imshow(img); axes[row, 0].axis("off")
        axes[row, 1].imshow(overlay(img, gt)); axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay(img, pred)); axes[row, 2].axis("off")
        fn_short = coco_test.imgs[iid]["file_name"][:40]
        axes[row, 0].set_title(f"{fn_short}\nmIoU={miou:.3f}", fontsize=7)
        axes[row, 1].set_title("GT", fontsize=8)
        axes[row, 2].set_title("Prediction", fontsize=8)
    # legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(COLORS[i]) / 255.0) for i in range(1, 8)]
    fig.legend(legend_handles, CLASS_NAMES[1:], loc="upper center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Worst 10 test images by per-image mIoU -- {name} (seed 42)", fontsize=13, y=1.01)
    plt.tight_layout()
    out = f"{RUN}/failure_analysis_{name}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out}")

with open(f"{RUN}/failure_analysis_worst_cases.json", "w") as f:
    json.dump(ALL_WORST, f, indent=2)
log("wrote failure_analysis_worst_cases.json")
