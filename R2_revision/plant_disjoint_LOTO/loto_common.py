"""
loto_common.py
==============
Shared code for the leave-one-treatment-out (LOTO) retraining of the seven R1 models.

It provides three things, used identically by all seven LOTO training scripts:

1. run_config(): the run settings (fold, seed, data folder, output folder, maximum epochs,
   early-stopping patience), taken from environment variables set by run_step1_training.sh.
   Nothing is hard-coded per model.

2. Ground truth and predictors that mirror run_full_test_evaluation.py (GitHub b2f792f), so that
   the epoch-selection score on the validation images is computed exactly as the test metrics:
     * ground truth: COCO polygons filled with cv2.fillPoly at the stored (640 x 640 letterboxed)
       resolution, then nearest-neighbour resized to 512 x 512;
     * U-Net / DeepLabv3 / SegFormer: PIL bilinear resize to 512, ToTensor (+ ImageNet
       normalisation for SegFormer), arg-max of the class logits;
     * Mask R-CNN: masks > 0.5 of detections with score >= SCORE_THR painted in ascending score
       order on the 512 grid;
     * simplified SOLOv2 baseline: 16 x 16 class and objectness logits bilinearly upsampled to
       512, arg-max class where sigmoid(objectness) > 0.5 (Section 2.5 of the R2 manuscript; R1
       selected this model on the native 16 x 16 grid);
     * YOLO11-seg / YOLO11-PGSS: Ultralytics predict at imgsz 512 with conf = SCORE_THR, polygons
       painted at native resolution in ascending score order, nearest resize to 512.
   The SAME score threshold (SCORE_THR) is used for every instance model.

3. EpochSelector: after every epoch it scores the current weights on the validation images with
   the pooled 8-class macro Dice (unified_pixel_dice.PooledDice: TP/FP/FN summed over all
   validation pixels, Dice per class, mean over classes present) and keeps the checkpoint of the
   best epoch (ties -> earliest epoch). The same score drives early stopping, with R1's rule for
   all seven models: at most 100 epochs, stop after 15 consecutive epochs without improvement.
   (In R1 the rule was the same but the score differed between models; that is what changes.)
   Every epoch's score is written to selection_log.csv; selection_summary.json records the
   selected epoch and the number of epochs run (report both in Table S9).

Do not change SCORE_THR, the epoch limit, the patience or the selection rule after looking at
test results.
"""
import csv
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # author_scripts/ (unified_pixel_dice.py)
from unified_pixel_dice import PooledDice  # noqa: E402

IMG = 512
NUM_CLASSES = 8
CLASS_NAMES = ["background", "Akebia", "Campsis", "Hydrangea", "Lonicera",
               "Madame Galen", "Parthenocissus", "Wisteria"]
SCORE_THR = 0.25        # common instance score threshold for semantic conversion (pre-specified)
SOLO_OBJ_THR = 0.5      # objectness threshold of the simplified SOLOv2 baseline (as in R1)
DEFAULT_EPOCHS = 100    # maximum number of epochs (as in R1; pre-specified)
DEFAULT_PATIENCE = 15   # stop after this many epochs without improvement (as in R1; pre-specified)


# ------------------------------------------------------------------------------------------------
# 1. Run configuration
# ------------------------------------------------------------------------------------------------
def run_config(model_name):
    """Read the run settings from the environment (see run_step1_training.sh)."""
    need = ["LOTO_FOLD", "LOTO_DATA_ROOT", "LOTO_OUT_ROOT"]
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"[loto] missing environment variables {missing}; "
                         f"launch through run_step1_training.sh")
    fold = os.environ["LOTO_FOLD"]
    seed = int(os.environ.get("LOTO_SEED", "42"))
    data_root = os.path.abspath(os.environ["LOTO_DATA_ROOT"])          # <out>/<fold>/rebuilt/final
    out_root = os.path.abspath(os.environ["LOTO_OUT_ROOT"])
    max_epochs = int(os.environ.get("LOTO_MAX_EPOCHS", str(DEFAULT_EPOCHS)))
    if max_epochs != DEFAULT_EPOCHS:
        print(f"[loto] WARNING: LOTO_MAX_EPOCHS={max_epochs} differs from the pre-specified "
              f"{DEFAULT_EPOCHS}; use this only for smoke tests.")
    patience = int(os.environ.get("LOTO_PATIENCE", str(DEFAULT_PATIENCE)))
    if patience != DEFAULT_PATIENCE:
        print(f"[loto] WARNING: LOTO_PATIENCE={patience} differs from the pre-specified "
              f"{DEFAULT_PATIENCE}; use this only for smoke tests.")

    for split in ("train", "valid"):
        if not os.path.isfile(os.path.join(data_root, split, "_annotations.coco.json")):
            raise SystemExit(f"[loto] {data_root}/{split}/_annotations.coco.json not found")
    # the fold marker written by step0_build_loto_datasets.py must name the same fold
    marker = os.path.join(os.path.dirname(os.path.dirname(data_root)), "FOLD.json")
    if not os.path.isfile(marker):
        raise SystemExit(f"[loto] {marker} not found: LOTO_DATA_ROOT must be "
                         f"<out>/<fold>/rebuilt/final as written by step 0")
    fold_info = json.load(open(marker))
    if fold_info.get("fold") != fold:
        raise SystemExit(f"[loto] LOTO_FOLD={fold} but {marker} says {fold_info.get('fold')}")

    save_dir = os.path.join(out_root, fold, f"seed{seed}", model_name)
    os.makedirs(save_dir, exist_ok=True)
    cfg = {
        "model": model_name, "fold": fold, "seed": seed, "data_root": data_root,
        "yolo_data": os.path.join(os.path.dirname(data_root), "final_yolo", "data.yaml"),
        "save_dir": save_dir, "max_epochs": max_epochs, "patience": patience,
        "selection_criterion": "pooled 8-class macro pixel Dice on the 512x512 grid, validation set",
        "early_stopping": f"stop after {patience} consecutive epochs without improvement of the "
                          f"selection criterion (at most {max_epochs} epochs)",
        "instance_score_threshold": SCORE_THR, "solo_objectness_threshold": SOLO_OBJ_THR,
        "fold_marker": fold_info,
    }
    json.dump(cfg, open(os.path.join(save_dir, "loto_run_config.json"), "w"), indent=2)
    print(f"[loto] {model_name} | fold {fold} | seed {seed} | max {max_epochs} epochs, patience "
          f"{patience} | data {data_root} | out {save_dir}")
    return cfg


# ------------------------------------------------------------------------------------------------
# 2. Ground truth and predictors (mirror run_full_test_evaluation.py)
# ------------------------------------------------------------------------------------------------
def gt_512(coco, iid):
    """run_full_test_evaluation.gt_pixel_map_native() followed by resize_map()."""
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
    return cv2.resize(canvas, (IMG, IMG), interpolation=cv2.INTER_NEAREST)


def load_split(split_dir):
    """[(image_path, gt_512)] for every image of a COCO split folder, in image-id order."""
    from pycocotools.coco import COCO
    coco = COCO(os.path.join(split_dir, "_annotations.coco.json"))
    return [(os.path.join(split_dir, coco.imgs[i]["file_name"]), gt_512(coco, i))
            for i in sorted(coco.imgs)]


def make_predictor(kind, device, segformer_forward=None, segformer_norm=None):
    """Returns predict(model, image_path) -> uint8 (512, 512) label map (0 = background)."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms as T
    to_tensor = T.ToTensor()

    if kind in ("unet", "deeplabv3", "segformer"):
        def predict(model, path):
            img = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
            x = to_tensor(img)
            if kind == "segformer":
                x = segformer_norm(x)
            x = x.unsqueeze(0).to(device)
            if kind == "unet":
                logits = model(x)
            elif kind == "deeplabv3":
                logits = model(x)["out"]
            else:
                logits = segformer_forward(model, x)
            return torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        return predict

    if kind == "maskrcnn":
        def predict(model, path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            x = to_tensor(cv2.resize(img, (IMG, IMG))).to(device)
            out = model([x])[0]
            lab = np.zeros((IMG, IMG), np.uint8)
            for i in torch.argsort(out["scores"]).tolist():          # ascending score
                if float(out["scores"][i]) < SCORE_THR:
                    continue
                m = (out["masks"][i, 0] > 0.5).cpu().numpy()
                lab[m] = int(out["labels"][i])
            return lab
        return predict

    if kind == "solov2":
        tf = T.Compose([T.ToPILImage(), T.Resize((IMG, IMG)), T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        def predict(model, path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            cat, obj = model(tf(img).unsqueeze(0).to(device))
            cat = F.interpolate(cat, size=(IMG, IMG), mode="bilinear", align_corners=False)[0]
            obj = torch.sigmoid(F.interpolate(obj, size=(IMG, IMG), mode="bilinear",
                                              align_corners=False))[0, 0]
            lab = torch.argmax(cat, dim=0)
            lab[obj <= SOLO_OBJ_THR] = 0
            return lab.cpu().numpy().astype(np.uint8)
        return predict

    if kind == "yolo":
        def predict(yolo, path):
            r = yolo.predict(path, imgsz=IMG, conf=SCORE_THR, device=device, verbose=False)[0]
            H0, W0 = r.orig_shape
            canvas = np.zeros((H0, W0), np.uint8)
            if r.masks is not None and len(r.masks) > 0:
                polys = r.masks.xy
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for idx in np.argsort(confs):                       # ascending score
                    poly = polys[idx]
                    if poly.shape[0] < 3:
                        continue
                    cv2.fillPoly(canvas, [poly.astype(np.int32)], int(clss[idx]) + 1)  # 0..6 -> 1..7
            return cv2.resize(canvas, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        return predict

    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------
# 3. Uniform epoch selection
# ------------------------------------------------------------------------------------------------
class EpochSelector:
    def __init__(self, save_dir, valid_dir, predict, model_name):
        self.save_dir, self.predict, self.model_name = save_dir, predict, model_name
        self.items = load_split(valid_dir)
        self.best, self.best_epoch, self.n_epochs = -1.0, None, 0
        self.log_path = os.path.join(save_dir, "selection_log.csv")
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "val_pooled_macro_dice_512", "selected_so_far",
                                    "best_epoch_so_far", "score_seconds"]
                                   + [f"dice_{c}" for c in CLASS_NAMES] + ["notes"])
        print(f"[loto-select] {model_name}: {len(self.items)} validation images; criterion = pooled "
              f"8-class macro Dice at {IMG}x{IMG}; instance score >= {SCORE_THR}")

    def score(self, model, toggle_mode=True):
        """Pooled macro Dice of `model` on the validation images.
        toggle_mode=True (plain PyTorch models): switch to eval() and restore train() afterwards.
        toggle_mode=False (Ultralytics YOLO objects): never call .eval()/.train() on them, because
        Ultralytics overrides Model.train() to START A TRAINING RUN (and nn.Module.eval() calls it)."""
        import torch
        was_training = bool(model.training) if toggle_mode else False
        if toggle_mode:
            model.eval()
        t0 = time.time()
        acc = PooledDice(NUM_CLASSES)
        with torch.no_grad():
            for path, gt in self.items:
                acc.update(self.predict(model, path).astype(np.int64), gt.astype(np.int64))
        if toggle_mode and was_training:
            model.train()
        self._last = (acc.macro(), acc.per_class(), time.time() - t0)
        return self._last[0]

    def update(self, epoch1, score, notes=""):
        """Record epoch `epoch1` (1-based). Returns True if it is the new best (strictly greater)."""
        self.n_epochs = max(self.n_epochs, epoch1)
        improved = score > self.best
        if improved:
            self.best, self.best_epoch = score, epoch1
        _, per_class, secs = self._last
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch1, f"{score:.6f}", int(improved), self.best_epoch, f"{secs:.1f}"]
                                   + ["" if np.isnan(d) else f"{d:.6f}" for d in per_class] + [notes])
        print(f"[loto-select] {self.model_name} epoch {epoch1}: pooled val Dice {score:.4f}"
              f"{'  <- new best' if improved else ''} (best {self.best:.4f} @ epoch {self.best_epoch})")
        return improved

    def finalize(self, epochs_run, checkpoint):
        summary = {"model": self.model_name, "epochs_run": int(epochs_run),
                   "selected_epoch": self.best_epoch, "val_pooled_macro_dice_512": self.best,
                   "n_validation_images": len(self.items), "checkpoint": checkpoint,
                   "criterion": "pooled 8-class macro pixel Dice, 512x512, validation set; ties -> earliest",
                   "instance_score_threshold": SCORE_THR}
        cfg_path = os.path.join(self.save_dir, "loto_run_config.json")
        if os.path.isfile(cfg_path):
            cfg = json.load(open(cfg_path))
            summary.update(max_epochs=cfg.get("max_epochs"), patience=cfg.get("patience"),
                           stopped_early=int(epochs_run) < int(cfg.get("max_epochs", 0)))
        json.dump(summary, open(os.path.join(self.save_dir, "selection_summary.json"), "w"), indent=2)
        print(f"[loto-select] {self.model_name}: selected epoch {self.best_epoch} of {epochs_run} "
              f"(pooled val Dice {self.best:.4f}) -> {checkpoint}")
        return summary


def yolo_selection_callbacks(valid_dir, run_dir, device, model_name, patience):
    """Ultralytics callbacks: after every epoch, score weights/last.pt on the validation images with
    the uniform criterion and copy it to weights/best_unified.pt when it is the best so far.
    Training stops (trainer.stop = True, as in R1's make_dice_early_stopper) after `patience`
    consecutive epochs without improvement. The R1 quantity (F1 from Ultralytics
    precision(M)/recall(M)) is logged in the notes column for transparency only."""
    from ultralytics import YOLO
    import torch
    selector = EpochSelector(run_dir, valid_dir, make_predictor("yolo", device), model_name)
    state = {"t0": None, "wait": 0, "in_epoch": False}

    def on_train_epoch_start(trainer):
        state["t0"] = time.time()
        state["in_epoch"] = True

    def on_fit_epoch_end(trainer):
        epoch1 = trainer.epoch + 1
        if not state["in_epoch"] or epoch1 > trainer.epochs or epoch1 <= selector.n_epochs:
            # Ultralytics' final_eval() validates best.pt after training, increments trainer.epoch
            # and fires this callback once more. No training epoch started before that call, so it
            # is recognised by the in_epoch flag (works after early stopping and after the last
            # epoch). It is not a training epoch and is ignored.
            print(f"[loto-select] {model_name}: ignoring post-training callback (epoch index {epoch1})")
            return
        state["in_epoch"] = False
        m = trainer.metrics or {}
        p = next((v for k, v in m.items() if "precision(M)" in k), None)
        r = next((v for k, v in m.items() if "recall(M)" in k), None)
        r1 = float("nan") if p is None or r is None else 2 * p * r / (p + r + 1e-9)
        last = str(trainer.last)
        yolo = YOLO(last)
        s = selector.score(yolo, toggle_mode=False)
        del yolo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        epoch_time = time.time() - state["t0"] if state["t0"] else float("nan")
        if selector.update(epoch1, s, notes=f"r1_F1_PR(M)={r1:.4f}; epoch_s={epoch_time:.0f}"):
            shutil.copy2(last, os.path.join(trainer.save_dir, "weights", "best_unified.pt"))
            state["wait"] = 0
        else:
            state["wait"] += 1
        if state["wait"] >= patience:
            print(f"[loto-select] {model_name}: early stopping, no improvement in {patience} epochs "
                  f"(best {selector.best:.4f} at epoch {selector.best_epoch})")
            trainer.stop = True

    return on_train_epoch_start, on_fit_epoch_end, selector
