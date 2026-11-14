"""
unified_pixel_dice.py
=====================
ONE checkpoint-selection criterion for all seven architectures (addresses Reviewer comment 2).

Criterion: macro-averaged per-class Dice over the 8 classes (background + 7 species), computed
from VALIDATION-SET-POOLED pixel counts (TP/FP/FN summed over all validation images, then Dice per
class, then the mean over classes present in the validation ground truth or prediction), on the
full model input grid (512 x 512) for every model. This is the same pixel-level quantity used for
test-set reporting, so selection and evaluation are aligned.

How each architecture is converted to a 512 x 512 semantic label map BEFORE scoring:
  * U-Net / DeepLabv3 / SegFormer : argmax over class logits at 512 x 512 (upsample logits first if
                                    the network outputs a lower resolution, e.g. SegFormer 1/4).
  * Simplified SOLO-style baseline: bilinearly upsample the stride-32 category logits (16 x 16) and
                                    objectness map to 512 x 512, then argmax / threshold. (The
                                    previous selection used the native 16 x 16 grid.)
  * Mask R-CNN / YOLO11-seg / YOLO11-PGSS: paint instance masks (binarised at 0.5) whose score >=
                                    `score_thr` onto a background map in ASCENDING score order, so
                                    that where instances overlap the highest-scoring instance wins.
                                    The SAME score threshold must be used for every instance model
                                    (default 0.25; report it).

Epoch selection (uniform across models):
  Score every epoch on the validation set with `pooled_macro_dice()` and keep the best epoch
  (ties -> earliest). The same score drives early stopping, with one rule for every model. The
  LOTO training scripts (loto_retraining/, class loto_common.EpochSelector) do this online after
  each epoch, with R1's rule (at most 100 epochs, stop after 15 epochs without improvement).
  If every checkpoint is saved instead, `select_best_epoch()` makes the same choice afterwards.
  Report the selected epoch and the number of epochs run for every model/fold/seed.

Run `python -I unified_pixel_dice.py --selftest` to check the metric on synthetic arrays.
(The self-test uses synthetic data only; it is NOT a result.)
"""
import numpy as np

NUM_CLASSES = 8


# ----------------------------------------------------------------------------------------------
# Core metric
# ----------------------------------------------------------------------------------------------
class PooledDice:
    """Accumulate per-class TP/FP/FN over many images; compute macro Dice once at the end."""

    def __init__(self, num_classes=NUM_CLASSES):
        self.k = num_classes
        self.tp = np.zeros(num_classes, dtype=np.int64)
        self.fp = np.zeros(num_classes, dtype=np.int64)
        self.fn = np.zeros(num_classes, dtype=np.int64)

    def update(self, pred, gt):
        pred = np.asarray(pred).ravel(); gt = np.asarray(gt).ravel()
        assert pred.shape == gt.shape, "prediction and ground truth must share the same grid"
        cm = np.bincount(gt * self.k + pred, minlength=self.k * self.k).reshape(self.k, self.k)
        tp = np.diag(cm)
        self.tp += tp
        self.fp += cm.sum(0) - tp
        self.fn += cm.sum(1) - tp

    def per_class(self):
        denom = 2 * self.tp + self.fp + self.fn
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(denom > 0, 2 * self.tp / denom, np.nan)
        return d

    def macro(self):
        return float(np.nanmean(self.per_class()))


def pooled_macro_dice(pairs, num_classes=NUM_CLASSES):
    """pairs: iterable of (pred_label_map, gt_label_map) on the SAME 512x512 grid."""
    acc = PooledDice(num_classes)
    for p, g in pairs:
        acc.update(p, g)
    return acc.macro()


# ----------------------------------------------------------------------------------------------
# Converters: model output -> 512x512 semantic label map
# ----------------------------------------------------------------------------------------------
def upsample_bilinear(arr, size):
    """arr: (C,h,w) float array -> (C,H,W). Pure-numpy separable bilinear (align_corners=False)."""
    C, h, w = arr.shape; H, W = size
    ys = (np.arange(H) + 0.5) * h / H - 0.5; xs = (np.arange(W) + 0.5) * w / W - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1); x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]; wx = np.clip(xs - x0, 0, 1)[None, :]
    out = np.empty((C, H, W), dtype=np.float32)
    for c in range(C):
        a = arr[c]
        top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
        bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
        out[c] = top * (1 - wy) + bot * wy
    return out


def semantic_from_logits(logits, size=(512, 512)):
    """logits: (C,h,w). Upsample if needed, then argmax."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape[1:] != tuple(size):
        logits = upsample_bilinear(logits, size)
    return logits.argmax(0).astype(np.int64)


def semantic_from_solo(category_logits, objectness_logit, size=(512, 512), obj_thr=0.5):
    """Simplified SOLO-style baseline: category logits (C,16,16) + objectness (1,16,16)."""
    cat = upsample_bilinear(np.asarray(category_logits, np.float32), size)
    obj = upsample_bilinear(np.asarray(objectness_logit, np.float32), size)[0]
    lab = cat.argmax(0)
    lab[(1 / (1 + np.exp(-obj))) <= obj_thr] = 0
    return lab.astype(np.int64)


def semantic_from_instances(masks, classes, scores, size=(512, 512), score_thr=0.25, mask_thr=0.5):
    """masks: (N,H,W) float/bool at `size`; classes: (N,) in 1..7; scores: (N,).
    Instances below score_thr are dropped; remaining are painted in ascending score order."""
    lab = np.zeros(size, dtype=np.int64)
    if masks is None or len(scores) == 0:
        return lab
    order = np.argsort(np.asarray(scores))
    for i in order:
        if scores[i] < score_thr:
            continue
        lab[np.asarray(masks[i]) > mask_thr] = int(classes[i])
    return lab


def ultralytics_result_to_semantic(result, size=(512, 512), score_thr=0.25):
    """Convert one ultralytics Results object (segment task) to a semantic map.
    Assumes the dataset's class indices 0..6 map to labels 1..7 (background = 0); adjust if not."""
    if result.masks is None:
        return np.zeros(size, dtype=np.int64)
    m = result.masks.data.cpu().numpy()                      # (N,h,w) at network resolution
    if m.shape[1:] != tuple(size):
        m = upsample_bilinear(m.astype(np.float32), size)
    cls = result.boxes.cls.cpu().numpy().astype(int) + 1
    conf = result.boxes.conf.cpu().numpy()
    return semantic_from_instances(m, cls, conf, size, score_thr)


# ----------------------------------------------------------------------------------------------
# Post-hoc epoch selection
# ----------------------------------------------------------------------------------------------
def select_best_epoch(epoch_checkpoints, predict_fn, val_items, num_classes=NUM_CLASSES):
    """epoch_checkpoints: dict {epoch: checkpoint_path}
    predict_fn(checkpoint_path, image) -> 512x512 semantic label map (use a converter above)
    val_items: list of (image, gt_label_map_512)
    Returns (best_epoch, {epoch: macro_dice})."""
    scores = {}
    for ep, ck in sorted(epoch_checkpoints.items()):
        scores[ep] = pooled_macro_dice(((predict_fn(ck, img), gt) for img, gt in val_items), num_classes)
    best = max(scores, key=lambda e: (scores[e], -e))  # ties -> earliest epoch
    return best, scores


# ----------------------------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 8, size=(512, 512))
    assert abs(pooled_macro_dice([(gt, gt)]) - 1.0) < 1e-12
    pred = gt.copy(); pred[:256] = 0
    acc = PooledDice(); acc.update(pred, gt)
    d = acc.per_class()
    # brute-force check of class 3
    tp = ((pred == 3) & (gt == 3)).sum(); fp = ((pred == 3) & (gt != 3)).sum(); fn = ((pred != 3) & (gt == 3)).sum()
    assert abs(d[3] - 2 * tp / (2 * tp + fp + fn)) < 1e-12
    # pooled != mean of per-image values in general (documented behaviour)
    logits = rng.normal(size=(8, 16, 16)); lab = semantic_from_logits(logits)
    assert lab.shape == (512, 512)
    m = np.zeros((2, 512, 512)); m[0, :100] = 1; m[1, 50:150] = 1
    sem = semantic_from_instances(m, np.array([2, 5]), np.array([0.9, 0.3]))
    assert sem[60, 0] == 2 and sem[120, 0] == 5 and sem[300, 0] == 0  # higher score wins overlap
    sem2 = semantic_from_instances(m, np.array([2, 5]), np.array([0.9, 0.1]))
    assert sem2[120, 0] == 0  # below threshold dropped
    print("unified_pixel_dice self-test passed (synthetic arrays only).")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
