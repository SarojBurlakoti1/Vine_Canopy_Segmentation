"""
Mask R-CNN (torchvision maskrcnn_resnet50_fpn, COCO-pretrained) — instance segmentation
on the leakage-corrected clean split, matched to the other models: 512² input, 30 epochs,
same clean split. Instance targets (boxes/labels/masks) decoded from the COCO polygons.
"""
import os
import json
import random
import sys
import time
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import cv2
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "rebuilt", "final"))
SAVE_DIR = "/workspace/vine-project-Run/MaskRCNN/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
IMG = 512
NUM_CLASSES = 8   # background + 7 species (torchvision reserves label 0 for background)
CLASS_NAMES = ['background', 'Akebia', 'Campsis', 'Hydrangea',
               'Lonicera', 'Madame Galen', 'Parthenocissus', 'Wisteria']

SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15  # epochs with no val-Dice improvement before stopping


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_run_metadata(model, device, seed=SEED):
    return {
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "package_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
        },
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "per_epoch_time_sec": [],
    }


def compute_dice(preds, labels, num_classes):
    """Mean per-class Dice = 2|P∩G| / (|P|+|G|), skipping classes absent from both."""
    dices = []
    preds = preds.reshape(-1); labels = labels.reshape(-1)
    for cls in range(num_classes):
        pi = preds == cls; li = labels == cls
        inter = (pi & li).sum().float()
        denom = pi.sum().float() + li.sum().float()
        dices.append(float('nan') if denom == 0 else (2 * inter / denom).item())
    return torch.tensor(dices).nanmean().item()


def instances_to_semantic_map(masks, labels, scores=None, mask_thresh=0.5, score_thresh=0.5, size=(IMG, IMG), device=None):
    """Composite per-instance masks onto one [H,W] canvas indexed by class id (0=background),
    so instance predictions/targets can be compared with a pixel-wise Dice score just like
    the other models. With scores given (predictions), instances below score_thresh are
    dropped and the rest are drawn in ascending-score order so the highest-confidence mask
    wins on any overlap; ground-truth instances (no scores) are drawn in their given order."""
    if len(masks):
        device = masks.device
    elif device is None:
        device = torch.device('cpu')
    canvas = torch.zeros(size, dtype=torch.long, device=device)
    order = range(len(masks))
    if scores is not None:
        keep = [i for i in range(len(masks)) if scores[i] >= score_thresh]
        order = sorted(keep, key=lambda i: scores[i].item())
    for i in order:
        m = masks[i]
        if m.dim() == 3:  # predicted masks arrive as [1, H, W] soft probability maps
            m = m[0]
        canvas[m > mask_thresh] = int(labels[i])
    return canvas


class VineInstanceDataset(Dataset):
    """Returns (image_tensor, target) with target = {boxes, labels, masks, ...} at 512²."""
    def __init__(self, split='train', base_dir=DATA_ROOT):
        self.split_dir = os.path.join(base_dir, split)
        self.coco = COCO(os.path.join(self.split_dir, '_annotations.coco.json'))
        self.image_ids = list(self.coco.imgs.keys())
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid = self.image_ids[idx]
        info = self.coco.loadImgs(iid)[0]
        H, W = info['height'], info['width']
        image = Image.open(os.path.join(self.split_dir, info['file_name'])).convert("RGB").resize((IMG, IMG), Image.BILINEAR)

        masks, boxes, labels = [], [], []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid)):
            if not isinstance(ann['segmentation'], list):
                continue
            m = coco_mask.decode(coco_mask.frPyObjects(ann['segmentation'], H, W))
            if m.ndim == 3:
                m = np.any(m, axis=2)
            m = cv2.resize(m.astype(np.uint8), (IMG, IMG), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(m)
            if xs.size == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            if x2 <= x1 or y2 <= y1:
                continue
            masks.append(m)
            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann['category_id']))   # 1..7

        if boxes:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),
                "image_id": torch.tensor([iid]),
            }
        else:  # image with no valid instances
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, IMG, IMG), dtype=torch.uint8),
                "image_id": torch.tensor([iid]),
            }
        return self.to_tensor(image), target


def build_model(num_classes=NUM_CLASSES):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                  min_size=IMG, max_size=IMG)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    BATCH, LR = 8, 5e-3
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    train_ds = VineInstanceDataset('train'); val_ds = VineInstanceDataset('valid')
    print(f"train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4,
                            collate_fn=collate_fn, pin_memory=True)

    model = build_model().to(device)
    print("params_M=%.1f" % (sum(p.numel() for p in model.parameters()) / 1e6))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=min(500, len(train_loader)))

    run_meta = collect_run_metadata(model, device, SEED)
    json.dump(run_meta, open(os.path.join(SAVE_DIR, 'run_metadata.json'), 'w'), indent=2)

    hist = {"train_loss": [], "val_loss": [], "dice": []}
    best_dice = -1.0
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        model.train(); running = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]"):
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if epoch == 0:
                warmup.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        # val loss (train-mode forward under no_grad to get the loss dict)
        val_running = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val-Loss]"):
                images_dev = [im.to(device) for im in images]
                targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images_dev, targets_dev)
                val_running += sum(loss_dict.values()).item()
        val_loss = val_running / len(val_loader)

        # val Dice (genuine eval-mode inference -> composite predicted instance masks vs
        # ground truth into semantic maps -> pixel-wise Dice, same recipe as the other models)
        model.eval()
        tdice = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val-Dice]"):
                images_dev = [im.to(device) for im in images]
                outputs = model(images_dev)
                batch_dice = 0.0
                for out, tgt in zip(outputs, targets):
                    gt_map = instances_to_semantic_map(tgt['masks'].to(device), tgt['labels'].to(device), device=device)
                    pred_map = instances_to_semantic_map(out['masks'], out['labels'], out['scores'], device=device)
                    batch_dice += compute_dice(pred_map, gt_map, NUM_CLASSES)
                tdice += batch_dice / len(images_dev)
        dice = tdice / len(val_loader)
        model.train()
        scheduler.step()

        hist["train_loss"].append(train_loss); hist["val_loss"].append(val_loss); hist["dice"].append(dice)
        epoch_time = time.time() - epoch_start
        run_meta["per_epoch_time_sec"].append(epoch_time)
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"Dice {dice:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | {epoch_time:.1f}s")
        if dice > best_dice:
            best_dice = dice
            epochs_no_improve = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'val_dice': dice, 'class_names': CLASS_NAMES},
                       os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"  ✓ best model saved (Dice {dice:.4f})")
        else:
            epochs_no_improve += 1
        json.dump(hist, open(os.path.join(SAVE_DIR, 'loss_history.json'), 'w'), indent=2)
        json.dump(run_meta, open(os.path.join(SAVE_DIR, 'run_metadata.json'), 'w'), indent=2)
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping: no val-Dice improvement in {PATIENCE} epochs "
                  f"(best {best_dice:.4f} at epoch {epoch + 1 - epochs_no_improve}).")
            break

    n_ran = len(hist['train_loss'])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(hist['train_loss'], label='Train Loss'); plt.plot(hist['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Mask R-CNN Training and Validation Loss ({n_ran} epochs)')
    plt.legend(); plt.grid(True); plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png')); plt.close()
    print(f"Done. Best val Dice {best_dice:.4f}. Saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
