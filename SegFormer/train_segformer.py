"""
SegFormer (HuggingFace MiT-b2 backbone, ImageNet-pretrained encoder) — transformer-based
semantic segmentation on the leakage-corrected clean split, matched to the other models:
512² input, 30 epochs, batch 8, AdamW. CrossEntropy on logits upsampled to full resolution.
"""
import os
import json
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import transformers
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import SegformerForSemanticSegmentation
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "rebuilt", "final"))
SAVE_DIR = "/workspace/vine-project-Run/SegFormer/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
IMG = 512
NUM_CLASSES = 8
CLASS_NAMES = ['background', 'Akebia', 'Campsis', 'Hydrangea',
               'Lonicera', 'Madame Galen', 'Parthenocissus', 'Wisteria']
BACKBONE = "nvidia/mit-b2"
NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "per_epoch_time_sec": [],
    }


class VineDataset(Dataset):
    def __init__(self, split='train', base_dir=DATA_ROOT):
        self.split_dir = os.path.join(base_dir, split)
        self.coco = COCO(os.path.join(self.split_dir, '_annotations.coco.json'))
        self.image_ids = list(self.coco.imgs.keys())
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        info = self.coco.loadImgs(self.image_ids[idx])[0]
        image = Image.open(os.path.join(self.split_dir, info['file_name'])).convert("RGB")
        H, W = info['height'], info['width']
        combined = np.zeros((H, W), dtype=np.uint8)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=info['id'])):
            if not isinstance(ann['segmentation'], list):
                continue
            m = coco_mask.decode(coco_mask.frPyObjects(ann['segmentation'], H, W))
            if m.ndim == 3:
                m = np.any(m, axis=2)
            combined[m.astype(bool)] = ann['category_id']
        image = image.resize((IMG, IMG), Image.BILINEAR)
        combined = cv2.resize(combined, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        return NORM(self.to_tensor(image)), torch.as_tensor(combined, dtype=torch.long)


def build_model():
    return SegformerForSemanticSegmentation.from_pretrained(
        BACKBONE, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)


def forward_logits(model, x):
    """Return full-resolution logits [B, C, IMG, IMG] (SegFormer outputs H/4)."""
    logits = model(pixel_values=x).logits
    return F.interpolate(logits, size=(IMG, IMG), mode="bilinear", align_corners=False)


def pixel_accuracy(preds, labels):
    return ((preds == labels).float().sum() / preds.numel()).item()


def compute_mIoU(preds, labels, num_classes):
    ious = []
    preds = preds.view(-1); labels = labels.view(-1)
    for cls in range(num_classes):
        pi = preds == cls; li = labels == cls
        inter = (pi & li).sum().float(); union = (pi | li).sum().float()
        ious.append(float('nan') if union == 0 else (inter / union).item())
    return torch.tensor(ious).nanmean().item()


def compute_dice(preds, labels, num_classes):
    """Mean per-class Dice = 2|P∩G| / (|P|+|G|), skipping classes absent from both."""
    dices = []
    preds = preds.view(-1); labels = labels.view(-1)
    for cls in range(num_classes):
        pi = preds == cls; li = labels == cls
        inter = (pi & li).sum().float()
        denom = pi.sum().float() + li.sum().float()
        dices.append(float('nan') if denom == 0 else (2 * inter / denom).item())
    return torch.tensor(dices).nanmean().item()


def main():
    BATCH, LR = 8, 6e-5   # SegFormer standard lr
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    train_ds = VineDataset('train'); val_ds = VineDataset('valid')
    print(f"train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(device)
    print("params_M=%.1f" % (sum(p.numel() for p in model.parameters()) / 1e6))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    run_meta = collect_run_metadata(model, device, SEED)
    json.dump(run_meta, open(os.path.join(SAVE_DIR, 'run_metadata.json'), 'w'), indent=2)

    hist = {k: [] for k in ["train_loss", "val_loss", "accuracy", "miou", "dice", "precision", "recall", "f1"]}
    best_dice = -1.0
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        model.train(); running = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]"):
            images = images.to(device); masks = masks.to(device)
            logits = forward_logits(model, images)
            loss = criterion(logits, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        model.eval(); vloss = 0.0; tacc = 0.0; tmiou = 0.0; tdice = 0.0; all_p, all_t = [], []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val]"):
                images = images.to(device); masks = masks.to(device)
                logits = forward_logits(model, images)
                vloss += criterion(logits, masks).item()
                preds = torch.argmax(logits, dim=1)
                tacc += pixel_accuracy(preds, masks); tmiou += compute_mIoU(preds, masks, NUM_CLASSES)
                tdice += compute_dice(preds, masks, NUM_CLASSES)
                all_p.append(preds.view(-1)[::16].cpu().numpy()); all_t.append(masks.view(-1)[::16].cpu().numpy())
        vloss /= len(val_loader); acc = tacc / len(val_loader); miou = tmiou / len(val_loader)
        dice = tdice / len(val_loader)
        yp = np.concatenate(all_p); yt = np.concatenate(all_t)
        prec = precision_score(yt, yp, average='macro', zero_division=0)
        rec = recall_score(yt, yp, average='macro', zero_division=0)
        f1 = f1_score(yt, yp, average='macro', zero_division=0)
        for k, v in zip(hist, [train_loss, vloss, acc, miou, dice, prec, rec, f1]):
            hist[k].append(v)
        epoch_time = time.time() - epoch_start
        run_meta["per_epoch_time_sec"].append(epoch_time)
        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] Train {train_loss:.4f} | Val {vloss:.4f} | "
              f"Acc {acc:.4f} | mIoU {miou:.4f} | Dice {dice:.4f} | P {prec:.4f} R {rec:.4f} F1 {f1:.4f} | "
              f"{epoch_time:.1f}s")
        if dice > best_dice:
            best_dice = dice
            epochs_no_improve = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_miou': miou, 'val_dice': dice, 'class_names': CLASS_NAMES},
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
    pd.DataFrame({'Epoch': range(1, n_ran + 1), 'Train Loss': hist['train_loss'], 'Val Loss': hist['val_loss'],
                  'Accuracy': hist['accuracy'], 'mIoU': hist['miou'], 'Dice': hist['dice'],
                  'Precision': hist['precision'], 'Recall': hist['recall'], 'F1-score': hist['f1']}).to_csv(
        os.path.join(SAVE_DIR, 'segformer_training_metrics.csv'), index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(hist['train_loss'], label='Train Loss'); plt.plot(hist['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'SegFormer Training and Validation Loss ({n_ran} epochs)')
    plt.legend(); plt.grid(True); plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png')); plt.close()
    print(f"Done. Best val Dice {best_dice:.4f}. Saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
