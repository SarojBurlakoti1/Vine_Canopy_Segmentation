"""
U-Net (semantic segmentation) — the user's original U-Net, adapted to run on the
leakage-corrected clean split (identical images to SOLOv2 / clean YOLO), resized to 512²
to match the other models. Architecture, loss (CrossEntropy), optimizer (Adam lr 1e-4),
and 30-epoch schedule are unchanged from the notebook.
"""
import os
import json
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "rebuilt", "final"))  # same for all models
SAVE_DIR = "/workspace/vine-project-Run/UNet/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
IMG = 512
NUM_CLASSES = 8
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


# ============ U-Net (unchanged from the original notebook) ============
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=8):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128); self.down2 = Down(128, 256)
        self.down3 = Down(256, 512); self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512); self.up2 = Up(512, 256)
        self.up3 = Up(256, 128); self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)


# ============ Dataset (adapted: clean split + resize to 512²) ============
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
            rle = coco_mask.frPyObjects(ann['segmentation'], H, W)
            m = coco_mask.decode(rle)
            if m.ndim == 3:
                m = np.any(m, axis=2)
            combined[m.astype(bool)] = ann['category_id']   # 1..7 species, 0 = background
        # resize both to 512 (image bilinear, mask nearest)
        image = image.resize((IMG, IMG), Image.BILINEAR)
        combined = cv2.resize(combined, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
        return self.to_tensor(image), torch.as_tensor(combined, dtype=torch.long)


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
    BATCH, LR = 8, 1e-4  # lr as original; batch 8 for GPU efficiency
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    train_ds = VineDataset('train'); val_ds = VineDataset('valid')
    print(f"train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(3, NUM_CLASSES).to(device)
    print("params_M=%.1f" % (sum(p.numel() for p in model.parameters()) / 1e6))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

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
            out = model(images); loss = criterion(out, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        model.eval(); vloss = 0.0; tacc = 0.0; tmiou = 0.0; tdice = 0.0
        all_p, all_t = [], []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val]"):
                images = images.to(device); masks = masks.to(device)
                out = model(images); vloss += criterion(out, masks).item()
                preds = torch.argmax(out, dim=1)
                tacc += pixel_accuracy(preds, masks); tmiou += compute_mIoU(preds, masks, NUM_CLASSES)
                tdice += compute_dice(preds, masks, NUM_CLASSES)
                # subsample pixels for macro P/R/F1 to keep memory sane
                all_p.append(preds.view(-1)[::16].cpu().numpy())
                all_t.append(masks.view(-1)[::16].cpu().numpy())
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
    pd.DataFrame({'Epoch': range(1, n_ran + 1), 'Train Loss': hist['train_loss'],
                  'Val Loss': hist['val_loss'], 'Accuracy': hist['accuracy'], 'mIoU': hist['miou'],
                  'Dice': hist['dice'], 'Precision': hist['precision'], 'Recall': hist['recall'],
                  'F1-score': hist['f1']}
                 ).to_csv(os.path.join(SAVE_DIR, 'unet_training_metrics.csv'), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(hist['train_loss'], label='Train Loss'); plt.plot(hist['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'U-Net Training and Validation Loss ({n_ran} epochs)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png')); plt.close()
    print(f"Done. Best val Dice {best_dice:.4f}. Saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
