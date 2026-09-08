"""
SimplifiedSOLOv2 training + evaluation on Vines_2025 v6 (Roboflow COCO-segmentation export).
Adapted from "SoloV2 Final.ipynb" — same dataset class, model architecture and training loop,
pointed at the new dataset version and a separate output folder (SoloV2/checkpoints).
"""
import os
import json
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "rebuilt", "final"))
SAVE_DIR = "/workspace/vine-project-Run/SoloV2/checkpoints_clean"
os.makedirs(SAVE_DIR, exist_ok=True)

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


def compute_batch_dice(category_pred, masks, labels, num_classes, device):
    """category_pred: [B,C,Hc,Wc] logits. masks/labels: per-image lists, same convention as
    evaluate_metrics(). Builds each image's ground-truth map on the model's native coarse
    grid (no upsampling) and returns the mean per-class Dice against argmax(category_pred)."""
    preds = torch.argmax(category_pred, dim=1)  # [B, Hc, Wc]
    B, Hc, Wc = preds.shape
    total = 0.0
    for i in range(B):
        gt_map = torch.zeros((Hc, Wc), dtype=torch.long, device=device)
        for j, lbl in enumerate(labels[i]):
            if j < len(masks[i]):
                mask_small = torch.nn.functional.interpolate(
                    masks[i][j].unsqueeze(0).unsqueeze(0), size=(Hc, Wc), mode='nearest'
                ).squeeze()
                gt_map[mask_small > 0.5] = int(lbl.item()) if torch.is_tensor(lbl) else int(lbl)
        total += compute_dice(preds[i], gt_map, num_classes)
    return total / B


# =========================
# Dataset
# =========================
class VineDataset(Dataset):
    """Dataset class for vine instance segmentation (Roboflow COCO)."""
    def __init__(self, split='train', transforms=None, root=ROOT):
        if split not in ['train', 'valid', 'test']:
            raise ValueError("split must be one of: 'train', 'valid', 'test'")
        self.split = split
        self.transforms = transforms

        self.image_dir = os.path.join(root, split)
        self.annotation_file = os.path.join(self.image_dir, "_annotations.coco.json")

        if not os.path.exists(self.image_dir):
            raise RuntimeError(f"Images folder not found: {self.image_dir}")
        if not os.path.exists(self.annotation_file):
            raise RuntimeError(f"Annotation file not found: {self.annotation_file}")

        self.coco = COCO(self.annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # exclude any dummy/root class with id==0
        raw_cat_ids = sorted(self.coco.getCatIds())
        cat_ids = [cid for cid in raw_cat_ids if cid != 0]
        cats = self.coco.loadCats(cat_ids)
        cats_sorted = sorted(cats, key=lambda c: c['id'])
        self.class_names = ["background"] + [c['name'] for c in cats_sorted]
        self.id_to_idx = {cid: i + 1 for i, cid in enumerate(cat_ids)}

        print(f"\nDataset Info ({split})")
        print(f"- images: {os.path.abspath(self.image_dir)}")
        print(f"- ann:    {os.path.abspath(self.annotation_file)}")
        print(f"- #images: {len(self.image_ids)}")
        print(f"- categories: {list(self.coco.cats.values())}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        processed_masks = []
        labels = []

        for ann in anns:
            if isinstance(ann['segmentation'], list):
                rles = coco_mask.frPyObjects(ann['segmentation'], image_info['height'], image_info['width'])
                mask = coco_mask.decode(rles)
            else:
                mask = coco_mask.decode(ann['segmentation'])

            if mask.ndim == 3:
                mask = mask[:, :, 0]

            mask_resized = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
            processed_masks.append(torch.tensor(mask_resized, dtype=torch.float32))
            labels.append(self.id_to_idx.get(ann['category_id'], 0))

        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        if not processed_masks:
            processed_masks = [torch.zeros((512, 512), dtype=torch.float32)]
            labels = [0]

        masks = torch.stack(processed_masks)
        labels = torch.tensor(labels, dtype=torch.long)

        return image, masks, labels


# =========================
# Model
# =========================
class SimplifiedSOLOv2(nn.Module):
    """Simplified SOLOv2-ish head on a ResNet50 backbone."""
    def __init__(self, num_classes=8):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        backbone_out = 2048

        self.category_head = nn.Sequential(
            nn.Conv2d(backbone_out, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(backbone_out, 384, 3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        feats = self.backbone(x)
        category_pred = self.category_head(feats)
        mask_pred = self.mask_head(feats)
        return category_pred, mask_pred


def focal_cross_entropy(logits, targets, gamma=2.0, alpha=None):
    ce = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    focal = (1 - pt) ** gamma * ce
    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            a = alpha.to(logits.device)
            focal = focal * a[targets.squeeze(0)]
        else:
            focal = focal * alpha
    return focal.mean()


def compute_loss(category_pred, mask_pred, masks, labels, device):
    batch_size = category_pred.size(0)
    total = 0.0

    for i in range(batch_size):
        target_hw = category_pred.shape[-2:]
        category_target = torch.zeros(target_hw, dtype=torch.long, device=device)

        if len(labels[i]) > 0:
            for j in range(min(len(labels[i]), len(masks[i]))):
                mask = masks[i][j]
                mask_small = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), size=target_hw, mode='nearest'
                ).squeeze()
                lbl = int(labels[i][j].item()) if torch.is_tensor(labels[i]) else int(labels[i][j])
                category_target[mask_small > 0.5] = lbl

        cat_loss = focal_cross_entropy(category_pred[i].unsqueeze(0), category_target.unsqueeze(0), gamma=2.0, alpha=None)

        if len(masks[i]) > 0:
            mask_pred_resized = torch.nn.functional.interpolate(
                mask_pred[i].unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False
            ).squeeze()
            mask_target = masks[i][0]
            mask_loss = nn.functional.binary_cross_entropy_with_logits(mask_pred_resized, mask_target)
        else:
            mask_loss = torch.tensor(0.0, device=device)

        total += (cat_loss + mask_loss)

    return total / batch_size


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def collate_fn(batch):
    images, all_masks, all_labels = [], [], []
    for image, masks, labels in batch:
        images.append(image)
        all_masks.append(masks)
        all_labels.append(labels)
    images = torch.stack(images)
    return images, all_masks, all_labels


# =========================
# Training
# =========================
def train_model(num_epochs=MAX_EPOCHS, batch_size=8, learning_rate=1e-4, root=ROOT, patience=PATIENCE):
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

    print("Loading datasets...")
    train_dataset = VineDataset('train', get_transforms(True), root=root)
    val_dataset = VineDataset('valid', get_transforms(False), root=root)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Empty train/valid split. Check your dataset export and paths.")

    num_workers = 4
    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin)

    steps_per_epoch = max(1, len(train_loader))
    model = SimplifiedSOLOv2(num_classes=len(train_dataset.class_names)).to(device)
    print("params_M=%.1f" % (sum(p.numel() for p in model.parameters()) / 1e6))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.3
    )

    run_meta = collect_run_metadata(model, device, SEED)
    json.dump(run_meta, open(os.path.join(SAVE_DIR, 'run_metadata.json'), 'w'), indent=2)

    train_losses, val_losses, val_dices = [], [], []
    best_dice = -1.0
    epochs_no_improve = 0
    num_classes = len(train_dataset.class_names)

    print(f"Starting training for up to {num_epochs} epochs (early stop patience={patience}, monitor=val Dice)...")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for images, masks, labels in pbar:
            images = images.to(device)
            masks = [m.to(device) for m in masks]
            labels = [l.to(device) for l in labels]

            optimizer.zero_grad()
            cat_pred, m_pred = model(images)
            loss = compute_loss(cat_pred, m_pred, masks, labels, device)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train = running / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        val_running = 0.0
        dice_running = 0.0
        vpbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for images, masks, labels in vpbar:
                images = images.to(device)
                masks = [m.to(device) for m in masks]
                labels = [l.to(device) for l in labels]
                cat_pred, m_pred = model(images)
                vloss = compute_loss(cat_pred, m_pred, masks, labels, device)
                val_running += vloss.item()
                dice_running += compute_batch_dice(cat_pred, masks, labels, num_classes, device)
                vpbar.set_postfix({'Loss': f'{vloss.item():.4f}'})

        avg_val = val_running / len(val_loader)
        avg_dice = dice_running / len(val_loader)
        val_losses.append(avg_val)
        val_dices.append(avg_dice)
        epoch_time = time.time() - epoch_start
        run_meta["per_epoch_time_sec"].append(epoch_time)
        print(f"Epoch {epoch+1}: Train {avg_train:.4f} | Val {avg_val:.4f} | Dice {avg_dice:.4f} | {epoch_time:.1f}s")

        if avg_dice > best_dice:
            best_dice = avg_dice
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train, 'val_loss': avg_val, 'val_dice': avg_dice,
                'class_names': train_dataset.class_names,
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"Best model saved (Dice: {avg_dice:.4f})")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train, 'val_loss': avg_val, 'val_dice': avg_dice,
                'class_names': train_dataset.class_names,
            }, os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth'))

        with open(os.path.join(SAVE_DIR, 'loss_history.json'), 'w') as f:
            json.dump({'train_losses': train_losses, 'val_losses': val_losses, 'val_dices': val_dices}, f, indent=2)
        json.dump(run_meta, open(os.path.join(SAVE_DIR, 'run_metadata.json'), 'w'), indent=2)

        if epochs_no_improve >= patience:
            print(f"Early stopping: no val-Dice improvement in {patience} epochs "
                  f"(best {best_dice:.4f} at epoch {epoch + 1 - epochs_no_improve}).")
            break

    n_ran = len(train_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Training and Validation Loss (SOLOv2, {n_ran} epochs)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png')); plt.close()

    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Model & curves saved in: {SAVE_DIR}")

    return model, train_dataset.class_names


# =========================
# Evaluation
# =========================
def build_val_loader(root=ROOT, batch_size=8, num_workers=2, device='cuda'):
    pin = (torch.cuda.is_available() and device == 'cuda')
    val_dataset = VineDataset('valid', get_transforms(False), root=root)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin)
    return val_loader, val_dataset.class_names


def iou_from_confusion_matrix(cm):
    eps = 1e-7
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn + eps
    ious = tp / denom
    return ious, np.mean(ious)


def evaluate_metrics(model, dataloader, device, class_names=None, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_preds, all_trues = [], []
    num_classes = len(class_names) if class_names is not None else 8

    with torch.no_grad():
        for images, masks, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            category_pred, _ = model(images)
            preds = torch.argmax(category_pred, dim=1)

            B, Hc, Wc = preds.shape
            for i in range(B):
                gt_map = torch.zeros((Hc, Wc), dtype=torch.long, device=device)
                for j, lbl in enumerate(labels[i]):
                    if j < len(masks[i]):
                        mask = masks[i][j].to(device)
                        mask_small = torch.nn.functional.interpolate(
                            mask.unsqueeze(0).unsqueeze(0), size=(Hc, Wc), mode="nearest"
                        ).squeeze()
                        gt_map[mask_small > 0.5] = lbl

                all_trues.append(gt_map.cpu().numpy().reshape(-1))
                all_preds.append(preds[i].cpu().numpy().reshape(-1))

    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    acc = accuracy_score(all_trues, all_preds)
    prec = precision_score(all_trues, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_trues, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_trues, all_preds, average='weighted', zero_division=0)

    labels_range = list(range(num_classes))
    cm = confusion_matrix(all_trues, all_preds, labels=labels_range)
    ious, miou = iou_from_confusion_matrix(cm)

    print(f"\n=== Validation Metrics (pixel-level) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (weighted)")
    print(f"Recall   : {rec:.4f} (weighted)")
    print(f"F1-score : {f1:.4f} (weighted)")
    print(f"mIoU     : {miou:.4f}")

    if class_names is None:
        class_names = [f"Class {i}" for i in labels_range]

    print("\nPer-class IoU:")
    for i, iou in enumerate(ious):
        cname = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"  {i:>2} ({cname}): {iou:.4f}")

    np.save(os.path.join(save_dir, "confusion_matrix.npy"), cm)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": acc, "precision_weighted": prec, "recall_weighted": rec,
            "f1_weighted": f1, "miou": float(miou),
            "per_class_iou": {class_names[i]: float(ious[i]) for i in range(num_classes)}
        }, f, indent=2)
    print(f"Metrics saved to: {os.path.join(save_dir, 'metrics.json')}")

    return acc, prec, rec, f1, miou, cm, class_names


# =========================
# Confusion matrix plot (matches confusion_matrix.png reference style)
# =========================
def plot_normalized_confusion_matrix(cm, class_names, save_dir=SAVE_DIR, model_label="SOLOv2"):
    """Row-normalized confusion matrix, species-only display (drops 'background' row/col),
    styled to match the reference confusion_matrix.png: seaborn Blues heatmap,
    'Ground Truth Class' / 'Predicted Class' axis labels, 2-decimal annotations.

    Row sums are taken from the FULL matrix (including the background column), not just
    the species-only submatrix — species pixels the model calls "background" are real
    errors and must stay in the denominator, otherwise the diagonal looks artificially
    perfect by simply excluding the model's dominant failure mode from the count."""
    species_names = [c for c in class_names if c != "background"]
    idx = [i for i, c in enumerate(class_names) if c != "background"]
    cm_species = cm[np.ix_(idx, idx)].astype('float')

    row_sums = cm[idx, :].sum(axis=1, keepdims=True).astype('float')
    row_sums[row_sums == 0] = 1
    cm_normalized = cm_species / row_sums

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=species_names, yticklabels=species_names,
                cbar=True, square=False)
    plt.ylabel("Ground Truth Class")
    plt.xlabel("Predicted Class")
    plt.title(f"Normalized Confusion Matrix ({model_label})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Normalized confusion matrix saved to: {out_path}")
    return out_path


# =========================
# Sample prediction comparison (matches sample_prediction_comparison.png reference style)
# =========================
def plot_sample_prediction_comparison(model, class_names, image_path, gt_species_name,
                                       device, save_dir=SAVE_DIR, conf_threshold=0.5,
                                       min_area=150, min_label_conf=0.5, max_boxes=12,
                                       model_label="SOLOv2"):
    """Side-by-side Original RGB vs prediction with red overlay masks + bounding boxes +
    confidence labels, styled to match sample_prediction_comparison.png."""
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = image_rgb.shape[:2]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        category_pred, mask_pred = model(input_tensor)
        category_probs = torch.softmax(category_pred, dim=1)  # [1, C, Hc, Wc]
        objectness = torch.sigmoid(mask_pred)                  # [1, 1, Hc, Wc]

        # Upsample continuous score maps (not the argmax) to full resolution with
        # bilinear interpolation so mask edges are smooth rather than blocky —
        # the model's native grid is coarse (32x downsampled, no decoder), so this
        # is a soft upsample rather than a true high-res prediction.
        category_probs_full = torch.nn.functional.interpolate(
            category_probs, size=(H0, W0), mode='bilinear', align_corners=False
        )[0]
        objectness_full = torch.nn.functional.interpolate(
            objectness, size=(H0, W0), mode='bilinear', align_corners=False
        )[0, 0]

        class_conf_full, class_ids_full = category_probs_full.max(dim=0)
        fg_full = (class_ids_full > 0) & (objectness_full > conf_threshold)

        class_ids_np = class_ids_full.cpu().numpy()
        class_conf_np = class_conf_full.cpu().numpy()
        fg_np = fg_full.cpu().numpy().astype(np.uint8)

    overlay = image_rgb.copy()
    boxes = []  # (x1, y1, x2, y2, class_name, conf)

    for cls_idx in np.unique(class_ids_np):
        if cls_idx == 0:
            continue
        cls_mask = ((class_ids_np == cls_idx) & (fg_np > 0)).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue
        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            comp_mask_full = (labels_cc == lbl).astype(np.uint8)
            if comp_mask_full.sum() == 0:
                continue

            red = np.zeros_like(overlay)
            red[:, :, 0] = 255
            alpha = 0.55
            m = comp_mask_full.astype(bool)
            overlay[m] = (overlay[m].astype(np.float32) * (1 - alpha) + red[m].astype(np.float32) * alpha).astype(np.uint8)

            ys, xs = np.where(comp_mask_full)
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            conf = float(class_conf_np[labels_cc == lbl].mean())
            cname = class_names[cls_idx] if cls_idx < len(class_names) else f"class{cls_idx}"
            boxes.append((x1, y1, x2, y2, cname, conf))

    # Only label the model's most confident detections — an untrained/uncertain model
    # can otherwise produce dozens of low-confidence stray components that bury the plot.
    boxes = [b for b in boxes if b[5] >= min_label_conf]
    boxes.sort(key=lambda b: b[5], reverse=True)
    boxes = boxes[:max_boxes]

    for (x1, y1, x2, y2, cname, conf) in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)
        label = f"{cname} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        tx1 = min(max(0, x1), W0 - tw - 10)
        ty1 = max(0, y1 - th - baseline - 6)
        cv2.rectangle(overlay, (tx1, ty1), (tx1 + tw + 10, ty1 + th + baseline + 6), (255, 0, 0), -1)
        cv2.putText(overlay, label, (tx1 + 5, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(20, 11.5))
    axes[0].imshow(image_rgb); axes[0].set_title("Original RGB", fontsize=16); axes[0].axis('off')
    axes[1].imshow(overlay); axes[1].set_title(f"{model_label} Prediction", fontsize=16); axes[1].axis('off')
    fig.suptitle(f"{gt_species_name} — instance segmentation + bounding boxes", fontsize=18)
    plt.tight_layout()
    out_path = os.path.join(save_dir, "sample_prediction_comparison.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Sample prediction comparison saved to: {out_path}")
    return out_path


# =========================
# Main
# =========================
if __name__ == "__main__":
    config = {'num_epochs': MAX_EPOCHS, 'batch_size': 8, 'learning_rate': 1e-4, 'patience': PATIENCE}

    trained_model, class_names = train_model(**config, root=ROOT)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader, _ = build_val_loader(root=ROOT, batch_size=8, num_workers=2, device=device.type)

    acc, prec, rec, f1, miou, cm, class_names = evaluate_metrics(
        trained_model, val_loader, device, class_names=class_names, save_dir=SAVE_DIR
    )

    plot_normalized_confusion_matrix(cm, class_names, save_dir=SAVE_DIR, model_label="SOLOv2")

    sample_image = None
    for _split in ("valid", "test"):
        for _f in os.listdir(os.path.join(ROOT, _split)):
            if _f.startswith("IMG_1534_aug1") and _f.endswith((".jpg", ".jpeg", ".png")):
                sample_image = os.path.join(ROOT, _split, _f)
                break
        if sample_image:
            break
    if sample_image and os.path.exists(sample_image):
        plot_sample_prediction_comparison(
            trained_model, class_names, sample_image, "Wisteria floribunda",
            device, save_dir=SAVE_DIR, model_label="SOLOv2"
        )
    else:
        print(f"Sample image not found, skipping comparison plot: {sample_image}")

    print("\nAll done. Outputs in:", SAVE_DIR)
