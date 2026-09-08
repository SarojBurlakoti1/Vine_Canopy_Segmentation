"""
Evaluate the trained best_model.pth on the held-out `test` split instead of `valid`,
and produce a matching-format confusion matrix + metrics.json for comparison.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from train_solov2 import (
    ROOT, SAVE_DIR, VineDataset, SimplifiedSOLOv2, get_transforms, collate_fn,
    evaluate_metrics, plot_normalized_confusion_matrix, plot_sample_prediction_comparison,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = VineDataset('test', get_transforms(False), root=ROOT)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
class_names = test_dataset.class_names

ckpt = torch.load(os.path.join(SAVE_DIR, 'best_model.pth'), map_location=device)
model = SimplifiedSOLOv2(num_classes=len(class_names)).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded best_model.pth (epoch {ckpt['epoch']+1}, val_loss {ckpt['val_loss']:.4f})")

TEST_SAVE_DIR = os.path.join(os.path.dirname(SAVE_DIR), "checkpoints_test")
os.makedirs(TEST_SAVE_DIR, exist_ok=True)

acc, prec, rec, f1, miou, cm, class_names = evaluate_metrics(
    model, test_loader, device, class_names=class_names, save_dir=TEST_SAVE_DIR
)

plot_normalized_confusion_matrix(cm, class_names, save_dir=TEST_SAVE_DIR, model_label="SOLOv2, test split")

print("\nSaved test-split evaluation to:", TEST_SAVE_DIR)
