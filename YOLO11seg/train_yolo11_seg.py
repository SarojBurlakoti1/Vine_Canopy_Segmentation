"""
Train stock YOLO11l-seg on the leakage-corrected clean split — the same
images SimplifiedSOLOv2 was trained/tested on. Config mirrors the notebook's
train_yolo11_seg() call (img 512, batch 8, AdamW, lr 5e-4).

Early stopping: Ultralytics' own `patience` early-stops on its internal
"fitness" metric (a weighted box+mask mAP composite), not on Dice. To get a
genuine, consistently-defined validation-Dice-monitored early stop (matching
the other six models in this project), Ultralytics' native patience is
disabled (set absurdly high) and a custom callback drives the actual stop
decision instead: each epoch it reads the mask precision/recall Ultralytics
already computes (trainer.metrics keys "metrics/precision(M)"/"recall(M)",
confirmed against this project's own historical results.csv), derives
Dice = 2PR/(P+R) -- which is exactly the F1 score, and F1 on binary masks
*is* the Dice coefficient -- and sets trainer.stop=True after `patience`
epochs with no improvement. The best-Dice checkpoint is copied alongside
Ultralytics' own best.pt (which still tracks Ultralytics' native fitness)
as best_dice.pt.
"""
import json
import os
import shutil
import sys
import time

import torch
import ultralytics
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(BASE, "..", "rebuilt", "final_yolo", "data.yaml"))

SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15  # epochs with no val-Dice improvement before stopping
RUN_NAME = "vine_clean_v6"


def make_dice_early_stopper(patience=PATIENCE):
    """Returns (on_train_epoch_start, on_fit_epoch_end) callbacks that together
    time each epoch and implement Dice-monitored early stopping via trainer.stop."""
    state = {"best_dice": -1.0, "wait": 0, "history": [], "epoch_start": None}

    def on_train_epoch_start(trainer):
        state["epoch_start"] = time.time()

    def on_fit_epoch_end(trainer):
        metrics = trainer.metrics or {}
        p = next((v for k, v in metrics.items() if "precision(M)" in k), None)
        r = next((v for k, v in metrics.items() if "recall(M)" in k), None)
        if p is None or r is None:
            print(f"[dice-early-stop] WARNING: no mask precision/recall found in "
                  f"trainer.metrics (keys={list(metrics.keys())}); skipping this epoch's check.")
            dice = float("nan")
        else:
            dice = 2 * p * r / (p + r + 1e-9)

        epoch_time = (time.time() - state["epoch_start"]) if state["epoch_start"] else None
        state["history"].append({"epoch": trainer.epoch + 1, "val_dice": dice, "epoch_time_sec": epoch_time})
        with open(os.path.join(trainer.save_dir, "dice_early_stop_log.json"), "w") as f:
            json.dump(state["history"], f, indent=2)

        if dice == dice and dice > state["best_dice"]:  # dice==dice is False for NaN
            state["best_dice"] = dice
            state["wait"] = 0
            best_dice_path = os.path.join(trainer.save_dir, "weights", "best_dice.pt")
            if os.path.exists(trainer.last):
                shutil.copy2(trainer.last, best_dice_path)
            print(f"[dice-early-stop] epoch {trainer.epoch + 1}: new best val Dice {dice:.4f} -> {best_dice_path}")
        else:
            state["wait"] += 1
            print(f"[dice-early-stop] epoch {trainer.epoch + 1}: val Dice {dice:.4f} "
                  f"(best {state['best_dice']:.4f}, no improvement {state['wait']}/{patience})")

        if state["wait"] >= patience:
            print(f"[dice-early-stop] stopping: no val-Dice improvement in {patience} epochs "
                  f"(best {state['best_dice']:.4f}).")
            trainer.stop = True

    return on_train_epoch_start, on_fit_epoch_end


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    run_dir = os.path.join("/workspace/vine-project-Run/YOLO11seg/runs", RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)

    model = YOLO("yolo11l-seg.pt")

    run_meta = {
        "seed": SEED,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "package_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": torch.version.cuda,
        },
        "total_parameters": sum(p.numel() for p in model.model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.model.parameters() if p.requires_grad),
    }
    json.dump(run_meta, open(os.path.join(run_dir, "run_metadata.json"), "w"), indent=2)

    def on_train_start(trainer):
        # Re-count post-setup: Ultralytics' trainer sets requires_grad (freeze schedule etc.)
        # only once training actually starts, so the counts taken right after YOLO(...)
        # construction above are wrong (trainable_parameters reads 0 pre-setup).
        params = list(trainer.model.parameters())
        run_meta["total_parameters"] = sum(p.numel() for p in params)
        run_meta["trainable_parameters"] = sum(p.numel() for p in params if p.requires_grad)
        json.dump(run_meta, open(os.path.join(run_dir, "run_metadata.json"), "w"), indent=2)

    on_epoch_start, on_epoch_end = make_dice_early_stopper(patience=PATIENCE)
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_fit_epoch_end", on_epoch_end)

    model.train(
        data=DATA,
        epochs=MAX_EPOCHS,
        imgsz=512,
        batch=8,
        lr0=5e-4,
        optimizer="AdamW",
        seed=SEED,
        patience=100000,  # Ultralytics' own fitness-based early stop, effectively disabled --
                          # the callback above drives the actual (Dice-monitored) stop decision
        device=device,
        project="/workspace/vine-project-Run/YOLO11seg/runs",
        name=RUN_NAME,
        workers=4,
        verbose=True,
        plots=True,
        exist_ok=True,
    )
    print("Training done. Run dir:", run_dir)
    print("Ultralytics best.pt (native fitness) and best_dice.pt (val-Dice-selected) both in",
          os.path.join(run_dir, "weights"))


if __name__ == "__main__":
    main()
