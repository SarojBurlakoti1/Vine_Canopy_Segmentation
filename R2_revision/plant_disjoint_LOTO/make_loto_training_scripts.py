#!/usr/bin/env python3
"""
make_loto_training_scripts.py
Generates the seven LOTO training scripts in models/ from the R1 training scripts (GitHub
SarojBurlakoti1/Vine_Canopy_Segmentation, commit b2f792f) by exact text replacements, and writes a
unified diff for each one to patches/. The R1 files are read only; their SHA-256 must match.

What changes (and nothing else: architectures, losses, optimisers, learning rates, batch sizes,
schedules, input size, training augmentation and data loading are untouched):
  1. data folder, output folder and seed come from loto_common.run_config() (environment variables
     set by run_step1_training.sh) instead of hard-coded /workspace paths and SEED = 42;
  2. the R1 early-stopping rule is kept for every model (at most MAX_EPOCHS = 100, stop after
     PATIENCE = 15 epochs without improvement; both from run_config, so the launcher records them),
     but it now runs on the uniform criterion instead of each model's own validation score. For the
     YOLO scripts the R1 Dice early-stopper callback is replaced by loto_common's callback, which
     applies the same rule; Ultralytics' native patience stays disabled as in R1;
  3. the checkpoint kept for testing is chosen by the same uniform criterion in
     loto_common.EpochSelector (pooled 8-class pixel Dice at 512 x 512 on the validation set,
     computed with the same predictor as the test evaluation). The R1 selection quantities are
     still computed and logged;
  4. simplified SOLOv2: the every-10-epoch full checkpoints and the optimizer state in best_model.pth
     are dropped (disk), and the qualitative sample image is taken from the validation split only
     (R1 also searched the test split);
  5. simplified SOLOv2: R1's geometric training augmentations (flips, rotation, affine, random
     resized crop; same parameters and order) are applied to the image and its masks together
     (torchvision.transforms.v2). R1 applied them to the image only, so targets were misaligned.
     Author decision of 9 Nov 2026.

Usage:  python make_loto_training_scripts.py <R1 repo folder at b2f792f>
"""
import difflib
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SRC = {  # model folder -> (R1 file, LOTO file, SHA-256 of the R1 file)
    "UNet": ("UNet/train_unet.py", "train_unet_loto.py",
             "d07ffb25d36f981adf6495c1f9fa0a06ead783b4107ac7fbb4df23f5140c8e1f"),
    "DeepLabV3": ("DeepLabV3/train_deeplabv3.py", "train_deeplabv3_loto.py",
                  "1c509359c07f415b88300e27fa0b94ade41b7bf56de4153baa352325746ef0a7"),
    "SegFormer": ("SegFormer/train_segformer.py", "train_segformer_loto.py",
                  "b6c391b580ee5c41b0383b1058b6190e6e4ea573c83643b7f4a05cc9b37454de"),
    "MaskRCNN": ("MaskRCNN/train_maskrcnn.py", "train_maskrcnn_loto.py",
                 "7235467c39a552c30f078e4fab90551db1ecf5a542d6ad0db51e872e64b9c902"),
    "SoloV2": ("SoloV2/train_solov2_clean.py", "train_solov2_loto.py",
               "badcfbe6e5d3bf011c4508d49f4f47ef466f991861871fdbb17fc11aa621ba95"),
    "YOLO11seg": ("YOLO11seg/train_yolo11_seg.py", "train_yolo11_seg_loto.py",
                  "7709a32bc4642017875ed20c72b9fe8536bdc0734e37c2420db724b34b697c66"),
    "PGSS_YOLO11": ("PGSS_YOLO11/train_pgss.py", "train_pgss_loto.py",
                    "8d895af685846654d5c75df938f544dce9fed616d5ba604e19282e71acc17740"),
}
COPY_UNCHANGED = {  # PGSS dependencies, copied byte-for-byte
    "PGSS_YOLO11/pgss_modules.py": "ac4717f7cc8fc1ba526ba02c5b8eac006a66e287e64460511d0c946e67190c07",
    "PGSS_YOLO11/yolo11s-pgss-seg.yaml": "2c76be246080f268e1478b06e533ceb73d88aaee6f6a4e83536aaf7d28ab5834",
}

LC_IMPORT = ('sys.path.insert(0, os.path.abspath(os.path.join({base}, "..", "..")))  # loto_retraining/\n'
             'import loto_common as LC  # noqa: E402\n'
             'CFG = LC.run_config("{name}")\n')

SEED_OLD = ("SEED = 42\n"
            "MAX_EPOCHS = 100\n"
            "PATIENCE = 15  # epochs with no val-Dice improvement before stopping\n")
SEED_NEW = ('SEED = CFG["seed"]\n'
            'MAX_EPOCHS = CFG["max_epochs"]   # as R1: at most 100 epochs\n'
            'PATIENCE = CFG["patience"]       # as R1: 15 epochs without improvement (now of the shared Dice)\n')

META_OLD = ("    run_meta = collect_run_metadata(model, device, SEED)\n"
            "    json.dump(run_meta, open(os.path.join(SAVE_DIR, 'run_metadata.json'), 'w'), indent=2)\n")


def meta_new(predictor_call, valid_dir_expr, name):
    return (META_OLD +
            '    run_meta["loto"] = CFG\n'
            f"    SELECTOR = LC.EpochSelector(SAVE_DIR, os.path.join({valid_dir_expr}, 'valid'),\n"
            f"                                {predictor_call}, '{name}')\n")


SEL_COMMENT = ("        # LOTO: the checkpoint kept for testing is chosen by the uniform criterion\n"
               "        # (loto_common.EpochSelector: pooled 8-class pixel Dice at 512 x 512 on the validation\n"
               "        # set, same predictor as the test evaluation). The R1 Dice above is still logged.\n")


def semantic_select(save_keys_old, save_keys_new):
    old = ("        if dice > best_dice:\n"
           "            best_dice = dice\n"
           "            epochs_no_improve = 0\n"
           "            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),\n"
           f"                        {save_keys_old}\n"
           "                       os.path.join(SAVE_DIR, 'best_model.pth'))\n"
           '            print(f"  ✓ best model saved (Dice {dice:.4f})")\n'
           "        else:\n"
           "            epochs_no_improve += 1\n")
    new = (SEL_COMMENT +
           "        sel_dice = SELECTOR.score(model)\n"
           "        if SELECTOR.update(epoch + 1, sel_dice):\n"
           "            best_dice = sel_dice\n"
           "            epochs_no_improve = 0\n"
           "            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),\n"
           f"                        {save_keys_new}\n"
           "                       os.path.join(SAVE_DIR, 'best_model.pth'))\n"
           '            print(f"  ✓ best model saved (pooled val Dice {sel_dice:.4f})")\n'
           "        else:\n"
           "            epochs_no_improve += 1\n")
    return old, new


FIN_OLD = "    n_ran = len(hist['train_loss'])\n"
FIN_NEW = FIN_OLD + "    SELECTOR.finalize(n_ran, os.path.join(SAVE_DIR, 'best_model.pth'))\n"

SEM_KEYS_OLD = "'val_miou': miou, 'val_dice': dice, 'class_names': CLASS_NAMES},"
SEM_KEYS_NEW = "'val_miou': miou, 'val_dice': dice, 'val_pooled_dice_512': sel_dice, 'class_names': CLASS_NAMES},"


def torch_paths(name, data_line):
    old = (data_line +
           f'SAVE_DIR = "/workspace/vine-project-Run/{name}/checkpoints"\n'
           "os.makedirs(SAVE_DIR, exist_ok=True)\n")
    new = (LC_IMPORT.format(base="BASE_DIR", name=name) +
           'DATA_ROOT = CFG["data_root"]   # <loto>/<fold>/rebuilt/final\n'
           'SAVE_DIR = CFG["save_dir"]     # <LOTO_OUT_ROOT>/<fold>/seed<seed>/' + name + "\n"
           "os.makedirs(SAVE_DIR, exist_ok=True)\n")
    return old, new


DATA_FINAL = 'DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "rebuilt", "final"))\n'
DATA_FINAL_UNET = ('DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "rebuilt", "final"))'
                   '  # same for all models\n')

REPL = {
    "UNet": [torch_paths("UNet", DATA_FINAL_UNET), (SEED_OLD, SEED_NEW),
             (META_OLD, meta_new("LC.make_predictor('unet', device)", "DATA_ROOT", "UNet")),
             semantic_select(SEM_KEYS_OLD, SEM_KEYS_NEW), (FIN_OLD, FIN_NEW)],
    "DeepLabV3": [torch_paths("DeepLabV3", DATA_FINAL), (SEED_OLD, SEED_NEW),
                  (META_OLD, meta_new("LC.make_predictor('deeplabv3', device)", "DATA_ROOT", "DeepLabV3")),
                  semantic_select(SEM_KEYS_OLD, SEM_KEYS_NEW), (FIN_OLD, FIN_NEW)],
    "SegFormer": [torch_paths("SegFormer", DATA_FINAL), (SEED_OLD, SEED_NEW),
                  (META_OLD, meta_new("LC.make_predictor('segformer', device, segformer_forward=forward_logits, "
                                      "segformer_norm=NORM)", "DATA_ROOT", "SegFormer")),
                  semantic_select(SEM_KEYS_OLD, SEM_KEYS_NEW), (FIN_OLD, FIN_NEW)],
    "MaskRCNN": [torch_paths("MaskRCNN", DATA_FINAL), (SEED_OLD, SEED_NEW),
                 (META_OLD, meta_new("LC.make_predictor('maskrcnn', device)", "DATA_ROOT", "MaskRCNN")),
                 semantic_select("'val_loss': val_loss, 'val_dice': dice, 'class_names': CLASS_NAMES},",
                                 "'val_loss': val_loss, 'val_dice': dice, 'val_pooled_dice_512': sel_dice, "
                                 "'class_names': CLASS_NAMES},"),
                 (FIN_OLD, FIN_NEW)],
    "SoloV2": [
        ('ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "rebuilt", "final"))\n'
         'SAVE_DIR = "/workspace/vine-project-Run/SoloV2/checkpoints_clean"\n'
         "os.makedirs(SAVE_DIR, exist_ok=True)\n",
         LC_IMPORT.format(base="BASE_DIR", name="SoloV2") +
         'ROOT = CFG["data_root"]        # <loto>/<fold>/rebuilt/final\n'
         'SAVE_DIR = CFG["save_dir"]     # <LOTO_OUT_ROOT>/<fold>/seed<seed>/SoloV2\n'
         "os.makedirs(SAVE_DIR, exist_ok=True)\n"),
        (SEED_OLD, SEED_NEW),
        (META_OLD, meta_new("LC.make_predictor('solov2', device)", "root", "SoloV2")),
        ("        if avg_dice > best_dice:\n"
         "            best_dice = avg_dice\n"
         "            epochs_no_improve = 0\n"
         "            torch.save({\n"
         "                'epoch': epoch, 'model_state_dict': model.state_dict(),\n"
         "                'optimizer_state_dict': optimizer.state_dict(),\n"
         "                'train_loss': avg_train, 'val_loss': avg_val, 'val_dice': avg_dice,\n"
         "                'class_names': train_dataset.class_names,\n"
         "            }, os.path.join(SAVE_DIR, 'best_model.pth'))\n"
         '            print(f"Best model saved (Dice: {avg_dice:.4f})")\n'
         "        else:\n"
         "            epochs_no_improve += 1\n"
         "\n"
         "        if (epoch + 1) % 10 == 0:\n"
         "            torch.save({\n"
         "                'epoch': epoch, 'model_state_dict': model.state_dict(),\n"
         "                'optimizer_state_dict': optimizer.state_dict(),\n"
         "                'train_loss': avg_train, 'val_loss': avg_val, 'val_dice': avg_dice,\n"
         "                'class_names': train_dataset.class_names,\n"
         "            }, os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth'))\n",
         SEL_COMMENT.replace("The R1 Dice above", "The R1 16 x 16-grid Dice above") +
         "        sel_dice = SELECTOR.score(model)\n"
         "        if SELECTOR.update(epoch + 1, sel_dice):\n"
         "            best_dice = sel_dice\n"
         "            epochs_no_improve = 0\n"
         "            torch.save({\n"
         "                'epoch': epoch, 'model_state_dict': model.state_dict(),\n"
         "                'train_loss': avg_train, 'val_loss': avg_val, 'val_dice': avg_dice,\n"
         "                'val_pooled_dice_512': sel_dice,\n"
         "                'class_names': train_dataset.class_names,\n"
         "            }, os.path.join(SAVE_DIR, 'best_model.pth'))\n"
         '            print(f"Best model saved (pooled val Dice: {sel_dice:.4f})")\n'
         "        else:\n"
         "            epochs_no_improve += 1\n"
         "        # (R1 also saved a full checkpoint with optimizer state every 10 epochs; omitted here.)\n"),
        ("    n_ran = len(train_losses)\n",
         "    n_ran = len(train_losses)\n"
         "    SELECTOR.finalize(n_ran, os.path.join(SAVE_DIR, 'best_model.pth'))\n"),
        ('    for _split in ("valid", "test"):\n',
         '    for _split in ("valid",):  # LOTO: the test fold is never opened during training\n'),
        # (5) LOTO: R1's geometric training augmentations now move the masks with the image
        ("        processed_masks = []\n",
         "        processed_masks = []\n"
         "        native_masks = []  # LOTO: masks at native resolution for the joint training transform\n"),
        ("            mask_resized = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)\n",
         "            native_masks.append(mask.astype(np.uint8))  # LOTO\n"
         "            mask_resized = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)\n"),
        ("        if self.transforms:\n"
         "            image = self.transforms(image)\n",
         "        if getattr(self.transforms, 'joint', False):\n"
         "            # LOTO: the same geometric augmentation is applied to the image and to its masks\n"
         "            image, joint_masks = self.transforms(\n"
         "                image, native_masks or [np.zeros(image.shape[:2], np.uint8)])\n"
         "            if native_masks:\n"
         "                processed_masks = list(joint_masks)\n"
         "        elif self.transforms:\n"
         "            image = self.transforms(image)\n"),
        ("def get_transforms(train=True):\n"
         "    if train:\n"
         "        return transforms.Compose([\n"
         "            transforms.ToPILImage(),\n"
         "            transforms.RandomHorizontalFlip(),\n"
         "            transforms.RandomRotation(10),\n"
         "            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),\n"
         "            transforms.RandomVerticalFlip(),\n"
         "            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),\n"
         "            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),\n"
         "            transforms.ToTensor(),\n"
         "            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n"
         "        ])\n",
         "class JointTrainTransform:\n"
         "    \"\"\"LOTO: R1's training augmentations (same operations, parameters and order) applied to the\n"
         "    image AND its instance masks with torchvision.transforms.v2. R1 applied them to the image\n"
         "    only, so image and masks were misaligned. Masks use nearest-neighbour resampling and fill 0;\n"
         "    ColorJitter and Normalize act on the image only. Output: normalised 3x512x512 image and\n"
         "    N x 512 x 512 float masks, as in R1.\"\"\"\n"
         "    joint = True\n"
         "\n"
         "    def __init__(self):\n"
         "        from torchvision.transforms import v2\n"
         "        self.t = v2.Compose([\n"
         "            v2.RandomHorizontalFlip(),\n"
         "            v2.RandomRotation(10),\n"
         "            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),\n"
         "            v2.RandomVerticalFlip(),\n"
         "            v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),\n"
         "            v2.RandomResizedCrop(512, scale=(0.8, 1.0)),\n"
         "            v2.ToDtype(torch.float32, scale=True),\n"
         "            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n"
         "        ])\n"
         "\n"
         "    def __call__(self, image_rgb, masks_native):\n"
         "        from torchvision import tv_tensors  # imported here: the object must stay picklable for DataLoader workers\n"
         "        img = tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1))\n"
         "        m = tv_tensors.Mask(torch.from_numpy(np.stack(masks_native)))\n"
         "        img, m = self.t(img, m)\n"
         "        return img.as_subclass(torch.Tensor), m.as_subclass(torch.Tensor).float()\n"
         "\n"
         "\n"
         "def get_transforms(train=True):\n"
         "    if train:\n"
         "        return JointTrainTransform()  # LOTO (R1: image-only torchvision.transforms pipeline)\n"),
    ],
}


def yolo_repl(name, data_old, run_name, device_expr, done_old, done_new_tail):
    return [
        (data_old,
         LC_IMPORT.format(base="BASE", name=name) +
         'DATA = CFG["yolo_data"]        # <loto>/<fold>/rebuilt/final_yolo/data.yaml\n'),
        (SEED_OLD + f'RUN_NAME = "{run_name}"\n',
         SEED_NEW +
         'RUN_DIR = CFG["save_dir"]        # <LOTO_OUT_ROOT>/<fold>/seed<seed>/' + name + "\n"),
        (f'    run_dir = os.path.join("/workspace/vine-project-Run/{name}/runs", RUN_NAME)\n',
         "    run_dir = RUN_DIR\n"),
        ("    on_epoch_start, on_epoch_end = make_dice_early_stopper(patience=PATIENCE)\n",
         "    # LOTO: replaces the R1 Dice early-stopper; same rule (PATIENCE), scored with the shared Dice\n"
         "    on_epoch_start, on_epoch_end, SELECTOR = LC.yolo_selection_callbacks(\n"
         f'        os.path.join(CFG["data_root"], "valid"), run_dir, {device_expr}, "{name}", patience=PATIENCE)\n'),
        (f'        project="/workspace/vine-project-Run/{name}/runs",\n'
         "        name=RUN_NAME,\n",
         "        project=os.path.dirname(run_dir),\n"
         "        name=os.path.basename(run_dir),\n"),
        (done_old,
         '    SELECTOR.finalize(SELECTOR.n_epochs, os.path.join(run_dir, "weights", "best_unified.pt"))\n'
         + done_new_tail),
    ]


REPL["YOLO11seg"] = yolo_repl(
    "YOLO11seg", 'DATA = os.path.abspath(os.path.join(BASE, "..", "rebuilt", "final_yolo", "data.yaml"))\n',
    "vine_clean_v6", "device",
    '    print("Training done. Run dir:", run_dir)\n'
    '    print("Ultralytics best.pt (native fitness) and best_dice.pt (val-Dice-selected) both in",\n'
    '          os.path.join(run_dir, "weights"))\n',
    '    print("Training done. Run dir:", run_dir)\n'
    '    print("Checkpoint for testing: weights/best_unified.pt (uniform criterion); Ultralytics best.pt "\n'
    '          "(native fitness) is not used.")\n')
REPL["PGSS_YOLO11"] = yolo_repl(
    "PGSS_YOLO11", 'DATA = os.path.abspath(os.path.join(BASE, "..", "rebuilt", "final_yolo", "data.yaml"))'
                   '  # same clean split as the other models\n',
    "pgss_vine_v6", "DEVICE",
    '    print("Training done. Weights:", os.path.join(run_dir, "weights", "best.pt"))\n'
    '    print("Ultralytics best.pt (native fitness) and best_dice.pt (val-Dice-selected) both in",\n'
    '          os.path.join(run_dir, "weights"))\n',
    '    print("Training done. Run dir:", run_dir)\n'
    '    print("Checkpoint for testing: weights/best_unified.pt (uniform criterion); Ultralytics best.pt "\n'
    '          "(native fitness) is not used.")\n')


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    repo = sys.argv[1]
    os.makedirs(os.path.join(HERE, "patches"), exist_ok=True)
    for name, (rel, out_name, ref) in SRC.items():
        src = os.path.join(repo, rel)
        got = sha256(src)
        if got != ref:
            raise SystemExit(f"{rel}: SHA-256 {got} != b2f792f {ref}")
        text = open(src, encoding="utf-8").read()
        new = text
        for old, rep in REPL[name]:
            n = new.count(old)
            if n != 1:
                raise SystemExit(f"{rel}: replacement anchor found {n} times (expected 1):\n{old}")
            new = new.replace(old, rep)
        header = (f"# LOTO version of {rel} (GitHub b2f792f, SHA-256 {ref[:16]}...), generated by\n"
                  f"# make_loto_training_scripts.py. Changes vs R1: see patches/{name}.diff. Launch through\n"
                  f"# run_step1_training.sh (it sets LOTO_FOLD, LOTO_SEED, LOTO_DATA_ROOT, LOTO_OUT_ROOT).\n")
        new = header + new
        compile(new, out_name, "exec")
        d = os.path.join(HERE, "models", name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, out_name), "w", encoding="utf-8") as f:
            f.write(new)
        diff = difflib.unified_diff(text.splitlines(True), new.splitlines(True),
                                    fromfile=f"R1/{rel}", tofile=f"models/{name}/{out_name}")
        with open(os.path.join(HERE, "patches", f"{name}.diff"), "w", encoding="utf-8") as f:
            f.writelines(diff)
        print(f"wrote models/{name}/{out_name}  ({len(REPL[name])} replacements)")
    for rel, ref in COPY_UNCHANGED.items():
        src = os.path.join(repo, rel)
        if sha256(src) != ref:
            raise SystemExit(f"{rel}: SHA-256 mismatch")
        shutil.copy2(src, os.path.join(HERE, "models", "PGSS_YOLO11", os.path.basename(rel)))
        print(f"copied {rel} unchanged")


if __name__ == "__main__":
    main()
