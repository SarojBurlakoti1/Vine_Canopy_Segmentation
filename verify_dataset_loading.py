#!/usr/bin/env python3
"""
Dry-run dataset-loading check for all seven model training scripts, without importing
torch/torchvision/transformers/ultralytics (not installed locally -- disk is at ~79%
capacity with only a few GB free, so the multi-GB ML stack was deliberately not installed
here). Instead this replicates exactly what each script's dataset-loading code does:
  - The 5 COCO-JSON models (DeepLabV3, MaskRCNN, UNet, SegFormer, SoloV2) all now compute
    DATA_ROOT/ROOT identically: os.path.abspath(os.path.join(<model folder>, "..", "rebuilt", "final")).
    This loads each split's _annotations.coco.json via pycocotools (the exact same call each
    script's Dataset.__init__ makes), lists image ids, and confirms every image file the
    JSON references actually exists on disk and opens cleanly with PIL.
  - The 2 Ultralytics models (YOLO11seg, PGSS_YOLO11) both now point at the same
    rebuilt/final_yolo/data.yaml. This parses it, resolves each split's images/ and labels/
    folders, and confirms image/label counts match and are non-zero.
"""
import os
import sys
import yaml
from PIL import Image
from pycocotools.coco import COCO

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

COCO_MODELS = ["DeepLabV3", "MaskRCNN", "UNet", "SegFormer", "SoloV2"]
SPLITS = ["train", "valid", "test"]


def check_coco_model(model_folder):
    data_root = os.path.abspath(os.path.join(REPO_ROOT, model_folder, "..", "rebuilt", "final"))
    print(f"\n=== {model_folder} ===")
    print(f"DATA_ROOT: {data_root}")
    ok = True
    for split in SPLITS:
        split_dir = os.path.join(data_root, split)
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            print(f"  [FAIL] {split}: annotation file not found: {ann_path}")
            ok = False
            continue
        try:
            coco = COCO(ann_path)
        except Exception as e:
            print(f"  [FAIL] {split}: COCO() failed to parse {ann_path}: {e}")
            ok = False
            continue
        image_ids = list(coco.imgs.keys())
        missing = []
        unreadable = []
        for iid in image_ids:
            info = coco.loadImgs(iid)[0]
            fpath = os.path.join(split_dir, info["file_name"])
            if not os.path.exists(fpath):
                missing.append(info["file_name"])
                continue
            try:
                with Image.open(fpath) as im:
                    im.verify()
            except Exception:
                unreadable.append(info["file_name"])
        status = "PASS" if not missing and not unreadable else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {split}: {len(image_ids)} images in JSON, "
              f"{len(coco.anns)} annotations, {len(missing)} missing files, "
              f"{len(unreadable)} unreadable files")
        if missing:
            print(f"         missing (first 5): {missing[:5]}")
        if unreadable:
            print(f"         unreadable (first 5): {unreadable[:5]}")
    return ok


def check_yolo_model(model_folder, data_yaml_rel):
    data_yaml_path = os.path.abspath(os.path.join(REPO_ROOT, model_folder, data_yaml_rel))
    print(f"\n=== {model_folder} ===")
    print(f"data.yaml: {data_yaml_path}")
    if not os.path.exists(data_yaml_path):
        print(f"  [FAIL] data.yaml not found")
        return False
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)
    base = cfg["path"]
    print(f"  path: {base}")
    ok = True
    for key, split in [("train", "train"), ("val", "valid"), ("test", "test")]:
        img_dir = os.path.join(base, cfg[key])
        lbl_dir = img_dir.replace("images", "labels")
        if not os.path.isdir(img_dir):
            print(f"  [FAIL] {split}: images dir not found: {img_dir}")
            ok = False
            continue
        if not os.path.isdir(lbl_dir):
            print(f"  [FAIL] {split}: labels dir not found: {lbl_dir}")
            ok = False
            continue
        imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        lbls = sorted(f for f in os.listdir(lbl_dir) if f.endswith(".txt"))
        img_stems = {os.path.splitext(f)[0] for f in imgs}
        lbl_stems = {os.path.splitext(f)[0] for f in lbls}
        status = "PASS" if imgs and img_stems == lbl_stems else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {split}: {len(imgs)} images, {len(lbls)} labels, "
              f"stem sets match: {img_stems == lbl_stems}")
    return ok


def main():
    results = {}
    for model in COCO_MODELS:
        results[model] = check_coco_model(model)

    results["YOLO11seg"] = check_yolo_model("YOLO11seg", os.path.join("..", "rebuilt", "final_yolo", "data.yaml"))
    results["PGSS_YOLO11"] = check_yolo_model("PGSS_YOLO11", os.path.join("..", "rebuilt", "final_yolo", "data.yaml"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for model in ["DeepLabV3", "MaskRCNN", "UNet", "SegFormer", "SoloV2", "YOLO11seg", "PGSS_YOLO11"]:
        ok = results[model]
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {model}")
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
