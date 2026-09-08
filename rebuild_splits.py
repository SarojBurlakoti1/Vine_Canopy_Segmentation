#!/usr/bin/env python3
"""
Merge the three COCO split folders (train/valid/test) of the Vines_2025
dataset into one in-memory dataset, then reassign every image to a new
train/valid/test split according to split_assignment_final.csv's
`split_new` column (matched by filename against `roboflow_filename`).

Writes rebuilt/train, rebuilt/valid, rebuilt/test with copied images and
a correct _annotations.coco.json in each.
"""
import csv
import json
import os
import shutil

SRC_ROOT = "Vines_2025.v8-vineo.coco-segmentation"
SRC_SPLITS = ["train", "valid", "test"]
CSV_PATH = "split_assignment_final.csv"
OUT_ROOT = "rebuilt"

# split_new values in the CSV use "val" while the dataset folders (and the
# output folders we're asked to produce) use "valid".
SPLIT_NEW_TO_FOLDER = {"train": "train", "val": "valid", "test": "test"}


def strip_subfolder(file_name):
    """The JSON's file_name might include a subfolder path; keep only the basename."""
    return file_name.replace("\\", "/").split("/")[-1]


def load_csv_mapping(csv_path):
    mapping = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["roboflow_filename"].strip()
            split_new = row["split_new"].strip()
            mapping[fname] = split_new
    return mapping


def merge_dataset():
    merged_images = []
    merged_annotations = []
    categories = None
    next_image_id = 1
    next_ann_id = 1

    # image basename -> (source split, source image path) for copying later
    basename_to_srcpath = {}

    for split in SRC_SPLITS:
        json_path = os.path.join(SRC_ROOT, split, "_annotations.coco.json")
        with open(json_path) as f:
            data = json.load(f)

        if categories is None:
            categories = data["categories"]
        else:
            if categories != data["categories"]:
                raise ValueError(f"Category mismatch in {split} vs earlier splits")

        old_to_new_image_id = {}
        for img in data["images"]:
            basename = strip_subfolder(img["file_name"])
            src_path = os.path.join(SRC_ROOT, split, basename)
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Image file missing on disk: {src_path}")

            new_img = dict(img)
            new_img["file_name"] = basename
            old_id = img["id"]
            new_img["id"] = next_image_id
            old_to_new_image_id[old_id] = next_image_id
            merged_images.append(new_img)
            basename_to_srcpath[basename] = src_path
            next_image_id += 1

        for ann in data["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new_image_id[ann["image_id"]]
            merged_annotations.append(new_ann)
            next_ann_id += 1

    return merged_images, merged_annotations, categories, basename_to_srcpath


def main():
    merged_images, merged_annotations, categories, basename_to_srcpath = merge_dataset()
    print(f"Merged dataset: {len(merged_images)} images, {len(merged_annotations)} annotations, "
          f"{len(categories)} categories")

    # sanity: duplicate basenames across original splits?
    seen = {}
    dup_count = 0
    for img in merged_images:
        bn = img["file_name"]
        if bn in seen:
            dup_count += 1
        seen[bn] = img["id"]
    if dup_count:
        print(f"WARNING: {dup_count} duplicate filenames across original splits (kept all as separate images)")

    csv_mapping = load_csv_mapping(CSV_PATH)
    print(f"CSV mapping entries: {len(csv_mapping)}")

    # Build image_id -> new split folder
    unmatched = []
    image_to_split = {}
    for img in merged_images:
        bn = img["file_name"]
        split_new = csv_mapping.get(bn)
        if split_new is None:
            unmatched.append(bn)
            continue
        folder = SPLIT_NEW_TO_FOLDER.get(split_new)
        if folder is None:
            raise ValueError(f"Unknown split_new value: {split_new!r} for {bn}")
        image_to_split[img["id"]] = folder

    if unmatched:
        print(f"WARNING: {len(unmatched)} images in dataset had no CSV match (excluded from output):")
        for bn in unmatched[:20]:
            print("   ", bn)
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more")

    # Also report CSV rows that never matched any image on disk
    dataset_basenames = set(img["file_name"] for img in merged_images)
    csv_only = [bn for bn in csv_mapping if bn not in dataset_basenames]
    if csv_only:
        print(f"NOTE: {len(csv_only)} CSV rows have no matching image file in the dataset")
        for bn in csv_only[:20]:
            print("   ", bn)
        if len(csv_only) > 20:
            print(f"    ... and {len(csv_only) - 20} more")

    # Group images/annotations per output split
    per_split_images = {"train": [], "valid": [], "test": []}
    for img in merged_images:
        folder = image_to_split.get(img["id"])
        if folder:
            per_split_images[folder].append(img)

    per_split_image_ids = {
        folder: set(img["id"] for img in imgs) for folder, imgs in per_split_images.items()
    }
    per_split_annotations = {"train": [], "valid": [], "test": []}
    for ann in merged_annotations:
        for folder, id_set in per_split_image_ids.items():
            if ann["image_id"] in id_set:
                per_split_annotations[folder].append(ann)
                break

    # Write output
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)

    for folder in ["train", "valid", "test"]:
        out_dir = os.path.join(OUT_ROOT, folder)
        os.makedirs(out_dir, exist_ok=True)

        for img in per_split_images[folder]:
            src_path = basename_to_srcpath[img["file_name"]]
            dst_path = os.path.join(out_dir, img["file_name"])
            shutil.copy2(src_path, dst_path)

        out_json = {
            "images": per_split_images[folder],
            "annotations": per_split_annotations[folder],
            "categories": categories,
        }
        with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
            json.dump(out_json, f)

    print("\nFinal counts:")
    for folder in ["train", "valid", "test"]:
        print(f"  {folder}: {len(per_split_images[folder])} images, "
              f"{len(per_split_annotations[folder])} annotations")


if __name__ == "__main__":
    main()
