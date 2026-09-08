#!/usr/bin/env python3
"""
Polygon-aware augmentation pipeline for rebuilt/train/ only.
rebuilt/valid/ and rebuilt/test/ are never opened by this script.

For every training image, generates exactly 3 augmented copies using:
  - Horizontal flip                       p=0.5
  - Vertical flip                         p=0.5
  - Rotation +/-15 deg                    p=0.5
  - Brightness adjustment +/-20%          p=0.4  (independent draw)
  - Contrast/exposure adjustment +/-15%   p=0.4  (independent draw)
  - Random crop-zoom, 0-20% zoom-in       p=0.3
    (interpreted as: crop a region covering 80-100% of the image AREA,
     centered/offset randomly, then resize back up to the original
     per-image width/height -- i.e. a zoom-in of up to ~20%, no stretch)

Segmentation polygons are carried through the geometric transforms as
albumentations keypoints (one flat list of (x, y) per image, tracked back
to their owning annotation/polygon), so masks move exactly with the image.
Pixel-only transforms (brightness/contrast) leave keypoints untouched.

A single global seed (42) is set on the albumentations Compose object,
which is built once and reused for every augmented sample in a fixed,
deterministic iteration order (images sorted by original image id,
augmentation copies 1..3) -- so the whole run is fully reproducible.

Output: rebuilt/train_augmented/ containing ONLY the 2367 (789 x 3)
augmented images (the untouched originals stay in rebuilt/train/) plus a
corrected _annotations.coco.json.
"""
import json
import os
import shutil

import albumentations as A
import cv2
import numpy as np

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

SRC_DIR = "rebuilt/train"
SRC_JSON = os.path.join(SRC_DIR, "_annotations.coco.json")
OUT_DIR = "rebuilt/train_augmented"
N_AUG_PER_IMAGE = 3
SEED = 42
FILL_GRAY = (114, 114, 114)


def build_pipeline(height, width, seed):
    aspect = width / height
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                fill=FILL_GRAY,
                p=0.5,
            ),
            # Brightness and contrast are independent p=0.4 draws, so each
            # limit is set on its own RandomBrightnessContrast instance
            # with the other channel's limit pinned to 0 (no-op).
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.15, p=0.4),
            # scale is an AREA fraction in albumentations' RandomResizedCrop;
            # 0.694 = (1/1.2)^2 so the linear zoom factor ranges 1.0x-1.2x
            # (i.e. "0-20% zoom"), ratio pinned to the image's own aspect
            # ratio so there is no stretching.
            A.RandomResizedCrop(
                size=(height, width),
                scale=(1.0 / 1.2 ** 2, 1.0),
                ratio=(aspect, aspect),
                p=0.3,
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        seed=seed,
    )


def polygons_to_keypoints(segmentation):
    """Flatten a COCO segmentation (list of polygons) into (x, y) keypoints
    plus the per-polygon point counts needed to regroup them afterward."""
    keypoints = []
    poly_lengths = []
    for poly in segmentation:
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        keypoints.extend(pts)
        poly_lengths.append(len(pts))
    return keypoints, poly_lengths


def keypoints_to_polygons(keypoints, poly_lengths, width, height):
    """Regroup flat keypoints back into COCO polygons, clipping to the
    image canvas so coordinates always stay within [0, width] x [0, height]."""
    polygons = []
    idx = 0
    for n in poly_lengths:
        pts = keypoints[idx: idx + n]
        idx += n
        flat = []
        for x, y in pts:
            cx = min(max(x, 0.0), width)
            cy = min(max(y, 0.0), height)
            flat.extend([cx, cy])
        polygons.append(flat)
    return polygons


def polygon_area_and_bbox(polygons):
    """Shoelace area summed over polygon parts (outer parts only, no hole
    subtraction -- matches how COCO area/bbox are derived from these
    single-part polygons) and the overall bounding box."""
    total_area = 0.0
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    any_point = False
    for flat in polygons:
        pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        if len(pts) < 3:
            continue
        any_point = True
        area = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            area += x1 * y2 - x2 * y1
        total_area += abs(area) / 2.0
        for x, y in pts:
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
    if not any_point:
        return 0.0, [0.0, 0.0, 0.0, 0.0]
    return total_area, [min_x, min_y, max_x - min_x, max_y - min_y]


def main():
    with open(SRC_JSON) as f:
        src = json.load(f)

    categories = src["categories"]
    images = sorted(src["images"], key=lambda im: im["id"])
    anns_by_image = {}
    for ann in src["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # One persistent, seeded Compose per unique (height, width) found in the
    # training set, reused across every image/augmentation of that size so
    # albumentations' internal RNG advances call-to-call instead of being
    # reset (which would otherwise make every image's Nth augmented copy
    # identical). Seeds are derived deterministically from SEED so the whole
    # run is still a pure function of one fixed seed.
    unique_sizes = sorted(set((im["height"], im["width"]) for im in images))
    pipeline_cache = {
        size: build_pipeline(size[0], size[1], seed=SEED + i)
        for i, size in enumerate(unique_sizes)
    }

    out_images = []
    out_annotations = []
    next_image_id = 1
    next_ann_id = 1
    dropped_annotations = 0

    for img in images:
        src_path = os.path.join(SRC_DIR, img["file_name"])
        image_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {src_path}")
        h, w = img["height"], img["width"]
        assert image_bgr.shape[0] == h and image_bgr.shape[1] == w, (
            f"Dimension mismatch for {img['file_name']}: "
            f"json says {w}x{h}, file is {image_bgr.shape[1]}x{image_bgr.shape[0]}"
        )

        anns = anns_by_image.get(img["id"], [])
        keypoints, structure = [], []  # structure: list of (num_polygons, poly_lengths) per ann
        for ann in anns:
            kps, poly_lengths = polygons_to_keypoints(ann["segmentation"])
            keypoints.extend(kps)
            structure.append(poly_lengths)

        pipeline = pipeline_cache[(h, w)]

        stem, ext = os.path.splitext(img["file_name"])

        for aug_idx in range(1, N_AUG_PER_IMAGE + 1):
            result = pipeline(image=image_bgr, keypoints=keypoints)
            aug_image = result["image"]
            aug_keypoints = [(x, y) for x, y in result["keypoints"]]
            out_h, out_w = aug_image.shape[0], aug_image.shape[1]

            new_file_name = f"{stem}__a{aug_idx}{ext}"
            cv2.imwrite(os.path.join(OUT_DIR, new_file_name), aug_image)

            new_image_id = next_image_id
            next_image_id += 1
            out_images.append({
                "id": new_image_id,
                "license": img.get("license", 1),
                "file_name": new_file_name,
                "height": out_h,
                "width": out_w,
                "date_captured": img.get("date_captured", ""),
                "extra": dict(img.get("extra", {}), augmented_from=img["file_name"], aug_index=aug_idx),
            })

            kp_cursor = 0
            for ann, poly_lengths in zip(anns, structure):
                n_pts = sum(poly_lengths)
                ann_kps = aug_keypoints[kp_cursor: kp_cursor + n_pts]
                kp_cursor += n_pts

                new_polygons = keypoints_to_polygons(ann_kps, poly_lengths, out_w, out_h)
                area, bbox = polygon_area_and_bbox(new_polygons)
                if area < 1.0:
                    dropped_annotations += 1
                    continue

                out_annotations.append({
                    "id": next_ann_id,
                    "image_id": new_image_id,
                    "category_id": ann["category_id"],
                    "bbox": bbox,
                    "iscrowd": ann.get("iscrowd", 0),
                    "area": area,
                    "segmentation": new_polygons,
                })
                next_ann_id += 1
            assert kp_cursor == len(aug_keypoints)

    out_json = {
        "images": out_images,
        "annotations": out_annotations,
        "categories": categories,
    }
    with open(os.path.join(OUT_DIR, "_annotations.coco.json"), "w") as f:
        json.dump(out_json, f)

    print(f"Source training images: {len(images)}")
    print(f"Augmented copies per image: {N_AUG_PER_IMAGE}")
    print(f"Final augmented training image count: {len(out_images)}")
    print(f"Final augmented training annotation count: {len(out_annotations)}")
    if dropped_annotations:
        print(f"Dropped {dropped_annotations} annotation(s) that degenerated to ~0 area "
              f"after a geometric transform (e.g. rotated/cropped fully out of frame)")


if __name__ == "__main__":
    main()
