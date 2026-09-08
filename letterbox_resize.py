#!/usr/bin/env python3
"""
Letterbox-resize (aspect-ratio-preserving resize + constant gray pad, value
114) all three final splits to 640x640:

  - rebuilt/train_augmented/  -> resized IN PLACE (this folder was created
                                  fresh by augment_train.py in this same
                                  pipeline, so overwriting it is safe)
  - rebuilt/valid/            -> left completely untouched; letterboxed
                                  copies written to rebuilt/valid_640/
  - rebuilt/test/             -> left completely untouched; letterboxed
                                  copies written to rebuilt/test_640/

rebuilt/valid_640 and rebuilt/test_640 are the "final" valid/test splits
(640x640, matching train_augmented); the original rebuilt/valid and
rebuilt/test stay exactly as verified by the previous integrity check, in
case the pristine un-letterboxed versions are ever needed again.

Each output image entry gets a "letterbox" field recording scale/pad/
original size, so the manuscript's canopy-area code can map a predicted
mask on the 640x640 canvas back to original image coordinates:
    x_orig = (x_640 - pad_left) / scale
    y_orig = (y_640 - pad_top)  / scale
"""
import json
import os
import shutil

import cv2

TARGET = 640
FILL_GRAY = (114, 114, 114)

JOBS = [
    ("rebuilt/train_augmented", "rebuilt/train_augmented", True),
    ("rebuilt/valid", "rebuilt/valid_640", False),
    ("rebuilt/test", "rebuilt/test_640", False),
]


def letterbox_geometry(orig_w, orig_h, target=TARGET):
    scale = min(target / orig_w, target / orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)
    pad_w, pad_h = target - new_w, target - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    return scale, new_w, new_h, left, top, right, bottom


def letterbox_image(image_bgr, new_w, new_h, left, top, right, bottom):
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=FILL_GRAY)


def transform_polygon(flat, scale, left, top, target=TARGET):
    out = []
    for i in range(0, len(flat), 2):
        x = flat[i] * scale + left
        y = flat[i + 1] * scale + top
        out.extend([min(max(x, 0.0), target), min(max(y, 0.0), target)])
    return out


def polygon_area_and_bbox(polygons):
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


def process(src_dir, dst_dir, in_place):
    src_json_path = os.path.join(src_dir, "_annotations.coco.json")
    with open(src_json_path) as f:
        data = json.load(f)

    if not in_place:
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir, exist_ok=True)

    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_annotations = []
    for img in data["images"]:
        src_path = os.path.join(src_dir, img["file_name"])
        image_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {src_path}")
        orig_w, orig_h = img["width"], img["height"]

        scale, new_w, new_h, left, top, right, bottom = letterbox_geometry(orig_w, orig_h)
        out_image = letterbox_image(image_bgr, new_w, new_h, left, top, right, bottom)

        dst_path = os.path.join(dst_dir, img["file_name"])
        cv2.imwrite(dst_path, out_image)

        new_img = dict(img)
        new_img["width"] = TARGET
        new_img["height"] = TARGET
        new_img["letterbox"] = {
            "scale": scale,
            "pad_left": left,
            "pad_top": top,
            "pad_right": right,
            "pad_bottom": bottom,
            "orig_width": orig_w,
            "orig_height": orig_h,
        }
        new_images.append(new_img)

        for ann in anns_by_image.get(img["id"], []):
            new_polys = [transform_polygon(p, scale, left, top) for p in ann["segmentation"]]
            area, bbox = polygon_area_and_bbox(new_polys)
            new_ann = dict(ann)
            new_ann["segmentation"] = new_polys
            new_ann["bbox"] = bbox
            new_ann["area"] = area
            new_annotations.append(new_ann)

    out_json = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": data["categories"],
    }
    with open(os.path.join(dst_dir, "_annotations.coco.json"), "w") as f:
        json.dump(out_json, f)

    return len(new_images), len(new_annotations)


def main():
    for src_dir, dst_dir, in_place in JOBS:
        n_img, n_ann = process(src_dir, dst_dir, in_place)
        tag = "(in place)" if in_place else "(new folder, source left untouched)"
        print(f"{src_dir} -> {dst_dir} {tag}: {n_img} images, {n_ann} annotations, resized to {TARGET}x{TARGET}")


if __name__ == "__main__":
    main()
