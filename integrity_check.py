#!/usr/bin/env python3
"""
Integrity check, generalized to run against any list of COCO split folders.

1. Every annotation's image_id must reference an image present in that
   same file's images list (no orphan annotations).
2. Every annotation's segmentation polygon coordinates must fall within
   [0, width] for x and [0, height] for y of the referenced image.
3. (optional) Total annotation count across all checked files must equal
   an expected value -- only meaningful for the original, unaugmented
   rebuilt/{train,valid,test} set; pass --no-total-check to skip it.

Usage:
  python3 integrity_check.py                       # rebuilt/{train,valid,test}, expects 1416 total
  python3 integrity_check.py DIR1 DIR2 DIR3 ...     # arbitrary folders, no total-count check
"""
import json
import sys

DEFAULT_SPLITS = ["rebuilt/train", "rebuilt/valid", "rebuilt/test"]
EXPECTED_TOTAL_ANNOTATIONS = 1416

# Small tolerance for floating point edges right at the boundary.
EPS = 1e-6


def check_orphans(split, images, annotations):
    img_ids = set(i["id"] for i in images)
    orphans = [a["id"] for a in annotations if a["image_id"] not in img_ids]
    return orphans


def check_bounds(split, images, annotations):
    img_by_id = {i["id"]: i for i in images}
    violations = []
    for ann in annotations:
        img = img_by_id.get(ann["image_id"])
        if img is None:
            continue  # already reported as orphan
        w, h = img["width"], img["height"]
        seg = ann.get("segmentation")
        if not seg:
            continue
        # segmentation can be a list of polygons (list of flat [x1,y1,x2,y2,...])
        # or an RLE dict (has "counts"/"size") - skip RLE, only check polygons.
        if isinstance(seg, dict):
            continue
        for poly in seg:
            if not isinstance(poly, list):
                continue
            xs = poly[0::2]
            ys = poly[1::2]
            bad_x = [x for x in xs if x < -EPS or x > w + EPS]
            bad_y = [y for y in ys if y < -EPS or y > h + EPS]
            if bad_x or bad_y:
                violations.append({
                    "annotation_id": ann["id"],
                    "image_id": ann["image_id"],
                    "width": w,
                    "height": h,
                    "bad_x_sample": bad_x[:3],
                    "bad_y_sample": bad_y[:3],
                })
    return violations


def main():
    args = sys.argv[1:]
    dirs = args if args else DEFAULT_SPLITS
    do_total_check = not args  # only meaningful for the default rebuilt/{train,valid,test} set

    all_orphans = {}
    all_violations = {}
    total_annotations = 0
    per_split_counts = {}

    for d in dirs:
        path = f"{d}/_annotations.coco.json"
        with open(path) as f:
            data = json.load(f)
        images = data["images"]
        annotations = data["annotations"]
        per_split_counts[d] = len(annotations)
        total_annotations += len(annotations)

        orphans = check_orphans(d, images, annotations)
        all_orphans[d] = orphans

        violations = check_bounds(d, images, annotations)
        all_violations[d] = violations

    print("=" * 60)
    print("CHECK 1: Orphan annotations (image_id not in images list)")
    print("=" * 60)
    check1_pass = True
    for d in dirs:
        orphans = all_orphans[d]
        status = "PASS" if not orphans else "FAIL"
        if orphans:
            check1_pass = False
        print(f"  [{status}] {d}: {len(orphans)} orphan annotation(s)")
        if orphans:
            print(f"         orphan annotation ids: {orphans[:20]}")

    print()
    print("=" * 60)
    print("CHECK 2: Segmentation coordinates within image bounds")
    print("=" * 60)
    check2_pass = True
    for d in dirs:
        violations = all_violations[d]
        status = "PASS" if not violations else "FAIL"
        if violations:
            check2_pass = False
        print(f"  [{status}] {d}: {len(violations)} out-of-bounds annotation(s)")
        for v in violations[:10]:
            print(f"         ann_id={v['annotation_id']} image_id={v['image_id']} "
                  f"size=({v['width']}x{v['height']}) "
                  f"bad_x={v['bad_x_sample']} bad_y={v['bad_y_sample']}")
        if len(violations) > 10:
            print(f"         ... and {len(violations) - 10} more")

    check3_pass = None
    if do_total_check:
        print()
        print("=" * 60)
        print("CHECK 3: Total annotation count across all folders")
        print("=" * 60)
        check3_pass = total_annotations == EXPECTED_TOTAL_ANNOTATIONS
        status = "PASS" if check3_pass else "FAIL"
        per_split_str = ", ".join(f"{s}={per_split_counts[s]}" for s in dirs)
        print(f"  [{status}] total={total_annotations} (expected {EXPECTED_TOTAL_ANNOTATIONS}) "
              f"[{per_split_str}]")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Check 1 (no orphan annotations):      {'PASS' if check1_pass else 'FAIL'}")
    print(f"  Check 2 (segmentation within bounds):  {'PASS' if check2_pass else 'FAIL'}")
    if do_total_check:
        print(f"  Check 3 (total annotation count = {EXPECTED_TOTAL_ANNOTATIONS}): {'PASS' if check3_pass else 'FAIL'}")
    overall = check1_pass and check2_pass and (check3_pass if do_total_check else True)
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
