#!/usr/bin/env python3
"""
step0_build_loto_datasets.py: STEP 0 of the R2 retraining (CPU only; run once).

Run it on the machine that holds the Roboflow export used in R1
(Vines_2025.v8-vineo.coco-segmentation, the SRC_ROOT of rebuild_splits.py). The fold file uses
that export's file names; Roboflow re-hashes file names on every export, so no other export
(including the public OSF one) will match.

  python step0_build_loto_datasets.py \
      --v8 /root/vine-project/Run/Vines_2025.v8-vineo.coco-segmentation \
      --folds loto_fold_assignment.csv --manifest split_assignment_final.csv \
      --r1_scripts /root/vine-project/Run --out /root/vine-project/Run/loto --checks_only
  (then the same command without --checks_only to build the three folds)

PRE-FLIGHT CHECKS (stop on failure unless stated):
  P1  every one of the 572 LOTO images is present exactly once in the export, in the split
      recorded for it in split_assignment_final.csv (split_old_v8);
  P2  the export holds ONE file per source image (no Roboflow augmentation): number of train files
      equals the 790 manifest rows with split_old_v8 == train, and no source-name prefix that is
      unique in the manifest occurs more than once in the export. If Roboflow had generated three
      versions per image, LOTO validation/test images taken from the old training split could be
      augmented copies (checklist item B9); the script then stops;
  P3  corner-fill scan of the 572 images (black = Roboflow rotation fill, grey 114 = albumentations
      fill); a warning for visual review, because dark backdrops can trigger it;
  P4  annotations: category ids within the export's categories, annotation species agrees with the
      manifest species, polygons inside the image (warnings, listed in the report);
  P5  augment_train.py and letterbox_resize.py are byte-identical to GitHub b2f792f (SHA-256).

BUILD (per fold, in <out>/<fold>):
  rebuilt/{train,valid,test}: same layout and JSON as rebuild_splits.py (image ids assigned in the
      same merged train -> valid -> test order, so augment_train.py processes images in the same
      order), files hard-linked from the export when possible;
  then the UNCHANGED R1 scripts augment_train.py and letterbox_resize.py are run inside <out>/<fold>;
  rebuilt/final/{train,valid,test} = train_augmented (3 copies per training image, as in R1),
      valid_640, test_640;
  rebuilt/final_yolo/{train,valid,test}/{images,labels} + data.yaml (class = category_id - 1, one
      label line per polygon, coordinates normalised to [0, 1]);
  FOLD.json (fold marker checked by every LOTO training script).

ANNOTATION CORRECTIONS (optional, --corrections annotation_corrections.json): author-confirmed fixes
  of single annotations, each identified by file name + original export annotation id + category
  name (all three must match, otherwise the script stops). The annotation is left out of the
  fold's rebuilt/ JSON before augmentation, so the augmented copies inherit the fix. The export
  itself is never changed. Every correction is recorded in step0_report.json and FOLD.json.

POST-BUILD CHECKS: image counts, source names of final/test == the fold's test list (and likewise
for train/valid), group-level disjointness, 640 x 640 images, YOLO image/label pairing.
Report: <out>/step0_report.json.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd

R1_SHA256 = {  # GitHub SarojBurlakoti1/Vine_Canopy_Segmentation, commit b2f792f
    "augment_train.py": "810ccb1d50ab4dcf3269a65aa2a508f163663784cab920ff43cfc6ecabc5d66d",
    "letterbox_resize.py": "78f9e10d406a694cc06d12a9f0c360a83fb6173769ebca16910b4a6da2fe580d",
}
SPLITS = ["train", "valid", "test"]
FOLD_VALUE_TO_FOLDER = {"train": "train", "val": "valid", "test": "test"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}\.(jpg|jpeg|png)$", re.I)
AUG_SUFFIX = re.compile(r"__a[1-9]\d*(\.[A-Za-z]+)$")
IMG_EXT = (".jpg", ".jpeg", ".png")


class Report:
    def __init__(self):
        self.d = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}, "warnings": [],
                  "folds": {}}
        self.failed = []

    def check(self, name, ok, detail, fatal=True):
        self.d["checks"][name] = {"ok": bool(ok), "detail": detail}
        tag = "PASS" if ok else ("FAIL" if fatal else "WARN")
        print(f"[{tag}] {name}: {detail if isinstance(detail, str) else json.dumps(detail, default=str)[:400]}")
        if not ok and fatal:
            self.failed.append(name)

    def warn(self, msg):
        self.d["warnings"].append(msg)
        print(f"[WARN] {msg}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prefix_of(name):
    return RF_SUFFIX.sub("", name)


COPY_FILES = False   # --copy_files: always copy instead of hard-linking export images


def link_or_copy(src, dst):
    if COPY_FILES:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_new(src, dst):
    """For files step 0 itself created (final/ -> final_yolo/): hard link to save disk."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_corrections(path, index, categories, rep):
    """Validate the author-confirmed annotation corrections. Returns {file_name: {ann_id, ...}}."""
    if not path:
        rep.d["corrections"] = []
        return {}
    items = json.load(open(path))
    names = {c["id"]: c["name"] for c in categories}
    problems, remove = [], collections.defaultdict(set)
    for c in items:
        e = index.get(c["file_name"])
        if e is None:
            problems.append(f"{c['file_name']}: not in the export")
            continue
        a = next((a for a in e["anns"] if a["id"] == c["remove_annotation_id"]), None)
        if a is None:
            problems.append(f"{c['file_name']}: no annotation id {c['remove_annotation_id']}")
        elif names.get(a["category_id"]) != c["remove_category"]:
            problems.append(f"{c['file_name']}: annotation {a['id']} is {names.get(a['category_id'])}, "
                            f"not {c['remove_category']}")
        else:
            remove[c["file_name"]].add(a["id"])
    rep.d["corrections"] = items
    rep.check("C1_annotation_corrections_valid", not problems,
              {"n": len(items), "problems": problems, "file": os.path.abspath(path), "sha256": sha256(path)})
    return dict(remove)


# ------------------------------------------------------------------------------------------------
def load_export(root):
    """Index the export exactly as rebuild_splits.merge_dataset() reads it."""
    index, order, categories, on_disk = {}, [], None, {}
    for split in SPLITS:
        jp = os.path.join(root, split, "_annotations.coco.json")
        if not os.path.isfile(jp):
            raise SystemExit(f"{jp} not found: is --v8 the Roboflow export folder?")
        data = json.load(open(jp))
        if categories is None:
            categories = data["categories"]
        elif categories != data["categories"]:
            raise SystemExit(f"category list in {split} differs from earlier splits")
        anns = collections.defaultdict(list)
        for a in data["annotations"]:
            anns[a["image_id"]].append(a)
        for img in data["images"]:
            bn = img["file_name"].replace("\\", "/").split("/")[-1]
            path = os.path.join(root, split, bn)
            if not os.path.exists(path):
                raise SystemExit(f"image listed in JSON but missing on disk: {path}")
            if bn in index:
                raise SystemExit(f"duplicate file name across splits: {bn}")
            index[bn] = {"split": split, "img": img, "anns": anns[img["id"]], "path": path}
            order.append(bn)
        on_disk[split] = sum(1 for f in os.listdir(os.path.join(root, split)) if f.lower().endswith(IMG_EXT))
    return index, order, categories, on_disk


def corner_fill(path):
    a = cv2.imread(path, cv2.IMREAD_COLOR)
    if a is None:
        return None
    k = max(4, min(a.shape[:2]) // 100)
    cs = [a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:]]
    black = sum(int(c.max() < 10) for c in cs)
    grey = sum(int(np.abs(c.astype(int) - 114).max() <= 3) for c in cs)
    return {"black_corners": black, "grey114_corners": grey, "h": a.shape[0], "w": a.shape[1]}


# ------------------------------------------------------------------------------------------------
def preflight(args, rep):
    folds = pd.read_csv(args.folds)
    man = pd.read_csv(args.manifest)
    fold_cols = [c for c in folds.columns if c.startswith("fold_")]
    rep.d["inputs"] = {"v8": os.path.abspath(args.v8), "folds": os.path.abspath(args.folds),
                       "folds_sha256": sha256(args.folds), "manifest": os.path.abspath(args.manifest),
                       "manifest_sha256": sha256(args.manifest), "fold_columns": fold_cols}
    index, order, categories, on_disk = load_export(args.v8)
    cat_name = {c["id"]: c["name"] for c in categories}
    rep.d["export"] = {"images_in_json": collections.Counter(v["split"] for v in index.values()),
                       "image_files_on_disk": on_disk, "categories": categories}
    print(f"export: {dict(rep.d['export']['images_in_json'])} images; categories {cat_name}")

    # P1 presence / split of origin
    mm = man.set_index("roboflow_filename")
    missing = [f for f in folds.roboflow_filename if f not in index]
    wrong_split = [f for f in folds.roboflow_filename if f in index and f in mm.index
                   and {"valid": "valid", "val": "valid"}.get(mm.at[f, "split_old_v8"], mm.at[f, "split_old_v8"])
                   != index[f]["split"]]
    rep.check("P1_all_572_present_in_recorded_split",
              len(folds) == 572 and not missing and not wrong_split and folds.roboflow_filename.is_unique,
              {"n_fold_rows": len(folds), "missing": missing[:20], "n_missing": len(missing),
               "wrong_split": wrong_split[:20], "n_wrong_split": len(wrong_split)})

    # P2 one file per source image. Decisive test: no original-file-name prefix occurs more often
    # in the export than in the manifest (a Roboflow 3x export repeats every training prefix 3x).
    man_train = int((man.split_old_v8 == "train").sum())
    exp_prefix = collections.Counter(prefix_of(f) for f in index)
    man_prefix = collections.Counter(prefix_of(f) for f in man.roboflow_filename)
    loto_prefixes = {prefix_of(f) for f in folds.roboflow_filename}
    repeated = {p: {"export": exp_prefix[p], "manifest": n} for p, n in man_prefix.items()
                if exp_prefix.get(p, 0) > n}
    repeated_loto = {p: v for p, v in repeated.items() if p in loto_prefixes}
    n_train = rep.d["export"]["images_in_json"].get("train", 0)
    not_in_manifest = [f for f in index if f not in mm.index]
    rep.check("P2_export_not_roboflow_augmented", not repeated_loto,
              {"prefixes_more_frequent_in_export_than_manifest": len(repeated),
               "of_which_used_by_LOTO_images": len(repeated_loto),
               "examples": dict(list(repeated.items())[:10]),
               "meaning": "PASS = one file per source image for every LOTO image, so the LOTO images "
                          "are unaugmented originals (closes checklist B9)."})
    rep.check("P2b_export_size_matches_manifest",
              n_train == man_train and not not_in_manifest,
              {"train_images_in_export": n_train, "manifest_train_rows": man_train,
               "export_images_not_in_manifest": len(not_in_manifest),
               "examples": not_in_manifest[:10],
               "note": "images not in the manifest are ignored (as in rebuild_splits.py); a large "
                       "excess (about 2x the manifest) would indicate an augmented export"},
              fatal=False)

    # P3 corner fill (warning)
    flagged, sizes = [], collections.Counter()
    for f in folds.roboflow_filename:
        if f not in index:
            continue
        cf = corner_fill(index[f]["path"])
        if cf is None:
            flagged.append({"file": f, "problem": "unreadable"})
            continue
        sizes[(cf["w"], cf["h"])] += 1
        if cf["black_corners"] >= 2 or cf["grey114_corners"] >= 2:
            flagged.append({"file": f, "split_old": index[f]["split"], **cf})
    rep.d["export"]["loto_image_sizes"] = {f"{w}x{h}": n for (w, h), n in sizes.most_common()}
    rep.check("P3_no_rotation_fill_in_corners", not flagged,
              {"n_flagged": len(flagged), "flagged": flagged[:40],
               "action": "inspect flagged images visually; dark backdrops can trigger this"},
              fatal=False)

    # P4 annotations (warnings)
    bad_cat, species_mismatch, no_ann, oob = [], [], [], []
    for r in folds.itertuples():
        e = index.get(r.roboflow_filename)
        if e is None:
            continue
        W, H = e["img"]["width"], e["img"]["height"]
        names = set()
        if not e["anns"]:
            no_ann.append(r.roboflow_filename)
        for a in e["anns"]:
            if a["category_id"] not in cat_name or a["category_id"] == 0:
                bad_cat.append((r.roboflow_filename, a["category_id"]))
            names.add(cat_name.get(a["category_id"], "?"))
            if not isinstance(a["segmentation"], list):
                continue
            for poly in a["segmentation"]:
                xs, ys = np.array(poly[0::2]), np.array(poly[1::2])
                if xs.min() < -1 or ys.min() < -1 or xs.max() > W + 1 or ys.max() > H + 1:
                    oob.append(r.roboflow_filename)
        if names and names != {r.species}:
            species_mismatch.append({"file": r.roboflow_filename, "manifest": r.species,
                                     "annotated": sorted(names)})
    rep.check("P4a_category_ids_valid", not bad_cat, {"bad": bad_cat[:20]}, fatal=False)
    rep.check("P4b_annotation_species_matches_manifest", not species_mismatch,
              {"n": len(species_mismatch), "examples": species_mismatch[:20]}, fatal=False)
    rep.check("P4c_every_image_annotated", not no_ann, {"n": len(no_ann), "files": no_ann[:20]}, fatal=False)
    rep.check("P4d_polygons_inside_image", not oob, {"n": len(set(oob)), "files": sorted(set(oob))[:20]},
              fatal=False)

    # P5 R1 pipeline scripts unchanged
    hashes = {}
    for name, ref in R1_SHA256.items():
        p = os.path.join(args.r1_scripts, name)
        hashes[name] = sha256(p) if os.path.isfile(p) else None
    ok = all(hashes[n] == R1_SHA256[n] for n in R1_SHA256)
    rep.check("P5_r1_pipeline_scripts_identical_to_b2f792f", ok or args.allow_script_mismatch,
              {"found": hashes, "expected": R1_SHA256,
               "note": "" if ok else "differs from GitHub b2f792f; --allow_script_mismatch was "
                                     f"{'set' if args.allow_script_mismatch else 'not set'}"})
    return folds, index, order, categories, fold_cols


# ------------------------------------------------------------------------------------------------
def build_fold(args, rep, folds, index, order, categories, col, remove=None):
    remove = remove or {}
    n_removed = 0
    fold = col.replace("fold_", "")                     # e.g. test_trt20
    fdir = os.path.join(args.out, fold)
    if os.path.exists(fdir):
        if not args.overwrite:
            raise SystemExit(f"{fdir} exists; pass --overwrite to rebuild it")
        shutil.rmtree(fdir)
    rb = os.path.join(fdir, "rebuilt")
    assign = dict(zip(folds.roboflow_filename, folds[col]))
    lists = {s: [] for s in SPLITS}

    # --- rebuilt/{train,valid,test}: rebuild_splits.py layout (merged-order ids) ---
    per = {s: {"images": [], "annotations": []} for s in SPLITS}
    next_img, next_ann = 1, 1
    for bn in order:                                     # train -> valid -> test, JSON order
        e = index[bn]
        new_id = next_img
        next_img += 1
        ann_ids = []
        for a in e["anns"]:
            ann_ids.append(next_ann)
            next_ann += 1
        if bn not in assign:
            continue
        split = FOLD_VALUE_TO_FOLDER[assign[bn]]
        img = dict(e["img"])
        img["file_name"] = bn
        img["id"] = new_id
        per[split]["images"].append(img)
        for a, aid in zip(e["anns"], ann_ids):
            if a["id"] in remove.get(bn, ()):
                n_removed += 1                      # author-confirmed correction (see --corrections)
                continue
            na = dict(a)
            na["id"], na["image_id"] = aid, new_id
            per[split]["annotations"].append(na)
        lists[split].append(bn)
    for s in SPLITS:
        d = os.path.join(rb, s)
        os.makedirs(d, exist_ok=True)
        for img in per[s]["images"]:
            link_or_copy(index[img["file_name"]]["path"], os.path.join(d, img["file_name"]))
        json.dump({"images": per[s]["images"], "annotations": per[s]["annotations"],
                   "categories": categories}, open(os.path.join(d, "_annotations.coco.json"), "w"))
    print(f"[{fold}] rebuilt/: " + ", ".join(f"{s} {len(lists[s])}" for s in SPLITS))

    # --- unchanged R1 scripts, run inside the fold folder (they use relative rebuilt/ paths) ---
    for script in ("augment_train.py", "letterbox_resize.py"):
        logp = os.path.join(fdir, f"step0_{script.replace('.py', '')}.log")
        with open(logp, "w") as log:
            res = subprocess.run([sys.executable, os.path.abspath(os.path.join(args.r1_scripts, script))],
                                 cwd=fdir, stdout=log, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            raise SystemExit(f"[{fold}] {script} failed; see {logp}")
        print(f"[{fold}] ran {script} (log: {logp})")

    # --- final/ (COCO) ---
    final = os.path.join(rb, "final")
    os.makedirs(final)
    for src, dst in (("train_augmented", "train"), ("valid_640", "valid"), ("test_640", "test")):
        os.rename(os.path.join(rb, src), os.path.join(final, dst))

    # --- final_yolo/ ---
    names = [c["name"] for c in sorted(categories, key=lambda c: c["id"]) if c["id"] != 0]
    fy = os.path.join(rb, "final_yolo")
    n_lines = collections.Counter()
    for s in SPLITS:
        data = json.load(open(os.path.join(final, s, "_annotations.coco.json")))
        idir, ldir = os.path.join(fy, s, "images"), os.path.join(fy, s, "labels")
        os.makedirs(idir)
        os.makedirs(ldir)
        anns = collections.defaultdict(list)
        for a in data["annotations"]:
            anns[a["image_id"]].append(a)
        for img in data["images"]:
            link_new(os.path.join(final, s, img["file_name"]), os.path.join(idir, img["file_name"]))
            W, H = float(img["width"]), float(img["height"])
            lines = []
            for a in anns[img["id"]]:
                if a["category_id"] == 0 or not isinstance(a["segmentation"], list):
                    continue
                for poly in a["segmentation"]:
                    if len(poly) < 6:
                        continue
                    xy = np.array(poly, float).reshape(-1, 2)
                    xy[:, 0] = np.clip(xy[:, 0] / W, 0, 1)
                    xy[:, 1] = np.clip(xy[:, 1] / H, 0, 1)
                    lines.append(f"{a['category_id'] - 1} " + " ".join(f"{v:.6f}" for v in xy.ravel()))
            n_lines[s] += len(lines)
            with open(os.path.join(ldir, os.path.splitext(img["file_name"])[0] + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
    with open(os.path.join(fy, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(fy)}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                f"nc: {len(names)}\nnames: {json.dumps(names)}\n")

    # --- post-build checks ---
    post = {}
    fin = {s: json.load(open(os.path.join(final, s, "_annotations.coco.json"))) for s in SPLITS}
    src_names = {s: sorted(AUG_SUFFIX.sub(r"\1", im["file_name"]) for im in fin[s]["images"]) for s in SPLITS}
    post["counts_final"] = {s: len(fin[s]["images"]) for s in SPLITS}
    post["train_is_3x"] = len(fin["train"]["images"]) == 3 * len(lists["train"])
    post["sources_match_fold_lists"] = all(
        sorted(set(src_names[s])) == sorted(lists[s]) for s in SPLITS)
    grp = dict(zip(folds.roboflow_filename, folds.group_id))
    g = {s: {grp[n] for n in lists[s]} for s in SPLITS}
    post["test_groups_disjoint_from_train_valid"] = g["test"].isdisjoint(g["train"] | g["valid"])
    post["no_test_source_in_train_or_valid"] = set(lists["test"]).isdisjoint(set(lists["train"]) | set(lists["valid"]))
    post["all_final_images_640"] = all(im["width"] == 640 and im["height"] == 640
                                       for s in SPLITS for im in fin[s]["images"])
    post["yolo_pairs_ok"] = all(
        {os.path.splitext(f)[0] for f in os.listdir(os.path.join(fy, s, "images"))}
        == {os.path.splitext(f)[0] for f in os.listdir(os.path.join(fy, s, "labels"))} for s in SPLITS)
    post["yolo_label_lines"] = dict(n_lines)
    expected_removed = sum(len(v) for k, v in remove.items() if k in assign)
    post["annotation_corrections_applied"] = n_removed == expected_removed
    post["n_annotations_removed_by_corrections"] = n_removed
    post["annotation_sha256"] = {s: sha256(os.path.join(final, s, "_annotations.coco.json")) for s in SPLITS}
    ok = all(v for k, v in post.items() if isinstance(v, bool))
    rep.check(f"POST_{fold}", ok, {k: v for k, v in post.items() if k != "annotation_sha256"})
    trt = folds.loc[folds[col] == "test", "treatment"].astype(str).unique().tolist()
    marker = {"fold": fold, "fold_column": col, "test_treatment": trt,
              "n_raw": {s: len(lists[s]) for s in SPLITS}, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
              "source_export": os.path.abspath(args.v8), "fold_file_sha256": sha256(args.folds),
              "post_checks": post, "annotation_corrections": rep.d.get("corrections", []),
              "files_copied_not_linked": COPY_FILES}
    json.dump(marker, open(os.path.join(fdir, "FOLD.json"), "w"), indent=2)
    pd.DataFrame([{"roboflow_filename": n, "split": s} for s in SPLITS for n in lists[s]]).to_csv(
        os.path.join(fdir, "fold_manifest.csv"), index=False)
    rep.d["folds"][fold] = marker
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v8", required=True, help="Roboflow export folder used in R1 (train/valid/test COCO)")
    ap.add_argument("--folds", required=True, help="loto_fold_assignment.csv")
    ap.add_argument("--manifest", required=True, help="split_assignment_final.csv")
    ap.add_argument("--r1_scripts", required=True, help="folder with R1 augment_train.py and letterbox_resize.py")
    ap.add_argument("--out", required=True, help="output folder, e.g. /root/vine-project/Run/loto")
    ap.add_argument("--checks_only", action="store_true", help="run the pre-flight checks only")
    ap.add_argument("--overwrite", action="store_true", help="rebuild folds that already exist")
    ap.add_argument("--allow_script_mismatch", action="store_true",
                    help="proceed although the R1 pipeline scripts differ from b2f792f (recorded)")
    ap.add_argument("--corrections", help="annotation_corrections.json (author-confirmed fixes)")
    ap.add_argument("--copy_files", action="store_true",
                    help="copy export images instead of hard-linking them")
    args = ap.parse_args()
    global COPY_FILES
    COPY_FILES = args.copy_files
    os.makedirs(args.out, exist_ok=True)
    rep = Report()
    folds, index, order, categories, fold_cols = preflight(args, rep)
    remove = load_corrections(args.corrections, index, categories, rep)
    report_path = os.path.join(args.out, "step0_report.json")
    if rep.failed:
        rep.d["result"] = f"STOPPED: pre-flight failed {rep.failed}"
        json.dump(rep.d, open(report_path, "w"), indent=1, default=str)
        print(f"\nSTOPPED: pre-flight check(s) failed: {rep.failed}. Report: {report_path}")
        sys.exit(1)
    if args.checks_only:
        rep.d["result"] = "pre-flight passed (checks only, nothing built)"
        json.dump(rep.d, open(report_path, "w"), indent=1, default=str)
        print(f"\nPre-flight passed. Report: {report_path}. Re-run without --checks_only to build.")
        return
    oks = [build_fold(args, rep, folds, index, order, categories, c, remove) for c in fold_cols]
    rep.d["result"] = "built; all post-build checks passed" if all(oks) else "built WITH FAILED post-build checks"
    rep.d["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(rep.d, open(report_path, "w"), indent=1, default=str)
    print(f"\n{rep.d['result']}. Report: {report_path}")
    sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()
