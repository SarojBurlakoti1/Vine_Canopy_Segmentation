#!/usr/bin/env python3
"""
area_analysis_matched.py
========================
Matched canopy-area comparison (reviewer comment 3, R2), exactly as pre-specified in
  AREA_ANALYSIS_PLAN.md               (GitHub commit 33da919, sha256 936fa12b...)
  AREA_ANALYSIS_PLAN_AMENDMENT_1.md   (GitHub commit acec144, sha256 2ffcafd6...)

For every measured image, three areas are computed in the measurement frame, inside the analyst's
rectangle and with the analyst's per-image px/cm:
  I  ImageJ area        = canopy pixels of the saved final mask / (px per cm)^2
  G  ground-truth area  = union of all annotated polygons (Run/rebuilt/test_640, cv2.fillPoly on the
                          640 x 640 dataset grid as in loto_common.gt_512), mapped like the image
  M  model area         = non-background pixels of the 512 x 512 LOTO prediction (step 4), mapped:
                          IMG_  512 -> 640 x 640 -> 853 x 640 -> 90 deg clockwise   (nearest)
                          TRT_  512 -> 480 x 640, + 80 px background left and right (nearest)

Primary analysis (Amendment 1): 164 unique photos (for the 5 photos measured twice the TRT_ version
is used); "unseen in LOTO" photos use the model of the fold in which the photo (or its identical
TRT_ copy) is a test image; "not in LOTO" photos use the mean of the three fold models' areas.
Pre-specified version (plan, commit 33da919): 169 images, tier A -> own fold, tier C/D -> mean.
NO SCALE images (scale_review.csv) and images with ImageJ area 0 are excluded and counted.

Model areas of 0 (no canopy predicted): see AREA_ANALYSIS_DEVIATIONS.md D1 (cm2 statistics keep them;
log-scale statistics use M > 0 and report the count).

Statistics: area_agreement_matched.agreement() (named in the plan), plus a percentile bootstrap
(B = 10,000, seed 42) of the mean log ratio, the decomposition log(M/I) = log(G/I) + log(M/G), and
the ImageJ repeatability (repeat_20 and, per Amendment 1 item 5, the 5 photos measured twice).

Backdrop (plan section 3, "also by backdrop"): the date-based labels of the data files are known to
be wrong for some sessions, so the backdrop is read from each image: the median grey value of the
measurement image inside the rectangle, outside the ground-truth polygons (dilated by 15 px) and
outside the ImageJ mask. The values are bimodal with an empty interval from 125 to 153 (black
backdrops <= 125, white >= 153; all 12 images between 106 and 169 were checked by eye on 13 Nov 2026).
The threshold is 140, the middle of that interval; Otsu's threshold (125.0) would put the black
image 045_TRT_20_WF (125.0) in the white group. Set before any agreement statistic was computed.

Usage:
  /opt/anaconda3/bin/python3 -I area_analysis_matched.py --stage checks   # geometry + backdrop only
  /opt/anaconda3/bin/python3 -I area_analysis_matched.py --stage all      # everything
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # author_scripts: area_agreement_matched.py
sys.path.insert(0, HERE)                    # make_measurement_images.py (frozen geometry)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from area_agreement_matched import agreement  # noqa: E402
from make_measurement_images import map_mask_to_measurement  # noqa: E402

REV = os.path.expanduser("~/Downloads/Revision")
V2 = os.path.join(REV, "ImageJ_remeasurement", "v2_dataset_images")
AUDIT = os.path.join(REV, "JAFR_Final_Revision", "analysis", "audit_outputs")
PRED = os.path.join(REV, "LOTO_eval_results", "imagej_pred")
GT_JSON = os.path.join(REV, "Run", "rebuilt", "test_640", "_annotations.coco.json")
FOLDS = ["test_trt20", "test_trt50", "test_trt80"]
MODELS = [("UNet", "U-Net"), ("DeepLabV3", "DeepLabv3"), ("SegFormer", "SegFormer"),
          ("MaskRCNN", "Mask R-CNN"), ("SoloV2", "SOLOv2 (simplified)"),
          ("YOLO11seg", "YOLO11-seg"), ("PGSS_YOLO11", "YOLO11-PGSS")]
B_BOOT, SEED = 10_000, 42
MIN_N_STRATUM = 10
TRT_PAD, TRT_W = 80, 480

# Files fixed by the plan (section 5) and by Amendment 1. The script stops if any differs.
FROZEN = {
    os.path.join(HERE, "AREA_ANALYSIS_PLAN.md"): "936fa12b41c29ea459d6b4e4ff56e6d11a7b2f5a5dbae29af461cb2187e2c962",
    os.path.join(HERE, "AREA_ANALYSIS_PLAN_AMENDMENT_1.md"): "2ffcafd606ffbde40ae829b2a0a54cda42e9ac9c1a5c83ed551b715da5485cbe",
    os.path.join(HERE, "Measure_Vine_Canopy.ijm"): "5a266bf4937eaf2c9f5fa02deb30af6f24a774d3e0255d1b7ddc67c8a5c9c3b6",
    os.path.join(HERE, "make_measurement_images.py"): "1740f19a3ecbac037d65302cab0a82e64a7db9dbc43a5bf1622447043742831d",
    os.path.join(HERE, "read_imagej_outputs.py"): "a8d94eb86311035c1729f775d8752a67b32c3d3a81bd24abb5eca7c662da7881",
    os.path.join(V2, "measurement_manifest.csv"): "34dc5ea9313a5afc91b34ed5ff87bfec0bdc8471caae08a46596c9ce14786dcb",
    os.path.join(V2, "repeat_key.csv"): "91497a455fc8aab00ea11d6af63119041be98d6876a14d257d53fcaf9c32fb14",
    os.path.join(V2, "images_to_measure_results", "measurements.csv"): "4e0a10f4bb0c2978914a1c3f4b45c7a689336c5af3a6c8ae33b4a25b1d790372",
    os.path.join(V2, "repeat_20_results", "measurements.csv"): "448d13a2d83a178687b6d752831d44e0a2c30781ce96d67c93b4daf6b6ad0c5c",
    os.path.join(V2, "scale_review.csv"): "3ab1a4cf27853ccdd950c8299d8327f2eb6ff0ac67c3b4778b884b5016028f90",
    os.path.join(V2, "repeat_scale_review.csv"): "06d0f0d10f6e7cd8eceb0bd93f86b4fdf62f8e1db7054594bb48b7a22a93ae69",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_frozen():
    bad = [(p, sha256(p)) for p, s in FROZEN.items() if sha256(p) != s]
    if bad:
        raise SystemExit("Frozen file changed: " + "; ".join(f"{p} ({s[:12]})" for p, s in bad))
    return {os.path.relpath(p, REV): s for p, s in FROZEN.items()}


# ------------------------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------------------------
def nearest(a, w, h):
    return np.asarray(Image.fromarray(np.asarray(a, np.uint8)).resize((w, h), Image.NEAREST))


def pred_to_measurement(p512, kind):
    if kind == "stretched":
        return map_mask_to_measurement(nearest(p512, 640, 640), kind)
    out = np.zeros((640, 640), np.uint8)
    out[:, TRT_PAD:TRT_PAD + TRT_W] = nearest(p512, TRT_W, 640)
    return out


def gt_640(anns):
    canvas = np.zeros((640, 640), np.uint8)
    for ann in anns:
        if ann["category_id"] == 0 or not isinstance(ann["segmentation"], list):
            continue
        for poly in ann["segmentation"]:
            pts = np.array(poly, np.float32).reshape(-1, 2).astype(np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(canvas, [pts], 1)
    return canvas


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


# ------------------------------------------------------------------------------------------------
# Build the per-image table
# ------------------------------------------------------------------------------------------------
def build_table():
    man = pd.read_csv(os.path.join(V2, "measurement_manifest.csv"))
    meas = pd.read_csv(os.path.join(V2, "images_to_measure_results", "measurements.csv"))
    chk = pd.read_csv(os.path.join(V2, "images_to_measure_results", "checked_measurements.csv"))
    assert len(man) == 169 and len(meas) == 169 and (chk["check"] == "OK").all()
    df = man.merge(meas, on="file", validate="1:1")
    assert (df.image_w == df.meas_w).all() and (df.image_h == df.meas_h).all()
    df["prefix"] = np.where(df.image_id.str.startswith("IMG_"), "IMG_", "TRT_")
    assert ((df.prefix == "IMG_") == (df.kind == "stretched")).all()
    lb = df[df.kind == "letterbox"]
    # detected uniform-grey bar width is 79-80 px (JPEG blur at the content edge); geometry = 80 px (COCO letterbox)
    assert len(lb) == 35 and lb.gray_bar_left.between(TRT_PAD - 1, TRT_PAD).all() and lb.gray_bar_right.between(TRT_PAD - 1, TRT_PAD).all()

    # scale status (photo property)
    sr = pd.read_csv(os.path.join(V2, "scale_review.csv"))
    no_scale = set(sr.loc[sr.status.str.startswith("NO SCALE"), "file"])
    assert len(no_scale) == 20 and no_scale <= set(df.file)
    df["no_scale"] = df.file.isin(no_scale)

    # LOTO membership, folds and duplicates
    fa = pd.read_csv(os.path.join(AUDIT, "loto_fold_assignment.csv"))
    fa_by = fa.set_index("roboflow_filename")

    def test_fold(name):
        r = fa_by.loc[name]
        f = [f for f in FOLDS if r[f"fold_{f}"] == "test"]
        assert len(f) == 1 and f[0] == f"test_trt{int(r.treatment)}", (name, f)
        return f[0]

    dup = pd.read_csv(os.path.join(AUDIT, "duplicate_photos.csv"))
    assert dup.image_a.str.startswith("IMG_").all() and dup.image_b.str.startswith("TRT_").all()
    copy_of = dict(zip(dup.image_a, dup.image_b))
    ids169 = set(df.image_id)
    rows = []
    for r in df.itertuples():
        if r.prefix == "TRT_":
            assert r.image_id in fa_by.index and fa_by.loc[r.image_id, "tier"].startswith("A"), r.image_id
            rows.append(dict(copy_trt="", loto_fold=test_fold(r.image_id), tier="A",
                             group="unseen_in_LOTO", amend_included=True))
        else:
            c = copy_of.get(r.image_id, "")
            if c:
                assert c in fa_by.index, c
            twin = bool(c) and c in ids169
            rows.append(dict(copy_trt=c, loto_fold=test_fold(c) if c else "", tier="C/D",
                             group="unseen_in_LOTO" if c else "not_in_LOTO", amend_included=not twin))
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    n_copy = int((df.copy_trt != "").sum()); n_twin = int((~df.amend_included).sum())
    assert n_copy == 41 and n_twin == 5, (n_copy, n_twin)
    inc = df[df.amend_included]
    assert len(inc) == 164 and (inc.group == "unseen_in_LOTO").sum() == 71 and (inc.group == "not_in_LOTO").sum() == 93
    # the twin's TRT_ file in the 169 (for Amendment 1 item 5)
    by_id = df.set_index("image_id")
    df["twin_file"] = [by_id.loc[c, "file"] if (not a and c) else "" for c, a in zip(df.copy_trt, df.amend_included)]

    # ground truth
    coco = json.load(open(GT_JSON))
    img_by_name = {im["file_name"]: im for im in coco["images"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    assert ids169 <= set(img_by_name), "GT file names differ from the image ids"

    preds = {(f, m): np.load(os.path.join(PRED, f, f"{m}.npz")) for f in FOLDS for m, _ in MODELS}
    pidx = {}
    for (f, m), z in preds.items():
        pidx[(f, m)] = {str(i): k for k, i in enumerate(z["image_id"])}
    pred_arr = {k: z["pred"] for k, z in preds.items()}

    out = []
    for r in df.itertuples():
        im = img_by_name[r.image_id]
        assert (im["width"], im["height"]) == (640, 640)
        if r.kind == "letterbox":
            lbx = im["letterbox"]
            assert abs(lbx["scale"] - 0.3125) < 1e-6 and lbx["pad_left"] == TRT_PAD and lbx["pad_top"] == 0, lbx
        g = map_mask_to_measurement(gt_640(anns_by_img.get(im["id"], [])), r.kind) > 0
        H, W = r.meas_h, r.meas_w
        assert g.shape == (H, W)
        roi = np.zeros((H, W), bool); roi[r.roi_y:r.roi_y + r.roi_h, r.roi_x:r.roi_x + r.roi_w] = True
        base = os.path.splitext(r.file)[0]
        ij = np.asarray(Image.open(os.path.join(V2, "images_to_measure_results", "masks", base + "_mask.png")).convert("L")) == 255
        assert ij.shape == (H, W) and not (ij & ~roi).any() and int(ij.sum()) == r.canopy_px
        ppc2 = r.px_per_cm ** 2
        rec = dict(file=r.file, I_px=int(ij.sum()), G_px=int((g & roi).sum()),
                   I=ij.sum() / ppc2, G=(g & roi).sum() / ppc2,
                   G_px_outside_roi=int((g & ~roi).sum()),
                   frac_I_inside_G=float((ij & g).sum() / max(ij.sum(), 1)),
                   iou_I_G=iou(ij, g & roi))
        for f in FOLDS:
            for m, _ in MODELS:
                p = pred_to_measurement(pred_arr[(f, m)][pidx[(f, m)][r.image_id]], r.kind) > 0
                assert p.shape == (H, W)
                rec[f"M_{m}_{f}"] = (p & roi).sum() / ppc2
                rec[f"iouGM_{m}_{f}"] = iou(g & roi, p & roi)
        # backdrop: grey value inside the rectangle, outside dilated GT and outside the ImageJ mask
        grey = np.asarray(Image.open(os.path.join(V2, "images_to_measure", r.file)).convert("L"))
        gd = cv2.dilate(g.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
        bg = roi & ~gd & ~ij
        if r.kind == "letterbox":
            bg[:, :TRT_PAD] = False; bg[:, TRT_PAD + TRT_W:] = False
        rec["bg_px"] = int(bg.sum())
        rec["backdrop_grey"] = float(np.median(grey[bg])) if bg.any() else float("nan")
        out.append(rec)
    df = df.merge(pd.DataFrame(out), on="file", validate="1:1")
    for m, _ in MODELS:
        df[f"M_{m}_mean3"] = df[[f"M_{m}_{f}" for f in FOLDS]].mean(axis=1)
        own = np.array([df.loc[i, f"M_{m}_{df.loc[i, 'loto_fold']}"] if df.loc[i, "loto_fold"] else np.nan
                        for i in df.index])
        # pre-specified: tier A -> own fold, tier C/D -> mean of three
        df[f"Mpre_{m}"] = np.where(df.tier == "A", own, df[f"M_{m}_mean3"])
        # Amendment 1: unseen in LOTO -> own fold (of the photo or its TRT_ copy), else mean of three
        df[f"Mam_{m}"] = np.where(df.group == "unseen_in_LOTO", own, df[f"M_{m}_mean3"])
    return df


# ------------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------------
def boot_mean(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def agree_row(ref, src):
    """cm2 statistics on all images; log-scale statistics on images with src > 0 (deviation D1)."""
    ref = np.asarray(ref, float); src = np.asarray(src, float)
    assert (ref > 0).all() and (src >= 0).all()
    pos = src > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        a = agreement(ref, src)                       # cm2 statistics: all images, M = 0 included
    lr = np.log(src[pos] / ref[pos])
    lo, hi = boot_mean(lr)
    sd = float(lr.std(ddof=1))
    a.update(n_src_zero=int((~pos).sum()), n_log=int(pos.sum()),
             ratio_geomean=float(np.exp(lr.mean())),
             ratio_loa_lo=float(np.exp(lr.mean() - 1.96 * sd)), ratio_loa_hi=float(np.exp(lr.mean() + 1.96 * sd)),
             mean_log_ratio=float(lr.mean()), sd_log_ratio=sd,
             mean_log_ratio_ci_lo=lo, mean_log_ratio_ci_hi=hi,
             ratio_ci_lo=float(np.exp(lo)), ratio_ci_hi=float(np.exp(hi)))
    return a


def run_agreement(df, model_col, strata, label):
    """df: analysed rows; model_col(m) -> column of M; strata: list of (name, level, mask)."""
    res, dec = [], []
    for sname, lev, mask in strata:
        d = df[mask]
        if len(d) < MIN_N_STRATUM:
            continue
        base = dict(analysis=label, stratum=sname, level=lev)
        res.append(dict(base, comparison="reference definition (G vs I)", model="(none)",
                        reference="ImageJ (I)", source="GT polygons (G)", **agree_row(d.I, d.G)))
        lgi = np.log(d.G / d.I)
        for m, disp in MODELS:
            M = d[model_col(m)]
            res.append(dict(base, comparison="segmentation error (M vs G)", model=disp,
                            reference="GT polygons (G)", source="model (M)", **agree_row(d.G, M)))
            e2e = agree_row(d.I, M)
            res.append(dict(base, comparison="end to end (M vs I)", model=disp,
                            reference="ImageJ (I)", source="model (M)", **e2e))
            pos = (M > 0).values                          # deviation D1: same photos for all three terms
            lgp = lgi[pos]; lmg = np.log(M[pos] / d.G[pos]); lmi = np.log(M[pos] / d.I[pos])
            assert np.allclose(lmi, lgp + lmg)
            excl0 = e2e["mean_log_ratio_ci_lo"] > 0 or e2e["mean_log_ratio_ci_hi"] < 0
            dec.append(dict(base, model=disp, n=int(pos.sum()), n_model_zero=int((~pos).sum()),
                            mean_log_GI=float(lgp.mean()), mean_log_MG=float(lmg.mean()), mean_log_MI=float(lmi.mean()),
                            ratio_GI=float(np.exp(lgp.mean())), ratio_MG=float(np.exp(lmg.mean())), ratio_MI=float(np.exp(lmi.mean())),
                            MI_ci_lo=e2e["mean_log_ratio_ci_lo"], MI_ci_hi=e2e["mean_log_ratio_ci_hi"],
                            MI_ci_excludes_0=excl0,
                            share_reference_definition=float(lgp.mean() / lmi.mean()) if excl0 else np.nan))
    return pd.DataFrame(res), pd.DataFrame(dec)


def strata_for(d, group_col, groups):
    s = [("all", "all", np.ones(len(d), bool))]
    s += [(group_col, g, (d[group_col] == g).values) for g in groups]
    s += [("backdrop", b, (d.backdrop == b).values) for b in ("white", "black")]
    return s


def repeat_stats(pairs, scale_ok_col="scale_ok"):
    """pairs: DataFrame with ppc_1, ppc_2, I_1, I_2, rect IoU, scale_ok."""
    out = dict(n_photos=len(pairs), n_scale_ok=int(pairs[scale_ok_col].sum()))
    s = pairs[pairs[scale_ok_col]]
    pct = 100 * (s.ppc_2 - s.ppc_1) / s.ppc_1
    lr = np.log(s.I_2 / s.I_1)
    out.update(ppc_pct_mean=float(pct.mean()), ppc_pct_sd=float(pct.std(ddof=1)),
               ppc_pct_loa_lo=float(pct.mean() - 1.96 * pct.std(ddof=1)), ppc_pct_loa_hi=float(pct.mean() + 1.96 * pct.std(ddof=1)),
               ppc_abs_pct_median=float(pct.abs().median()), ppc_abs_pct_max=float(pct.abs().max()),
               area_log_mean=float(lr.mean()), area_log_sd=float(lr.std(ddof=1)),
               area_ratio_geomean=float(np.exp(lr.mean())),
               area_ratio_loa_lo=float(np.exp(lr.mean() - 1.96 * lr.std(ddof=1))),
               area_ratio_loa_hi=float(np.exp(lr.mean() + 1.96 * lr.std(ddof=1))),
               area_abs_pct_median=float((100 * (np.exp(lr) - 1)).abs().median()),
               area_abs_pct_max=float((100 * (np.exp(lr) - 1)).abs().max()),
               rect_iou_n=len(pairs), rect_iou_median=float(pairs.rect_iou.median()),
               rect_iou_min=float(pairs.rect_iou.min()), rect_iou_mean=float(pairs.rect_iou.mean()))
    return out


def rect_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def repeatability(df):
    key = pd.read_csv(os.path.join(V2, "repeat_key.csv"))
    rep = pd.read_csv(os.path.join(V2, "repeat_20_results", "measurements.csv"))
    rchk = pd.read_csv(os.path.join(V2, "repeat_20_results", "checked_measurements.csv"))
    assert len(rep) == 20 and (rchk["check"] == "OK").all()
    rsr = pd.read_csv(os.path.join(V2, "repeat_scale_review.csv"))
    rep_no_scale = set(rsr.loc[rsr.status.str.startswith("NO SCALE"), "repeat_file"])
    first = df.set_index("file")
    rows = []
    for r in key.merge(rep, left_on="repeat_file", right_on="file", suffixes=("", "_rep")).itertuples():
        f = first.loc[r.file]
        # photo status (Amendment 1 item 8) must agree with the repeat review
        assert (r.repeat_file in rep_no_scale) == bool(f.no_scale), r.repeat_file
        t1 = pd.Timestamp(f.date_time); t2 = pd.Timestamp(r.date_time)
        rows.append(dict(repeat_file=r.repeat_file, file=r.file, image_id=r.image_id,
                         amend_included=bool(f.amend_included), scale_ok=not f.no_scale,
                         ppc_1=f.px_per_cm, ppc_2=r.px_per_cm,
                         I_1=f.I, I_2=r.canopy_px / r.px_per_cm ** 2,
                         rect_iou=rect_iou((f.roi_x, f.roi_y, f.roi_w, f.roi_h), (r.roi_x, r.roi_y, r.roi_w, r.roi_h)),
                         t_first=str(t1), t_repeat=str(t2), gap_h=(t2 - t1).total_seconds() / 3600))
    per = pd.DataFrame(rows)
    # Amendment 1 item 7: R03 (IMG_1532) = R02 (TRT_20_LH); the IMG_ twin is left out
    am = per[per.amend_included]
    assert len(am) == 19 and am.scale_ok.sum() == 15 and per.scale_ok.sum() == 16
    summ = [dict(analysis="primary (Amendment 1): 19 unique photos", **repeat_stats(am)),
            dict(analysis="pre-specified: 20 repeat images", **repeat_stats(per))]
    gaps = dict(gap_h_min=float(per.gap_h.min()), gap_h_median=float(per.gap_h.median()), gap_h_max=float(per.gap_h.max()))

    # Amendment 1 item 5: the 5 photos measured twice (IMG_ frame 640 x 853 -> TRT_ frame: x*0.75+80, y*640/853)
    sx, sy = TRT_W / 640, 640 / 853
    dp = []
    for r in df[~df.amend_included].itertuples():
        t = first.loc[r.twin_file]
        L_img_in_trt = np.hypot((r.line_x2 - r.line_x1) * sx, (r.line_y2 - r.line_y1) * sy)
        rect_img_in_trt = (r.roi_x * sx + TRT_PAD, r.roi_y * sy, r.roi_w * sx, r.roi_h * sy)
        dp.append(dict(img_file=r.file, trt_file=r.twin_file, scale_ok=(not r.no_scale) and (not t.no_scale),
                       no_scale_img=bool(r.no_scale), no_scale_trt=bool(t.no_scale),
                       ppc_1=L_img_in_trt / 121.92, ppc_2=t.px_per_cm,  # both in the TRT_ frame
                       I_1=r.I, I_2=t.I,
                       rect_iou=rect_iou(rect_img_in_trt, (t.roi_x, t.roi_y, t.roi_w, t.roi_h)),
                       t_img=r.date_time, t_trt=t.date_time))
    dp = pd.DataFrame(dp)
    summ.append(dict(analysis="Amendment 1 item 5: 5 photos measured twice (IMG_ vs TRT_ version)", **repeat_stats(dp)))
    return per, dp, pd.DataFrame(summ), gaps


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["checks", "all"], required=True)
    ap.add_argument("--out", default=os.path.join(REV, "ImageJ_remeasurement", "area_analysis_results"))
    ap.add_argument("--backdrop_threshold", type=float, default=140.0,
                    help="grey value separating black (below) from white (above); see the docstring")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frozen = check_frozen()
    df = build_table()
    thr = a.backdrop_threshold
    df["backdrop"] = np.where(df.backdrop_grey >= thr, "white", "black")
    gap = df.backdrop_grey[(df.backdrop_grey > thr - 15) & (df.backdrop_grey < thr + 13)]
    assert gap.empty, f"backdrop value inside the empty interval: {gap.tolist()}"
    df["G_frac_outside_roi"] = df.G_px_outside_roi / (df.G_px + df.G_px_outside_roi)

    # geometry checks (independent of any agreement statistic)
    geo = []
    for kind, d in df.groupby("kind"):
        rec = dict(kind=kind, n=len(d), G_px_outside_roi_median=float(d.G_px_outside_roi.median()),
                   G_frac_outside_roi_median=float(d.G_frac_outside_roi.median()),
                   G_frac_outside_roi_max=float(d.G_frac_outside_roi.max()),
                   frac_I_inside_G_median=float(d.frac_I_inside_G.median()), frac_I_inside_G_min=float(d.frac_I_inside_G.min()),
                   iou_I_G_median=float(d.iou_I_G.median()))
        for m, disp in MODELS:
            own = [d.loc[i, f"iouGM_{m}_{d.loc[i, 'loto_fold']}"] for i in d.index if d.loc[i, "loto_fold"]]
            rec[f"iouGM_ownfold_median_{m}"] = float(np.median(own)) if own else np.nan
        geo.append(rec)
    geo = pd.DataFrame(geo)
    geo.to_csv(os.path.join(a.out, "geometry_check.csv"), index=False)
    df[["file", "image_id", "kind", "group", "amend_included", "bg_px", "backdrop_grey", "backdrop"]].sort_values(
        "backdrop_grey").to_csv(os.path.join(a.out, "backdrop_check.csv"), index=False)
    print(geo.T.to_string())
    hist = np.histogram(df.backdrop_grey, bins=np.arange(0, 265, 10))
    print("backdrop grey histogram (bin start: count):", {int(b): int(c) for b, c in zip(hist[1], hist[0]) if c})
    print(f"backdrop threshold {thr:.1f}: white {int((df.backdrop == 'white').sum())}, black {int((df.backdrop == 'black').sum())}")
    if a.stage == "checks":
        return

    # exclusions
    zero_I = df.I <= 0
    excl = dict(n_measured=len(df), no_scale=int(df.no_scale.sum()), imagej_area_zero=int(zero_I.sum()),
                amend_twins_left_out=int((~df.amend_included).sum()),
                primary_n_photos=int(df.amend_included.sum()),
                primary_n_analysed=int((df.amend_included & ~df.no_scale & ~zero_I).sum()),
                prespec_n_analysed=int((~df.no_scale & ~zero_I).sum()))
    ana = df[df.amend_included & ~df.no_scale & ~zero_I].reset_index(drop=True)
    pre = df[~df.no_scale & ~zero_I].reset_index(drop=True)
    for g in ("unseen_in_LOTO", "not_in_LOTO"):
        excl[f"primary_n_{g}"] = int((ana.group == g).sum())
        excl[f"primary_no_scale_{g}"] = int((df.amend_included & df.no_scale & (df.group == g)).sum())
    for t in ("A", "C/D"):
        excl[f"prespec_n_tier_{t}"] = int((pre.tier == t).sum())
    for b in ("white", "black"):
        excl[f"primary_n_backdrop_{b}"] = int((ana.backdrop == b).sum())
    excl["model_area_zero_primary"] = {m: int((ana[f"Mam_{m}"] <= 0).sum()) for m, _ in MODELS}
    excl["model_area_zero_prespecified"] = {m: int((pre[f"Mpre_{m}"] <= 0).sum()) for m, _ in MODELS}

    # primary (Amendment 1) and pre-specified analyses
    agr1, dec1 = run_agreement(ana, lambda m: f"Mam_{m}", strata_for(ana, "group", ["unseen_in_LOTO", "not_in_LOTO"]),
                               "primary (Amendment 1)")
    agr2, dec2 = run_agreement(pre, lambda m: f"Mpre_{m}", strata_for(pre, "tier", ["A", "C/D"]),
                               "pre-specified (plan 33da919)")
    # sensitivity: each fold model separately for the photos that use the mean of three
    sens = []
    for label, d, col, lev in (("primary (Amendment 1)", ana, "group", "not_in_LOTO"), ("pre-specified (plan 33da919)", pre, "tier", "C/D")):
        dd = d[d[col] == lev]
        for f in FOLDS:
            r_, _ = run_agreement(dd, lambda m, f=f: f"M_{m}_{f}", [(col, lev, np.ones(len(dd), bool))],
                                  f"{label}; sensitivity: {f} model only")
            sens.append(r_[r_.comparison != "reference definition (G vs I)"])
    sens = pd.concat(sens, ignore_index=True)

    rep_per, dup_pairs, rep_sum, gaps = repeatability(df)

    # write
    keep = [c for c in df.columns if not c.startswith("iouGM_")]
    df[keep].to_csv(os.path.join(a.out, "per_image_areas.csv"), index=False)
    pd.concat([agr1, agr2], ignore_index=True).to_csv(os.path.join(a.out, "agreement.csv"), index=False)
    pd.concat([dec1, dec2], ignore_index=True).to_csv(os.path.join(a.out, "decomposition.csv"), index=False)
    sens.to_csv(os.path.join(a.out, "sensitivity_single_fold_models.csv"), index=False)
    rep_per.to_csv(os.path.join(a.out, "repeat_per_image.csv"), index=False)
    dup_pairs.to_csv(os.path.join(a.out, "duplicate_photo_pairs.csv"), index=False)
    rep_sum.to_csv(os.path.join(a.out, "repeatability_summary.csv"), index=False)
    info = dict(run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), script_sha256=sha256(os.path.abspath(__file__)),
                frozen_inputs=frozen, deviations_sha256=sha256(os.path.join(HERE, "AREA_ANALYSIS_DEVIATIONS.md")), gt_json_sha256=sha256(GT_JSON),
                pred_manifest_sha256=sha256(os.path.join(PRED, "manifest.json")),
                backdrop_threshold=thr, bootstrap=dict(B=B_BOOT, seed=SEED), min_n_stratum=MIN_N_STRATUM,
                exclusions=excl, repeat_gaps_hours=gaps,
                versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, opencv=cv2.__version__))
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=2)
    print(json.dumps(excl, indent=1)); print(json.dumps(gaps))
    print("written to", a.out)


if __name__ == "__main__":
    main()
