#!/usr/bin/env python3
"""
read_imagej_outputs.py
======================
Reads and checks the output of Measure_Vine_Canopy.ijm (one results folder):
  measurements.csv, masks/<image>_mask.png, rois/<image>_rois.zip
For every row it checks, independently of ImageJ:
  * the mask exists, has the image size, is binary (0/255) and has no canopy pixel outside the region;
  * the canopy pixel count recomputed from the mask equals canopy_px;
  * the area recomputed as canopy_px / px_per_cm^2 equals canopy_area_cm2 (to 0.01 cm2);
  * px_per_cm = line_px / 121.92 and line_px = the length of the saved scale line;
  * the saved region rectangle equals roi_x, roi_y, roi_w, roi_h.
Writes <results>/checked_measurements.csv (adds check columns); exits 1 if any check fails.
Usage: python -I read_imagej_outputs.py <results_folder>
"""
import csv, io, math, os, struct, sys, zipfile
import numpy as np
from PIL import Image

KNOWN_CM = 121.92


def read_roi(b):
    """Minimal ImageJ .roi reader: returns (type, (top, left, bottom, right), line (x1, y1, x2, y2))."""
    if b[:4] != b"Iout":
        raise ValueError("not an ImageJ roi")
    rtype = b[6]
    top, left, bottom, right = struct.unpack(">hhhh", b[8:16])
    x1, y1, x2, y2 = struct.unpack(">ffff", b[18:34])
    return rtype, (top, left, bottom, right), (x1, y1, x2, y2)


def check_folder(res):
    rows = list(csv.DictReader(open(os.path.join(res, "measurements.csv"))))
    out, bad = [], 0
    for r in rows:
        base = os.path.splitext(r["file"])[0]
        msgs = []
        m = np.asarray(Image.open(os.path.join(res, "masks", base + "_mask.png")).convert("L"))
        W, H = int(r["image_w"]), int(r["image_h"])
        if m.shape != (H, W): msgs.append(f"mask shape {m.shape}")
        if not set(np.unique(m)) <= {0, 255}: msgs.append("mask not binary")
        x, y, w, h = (int(r[k]) for k in ("roi_x", "roi_y", "roi_w", "roi_h"))
        inside = np.zeros_like(m, bool); inside[y:y + h, x:x + w] = True
        n = int((m == 255).sum()); n_out = int(((m == 255) & ~inside).sum())
        if n_out: msgs.append(f"{n_out} canopy px outside region")
        if n != int(r["canopy_px"]): msgs.append(f"canopy_px {r['canopy_px']} != mask {n}")
        ppc = float(r["px_per_cm"]); L = float(r["line_px"])
        if abs(ppc - L / KNOWN_CM) > 1e-5: msgs.append("px_per_cm != line_px/121.92")
        if abs(n / ppc ** 2 - float(r["canopy_area_cm2"])) > 0.01: msgs.append("area != canopy_px/px_per_cm^2")
        z = zipfile.ZipFile(os.path.join(res, "rois", base + "_rois.zip"))
        rois = {nm: read_roi(z.read(nm)) for nm in z.namelist()}
        line = [v for k, v in rois.items() if k.startswith("scale_line")]
        rect = [v for k, v in rois.items() if k.startswith("region")]
        if len(line) != 1 or len(rect) != 1:
            msgs.append(f"roi zip has {list(rois)}")
        else:
            x1, y1, x2, y2 = line[0][2]
            if abs(math.hypot(x2 - x1, y2 - y1) - L) > 0.01: msgs.append("saved line length != line_px")
            top, left, bottom, right = rect[0][1]
            if (left, top, right - left, bottom - top) != (x, y, w, h): msgs.append("saved rectangle != roi columns")
        r2 = dict(r); r2["check_canopy_px"] = n; r2["check"] = "OK" if not msgs else "; ".join(msgs)
        bad += bool(msgs); out.append(r2)
    with open(os.path.join(res, "checked_measurements.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
    for r in out:
        print(f"{r['file']}: {r['check']}  ({r['canopy_area_cm2']} cm2)")
    print(f"{len(out)} rows, {bad} with problems")
    return bad


if __name__ == "__main__":
    sys.exit(1 if check_folder(sys.argv[1]) else 0)
