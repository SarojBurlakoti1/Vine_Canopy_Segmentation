#!/usr/bin/env python3
"""
make_measurement_images.py
==========================
Builds the images on which canopy area is re-measured in ImageJ (reviewer comment 3, R2).

The ImageJ measurement is made on the SAME pixels that the annotators and the models used: the
640 x 640 dataset (Roboflow export) image of each of the 169 test images of the plant-overlapping
split. Only the geometry is restored so that pixels are square and the plant is upright:

  * 134 iPhone images ("IMG_..."): the export stored the raw sensor layout (4:3 landscape, EXIF
    orientation 6 not applied) STRETCHED to 640 x 640. They are resized to 853 x 640 (4:3, bicubic)
    and rotated 90 degrees clockwise -> 640 x 853, upright, square pixels.
  * 35 "TRT_..." images: the export LETTERBOXED an upright 3:4 photo into 640 x 640 with gray
    (114) bars left and right; pixels are already square and the plant upright, so the image is
    used unchanged (only re-saved losslessly as PNG).

The kind of each image is detected from the image itself (gray side bars) and must agree with
the file name (IMG_ = stretched, TRT_ = letterbox); the script stops otherwise.

Outputs (in --out):
  images_to_measure/NNN_<photo>.png   169 images, NNN = random measuring order (seed below)
  repeat_20/RNN.png                   20 images drawn at random (seed below) for a blind repeat
                                      measurement on another day, under neutral names
  measurement_manifest.csv            one row per file: order, file, image_id, kind, transform,
                                      dataset->measurement geometry, sha256 of source and output
  repeat_key.csv                      RNN -> image_id (do not open before the repeat is done)

Geometry (continuous pixel coordinates, dataset frame (xd, yd) in [0,640)):
  stretched: xr = xd * 853/640, yr = yd   (raw layout, 853 x 640)
             rotate 90 deg clockwise:  u = 640 - yr,  v = xr   (measurement frame, 640 x 853)
  letterbox: u = xd, v = yd
Label masks of the dataset frame (ground truth, model predictions) are brought into the measurement
frame with the same resize (nearest neighbour) and rotation; see map_mask_to_measurement().
"""
import argparse, csv, hashlib, os, random, re
import numpy as np
from PIL import Image

ORDER_SEED = 20261110
REPEAT_SEED = 20261111
N_REPEAT = 20
W_STRETCH = 853          # round(640 * 4 / 3)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def detect_kind(a):
    """'letterbox' if both sides have >20 columns of uniform gray (114), else 'stretched'."""
    colstd = a.std(axis=(0, 2)); colmean = a.mean(axis=(0, 2))
    gray = (colstd < 6) & (np.abs(colmean - 114) < 12)
    L = int(np.argmin(gray)) if not gray.all() else 640
    R = int(np.argmin(gray[::-1])) if not gray.all() else 640
    return ("letterbox" if (L > 20 and R > 20) else "stretched"), L, R


def to_measurement(img, kind):
    if kind == "stretched":
        return img.resize((W_STRETCH, 640), Image.BICUBIC).transpose(Image.ROTATE_270)  # 90 deg CW
    return img


def map_mask_to_measurement(mask, kind):
    """mask: (640, 640) label array in the dataset frame -> measurement frame (nearest neighbour)."""
    m = Image.fromarray(np.asarray(mask).astype(np.uint8))
    if kind == "stretched":
        m = m.resize((W_STRETCH, 640), Image.NEAREST).transpose(Image.ROTATE_270)
    return np.asarray(m)


def short_name(image_id):
    b = re.split(r"(?:_aug1)?_(?:jpe?g)\.rf\.", image_id)[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="text file, one dataset image_id per line (169)")
    ap.add_argument("--src", required=True, help="folder with the 640x640 dataset images")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    assert len(ids) == len(set(ids))
    order = ids[:]; random.Random(ORDER_SEED).shuffle(order)
    d_img = os.path.join(a.out, "images_to_measure"); d_rep = os.path.join(a.out, "repeat_20")
    os.makedirs(d_img, exist_ok=False); os.makedirs(d_rep, exist_ok=False)
    rows = []
    for k, iid in enumerate(order, 1):
        src = os.path.join(a.src, iid)
        img = Image.open(src).convert("RGB")
        arr = np.asarray(img).astype(float)
        assert arr.shape == (640, 640, 3), (iid, arr.shape)
        kind, L, R = detect_kind(arr)
        expect = "stretched" if iid.startswith("IMG_") else "letterbox" if iid.startswith(("TRT_", "Trt-")) else None
        if kind != expect:
            raise SystemExit(f"{iid}: detected {kind}, name suggests {expect}")
        out = to_measurement(img, kind)
        fn = f"{k:03d}_{short_name(iid)}.png"
        out.save(os.path.join(d_img, fn))
        rows.append(dict(order=k, file=fn, image_id=iid, kind=kind,
                         transform=("resize 640x640->853x640 bicubic; rotate 90 cw" if kind == "stretched" else "none"),
                         meas_w=out.size[0], meas_h=out.size[1], gray_bar_left=L if kind == "letterbox" else 0,
                         gray_bar_right=R if kind == "letterbox" else 0,
                         sha256_source=sha256(src), sha256_file=sha256(os.path.join(d_img, fn))))
    rep = random.Random(REPEAT_SEED).sample(rows, N_REPEAT)
    key = []
    for j, r in enumerate(rep, 1):
        fn = f"R{j:02d}.png"
        Image.open(os.path.join(d_img, r["file"])).save(os.path.join(d_rep, fn))
        key.append(dict(repeat_file=fn, file=r["file"], image_id=r["image_id"]))
    with open(os.path.join(a.out, "measurement_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["order"]))
    with open(os.path.join(a.out, "repeat_key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    print(f"{len(rows)} images ({sum(r['kind']=='stretched' for r in rows)} stretched, "
          f"{sum(r['kind']=='letterbox' for r in rows)} letterbox); {len(key)} repeat images")


if __name__ == "__main__":
    main()
