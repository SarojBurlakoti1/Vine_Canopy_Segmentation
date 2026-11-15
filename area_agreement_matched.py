"""
area_agreement_matched.py
=========================
Matched-calibration, matched-region comparison of model canopy area with ImageJ
(addresses Reviewer comment 3), plus a decomposition of the bias into
  (i)  reference-definition difference  : ground-truth polygon area  vs ImageJ area
  (ii) model segmentation error         : model area                 vs ground-truth polygon area
All three areas are computed for the SAME image, inside the SAME region of interest (ROI) and with
the SAME per-image scale (px per cm), so calibration and region differences are removed by design.

INPUT 1  imagej.csv  (one row per image; original-resolution pixel coordinates)
    image_id, session, species, backdrop,
    imagej_area_cm2,            # ImageJ result
    px_per_cm,                  # per-image scale from ImageJ "Set Scale" (trellis-post distance)
    roi_x0, roi_y0, roi_x1, roi_y1,   # the ImageJ crop rectangle actually used for THIS image
    orig_w, orig_h              # original image size in pixels
INPUT 2  masks directory with one PNG per (source, image_id) at ORIGINAL resolution
    masks/<source>/<image_id>.png ; nonzero = canopy. <source> = gt_polygon, unet, deeplabv3, ...
    (Map each 512x512 prediction back to the original frame with the exact inverse of the
     resize used at export/training -- for a stretched resize: x_orig = x_512 * orig_w/512,
     y_orig = y_512 * orig_h/512 -- before saving.)

OUTPUT  agreement_matched.csv  (one row per source and stratum) with:
    n, mean_ref, mean_src, bias (src - ref), bias_95CI, LoA_lower, LoA_upper (Bland-Altman),
    median_log_ratio and its 95% LoA (multiplicative agreement, recommended because areas span
    >1 order of magnitude), MAE, MAPE, Lin's CCC (with Fisher-z 95% CI), Pearson r,
    Deming regression slope/intercept (error-variance ratio 1).
    plus per-image table area_per_image.csv.

Run `python -I area_agreement_matched.py --selftest` to check the code on SYNTHETIC data
(the self-test output is not a result and must not be reported).
"""
import argparse, json, os
import numpy as np
import pandas as pd


def area_cm2(mask, px_per_cm, roi):
    x0, y0, x1, y1 = [int(round(v)) for v in roi]
    sub = mask[y0:y1, x0:x1]
    return float((sub > 0).sum()) / (px_per_cm ** 2)


def lins_ccc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    mx, my = x.mean(), y.mean(); sx2, sy2 = x.var(), y.var(); sxy = ((x - mx) * (y - my)).mean()
    ccc = 2 * sxy / (sx2 + sy2 + (mx - my) ** 2)
    z = np.arctanh(ccc); se = 1 / np.sqrt(n - 3)
    return ccc, (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))


def deming(x, y, lam=1.0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = x.mean(), y.mean(); sxx = ((x - mx) ** 2).mean(); syy = ((y - my) ** 2).mean(); sxy = ((x - mx) * (y - my)).mean()
    slope = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return slope, my - slope * mx


def agreement(ref, src):
    ref = np.asarray(ref, float); src = np.asarray(src, float); d = src - ref; n = len(d)
    sd = d.std(ddof=1); se = sd / np.sqrt(n)
    lr = np.log(src / ref)
    ccc, ccc_ci = lins_ccc(ref, src); slope, icpt = deming(ref, src)
    return dict(n=n, mean_ref=ref.mean(), mean_src=src.mean(), bias=d.mean(),
                bias_ci_lo=d.mean() - 1.96 * se, bias_ci_hi=d.mean() + 1.96 * se,
                loa_lo=d.mean() - 1.96 * sd, loa_hi=d.mean() + 1.96 * sd,
                ratio_geomean=float(np.exp(lr.mean())), ratio_loa_lo=float(np.exp(lr.mean() - 1.96 * lr.std(ddof=1))),
                ratio_loa_hi=float(np.exp(lr.mean() + 1.96 * lr.std(ddof=1))),
                mae=np.abs(d).mean(), mape=float(np.mean(np.abs(d) / ref) * 100),
                ccc=ccc, ccc_ci_lo=ccc_ci[0], ccc_ci_hi=ccc_ci[1], pearson_r=float(np.corrcoef(ref, src)[0, 1]),
                deming_slope=slope, deming_intercept=icpt)


def run(imagej_csv, mask_dir, out_dir, sources, strata=("session", "backdrop", "species")):
    from PIL import Image
    ij = pd.read_csv(imagej_csv)
    rows = []
    for r in ij.itertuples():
        roi = (r.roi_x0, r.roi_y0, r.roi_x1, r.roi_y1)
        rec = dict(image_id=r.image_id, session=r.session, species=r.species, backdrop=r.backdrop, imagej=r.imagej_area_cm2)
        for s in ["gt_polygon"] + list(sources):
            m = np.asarray(Image.open(os.path.join(mask_dir, s, f"{r.image_id}.png")))
            if m.ndim == 3: m = m[..., 0]
            assert m.shape == (r.orig_h, r.orig_w), f"{s}/{r.image_id}: mask must be at original resolution"
            rec[s] = area_cm2(m, r.px_per_cm, roi)
        rows.append(rec)
    per = pd.DataFrame(rows); per.to_csv(os.path.join(out_dir, "area_per_image.csv"), index=False)
    res = []
    comparisons = [("imagej", "gt_polygon", "reference definition: GT polygon vs ImageJ")] + \
                  [("imagej", s, "model vs ImageJ (matched)") for s in sources] + \
                  [("gt_polygon", s, "model vs GT polygon (matched)") for s in sources]
    for ref, src, label in comparisons:
        res.append(dict(comparison=label, reference=ref, source=src, stratum="all", level="all", **agreement(per[ref], per[src])))
        for st in strata:
            for lev, g in per.groupby(st):
                if len(g) >= 10:
                    res.append(dict(comparison=label, reference=ref, source=src, stratum=st, level=lev, **agreement(g[ref], g[src])))
    out = pd.DataFrame(res); out.to_csv(os.path.join(out_dir, "agreement_matched.csv"), index=False)
    return per, out


def _selftest(tmp):
    """Synthetic check: GT = 1.6 x ImageJ (definition gap); model = GT x N(1, 0.05)."""
    from PIL import Image
    rng = np.random.default_rng(1)
    os.makedirs(tmp, exist_ok=True); rows = []
    srcs = ["model_a"]
    for s in ["gt_polygon"] + srcs: os.makedirs(os.path.join(tmp, "masks", s), exist_ok=True)
    for i in range(40):
        W, H = 400, 300; ppc = rng.uniform(2.0, 3.0)
        ij_px = int(rng.uniform(2000, 20000)); gt_px = int(ij_px * 1.6); md_px = int(gt_px * rng.normal(1, 0.05))
        def mk(n):
            a = np.zeros(H * W, np.uint8); a[:min(n, H * W)] = 255; return a.reshape(H, W)
        Image.fromarray(mk(gt_px)).save(os.path.join(tmp, "masks", "gt_polygon", f"im{i}.png"))
        Image.fromarray(mk(md_px)).save(os.path.join(tmp, "masks", "model_a", f"im{i}.png"))
        rows.append(dict(image_id=f"im{i}", session="s1" if i < 20 else "s2", species="x", backdrop="w",
                         imagej_area_cm2=ij_px / ppc ** 2, px_per_cm=ppc, roi_x0=0, roi_y0=0, roi_x1=W, roi_y1=H, orig_w=W, orig_h=H))
    pd.DataFrame(rows).to_csv(os.path.join(tmp, "imagej.csv"), index=False)
    per, out = run(os.path.join(tmp, "imagej.csv"), os.path.join(tmp, "masks"), tmp, srcs, strata=("session",))
    g = out[(out.comparison.str.startswith("reference")) & (out.stratum == "all")].iloc[0]
    assert abs(g.ratio_geomean - 1.6) < 0.01, g.ratio_geomean
    m = out[(out.comparison == "model vs GT polygon (matched)") & (out.stratum == "all")].iloc[0]
    assert abs(m.ratio_geomean - 1.0) < 0.05
    print("area_agreement_matched self-test passed (synthetic data only; not a result).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagej_csv"); ap.add_argument("--mask_dir"); ap.add_argument("--out_dir")
    ap.add_argument("--sources", nargs="*", default=["unet", "deeplabv3", "segformer", "maskrcnn", "solo_simplified", "yolo11seg", "yolo11pgss"])
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--tmp", default="_selftest_area")
    a = ap.parse_args()
    if a.selftest:
        _selftest(a.tmp)
    else:
        run(a.imagej_csv, a.mask_dir, a.out_dir, a.sources)
