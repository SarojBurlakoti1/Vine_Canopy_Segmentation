#!/usr/bin/env python3
"""
area_table6_fig5_correlation.py
===============================
Table 6 and Fig. 5 of the R2 manuscript in the format of the first revision (regression of area
on ImageJ area, R2, r, MAE, RMSE, bias; scatter plots with 1:1 line, points coloured by species),
recalculated on the MATCHED comparison requested in reviewer comment 3: ImageJ, ground-truth
polygon and model-derived areas inside the analyst's rectangle and converted with the analyst's
per-image scale (primary analysis, Amendment 1: 144 unique photographs with a complete 4 ft scale).

Inputs (written by area_analysis_matched.py):
  ImageJ_remeasurement/area_analysis_results/per_image_areas.csv   per-photo areas (cm2)
  ImageJ_remeasurement/area_analysis_results/agreement.csv          geometric-mean ratios with
                                                                   bootstrap 95% CI (same run)
Species per photo from analysis/audit_outputs/split_assignment_final.csv.

cm2 statistics include photos where a model predicted no canopy (area 0); geometric-mean ratios
exclude them (as in the analysis plan). OLS fit: source area = a * ImageJ area + b.

Output: figures/Fig5.png/.tif/.pdf (190 mm wide, 300 dpi) and
        ImageJ_remeasurement/area_analysis_results/table6_correlation_matched.csv
Usage:  /opt/anaconda3/bin/python3 -I area_table6_fig5_correlation.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REV = os.path.expanduser("~/Downloads/Revision")
RES = os.path.join(REV, "ImageJ_remeasurement", "area_analysis_results")
SPLIT = os.path.join(REV, "JAFR_Final_Revision", "analysis", "audit_outputs", "split_assignment_final.csv")
OUT = os.path.join(REV, "JAFR_Final_Revision", "figures")
SOURCES = [("G", "Ground-truth polygon"), ("UNet", "U-Net"), ("DeepLabV3", "DeepLabv3"),
           ("SegFormer", "SegFormer"), ("MaskRCNN", "Mask R-CNN"), ("SoloV2", "Simplified SOLOv2"),
           ("YOLO11seg", "YOLO11-seg"), ("PGSS_YOLO11", "YOLO11-PGSS")]
AGR_NAME = {"UNet": "U-Net", "DeepLabV3": "DeepLabv3", "SegFormer": "SegFormer", "MaskRCNN": "Mask R-CNN",
            "SoloV2": "SOLOv2 (simplified)", "YOLO11seg": "YOLO11-seg", "PGSS_YOLO11": "YOLO11-PGSS"}
# species colours of the first-revision Fig. 5
SPECIES = {"Akebia": "#1f77b4", "Campsis": "#ff7f0e", "Hydrangea": "#2ca02c", "Lonicera": "#d62728",
           "Madame Galen": "#9467bd", "Parthenocissus": "#8c564b", "Wisteria": "#e377c2"}
INK, INK2 = "#111111", "#4a4a4a"
MM = 1 / 25.4
LIM = 32000

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42, "savefig.dpi": 300})


def load():
    per = pd.read_csv(os.path.join(RES, "per_image_areas.csv"))
    d = per[per.amend_included & ~per.no_scale].reset_index(drop=True)
    sp = pd.read_csv(SPLIT)[["roboflow_filename", "species"]]
    d = d.merge(sp, left_on="image_id", right_on="roboflow_filename", how="left")
    assert len(d) == 144 and d.species.notna().all()
    return d


def col(code):
    return "G" if code == "G" else f"Mam_{code}"


def stats(x, y):
    a, b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    e = y - x
    return dict(slope=a, intercept=b, r=r, r2=r * r, mae=np.abs(e).mean(),
                rmse=np.sqrt((e ** 2).mean()), bias=e.mean(), n=len(x), n_zero=int((y == 0).sum()))


def table(d):
    agr = pd.read_csv(os.path.join(RES, "agreement.csv"))
    agr = agr[agr.analysis == "primary (Amendment 1)"]
    rows = []
    for level, sub in (("all", d), ("unseen_in_LOTO", d[d.group == "unseen_in_LOTO"])):
        for code, name in SOURCES:
            s = stats(sub.I.values, sub[col(code)].values)
            st = "all" if level == "all" else "group"
            g = agr[(agr.stratum == st) & (agr.level == level)]
            ri = (g[g.comparison.str.startswith("reference definition")] if code == "G"
                  else g[(g.model == AGR_NAME[code]) & g.comparison.str.startswith("end to end")])
            assert len(ri) == 1, (level, code)
            ri = ri.iloc[0]
            # the cm2 statistics must equal those of the frozen analysis run
            assert abs(ri.bias - s["bias"]) < 0.01 and abs(ri.mae - s["mae"]) < 0.01, (level, code)
            assert abs(ri.pearson_r - s["r"]) < 1e-6, (level, code)
            s.update(level=level, source=name, gm_I=ri.ratio_geomean, gm_I_lo=ri.ratio_ci_lo, gm_I_hi=ri.ratio_ci_hi)
            if code != "G":
                rg = g[(g.model == AGR_NAME[code]) & g.comparison.str.startswith("segmentation error")]
                assert len(rg) == 1, (level, code)
                rg = rg.iloc[0]
                s.update(gm_G=rg.ratio_geomean, gm_G_lo=rg.ratio_ci_lo, gm_G_hi=rg.ratio_ci_hi)
            rows.append(s)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "table6_correlation_matched.csv"), index=False)
    return t


def panel(ax, d, code, name, s, letter):
    x = d.I.values; y = d[col(code)].values
    ax.plot([0, LIM], [0, LIM], ls=(0, (4, 3)), color="#7a7a7a", lw=0.9, zorder=1)
    for spn, c in SPECIES.items():
        m = (d.species == spn).values
        ax.scatter(x[m], y[m], s=11, color=c, edgecolors="white", linewidths=0.35, zorder=3)
    xs = np.array([0, x.max()])
    ax.plot(xs, s["slope"] * xs + s["intercept"], color=INK, lw=1.1, zorder=4)
    sign = "+" if s["intercept"] >= 0 else "−"
    txt = (f"y = {s['slope']:.3f}x {sign} {abs(s['intercept']):,.0f}\nMAE = {s['mae']:,.0f} cm²\n"
           f"RMSE = {s['rmse']:,.0f} cm²\nR² = {s['r2']:.3f}\nr = {s['r']:.3f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.2, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=INK2, lw=0.5), zorder=5)
    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_aspect("equal")
    loc = mt.MultipleLocator(10000)
    ax.xaxis.set_major_locator(loc); ax.yaxis.set_major_locator(mt.MultipleLocator(10000))
    f = mt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(f); ax.yaxis.set_major_formatter(f)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"({letter}) {name}", loc="left", fontsize=7.6, fontweight="bold", pad=3)
    ax.set_xlabel("ImageJ area (cm²)", fontsize=6.8, labelpad=2)
    ax.set_ylabel(("Polygon area" if code == "G" else "Predicted area") + " (cm²)", fontsize=6.8, labelpad=2)


def figure(d, t):
    fig, axes = plt.subplots(3, 3, figsize=(190 * MM, 196 * MM))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.975, bottom=0.06, wspace=0.42, hspace=0.42)
    tall = t[t.level == "all"].set_index("source")
    for k, (code, name) in enumerate(SOURCES):
        panel(axes.flat[k], d, code, name, tall.loc[name], "abcdefgh"[k])
    lg = axes.flat[8]; lg.axis("off")
    h = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=c, mec="white", mew=0.35, label=s) for s, c in SPECIES.items()]
    h += [Line2D([], [], ls=(0, (4, 3)), color="#7a7a7a", lw=0.9, label="1:1 line"),
          Line2D([], [], color=INK, lw=1.1, label="Regression fit")]
    lg.legend(handles=h, loc="center", frameon=True, fontsize=6.8, title="Species / lines", title_fontsize=7.2,
              edgecolor=INK2, borderpad=0.8, labelspacing=0.55, handlelength=2.2)
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (("png", {}), ("tif", {"pil_kwargs": {"compression": "tiff_lzw"}}), ("pdf", {})):
        fig.savefig(os.path.join(OUT, f"Fig5.{ext}"), dpi=300, **kw)
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    t = table(d)
    figure(d, t)
    pd.set_option("display.width", 250)
    print(t[["level", "source", "n", "n_zero", "slope", "intercept", "r2", "r", "mae", "rmse", "bias",
             "gm_I", "gm_I_lo", "gm_I_hi", "gm_G", "gm_G_lo", "gm_G_hi"]].round(3).to_string(index=False))
