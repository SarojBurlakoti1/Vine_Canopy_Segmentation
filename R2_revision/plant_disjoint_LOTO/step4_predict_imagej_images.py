#!/usr/bin/env python3
"""
step4_predict_imagej_images.py
==============================
Predicted label maps (512 x 512) of the 21 LOTO checkpoints (7 models x 3 folds, seed 42) for the
169 ImageJ-measured images (the R1 test set), for the matched canopy-area comparison of reviewer
comment 3 (area_agreement_matched.py).

It uses the same loading code (step2_evaluate_test_folds.load_model, with the selected-epoch check)
and the same predictors (loto_common.make_predictor) as the test evaluation. Every model of every
fold is run on every image, and nothing is combined here. Which fold model is used for which image
is decided in the area analysis.

Input: a folder with the 169 dataset images (640 x 640 export images, file names = image_id of
ImageJ_remeasurement/imagej_measurements.csv).
Output: <out_root>/<fold>/<model>.npz with image_id (N,) and pred (N, 512, 512) uint8, and
manifest.json.

Usage (GPU machine):
  python step4_predict_imagej_images.py --images /workspace/imagej_images --ids imagej_ids.txt \
      --out_root /workspace/imagej_pred
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import step2_evaluate_test_folds as S2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ids", required=True, help="text file, one image_id per line")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--folds_root", default="/workspace/Revision/LOTO_folds")
    ap.add_argument("--runs_root", default="/workspace/loto_runs")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = [x.strip() for x in open(a.ids) if x.strip()]
    paths = [os.path.join(a.images, i) for i in ids]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[:3]}")
    os.makedirs(a.out_root, exist_ok=True)
    scratch = os.path.join(a.out_root, "_import_scratch")
    man = {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "n_images": len(ids),
           "sha256_script": S2.sha256(os.path.abspath(__file__)), "runs": []}
    for fold in S2.FOLDS:
        os.makedirs(os.path.join(a.out_root, fold), exist_ok=True)
        for model in S2.MODELS:
            run_dir = os.path.join(a.runs_root, fold, f"seed{a.seed}", model)
            summ = json.load(open(os.path.join(run_dir, "selection_summary.json")))
            net, predict, ckpt, epoch1, _ = S2.load_model(model, fold, run_dir, a, scratch, device)
            if epoch1 != summ["selected_epoch"]:
                raise SystemExit(f"{fold}/{model}: checkpoint epoch {epoch1} != selected {summ['selected_epoch']}")
            t0 = time.time()
            with torch.no_grad():
                preds = np.stack([predict(net, p).astype(np.uint8) for p in paths])
            np.savez_compressed(os.path.join(a.out_root, fold, f"{model}.npz"), image_id=np.array(ids), pred=preds)
            man["runs"].append({"fold": fold, "model": model, "checkpoint": ckpt, "epoch": epoch1,
                                "checkpoint_sha256": S2.sha256(ckpt), "seconds": round(time.time() - t0, 1)})
            S2.log(f"{fold}/{model}: {len(ids)} images predicted")
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    man["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    json.dump(man, open(os.path.join(a.out_root, "manifest.json"), "w"), indent=2)
    S2.log("ALL DONE")


if __name__ == "__main__":
    main()
