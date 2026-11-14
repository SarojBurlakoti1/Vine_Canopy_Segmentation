#!/usr/bin/env bash
# STEP 1: train the seven models on the three LOTO folds (21 runs per seed), on the GPU machine.
#
#   bash run_step1_training.sh <LOTO_DIR> <OUT_DIR> [seed ...]
#     LOTO_DIR  folder written by step0_build_loto_datasets.py (contains test_trt20/ test_trt50/ test_trt80/)
#     OUT_DIR   where checkpoints and logs go, e.g. /workspace/loto_runs
#     seeds     default: 42 (the seed list must match the pre-specified analysis plan)
#
#   Optional: MODELS="UNet SegFormer" to run a subset; FOLDS="test_trt20" likewise.
#
# Each run writes <OUT_DIR>/<fold>/seed<seed>/<model>/ with
#   best_model.pth (YOLO: weights/best_unified.pt)  - the checkpoint to test
#   selection_log.csv                                - pooled validation Dice for every epoch
#   selection_summary.json                           - selected epoch (Table S9); marks the run complete
#   loto_run_config.json, run_metadata.json, training curves/logs as in R1
# Completed runs (selection_summary.json present) are skipped, so the script can be re-launched
# after an interruption. A failed run is logged and the loop continues.
set -uo pipefail

LOTO_DIR=$(cd "${1:?usage: run_step1_training.sh LOTO_DIR OUT_DIR [seed ...]}" && pwd)
OUT_DIR=${2:?usage: run_step1_training.sh LOTO_DIR OUT_DIR [seed ...]}
shift 2
SEEDS=${*:-42}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT_DIR/logs"
OUT_DIR=$(cd "$OUT_DIR" && pwd)

script_for() {
  case "$1" in
    UNet) echo UNet/train_unet_loto.py ;;
    DeepLabV3) echo DeepLabV3/train_deeplabv3_loto.py ;;
    SegFormer) echo SegFormer/train_segformer_loto.py ;;
    MaskRCNN) echo MaskRCNN/train_maskrcnn_loto.py ;;
    SoloV2) echo SoloV2/train_solov2_loto.py ;;
    YOLO11seg) echo YOLO11seg/train_yolo11_seg_loto.py ;;
    PGSS_YOLO11) echo PGSS_YOLO11/train_pgss_loto.py ;;
    *) echo "unknown model $1" >&2; return 1 ;;
  esac
}
MODELS=${MODELS:-"UNet DeepLabV3 SegFormer MaskRCNN SoloV2 YOLO11seg PGSS_YOLO11"}
FOLDS=${FOLDS:-"test_trt20 test_trt50 test_trt80"}

if [ -n "${LOTO_PATIENCE:-}" ] && [ "${LOTO_PATIENCE}" != "15" ]; then
  echo "WARNING: LOTO_PATIENCE=${LOTO_PATIENCE} (pre-specified patience is 15). Smoke tests only."
fi
if [ -n "${LOTO_MAX_EPOCHS:-}" ] && [ "${LOTO_MAX_EPOCHS}" != "100" ]; then
  echo "WARNING: LOTO_MAX_EPOCHS=${LOTO_MAX_EPOCHS} (pre-specified maximum is 100). Smoke tests only."
fi
for f in $FOLDS; do
  [ -f "$LOTO_DIR/$f/FOLD.json" ] || { echo "missing $LOTO_DIR/$f/FOLD.json: run step 0 first"; exit 1; }
done

FAILED=()
PY=${PYTHON:-python}
for seed in $SEEDS; do
  for fold in $FOLDS; do
    for m in $MODELS; do
      run="$OUT_DIR/$fold/seed$seed/$m"
      if [ -f "$run/selection_summary.json" ]; then
        echo "[skip] $fold seed$seed $m (already complete)"; continue
      fi
      log="$OUT_DIR/logs/${fold}_seed${seed}_${m}.log"
      echo "[$(date '+%F %T')] start $fold seed$seed $m  (log: $log)"
      script=$(script_for "$m") || { FAILED+=("$fold/seed$seed/$m"); continue; }
      ( cd "$HERE/models/$(dirname "$script")" && \
        LOTO_FOLD="$fold" LOTO_SEED="$seed" LOTO_DATA_ROOT="$LOTO_DIR/$fold/rebuilt/final" \
        LOTO_OUT_ROOT="$OUT_DIR" "$PY" "$(basename "$script")" ) > "$log" 2>&1
      status=$?
      if [ $status -ne 0 ] || [ ! -f "$run/selection_summary.json" ]; then
        echo "[$(date '+%F %T')] FAILED $fold seed$seed $m (exit $status); see $log"
        FAILED+=("$fold/seed$seed/$m")
      else
        echo "[$(date '+%F %T')] done  $fold seed$seed $m: $("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print('selected epoch', d['selected_epoch'], 'of', d['epochs_run'], '| pooled val Dice %.4f' % d['val_pooled_macro_dice_512'])" "$run/selection_summary.json")"
      fi
    done
  done
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All runs complete. Next: step 2 (test-fold evaluation)."
else
  echo "Failed runs (${#FAILED[@]}): ${FAILED[*]}"
  echo "Fix the cause and re-run the same command; completed runs are skipped."
  exit 1
fi
