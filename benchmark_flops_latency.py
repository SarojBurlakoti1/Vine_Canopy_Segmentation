"""
Per-model FLOPs and batch-size-1 inference-latency benchmark, same GPU, same test images.

FLOPs: single forward pass at the model's native working resolution (512x512), batch=1,
measured with torch.utils.flop_counter.FlopCounterMode for the five plain-PyTorch models
(UNet, DeepLabV3, SegFormer, Mask R-CNN, SOLOv2) and with Ultralytics' own get_flops()
(same underlying thop-style counter) for the two YOLO11 models. FLOPs exclude NMS/mask
post-processing for the YOLO models, matching standard YOLO-paper reporting convention.

Latency: wall-clock time for one image at a time (batch=1) on the same GPU used for the
main evaluation. All 169 test images are pre-loaded into memory first so disk I/O is
excluded from the timed region. For the five PyTorch models, the timed region is
preprocessing-to-tensor + model forward + argmax/threshold (i.e. everything needed to
produce the final mask). For the two YOLO11 models, the timed region is Ultralytics'
full model.predict() call (letterbox resize + forward + NMS + mask decode), since that is
how these models are actually used in practice. 10 warm-up iterations are discarded before
timing; CUDA is synchronized around every timed call.

Run: python3 benchmark_flops_latency.py 2>&1 | tee benchmark_run.log
Output: benchmark_results.json and benchmark_results.csv
"""
import os
import sys
import json
import time
import importlib.util

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.flop_counter import FlopCounterMode
from pycocotools.coco import COCO

RUN = "/root/vine-project/Run"
CKPT_ROOT = "/workspace/vine-project-Run"
COCO_TEST_DIR = f"{RUN}/rebuilt/final/test"
COCO_TEST_JSON = f"{COCO_TEST_DIR}/_annotations.coco.json"
YOLO_TEST_IMG_DIR = f"{RUN}/rebuilt/final_yolo/test/images"
IMG = 512
N_WARMUP = 10

assert torch.cuda.is_available(), "CUDA GPU required for a fair, reproducible latency benchmark"
device = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
RESULTS = {}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def time_calls(fn, inputs, n_warmup=N_WARMUP):
    """fn(x) -> anything; inputs is a list of pre-loaded per-image payloads.
    Returns (mean_ms, std_ms, median_ms, fps) over len(inputs) - n_warmup timed calls."""
    for i in range(n_warmup):
        fn(inputs[i % len(inputs)])
    torch.cuda.synchronize()
    times = []
    for x in inputs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "mean_ms": float(times.mean()), "std_ms": float(times.std()),
        "median_ms": float(np.median(times)), "min_ms": float(times.min()),
        "max_ms": float(times.max()), "fps": float(1000.0 / times.mean()),
        "n_timed_images": len(times),
    }


coco_test = COCO(COCO_TEST_JSON)
_smoke_n = os.environ.get("SMOKE_N")
if _smoke_n:
    _keep = list(coco_test.imgs.keys())[: int(_smoke_n)]
    coco_test.imgs = {k: coco_test.imgs[k] for k in _keep}
    log(f"SMOKE TEST MODE: trimmed to {len(coco_test.imgs)} images")
N_TEST = len(coco_test.imgs)
log(f"Loaded COCO test set: {N_TEST} images. GPU: {GPU_NAME}")

# Pre-load every test image once (both as raw RGB arrays and PIL), so disk I/O never
# enters the timed region for any model.
RAW_RGB = {}   # iid -> (H0,W0,3) uint8 RGB, native resolution
RAW_PIL = {}   # iid -> PIL.Image RGB, native resolution
for iid, info in coco_test.imgs.items():
    path = os.path.join(COCO_TEST_DIR, info["file_name"])
    img_bgr = cv2.imread(path)
    RAW_RGB[iid] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    RAW_PIL[iid] = Image.open(path).convert("RGB")
IIDS = list(coco_test.imgs.keys())


def flops_from_counter(model, x, is_dict_input=False):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            if is_dict_input:
                model(pixel_values=x)
            else:
                model(x)
    return fc.get_total_flops()


# ===========================================================================
def bench_unet():
    log("=== UNet ===")
    mod = load_module("train_unet_mod", f"{RUN}/UNet/train_unet.py")
    model = mod.UNet(3, 8).to(device)
    ckpt = torch.load(f"{CKPT_ROOT}/UNet/checkpoints/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()

    x0 = to_tensor(RAW_PIL[IIDS[0]].resize((IMG, IMG), Image.BILINEAR)).unsqueeze(0).to(device)
    flops = flops_from_counter(model, x0)

    tensors = [to_tensor(RAW_PIL[iid].resize((IMG, IMG), Image.BILINEAR)).unsqueeze(0).to(device) for iid in IIDS]

    def step(x):
        with torch.no_grad():
            return torch.argmax(model(x), dim=1)

    lat = time_calls(step, tensors)
    RESULTS["UNet"] = {"model": "UNet", "gpu": GPU_NAME, "batch_size": 1, "input_size": IMG,
                        "GFLOPs": flops / 1e9, "params_M": count_params(model) / 1e6, "latency": lat}
    log(f"UNet: {flops/1e9:.2f} GFLOPs, {lat['mean_ms']:.2f}ms/img")


def bench_deeplabv3():
    log("=== DeepLabV3 ===")
    mod = load_module("train_deeplabv3_mod", f"{RUN}/DeepLabV3/train_deeplabv3.py")
    model = mod.build_model().to(device)
    ckpt = torch.load(f"{CKPT_ROOT}/DeepLabV3/checkpoints/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()

    x0 = to_tensor(RAW_PIL[IIDS[0]].resize((IMG, IMG), Image.BILINEAR)).unsqueeze(0).to(device)
    flops = flops_from_counter(model, x0)

    tensors = [to_tensor(RAW_PIL[iid].resize((IMG, IMG), Image.BILINEAR)).unsqueeze(0).to(device) for iid in IIDS]

    def step(x):
        with torch.no_grad():
            return torch.argmax(model(x)["out"], dim=1)

    lat = time_calls(step, tensors)
    RESULTS["DeepLabV3"] = {"model": "DeepLabV3", "gpu": GPU_NAME, "batch_size": 1, "input_size": IMG,
                             "GFLOPs": flops / 1e9, "params_M": count_params(model) / 1e6, "latency": lat}
    log(f"DeepLabV3: {flops/1e9:.2f} GFLOPs, {lat['mean_ms']:.2f}ms/img")


def bench_segformer():
    log("=== SegFormer ===")
    mod = load_module("train_segformer_mod", f"{RUN}/SegFormer/train_segformer.py")
    model = mod.build_model().to(device)
    ckpt = torch.load(f"{CKPT_ROOT}/SegFormer/checkpoints/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()
    norm = mod.NORM

    x0 = norm(to_tensor(RAW_PIL[IIDS[0]].resize((IMG, IMG), Image.BILINEAR))).unsqueeze(0).to(device)
    flops = flops_from_counter(model, x0, is_dict_input=True)

    tensors = [norm(to_tensor(RAW_PIL[iid].resize((IMG, IMG), Image.BILINEAR))).unsqueeze(0).to(device) for iid in IIDS]

    def step(x):
        with torch.no_grad():
            return torch.argmax(mod.forward_logits(model, x), dim=1)

    lat = time_calls(step, tensors)
    RESULTS["SegFormer"] = {"model": "SegFormer", "gpu": GPU_NAME, "batch_size": 1, "input_size": IMG,
                             "GFLOPs": flops / 1e9, "params_M": count_params(model) / 1e6, "latency": lat}
    log(f"SegFormer: {flops/1e9:.2f} GFLOPs, {lat['mean_ms']:.2f}ms/img")


def bench_maskrcnn():
    log("=== Mask R-CNN ===")
    mod = load_module("train_maskrcnn_mod", f"{RUN}/MaskRCNN/train_maskrcnn.py")
    model = mod.build_model().to(device)
    ckpt = torch.load(f"{CKPT_ROOT}/MaskRCNN/checkpoints/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    to_tensor = T.ToTensor()

    x0 = to_tensor(cv2.resize(RAW_RGB[IIDS[0]], (IMG, IMG))).to(device)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model([x0])
    flops = fc.get_total_flops()

    tensors = [to_tensor(cv2.resize(RAW_RGB[iid], (IMG, IMG))).to(device) for iid in IIDS]

    def step(x):
        with torch.no_grad():
            return model([x])

    lat = time_calls(step, tensors)
    RESULTS["MaskRCNN"] = {"model": "Mask R-CNN", "gpu": GPU_NAME, "batch_size": 1, "input_size": IMG,
                            "GFLOPs": flops / 1e9, "params_M": count_params(model) / 1e6, "latency": lat,
                            "note": "FLOPs/latency are data-dependent (two-stage detector: RPN "
                                    "proposal count and ROI-head batch vary per image); measured "
                                    "on this test set's actual images, not a synthetic input."}
    log(f"Mask R-CNN: {flops/1e9:.2f} GFLOPs (this image), {lat['mean_ms']:.2f}ms/img")


def bench_solov2():
    log("=== SOLOv2 ===")
    mod = load_module("train_solov2_clean_mod", f"{RUN}/SoloV2/train_solov2_clean.py")
    model = mod.SimplifiedSOLOv2(num_classes=8).to(device)
    ckpt = torch.load(f"{CKPT_ROOT}/SoloV2/checkpoints_clean/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    transform = T.Compose([
        T.ToPILImage(), T.Resize((IMG, IMG)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x0 = transform(RAW_RGB[IIDS[0]]).unsqueeze(0).to(device)
    flops = flops_from_counter(model, x0)

    tensors = [transform(RAW_RGB[iid]).unsqueeze(0).to(device) for iid in IIDS]

    def step(x):
        with torch.no_grad():
            return model(x)

    lat = time_calls(step, tensors)
    RESULTS["SoloV2"] = {"model": "SOLOv2", "gpu": GPU_NAME, "batch_size": 1, "input_size": IMG,
                          "GFLOPs": flops / 1e9, "params_M": count_params(model) / 1e6, "latency": lat}
    log(f"SOLOv2: {flops/1e9:.2f} GFLOPs, {lat['mean_ms']:.2f}ms/img")


def bench_yolo_model(name, weights_path, extra_module_dir=None):
    log(f"=== {name} ===")
    if extra_module_dir:
        sys.path.insert(0, extra_module_dir)
        import pgss_modules  # noqa: F401
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    model = YOLO(weights_path)
    yolo_device = 0

    flops = get_flops(model.model, imgsz=IMG)  # GFLOPs, single image, excludes NMS

    paths = [os.path.join(YOLO_TEST_IMG_DIR, coco_test.imgs[iid]["file_name"]) for iid in IIDS]

    def step(p):
        return model.predict(p, imgsz=IMG, conf=0.25, device=yolo_device, verbose=False)

    lat = time_calls(step, paths)
    RESULTS[name] = {"model": name, "gpu": GPU_NAME, "batch_size": 1, "input_size": IMG,
                      "GFLOPs": flops, "params_M": sum(p.numel() for p in model.model.parameters()) / 1e6,
                      "latency": lat,
                      "note": "Latency is the full Ultralytics predict() call (letterbox resize + "
                              "forward + NMS + mask decode), the standard end-to-end usage pattern. "
                              "GFLOPs is the forward-pass-only figure Ultralytics itself reports "
                              "(excludes NMS), matching standard YOLO-paper convention."}
    log(f"{name}: {flops:.2f} GFLOPs, {lat['mean_ms']:.2f}ms/img")
    if extra_module_dir:
        sys.path.pop(0)


if __name__ == "__main__":
    t0 = time.time()
    steps = [
        ("UNet", bench_unet),
        ("DeepLabV3", bench_deeplabv3),
        ("SegFormer", bench_segformer),
        ("MaskRCNN", bench_maskrcnn),
        ("SoloV2", bench_solov2),
        ("YOLO11seg", lambda: bench_yolo_model(
            "YOLO11-seg", f"{CKPT_ROOT}/YOLO11seg/runs/vine_clean_v6/weights/best_dice.pt")),
        ("PGSS_YOLO11", lambda: bench_yolo_model(
            "YOLO11-PGSS", f"{CKPT_ROOT}/PGSS_YOLO11/runs/pgss_vine_v6/weights/best_dice.pt",
            extra_module_dir=f"{RUN}/PGSS_YOLO11")),
    ]
    for step_name, fn in steps:
        try:
            fn()
        except Exception as e:
            log(f"!!! {step_name} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            RESULTS[step_name] = {"model": step_name, "error": str(e)}
        with open(f"{RUN}/benchmark_results.json", "w") as f:
            json.dump(RESULTS, f, indent=2)
        # free GPU memory between models
        torch.cuda.empty_cache()

    import csv
    with open(f"{RUN}/benchmark_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "gpu", "batch_size", "input_size", "GFLOPs", "params_M",
                    "latency_mean_ms", "latency_std_ms", "latency_median_ms", "fps", "n_timed_images"])
        for name, r in RESULTS.items():
            if "error" in r:
                w.writerow([name, "ERROR: " + r["error"]] + [""] * 9)
                continue
            lat = r["latency"]
            w.writerow([r["model"], r["gpu"], r["batch_size"], r["input_size"], r["GFLOPs"], r["params_M"],
                        lat["mean_ms"], lat["std_ms"], lat["median_ms"], lat["fps"], lat["n_timed_images"]])

    log(f"ALL DONE in {time.time()-t0:.1f}s. Wrote benchmark_results.json and benchmark_results.csv")
