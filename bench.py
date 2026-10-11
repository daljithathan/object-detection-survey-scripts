# Benchmarking script for Object Detection Survey
# Runs latency and AP evaluation on COCO val2017

import os
import sys
import time
import json
import csv
import argparse
import datetime
import platform
import subprocess
import types

import cv2
import numpy as np
import torch
import torchvision

# Stops HF transformers from complaining about missing packages
os.environ.setdefault("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Dataset paths - change if running on a different machine
ROOT = os.path.expanduser("~/datasets/coco")
IMG_DIR = f"{ROOT}/val2017"
ANN = f"{ROOT}/annotations/instances_val2017.json"

SZ = 640
DEV = "cuda"

# NMS thresholds
NMS_CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300      
DEPLOY_CONF = 0.25 # used for realistic latency timing                             

FIELDS = [
    "detector", "year", "slice", "lineage", "input", "ap", "ap50", "ap_native", "lat_ms", 
    "lat_range", "lat_mean", "lat_p95", "lat_model_ms", "timing", "lat_eager_ms", 
    "lat_conf001_ms", "params_M", "flops_G", "gpu_util_pct", "gpu", "precision", "n_images"
]

NATIVE_SHORT = 800
NATIVE_LONG = 1333              

def letterbox(img):
    h, w = img.shape[:2]
    r = min(SZ / h, SZ / w)
    nh, nw = round(h * r), round(w * r)
    
    top = (SZ - nh) // 2
    left = (SZ - nw) // 2
    
    out = np.full((SZ, SZ, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, [r, r, left, top, w, h]


def stretch(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_LINEAR), [SZ / w, SZ / h, 0, 0, w, h]


def native_resize(img):
    # DETR-style test resolution: shorter side 800 (longer side <= 1333), aspect ratio kept, no padding.
    # Boxes are normalised w.r.t. the resized image, so un-scaling is identical to stretch.
    h, w = img.shape[:2]
    r = min(NATIVE_SHORT / min(h, w), NATIVE_LONG / max(h, w))
    out = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_LINEAR)
    return out, [SZ / w, SZ / h, 0, 0, w, h]

PRE = {"letterbox": letterbox, "stretch": stretch, "native": native_resize}

class DS(torch.utils.data.Dataset):
    # Needs to be picklable for dataloader workers
    def __init__(self, ids, files, mode, norm):
        self.ids = ids
        self.files = files
        self.mode = mode
        self.norm = norm

    def __len__(self): 
        return len(self.ids)

    def __getitem__(self, i):
        img_path = os.path.join(IMG_DIR, self.files[i])
        img = cv2.imread(img_path)
        
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
            
        img, meta = PRE[self.mode](img)
        
        # BGR to RGB and HWC to CHW
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1).float() / 255
        
        if self.norm:
            m, s = self.norm
            x = (x - torch.tensor(m).view(3, 1, 1)) / torch.tensor(s).view(3, 1, 1)
            
        return self.ids[i], x, meta


def nms_xywh(pred, cat_ids, conf=NMS_CONF):
    # pred (N, 4+nc) float32: xywh in input px + per-class scores (objectness already folded in)
    xy = pred[:, :2]
    wh = pred[:, 2:4]
    sc = pred[:, 4:]
    
    xyxy = torch.cat([xy - wh / 2, xy + wh / 2], 1)
    i, j = (sc > conf).nonzero(as_tuple=True)
    b, s = xyxy[i], sc[i, j]
    
    if s.numel() > 30000:
        s, k = s.topk(30000)
        b, j = b[k], j[k]
        
    if s.numel() == 0: 
        return b, s, cat_ids[j]
        
    # class-offset trick to use one NMS call instead of looping
    keep = torchvision.ops.nms(b + j.to(b.dtype)[:, None] * (b.max() + 1), s, NMS_IOU)[:MAX_DET]
    return b[keep], s[keep], cat_ids[j[keep]]


def _stub_utils_plots():
    # Hack to bypass YOLOv7's seaborn import error during inference 
    # since we don't actually use the plotting functions
    if "utils.plots" in sys.modules:
        return
    stub = types.ModuleType("utils.plots")
    stub.color_list = ["#FF0000"] * 20
    stub.plot_one_box = lambda *a, **k: None
    stub.plot_one_box_PIL = lambda *a, **k: None
    stub.Annotator = None
    sys.modules["utils.plots"] = stub


class Ultra:
    # Wrapper for ultralytics pip package (YOLOv12, YOLOv13)
    mode = "letterbox"
    norm = None
    
    def __init__(self, weights): 
        self.w = os.path.expanduser(weights)
        
    def load(self, cat80):
        from ultralytics import YOLO
        self.m = YOLO(self.w).model.fuse().half().to(DEV).eval()
        self.cat = cat80
        
    @torch.inference_mode()
    def forward(self, x):
        y = self.m(x)
        if isinstance(y, (list, tuple)):
            return y[0]
        return y
        
    @torch.inference_mode()
    def post(self, y, conf=NMS_CONF): 
        return nms_xywh(y[0].T.float(), self.cat, conf)


class Legacy:
    # Wrapper for older repos (YOLOv5 / YOLOv7)
    mode = "letterbox"
    norm = None
    
    def __init__(self, repo, weights): 
        self.repo = os.path.expanduser(repo)
        self.w = weights
        
    def load(self, cat80):
        # Purge cached modules so switching repos works reliably
        for k in [k for k in list(sys.modules) if k.split(".")[0] in ("models", "utils")]:
            del sys.modules[k]
            
        sys.path[:] = [p for p in sys.path if os.path.basename(p) not in ("yolov5", "yolov7")]
        sys.path.insert(0, self.repo)

        # old checkpoints are full pickles; restore torch.load afterwards
        _orig_load = torch.load
        def _safe_load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig_load(*a, **k)
            
        torch.load = _safe_load
        
        try:
            try:
                from models.experimental import attempt_load
            except ImportError as e:
                msg = str(e).lower()
                if any(d in msg for d in ("seaborn", "matplotlib", "pandas")):
                    # print("Stubbing missing plot dependencies for YOLOv7...")
                    for k in [k for k in list(sys.modules) if k.split(".")[0] in ("models", "utils")]:
                        del sys.modules[k]
                    _stub_utils_plots()
                    from models.experimental import attempt_load
                else:
                    raise

            path = os.path.join(self.repo, self.w)
            if not os.path.exists(path):
                raise FileNotFoundError(f"checkpoint not found: {path}")
                
            try:
                m = attempt_load(path, device=DEV)            
            except TypeError:
                m = attempt_load(path, map_location=DEV)      
        finally:
            torch.load = _orig_load

        self.m = m.half().to(DEV).eval()
        self.cat = cat80

    @torch.inference_mode()
    def forward(self, x):
        y = self.m(x)
        if isinstance(y, (list, tuple)):
            return y[0]
        return y
        
    @torch.inference_mode()
    def post(self, y, conf=NMS_CONF):
        p = y[0].float()
        return nms_xywh(torch.cat([p[:, :4], p[:, 5:] * p[:, 4:5]], 1), self.cat, conf)


class HF:
    # Wrapper for HuggingFace models (DETR, RT-DETRv2, D-FINE)
    mode = "stretch"
    
    def __init__(self, repo, kind, k=100, native=False): 
        self.repo = repo
        self.kind = kind
        self.k = k
        self.native = native
        
    def load(self, cat80):
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        self.m = AutoModelForObjectDetection.from_pretrained(self.repo).half().to(DEV).eval()
        p = AutoImageProcessor.from_pretrained(self.repo)
        
        if getattr(p, "do_normalize", False):
            self.norm = (list(p.image_mean), list(p.image_std))
        else:
            self.norm = None
            
        self.map = None if len(self.m.config.id2label) >= 91 else cat80
        core = getattr(self.m, "model", None)
        
        for n in ("anchors", "valid_mask"):
            t = getattr(core, n, None)
            if isinstance(t, torch.Tensor) and t.device.type == "cpu": 
                setattr(core, n, t.to(DEV))
                
    @torch.inference_mode()
    def forward(self, x): 
        return self.m(pixel_values=x)
        
    @torch.inference_mode()
    def post(self, out, conf=None):            
        # conf is unused here: DETR family has no score-threshold/NMS
        lg = out.logits[0].float()
        bx = out.pred_boxes[0].float()
        
        if self.kind == "softmax":
            s, l = lg.softmax(-1)[:, :-1].max(-1)
        else:
            p = lg.sigmoid()
            s, idx = p.flatten().topk(self.k)
            bx, l = bx[idx // p.shape[1]], idx % p.shape[1]
            
        xyxy = torch.cat([bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2], 1) * SZ
        
        if self.map is not None:
            return xyxy, s, self.map[l]
        return xyxy, s, l


Y1 = "2020–2022"
Y2 = "2024–2025"

RELEASE = {
    "YOLOv5-S": 2020, "YOLOv5-M": 2020, "YOLOv5-L": 2020, 
    "YOLOv7": 2022, "DETR-R50": 2020, "Deformable DETR-R50": 2021, 
    "YOLOv12-S": 2025, "YOLOv12-M": 2025, "YOLOv12-L": 2025,
    "YOLOv13-S": 2025, "YOLOv13-L": 2025, 
    "RT-DETRv2-S": 2024, "RT-DETRv2-M": 2024, "RT-DETRv2-L": 2024, 
    "D-FINE-L": 2025
}

SOURCES = {   
  "YOLOv5-S/M/L": "github.com/ultralytics/yolov5 (original anchor-based; weights v7.0 yolov5{s,m,l}.pt)",
  "YOLOv7": "github.com/WongKinYiu/yolov7 (weights v0.1 yolov7.pt)",
  "YOLOv12-S/M/L": "ultralytics package, official yolo12{s,m,l}.pt",
  "YOLOv13-S/L": "github.com/iMoonLab/yolov13, official yolov13{s,l}.pt",
  "DETR-R50": "HF transformers port of facebook/detr-resnet-50",
  "Deformable DETR-R50": "HF transformers port of SenseTime/deformable-detr",
  "RT-DETRv2-S/M/L": "HF transformers port of PekingU/rtdetr_v2_r18vd / r34vd / r50vd",
  "D-FINE-L": "HF transformers port of ustc-community/dfine-large-coco"
}

def registry():
    return {
      "YOLOv5-S": (Y1, "YOLO", lambda: Legacy("~/yolov5", "yolov5s.pt")),
      "YOLOv5-M": (Y1, "YOLO", lambda: Legacy("~/yolov5", "yolov5m.pt")),
      "YOLOv5-L": (Y1, "YOLO", lambda: Legacy("~/yolov5", "yolov5l.pt")),
      "YOLOv7": (Y1, "YOLO", lambda: Legacy("~/yolov7", "yolov7.pt")),
      
      "DETR-R50": (Y1, "DETR", lambda: HF("facebook/detr-resnet-50", "softmax", native=True)),
      "Deformable DETR-R50": (Y1, "DETR", lambda: HF("SenseTime/deformable-detr", "sigmoid", 100, native=True)),
      
      "YOLOv12-S": (Y2, "YOLO", lambda: Ultra("yolo12s.pt")),
      "YOLOv12-M": (Y2, "YOLO", lambda: Ultra("yolo12m.pt")),
      "YOLOv12-L": (Y2, "YOLO", lambda: Ultra("yolo12l.pt")),
      
      "YOLOv13-S": (Y2, "YOLO", lambda: Ultra("~/yolov13/yolov13s.pt")),   
      "YOLOv13-L": (Y2, "YOLO", lambda: Ultra("~/yolov13/yolov13l.pt")),
      
      "RT-DETRv2-S": (Y2, "DETR", lambda: HF("PekingU/rtdetr_v2_r18vd", "sigmoid", 300)),
      "RT-DETRv2-M": (Y2, "DETR", lambda: HF("PekingU/rtdetr_v2_r34vd", "sigmoid", 300)),
      "RT-DETRv2-L": (Y2, "DETR", lambda: HF("PekingU/rtdetr_v2_r50vd", "sigmoid", 300)),
      
      "D-FINE-L": (Y2, "DETR", lambda: HF("ustc-community/dfine-large-coco", "sigmoid", 300)),
    }

class _ConstCache:
    # Context manager to help with CUDA graph capture issues in some models
    def __enter__(self):
        self.o = (torch.tensor, torch.as_tensor)
        self.c = {}
        
        def wrap(fn):
            def f(data, *a, **k):
                if isinstance(data, (list, tuple, int, float, bool)) and "cuda" in str(k.get("device", "")):
                    key = (repr(data), repr(a), repr(sorted((x, str(y)) for x, y in k.items())))
                    if key not in self.c: 
                        self.c[key] = fn(data, *a, **k)
                    return self.c[key]
                return fn(data, *a, **k)
            return f
            
        torch.tensor = wrap(self.o[0])
        torch.as_tensor = wrap(self.o[1])
        
        self.si = torch.Tensor.__setitem__
        orig = self.si
        
        def setitem(t, idx, value):
            ix = idx if isinstance(idx, tuple) else (idx,)
            if (t.is_cuda and isinstance(value, (int, float, bool)) and all(isinstance(i, (int, slice)) for i in ix)):
                t[idx].fill_(value)
                return
            orig(t, idx, value)
            
        torch.Tensor.__setitem__ = setitem
        return self
        
    def __exit__(self, *e):
        torch.tensor, torch.as_tensor = self.o
        torch.Tensor.__setitem__ = self.si


def graphed(ad, x):
    try:
        with _ConstCache() as cc:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(3): 
                    ad.forward(x)
                    
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            
            g = torch.cuda.CUDAGraph()
            sx = x.clone()
            
            with torch.inference_mode(), torch.cuda.graph(g):
                sy = ad.forward(sx)
                
        keep = (cc.c, sx, sy)                      
        def fwd(xx): 
            keep[1].copy_(xx)
            g.replay()
            return keep[2]
            
        return fwd, "cuda_graph"
        
    except Exception as e:
        import traceback
        root = e
        while root.__context__ is not None: 
            root = root.__context__   
            
        where = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                            for f in reversed(traceback.extract_tb(root.__traceback__)[-6:]))
        print("  [CUDA-graph capture failed -> falling back to eager]", repr(root)[:250], "\n   ROOT CAUSE at", where)
        torch.cuda.synchronize()
        return ad.forward, "eager"


def measure(ad, fwd, tensors, conf, warmup=50, iters=300, repeats=3):
    meds = []
    mmeds = []
    e2e_all = []
    
    with torch.inference_mode():
        for _ in range(repeats):
            e2e = []
            mdl = []
            for i in range(warmup + iters):
                x = tensors[i % len(tensors)]
                a, b, c = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                
                a.record()
                raw = fwd(x)
                b.record()
                ad.post(raw, conf)
                c.record()
                torch.cuda.synchronize()
                
                if i >= warmup: 
                    mdl.append(a.elapsed_time(b))
                    e2e.append(a.elapsed_time(c))
                    
            meds.append(np.median(e2e))
            mmeds.append(np.median(mdl))
            e2e_all += e2e
            
    return dict(med=float(np.median(meds)), rng=float(max(meds) - min(meds)), mean=float(np.mean(e2e_all)),
                p95=float(np.percentile(e2e_all, 95)), mdl=float(np.median(mmeds)))


def latency(ad, tensors, use_graph=True, sensitivity=False):
    fwd, timing = graphed(ad, tensors[0]) if use_graph else (ad.forward, "eager")
    r = measure(ad, fwd, tensors, DEPLOY_CONF)
    
    out = dict(lat_ms=r["med"], lat_range=r["rng"], lat_mean=r["mean"], lat_p95=r["p95"],
               lat_model_ms=r["mdl"], timing=timing)
               
    if sensitivity:    
        out["lat_conf001_ms"] = measure(ad, fwd, tensors, NMS_CONF, repeats=1)["med"]
        if timing == "cuda_graph": 
            out["lat_eager_ms"] = measure(ad, ad.forward, tensors, DEPLOY_CONF, repeats=1)["med"]
            
    return out


def count_flops_g(ad, x):
    from torch.utils.flop_counter import FlopCounterMode
    
    def run_inference():
        with torch.inference_mode(), FlopCounterMode(display=False) as fc: 
            ad.forward(x)
        return fc.get_total_flops()
        
    def run_grad():                      
        # some HF modules break FlopCounterMode's hooks without autograd graphs
        with torch.enable_grad(), FlopCounterMode(display=False) as fc:
            if hasattr(ad, "kind"):
                ad.m(pixel_values=x) 
            else:
                ad.m(x)
        return fc.get_total_flops()
        
    for fn in (run_inference, run_grad):
        try: 
            return fn() / 1e9
        except Exception as e: 
            pass # print(f"  [FLOP count via {fn.__name__} failed]", repr(e)[:160])
            
    return None


def gpu_state():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                     "--format=csv,noheader,nounits"], text=True).strip().split("\n")[0]
        u, m = [float(v) for v in o.split(",")]
        return u, m
    except Exception: 
        return None, None


def gpu_util_idle(n=3, gap=0.3):
    vals = []
    for _ in range(n):
        u, _m = gpu_state()
        if u is not None: 
            vals.append(u)
        time.sleep(gap)
        
    if vals:
        return round(float(np.mean(vals)), 1)
    return None


def write_env(out, gpu, models):
    def sh(cmd):
        try: 
            return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: 
            return None
            
    def ver(pkg):
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception: 
            return None
            
    u, m = gpu_state()
    entry = dict(
        date=datetime.datetime.now().isoformat(timespec="seconds"), 
        models=models, gpu=gpu,
        gpu_mem_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30),
        driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        baseline_gpu_util_pct=u, 
        baseline_gpu_mem_used_mib=m,
        python=platform.python_version(), 
        torch=torch.__version__, 
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), 
        torchvision=torchvision.__version__,
        ultralytics=ver("ultralytics"), 
        transformers=ver("transformers"),
        pycocotools=ver("pycocotools"), 
        numpy=np.__version__
    )
    
    try:
        import ultralytics
        entry["ultralytics_path"] = os.path.dirname(ultralytics.__file__)
    except Exception: 
        pass
        
    p = os.path.join(out, "env.json")
    env = {}
    if os.path.exists(p):
        env = json.load(open(p)) 
        
    env.update(
        protocol=dict(input=SZ, batch=1, precision="FP16", warmup=50, iters=300, repeats=3,
                      nms_accuracy=dict(conf=NMS_CONF, iou=NMS_IOU, max_det=MAX_DET),
                      nms_latency=dict(conf=DEPLOY_CONF, iou=NMS_IOU, max_det=MAX_DET)),
        sources=SOURCES, 
        transformers_disable_torch_check=os.environ.get("TRANSFORMERS_DISABLE_TORCH_CHECK"),
        repos={n: sh(f"git -C ~/{n} rev-parse --short HEAD") for n in ("yolov5", "yolov7", "yolov13")}
    )
    
    env.setdefault("runs", []).append(entry)
    json.dump(env, open(p, "w"), indent=2)
    print(f"  baseline GPU: util={u}%  mem_used={m} MiB (other jobs)  -> {p}")


def accuracy(ad, gt, ids, files, outjson, mode=None):
    dl = torch.utils.data.DataLoader(DS(ids, files, mode or ad.mode, ad.norm), batch_size=None,
                                     num_workers=6, prefetch_factor=4)
    dets = []
    
    with torch.inference_mode():
        for img_id, x, (sx, sy, px, py, w, h) in dl:
            b, s, c = ad.post(ad.forward(x[None].to(DEV).half()))
            b = b.cpu().numpy()
            s = s.cpu().numpy()
            c = c.cpu().numpy()
            
            b[:, [0, 2]] = ((b[:, [0, 2]] - px) / sx).clip(0, w)
            b[:, [1, 3]] = ((b[:, [1, 3]] - py) / sy).clip(0, h)
            
            for bb, ss, cc in zip(b, s, c):
                dets.append(dict(image_id=int(img_id), category_id=int(cc), score=round(float(ss), 5),
                                 bbox=[round(float(bb[0]), 2), round(float(bb[1]), 2),
                                       round(float(bb[2] - bb[0]), 2), round(float(bb[3] - bb[1]), 2)]))
                                       
    json.dump(dets, open(outjson, "w"))
    if not dets: 
        return 0.0, 0.0
        
    E = COCOeval(gt, gt.loadRes(outjson), "bbox")
    E.params.imgIds = ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    return float(E.stats[0]), float(E.stats[1])


def save(csvp, rows):
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="names from registry (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="use first N val images for a quick test")
    ap.add_argument("--latency-only", action="store_true", help="re-time models without running AP again")
    ap.add_argument("--native-ap", action="store_true", help="re-eval DETR at 800/1333")
    ap.add_argument("--sensitivity", action="store_true", help="check how nms conf affects latency")
    ap.add_argument("--no-graph", action="store_true", help="force eager mode timing")
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true")
    
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    
    gt = COCO(ANN)
    ids = sorted(gt.getImgIds())
    if a.limit:
        ids = ids[:a.limit]
        
    files = [gt.imgs[i]["file_name"] for i in ids]
    cat80 = torch.tensor(sorted(gt.getCatIds()), device=DEV)
    reg = registry()
    gpu = torch.cuda.get_device_name(0)
    names = a.models or list(reg)
    
    csvp = os.path.join(a.out, "table10.csv")
    done = {}
    if os.path.exists(csvp):
        done = {r["detector"]: r for r in csv.DictReader(open(csvp))}
        
    if not a.limit: 
        write_env(a.out, gpu, names)
        
    for name in names:
        if (a.latency_only or a.native_ap) and name not in done: 
            print("no AP row yet, skipping", name)
            continue
            
        if a.native_ap and not getattr(reg[name][2](), "native", False): 
            continue
            
        if name in done and not (a.force or a.limit or a.latency_only or a.native_ap): 
            print("skipping", name)
            continue
            
        sl, lin, make = reg[name]
        print(f"\n===== {name}")
        
        ad = make()
        ad.load(cat80)
        
        if a.native_ap:                        
            torch.backends.cudnn.benchmark = False
            ap_n, _ = accuracy(ad, gt, ids, files, os.path.join(a.out, f"dets_{name}_native.json"), mode="native")
            torch.backends.cudnn.benchmark = True
            
            done[name]["ap_native"] = round(ap_n * 100, 1)
            print(name, "AP@native =", done[name]["ap_native"])
            
            if not a.limit: 
                save(csvp, done)
            del ad
            torch.cuda.empty_cache()
            continue
            
        pre = DS(ids[:100], files[:100], ad.mode, ad.norm)
        tens = [pre[i][1][None].to(DEV).half() for i in range(len(pre))]
        
        params = round(sum(p.numel() for p in ad.m.parameters()) / 1e6, 1)
        flops = count_flops_g(ad, tens[0])
        flops = round(flops, 1) if flops else ""
        
        util = gpu_util_idle()
        if util is not None and util > 5: 
            print(f"  [WARNING] GPU is {util}% busy before timing. Latency might be off.")
            
        lat = latency(ad, tens, use_graph=not a.no_graph, sensitivity=a.sensitivity)
        lat = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in lat.items()}
        
        extra = dict(year=RELEASE.get(name, ""), slice=sl, params_M=params, flops_G=flops, gpu_util_pct=util)
        
        if a.latency_only:
            row = {**done[name], **extra, **lat, "gpu": gpu, "precision": "FP16"}
        else:
            ap_, ap50 = accuracy(ad, gt, ids, files, os.path.join(a.out, f"dets_{name}.json"))
            row = dict(detector=name, lineage=lin, input=f"{SZ}x{SZ}", ap=round(ap_ * 100, 1),
                       ap50=round(ap50 * 100, 1), gpu=gpu, precision="FP16", n_images=len(ids), **extra, **lat)
                       
        print(row)
        if not a.limit:
            done[name] = row
            save(csvp, done)
            
        del ad, tens
        torch.cuda.empty_cache()
        
    modes = {r.get("timing", "?") for r in done.values()}
    print("\n===== summary (timing modes in CSV: %s)" % sorted(modes))
    for n, r in done.items():
        print(f"  {n:22s} AP={str(r.get('ap')):>5}  lat={str(r.get('lat_ms')):>6} ms  model={str(r.get('lat_model_ms')):>6}  "
              f"{r.get('timing')}  params={r.get('params_M')}M  flops={r.get('flops_G')}G")
              
    if len(modes) > 1: 
        print("  [WARNING] MIXED timing modes detected in CSV. Might want to re-run with --latency-only")

if __name__ == "__main__":
    main()