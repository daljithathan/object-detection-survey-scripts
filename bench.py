#!/usr/bin/env python
"""Unified benchmark for Table 10 / Figs. 5-6  (v4).
Protocol: COCO val2017, 640x640 input (RF-DETR: its native square resolution), batch 1, FP16, PyTorch,
one script for all models.

v4 additions
  * YOLOv10-S/M/L and YOLO26-S/M/L, each with BOTH heads: one-to-many + NMS ("(NMS)") and one-to-one
    NMS-free ("(NMS-free)"); the Ultralytics default head is recorded in column 'head_default'.
  * D-FINE-S/M (HF ports) to fill the medium scale; RF-DETR-S/M/L via the rfdetr package.
  * --repeats (default 10); 'lat_ci95' = 95% t-interval half-width of the repeat medians.
  * YOLOv13 must run in its own environment (PYTHONPATH=<yolov13 clone>); the script detects which
    ultralytics is imported and only runs the models that belong to that environment.

ACCURACY : all 5000 val images, pycocotools bbox AP. NMS models: conf=0.001 / IoU=0.7 / 300 dets.
           One-to-one (NMS-free) heads and DETR-family models: top-300, no NMS.
LATENCY  : GPU tensor in -> (boxes, scores, classes) out = forward + post-processing.
           Forward replayed as a CUDA graph (falls back to eager; column 'timing' records which).
           NMS models timed at conf=0.25; NMS-free/DETR heads use a score filter / top-k only.
           Excluded: decoding, resize/normalise, host->device copy, un-letterboxing.
           Median of 300 runs after 50 warm-up, repeated --repeats times.
"""
import argparse, csv, datetime, json, os, platform, subprocess, sys, time, types
import cv2, numpy as np, torch, torchvision
from scipy import stats
os.environ.setdefault("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.path.expanduser("~/datasets/coco")
IMG_DIR, ANN = f"{ROOT}/val2017", f"{ROOT}/annotations/instances_val2017.json"
SZ, DEV = 640, "cuda"
NMS_CONF, NMS_IOU, MAX_DET = 0.001, 0.7, 300
DEPLOY_CONF = 0.25
IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
FIELDS = ["detector", "year", "slice", "lineage", "head", "head_default", "input", "ap", "ap50", "ap_native",
          "lat_ms", "lat_range", "lat_ci95", "n_repeats", "lat_mean", "lat_p95", "lat_model_ms", "timing",
          "lat_eager_ms", "lat_conf001_ms", "params_M", "flops_G", "gpu_util_pct", "gpu", "precision", "n_images"]
NATIVE_SHORT, NATIVE_LONG = 800, 1333

# ---------------------------------------------------------------- preprocessing
def letterbox(img, sz):
    h, w = img.shape[:2]; r = min(sz / h, sz / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (sz - nh) // 2, (sz - nw) // 2
    out = np.full((sz, sz, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, [r, r, left, top, w, h]

def stretch(img, sz):
    h, w = img.shape[:2]
    return cv2.resize(img, (sz, sz), interpolation=cv2.INTER_LINEAR), [sz / w, sz / h, 0, 0, w, h]

def native_resize(img, sz):
    """DETR test resolution: shorter side 800 (longer <= 1333), aspect kept. Boxes are normalised, so the
    un-scaling meta is the same as `stretch` at size sz."""
    h, w = img.shape[:2]; r = min(NATIVE_SHORT / min(h, w), NATIVE_LONG / max(h, w))
    return cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_LINEAR), [sz / w, sz / h, 0, 0, w, h]

PRE = {"letterbox": letterbox, "stretch": stretch, "native": native_resize}

class DS(torch.utils.data.Dataset):
    def __init__(self, ids, files, mode, norm, size=SZ):
        self.ids, self.files, self.mode, self.norm, self.size = ids, files, mode, norm, size
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = cv2.imread(os.path.join(IMG_DIR, self.files[i]))
        if img is None: raise FileNotFoundError(os.path.join(IMG_DIR, self.files[i]))
        img, meta = PRE[self.mode](img, self.size)
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1).float() / 255
        if self.norm:
            m, s = self.norm
            x = (x - torch.tensor(m).view(3, 1, 1)) / torch.tensor(s).view(3, 1, 1)
        return self.ids[i], x, meta

# ---------------------------------------------------------------- adapters
def nms_xywh(pred, cat_ids, conf=NMS_CONF):
    """pred (N, 4+nc): xywh in input px + per-class scores."""
    xy, wh, sc = pred[:, :2], pred[:, 2:4], pred[:, 4:]
    xyxy = torch.cat([xy - wh / 2, xy + wh / 2], 1)
    i, j = (sc > conf).nonzero(as_tuple=True)
    b, s = xyxy[i], sc[i, j]
    if s.numel() > 30000:
        s, k = s.topk(30000); b, j = b[k], j[k]
    if s.numel() == 0: return b, s, cat_ids[j]
    keep = torchvision.ops.nms(b + j.to(b.dtype)[:, None] * (b.max() + 1), s, NMS_IOU)[:MAX_DET]
    return b[keep], s[keep], cat_ids[j[keep]]

def topk_sigmoid(logits, boxes_cxcywh, size, k, cat80):
    """DETR-family post-processing: sigmoid scores, top-k over (query, class), boxes normalised cxcywh."""
    p = logits.sigmoid(); s, idx = p.flatten().topk(min(k, p.numel()))
    bx, l = boxes_cxcywh[idx // p.shape[1]], idx % p.shape[1]
    xyxy = torch.cat([bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2], 1) * size
    return xyxy, s, (l if p.shape[1] >= 91 else cat80[l])     # >=91 logits: index == COCO category id

def _stub_utils_plots():
    if "utils.plots" in sys.modules: return
    stub = types.ModuleType("utils.plots")
    stub.color_list = ["#FF0000"] * 20
    stub.plot_one_box = lambda *a, **k: None
    stub.plot_one_box_PIL = lambda *a, **k: None
    stub.Annotator = None
    sys.modules["utils.plots"] = stub


class Ultra:                     # YOLOv10, YOLOv12, YOLO26; YOLOv13 via PYTHONPATH=<yolov13 clone>
    mode, norm, size = "letterbox", None, SZ
    def __init__(self, weights, head="default"):          # head: default | o2m | o2o
        self.w, self.head = os.path.expanduser(weights), head
    def load(self, cat80):
        from ultralytics import YOLO
        net = YOLO(self.w).model
        det = net.model[-1]
        e2e = getattr(det, "end2end", None)
        self.head_default = "o2m" if not e2e else "o2o"    # heads without the switch are one-to-many + NMS
        if self.head in ("o2m", "o2o"):
            if e2e is None:
                raise RuntimeError(f"{self.w}: head {type(det).__name__} has no 'end2end' attribute, cannot "
                                   f"select '{self.head}'. Send this message so the adapter can be adjusted.")
            want = self.head == "o2o"
            det.end2end = want
            try: net.end2end = want
            except Exception: pass
            if want:
                det.max_det = MAX_DET
            elif hasattr(det, "fuse"):
                det.fuse = lambda *a, **k: None             # Detect.fuse() would delete the one-to-many branch
        self.used = "o2o" if getattr(det, "end2end", False) else "o2m"
        self.m = net.fuse().half().to(DEV).eval(); self.cat = cat80
        print(f"  head: default={self.head_default}  used={self.used}")
    @torch.inference_mode()
    def forward(self, x):
        y = self.m(x)
        if isinstance(y, dict): raise RuntimeError(f"unexpected dict output, keys={list(y)}")
        return y[0] if isinstance(y, (list, tuple)) else y
    @torch.inference_mode()
    def post(self, y, conf=NMS_CONF):
        if self.used == "o2o":                              # (1, max_det, 6): x1 y1 x2 y2 score cls
            p = y[0].float()
            if p.shape[-1] != 6: raise RuntimeError(f"one-to-one output has shape {tuple(y.shape)}, expected (*, 6)")
            p = p[p[:, 4] > conf]
            return p[:, :4], p[:, 4], self.cat[p[:, 5].long()]
        return nms_xywh(y[0].T.float(), self.cat, conf)


class Legacy:                    # original YOLOv5 / YOLOv7 repos (objectness head)
    mode, norm, size = "letterbox", None, SZ
    head_default = used = "o2m"
    def __init__(self, repo, weights): self.repo, self.w = os.path.expanduser(repo), weights
    def load(self, cat80):
        for k in [k for k in list(sys.modules) if k.split(".")[0] in ("models", "utils")]:
            del sys.modules[k]
        sys.path[:] = [p for p in sys.path if os.path.basename(p) not in ("yolov5", "yolov7")]
        sys.path.insert(0, self.repo)
        _orig_load = torch.load
        def _safe_load(*a, **k):
            k.setdefault("weights_only", False); return _orig_load(*a, **k)
        torch.load = _safe_load
        try:
            try:
                from models.experimental import attempt_load
            except ImportError as e:
                if any(d in str(e).lower() for d in ("seaborn", "matplotlib", "pandas")):
                    for k in [k for k in list(sys.modules) if k.split(".")[0] in ("models", "utils")]:
                        del sys.modules[k]
                    _stub_utils_plots()
                    from models.experimental import attempt_load
                else: raise
            path = os.path.join(self.repo, self.w)
            if not os.path.exists(path): raise FileNotFoundError(f"checkpoint not found: {path}")
            try: m = attempt_load(path, device=DEV)
            except TypeError: m = attempt_load(path, map_location=DEV)
        finally:
            torch.load = _orig_load
        self.m = m.half().to(DEV).eval(); self.cat = cat80
    @torch.inference_mode()
    def forward(self, x):
        y = self.m(x); return y[0] if isinstance(y, (list, tuple)) else y
    @torch.inference_mode()
    def post(self, y, conf=NMS_CONF):
        p = y[0].float()
        return nms_xywh(torch.cat([p[:, :4], p[:, 5:] * p[:, 4:5]], 1), self.cat, conf)


class HF:                        # DETR, Deformable DETR, RT-DETRv2, D-FINE
    mode, size = "stretch", SZ
    head_default = used = "set"
    def __init__(self, repo, kind, k=100, native=False):
        self.repo, self.kind, self.k, self.native = repo, kind, k, native
    def load(self, cat80):
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        self.m = AutoModelForObjectDetection.from_pretrained(self.repo).half().to(DEV).eval()
        p = AutoImageProcessor.from_pretrained(self.repo)
        self.norm = (list(p.image_mean), list(p.image_std)) if getattr(p, "do_normalize", False) else None
        self.cat = cat80
        core = getattr(self.m, "model", None)
        for n in ("anchors", "valid_mask"):
            t = getattr(core, n, None)
            if isinstance(t, torch.Tensor) and t.device.type == "cpu": setattr(core, n, t.to(DEV))
    @torch.inference_mode()
    def forward(self, x): return self.m(pixel_values=x)
    @torch.inference_mode()
    def post(self, out, conf=None):
        lg, bx = out.logits[0].float(), out.pred_boxes[0].float()
        if self.kind == "softmax":
            s, l = lg.softmax(-1)[:, :-1].max(-1)
            xyxy = torch.cat([bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2], 1) * self.size
            return xyxy, s, (l if lg.shape[1] - 1 >= 91 else self.cat[l])
        return topk_sigmoid(lg, bx, self.size, self.k, self.cat)


class RFDETR:                    # rfdetr package; runs at the model's own square resolution
    mode = "stretch"
    head_default = used = "set"
    use_export = False                                       # set by --rfdetr-export
    # def __init__(self, cls_name, k=300): self.cls, self.k = cls_name, k
    def __init__(self, cls_name, k=300, res=None): self.cls, self.k, self.res = cls_name, k, res

    def load(self, cat80):
        import rfdetr
        C = getattr(rfdetr, self.cls, None)
        if C is None:
            raise RuntimeError(f"rfdetr {getattr(rfdetr, '__version__', '?')} has no {self.cls}; available: "
                               f"{[n for n in dir(rfdetr) if n.startswith('RFDETR')]}")
        # wrap = C()                                           # downloads the official COCO checkpoint
        wrap = C(resolution=self.res) if self.res else C()
        inner = getattr(wrap, "model", None)
        net = getattr(inner, "model", None)
        res = getattr(inner, "resolution", None) or getattr(getattr(wrap, "model_config", None), "resolution", None)
        if not isinstance(net, torch.nn.Module) or not res:
            raise RuntimeError(f"unexpected rfdetr layout: wrap.model={type(inner).__name__}, "
                               f"attrs={[a for a in dir(inner) if not a.startswith('_')][:40]}")
        self.size = int(res)
        self.norm = (list(getattr(wrap, "means", IMNET_MEAN)), list(getattr(wrap, "stds", IMNET_STD)))
        if self.use_export and hasattr(net, "export"): net.export()
        self.m = net.half().to(DEV).eval(); self.cat = cat80
        print(f"  RF-DETR {self.cls}: resolution {self.size}, export={self.use_export}")
    @torch.inference_mode()
    def forward(self, x): return self.m(x)
    @staticmethod
    def _split(out):
        if isinstance(out, dict): lg, bx = out["pred_logits"], out["pred_boxes"]
        else:
            ts = [t for t in out if isinstance(t, torch.Tensor)]
            bx = next(t for t in ts if t.shape[-1] == 4); lg = next(t for t in ts if t.shape[-1] > 4)
        while lg.dim() > 3: lg = lg[-1]
        while bx.dim() > 3: bx = bx[-1]
        return lg, bx
    @torch.inference_mode()
    def post(self, out, conf=None):
        lg, bx = self._split(out)
        return topk_sigmoid(lg[0].float(), bx[0].float(), self.size, self.k, self.cat)


Y1, Y2 = "2020–2022", "2024–2026"
RELEASE = {"YOLOv5-S": 2020, "YOLOv5-M": 2020, "YOLOv5-L": 2020, "YOLOv7": 2022, "DETR-R50": 2020,
           "Deformable DETR-R50": 2021, "YOLOv12-S": 2025, "YOLOv12-M": 2025, "YOLOv12-L": 2025,
           "YOLOv13-S": 2025, "YOLOv13-L": 2025, "RT-DETRv2-S": 2024, "RT-DETRv2-M": 2024,
           "RT-DETRv2-L": 2024, "D-FINE-S": 2025, "D-FINE-M": 2025, "D-FINE-L": 2025,
           "RF-DETR-S": 2026, "RF-DETR-M": 2026, "RF-DETR-L": 2026}
RELEASE.update({
    "RF-DETR-S@640": 2026,
    "RF-DETR-M@640": 2026,
    "RF-DETR-L@640": 2026
})
for _s in "SML":
    for _h in ("(NMS)", "(NMS-free)"):
        RELEASE[f"YOLOv10-{_s} {_h}"] = 2024; RELEASE[f"YOLO26-{_s} {_h}"] = 2026
SOURCES = {
  "YOLOv5-S/M/L": "github.com/ultralytics/yolov5 (original anchor-based; weights v7.0 yolov5{s,m,l}.pt)",
  "YOLOv7": "github.com/WongKinYiu/yolov7 (weights v0.1 yolov7.pt)",
  "YOLOv10-S/M/L": "ultralytics package, yolov10{s,m,l}.pt; both heads evaluated",
  "YOLOv12-S/M/L": "ultralytics package, official yolo12{s,m,l}.pt",
  "YOLOv13-S/L": "github.com/iMoonLab/yolov13, official yolov13{s,l}.pt",
  "YOLO26-S/M/L": "ultralytics package, yolo26{s,m,l}.pt; both heads evaluated",
  "DETR-R50": "HF transformers port of facebook/detr-resnet-50",
  "Deformable DETR-R50": "HF port of SenseTime/deformable-detr (PyTorch deformable attention, not the CUDA kernel)",
  "RT-DETRv2-S/M/L": "HF ports of PekingU/rtdetr_v2_r18vd / r34vd / r50vd",
  "D-FINE-S/M/L": "HF ports of ustc-community/dfine-{small,medium,large}-coco",
  "RF-DETR-S/M/L": "rfdetr package, RFDETRSmall/Medium/Large official COCO checkpoints, native resolution"}

def registry():
    r = {
      "YOLOv5-S": (Y1, "YOLO", lambda: Legacy("~/yolov5", "yolov5s.pt")),
      "YOLOv5-M": (Y1, "YOLO", lambda: Legacy("~/yolov5", "yolov5m.pt")),
      "YOLOv5-L": (Y1, "YOLO", lambda: Legacy("~/yolov5", "yolov5l.pt")),
      "YOLOv7":   (Y1, "YOLO", lambda: Legacy("~/yolov7", "yolov7.pt")),
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
      "D-FINE-S": (Y2, "DETR", lambda: HF("ustc-community/dfine-small-coco", "sigmoid", 300)),
      "D-FINE-M": (Y2, "DETR", lambda: HF("ustc-community/dfine-medium-coco", "sigmoid", 300)),
      "D-FINE-L": (Y2, "DETR", lambda: HF("ustc-community/dfine-large-coco", "sigmoid", 300)),
      "RF-DETR-S": (Y2, "DETR", lambda: RFDETR("RFDETRSmall")),
      "RF-DETR-M": (Y2, "DETR", lambda: RFDETR("RFDETRMedium")),
      "RF-DETR-L": (Y2, "DETR", lambda: RFDETR("RFDETRLarge")),
      "RF-DETR-S@640": (Y2, "DETR", lambda: RFDETR("RFDETRSmall", res=640)),
      "RF-DETR-M@640": (Y2, "DETR", lambda: RFDETR("RFDETRMedium", res=640)),
      "RF-DETR-L@640": (Y2, "DETR", lambda: RFDETR("RFDETRLarge", res=640)),
    }
    for s in "sml":
        S = s.upper()
        for tag, head in (("(NMS)", "o2m"), ("(NMS-free)", "o2o")):
            r[f"YOLOv10-{S} {tag}"] = (Y2, "YOLO", lambda s=s, h=head: Ultra(f"yolov10{s}.pt", h))
            r[f"YOLO26-{S} {tag}"] = (Y2, "YOLO", lambda s=s, h=head: Ultra(f"yolo26{s}.pt", h))
    return r

def uses_ultralytics(n): return n.startswith(("YOLOv10", "YOLOv12", "YOLOv13", "YOLO26"))

# ---------------------------------------------------------------- measurement
class _ConstCache:
    """During warm-up, remember small CPU->CUDA constant tensors (shape tables etc.); during capture, return
    the remembered copy, since host->device copies are illegal while capturing. Keyed by content, so a
    tensor whose values change is never served from the cache."""
    @staticmethod
    def _tkey(tag, t, a, k):
        if t.numel() > 4096: return None
        try: data = t.detach().contiguous().numpy().tobytes()
        except Exception: data = repr(t.detach().tolist())
        return (tag, str(t.dtype), tuple(t.shape), data, repr(a), repr(sorted((x, str(y)) for x, y in k.items())))

    def __enter__(self):
        self.o, self.c = (torch.tensor, torch.as_tensor), {}
        cache = self.c
        def wrap(fn):
            def f(data, *a, **k):
                if "cuda" in str(k.get("device", "")):
                    if isinstance(data, (list, tuple, int, float, bool)):
                        key = (repr(data), repr(a), repr(sorted((x, str(y)) for x, y in k.items())))
                    elif isinstance(data, torch.Tensor) and data.device.type == "cpu":
                        key = _ConstCache._tkey("as", data, a, k)
                    else: key = None
                    if key is not None:
                        if key not in cache: cache[key] = fn(data, *a, **k)
                        return cache[key]
                return fn(data, *a, **k)
            return f
        torch.tensor, torch.as_tensor = wrap(self.o[0]), wrap(self.o[1])

        self.to0, self.cuda0 = torch.Tensor.to, torch.Tensor.cuda
        def mk(orig, tag):
            def g(t, *a, **k):
                if t.device.type != "cpu": return orig(t, *a, **k)
                key = _ConstCache._tkey(tag, t, a, k)
                if key is not None and key in cache: return cache[key]
                out = orig(t, *a, **k)
                if key is not None and isinstance(out, torch.Tensor) and out.is_cuda: cache[key] = out
                return out
            return g
        torch.Tensor.to, torch.Tensor.cuda = mk(self.to0, "to"), mk(self.cuda0, "cuda")

        self.si = torch.Tensor.__setitem__
        orig = self.si
        def setitem(t, idx, value):
            ix = idx if isinstance(idx, tuple) else (idx,)
            if t.is_cuda and isinstance(value, (int, float, bool)) and all(isinstance(i, (int, slice)) for i in ix):
                t[idx].fill_(value); return
            orig(t, idx, value)
        torch.Tensor.__setitem__ = setitem
        return self

    def __exit__(self, *e):
        torch.tensor, torch.as_tensor = self.o
        torch.Tensor.to, torch.Tensor.cuda = self.to0, self.cuda0
        torch.Tensor.__setitem__ = self.si

def graphed(ad, x):
    try:
        with _ConstCache() as cc:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(3): ad.forward(x)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            g, sx = torch.cuda.CUDAGraph(), x.clone()
            with torch.inference_mode(), torch.cuda.graph(g):
                sy = ad.forward(sx)
        keep = (cc.c, sx, sy)
        def fwd(xx): keep[1].copy_(xx); g.replay(); return keep[2]
        return fwd, "cuda_graph"
    except Exception as e:
        import traceback
        root = e
        while root.__context__ is not None: root = root.__context__
        where = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                            for f in reversed(traceback.extract_tb(root.__traceback__)[-6:]))
        print("  [CUDA-graph capture failed -> eager]", repr(root)[:250], "\n   ROOT CAUSE at", where)
        torch.cuda.synchronize(); return ad.forward, "eager"

def measure(ad, fwd, tensors, conf, warmup=50, iters=300, repeats=10):
    meds, mmeds, e2e_all = [], [], []
    with torch.inference_mode():
        for _ in range(repeats):
            e2e, mdl = [], []
            for i in range(warmup + iters):
                x = tensors[i % len(tensors)]
                a, b, c = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                a.record(); raw = fwd(x); b.record(); ad.post(raw, conf); c.record()
                torch.cuda.synchronize()
                if i >= warmup: mdl.append(a.elapsed_time(b)); e2e.append(a.elapsed_time(c))
            meds.append(np.median(e2e)); mmeds.append(np.median(mdl)); e2e_all += e2e
    m = np.asarray(meds)
    ci = float(stats.t.ppf(0.975, len(m) - 1) * m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else None
    return dict(med=float(np.median(m)), rng=float(m.max() - m.min()), ci=ci, mean=float(np.mean(e2e_all)),
                p95=float(np.percentile(e2e_all, 95)), mdl=float(np.median(mmeds)))

def latency(ad, tensors, use_graph=True, sensitivity=False, repeats=10):
    fwd, timing = graphed(ad, tensors[0]) if use_graph else (ad.forward, "eager")
    r = measure(ad, fwd, tensors, DEPLOY_CONF, repeats=repeats)
    out = dict(lat_ms=r["med"], lat_range=r["rng"], lat_ci95=r["ci"], n_repeats=repeats, lat_mean=r["mean"],
               lat_p95=r["p95"], lat_model_ms=r["mdl"], timing=timing)
    if sensitivity:
        out["lat_conf001_ms"] = measure(ad, fwd, tensors, NMS_CONF, repeats=3)["med"]
        if timing == "cuda_graph":
            out["lat_eager_ms"] = measure(ad, ad.forward, tensors, DEPLOY_CONF, repeats=3)["med"]
    return out

def count_flops_g(ad, x):
    from torch.utils.flop_counter import FlopCounterMode
    def run_inference():
        with torch.inference_mode(), FlopCounterMode(display=False) as fc: ad.forward(x)
        return fc.get_total_flops()
    def run_grad():
        with torch.enable_grad(), FlopCounterMode(display=False) as fc:
            ad.m(pixel_values=x) if isinstance(ad, HF) else ad.m(x)
        return fc.get_total_flops()
    for fn in (run_inference, run_grad):
        try: return fn() / 1e9
        except Exception as e: print(f"  [FLOP count via {fn.__name__} failed]", repr(e)[:160])
    return None

def gpu_state():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                     "--format=csv,noheader,nounits"], text=True).strip().split("\n")[0]
        u, m = [float(v) for v in o.split(",")]; return u, m
    except Exception: return None, None

def gpu_util_idle(n=3, gap=0.3):
    vals = []
    for _ in range(n):
        u, _m = gpu_state()
        if u is not None: vals.append(u)
        time.sleep(gap)
    return round(float(np.mean(vals)), 1) if vals else None

def write_env(out, gpu, models, repeats):
    def sh(cmd):
        try: return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: return None
    def ver(pkg):
        try:
            from importlib.metadata import version; return version(pkg)
        except Exception: return None
    u, m = gpu_state()
    entry = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), models=models, gpu=gpu,
                 gpu_mem_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30),
                 driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
                 clocks=sh("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader"),
                 baseline_gpu_util_pct=u, baseline_gpu_mem_used_mib=m,
                 python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version(), torchvision=torchvision.__version__,
                 ultralytics=ver("ultralytics"), transformers=ver("transformers"), rfdetr=ver("rfdetr"),
                 pycocotools=ver("pycocotools"), numpy=np.__version__)
    try:
        import ultralytics; entry["ultralytics_path"] = os.path.dirname(ultralytics.__file__)
    except Exception: pass
    p = os.path.join(out, "env.json")
    env = json.load(open(p)) if os.path.exists(p) else {}
    env.update(protocol=dict(input=SZ, batch=1, precision="FP16", warmup=50, iters=300, repeats=repeats,
                             nms_accuracy=dict(conf=NMS_CONF, iou=NMS_IOU, max_det=MAX_DET),
                             nms_latency=dict(conf=DEPLOY_CONF, iou=NMS_IOU, max_det=MAX_DET)),
               sources=SOURCES, transformers_disable_torch_check=os.environ.get("TRANSFORMERS_DISABLE_TORCH_CHECK"),
               repos={n: sh(f"git -C ~/{n} rev-parse --short HEAD") for n in ("yolov5", "yolov7", "yolov13")})
    env.setdefault("runs", []).append(entry)
    json.dump(env, open(p, "w"), indent=2)
    print(f"  baseline GPU: util={u}%  mem_used={m} MiB  -> {p}")

def accuracy(ad, gt, ids, files, outjson, mode=None):
    dl = torch.utils.data.DataLoader(DS(ids, files, mode or ad.mode, ad.norm, ad.size), batch_size=None,
                                     num_workers=6, prefetch_factor=4)
    dets = []
    with torch.inference_mode():
        for img_id, x, (sx, sy, px, py, w, h) in dl:
            b, s, c = ad.post(ad.forward(x[None].to(DEV).half()))
            b, s, c = b.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy()
            b[:, [0, 2]] = ((b[:, [0, 2]] - px) / sx).clip(0, w)
            b[:, [1, 3]] = ((b[:, [1, 3]] - py) / sy).clip(0, h)
            for bb, ss, cc in zip(b, s, c):
                dets.append(dict(image_id=int(img_id), category_id=int(cc), score=round(float(ss), 5),
                                 bbox=[round(float(bb[0]), 2), round(float(bb[1]), 2),
                                       round(float(bb[2] - bb[0]), 2), round(float(bb[3] - bb[1]), 2)]))
    json.dump(dets, open(outjson, "w"))
    if not dets: return 0.0, 0.0
    E = COCOeval(gt, gt.loadRes(outjson), "bbox"); E.params.imgIds = ids
    E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0]), float(E.stats[1])

def save(csvp, rows):
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        w.writeheader(); w.writerows(rows.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="names from registry (default: all models of this environment)")
    ap.add_argument("--list", action="store_true", help="print registry names and exit")
    ap.add_argument("--limit", type=int, default=0, help="first N val images (smoke test, no CSV write)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--latency-only", action="store_true")
    ap.add_argument("--native-ap", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--rfdetr-export", action="store_true", help="call rfdetr's export() (use if graph capture fails)")
    ap.add_argument("--out", default="results_v4"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    reg = registry()
    if a.list: print("\n".join(reg)); return
    RFDETR.use_export = a.rfdetr_export
    os.makedirs(a.out, exist_ok=True)

    import ultralytics
    v13_env = "yolov13" in os.path.dirname(ultralytics.__file__)
    print(f"ultralytics from {os.path.dirname(ultralytics.__file__)}  -> {'YOLOv13' if v13_env else 'standard'} environment")
    names = a.models or [n for n in reg if (n.startswith("YOLOv13") == v13_env)
                         and (v13_env or True) and (not v13_env or uses_ultralytics(n))]
    for n in names:
        if n not in reg: sys.exit(f"unknown model '{n}'. Use --list.")
        if uses_ultralytics(n) and n.startswith("YOLOv13") != v13_env:
            sys.exit(f"'{n}' needs the {'YOLOv13' if n.startswith('YOLOv13') else 'standard'} ultralytics; "
                     f"see the run instructions (PYTHONPATH=~/yolov13 only for YOLOv13).")

    torch.backends.cudnn.benchmark = True
    gt = COCO(ANN); ids = sorted(gt.getImgIds()); ids = ids[:a.limit] if a.limit else ids
    files = [gt.imgs[i]["file_name"] for i in ids]
    cat80 = torch.tensor(sorted(gt.getCatIds()), device=DEV)
    gpu = torch.cuda.get_device_name(0)
    csvp = os.path.join(a.out, "table10.csv")
    done = {r["detector"]: r for r in csv.DictReader(open(csvp))} if os.path.exists(csvp) else {}
    if not a.limit: write_env(a.out, gpu, names, a.repeats)
    for name in names:
        if (a.latency_only or a.native_ap) and name not in done: print("no AP row yet, skip", name); continue
        if a.native_ap and not getattr(reg[name][2](), "native", False): continue
        if name in done and not (a.force or a.limit or a.latency_only or a.native_ap): print("skip", name); continue
        sl, lin, make = reg[name]; print(f"\n===== {name}")
        try:
            ad = make(); ad.load(cat80)
        except Exception as e:
            print(f"  [LOAD FAILED] {name}: {e!r}"); continue
        if a.native_ap:
            torch.backends.cudnn.benchmark = False
            ap_n, _ = accuracy(ad, gt, ids, files, os.path.join(a.out, f"dets_{name}_native.json"), mode="native")
            torch.backends.cudnn.benchmark = True
            done[name]["ap_native"] = round(ap_n * 100, 1); print(name, "AP@native =", done[name]["ap_native"])
            if not a.limit: save(csvp, done)
            del ad; torch.cuda.empty_cache(); continue
        pre = DS(ids[:100], files[:100], ad.mode, ad.norm, ad.size)
        tens = [pre[i][1][None].to(DEV).half() for i in range(len(pre))]
        params = round(sum(p.numel() for p in ad.m.parameters()) / 1e6, 1)
        flops = count_flops_g(ad, tens[0]); flops = round(flops, 1) if flops else ""
        util = gpu_util_idle()
        if util is not None and util > 5: print(f"  [WARNING] GPU already {util}% busy before timing")
        lat = latency(ad, tens, use_graph=not a.no_graph, sensitivity=a.sensitivity, repeats=a.repeats)
        lat = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in lat.items()}
        extra = dict(year=RELEASE.get(name, ""), slice=sl, params_M=params, flops_G=flops, gpu_util_pct=util,
                     head=ad.used, head_default=ad.head_default, input=f"{ad.size}x{ad.size}")
        if a.latency_only:
            row = {**done[name], **extra, **lat, "gpu": gpu, "precision": "FP16"}
        else:
            ap_, ap50 = accuracy(ad, gt, ids, files, os.path.join(a.out, f"dets_{name}.json"))
            if ap_ < 0.10: print(f"  [WARNING] {name}: AP {ap_:.3f} is implausibly low -> head/output format problem")
            row = dict(detector=name, lineage=lin, ap=round(ap_ * 100, 1), ap50=round(ap50 * 100, 1), gpu=gpu,
                       precision="FP16", n_images=len(ids), **extra, **lat)
        print(row)
        if not a.limit:
            done[name] = row; save(csvp, done)
        del ad, tens; torch.cuda.empty_cache()
    modes = {r.get("timing", "?") for r in done.values()}
    print("\n===== summary (timing modes in CSV: %s)" % sorted(modes))
    for n, r in done.items():
        print(f"  {n:24s} AP={str(r.get('ap')):>5}  lat={str(r.get('lat_ms')):>6} ±{str(r.get('lat_ci95')):>5} ms  "
              f"{r.get('timing'):10s} head={r.get('head')}/{r.get('head_default')}  in={r.get('input')}")
    if len(modes) > 1: print("  [WARNING] MIXED timing modes: fix before building Table 10")

if __name__ == "__main__":
    main()
