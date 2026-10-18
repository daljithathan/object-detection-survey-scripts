#!/usr/bin/env python
"""Build Table 10, Fig. 5 (AP vs latency) and Fig. 6 (AP vs FLOPs) from results_v4/table10.csv  (v4).

usage:
  python make_outputs.py --csv results_v4/table10.csv --report-only
  python make_outputs.py --csv results_v4/table10.csv --env results_v4/env.json \
         --docx Survey_FHHM_Manuscript.docx --out Survey_v4_filled.docx [--rfdetr 640|native] [--rewrite-caption] [--force]

Main groups use RF-DETR at the common 640x640 input by default (--rfdetr native swaps); the other RF-DETR set is
shown as a sensitivity group in Table 10 and left out of the figures. NMS-free YOLO heads are drawn hollow and
linked to their NMS twin. A text report with every derived number used in the paper is always printed.
"""
import argparse, copy, csv, json, os, sys
import numpy as np
from scipy.stats import kendalltau
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from docx.text.paragraph import Paragraph

G1, G2 = "2020–2022", "2024–2026"
COL = {"YOLO": "#1f5fbf", "DETR": "#d9480f"}
MRK = {G1: "o", G2: "s"}
FAM = ["YOLOv5", "YOLOv7", "DETR-R50", "Deformable", "YOLOv10", "YOLOv12", "YOLOv13", "YOLO26",
       "RT-DETRv2", "D-FINE", "RF-DETR"]
HEAD = {"o2m": "O2M + NMS", "o2o": "O2O (NMS-free)", "set": "Set (top-k)"}


def group(y): return G1 if y <= 2022 else G2
def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None
def is_rf_native(n): return n.startswith("RF-DETR") and not n.endswith("@640")
def is_rf_640(n): return n.endswith("@640")
def display(n): return n.replace("@640", "")


def short(n):
    s = n.replace("Deformable DETR-R50", "Def-DETR").replace("DETR-R50", "DETR").replace("RT-DETRv2", "RT2")
    s = s.replace("D-FINE", "DF").replace("RF-DETR", "RF").replace("YOLOv", "v").replace("YOLO26", "26")
    return s.replace(" (NMS-free)", "†").replace(" (NMS)", "").replace("@640", "")


def sort_key(r):
    n = r["detector"]
    fam = next((i for i, f in enumerate(FAM) if n.startswith(f)), 99)
    sc = next((i for i, s in enumerate(("-S", "-M", "-L")) if s in n), 0)
    return (r["group"] != G1, fam, sc, "NMS-free" in n)


def load(p):
    rows = []
    for r in csv.DictReader(open(p)):
        y = int(float(r["year"]))
        rows.append(dict(detector=r["detector"], lineage=r["lineage"], year=y, group=group(y),
                         head=r.get("head", ""), head_default=r.get("head_default", ""), input=r.get("input", ""),
                         ap=num(r["ap"]), ap_native=num(r.get("ap_native")), lat=num(r.get("lat_ms")),
                         ci=num(r.get("lat_ci95")), rng=num(r.get("lat_range")), model=num(r.get("lat_model_ms")),
                         eager=num(r.get("lat_eager_ms")), c001=num(r.get("lat_conf001_ms")),
                         params=num(r.get("params_M")), flops=num(r.get("flops_G")),
                         util=num(r.get("gpu_util_pct")), timing=(r.get("timing") or "").strip(),
                         gpu=r.get("gpu", ""), nrep=r.get("n_repeats", "")))
    return sorted(rows, key=sort_key)


def split(rows, rf):
    drop = is_rf_native if rf == "640" else is_rf_640
    return [r for r in rows if not drop(r["detector"])], [r for r in rows if drop(r["detector"])]


def validate(rows):
    err, warn = [], []
    for r in rows:
        if r["lat"] is None: err.append(f"{r['detector']}: no latency")
        if r["ap"] is None: err.append(f"{r['detector']}: no AP")
        if not r["timing"]: err.append(f"{r['detector']}: no timing mode")
    modes = {r["timing"] for r in rows if r["timing"]}
    if len(modes) > 1: err.append(f"MIXED timing modes {sorted(modes)}")
    for r in rows:
        if r["util"] is not None and r["util"] > 5:
            warn.append(f"{r['detector']}: GPU {r['util']}% busy before timing")
        if r["ci"] is not None and r["lat"] and r["ci"] / r["lat"] > 0.02:
            warn.append(f"{r['detector']}: 95% CI {r['ci']} ms = {100 * r['ci'] / r['lat']:.1f}% of latency")
    return err, warn


def pareto(pts):                       # pts: (x, ap, name); minimise x, maximise ap
    out, best = [], -1
    for x, ap, n in sorted(pts, key=lambda p: (p[0], -p[1])):
        if ap > best: out.append((x, ap, n)); best = ap
    return out


# ----------------------------------------------------------------------------- report
def report(main, sens):
    L = []
    def p(s=""): L.append(s)
    for key, lab in (("lat", "latency"), ("flops", "FLOPs")):
        for g in (G1, G2):
            fr = pareto([(r[key], r["ap"], r["detector"]) for r in main if r["group"] == g and r[key] is not None])
            p(f"Pareto front, {lab}, {g}: " + "; ".join(f"{display(n)} ({x:g}, {a:.1f})" for x, a, n in fr))
    p("\nNMS-free vs NMS head (same weights):")
    for r in main:
        if r["head"] != "o2o": continue
        t = next((q for q in main if q["detector"] == r["detector"].replace("(NMS-free)", "(NMS)")), None)
        if not t: continue
        d = t["lat"] - r["lat"]
        p(f"  {r['detector'].split(' ')[0]:10s} lat {t['lat']:.3f} -> {r['lat']:.3f} ms (-{d:.3f} ms, -{100 * d / t['lat']:.1f}%), "
          f"AP {t['ap']:.1f} -> {r['ap']:.1f} ({r['ap'] - t['ap']:+.1f}); post-proc {t['lat'] - t['model']:.3f} -> "
          f"{r['lat'] - r['model']:.3f} ms; forward {t['model']:.3f} -> {r['model']:.3f} ms")
    nms = [r["lat"] - r["model"] for r in main if r["head"] == "o2m"]
    det = [r["lat"] - r["model"] for r in main if r["head"] == "set"]
    p(f"\nPost-processing time: NMS (O2M YOLO) {min(nms):.2f}-{max(nms):.2f} ms; top-k (DETR) {min(det):.2f}-{max(det):.2f} ms")
    for lin in ("YOLO", "DETR"):
        rr = [r["eager"] / r["lat"] for r in main if r["lineage"] == lin and r["eager"]]
        p(f"Eager / CUDA-graph latency, {lin}: {min(rr):.1f}-{max(rr):.1f}x")
    rt = [r["eager"] / r["lat"] for r in main if r["lineage"] == "DETR" and r["eager"] and r["group"] == G2]
    p(f"Eager / CUDA-graph latency, real-time DETR (2024-2026): {min(rt):.1f}-{max(rt):.1f}x")
    for lab, sub in (("all main rows", main), (G2, [r for r in main if r["group"] == G2])):
        s = [r for r in sub if r["eager"]]
        tau = kendalltau([r["lat"] for r in s], [r["eager"] for r in s])[0]
        p(f"Kendall tau, CUDA-graph vs eager latency ranking, {lab} (n={len(s)}): {tau:.2f}")
    c = [r["c001"] - r["lat"] for r in main if r["head"] == "o2m" and r["c001"]]
    p(f"NMS timed at conf 0.001 instead of 0.25 (O2M YOLO): {min(c):+.2f} to {max(c):+.2f} ms")
    for r in main:
        if r["ap_native"] is not None:
            p(f"{r['detector']}: AP 640 = {r['ap']:.1f}, native 800/1333 = {r['ap_native']:.1f} ({r['ap'] - r['ap_native']:+.1f})")
    for r in sens:
        t = next((q for q in main if display(q["detector"]) == display(r["detector"])), None)
        if t:
            p(f"{display(r['detector'])}: {r['input']} -> {r['ap']:.1f} AP, {r['lat']:.3f} ms; "
              f"{t['input']} -> {t['ap']:.1f} AP, {t['lat']:.3f} ms; FLOPs {r['flops']} -> {t['flops']} G")
    hd = sorted({(r["detector"].split(" ")[0].rsplit("-", 1)[0], r["head_default"]) for r in main if r["head"] == "o2o"})
    p("Default head loaded by Ultralytics: " + ", ".join(f"{a}={b}" for a, b in hd))
    return "\n".join(L)


# ----------------------------------------------------------------------------- figures
LABEL_OFF = {}   # manual label offsets if any still collide, e.g. {"YOLOv12-M": (10, -8, "left", "top")}
XCAP = {"lat": 7.0, "flops": None}   # latency points beyond 7 ms are marked at the right edge with their value


def place_labels(ax, fig, pts, key):
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    base = [(6, 4, "left", "bottom"), (6, -4, "left", "top"), (-6, 4, "right", "bottom"), (-6, -4, "right", "top"),
            (0, 8, "center", "bottom"), (0, -8, "center", "top"), (8, 0, "left", "center"), (-8, 0, "right", "center"),
            (14, 10, "left", "bottom"), (-14, -10, "right", "top"), (14, -10, "left", "top"), (-14, 10, "right", "bottom")]
    dots = []
    for r in pts:
        x, y = ax.transData.transform((r[key], r["ap"])); dots.append((x - 7, y - 7, x + 7, y + 7))
    def hit(a, b): return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
    placed, axb = [], ax.get_window_extent(rend)
    for r in sorted(pts, key=lambda q: -q["ap"]):
        cand = [LABEL_OFF[r["detector"]]] if r["detector"] in LABEL_OFF else base
        best = None
        for dx, dy, ha, va in cand:
            t = ax.annotate(short(r["detector"]), (r[key], r["ap"]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=6.5, ha=ha, va=va,
                            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.45", shrinkA=0, shrinkB=3))
            bb = t.get_window_extent(rend); box = (bb.x0, bb.y0, bb.x1, bb.y1)
            inside = box[0] >= axb.x0 and box[2] <= axb.x1 and box[1] >= axb.y0 and box[3] <= axb.y1
            score = sum(hit(box, o) for o in placed) * 2 + sum(hit(box, d) for d in dots) + (0 if inside else 5)
            if best is None or score < best[0]:
                if best is not None: best[1].remove()
                best = (score, t, box)
            else: t.remove()
            if score == 0: break
        placed.append(best[2])


def make_fig(rows, key, xlabel, base, draft):
    pts = [r for r in rows if r[key] is not None]
    cap = XCAP.get(key)
    vis = [r for r in pts if cap is None or r[key] <= cap]
    off = [r for r in pts if cap is not None and r[key] > cap]
    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=220)
    for r in vis:
        if r["head"] != "o2o": continue
        t = next((q for q in vis if q["detector"] == r["detector"].replace("(NMS-free)", "(NMS)")), None)
        if t: ax.plot([t[key], r[key]], [t["ap"], r["ap"]], c=COL[r["lineage"]], lw=0.8, alpha=0.55, zorder=2)
    for r in vis:
        hollow = r["head"] == "o2o"
        ax.scatter(r[key], r["ap"], s=46, marker=MRK[r["group"]], zorder=3,
                   facecolors="white" if hollow else COL[r["lineage"]],
                   edgecolors=COL[r["lineage"]] if hollow else "k", linewidths=1.1 if hollow else 0.5)
    for g, ls in ((G1, ":"), (G2, "--")):
        fr = pareto([(r[key], r["ap"], r["detector"]) for r in vis if r["group"] == g])
        if len(fr) > 1: ax.step([x for x, _, _ in fr], [a for _, a, _ in fr], where="post", ls=ls, c="gray", lw=1, zorder=1)
    xs = [r[key] for r in vis]; lo = min(xs) * 0.9; hi = cap * 1.04 if off else max(xs) * 1.15
    ax.set_xscale("log"); ax.set_xlim(lo, hi)
    ticks = [t for t in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 40, 60, 80, 100, 150, 200) if lo <= t <= hi]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    for r in off:
        ax.scatter(hi / 1.015, r["ap"], s=46, marker=">", c=COL[r["lineage"]], edgecolors="k", linewidths=0.5,
                   zorder=3, clip_on=False)
        ax.annotate(f"{short(r['detector'])} ({r[key]:.1f} ms) →", (hi / 1.015, r["ap"]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=6.5)
    ax.set_xlabel(xlabel + " (log scale)"); ax.set_ylabel("COCO AP@[0.5:0.95] (%)"); ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(min(r["ap"] for r in pts) - 1.5, max(r["ap"] for r in pts) + 1.5)
    h = [plt.Line2D([], [], marker="o", ls="", c=COL["YOLO"], mec="k", label="YOLO lineage"),
         plt.Line2D([], [], marker="o", ls="", c=COL["DETR"], mec="k", label="DETR lineage"),
         plt.Line2D([], [], marker="o", ls="", c="w", mec="k", label=f"Released {G1}"),
         plt.Line2D([], [], marker="s", ls="", c="w", mec="k", label=f"Released {G2}"),
         plt.Line2D([], [], marker="s", ls="-", c=COL["YOLO"], mfc="w", mec=COL["YOLO"],
                    label="NMS-free head (†), linked to NMS head"),
         plt.Line2D([], [], ls=":", c="gray", label=f"Pareto front {G1}"),
         plt.Line2D([], [], ls="--", c="gray", label=f"Pareto front {G2}")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    place_labels(ax, fig, vis, key)
    if draft: ax.text(0.5, 0.5, "DRAFT", transform=ax.transAxes, fontsize=60, color="red", alpha=0.15,
                      ha="center", va="center", rotation=25)
    fig.savefig(base + ".png"); fig.savefig(base + ".pdf"); plt.close(fig)


# ----------------------------------------------------------------------------- docx
def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), fill); tcPr.append(s)


def put(cell, text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=8):
    cell.text = ""; p = cell.paragraphs[0]; p.alignment = align
    p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text); r.bold, r.italic = bold, italic; r.font.size = Pt(size)


def set_runs(p, parts, size):
    for r in list(p.runs): r._r.getparent().remove(r._r)
    for text, bold in parts:
        r = p.add_run(text); r.bold = bold; r.font.size = size


def find_table(doc):
    for t in doc.tables:
        hdr = [c.text.strip() for c in t.rows[0].cells]
        if hdr and hdr[0] == "Detector" and any(h.startswith("Latency") for h in hdr): return t
    raise SystemExit("Table 10 not found in the docx")


def f(v, d): return "–" if v is None else f"{v:.{d}f}"


def build_table(doc, old, main, sens, rf):
    cols = [("Detector", 3.2), ("Year", 1.0), ("Lineage", 1.3), ("Head", 1.8), ("Input", 1.3), ("Params (M)", 1.3),
            ("FLOPs (G)", 1.3), ("AP (%)", 1.3), ("Latency (ms)", 1.5), ("±95% CI", 1.2), ("Eager (ms)", 1.3)]
    t = doc.add_table(rows=1, cols=len(cols))
    try: t.style = old.style if old.style is not None else "Table Grid"
    except Exception: pass
    t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    hdr = t.rows[0]
    for i, (name, w) in enumerate(cols):
        put(hdr.cells[i], name, bold=True); hdr.cells[i].width = Cm(w); shade(hdr.cells[i], "D9D9D9")
    trPr = hdr._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    other = "native input resolution" if rf == "640" else "640×640 input"
    blocks = [(f"Released {G1}", [r for r in main if r["group"] == G1]),
              (f"Released {G2}", [r for r in main if r["group"] == G2]),
              (f"Sensitivity: RF-DETR at {other} (not in Figs. 5 and 6)", sens)]
    for title, grp in blocks:
        if not grp: continue
        gr = t.add_row(); m = gr.cells[0].merge(gr.cells[-1])
        put(m, title, bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT); shade(m, "F2F2F2")
        for r in grp:
            row = t.add_row().cells
            vals = [display(r["detector"]), str(r["year"]), r["lineage"], HEAD.get(r["head"], r["head"]),
                    r["input"].replace("x", "×"), f(r["params"], 1), f(r["flops"], 1), f(r["ap"], 1),
                    f(r["lat"], 2), f(r["ci"], 3), f(r["eager"], 2)]
            for i, v in enumerate(vals):
                put(row[i], v, align=WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER)
                row[i].width = Cm(cols[i][1])
    old._tbl.addprevious(t._tbl); old._tbl.getparent().remove(old._tbl)
    return t


def protocol_text(rows, env):
    runs = (env or {}).get("runs", []); last = runs[0] if runs else {}
    gpu = (last.get("gpu") or rows[0]["gpu"]) + (f" ({last['gpu_mem_gib']} GB)" if last.get("gpu_mem_gib") else "")
    pr = (env or {}).get("protocol", {}); nl, na = pr.get("nms_latency", {}), pr.get("nms_accuracy", {})
    return (f" COCO val2017 (5,000 images); 640×640 input for every model in the two release-year groups (letterbox for "
            f"YOLO models, direct resize for DETR-family models); one {gpu} GPU; batch size 1; FP16; PyTorch with the "
            f"forward pass replayed as a CUDA graph; the same evaluation and timing script for all models. Latency is "
            f"the median of {pr.get('repeats', 10)} repeats of {pr.get('iters', 300)} timed runs, each after "
            f"{pr.get('warmup', 50)} warm-up runs, with the 95% t-interval of the repeat medians (±95% CI). It covers the "
            f"forward pass and post-processing (NMS at confidence {nl.get('conf', 0.25)} and IoU {nl.get('iou', 0.7)} for "
            f"one-to-many YOLO heads; score filtering of the one-to-one output for NMS-free heads; top-k selection for "
            f"DETR-family models) and excludes image decoding, resizing and host-to-device transfer. Eager is the "
            f"latency of the same model in eager PyTorch. AP uses NMS at confidence {na.get('conf', 0.001)}, IoU "
            f"{na.get('iou', 0.7)} and at most {na.get('max_det', 300)} detections for one-to-many heads and the top "
            f"{na.get('max_det', 300)} outputs otherwise.")


def note_text(main, sens, env):
    ver = next((r.get("ultralytics") for r in (env or {}).get("runs", []) if r.get("ultralytics")), "")
    s = ("Note. Groups are defined by release year, consistent with Tables 4 and 5. YOLOv10 and YOLO26 are each "
         f"evaluated with both heads from the same checkpoint: the one-to-many head with NMS, which Ultralytics {ver} "
         "loads by default, and the one-to-one NMS-free head (†), selected explicitly. ")
    for r in main:
        if r["detector"] == "DETR-R50": a = r["ap_native"]
        if r["detector"] == "Deformable DETR-R50": b = r["ap_native"]
    try:
        s += (f"DETR-R50 and Deformable DETR-R50 are normally tested at a shorter side of 800 pixels, where they reach "
              f"{a:.1f} and {b:.1f} AP. ")
    except NameError: pass
    if sens:
        s += ("RF-DETR is shown at both its native square input and 640×640; only the row set in the 2024–2026 group "
              "enters Figs. 5 and 6. ")
    s += ("FLOPs are counted for one forward pass at the listed input (convolution and matrix-multiplication operations, "
          "2×MACs); deformable-attention sampling is not counted, so values for models that use it are slight "
          "under-estimates. DETR, Deformable DETR, RT-DETRv2 and D-FINE use Hugging Face Transformers ports of the "
          "official checkpoints (the Deformable DETR port uses PyTorch deformable attention, not the original CUDA "
          "kernel); RF-DETR uses the official rfdetr package; YOLOv13 was run from its official repository in a separate "
          "environment.")
    return s


def insert_picture(doc, prefix, png):
    cap = next((p for p in doc.paragraphs if p.text.strip().startswith(prefix)), None)
    pic = None
    if cap is not None:
        prev = cap._p.getprevious()
        if prev is not None and prev.tag == qn("w:p"):
            q = Paragraph(prev, cap._parent)
            if q._p.xpath(".//w:drawing") or "PLACEHOLDER" in q.text: pic = q
    if pic is None:
        pic = next((p for p in doc.paragraphs if f"FIGURE {prefix.split()[1].rstrip('.')} PLACEHOLDER" in p.text), None)
    if pic is None: print(f"WARNING: no picture slot found before '{prefix}' caption; {png} saved as a file only"); return cap
    for rid in [b.get(qn("r:embed")) for b in pic._p.xpath(".//a:blip")]:
        try: doc.part.drop_rel(rid)
        except Exception: pass
    for r in list(pic.runs): r._r.getparent().remove(r._r)
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER; pic.add_run().add_picture(png, width=Inches(6.2))
    return cap


def rewrite_caption(cap, text):
    if cap is None: return
    sz = cap.runs[0].font.size if cap.runs else None
    head, body = text.split(" ", 2)[0] + " " + text.split(" ", 2)[1], " " + text.split(" ", 2)[2]
    for r in list(cap.runs): r._r.getparent().remove(r._r)
    for t, b in ((head, True), (body, False)):
        r = cap.add_run(t); r.bold = b
        if sz: r.font.size = sz


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_v4/table10.csv"); ap.add_argument("--env", default="results_v4/env.json")
    ap.add_argument("--docx"); ap.add_argument("--out"); ap.add_argument("--rfdetr", choices=("640", "native"), default="640")
    ap.add_argument("--report-only", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--rewrite-caption", action="store_true")
    a = ap.parse_args()
    rows = load(a.csv); env = json.load(open(a.env)) if os.path.exists(a.env) else None
    main_rows, sens = split(rows, a.rfdetr)
    err, warn = validate(main_rows)
    for w in warn: print("WARNING:", w)
    for e in err: print("ERROR:  ", e)
    rep = report(main_rows, sens); print("\n===== REPORT\n" + rep)
    open("report_v4.txt", "w").write(rep + "\n")
    if a.report_only: return
    if err and not a.force: sys.exit("\nNot building: fix the errors above (or --force for a DRAFT).")
    if not (a.docx and a.out): sys.exit("--docx and --out are required unless --report-only")
    gpu = rows[0]["gpu"]
    make_fig(main_rows, "lat", f"Latency (ms), batch 1, FP16, {gpu}", "fig5", bool(err))
    make_fig(main_rows, "flops", "FLOPs (G), one forward pass", "fig6", bool(err))

    doc = Document(a.docx)
    old = find_table(doc); new = build_table(doc, old, main_rows, sens, a.rfdetr)
    prot = next((p for p in doc.paragraphs if p.text.startswith("Benchmark protocol:")), None)
    if prot is not None:
        set_runs(prot, [("Benchmark protocol:", True), (protocol_text(rows, env), False)],
                 prot.runs[0].font.size if prot.runs and prot.runs[0].font.size else Pt(9))
    nxt = new._tbl.getnext()
    if nxt is not None and nxt.tag == qn("w:p") and Paragraph(nxt, new._parent).text.startswith("Note"):
        note = Paragraph(nxt, new._parent)
    else:
        el = OxmlElement("w:p"); new._tbl.addnext(el); note = Paragraph(el, new._parent)
    set_runs(note, [("Note. ", True), (note_text(main_rows, sens, env)[len("Note. "):], False)],
             note.runs[0].font.size if note.runs and note.runs[0].font.size else Pt(9))
    c5 = insert_picture(doc, "Fig. 5.", "fig5.png"); c6 = insert_picture(doc, "Fig. 6.", "fig6.png")
    if a.rewrite_caption:
        rewrite_caption(c5, f"Fig. 5. COCO AP@[0.5:0.95] against single-image (batch 1) latency, on a logarithmic axis, "
                            f"for YOLO and DETR detectors grouped by release year ({G1} and {G2}), measured with one script "
                            f"on one {gpu} GPU in FP16 at a common 640×640 input (Table 10). Latency covers the forward pass "
                            f"and post-processing. Hollow markers (†) are NMS-free one-to-one heads, linked to the NMS head "
                            f"of the same checkpoint. Dotted and dashed lines trace the Pareto front of each group."
                            f" Deformable DETR-R50 (12.1 ms) lies beyond the axis and is marked at its right edge.")
        rewrite_caption(c6, f"Fig. 6. COCO AP@[0.5:0.95] against FLOPs of one 640×640 forward pass, on a logarithmic axis, "
                            f"for the detectors of Fig. 5. FLOPs count convolution and matrix-multiplication operations "
                            f"(2×MACs) and exclude deformable-attention sampling. Markers and lines as in Fig. 5.")
    doc.save(a.out)
    print(f"\nwrote {a.out}, fig5.png/.pdf, fig6.png/.pdf, report_v4.txt" + ("  [DRAFT]" if err else ""))


if __name__ == "__main__":
    main()
