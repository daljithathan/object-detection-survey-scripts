import argparse
import copy
import csv
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from docx.text.paragraph import Paragraph

ORDER = ["YOLOv5-S", "YOLOv5-M", "YOLOv5-L", "YOLOv7", "DETR-R50", "Deformable DETR-R50",
         "YOLOv12-S", "YOLOv12-M", "YOLOv12-L", "YOLOv13-S", "YOLOv13-L",
         "RT-DETRv2-S", "RT-DETRv2-M", "RT-DETRv2-L", "D-FINE-L"]

YEAR = {"YOLOv5-S": 2020, "YOLOv5-M": 2020, "YOLOv5-L": 2020, "YOLOv7": 2022, "DETR-R50": 2020,
        "Deformable DETR-R50": 2021, "YOLOv12-S": 2025, "YOLOv12-M": 2025, "YOLOv12-L": 2025,
        "YOLOv13-S": 2025, "YOLOv13-L": 2025, "RT-DETRv2-S": 2024, "RT-DETRv2-M": 2024,
        "RT-DETRv2-L": 2024, "D-FINE-L": 2025}

FAMILIES = [["YOLOv5-S", "YOLOv5-M", "YOLOv5-L"], ["YOLOv12-S", "YOLOv12-M", "YOLOv12-L"],
            ["YOLOv13-S", "YOLOv13-L"], ["RT-DETRv2-S", "RT-DETRv2-M", "RT-DETRv2-L"]]

G1, G2 = "2020–2022", "2024–2025"
COL = {"YOLO": "#1f5fbf", "DETR": "#d9480f"}
MRK = {G1: "o", G2: "s"}

def group(year):
    if year <= 2022:
        return G1
    return G2

def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def load(csvp):
    rows = []
    for r in csv.DictReader(open(csvp)):
        n = r["detector"]
        year = int(num(r.get("year")) or YEAR.get(n, 0))
        rows.append(dict(
            detector=n, lineage=r["lineage"], year=year, group=group(year), ap=num(r["ap"]),
            lat=num(r.get("lat_ms")), rng=num(r.get("lat_range")), params=num(r.get("params_M")),
            flops=num(r.get("flops_G")), timing=(r.get("timing") or "").strip(),
            util=num(r.get("gpu_util_pct")), ap_native=num(r.get("ap_native")),
            lat_eager=num(r.get("lat_eager_ms")), lat_c001=num(r.get("lat_conf001_ms")),
            gpu=r.get("gpu", "")
        ))
    rows.sort(key=lambda x: ORDER.index(x["detector"]) if x["detector"] in ORDER else 999)
    return rows

def validate(rows):
    err = []
    warn = []
    for r in rows:
        if r["lat"] is None:
            err.append(f"{r['detector']}: no latency")
        if r["ap"] is None:
            err.append(f"{r['detector']}: no AP")
        if not r["timing"]:
            err.append(f"{r['detector']}: no 'timing' value -> run  bench.py --latency-only")
            
    modes = {r["timing"] for r in rows if r["timing"]}
    if len(modes) > 1:
        err.append("MIXED timing modes %s (%s) -> numbers are not comparable. Re-run bench.py --latency-only for "
                   "ALL models (or --no-graph for all)." % (sorted(modes),
                   ", ".join(f"{r['detector']}={r['timing']}" for r in rows if r["timing"] != "cuda_graph")))
                   
    if modes == {"eager"}:
        warn.append("all rows are eager-mode: valid but launch-overhead-bound; small models will look similar")
        
    lat = {r["detector"]: r["lat"] for r in rows}
    for fam in FAMILIES:
        v = [lat.get(n) for n in fam if lat.get(n) is not None]
        if len(v) == len(fam) and any(b < a for a, b in zip(v, v[1:])):
            warn.append(f"latency not increasing with size in {fam[0].split('-')[0]}: "
                        + ", ".join(f"{n}={lat[n]}" for n in fam))
                        
    for r in rows:
        if r["util"] is not None and r["util"] > 5:
            warn.append(f"{r['detector']}: GPU {r['util']}% busy before timing")
        if r["rng"] is not None and r["lat"] and r["rng"] / r["lat"] > 0.05:
            warn.append(f"{r['detector']}: repeat spread {r['rng']} ms is >5% of latency {r['lat']} ms")
    return err, warn

def pareto(pts):
    pts = sorted(pts)
    best = -1
    out = []
    for x, ap in pts:
        if ap > best:
            out.append((x, ap))
            best = ap
    return out

def place_labels(ax, fig, pts, key):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    cand = [(7, 5, "left", "bottom"), (7, -5, "left", "top"), (-7, 5, "right", "bottom"), (-7, -5, "right", "top"),
            (0, 9, "center", "bottom"), (0, -9, "center", "top"), (9, 0, "left", "center"), (-9, 0, "right", "center")]
    dots = []
    for r in pts:
        x, y = ax.transData.transform((r[key], r["ap"]))
        dots.append((x - 9, y - 9, x + 9, y + 9))
        
    def hit(b1, b2):
        return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])
        
    placed = []
    for r in sorted(pts, key=lambda q: -q["ap"]):
        best = None
        for dx, dy, ha, va in cand:
            t = ax.annotate(r["detector"], (r[key], r["ap"]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=6.5, ha=ha, va=va,
                            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.45", shrinkA=0, shrinkB=4))
            bb = t.get_window_extent(rend)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            ax_bb = ax.get_window_extent(rend)
            inside = box[0] >= ax_bb.x0 and box[2] <= ax_bb.x1 and box[1] >= ax_bb.y0 and box[3] <= ax_bb.y1
            score = sum(hit(box, o) for o in placed) * 2 + sum(hit(box, d) for d in dots) + (0 if inside else 5)
            
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, t, box)
            else:
                t.remove()
            if score == 0:
                break
        placed.append(best[2])

def make_fig(rows, key, xlabel, png, pdf, draft):
    pts = [r for r in rows if r[key] is not None]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    
    for r in pts:
        ax.scatter(r[key], r["ap"], s=70, marker=MRK[r["group"]], c=COL[r["lineage"]],
                   edgecolors="k", linewidths=0.6, zorder=3)
                   
    for g, ls in ((G1, ":"), (G2, "--")):
        p = pareto([(r[key], r["ap"]) for r in pts if r["group"] == g])
        if len(p) > 1:
            ax.step([x for x, _ in p], [y for _, y in p], where="post", ls=ls, c="gray", lw=1, zorder=1)
            
    h = [plt.Line2D([], [], marker="o", ls="", c=COL["YOLO"], mec="k", label="YOLO lineage"),
         plt.Line2D([], [], marker="o", ls="", c=COL["DETR"], mec="k", label="DETR lineage"),
         plt.Line2D([], [], marker="o", ls="", c="w", mec="k", label=f"Released {G1}"),
         plt.Line2D([], [], marker="s", ls="", c="w", mec="k", label=f"Released {G2}"),
         plt.Line2D([], [], ls=":", c="gray", label=f"Pareto front {G1}"),
         plt.Line2D([], [], ls="--", c="gray", label=f"Pareto front {G2}")]
         
    ax.set_xscale("log")
    xs = [r[key] for r in pts]
    lo, hi = min(xs) * 0.85, max(xs) * 1.25
    ax.set_xlim(lo, hi)
    
    ticks = [t for t in (1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel + " (log scale)")
    ax.set_ylabel("COCO AP@[0.5:0.95] (%)")
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(min(r["ap"] for r in pts) - 1.5, max(r["ap"] for r in pts) + 1.5)
    ax.legend(handles=h, fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    place_labels(ax, fig, pts, key)
    
    if draft:
        ax.text(0.5, 0.5, "DRAFT", transform=ax.transAxes, fontsize=60, color="red", alpha=0.15,
                ha="center", va="center", rotation=25)
                
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)

def put(cell, text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=9):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)

def set_runs(p, parts, size):
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    for text, bold in parts:
        r = p.add_run(text)
        r.bold = bold
        r.font.size = size

def find_table(doc):
    for t in doc.tables:
        hdr = [c.text.strip() for c in t.rows[0].cells]
        if hdr and hdr[0] == "Detector" and any(h.startswith("Latency (ms)") for h in hdr):
            return t
    for t in doc.tables:
        hdr = [c.text.strip() for c in t.rows[0].cells]
        if hdr[:2] == ["Detector", "Time slice"]:
            return t
    raise SystemExit("Table 10 not found in the docx")

def build_table(doc, old, rows):
    has_p = any(r["params"] is not None for r in rows)
    has_f = any(r["flops"] is not None for r in rows)
    cols = [("Detector", 4.3), ("Year", 1.4), ("Lineage", 1.9)]
    
    if has_p: cols.append(("Params (M)", 2.0))
    if has_f: cols.append(("FLOPs (G)", 2.0))
    cols += [("COCO AP@[0.5:0.95] (%)", 3.2), ("Latency (ms)", 2.4)]
    
    t = doc.add_table(rows=1, cols=len(cols))
    try:
        t.style = old.style if old.style is not None else "Table Grid"
    except Exception:
        pass
        
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0]
    
    for i, (name, w) in enumerate(cols):
        put(hdr.cells[i], name, bold=True)
        hdr.cells[i].width = Cm(w)
        shade(hdr.cells[i], "D9D9D9")
        
    trPr = hdr._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    
    for g in (G1, G2):
        grp = [r for r in rows if r["group"] == g]
        if not grp:
            continue
        gr = t.add_row()
        m = gr.cells[0].merge(gr.cells[-1])
        put(m, f"Released {g}", bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT)
        shade(m, "F2F2F2")
        
        for r in grp:
            row = t.add_row().cells
            vals = [r["detector"], str(r["year"]), r["lineage"]]
            if has_p:
                vals.append("–" if r["params"] is None else f"{r['params']:.1f}")
            if has_f:
                vals.append("–" if r["flops"] is None else f"{r['flops']:.1f}")
            vals += [f"{r['ap']:.1f}", f"{r['lat']:.1f}"]
            
            for i, v in enumerate(vals):
                put(row[i], v, align=WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER)
                row[i].width = Cm(cols[i][1])
                
    old._tbl.addprevious(t._tbl)
    old._tbl.getparent().remove(old._tbl)
    return t

def protocol_text(rows, env):
    runs = (env or {}).get("runs", [])
    last = runs[-1] if runs else {}
    gpu = last.get("gpu") or (rows[0]["gpu"] if rows else "GPU")
    if last.get("gpu_mem_gib"):
        gpu += f" ({last['gpu_mem_gib']} GB)"
        
    if rows:
        if rows[0]["timing"] == "cuda_graph":
            mode = "CUDA-graph replay of the PyTorch forward pass"
        elif rows[0]["timing"] == "eager":
            mode = "eager PyTorch"
        else:
            mode = "PyTorch"
    else:
        mode = "PyTorch"

    prot = (env or {}).get("protocol", {})
    nl = prot.get("nms_latency", {})
    na = prot.get("nms_accuracy", {})
    
    rep_list = [r["rng"] for r in rows if r["rng"] is not None]
    rep = max(rep_list) if rep_list else 0
    
    s = (f" COCO val2017 (5,000 images); 640×640 input (letterbox for YOLO models, direct resize for DETR-family "
         f"models); {gpu}; batch size 1; FP16; {mode}; same evaluation and timing script for all models. "
         f"Latency is the median of {prot.get('repeats', 3)}×{prot.get('iters', 300)} timed runs after "
         f"{prot.get('warmup', 50)} warm-up runs (largest spread between repeats {rep:.2f} ms) and covers the model forward "
         f"pass plus post-processing (NMS at confidence {nl.get('conf', 0.25)} and IoU {nl.get('iou', 0.7)} for YOLO models; "
         f"top-k selection for DETR-family models); image decoding, resizing and host-to-device transfer are excluded. "
         f"AP uses NMS at confidence {na.get('conf', 0.001)}, IoU {na.get('iou', 0.7)}, at most {na.get('max_det', 300)} detections.")
    return s

def note_text(rows, env):
    d = {r["detector"]: r for r in rows}
    s = ("Note. Groups are defined by release year (≤2022 vs ≥2024), consistent with Tables 4 and 5. "
         "YOLOv5 and YOLOv7 are the original anchor-based implementations; YOLOv12 and YOLOv13 use the official "
         "checkpoints; DETR-family models use Hugging Face Transformers ports of the official checkpoints, and the "
         "Deformable DETR port uses PyTorch deformable attention rather than the original CUDA kernel. ")
         
    n1 = d.get("DETR-R50")
    n2 = d.get("Deformable DETR-R50")
    if n1 and n2 and n1["ap_native"] is not None and n2["ap_native"] is not None:
        s += (f"DETR-R50 and Deformable DETR-R50 were trained and are normally tested at a shorter side of 800 px; "
              f"at that resolution they reach {n1['ap_native']:.1f} and {n2['ap_native']:.1f} AP, versus "
              f"{n1['ap']:.1f} and {n2['ap']:.1f} AP under the unified 640×640 protocol. ")
              
    s += ("FLOPs are counted for one 640×640 forward pass (convolution and matrix-multiplication operations, "
          "2×MACs); deformable-attention sampling is not counted, so DETR-family values are slight under-estimates. ")
          
    es = [r["lat_eager"] / r["lat"] for r in rows if r["lat_eager"] and r["lat"]]
    if es:
        s += (f"Timing in eager mode instead of CUDA-graph replay changes latency by {min(es):.1f}–{max(es):.1f}×, "
              f"which is why the timing method is fixed for all models. ")
              
    ns = [r["lat_c001"] - r["lat"] for r in rows if r["lat_c001"] and r["lat"] and r["lineage"] == "YOLO"]
    if ns:
        s += (f"Timing NMS at the accuracy threshold (0.001) instead of 0.25 adds {min(ns):.1f}–{max(ns):.1f} ms "
              f"for YOLO models. ")
    return s.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/table10.csv")
    ap.add_argument("--env", default="results/env.json")
    ap.add_argument("--docx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default="fig5")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rewrite-caption", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.env):
        env = json.load(open(a.env))
    else:
        env = None
        
    rows = load(a.csv)
    err, warn = validate(rows)
    
    for w in warn:
        print("WARNING:", w)
    for e in err:
        print("ERROR:  ", e)
        
    if err and not a.force:
        sys.exit("\nNot building: fix the errors above (or use --force for a DRAFT).")
        
    draft = bool(err)
    gpu = rows[0]["gpu"] if rows else "GPU"
    
    make_fig(rows, "lat", f"Latency (ms), batch 1, FP16, {gpu}", f"{a.fig}.png", f"{a.fig}.pdf", draft)
    if any(r["flops"] is not None for r in rows):
        make_fig(rows, "flops", "FLOPs (G), one 640×640 forward pass", f"{a.fig}_flops.png", f"{a.fig}_flops.pdf", draft)

    doc = Document(a.docx)
    old = find_table(doc)
    prot = next((p for p in doc.paragraphs if p.text.startswith("Benchmark protocol:")), None)
    new = build_table(doc, old, rows)
    
    if prot is not None:
        sz = prot.runs[0].font.size if prot.runs and prot.runs[0].font.size else Pt(9.5)
        set_runs(prot, [("Benchmark protocol:", True), (protocol_text(rows, env), False)], sz)
        
    nxt = new._tbl.getnext()
    if nxt is not None and nxt.tag == qn("w:p") and Paragraph(nxt, new._parent).text.startswith("Note."):
        note = Paragraph(nxt, new._parent)
    else:
        el = copy.deepcopy(prot._p) if prot is not None else OxmlElement("w:p")
        new._tbl.addnext(el)
        note = Paragraph(el, new._parent)
        
    sz = note.runs[0].font.size if note.runs and note.runs[0].font.size else Pt(9)
    set_runs(note, [("Note. ", True), (note_text(rows, env)[len("Note. "):], False)], sz)

    cap = next((p for p in doc.paragraphs if p.text.startswith("Fig. 5.")), None)
    pic_par = None
    
    if cap is not None:
        prev = cap._p.getprevious()
        if prev is not None and prev.tag == qn("w:p"):
            pic_par = Paragraph(prev, cap._parent)
            
    if pic_par is None or not (pic_par._p.xpath(".//w:drawing") or "FIGURE 5 PLACEHOLDER" in pic_par.text):
        pic_par = next((p for p in doc.paragraphs if "FIGURE 5 PLACEHOLDER" in p.text), None)
        
    if pic_par is not None:
        rids = [b.get(qn("r:embed")) for b in pic_par._p.xpath(".//a:blip")]
        for r in list(pic_par.runs):
            r._r.getparent().remove(r._r)
        for rid in rids:
            try:
                doc.part.drop_rel(rid)
            except Exception:
                pass
        pic_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic_par.add_run().add_picture(f"{a.fig}.png", width=Inches(6.0))
    else:
        print("WARNING: no Fig. 5 placeholder/picture paragraph found; figure saved only as a file")
        
    if a.rewrite_caption and cap is not None:
        sz = cap.runs[0].font.size if cap.runs else None
        for r in list(cap.runs):
            r._r.getparent().remove(r._r)
        for text, bold in (("Fig. 5.", True), (
                f" COCO AP@[0.5:0.95] versus single-image (batch 1) latency of YOLO- and DETR-lineage detectors, "
                f"grouped by release year ({G1} and {G2}), measured with one script on one {gpu} GPU in FP16 "
                f"(Table 10). Latency covers the model forward pass and post-processing. Dotted and dashed lines "
                f"trace the Pareto front of each group.", False)):
            r = cap.add_run(text)
            r.bold = bold
            if sz:
                r.font.size = sz
                
    for old_note in [p for p in doc.paragraphs if p.text.startswith("Note. The blank")]:
        old_note._p.getparent().remove(old_note._p)
        
    doc.save(a.out)
    
    output_msg = f"\nwrote {a.out}, {a.fig}.png/.pdf"
    if os.path.exists(f"{a.fig}_flops.png"):
        output_msg += f", {a.fig}_flops.png"
    print(output_msg)
    
    if draft:
        print("NOTE: validation errors were overridden with --force -> figure is stamped DRAFT, do not submit")

if __name__ == "__main__":
    main()