# Unified COCO Benchmark of YOLO and DETR Detectors

This repository contains the scripts, results and environment record for the unified benchmark reported in:

**From Histograms to Hungarian Matching: A Component-Ledger Survey of Object Detection and the YOLO–DETR Convergence**
Daljit Singh, Ajay Pal Singh Chauhan and Alka Singla (manuscript submitted for publication)

The benchmark evaluates **26 official checkpoints in 32 configurations** from the YOLO and DETR lineages on COCO val2017 under one protocol on one GPU. YOLOv10 and YOLO26 are each evaluated with both heads of the same checkpoint: the one-to-many head with NMS and the one-to-one NMS-free head. The results appear in Table 9 and Figs. 5 and 6 of the article and in Supplementary Section S6.

COCO images, annotations and model weights are not included.

## Contents

| File | Purpose |
| --- | --- |
| `bench.py` | Evaluates COCO AP and measures latency, parameters and FLOPs for every model in the registry; records the environment. |
| `make_outputs.py` | Builds Figs. 5 and 6, rebuilds Table 9 in a Word manuscript, and writes `report_v4.txt` with all derived statistics used in the text. |
| `requirements.txt` | Python packages and versions used for the recorded runs. |
| `results_v4/table10.csv` | Final results: one row per configuration, including the RF-DETR resolution-sensitivity rows. |
| `results_v4/env.json` | Hardware, driver, clocks, package versions, repository commits, checkpoint sources and protocol settings for every run. |
| `results_v4/report_v4.txt` | Pareto fronts, NMS/NMS-free decomposition, eager-mode ratios, Kendall's τ and resolution effects. |
| `results_v4/logs/` | Console logs of the final runs. |

## Models

| Lineage | Models | Source |
| --- | --- | --- |
| YOLO, 2020–2022 | YOLOv5-S/M/L, YOLOv7 | Original repositories (anchor-based code) |
| DETR, 2020–2021 | DETR-R50, Deformable DETR-R50 | Hugging Face Transformers ports |
| YOLO, 2024–2026 | YOLOv10-S/M/L (NMS and NMS-free), YOLOv12-S/M/L, YOLOv13-S/L, YOLO26-S/M/L (NMS and NMS-free) | Ultralytics package; YOLOv13 from its official repository |
| DETR, 2024–2026 | RT-DETRv2-S/M/L, D-FINE-S/M/L | Hugging Face Transformers ports |
| DETR, 2026 | RF-DETR-S/M/L at 640×640, plus native 512/576/704 as sensitivity rows | `rfdetr` package |

Run `python bench.py --list` to print all registry names.

## Environment used

- GPU: NVIDIA H100 NVL (94 GB), driver 595.91.07
- Python 3.14.4; PyTorch 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24); torchvision 0.29.1
- Ultralytics 8.4.174; Transformers 5.19.0; rfdetr 1.11.2; pycocotools 2.0.11; NumPy 2.5.2

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

## Data and external repositories

COCO val2017 is expected at:

```text
~/datasets/coco/val2017/
~/datasets/coco/annotations/instances_val2017.json
```

Legacy and external repositories are expected at:

```text
~/yolov5/    # commit f9578796; yolov5s.pt, yolov5m.pt, yolov5l.pt (release v7.0)
~/yolov7/    # commit a207844; yolov7.pt (release v0.1)
~/yolov13/   # commit 7328994; yolov13s.pt, yolov13l.pt
```

YOLOv10, YOLOv12 and YOLO26 weights are downloaded by Ultralytics; DETR, Deformable DETR, RT-DETRv2 and D-FINE by Hugging Face Transformers; RF-DETR by the `rfdetr` package. The YOLOv5 and YOLOv7 checkpoints are pickle-based; load only official weights you trust.

YOLOv13 needs the modified `ultralytics` package shipped with its repository, which does not contain YOLO26. `bench.py` detects which `ultralytics` is imported and only runs the models that belong to that environment.

## Reproducing the results

Run these with no other process on the GPU (`nvidia-smi` should list none).

```bash
# 1. All models except YOLOv13 (about 2-3 hours)
python bench.py --out results_v4 --repeats 10 --sensitivity

# 2. YOLOv13 in its own environment
PYTHONPATH="$HOME/yolov13" python bench.py --out results_v4 --repeats 10 --sensitivity

# 3. RF-DETR at the common 640x640 input
python bench.py --out results_v4 --repeats 10 --sensitivity \
  --models RF-DETR-S@640 RF-DETR-M@640 RF-DETR-L@640

# 4. DETR and Deformable DETR AP at their native resolution (shorter side 800, longer side <= 1333)
python bench.py --out results_v4 --native-ap --models DETR-R50 "Deformable DETR-R50"

# Quick check on 200 images (writes nothing to the CSV)
python bench.py --limit 200 --repeats 3 --models "YOLO26-S (NMS-free)" RF-DETR-M
```

Each model is saved to the CSV as soon as it finishes, so an interrupted run can be restarted; completed models are skipped unless `--force` is given. `--latency-only` re-times models already in the CSV and keeps their AP.

Build the figures, the derived statistics and the manuscript table:

```bash
# Statistics only (writes report_v4.txt)
python make_outputs.py --csv results_v4/table10.csv --report-only

# Figures plus Table 9 inserted into a copy of the manuscript
python make_outputs.py --csv results_v4/table10.csv --env results_v4/env.json \
  --docx manuscript.docx --out manuscript_with_benchmark.docx --rewrite-caption
```

`make_outputs.py` refuses to build if timing modes are mixed and warns when a row was timed on a busy GPU or has a wide confidence interval. It never overwrites the input document.

## Benchmark protocol

- **Data:** COCO val2017, all 5,000 images, pycocotools bbox AP.
- **Input:** 640×640 for every model in Figs. 5 and 6; letterbox for YOLO models, direct resize for DETR-family models. RF-DETR is also reported at its native inputs as a sensitivity check.
- **Precision and batch:** batch size 1, FP16.
- **Accuracy:** for one-to-many YOLO heads, NMS at confidence 0.001, IoU 0.7, max 300 detections. For one-to-one heads, the top 300 outputs. For DETR-family models, top-k query–class pairs.
- **Latency scope:** GPU tensor in to boxes, scores and classes out (forward pass plus post-processing). Image decoding, resizing and host-to-device transfer are excluded.
- **Execution:** the forward pass is replayed as a CUDA graph. Eager-mode latency is recorded separately as a sensitivity measurement.
- **Timing schedule:** 100 pre-loaded images, 50 warm-up and 300 timed runs per repeat, 10 repeats. Reported latency is the median of repeat medians with a 95% t-interval.
- **NMS threshold for timing:** confidence 0.25, IoU 0.7. Timing at confidence 0.001 is recorded as a sensitivity measurement.
- **Head selection:** for YOLOv10 and YOLO26, the head is selected through the Ultralytics end-to-end switch before layer fusion, and the default head is recorded in the `head_default` column. Both releases load the NMS head by default in Ultralytics 8.4.174.
- **FLOPs:** counted for one forward pass at the evaluated input (convolution and matrix multiplication, 2×MACs). Deformable-attention sampling is not counted.

## Notes on the recorded runs

- All reported rows were timed with no more than 3.3% prior GPU utilization.
- The YOLOv13 and RF-DETR rows were re-timed in a later session after contention from an unrelated process was detected during the first pass. Their AP is unchanged, and the re-timed native RF-DETR rows match the first measurement within 0.5%.
- Earlier pilot measurements differed from the final session by up to 13%. Only the final session (`results_v4`) is used in the article.

## Limitations

- One high-end GPU, PyTorch with CUDA-graph replay, and one dataset. TensorRT and edge devices were not measured, and latency rankings may differ on compute-bound hardware.
- DETR, Deformable DETR, RT-DETRv2 and D-FINE are Hugging Face ports. The Deformable DETR port uses PyTorch deformable attention, not the original CUDA kernel, so its latency is an upper bound.
- RF-DETR uses a different backbone and pretraining from the other models; input resolution is controlled, pretraining is not.
- Confidence intervals describe precision within one measurement session.

The results describe these models under this setup and should not be read as a universal ranking.

## Citation

If you use the scripts or results, please cite the article and this repository:

> D. Singh, A. P. S. Chauhan and A. Singla, From Histograms to Hungarian Matching: A Component-Ledger Survey of Object Detection and the YOLO–DETR Convergence, manuscript submitted for publication, 2026.
>
> D. Singh, *object-detection-survey-scripts*, GitHub, 2026. https://github.com/daljithathan/object-detection-survey-scripts

## License

The code is released under the MIT License (see `LICENSE`). COCO, third-party repositories, pretrained weights and external packages remain subject to their own licenses and terms.

## Contact

[Daljit Singh](https://daljithathan.github.io/) – daljithathan@gmail.com
