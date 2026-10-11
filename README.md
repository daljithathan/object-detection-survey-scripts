# COCO Benchmark for YOLO and DETR Detectors

I used these scripts to evaluate 15 object detectors from the YOLO and DETR families on COCO val2017. The results are used in Table 10 and Figs. 5 and 6 of our survey paper, **From Histograms to Hungarian Matching: A Critical Survey of the Evolution of Object Detection, 2001–2026**.

**Authors:** Daljit Singh, Ajay Pal Singh Chauhan and Alka Singla

The repository contains the benchmark script, the results used for the paper and a record of the software environment and evaluation settings. COCO images, annotations and model weights are not included.

## Files

| File | What it does |
| --- | --- |
| `bench.py` | Evaluates COCO AP and records latency, parameter count and FLOPs for each model. |
| `make_outputs.py` | Builds the benchmark plots and updates Table 10 and the benchmark text in a Word manuscript. |
| `requirements.txt` | Lists the Python packages and versions used for the recorded runs. |
| `results/table10.csv` | Results for the 15 evaluated models. |
| `results/env.json` | Hardware, software versions, repository revisions and benchmark settings recorded during the runs. |

## Environment used

The recorded runs used:

- GPU: NVIDIA H100 NVL (93 GiB memory)
- Python: 3.14.4
- PyTorch: 2.14.1+cu130
- CUDA: 13.0
- torchvision: 0.29.1+cu130
- Ultralytics: 8.4.174
- Transformers: 5.19.0
- pycocotools: 2.0.11
- NumPy: 2.5.2

Install the listed packages in a suitable Python environment:

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

## Data and model files

The script expects COCO val2017 in the following locations under your home directory:

```text
~/datasets/coco/val2017/
~/datasets/coco/annotations/instances_val2017.json
```

The legacy YOLO repositories and weights are expected at these paths:

```text
~/yolov5/    # commit f9578796; yolov5s.pt, yolov5m.pt, yolov5l.pt from release v7.0
~/yolov7/    # commit a207844; yolov7.pt from release v0.1
~/yolov13/   # commit 7328994; yolov13s.pt and yolov13l.pt
```

YOLOv12 weights (`yolo12s.pt`, `yolo12m.pt` and `yolo12l.pt`) are obtained through the Ultralytics package. DETR, Deformable DETR, RT-DETRv2 and D-FINE are loaded through Hugging Face Transformers. The YOLOv13 runs use the `ultralytics` package from the local `~/yolov13` checkout.

The YOLOv5 and YOLOv7 checkpoint files are older pickle-based weights. Use the official repositories and weights and only load checkpoints you trust.

## Running the benchmark

Run all listed models with the additional eager-mode and NMS-threshold timing checks:

```bash
python bench.py --sensitivity
```

Run YOLOv13-S and YOLOv13-L with the local YOLOv13 package on the Python import path:

```bash
PYTHONPATH="$HOME/yolov13" python bench.py --models YOLOv13-S YOLOv13-L --sensitivity
```

Run DETR and Deformable DETR at their native resolution (shorter side 800 pixels):

```bash
python bench.py --native-ap --models "DETR-R50" "Deformable DETR-R50"
```

Run a quick check on 50 images with YOLOv12-S:

```bash
python bench.py --limit 50 --models YOLOv12-S
```

Build the plots and insert the updated benchmark table and text into a copy of your Word manuscript. Replace `manuscript.docx` with your source file name if it is different:

```bash
python make_outputs.py --csv results/table10.csv --env results/env.json --docx manuscript.docx --out manuscript_with_benchmark_results.docx
```

The script writes the updated Word file and plot files in the working directory. Review the generated table, plots, captions and benchmark description before using them in a manuscript.

## Benchmark protocol

The recorded protocol is:

- COCO val2017 (5,000 images)
- 640 × 640 input; letterbox resizing for YOLO and direct resizing for DETR models
- Batch size 1, FP16 precision
- CUDA graph replay for the reported latency measurements
- 50 warm-up runs, then 300 runs per repeat, with three repeats
- Latency includes the model forward pass and post-processing; timing was recorded on the first 100 images
- NMS settings for AP: confidence 0.001, IoU 0.7, maximum 300 detections
- NMS settings for latency: confidence 0.25, IoU 0.7, maximum 300 detections

The detailed environment and run records are in `results/env.json`.

## Limitations

- The comparison was run on one GPU. Other jobs shared the GPU during some timing runs.
- TensorRT and edge-device performance were not measured.
- The Deformable DETR implementation uses PyTorch deformable attention rather than the original CUDA kernel, so its measured latency is an upper bound for that implementation.
- YOLOv10, YOLO26 and RF-DETR are not included.

These results describe the tested models under this setup; they should not be read as a universal ranking across hardware or deployment environments.

## Citation

If you use the scripts or the recorded results, please cite this repository:

> Singh, D. (2026). *object-detection-survey-scripts*. GitHub. https://github.com/daljithathan/object-detection-survey-scripts

Associated manuscript: Daljit Singh, Ajay Pal Singh Chauhan and Alka Singla, *From Histograms to Hungarian Matching: A Critical Survey of the Evolution of Object Detection, 2001–2026* (manuscript in preparation). The final journal citation will be added here when available.

## License

The code in this repository is released under the MIT License; see `LICENSE`. COCO, third-party repositories, pretrained weights and external packages remain subject to their own licenses and terms.

## Contact

<a href="https://daljithathan.github.io/">Daljit Singh</a> — daljithathan@gmail.com
