# Auto YOLO — Dual-Engine Industrial Defect Detection

<p align="center">
  <a href="https://arxiv.org/abs/2602.23013">SubspaceAD (CVPR 2026)</a> •
  <a href="https://github.com/ultralytics/ultralytics">YOLOv8</a> •
  <a href="https://karpathy.github.io/2025/03/17/autoresearch/">AutoResearch</a>
</p>

---

## Overview

This project implements a **dual-engine industrial defect detection system** that combines:

1. **YOLOv8** (supervised) — trained on the NEU-DET steel defect dataset for detecting 6 known defect classes
2. **SubspaceAD** (self-supervised) — training-free anomaly detection using PCA subspace modeling on DINOv2 features

The two engines are combined via **OR fusion**: a defect is reported if either engine detects an anomaly. This captures both known defect types (via YOLO) and unknown/unseen anomalies (via SubspaceAD cross-domain detection).

---

## Project Structure

```
auto_yolo/
├── yolo-autoresearch-v2/     # YOLO training + dual-engine inference
│   ├── data/                 # NEU-DET dataset (train/val splits)
│   ├── models/               # Trained YOLO weights and PCA models
│   ├── runs/detect/          # YOLO training outputs
│   ├── NEU-DET/              # Raw NEU-DET dataset
│   ├── src/                  # Custom assigner, loss, search space
│   ├── inference.py          # Phase 3 dual-engine inference
│   ├── train.py              # YOLO training script (agent-editable)
│   └── prepare_dataset.py    # Dataset preparation
├── SubspaceAD/              # SubspaceAD official implementation
│   ├── main.py               # SubspaceAD evaluation
│   ├── src/subspacead/       # Core: PCA, feature extraction, patching
│   ├── datasets/             # MVTec-AD dataset
│   └── results_full/         # Benchmark results
├── autoresearch/            # AutoResearch framework (karpathy style)
│   ├── train.py              # Agent-editable training script
│   ├── prepare.py            # Data preparation
│   └── program.md            # Agent instructions
├── dinov2-base/             # DINOv2-base model cache
└── models_cache/            # Cached models
```

---

## Installation

### YOLO Environment

```bash
pip install ultralytics torch torchvision
```

### SubspaceAD Environment

```bash
cd SubspaceAD
conda create -n subspacead python=3.10
conda activate subspacead
pip install -r requirements.txt
pip install -e .
```

---

## Phase 1: SubspaceAD Benchmark (MVTec-AD)

Validate SubspaceAD on the standard MVTec-AD benchmark using DINOv2-base.

```bash
cd SubspaceAD
./run_full_benchmark.sh
```

**Results — MVTec-AD (5 categories, dinov2-base, mean aggregation):**

| Category   | Image AUROC | Image AUPR | Pixel AUROC | AU-PRO |
|------------|-------------|------------|-------------|--------|
| screw      | 0.8768      | 0.9373     | 0.9846      | 0.9417 |
| tile       | 1.0000      | 1.0000     | 0.9591      | 0.9149 |
| toothbrush | 0.9639      | 0.9844     | 0.9839      | 0.9561 |
| transistor | 0.9408      | 0.9105     | 0.8838      | 0.6009 |
| wood       | 0.9947      | 0.9984     | 0.9409      | 0.9432 |
| zipper     | 0.9819      | 0.9949     | 0.9742      | 0.9238 |
| **Average**| **0.9597**  | **0.9709** | **0.9544**  | **0.8801** |

---

## Phase 2: YOLO Training (NEU-DET)

Train YOLOv8n on the NEU-DET steel surface defect dataset.

```bash
cd yolo-autoresearch-v2
python train.py --data neu-det.yaml --epochs 50 --imgsz 200
```

**Dataset:** NEU-DET (6 defect classes)
- Train: 1729 images
- Val: 649 images

| Metric           | Value  |
|------------------|--------|
| Precision        | 0.854  |
| Recall           | 0.768  |
| **mAP@0.5**      | **0.862** |
| mAP@0.5:0.95    | 0.513  |
| YOLO Recall (val)| 99.1% (643/649) |

### NEU-DET Defect Classes

| ID | Class Name      | Description                    |
|----|-----------------|--------------------------------|
| 0  | crazing         | Cracking                       |
| 1  | inclusion       | Non-metallic inclusions        |
| 2  | patches         | Surface patches                |
| 3  | pitted_surface | Pitting corrosion              |
| 4  | rolled-in_scale | Rolled oxide scale             |
| 5  | scratches       | Surface scratches              |

---

## Phase 3: Dual-Engine Fusion

Combine YOLO (known defect detection) with SubspaceAD (unknown anomaly scoring).

```bash
cd yolo-autoresearch-v2
python inference.py
```

### Fusion Logic

```
is_defect = (YOLO detects defect) OR (SubspaceAD anomaly_score > threshold)
```

### OR Fusion Results (NEU-DET val, 649 images):

| Engine           | Recall          | Notes                              |
|------------------|-----------------|------------------------------------|
| YOLO alone       | 99.1% (643/649) | Supervised, NEU-DET domain        |
| SubspaceAD alone | ~50% (325/649)  | Cross-domain, fitted on MVTec-AD  |
| **OR Fusion**    | **99.5% (646/649)** | YOLO + SubspaceAD union       |

**YOLO missed 6 images; SubspaceAD recovered 3 of them.**

---

## How It Works

### SubspaceAD Algorithm

1. **Feature Extraction:** Extract DINOv2-base patch features (layers -1, -2, -3, -4) from normal reference images
2. **Subspace Modeling:** Fit PCA on features, retaining 99% variance to estimate the low-dimensional manifold of normal appearance
3. **Anomaly Detection:** Project test image features into PCA subspace, compute reconstruction residual — high residual indicates anomaly

### Dual-Engine OR Fusion

- **YOLO** excels at detecting known defect types it was trained on
- **SubspaceAD** detects any deviation from normal appearance (domain-aware, training-free)
- OR fusion captures both known and unknown defect patterns

---

## AutoResearch Workflow

The `yolo-autoresearch-v2` module uses an AutoResearch-style autonomous training loop:

1. Agent reads `results.tsv` for experiment history
2. Agent proposes a hypothesis and modifies `train.py` or `src/` modules
3. Runs `python train.py '{"name":"exp_XXX", ...}'` with experimental config
4. Parses output for `RESULT map50=...`
5. If improvement: commits to git; otherwise: resets
6. Results appended to `results.tsv`, next iteration begins

### Known Best Practices (from V1 experiments)

- **Mosaic must be OFF (0.0):** 200×200 small images lose defects after mosaic拼接
- **Learning rate:** Start from 0.001 (baseline's 0.01 is too high for small datasets)
- **degrees=5:** Slight rotation helps; beyond 8° is harmful
- **yolov8n preferred:** yolov8m overfits on 1800-image dataset

---

## Models & Artifacts

| File | Description |
|------|-------------|
| `yolo-autoresearch-v2/runs/detect/runs/baseline/yolov8n_baseline/weights/best.pt` | YOLOv8n trained on NEU-DET |
| `yolo-autoresearch-v2/models/subspace_pca_neu.pt` | SubspaceAD PCA fitted on MVTec-AD bottle |
| `SubspaceAD/results_full/benchmark_results.csv` | Full MVTec-AD benchmark results |

---

## References

- [SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling (CVPR 2026)](https://arxiv.org/abs/2602.23013)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [AutoResearch (karpathy)](https://karpathy.github.io/2025/03/17/autoresearch/)

---

## License

MIT
