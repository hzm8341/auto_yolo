# YOLO + SubspaceAD Dual-Engine Anomaly Detection

A dual-engine industrial defect detection system combining YOLOv8 (supervised detection) with SubspaceAD (self-supervised anomaly scoring).

## Project Structure

```
auto_yolo/
├── yolo-autoresearch-v2/     # YOLO training + dual-engine inference
│   ├── data/                 # NEU-DET dataset (train/val splits)
│   ├── models/               # Trained models and PCA
│   │   ├── subspace_pca_neu.pt   # SubspaceAD PCA fitted on MVTec-AD bottle
│   │   └── subspace_pca.pkl      # Original SubspaceAD PCA
│   ├── runs/detect/          # YOLO training outputs
│   ├── NEU-DET/              # Raw NEU-DET dataset
│   ├── inference.py          # Phase 3 dual-engine inference
│   ├── train.py              # YOLO training script
│   └── prepare_dataset.py    # Dataset preparation
└── SubspaceAD/               # SubspaceAD benchmark
    ├── main.py               # SubspaceAD evaluation
    ├── datasets/             # MVTec-AD dataset
    └── results_full/         # Benchmark results
```

## Installation

```bash
# YOLO environment
pip install ultralytics torch torchvision

# SubspaceAD environment
cd SubspaceAD
pip install transformers scikit-learn opencv-python pandas tqdm anomalib
```

## Phase 1: SubspaceAD Benchmark (MVTec-AD)

Validate SubspaceAD on the standard MVTec-AD benchmark using DINOv2-base.

```bash
cd /media/hzm/Data/auto_yolo/SubspaceAD
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

## Phase 2: YOLO Training (NEU-DET)

Train YOLOv8n on the NEU-DET steel surface defect dataset.

```bash
cd yolo-autoresearch-v2
python train.py --data neu-det.yaml --epochs 50 --imgsz 200
```

**Dataset:** NEU-DET (6 defect classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)
- Train: 1729 images
- Val: 649 images

**Results — YOLOv8n Baseline (epoch 50):**

| Metric           | Value  |
|------------------|--------|
| Precision        | 0.854  |
| Recall           | 0.768  |
| **mAP@0.5**      | **0.862** |
| mAP@0.5:0.95    | 0.513  |
| YOLO Recall (val)| 99.1% (643/649) |

## Phase 3: Dual-Engine Fusion

Combine YOLO (known defect detection) with SubspaceAD (unknown anomaly scoring) using **OR fusion**: report defect if either engine detects an anomaly.

```bash
cd yolo-autoresearch-v2
python inference.py
```

### Fusion Logic

```
is_defect = (YOLO detects defect) OR (SubspaceAD anomaly_score > threshold)
```

### OR Fusion Results (NEU-DET val, 649 defect images):

| Engine            | Recall       | Notes                          |
|-------------------|--------------|--------------------------------|
| YOLO alone        | 99.1% (643/649) | Supervised, NEU-DET domain    |
| SubspaceAD alone  | ~50% (325/649) | Cross-domain, fitted on MVTec-AD |
| **OR Fusion**     | **99.5% (646/649)** | YOLO + SubspaceAD union   |

**YOLO missed 6 images; SubspaceAD recovered 3 of them.**

## Models & Artifacts

| File                                | Description                              |
|-------------------------------------|------------------------------------------|
| `runs/detect/runs/baseline/yolov8n_baseline/weights/best.pt` | YOLOv8n trained on NEU-DET |
| `models/subspace_pca_neu.pt`        | SubspaceAD PCA fitted on MVTec-AD bottle good images |
| `SubspaceAD/results_full/benchmark_results.csv` | Full MVTec-AD benchmark results |

## How It Works

### SubspaceAD
1. Extract DINOv2-base patch features (layers -1, -2, -3, -4)
2. Fit PCA on clean reference images (retaining 99% variance)
3. Project test image features into PCA subspace
4. Compute reconstruction residual — high residual = anomaly

### Dual-Engine OR Fusion
- **YOLO** excels at detecting known defect types it was trained on
- **SubspaceAD** detects any deviation from normal appearance (domain-aware)
- OR fusion captures both known and unknown defect patterns

## NEU-DET Defect Classes

| ID | Class Name     | Description                    |
|----|----------------|--------------------------------|
| 0  | crazing        | Cracking /龟裂                 |
| 1  | inclusion      | Non-metallic inclusions /夹杂   |
| 2  | patches        | Surface patches /斑块          |
| 3  | pitted_surface | Pitting corrosion /麻面        |
| 4  | rolled-in_scale | Rolled oxide scale /氧化铁皮   |
| 5  | scratches      | Surface scratches /划痕        |
