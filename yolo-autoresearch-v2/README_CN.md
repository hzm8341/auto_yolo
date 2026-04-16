# YOLO + SubspaceAD 双引擎工业缺陷检测系统

融合 YOLOv8（监督目标检测）与 SubspaceAD（自监督异常评分）的双引擎工业缺陷检测系统。

## 项目结构

```
auto_yolo/
├── yolo-autoresearch-v2/     # YOLO 训练 + 双引擎推理
│   ├── data/                 # NEU-DET 数据集（train/val 划分）
│   ├── models/               # 模型权重和 PCA 文件
│   │   ├── subspace_pca_neu.pt   # SubspaceAD PCA（MVTec-AD bottle 拟合）
│   │   └── subspace_pca.pkl      # 原始 SubspaceAD PCA
│   ├── runs/detect/          # YOLO 训练输出
│   ├── NEU-DET/              # 原始 NEU-DET 数据集
│   ├── inference.py           # Phase 3 双引擎推理脚本
│   ├── train.py               # YOLO 训练脚本
│   └── prepare_dataset.py     # 数据集准备脚本
└── SubspaceAD/               # SubspaceAD 基准测试
    ├── main.py               # SubspaceAD 评估主程序
    ├── datasets/             # MVTec-AD 数据集
    └── results_full/         # 基准测试结果
```

## 环境安装

```bash
# YOLO 环境
pip install ultralytics torch torchvision

# SubspaceAD 环境
cd SubspaceAD
pip install transformers scikit-learn opencv-python pandas tqdm anomalib
```

## Phase 1: SubspaceAD 基准测试（MVTec-AD）

使用 DINOv2-base 在标准 MVTec-AD 基准上验证 SubspaceAD。

```bash
cd /media/hzm/Data/auto_yolo/SubspaceAD
./run_full_benchmark.sh
```

**MVTec-AD 基准结果（5 类，dinov2-base，mean 聚合）：**

| 类别      | Image AUROC | Image AUPR | Pixel AUROC | AU-PRO |
|-----------|-------------|------------|-------------|--------|
| screw     | 0.8768      | 0.9373     | 0.9846      | 0.9417 |
| tile      | 1.0000      | 1.0000     | 0.9591      | 0.9149 |
| toothbrush| 0.9639      | 0.9844     | 0.9839      | 0.9561 |
| transistor| 0.9408     | 0.9105     | 0.8838      | 0.6009 |
| wood      | 0.9947      | 0.9984     | 0.9409      | 0.9432 |
| zipper    | 0.9819      | 0.9949     | 0.9742      | 0.9238 |
| **平均**  | **0.9597** | **0.9709** | **0.9544** | **0.8801** |

## Phase 2: YOLO 训练（NEU-DET）

在 NEU-DET 钢表面缺陷数据集上训练 YOLOv8n。

```bash
cd yolo-autoresearch-v2
python train.py --data neu-det.yaml --epochs 50 --imgsz 200
```

**数据集：** NEU-DET（6 类缺陷：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches）
- 训练集：1729 张
- 验证集：649 张

**YOLOv8n Baseline 结果（epoch 50）：**

| 指标           | 数值    |
|----------------|---------|
| Precision      | 0.854   |
| Recall         | 0.768   |
| **mAP@0.5**   | **0.862** |
| mAP@0.5:0.95 | 0.513   |
| YOLO 召回率（val）| 99.1%（643/649） |

## Phase 3: 双引擎融合

通过 **OR 融合** 组合 YOLO（已知缺陷检测）与 SubspaceAD（未知异常评分）：

```
is_defect = (YOLO 检测到缺陷) OR (SubspaceAD 异常分数 > 阈值)
```

**NEU-DET val 融合结果（649 张缺陷图）：**

| 引擎          | 召回率        | 说明                    |
|-------------|-------------|------------------------|
| YOLO 单独     | 99.1%（643/649） | 监督学习，NEU-DET 域内  |
| SubspaceAD 单独 | ~50%（325/649） | 跨域拟合（MVTec-AD）   |
| **OR 融合**  | **99.5%（646/649）** | YOLO + SubspaceAD 并集 |

YOLO 漏检 6 张，SubspaceAD 从中额外捕获 3 张。

## 模型与产物

| 文件                                                | 说明                      |
|---------------------------------------------------|-------------------------|
| `runs/detect/runs/baseline/yolov8n_baseline/weights/best.pt` | YOLOv8n NEU-DET 权重 |
| `models/subspace_pca_neu.pt`                      | SubspaceAD PCA（MVTec-AD bottle 拟合） |
| `SubspaceAD/results_full/benchmark_results.csv`    | MVTec-AD 完整基准测试结果  |

## 工作原理

### SubspaceAD
1. 提取 DINOv2-base patch 特征（层 -1, -2, -3, -4）
2. 在干净参考图上拟合 PCA（保留 99% 方差）
3. 将测试图特征投影到 PCA 子空间
4. 计算重建残差 — 残差越大 = 越异常

### 双引擎 OR 融合
- **YOLO** 擅长检测训练过的已知缺陷类型
- **SubspaceAD** 检测任何偏离正常外观的异常（依赖域）
- OR 融合同时覆盖已知和未知缺陷模式

## NEU-DET 缺陷类别

| ID | 类别名          | 说明       |
|----|----------------|------------|
| 0  | crazing        | 龟裂        |
| 1  | inclusion      | 夹杂        |
| 2  | patches        | 斑块        |
| 3  | pitted_surface | 麻面        |
| 4  | rolled-in_scale | 氧化铁皮压入 |
| 5  | scratches      | 划痕        |
