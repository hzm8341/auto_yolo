# Auto YOLO — 双引擎工业缺陷检测系统

<p align="center">
  <a href="https://arxiv.org/abs/2602.23013">SubspaceAD (CVPR 2026)</a> •
  <a href="https://github.com/ultralytics/ultralytics">YOLOv8</a> •
  <a href="https://karpathy.github.io/2025/03/17/autoresearch/">AutoResearch</a>
</p>

---

## 概述

本项目实现了一个**双引擎工业缺陷检测系统**，结合了：

1. **YOLOv8** (监督学习) — 在 NEU-DET 钢铁缺陷数据集上训练，检测 6 类已知缺陷
2. **SubspaceAD** (自监督学习) — 基于 PCA 子空间建模的无监督异常检测，使用 DINOv2 特征

两个引擎通过 **OR 融合**结合：只要任一引擎检测到异常即报告缺陷。这种方法既能捕获已知缺陷类型（通过 YOLO），也能检测未知/未见过的异常（通过 SubspaceAD 跨域检测）。

---

## 项目结构

```
auto_yolo/
├── yolo-autoresearch-v2/     # YOLO 训练 + 双引擎推理
│   ├── data/                 # NEU-DET 数据集（训练/验证划分）
│   ├── models/               # 训练好的 YOLO 权重和 PCA 模型
│   ├── runs/detect/          # YOLO 训练输出
│   ├── NEU-DET/              # 原始 NEU-DET 数据集
│   ├── src/                  # 自定义 assigner、loss、search_space
│   ├── inference.py          # Phase 3 双引擎推理
│   ├── train.py              # YOLO 训练脚本（Agent 可编辑）
│   └── prepare_dataset.py    # 数据集准备
├── SubspaceAD/              # SubspaceAD 官方实现
│   ├── main.py               # SubspaceAD 评估
│   ├── src/subspacead/       # 核心：PCA、特征提取、patching
│   ├── datasets/             # MVTec-AD 数据集
│   └── results_full/         # 基准测试结果
├── autoresearch/             # AutoResearch 框架（karpathy 风格）
│   ├── train.py              # Agent 可编辑的训练脚本
│   ├── prepare.py            # 数据准备
│   └── program.md            # Agent 指令
├── dinov2-base/             # DINOv2-base 模型缓存
└── models_cache/            # 缓存模型
```

---

## 安装

### YOLO 环境

```bash
pip install ultralytics torch torchvision
```

### SubspaceAD 环境

```bash
cd SubspaceAD
conda create -n subspacead python=3.10
conda activate subspacead
pip install -r requirements.txt
pip install -e .
```

---

## Phase 1: SubspaceAD 基准测试 (MVTec-AD)

使用 DINOv2-base 在标准 MVTec-AD 基准上验证 SubspaceAD。

```bash
cd SubspaceAD
./run_full_benchmark.sh
```

**MVTec-AD 结果（5 个类别，dinov2-base，平均聚合）：**

| 类别       | Image AUROC | Image AUPR | Pixel AUROC | AU-PRO |
|------------|-------------|------------|-------------|--------|
| screw      | 0.8768      | 0.9373     | 0.9846      | 0.9417 |
| tile       | 1.0000      | 1.0000     | 0.9591      | 0.9149 |
| toothbrush | 0.9639      | 0.9844     | 0.9839      | 0.9561 |
| transistor | 0.9408      | 0.9105     | 0.8838      | 0.6009 |
| wood       | 0.9947      | 0.9984     | 0.9409      | 0.9432 |
| zipper     | 0.9819      | 0.9949     | 0.9742      | 0.9238 |
| **平均**   | **0.9597**  | **0.9709** | **0.9544**  | **0.8801** |

---

## Phase 2: YOLO 训练 (NEU-DET)

在 NEU-DET 钢铁表面缺陷数据集上训练 YOLOv8n。

```bash
cd yolo-autoresearch-v2
python train.py --data neu-det.yaml --epochs 50 --imgsz 200
```

**数据集：** NEU-DET（6 类缺陷）
- 训练集：1729 张图像
- 验证集：649 张图像

| 指标           | 数值    |
|----------------|---------|
| Precision      | 0.854   |
| Recall         | 0.768   |
| **mAP@0.5**    | **0.862** |
| mAP@0.5:0.95  | 0.513   |
| YOLO Recall    | 99.1% (643/649) |

### NEU-DET 缺陷类别

| ID | 类别名称        | 描述           |
|----|-----------------|----------------|
| 0  | crazing         | 龟裂           |
| 1  | inclusion       | 非金属夹杂     |
| 2  | patches        | 斑块           |
| 3  | pitted_surface | 麻面           |
| 4  | rolled-in_scale | 氧化铁皮       |
| 5  | scratches      | 划痕           |

---

## Phase 3: 双引擎融合

结合 YOLO（已知缺陷检测）和 SubspaceAD（未知异常评分）。

```bash
cd yolo-autoresearch-v2
python inference.py
```

### 融合逻辑

```
is_defect = (YOLO 检测到缺陷) OR (SubspaceAD 异常分数 > 阈值)
```

### OR 融合结果（NEU-DET 验证集，649 张图像）：

| 引擎            | Recall         | 说明                      |
|-----------------|----------------|---------------------------|
| 仅 YOLO         | 99.1% (643/649) | 监督学习，NEU-DET 域     |
| 仅 SubspaceAD   | ~50% (325/649)  | 跨域，在 MVTec-AD 上拟合 |
| **OR 融合**     | **99.5% (646/649)** | YOLO + SubspaceAD 并集 |

**YOLO 漏检 6 张图像；SubspaceAD 恢复了其中 3 张。**

---

## 工作原理

### SubspaceAD 算法

1. **特征提取：** 从正常参考图像中提取 DINOv2-base patch 特征（层 -1, -2, -3, -4）
2. **子空间建模：** 在特征上拟合 PCA，保留 99% 方差以估计正常外观的低维流形
3. **异常检测：** 将测试图像特征投影到 PCA 子空间，计算重建残差——高残差表示异常

### 双引擎 OR 融合

- **YOLO** 擅长检测它训练过的已知缺陷类型
- **SubspaceAD** 检测任何偏离正常外观的情况（域感知、无监督）
- OR 融合捕获已知和未知缺陷模式

---

## AutoResearch 工作流程

`yolo-autoresearch-v2` 模块使用 AutoResearch 风格的自主训练循环：

1. Agent 读取 `results.tsv` 了解实验历史
2. Agent 提出假设并修改 `train.py` 或 `src/` 模块
3. 运行 `python train.py '{"name":"exp_XXX", ...}'` 进行实验配置
4. 解析输出中的 `RESULT map50=...`
5. 如有提升：提交到 git；否则：重置
6. 结果追加到 `results.tsv`，开始下一轮迭代

### 已知的最佳实践（来自 V1 实验）

- **Mosaic 必须关闭 (0.0)：** 200×200 小图经 mosaic 拼接后缺陷会消失
- **学习率：** 从 0.001 开始（基线的 0.01 对小数据集太高）
- **degrees=5：** 轻微旋转有帮助；超过 8° 有害
- **优先使用 yolov8n：** yolov8m 在 1800 张图像的数据集上容易过拟合

---

## 模型和产物

| 文件 | 描述 |
|------|------|
| `yolo-autoresearch-v2/runs/detect/runs/baseline/yolov8n_baseline/weights/best.pt` | 在 NEU-DET 上训练的 YOLOv8n |
| `yolo-autoresearch-v2/models/subspace_pca_neu.pt` | 在 MVTec-AD bottle 上拟合的 SubspaceAD PCA |
| `SubspaceAD/results_full/benchmark_results.csv` | 完整的 MVTec-AD 基准测试结果 |

---

## 参考

- [SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling (CVPR 2026)](https://arxiv.org/abs/2602.23013)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [AutoResearch (karpathy)](https://karpathy.github.io/2025/03/17/autoresearch/)

---

## 许可证

MIT
