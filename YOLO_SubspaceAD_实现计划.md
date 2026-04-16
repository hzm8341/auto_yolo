# YOLO + SubspaceAD 实现计划

> 基于《YOLO + SubspaceAD：一张良品图，检出所有未知缺陷》的技术复现与实施方案
> 论文来源：arxiv 2602.23013 | 代码：github.com/CLendering/SubspaceAD

---

## 技术可行性评估

### 可信的部分

SubspaceAD的核心技术路线完全成立：DINOv2冻结特征提取 + PCA子空间建模 + 重建残差异常打分，是经典one-class classification思路的现代化版本。GitHub仓库可直接使用，MVTec-AD上1-shot AUROC 98.0%的数据与近两年异常检测领域进展吻合。

### 需要保持谨慎的部分

| 说法 | 风险等级 | 说明 |
|------|----------|------|
| "漏检率直降96%" | ⚠️ 高 | 无对比基准，营销性描述，不可作为工程决策依据 |
| "导入周期缩短80%" | ⚠️ 高 | 同上 |
| CVPR 2026收录 | ⚠️ 中 | 预印本阶段，正式录用情况需自行核实 |
| DINOv2-G推理速度 | ⚠️ 中 | 模型约1.1B参数，边缘设备可能是瓶颈，文章未提及 |
| 双引擎OR融合逻辑 | ⚠️ 中 | 提高召回同时会增加误报，阈值调校是关键工作 |

### 适用场景判断

这套方案最适合以下场景，越符合越推荐：

- 已知缺陷类型有限，但担心线上出现未见过的新缺陷
- 缺陷样本极度稀缺（千级甚至百级）
- 新品类切换频繁，重新标注成本高
- 对漏检容忍度低，对误报有一定容忍空间

---

## 架构总览

```
输入图像
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  ▼
有监督 YOLO                      SubspaceAD
（已知缺陷精确定位）              （未知缺陷异常打分）
    │                                  │
    │  检测框 + 类别                   │  像素级异常热图
    │                                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
            双引擎融合决策
    YOLO命中 OR 异常分数 > 阈值
                   │
                   ▼
            标记为缺陷区域
```

**两个引擎的职责分工：**

| 引擎 | 负责范围 | 依赖 | 特点 |
|------|----------|------|------|
| 有监督YOLO | 已知缺陷类型，精确框定位 | 标注数据 | 高精度，强可解释性 |
| SubspaceAD | 未知/新型缺陷，区域异常打分 | 仅需良品图 | 零训练，泛化强 |

---

## 阶段一：环境准备

### 1.1 依赖安装

```bash
# 核心依赖
pip install torch torchvision
pip install transformers          # DINOv2
pip install scikit-learn          # PCA
pip install opencv-python
pip install ultralytics           # YOLOv8
pip install albumentations        # 数据增强

# SubspaceAD官方代码
git clone https://github.com/CLendering/SubspaceAD
cd SubspaceAD
```

### 1.2 硬件需求说明

| 组件 | 最低配置 | 推荐配置 | 备注 |
|------|----------|----------|------|
| YOLO训练 | RTX 3080 | A100/H100 | 可云端租用 |
| DINOv2-G推理 | 16GB显存 | 24GB显存 | G版模型较大 |
| SubspaceAD拟合 | CPU可运行 | GPU加速 | PCA计算量不大 |
| 边缘部署 | - | Jetson AGX Orin | 需提前评估延迟 |

> 注意：DINOv2-G约1.1B参数，如显存不足可降级使用DINOv2-B（86M参数），性能会有一定损失，但部署成本大幅下降。

### 1.3 目录结构

```
project/
├── data/
│   ├── good/               # 良品图（SubspaceAD拟合用）
│   ├── train/              # 已知缺陷训练集（YOLO用）
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── models/
│   ├── yolo_best.pt        # 训练好的YOLO权重
│   └── subspace_pca.pkl    # 拟合好的PCA模型
├── subspace_ad/            # SubspaceAD相关代码
├── train_yolo.py           # 有监督YOLO训练脚本
├── fit_subspace.py         # SubspaceAD拟合脚本
├── inference.py            # 双引擎融合推理
└── evaluate.py             # 评估脚本
```

---

## 阶段二：有监督YOLO训练（第一引擎）

### 2.1 数据增强策略

针对缺陷样本少、类别失衡的核心问题，采用模拟光学成像增强：

```python
import albumentations as A

# 模拟光学成像数据增强（小样本场景必备）
transform = A.Compose([
    # 光照变化
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    # 透视变换（模拟相机角度偏差）
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    # 高斯噪声（模拟传感器噪声）
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    # 运动模糊（模拟产线振动）
    A.MotionBlur(blur_limit=5, p=0.3),
    # 标准翻转
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
])
```

> 注意：与第一篇文章的NEU-DET实验结论保持一致——mosaic增强对小目标/小分辨率图像有害，建议关闭（mosaic=0.0）。

### 2.2 加权损失函数处理类别失衡

```python
def compute_class_weights(dataset_labels, num_classes):
    """根据各类样本数量自动计算权重"""
    class_counts = [0] * num_classes
    for label_file in dataset_labels:
        with open(label_file) as f:
            for line in f:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1

    # 反频率加权：样本越少，权重越大
    total = sum(class_counts)
    weights = [total / (num_classes * count) if count > 0 else 1.0
               for count in class_counts]
    return weights

def reweighted_loss(cls_loss, cls_weights):
    """对各类别损失进行加权"""
    return sum(cls_weights[cls] * loss
               for cls, loss in cls_loss.items())
```

### 2.3 YOLO训练配置

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # 小数据集优先用nano，参考第一篇文章结论

model.train(
    data="defect.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    lr0=0.005,          # 参考AutoResearch实验结论：0.004-0.005最优
    optimizer="AdamW",
    mosaic=0.0,         # 关闭mosaic（小目标场景）
    degrees=5.0,        # 微旋转
    scale=0.15,
    # 轻量化：在Neck使用GSConv（需自定义模型配置）
    project="runs/yolo",
    name="defect_detector",
)
```

### 2.4 Neck轻量化（GSConv）

文章提到在Neck网络使用幻影卷积GSConv，降低复杂度同时增强非线性能力：

```python
# 在自定义YOLO配置文件中替换Neck层
# defect_yolov8n_gsconv.yaml 关键修改：
# 将Neck中标准Conv替换为GSConv
# GSConv = 标准卷积 + 深度可分离卷积的组合
# 参数量减少约30%，感受野不变

class GSConv(nn.Module):
    """幻影卷积：标准卷积 + DWConv的并联结构"""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)
```

---

## 阶段三：SubspaceAD拟合（第二引擎）

### 3.1 核心原理

```
良品图输入
    │
    ▼
DINOv2-G（冻结权重）
    │ 每张图切14×14=196个patch
    │ 每个patch → 1536维特征向量
    ▼
随机旋转增强×30次
    │ 扩充正常样本覆盖范围
    ▼
PCA拟合
    │ 估计正常变化的低维子空间
    │ 保留解释95%方差的主成分
    ▼
保存PCA模型（<1MB）
```

### 3.2 拟合脚本

```python
import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pickle, torchvision.transforms.functional as TF

class SubspaceADFitter:
    def __init__(self, model_name="facebook/dinov2-giant", pca_ev=0.95):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().cuda()
        self.pca = PCA(n_components=pca_ev)  # 保留95%方差
        self.pca_ev = pca_ev

    @torch.no_grad()
    def extract_features(self, image_path, n_rotations=30):
        """提取单张图像的patch特征，含旋转增强"""
        img = Image.open(image_path).convert("RGB")
        all_features = []

        angles = np.linspace(0, 360, n_rotations, endpoint=False)
        for angle in angles:
            rotated = TF.rotate(img, float(angle))
            inputs = self.processor(images=rotated, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # 取patch tokens（去掉CLS token）
            patch_features = outputs.last_hidden_state[:, 1:, :]  # [1, 196, 1536]
            all_features.append(patch_features.squeeze(0).cpu().numpy())

        return np.concatenate(all_features, axis=0)  # [196*30, 1536]

    def fit(self, good_image_paths):
        """用所有良品图拟合PCA子空间"""
        all_features = []
        for path in good_image_paths:
            features = self.extract_features(path)
            all_features.append(features)

        all_features = np.concatenate(all_features, axis=0)
        print(f"拟合PCA，特征矩阵大小：{all_features.shape}")
        self.pca.fit(all_features)
        print(f"保留主成分数：{self.pca.n_components_}")
        return self

    def save(self, path="models/subspace_pca.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.pca, f)
        print(f"PCA模型已保存至 {path}，大小：{os.path.getsize(path)/1024:.1f}KB")


# 使用方式：一张良品图即可完成拟合
fitter = SubspaceADFitter()
fitter.fit(["data/good/good_sample_001.jpg"])  # 可以多张
fitter.save("models/subspace_pca.pkl")
```

### 3.3 推理与异常热图生成

```python
class SubspaceADInferencer:
    def __init__(self, pca_path, model_name="facebook/dinov2-giant"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().cuda()
        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)

    @torch.no_grad()
    def get_anomaly_map(self, image_path):
        """返回像素级异常热图"""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model(**inputs)
        patch_features = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()
        # [196, 1536]

        # 投影到PCA子空间，计算重建残差
        projected = self.pca.transform(patch_features)
        reconstructed = self.pca.inverse_transform(projected)
        residuals = np.sum((patch_features - reconstructed) ** 2, axis=1)
        # [196] → reshape为14×14

        patch_grid = residuals.reshape(14, 14)

        # 上采样到原图尺寸
        import cv2
        anomaly_map = cv2.resize(patch_grid, (w, h),
                                  interpolation=cv2.INTER_CUBIC)
        return anomaly_map

    def get_anomaly_score(self, image_path):
        """返回图像级异常分数（用于二分类决策）"""
        anomaly_map = self.get_anomaly_map(image_path)
        return float(anomaly_map.max())  # 也可用均值或top-k均值
```

---

## 阶段四：双引擎融合推理

### 4.1 融合决策逻辑

```python
from ultralytics import YOLO

class DualEngineInspector:
    def __init__(self, yolo_path, pca_path, anomaly_threshold=None):
        self.yolo = YOLO(yolo_path)
        self.subspace = SubspaceADInferencer(pca_path)
        # 阈值需在验证集上标定，不能拍脑袋定
        self.anomaly_threshold = anomaly_threshold

    def inspect(self, image_path):
        results = {
            "yolo_detections": [],
            "anomaly_map": None,
            "anomaly_score": 0.0,
            "is_defect": False,
            "trigger": None,   # "yolo" | "subspace" | "both" | None
        }

        # 引擎一：YOLO检测已知缺陷
        yolo_results = self.yolo(image_path, conf=0.25)
        detections = yolo_results[0].boxes
        if len(detections) > 0:
            results["yolo_detections"] = detections
            results["is_defect"] = True
            results["trigger"] = "yolo"

        # 引擎二：SubspaceAD检测未知异常
        anomaly_map = self.subspace.get_anomaly_map(image_path)
        anomaly_score = float(anomaly_map.max())
        results["anomaly_map"] = anomaly_map
        results["anomaly_score"] = anomaly_score

        if anomaly_score > self.anomaly_threshold:
            results["is_defect"] = True
            results["trigger"] = "both" if results["trigger"] == "yolo" else "subspace"

        return results
```

### 4.2 阈值标定（关键步骤，文章未展开）

这是工程落地的核心工作，不能跳过：

```python
def calibrate_threshold(inspector, val_good_paths, val_defect_paths,
                         target_recall=0.95):
    """
    在验证集上标定SubspaceAD的异常阈值。
    target_recall：目标召回率，通常工业场景设0.95-0.99
    """
    good_scores = [inspector.subspace.get_anomaly_score(p)
                   for p in val_good_paths]
    defect_scores = [inspector.subspace.get_anomaly_score(p)
                     for p in val_defect_paths]

    all_scores = good_scores + defect_scores
    all_labels = [0] * len(good_scores) + [1] * len(defect_scores)

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

    # 找到满足目标召回率的最小阈值
    valid_idx = np.where(tpr >= target_recall)[0]
    best_threshold = thresholds[valid_idx[0]] if len(valid_idx) > 0 else thresholds[-1]

    precision_at_threshold = compute_precision(all_labels, all_scores, best_threshold)
    print(f"阈值：{best_threshold:.4f}")
    print(f"召回率：{tpr[valid_idx[0]]:.3f}")
    print(f"精确率：{precision_at_threshold:.3f}")
    print(f"误报率（FPR）：{fpr[valid_idx[0]]:.3f}")

    return best_threshold
```

> ⚠️ 重要提示：OR融合逻辑（YOLO或SubspaceAD任一触发即报缺陷）必然提高召回率，同时增加误报。在实际工程中需要在"漏检成本"和"误报成本"之间做明确的业务决策，再回推阈值设定。不同产品、不同缺陷类型的容忍度完全不同。

---

## 阶段五：评估与迭代

### 5.1 评估指标体系

```
有监督YOLO部分：
├── mAP@0.5（整体）
├── Per-class AP（重点关注尾部类别）
└── 参考目标：已知缺陷mAP > 90%

SubspaceAD部分：
├── 图像级 AUROC（区分良品/缺陷品的能力）
├── 像素级 AUROC（定位准确性）
└── 参考目标：图像级AUROC > 95%（MVTec基准98%）

双引擎融合：
├── 整体召回率（漏检率 = 1 - 召回率）
├── 整体精确率（误报率 = 1 - 精确率）
├── F1 Score
└── 业务指标：根据实际容忍度定义
```

### 5.2 SubspaceAD常见失效模式

| 失效模式 | 原因 | 解决方案 |
|----------|------|----------|
| 良品也被报异常 | 阈值过低 / 良品本身有正常纹理变化 | 拟合时加入更多良品变体；提高阈值 |
| 明显缺陷漏检 | pca_ev太低，子空间维度不足 | 提高pca_ev至0.98或0.99 |
| 定位不准 | patch粒度（14×14）太粗 | 换用更小patch的DINOv2变体；多尺度融合 |
| 推理速度慢 | DINOv2-G太大 | 降级到DINOv2-B；量化；ONNX导出 |

### 5.3 与AutoResearch结合的机会

将这套双引擎方案与第一篇文章的Karpathy Loop结合，可以对以下部分做自动化优化：

```
可纳入Karpathy Loop的参数：
YOLO侧：
├── 学习率、batch、优化器（已验证有效）
├── 类别权重比例（reweighted_loss的weights）
└── GSConv是否启用（结构改造，谨慎）

SubspaceAD侧：
├── pca_ev（0.90 / 0.95 / 0.98 / 0.99）
├── n_rotations（10 / 20 / 30 / 50）
└── 融合阈值（在固定验证集上自动搜索）

主指标：双引擎整体F1 Score（或业务定义的加权指标）
```

---

## 预期性能参考

| 指标 | 仅YOLO基线 | 加入SubspaceAD后 | 说明 |
|------|------------|-----------------|------|
| 已知缺陷mAP | ~75-93% | 持平（YOLO负责） | 视数据量和调优程度 |
| 已知缺陷召回率 | ~85% | 略提升 | SubspaceAD兜底 |
| 未知缺陷召回率 | ~0% | ~90%+ | SubspaceAD核心价值 |
| 整体误报率 | 低 | 有所上升 | OR逻辑的代价，需阈值调校 |
| 新品类导入时间 | 需重新标注+训练 | 拍1张良品图即可 | SubspaceAD核心优势 |
| 每品类存储 | 模型权重（MB级） | 额外+<1MB | PCA模型极小 |

---

## 资源链接

| 资源 | 链接 |
|------|------|
| SubspaceAD论文 | https://arxiv.org/abs/2602.23013 |
| SubspaceAD代码 | https://github.com/CLendering/SubspaceAD |
| DINOv2模型 | https://huggingface.co/facebook/dinov2-giant |
| YOLOv8文档 | https://docs.ultralytics.com |
| MVTec-AD数据集 | https://www.mvtec.com/company/research/datasets/mvtec-ad |

---

## 与第一篇AutoResearch方案的对比与互补

| 维度 | AutoResearch + YOLO | YOLO + SubspaceAD |
|------|---------------------|-------------------|
| 核心问题 | 如何把已知缺陷检测调到最优 | 如何检出未知/未见过的缺陷 |
| 数据需求 | 需要标注缺陷数据 | SubspaceAD只需良品图 |
| 优化方式 | Agent自主跑实验循环 | 工程集成+阈值标定 |
| 适合阶段 | 产线稳定期，持续调优 | 新品导入期，快速上线 |
| 互补关系 | 两者可以结合：用AutoResearch优化YOLO部分，SubspaceAD负责兜底未知缺陷 ||

**推荐的组合策略：**

1. 新品上线初期：以SubspaceAD为主，YOLO为辅（因为缺陷标注数据还少）
2. 数据积累后：用AutoResearch持续优化YOLO，提升已知缺陷精度
3. 稳定运行期：双引擎并行，YOLO负责精确定位+分类，SubspaceAD负责兜底新异常

---

*文档整理自《YOLO + SubspaceAD：一张良品图，检出所有未知缺陷》，结合AutoResearch实验方法论补充工程细节。SubspaceAD基于埃因霍温理工大学团队CVPR 2026预印本工作。*
