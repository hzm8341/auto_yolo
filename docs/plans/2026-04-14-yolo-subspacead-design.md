# YOLO + SubspaceAD 落地验证设计

> 日期：2026-04-14
> 目标：验证技术可行性，先跑通全流程
> 论文来源：arxiv 2602.23013 | 代码：github.com/CLendering/SubspaceAD

---

## 三阶段验证计划

### Phase 1：SubspaceAD + MVTec-AD（优先级最高）

**目标**：用 SubspaceAD 官方代码在 MVTec-AD 上复现论文指标（图像级 AUROC ≥ 95%）

**环境**：
- GPU：RTX 3090 24GB，直接跑 DINOv2-Giant
- Python 3.10+，CUDA 11.8+

**步骤**：
1. clone SubspaceAD 官方仓库
2. 下载 MVTec-AD 数据集（官方脚本）
3. 用 DINOv2-Giant 提取特征
4. PCA 拟合良品子空间
5. 在测试集上评估 AUROC

**验证指标**：
- 图像级 AUROC（核心指标）
- 像素级 AUROC（定位准确性）
- 推理速度 FPS

**输出**：
- 复现结果 vs 论文声明对比表
- 确定最优模型尺寸（DINOv2-G vs B）
- 依赖清单（Phase 3 用）

---

### Phase 2：YOLO + NEU-DET

**目标**：用 NEU-DET 数据集训练有监督 YOLO 基线

**数据**：
- 来源：`/media/hzm/Data/auto_yolo/NEU-DET.zip`
- 类别：6 类钢铁表面缺陷
- 分割：官方 train/val 分割

**配置**：
- 模型：YOLOv8n（nano，小数据集起步）
- mosaic=0.0（已验证对小目标有害）
- 数据增强：光照变化、透视变换、高斯噪声、运动模糊
- 加权损失处理类别失衡

**验证指标**：
- mAP@0.5（整体）
- Per-class AP（尾部类别）

**输出**：
- yolo_best.pt 基线权重
- 确认数据增强策略有效性

---

### Phase 3：双引擎融合

**目标**：将 Phase 1 的 SubspaceAD 和 Phase 2 的 YOLO 串联为端到端推理流水线

**融合逻辑**：
```
YOLO 检测到缺陷 → 报缺陷（引擎一）
SubspaceAD 异常分数 > 阈值 → 报缺陷（引擎二）
OR 逻辑：任一触发即报缺陷
```

**数据分工**：
- YOLO：NEU-DET 全量数据（有标注）
- SubspaceAD：仅用 NEU-DET 良品图拟合 PCA

**关键组件**：
- `DualEngineInspector`：双引擎并行推理类
- `calibrate_threshold`：在验证集上标定异常阈值（必须）

**验证指标**：
- 整体召回率（漏检率 = 1 - 召回率）
- 整体精确率（误报率 = 1 - 精确率）
- F1 Score

**输出**：
- `inference.py`：端到端推理脚本
- 阈值标定结果
- 融合评估报告

---

## 依赖清单

```bash
pip install torch torchvision
pip install transformers      # DINOv2
pip install scikit-learn      # PCA
pip install opencv-python
pip install ultralytics       # YOLOv8
pip install albumentations    # 数据增强
git clone https://github.com/CLendering/SubspaceAD
```

---

## 已知风险

| 风险点 | 等级 | 应对 |
|--------|------|------|
| DINOv2-G 推理慢 | 中 | 先用 G 跑通，再用 B 对比速度 |
| OR 融合提高误报 | 中 | 阈值标定是必做工作 |
| 论文营销性数据 | 高 | 以复现结果为准，不盲信论文数字 |
| NEU-DET 和真实工业场景差异 | 中 | Phase 3 集成后用真实数据验证 |

---

## 后续路径

Phase 1-3 全部验证通过后，可选方向：
1. **生产落地**：替换为真实工业数据，扩大规模
2. **AutoResearch 集成**：接入 Karpathy Loop 自动化调优
3. **边缘部署**：用 DINOv2-B + 量化，评估 Jetson Orin 推理延迟
