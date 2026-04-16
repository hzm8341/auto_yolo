# YOLO AutoResearch v2: 突破 0.80 mAP 设计文档

> 日期：2026-04-13
> 目标：基于 AutoResearch-YOLO 框架，通过 Loss/Assigner 多路探索，突破 0.80 mAP

## 1. 背景与目标

### 1.1 现状
- **当前最优**：mAP@0.5 = 0.773（长跑冠军 conv_loss_015）
- **crazing AP**：0.469（困难类别）
- **对标**：YOLOv8-SOE = 80.7%

### 1.2 目标
- **主目标**：突破 0.80 mAP@0.5
- **底线目标**：稳定超越 0.78

---

## 2. 框架设计

### 2.1 核心原则
沿用 Karpathy AutoResearch 框架核心设计：
- `prepare_dataset.py` — 锁死，仅做数据准备
- `train.py` — Agent 唯一可编辑文件
- `program.md` — 任务说明书，人类编写
- 固定周期评估，keep/discard 循环

### 2.2 项目结构
```
yolo-autoresearch-v2/
├── prepare_dataset.py          # 锁死，数据准备
├── neu-det.yaml                # 锁死，数据集配置
├── train.py                    # 唯一可编辑
├── program.md                  # 任务说明书
├── results.tsv                 # 实验日志
├── src/                        # 核心模块
│   ├── __init__.py
│   ├── assigner.py             # MultiClassFocus Assigner
│   ├── losses.py               # SIoU/EIoU/Varifocal Loss
│   └── search_space.py         # 搜索空间定义
├── artifacts/                  # 实验产物
└── monitor_dashboard.py        # 监控面板
```

---

## 3. 分阶段探索策略

### Phase 1：Baseline 确认（3轮）
**目标**：复现 0.773 基线，确认实验环境正常

**配置**：
```json
{
  "model": "yolov8n.pt",
  "epochs": 100,
  "tal_topk": 24,
  "tal_beta": 2.9,
  "crazing_boost": 2.0,
  "lr0": 0.001,
  "batch": 64,
  "mosaic": 0.0,
  "degrees": 5.0,
  "optimizer": "AdamW"
}
```

**退出条件**：3 轮内达到 >= 0.770

---

### Phase 2：多方向探索（~20轮）
**目标**：4 个方向各尝试至少 3 轮，记录有效性

#### 方向 A - TAL 参数扩展
**搜索空间**：
- topk: [15, 18, 21, 24, 27, 30]
- beta: [2.0, 2.5, 2.9, 3.5, 4.0, 5.0]
- alpha: [0.3, 0.4, 0.5, 0.6, 0.7]

**策略**：网格搜索 + 贝叶斯优化结合

#### 方向 B - 多类别加权
**搜索空间**：
- 只加 crazing(0): {0: 2.0} → {0: 2.5}, {0: 3.0}
- 加 rolled-in_scale(4): {0: 2.0, 4: 1.5}
- 加 inclusion(1): {0: 2.0, 1: 1.3}
- 动态 boosting：根据训练 loss 自动调整

#### 方向 C - 定位 Loss
**候选**：
- SIoU (Shape-IoU)
- EIoU (Efficient-IoU)
- VarifocalLoss
- 组合：SIoU + Varifocal

#### 方向 D - Focal Loss 变体
**候选**：
- Quality Focal Loss
- Varifocal Loss
- OHEM (Online Hard Example Mining)

**记录格式**：每个方向记录 best_mAP 和有效性评分 (1-5)

---

### Phase 3：收敛深耕（~10轮）
**策略**：
1. 选择 Phase 2 有效评分最高的 1-2 个方向
2. 在最优参数附近精细搜索
3. 尝试方向组合

---

### Phase 4：组合突破（~15轮）
**目标**：组合多个有效元素，冲击 0.80

**策略**：
- 最佳 TAL 参数 + 最佳类别加权 + 最佳 Loss
- 尝试更激进的参数（如更高 boost、更大 topk）

---

## 4. 核心模块设计

### 4.1 MultiClassFocus Assigner

```python
class MultiClassFocusedAssigner(TaskAlignedAssigner):
    """可配置的多类别加权 Assigner"""
    def __init__(self, *args, class_boosts: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_boosts = class_boosts or {}  # {0: 2.0, 4: 1.5, ...}

    def get_box_metrics(self, pd_scores, pd_bboxes,
                        gt_labels, gt_bboxes, mask_gt):
        align_metric, overlaps = super().get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

        if self.class_boosts:
            gt_labels_flat = gt_labels.squeeze(-1)
            for cls_id, boost in self.class_boosts.items():
                mask = gt_labels_flat.eq(cls_id).unsqueeze(-1)
                boost_matrix = torch.where(
                    mask,
                    torch.full_like(align_metric, boost),
                    torch.ones_like(align_metric)
                )
                align_metric = align_metric * boost_matrix

        return align_metric, overlaps
```

### 4.2 Loss 模块

```python
# losses.py
class SIoULoss(nn.Module):
    """Shape-IoU Loss"""
    ...

class EIoULoss(nn.Module):
    """Efficient-IoU Loss"""
    ...

class VarifocalLoss(nn.Module):
    """Varifocal Loss for object detection"""
    ...
```

### 4.3 train.py 集成

```python
def run(cfg: dict) -> float:
    # 注入 Assigner
    if cfg.get("class_boosts"):
        inject_focused_tal(crazing_boost=cfg.get("crazing_boost", 1.0),
                           class_boosts=cfg.get("class_boosts"))

    # 设置 Loss（如果需要）
    if cfg.get("loss_type"):
        set_loss_type(cfg["loss_type"])

    model = YOLO(cfg.get("model", "yolov8n.pt"))
    model.train(...)
```

---

## 5. program.md 核心内容

```markdown
# YOLO 钢铁缺陷检测 - 突破 0.80 mAP

## 目标
最大化 NEU-DET 验证集的 mAP@0.5，目标是突破 0.80

## 文件规则
- train.py **唯一可修改**
- prepare_dataset.py 和 neu-det.yaml **不允许修改**

## 分阶段策略

### Phase 1：Baseline 确认（3轮）
复现 conv_loss_015 配置，确认能达到 0.773

### Phase 2：多方向探索（~20轮）
方向 A - TAL 参数扩展
方向 B - 多类别加权
方向 C - 定位 Loss
方向 D - Focal Loss 变体

### Phase 3：收敛深耕（~10轮）
选择 Phase 2 最佳方向精细搜索

### Phase 4：组合突破（~15轮）
组合多个有效元素，冲击 0.80

## 退出条件
连续 10 轮无提升且方向已充分探索
```

---

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 结构改造负收益 | 高 | 中 | Phase 2 只做 Loss/Assigner |
| 过拟合 | 中 | 高 | 严格 100ep 长跑验证 |
| 方向探索迷失 | 中 | 中 | 明确分阶段策略 |
| Agent crash | 低 | 高 | git commit 前检查，crash 即回滚 |

---

## 7. 监控指标

| 指标 | 用途 |
|------|------|
| mAP@0.5 | 主指标，keep/discard 依据 |
| crazing AP | 短板指标 |
| val_loss | 过拟合监测 |
| training_time | 确保不超时 |

---

## 8. 预期成果

| 里程碑 | 目标 mAP | 说明 |
|--------|---------|------|
| Phase 1 | 0.773 | 基线确认 |
| Phase 2 | 0.78+ | 找到有效方向 |
| Phase 3 | 0.79+ | 精细调优 |
| Phase 4 | 0.80+ | 突破目标 |
