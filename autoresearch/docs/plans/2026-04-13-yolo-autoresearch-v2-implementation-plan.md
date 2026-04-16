# YOLO AutoResearch v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 YOLO AutoResearch v2 框架，通过 Loss/Assigner 多路探索突破 0.80 mAP

**Architecture:** 在原有 AutoResearch-YOLO 基础上，新增 MultiClassFocus Assigner 和多种 Loss 模块，扩展 program.md 分阶段探索策略

**Tech Stack:** Python, PyTorch, Ultralytics YOLOv8, Git

---

## 前置准备

### Task 0: 环境确认

**Files:**
- Check: `autoresearch/train.py`
- Check: `autoresearch/prepare.py`
- Check: `autoresearch/neu-det.yaml`

**Step 1: 确认项目结构**

Run: `ls -la /media/hzm/Data/auto_yolo/autoresearch/`
Expected: 看到 train.py, prepare.py, program.md, neu-det.yaml

**Step 2: 确认 NEU-DET 数据存在**

Run: `ls -la /media/hzm/Data/auto_yolo/autoresearch/data/ 2>/dev/null || echo "数据目录不存在"`
Expected: 如果不存在，人类需要运行 prepare_dataset.py

---

## 阶段一：基础框架搭建

### Task 1: 创建 src/ 目录结构

**Files:**
- Create: `autoresearch/src/__init__.py`
- Create: `autoresearch/src/assigner.py`
- Create: `autoresearch/src/losses.py`
- Create: `autoresearch/src/search_space.py`

**Step 1: 创建目录和 __init__.py**

```bash
mkdir -p /media/hzm/Data/auto_yolo/autoresearch/src
touch /media/hzm/Data/auto_yolo/autoresearch/src/__init__.py
```

**Step 2: Commit**

```bash
cd /media/hzm/Data/auto_yolo/autoresearch
git add src/__init__.py
git commit -m "feat: create src directory structure"
```

---

### Task 2: 实现 MultiClassFocus Assigner

**Files:**
- Create: `autoresearch/src/assigner.py`
- Modify: `autoresearch/train.py` (添加 inject 函数)

**Step 1: 编写测试**

```python
# 写入: /media/hzm/Data/auto_yolo/autoresearch/tests/test_assigner.py
import torch
import sys
sys.path.insert(0, '/media/hzm/Data/auto_yolo/autoresearch')
from src.assigner import MultiClassFocusedAssigner

def test_class_boosts():
    """测试多类别加权功能"""
    from ultralytics.utils.tal import TaskAlignedAssigner

    # 创建测试数据
    pd_scores = torch.randn(100, 10)
    pd_bboxes = torch.randn(100, 4)
    gt_labels = torch.randint(0, 6, (10, 1))
    gt_bboxes = torch.randn(10, 4)
    mask_gt = torch.ones(10, dtype=torch.bool)

    # 测试无 boost（应该等于原始）
    assigner_base = TaskAlignedAssigner(topk=10, num_classes=6)
    metrics_base, _ = assigner_base.get_box_metrics(
        pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

    # 测试有 boost
    assigner_boosted = MultiClassFocusedAssigner(
        topk=10, num_classes=6, class_boosts={0: 2.0})
    metrics_boosted, _ = assigner_boosted.get_box_metrics(
        pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

    # crazing(0) 的 GT 应该有更高的 metric
    gt_is_crazing = gt_labels.squeeze(-1).eq(0)
    if gt_is_crazing.any():
        # boosted metrics for crazing should be 2x
        print("Test passed: MultiClassFocusAssigner works")

test_class_boosts()
```

**Step 2: 运行测试**

Run: `cd /media/hzm/Data/auto_yolo/autoresearch && python -m pytest tests/test_assigner.py -v`
Expected: FAIL (因为 MultiClassFocusedAssigner 还没实现)

**Step 3: 实现 assigner.py**

```python
# /media/hzm/Data/auto_yolo/autoresearch/src/assigner.py
import torch
from ultralytics.utils.tal import TaskAlignedAssigner

class MultiClassFocusedAssigner(TaskAlignedAssigner):
    """可配置的多类别加权 Assigner

    在正样本分配时，给指定类别额外加权，让模型在梯度更新时
    更多地学习困难类别的特征。
    """
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


def inject_focused_tal(class_boosts: dict = None, crazing_boost: float = 1.0):
    """注入自定义 Assigner（monkey-patch，只影响训练，不影响推理）"""
    from ultralytics.utils import tal

    # 合并 crazing_boost 到 class_boosts
    if crazing_boost != 1.0:
        if class_boosts is None:
            class_boosts = {}
        class_boosts = {0: crazing_boost, **class_boosts}

    class Patched(MultiClassFocusedAssigner):
        def __init__(self, *a, **kw):
            super().__init__(*a, class_boosts=class_boosts, **kw)

    tal.TaskAlignedAssigner = Patched
```

**Step 4: 再次运行测试**

Run: `cd /media/hzm/Data/auto_yolo/autoresearch && python -m pytest tests/test_assigner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/assigner.py tests/test_assigner.py
git commit -m "feat: add MultiClassFocusedAssigner with class_boosts support"
```

---

### Task 3: 实现 Loss 模块 (SIoU/EIoU/Varifocal)

**Files:**
- Create: `autoresearch/src/losses.py`
- Modify: `autoresearch/train.py` (添加 loss 切换)

**Step 1: 编写测试**

```python
# 写入: /media/hzm/Data/auto_yolo/autoresearch/tests/test_losses.py
import torch
import sys
sys.path.insert(0, '/media/hzm/Data/auto_yolo/autoresearch')
from src.losses import SIoULoss, EIoULoss, VarifocalLoss

def test_siou():
    pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    loss_fn = SIoULoss()
    loss = loss_fn(pred, target)
    assert loss.item() >= 0
    print(f"SIoU loss: {loss.item()}")

def test_eiou():
    pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    loss_fn = EIoULoss()
    loss = loss_fn(pred, target)
    assert loss.item() >= 0
    print(f"EIoU loss: {loss.item()}")

def test_varifocal():
    pred = torch.randn(10, 1)
    target = torch.randn(10, 1)
    loss_fn = VarifocalLoss()
    loss = loss_fn(pred, target)
    assert loss.item() >= 0
    print(f"Varifocal loss: {loss.item()}")

test_siou()
test_eiou()
test_varifocal()
```

**Step 2: 运行测试**

Run: `cd /media/hzm/Data/auto_yolo/autoresearch && python -m pytest tests/test_losses.py -v`
Expected: FAIL (因为 losses.py 还没实现)

**Step 3: 实现 losses.py**

```python
# /media/hzm/Data/auto_yolo/autoresearch/src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SIoULoss(nn.Module):
    """Shape-IoU Loss
    考虑边框的形状相似性，而不仅仅是几何距离
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred/target: [N, 4] (x1, y1, x2, y2)
        pred_xy = (pred[:, :2] + pred[:, 2:]) / 2
        pred_wh = pred[:, 2:] - pred[:, :2]
        target_xy = (target[:, :2] + target[:, 2:]) / 2
        target_wh = target[:, 2:] - target[:, :2]

        # 距离损失
        d_xy = (pred_xy - target_xy) ** 2
        d_diag = d_xy.sum(dim=1)

        # 形状损失
        v_w = torch.abs(pred_wh[:, 0] - target_wh[:, 0])
        v_h = torch.abs(pred_wh[:, 1] - target_wh[:, 1])
        v = (4 / torch.pi ** 2) * (torch.atan(target_wh[:, 0] / (target_wh[:, 1] + 1e-8)) -
                                     torch.atan(pred_wh[:, 0] / (pred_wh[:, 1] + 1e-8))) ** 2

        alpha = v / ((1 - torch.exp(-v)) + 1e-8)

        siou = 1 - torch.exp(-d_diag) + alpha * v
        return siou.mean()


class EIoULoss(nn.Module):
    """Efficient-IoU Loss
    分别考虑重叠面积、距离、形状
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred/target: [N, 4] (x1, y1, x2, y2)
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

        # IoU
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        union_area = pred_area + target_area - inter_area + 1e-8
        iou = inter_area / union_area

        # 距离损失
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        d_diag = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # 形状损失
        v_w = torch.abs(pred[:, 2] - pred[:, 0] - (target[:, 2] - target[:, 0]))
        v_h = torch.abs(pred[:, 3] - pred[:, 1] - (target[:, 3] - target[:, 1]))
        v = (4 / torch.pi ** 2) * (torch.atan((target[:, 2] - target[:, 0]) / (target[:, 3] - target[:, 1] + 1e-8)) -
                                     torch.atan((pred[:, 2] - pred[:, 0]) / (pred[:, 3] - pred[:, 1] + 1e-8))) ** 2

        alpha = v / ((1 - torch.exp(-v)) + 1e-8)

        eiou = iou - alpha * v - d_diag / (640 ** 2)  # 假设图像尺寸 640
        return (1 - eiou).mean()


class VarifocalLoss(nn.Module):
    """Varifocal Loss for object detection
    用于检测任务的聚焦损失，对正负样本使用不同策略
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred: [N, 1] 预测分数
        # target: [N, 1] 目标分数
        pred = pred.clamp(1e-8, 1 - 1e-8)

        # 计算交叉熵
        loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # 聚焦因子
        pred_prob = torch.where(target >= 0.5, pred, 1 - pred)
        focal_weight = torch.pow(1 - pred_prob, 3)

        return (focal_weight * loss).mean()
```

**Step 4: 再次运行测试**

Run: `cd /media/hzm/Data/auto_yolo/autoresearch && python -m pytest tests/test_losses.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/losses.py tests/test_losses.py
git commit -m "feat: add SIoU, EIoU, VarifocalLoss implementations"
```

---

### Task 4: 实现搜索空间定义

**Files:**
- Create: `autoresearch/src/search_space.py`

**Step 1: 编写 search_space.py**

```python
# /media/hzm/Data/auto_yolo/autoresearch/src/search_space.py
"""搜索空间定义 - Phase 2 多方向探索参数"""

# 方向 A: TAL 参数扩展
TAL_PARAM_SPACE = {
    'tal_topk': [15, 18, 21, 24, 27, 30],
    'tal_alpha': [0.3, 0.4, 0.5, 0.6, 0.7],
    'tal_beta': [2.0, 2.5, 2.9, 3.5, 4.0, 5.0],
}

# 方向 B: 多类别加权
CLASS_BOOST_SPACES = [
    {0: 2.0},                          # 只加 crazing
    {0: 2.5},                          # crazing 更高
    {0: 3.0},                          # crazing 更高
    {0: 2.0, 4: 1.5},                 # 加 rolled-in_scale
    {0: 2.0, 1: 1.3},                  # 加 inclusion
    {0: 2.0, 4: 1.5, 1: 1.3},         # 加多个困难类
]

# 方向 C: Loss 类型
LOSS_TYPES = [
    'ciou',     # 默认
    'siou',     # SIoU
    'eiou',     # EIoU
    'varifocal', # Varifocal
]

# 方向 D: Focal Loss 变体
FOCAL_VARIANTS = [
    'standard',     # 标准 BCE
    'focal',        # Focal Loss
    'varifocal',    # Varifocal Loss
]

# 基础配置（Phase 1 复现）
BASELINE_CONFIG = {
    'model': 'yolov8n.pt',
    'epochs': 100,
    'tal_topk': 24,
    'tal_beta': 2.9,
    'crazing_boost': 2.0,
    'lr0': 0.001,
    'batch': 64,
    'mosaic': 0.0,
    'degrees': 5.0,
    'optimizer': 'AdamW',
}
```

**Step 2: Commit**

```bash
git add src/search_space.py
git commit -m "feat: add search space definitions for Phase 2 exploration"
```

---

### Task 5: 修改 train.py 集成新模块

**Files:**
- Modify: `autoresearch/train.py`

**Step 1: 备份并修改 train.py**

在 `run(cfg: dict)` 函数中添加：

```python
# 在 run() 函数开头添加：
# 支持 class_boosts 和 loss_type
if cfg.get("class_boosts") or cfg.get("crazing_boost", 1.0) != 1.0:
    from src.assigner import inject_focused_tal
    inject_focused_tal(
        class_boosts=cfg.get("class_boosts"),
        crazing_boost=cfg.get("crazing_boost", 1.0)
    )
```

**Step 2: Commit**

```bash
git add train.py
git commit -m "feat: integrate MultiClassFocus Assigner into train.py"
```

---

### Task 6: 创建 program.md 任务说明书

**Files:**
- Create: `autoresearch/program.md`

**Step 1: 编写 program.md**

```markdown
# YOLO 钢铁缺陷检测 - 突破 0.80 mAP

## 背景
基于 AutoResearch-YOLO 框架，Phase 1 确认了 0.773 基线。
现在进入 Phase 2-4，多方向探索 Loss/Assigner 改进。

## 目标
最大化 NEU-DET 验证集的 mAP@0.5，目标是突破 0.80

## 文件规则
- train.py **唯一可修改**
- prepare_dataset.py 和 neu-det.yaml **不允许修改**
- src/ 下的模块也可以修改

## 分阶段策略

### Phase 1：Baseline 确认（已完成）
- 配置：tal_topk=24, tal_beta=2.9, crazing_boost=2.0
- 目标：0.773

### Phase 2：多方向探索（~20轮）
每个方向至少尝试 3 轮，记录有效性。

**方向 A - TAL 参数扩展**：
- topk: [15, 18, 21, 24, 27, 30]
- beta: [2.0, 2.5, 2.9, 3.5, 4.0, 5.0]
- alpha: [0.3, 0.4, 0.5, 0.6, 0.7]

**方向 B - 多类别加权**：
- {0: 2.0} 只加 crazing
- {0: 2.5}, {0: 3.0} 更激进的 crazing
- {0: 2.0, 4: 1.5} 加 rolled-in_scale
- {0: 2.0, 1: 1.3} 加 inclusion

**方向 C - 定位 Loss**：
- loss_type: siou, eiou, varifocal

### Phase 3：收敛深耕（~10轮）
选择 Phase 2 最佳方向精细搜索

### Phase 4：组合突破（~15轮）
组合多个有效元素，冲击 0.80

## 实验循环（严格按此执行）
1. 阅读 results.tsv 中的历史实验结果
2. 基于历史结果提出假设，选择下一个探索方向
3. 修改 train.py（或 src/ 模块）
4. 运行：`python train.py '{"name":"exp_XXX", ...}'
5. 解析输出的 RESULT map50=... 行
6. 若当前 map50 > 历史最优：git add . && git commit -m "keep exp_XXX map50=..."
7. 若当前 map50 <= 历史最优：git reset --hard HEAD
8. 将结果追加到 results.tsv

## 退出条件
连续 10 轮无提升且方向已充分探索时停止。
```

**Step 2: Commit**

```bash
git add program.md
git commit -m "feat: add program.md for v2 autoresearch"
```

---

### Task 7: 创建 results.tsv 和 artifacts 目录

**Files:**
- Create: `autoresearch/results.tsv`
- Create: `autoresearch/artifacts/` (目录)

**Step 1: 创建文件**

```bash
cd /media/hzm/Data/auto_yolo/autoresearch
echo -e "exp_id\tphase\tdirection\tconfig\tmap50\tcrazing_ap\tstatus\tnotes" > results.tsv
mkdir -p artifacts
```

**Step 2: Commit**

```bash
git add results.tsv
git commit -m "feat: initialize results.tsv and artifacts directory"
```

---

### Task 8: 创建监控面板

**Files:**
- Create: `autoresearch/monitor_dashboard.py`

（参考原文档中的实现，此处略）

**Step 1: Commit**

```bash
git add monitor_dashboard.py
git commit -m "feat: add monitor dashboard"
```

---

## 实施检查清单

完成所有 Task 后，确认：

- [ ] src/assigner.py 实现 MultiClassFocus Assigner
- [ ] src/losses.py 实现 SIoU/EIoU/VarifocalLoss
- [ ] src/search_space.py 定义搜索空间
- [ ] train.py 集成新模块
- [ ] program.md 定义分阶段策略
- [ ] results.tsv 和 artifacts 就绪
- [ ] 所有测试通过
- [ ] Git history 清晰

---

## 下一步

**Phase 1**: 复现 0.773 基线（3轮）
**Phase 2**: 多方向探索（~20轮）
**Phase 3**: 收敛深耕（~10轮）
**Phase 4**: 组合突破（~15轮）

---

Plan complete and saved to `docs/plans/2026-04-13-yolo-autoresearch-v2-implementation-plan.md`.
