# AutoResearch 自主调 YOLO 完整实现指南

> 基于 Karpathy AutoResearch 框架，使用 Claude Code + H100，10 小时、64 轮实验，将 YOLOv8 在 NEU-DET 钢铁缺陷检测数据集上的 mAP@0.5 从 0.729 提升至 0.773（相对提升 6.0%）。

---

## 目录

1. [核心原理：Karpathy Loop](#1-核心原理karpathy-loop)
2. [数据集：NEU-DET 钢铁缺陷](#2-数据集neu-det-钢铁缺陷)
3. [环境搭建](#3-环境搭建)
4. [项目结构](#4-项目结构)
5. [三个核心文件详解](#5-三个核心文件详解)
   - [5.1 prepare_dataset.py（锁死）](#51-prepare_datasetpy锁死不允许修改)
   - [5.2 train.py（Agent 可编辑）](#52-trainpyagent-唯一可编辑文件)
   - [5.3 program.md（任务说明书）](#53-programmd任务说明书)
6. [V1：本地验证版（M4 Pro + MiniMax）](#6-v1本地验证版m4-pro--minimax)
7. [V2：正式版（H100 + Claude Code）](#7-v2正式版h100--claude-code)
8. [四阶段实验策略](#8-四阶段实验策略)
9. [Phase 4b 核心突破：focused-TAL](#9-phase-4b-核心突破focused-tal)
10. [监控面板](#10-监控面板)
11. [实验结果汇总](#11-实验结果汇总)
12. [三条可迁移结论](#12-三条可迁移结论)
13. [迁移到其他企业场景](#13-迁移到其他企业场景)
14. [成本与时间参考](#14-成本与时间参考)

---

## 1. 核心原理：Karpathy Loop

### 什么是 AutoResearch

Andrej Karpathy 开源的框架，让 AI Agent（Claude Code 或 Codex）在 GPU 上**自主跑实验循环**，无需人工干预。

### 四要素框架

| 要素 | 定义 | 原版（GPT） | 本项目（YOLO） |
|------|------|------------|---------------|
| 可编辑文件 | Agent 唯一被允许修改的文件 | train.py（GPT 训练） | train.py（YOLO 训练） |
| 标量指标 | 一个数字判断好坏 | val_bpb（越低越好） | mAP@0.5（越高越好） |
| 固定周期 | 每轮实验耗时基本一致 | 5 分钟（H100） | 15 epoch ≈ 3~7 分钟（H100） |
| Keep/Discard | 比上轮好就留，差就回滚 | git commit / reset | git commit / reset |

### 实验循环流程

```
Agent 修改 train.py
       ↓
  H100 训练（15 epoch，约 3~7 分钟）
       ↓
  读取 mAP@0.5
       ↓
  比上轮好？
  ├── 是 → git commit（保留）
  └── 否 → git reset --hard（回滚）
       ↓
  追加结果到 results.tsv
       ↓
  永不停止，进入下一轮
```

### 与传统 AutoML 的本质区别

| 维度 | 传统 AutoML（Optuna/Grid Search） | Karpathy Loop |
|------|----------------------------------|---------------|
| 搜索对象 | 数字参数（如 lr 从 0.001 到 0.1） | 代码（可改模型结构、loss 函数、assigner） |
| 搜索方式 | 网格采样，盲目采点 | 看懂历史实验后提出假设、验证假设 |
| 能力边界 | 超参数层面 | 任意代码改动 |

---

## 2. 数据集：NEU-DET 钢铁缺陷

### 基本信息

| 属性 | 说明 |
|------|------|
| 来源 | 东北大学发布 |
| 图像总数 | 1800 张 |
| 图像尺寸 | 200×200 像素，灰度图 |
| 缺陷类别 | 6 类，每类 300 张（完美均衡） |
| 标注格式 | Pascal VOC XML（需转换为 YOLO txt） |
| 数据划分 | 80% 训练（1440 张）/ 20% 验证（360 张） |
| 下载地址 | http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270 |

### 6 类缺陷及检测难度

| 缺陷类型 | 中文 | 特征 | 基线 AP |
|---------|------|------|---------|
| patches | 斑块 | 大面积色差，特征明显 | ~95% |
| scratches | 划痕 | 线性特征，方向明确 | ~90% |
| pitted_surface | 麻面 | 密集凹坑，纹理规律 | ~85% |
| inclusion | 夹杂 | 嵌入式异物，边界模糊 | ~75% |
| rolled-in_scale | 氧化铁皮压入 | 不规则暗色斑块 | ~65% |
| crazing | 龟裂 | 细密裂纹网络，极难检测 | ~30~41% |

### 已知局限

- 200×200 分辨率偏低，crazing 类缺陷肉眼难辨
- YOLOv8 默认输入 640×640，原图放大 3 倍引入插值伪影
- 数据量小，yolov8m 以上容易过拟合

### 公开 Benchmark 对比

| 方法 | mAP@0.5 |
|------|---------|
| Faster R-CNN (ResNet-50) | 76.6% |
| YOLOv8 默认配置 | 75.9% |
| **本项目 V2 最终成绩** | **77.3%** |
| YOLOv8-TLC | 79.8% |
| YOLOv8-SOE | 80.7% |
| DGYOLOv8 | 91.2%（评估口径存疑） |

> 可靠对标区间：**76%~81%**

---

## 3. 环境搭建

### 硬件选择

| 版本 | 硬件 | 单轮耗时 | 适用场景 |
|------|------|---------|---------|
| V1 | Mac M4 Pro / 普通 PC（MPS/CPU） | ~22 分钟 | 流程验证 |
| V2 | NVIDIA H100 80GB（CUDA） | ~3~7 分钟 | 正式实验 |

H100 按小时租用，约 3 美元/小时，推荐使用 CUDA 12.9 Base 镜像。

### 安装依赖

```bash
pip install ultralytics torch torchvision
pip install gitpython flask paramiko
pip install anthropic          # Claude Code API（V2）
pip install openai             # OpenRouter / MiniMax（V1）
```

### 初始化 git 仓库

```bash
git init yolo-autoresearch && cd yolo-autoresearch
touch train.py program.md results.tsv
git add . && git commit -m "baseline"
```

---

## 4. 项目结构

```
yolo-autoresearch/
├── prepare_dataset.py     # 数据准备 + 评估（锁死，不允许 Agent 修改）
├── train.py               # 训练逻辑（Agent 唯一可编辑文件）
├── program.md             # 给 Agent 的任务说明书（人类写）
├── neu-det.yaml           # 数据集配置（锁死）
├── results.tsv            # 实验结果日志
├── auto_experiment.py     # V1 循环脚本（MiniMax 驱动）
├── monitor_dashboard.py   # 实时监控面板
└── artifacts/             # 每轮实验产物（权重/曲线/混淆矩阵）
    ├── exp_001/
    │   ├── weights/best.pt
    │   ├── results.csv
    │   └── confusion_matrix.png
    └── ...
```

---

## 5. 三个核心文件详解

### 5.1 `prepare_dataset.py`（锁死，不允许修改）

```python
import os, xml.etree.ElementTree as ET, shutil
from pathlib import Path

# NEU-DET 类别映射（顺序固定，对应 crazing=0）
CLASS_MAP = {
    'crazing': 0, 'inclusion': 1, 'patches': 2,
    'pitted_surface': 3, 'rolled-in_scale': 4, 'scratches': 5
}

def convert_voc_to_yolo(xml_path: str, img_w: int, img_h: int) -> list[str]:
    """Pascal VOC XML → YOLO txt 格式"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        # 转换为 YOLO 归一化格式
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines

def prepare(src_dir: str, out_dir: str, val_ratio: float = 0.2):
    """处理数据集，输出 train/ val/ 目录"""
    import random, glob
    random.seed(42)
    all_imgs = sorted(glob.glob(f"{src_dir}/**/*.jpg", recursive=True))
    random.shuffle(all_imgs)
    n_val = int(len(all_imgs) * val_ratio)
    splits = {'val': all_imgs[:n_val], 'train': all_imgs[n_val:]}

    for split, imgs in splits.items():
        img_out = Path(out_dir) / split / 'images'
        lbl_out = Path(out_dir) / split / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            xml_path = img_path.replace('.jpg', '.xml')
            shutil.copy(img_path, img_out / Path(img_path).name)
            lines = convert_voc_to_yolo(xml_path, 200, 200)
            lbl_file = lbl_out / Path(img_path).stem
            lbl_file.with_suffix('.txt').write_text('\n'.join(lines))

    print(f"完成：train {len(splits['train'])} 张，val {len(splits['val'])} 张")

if __name__ == '__main__':
    prepare('NEU-DET', 'data')
```

**neu-det.yaml（配套数据集配置）：**

```yaml
path: /root/yolo-autoresearch/data
train: train/images
val: val/images
nc: 6
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
```

---

### 5.2 `train.py`（Agent 唯一可编辑文件）

这是 Agent 可以自由修改的文件，包含训练逻辑、超参数、以及可选的 loss/assigner 改造。

```python
from ultralytics import YOLO
import json, sys

# ── 可选：focused-TAL Assigner（Phase 4b 引入）──────────────────────
import torch
from ultralytics.utils.tal import TaskAlignedAssigner

class CrazingFocusedTaskAlignedAssigner(TaskAlignedAssigner):
    """给 crazing 类（索引 0，最难检测的龟裂）在正样本分配时额外加权"""
    def __init__(self, *args, crazing_boost: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.crazing_boost = crazing_boost

    def get_box_metrics(self, pd_scores, pd_bboxes,
                        gt_labels, gt_bboxes, mask_gt):
        align_metric, overlaps = super().get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
        if self.crazing_boost != 1.0:
            gt_is_crazing = gt_labels.squeeze(-1).eq(0).unsqueeze(-1)
            boost = torch.where(
                gt_is_crazing,
                torch.full_like(align_metric, self.crazing_boost),
                torch.ones_like(align_metric)
            )
            align_metric = align_metric * boost
        return align_metric, overlaps

def inject_focused_tal(crazing_boost: float = 2.0):
    """Monkey-patch 注入自定义 Assigner（只影响训练，不影响推理）"""
    from ultralytics.utils import tal

    class Patched(CrazingFocusedTaskAlignedAssigner):
        def __init__(self, *a, **kw):
            super().__init__(*a, crazing_boost=crazing_boost, **kw)

    tal.TaskAlignedAssigner = Patched
# ────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> float:
    # Phase 4b：注入 focused-TAL（crazing_boost > 1.0 时启用）
    boost = cfg.get("crazing_boost", 1.0)
    if boost > 1.0:
        inject_focused_tal(crazing_boost=boost)

    model = YOLO(cfg.get("model", "yolov8n.pt"))
    model.train(
        data      = "neu-det.yaml",
        epochs    = cfg.get("epochs", 15),
        imgsz     = cfg.get("imgsz", 640),
        lr0       = cfg.get("lr0", 0.005),       # V1 发现：0.004~0.005 最优
        lrf       = cfg.get("lrf", 0.01),
        batch     = cfg.get("batch", 32),
        optimizer = cfg.get("optimizer", "auto"),
        mosaic    = cfg.get("mosaic", 0.0),       # V1 发现：必须关闭
        degrees   = cfg.get("degrees", 5.0),      # V1 发现：5度有效，>8度有害
        scale     = cfg.get("scale", 0.15),
        fliplr    = cfg.get("fliplr", 0.5),
        warmup_epochs  = cfg.get("warmup_epochs", 3),
        weight_decay   = cfg.get("weight_decay", 0.0005),
        # focused-TAL 参数（Phase 4b）
        tal_topk  = cfg.get("tal_topk", 10),      # 长跑冠军：24
        tal_alpha = cfg.get("tal_alpha", 0.5),
        tal_beta  = cfg.get("tal_beta", 6.0),     # 长跑冠军：2.9
        project   = "runs",
        name      = cfg.get("name", "exp"),
        exist_ok  = True,
        verbose   = False,
    )
    m = model.val(data="neu-det.yaml", verbose=False)
    return float(m.box.map50)

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    map50 = run(cfg)
    # 固定输出格式，供循环脚本解析
    print(f"RESULT map50={map50:.6f}")
```

**V1 已知先验（作为搜索起点）：**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `mosaic` | 0.0（关闭） | 200×200 小图四合一后缺陷消失 |
| `lr0` | 0.004~0.005 | baseline 的 0.01 太高 |
| `degrees` | 5.0 | 微旋转有效，>8 度有害 |
| `scale` | 0.15 | 缩放幅度适中 |
| `model` | yolov8n.pt | 小数据集首选，避免过拟合 |

---

### 5.3 `program.md`（任务说明书）

这是人写给 Agent 的操作手册，决定搜索边界和策略。

```markdown
# YOLO 钢铁缺陷检测优化任务

## 目标
最大化 NEU-DET 验证集的 mAP@0.5。

## 文件规则
- 你**唯一可修改**的文件是 train.py
- prepare_dataset.py 和 neu-det.yaml **不允许修改**
- 每轮结束后必须将结果追加到 results.tsv

## 实验循环（严格按此执行）
1. 阅读 results.tsv 中的历史实验结果
2. 基于历史结果提出假设，修改 train.py
3. 运行：python train.py '{"name":"exp_XXX", ...}'
4. 解析输出的 RESULT map50=... 行
5. 若当前 map50 > 历史最优：git add . && git commit -m "keep exp_XXX map50=..."
6. 若当前 map50 <= 历史最优：git reset --hard HEAD
7. 将结果追加到 results.tsv，立即开始下一轮

## 分阶段策略
- Phase 1（3轮）：yolov8n / yolov8s / yolov8m 各跑一次 15ep 基线
- Phase 2（~15轮）：超参搜索（lr0, batch, optimizer, augmentation）
- Phase 3（2轮）：Top-2 配置以 100 epoch 长跑验证
- Phase 4a（~10轮）：结构改造探索（注意力模块、多尺度头）
- Phase 4b（~30轮）：Loss / TAL Assigner 改造

## 重要先验（来自 V1 实验，作为搜索起点）
- mosaic 必须关闭（200×200 小图拼接后缺陷消失）
- lr0 建议从 0.004~0.005 开始（baseline 的 0.01 太高）
- degrees=5 有效，超过 8 有害
- yolov8m 在 1800 张小数据集上会过拟合，优先 n/s

## Phase 4b 具体指导
超参搜索天花板出现后，尝试修改 TAL Assigner 参数和 crazing_boost：
- tal_topk: 搜索范围 15~25
- tal_beta: 搜索范围 2.0~4.0
- crazing_boost: 搜索范围 1.5~2.5（crazing 是最难的类，索引为 0）
成功信号：loss_n_005 首次突破超参天花板后，继续精调收敛区间。

## results.tsv 格式
exp_id\tphase\tmodel\tepochs\tlr0\tbatch\tmap50\tstatus\tnotes

## 退出条件
边际收益低于继续投入的 GPU 成本时停止（连续 5 轮无提升且改动方向已充分探索）。
```

---

## 6. V1：本地验证版（M4 Pro + MiniMax）

V1 不需要 H100，在本地 Mac 上验证整个循环是否可行。

### `auto_experiment.py`（V1 循环脚本）

```python
import subprocess, json, pathlib
from openai import OpenAI  # 或 MiniMax API

BEST    = 0.0
HISTORY = []
client  = OpenAI(api_key="YOUR_KEY", base_url="https://openrouter.ai/api/v1")

def ask_llm(history: list) -> dict:
    """LLM 看历史实验结果，输出下一组超参 JSON"""
    prompt = f"""你是 YOLO 调参专家。以下是历史实验（最近 10 轮）：
{json.dumps(history[-10:], ensure_ascii=False, indent=2)}

当前最优 mAP@0.5: {BEST:.4f}

根据历史趋势提出下一组配置。只输出 JSON，无其他内容：
{{"model":"yolov8n.pt","lr0":0.005,"batch":32,"mosaic":0.0,"degrees":5,"optimizer":"auto"}}"""

    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",   # 可替换为 minimax/text-01
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return json.loads(resp.choices[0].message.content)

def run_one(cfg: dict, exp_id: str) -> float:
    cfg["name"] = exp_id
    r = subprocess.run(
        ["python", "train.py", json.dumps(cfg)],
        capture_output=True, text=True, timeout=7200
    )
    for line in r.stdout.split("\n"):
        if line.startswith("RESULT map50="):
            return float(line.split("=")[1])
    print(f"  [警告] 未能解析输出，stderr: {r.stderr[-200:]}")
    return 0.0

def main(n_rounds: int = 25):
    global BEST
    pathlib.Path("results.tsv").touch()

    for i in range(n_rounds):
        exp_id = f"exp_{i:03d}"
        print(f"\n── Round {i+1}/{n_rounds} ({exp_id}) ──")

        cfg   = ask_llm(HISTORY)
        print(f"  配置: {cfg}")
        map50 = run_one(cfg, exp_id)

        if map50 > BEST:
            status = "keep"
            subprocess.run(["git", "add", "."])
            subprocess.run(["git", "commit", "-m",
                f"keep {exp_id} map50={map50:.4f}"])
            BEST = map50
        else:
            status = "discard"
            subprocess.run(["git", "reset", "--hard", "HEAD"])

        rec = {**cfg, "exp_id": exp_id, "map50": map50, "status": status}
        HISTORY.append(rec)
        with open("results.tsv", "a") as f:
            f.write(f"{exp_id}\tauto\t{cfg.get('model','')}\t"
                    f"{cfg.get('epochs',15)}\t{map50:.6f}\t{status}\t\n")
        print(f"  → {status.upper()} | mAP={map50:.4f} | best={BEST:.4f}")

if __name__ == "__main__":
    # Mac 上建议挂 caffeinate 防止睡眠
    # caffeinate -i python auto_experiment.py
    main()
```

### V1 注意事项

- Mac MPS 单轮约 14~38 分钟，25 轮约需 2 个夜间 session
- 使用 `caffeinate -i python auto_experiment.py` 防止 macOS 睡眠
- 每轮后清理缓存避免磁盘撑满：`rm -rf /tmp/torch_*`

### V1 最终成绩

- mAP：0.719 → 0.759（+5.6%）
- 关键发现：lr=0.004~0.005 最优，关闭 mosaic，微旋转 5 度有效

---

## 7. V2：正式版（H100 + Claude Code）

### V1 vs V2 对比

| 维度 | V1 | V2 |
|------|----|----|
| 硬件 | Mac M4 Pro（MPS） | NVIDIA H100 80GB（CUDA） |
| Agent | MiniMax（只输出 JSON 参数） | Claude Code（直接读写代码、执行命令） |
| 搜索空间 | 超参数 | 超参数 + 模型结构 + Loss 函数 + Assigner 策略 |
| 单轮耗时 | ~22 分钟 | ~3~7 分钟 |
| 实验总数 | 25 | 64 |

### V2 启动方式

在 H100 实例上，直接让 Claude Code 读取 `program.md` 并开始执行：

```bash
# 1. SSH 连接到 H100
ssh root@YOUR_H100_IP

# 2. 进入项目目录
cd yolo-autoresearch

# 3. 启动 Claude Code，指向 program.md
claude --model claude-sonnet-4-5 < program.md

# 或者直接在 Claude Code 对话框中粘贴 program.md 内容
# 然后说：请严格按照 program.md 的规则，开始自主实验循环
```

---

## 8. 四阶段实验策略

### 阶段总览

| 阶段 | 目标 | 实验数 | 退出条件 |
|------|------|--------|---------|
| Phase 1：基线建立 | n/s/m 三尺寸各跑一次 15ep | 3 轮 | 所有候选模型跑完 |
| Phase 2：超参搜索 | lr、batch、optimizer、augmentation | ~15 轮 | 连续多轮无法突破当前最优 |
| Phase 3：长跑验证 | Top-2 配置以 100 epoch 验证 | 2 轮 | Top-N 全部完成长跑 |
| Phase 4：结构+Loss 探索 | 结构改造、loss 函数、assigner | ~45 轮 | 边际收益低于 GPU 成本 |

### Phase 1 结果（关键发现：大模型不适合小数据集）

| 模型 | 参数量 | mAP@0.5 | 结论 |
|------|--------|---------|------|
| yolov8n | ~3M | 0.7294 | 保留 |
| yolov8s | ~11M | 0.7304 | 保留 |
| yolov8m | ~26M | 0.7193 | **淘汰**（过拟合） |

### Phase 2 关键节点

- `hp_s_002`（batch=64）→ mAP 0.7625，阶段冠军
- `hp_n_005`（AdamW, lr=0.001）→ mAP 0.7553
- 天花板确认：imgsz=800、degrees=3、scale=0.10/0.20 全部 discard

### Phase 3 短跑冠军 ≠ 长跑冠军（重要发现）

| 模型 | 15ep 短跑 | 100ep 长跑 | 排名变化 |
|------|-----------|-----------|---------|
| yolov8s（hp_s_002） | 0.7625（第 1） | 0.7472 | 跌至第 2 |
| yolov8n（hp_n_005） | 0.7553（第 2） | 0.7601 | 升至第 1 |

> **结论：Phase 4 全部基于 yolov8n，不再在 s 上花预算。**

### Phase 4a 结构改造全部负收益（10 轮）

| 实验 | 改动 | mAP@0.5 | vs 基线 0.760 |
|------|------|---------|--------------|
| struct_n_001 | C2PSA backbone P5 | 0.7408 | -0.019 |
| struct_n_002 | PSA after SPPF | 0.6586 | -0.101 |
| struct_n_003 | C2PSA head P5 | 0.7179 | -0.042 |
| struct_n_006 | A2C2f backbone P5 | 0.7285 | -0.032 |
| neck_n_001 | YOLOv8-P2 多尺度头 | 0.5143 | **-0.246** |

> **结论：1800 张 200×200 小图，YOLOv8n 的 3M 参数已经是最优平衡点，往上堆结构只会帮倒忙。**

---

## 9. Phase 4b 核心突破：focused-TAL

### 什么是 TAL

Task-Aligned Assigner（TAL）决定训练时"哪些预测框算正样本"。核心参数：
- `topk`：每个真实目标最多匹配几个预测框
- `alpha`：匹配时分类得分的权重
- `beta`：匹配时定位得分的权重

### 核心代码

```python
from ultralytics.utils.tal import TaskAlignedAssigner
import torch

class CrazingFocusedTaskAlignedAssigner(TaskAlignedAssigner):
    """
    在正样本分配时，给 crazing（龟裂，类别索引=0）额外加权，
    让模型在梯度更新时更多地学习这个最难类别的特征。
    """
    def __init__(self, *args, crazing_boost: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.crazing_boost = crazing_boost

    def get_box_metrics(self, pd_scores, pd_bboxes,
                        gt_labels, gt_bboxes, mask_gt):
        align_metric, overlaps = super().get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

        if self.crazing_boost != 1.0:
            gt_is_crazing = gt_labels.squeeze(-1).eq(0).unsqueeze(-1)
            boost = torch.where(
                gt_is_crazing,
                torch.full_like(align_metric, self.crazing_boost),
                torch.ones_like(align_metric)
            )
            align_metric = align_metric * boost
        return align_metric, overlaps


def inject_focused_tal(crazing_boost: float = 2.0):
    """注入自定义 Assigner（monkey-patch，只影响训练，不影响推理）"""
    from ultralytics.utils import tal

    class Patched(CrazingFocusedTaskAlignedAssigner):
        def __init__(self, *a, **kw):
            super().__init__(*a, crazing_boost=crazing_boost, **kw)

    tal.TaskAlignedAssigner = Patched
```

### 参数搜索路径

| 阶段 | 搜索范围 | 转折点 |
|------|---------|--------|
| 初期试水 | crazing 分类 loss 加权 x2、x3 | 效果一般，未超基线 |
| 调 TAL 参数 | topk 15~24, beta 2.3~4.0, boost 1.5~2.3 | `loss_n_005` 首次突破 0.762 |
| 精调收敛 | topk 21~24, beta 2.6~2.9, boost 2.0~2.15 | 最优区间确认 |

### 最优参数（长跑冠军 conv_loss_015）

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

### 效果对比

| 指标 | 结构改造（Phase 4a） | Loss/Assigner（Phase 4b） |
|------|---------------------|--------------------------|
| 实验轮数 | 10 轮 | ~30 轮 |
| 正收益次数 | 0 | 多次 |
| 最佳 mAP 提升 | 无（全部负收益） | +0.013（vs 超参天花板） |
| 额外推理成本 | 有（增加计算量） | **零**（Assigner 只影响训练） |

---

## 10. 监控面板

### `monitor_dashboard.py`

```python
from flask import Flask, jsonify
import paramiko

app = Flask(__name__)
HOST = "YOUR_H100_IP"
USER = "root"
KEY  = "/path/to/your/id_rsa"

def ssh_run(cmd: str) -> str:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, key_filename=KEY)
    _, out, _ = c.exec_command(cmd)
    return out.read().decode().strip()

@app.route("/api/status")
def status():
    # GPU 状态（每 10 秒轮询）
    gpu = ssh_run(
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,"
        "memory.total --format=csv,noheader,nounits"
    )
    util, mem_used, mem_total = [x.strip() for x in gpu.split(",")]

    # 实验结果
    tsv = ssh_run("tail -30 ~/yolo-autoresearch/results.tsv")
    rows = [r.split("\t") for r in tsv.strip().split("\n") if r]
    best = max((float(r[4]) for r in rows if len(r) > 4), default=0)

    return jsonify({
        "gpu_util":     int(util),
        "mem_used_mb":  int(mem_used),
        "mem_total_mb": int(mem_total),
        "best_map50":   round(best, 4),
        "n_experiments": len(rows),
        "recent_5":     rows[-5:]
    })

if __name__ == "__main__":
    # 本地运行：浏览器打开 http://localhost:8765/api/status
    app.run(port=8765, debug=True)
```

### 运行提示

```bash
# 本地启动监控
python monitor_dashboard.py

# 查看进度（直接在远端执行）
tail -f results.tsv          # 实时追踪实验日志
watch -n 10 cat results.tsv  # 每 10 秒刷新
nvidia-smi dmon -s u -d 5    # GPU 使用率实时监控

# H100 上训练间隙 Agent 在思考，看到无进程是正常的
ps aux | grep python
```

---

## 11. 实验结果汇总

### 里程碑成绩

| 里程碑 | 实验 | mAP@0.5 | crazing AP | 提升来源 |
|--------|------|---------|------------|---------|
| 基线 | baseline_n（15ep） | 0.7294 | 0.305 | — |
| 超参天花板 | conv_n_001（100ep） | 0.7601 | 0.443 | 超参搜索 +4.2% |
| 短跑最优 | loss_n_015（15ep） | 0.7745 | 0.438 | focused-TAL |
| **长跑冠军** | **conv_loss_015（100ep）** | **0.7726** | **0.469** | +1.7% vs 超参 |

- **总提升**：0.729 → 0.773，+5.9%（相对 +6.0%）
- **crazing 类 AP**：0.305 → 0.469（+53.8%）
- 超参搜索贡献前 4.2 个百分点，focused-TAL 贡献最后 1.7 个百分点

### 结构改造 vs Loss 改造对比

| 维度 | 结构改造（Phase 4a） | Loss/Assigner（Phase 4b） |
|------|---------------------|--------------------------|
| 轮数 | 10 | ~30 |
| 正收益 | 0 | 多次 |
| 最优提升 | 无 | +1.7% |
| 推理额外成本 | 有 | **零** |
| 实现复杂度 | 改 YAML 结构定义 | 改 Assigner 分配逻辑 |

---

## 12. 三条可迁移结论

### 结论一：迭代次数要和数据规模匹配

**现象**：短跑（15ep）冠军在长跑（100ep）中反转排名，发生了两次。

**原因**：NEU-DET 只有 1800 张 200×200 灰度图，模型容量超出数据规模时，短跑阶段实际上是在"记忆训练集"，拉长训练就暴露过拟合。

**实践原则**：
- 小数据集（千级别）：短跑快速筛选，Top-N 配置必须做长跑验证
- 大数据集（万级以上）：短跑和长跑排名一致性更高
- **"短跑筛选 + 长跑确认"是标准流程，不是可选步骤**

### 结论二：小数据集上，加模块不如减模块

**现象**：10 轮结构改造（注意力模块、多尺度检测头）全部负收益，最惨跌 24.6%。

**原因**：在 1800 张小图上，YOLOv8n 的 3M 参数已经是模型容量和数据规模之间比较好的平衡点，额外模块引入的参数量超过了带来的表达能力增益。

**判断标准**：训练集 < 5000 张时，先确认基础模型是否已够用。val_loss 后期上升 → 应减模型或加数据，而非堆模块。

### 结论三：改 Loss 是性价比最高的突破口

**现象**：超参搜索天花板在 0.762，结构改造全部失败，focused-TAL 用 30 轮实验把 mAP 从 0.760 推到 0.773。

**核心优势**：Assigner 只在训练时参与计算，部署推理时完全不存在，是**真正的零成本优化**。

**通用推论**：对于检测任务，超参已到位但指标未达目标时，优先顺序应为：
1. 改 Loss / Assigner 策略
2. 改模型结构（慎用，小数据集风险高）

---

## 13. 迁移到其他企业场景

判断一个场景能否套 Karpathy Loop 的核心标准：**能否把四要素说清楚**。

| 条件 | 为什么重要 |
|------|-----------|
| 能定义单一主指标 | 没有主指标，无法自动 keep/discard |
| 单轮实验可自动运行 | 否则 Agent 无法闭环 |
| 可编辑面能被收窄 | 改动范围过大，搜索空间失控 |
| 有一套相对稳定的评测集 | 没有评测集，只是在"盲调" |
| 有明确的业务收益 | 否则调出来也很难落地 |

### 四个优先迁移场景

**场景 1：RAG 检索与上下文组装优化**

| 要素 | 对应 |
|------|------|
| 可编辑文件 | rag_pipeline.py（chunk 策略、rerank、query rewrite） |
| 标量指标 | Recall@10 / MRR / 答案准确率 |
| 固定周期 | 跑完评测集一遍（通常 1~5 分钟） |
| Keep/Discard | 同 YOLO |

优化链路：chunk 策略 → metadata filter → hybrid search → rerank → query rewrite → answer prompt

**场景 2：Prompt / Context 优化**

| 要素 | 对应 |
|------|------|
| 可编辑文件 | prompt_config.py（system prompt、few-shot、上下文拼装） |
| 标量指标 | 任务成功率 / 格式合规率 / 人工评分均值 |
| 固定周期 | 跑完评测集（通常 < 3 分钟） |
| Keep/Discard | 同 YOLO |

适用任务：客服、摘要、结构化生成、Agent 工作流

**场景 3：OCR / 文档抽取流水线优化**

| 要素 | 对应 |
|------|------|
| 可编辑文件 | pipeline.py（预处理、版面分块、字段抽取、后处理规则） |
| 标量指标 | 字段抽取准确率 / F1 |
| 固定周期 | 跑完测试文档集（通常 2~10 分钟） |
| Keep/Discard | 同 YOLO |

**场景 4：代码性能与质量优化**

| 要素 | 对应 |
|------|------|
| 可编辑文件 | 只允许改 1~3 个热点文件（边界必须收紧） |
| 标量指标 | P95 延迟 / 吞吐量 / 测试通过率 |
| 固定周期 | 跑完 benchmark 脚本（通常 1~5 分钟） |
| Keep/Discard | 同 YOLO |

---

## 14. 成本与时间参考

### V2 实验成本分布

| 阶段 | 轮数 | 时长 | H100 成本（约 $3/h） |
|------|------|------|---------------------|
| Phase 1 基线 | 3 | ~30 分钟 | ~$1.5 |
| Phase 2 超参 | ~15 | ~2~3 小时 | ~$9 |
| Phase 3 长跑 | 2 | ~1~2 小时 | ~$5 |
| Phase 4 探索 | ~45 | ~5~6 小时 | ~$18 |
| **合计** | **~64** | **~10 小时** | **~$33** |

### 效率对比

| 指标 | V1（M4 Pro） | V2（H100） |
|------|-------------|-----------|
| 单轮耗时 | ~22 分钟 | ~3~7 分钟 |
| 10 小时实验轮数 | ~25 轮 | ~64 轮 |
| 搜索空间 | 超参数 JSON | 任意代码 |
| mAP 提升 | +5.6% | +6.0% |

### 关键工程建议

1. **数据先落盘**：每轮实验产物（权重、训练曲线、混淆矩阵）先存远端 `artifacts/`，确认后再同步本地
2. **短跑筛选 + 长跑确认**：15ep 快速淘汰，只对 Top-N 做 100ep 长跑，避免在没前途的方向上烧钱
3. **SSH 上看到空进程是正常的**：Agent 在分析历史数据、决策下一步，这个思考时间有时比训练本身还长
4. **分阶段给 Agent 课程表**：全参数空间太大容易迷失，每阶段打开一扇门，充分探索后再进入下一阶段
5. **H100 磁盘管理**：每轮后删除不需要的中间 checkpoint，避免 500GB 磁盘被撑满

---

*文档基于韦东东的实战案例整理，实验环境：NVIDIA H100 80GB + Claude Code，数据集：NEU-DET，模型：YOLOv8n*
