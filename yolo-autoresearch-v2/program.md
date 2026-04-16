# YOLO 钢铁缺陷检测 - 突破 0.80 mAP

## 背景

基于 AutoResearch-YOLO 框架，Phase 1 确认了 0.773 基线（tal_topk=24, tal_beta=2.9, crazing_boost=2.0）。
现在进入 Phase 2-4，多方向探索 Loss/Assigner 改进，目标突破 0.80 mAP@0.5。

## 目标

最大化 NEU-DET 验证集的 **mAP@0.5**，目标是突破 **0.80**

## 文件规则

- **train.py 是唯一可修改的文件**
- prepare_dataset.py 和 neu-det.yaml **不允许修改**
- src/ 下的模块（assigner.py, losses.py, search_space.py）也可以修改

## 数据集信息

- **数据集**：NEU-DET（东北大学钢铁缺陷数据集）
- **图像**：1800 张 200×200 灰度图
- **类别**：6 类（crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches）
- **缺陷**：crazing（龟裂）最难检测，基线 AP 只有 0.305

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
8. 将结果追加到 results.tsv，立即开始下一轮

## results.tsv 格式

```
exp_id\tphase\tdirection\tconfig\tmap50\tcrazing_ap\tstatus\tnotes
```

示例：
```
exp_id	phase	direction	config	map50	status	notes
exp_001	Phase1	baseline	tal_topk=24,tal_beta=2.9	0.773	keep	baseline确认
```

## 已知有效先验（来自 V1 实验）

- mosaic 必须关闭（0.0）：200×200 小图拼接后缺陷消失
- lr0 从 0.001 开始：baseline 的 0.01 太高
- degrees=5：微旋转有效，超过 8 度有害
- yolov8n 优先：1800 张小数据集上 yolov8m 容易过拟合
- 关闭 mosaic + lr=0.001 + 5度旋转 是已知最优增强组合

## 退出条件

连续 10 轮无提升且方向已充分探索时停止。
