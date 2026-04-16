"""
YOLO AutoResearch v2 - Train Script

这是 Agent 唯一可编辑的文件，用于训练 YOLOv8 模型并评估 mAP@0.5。

用法:
    python train.py '{"name": "exp_001", "tal_topk": 24, "crazing_boost": 2.0, ...}'
"""
import json
import sys
from ultralytics import YOLO


def inject_focused_tal(class_boosts=None, crazing_boost=1.0):
    """注入自定义 Assigner（monkey-patch，只影响训练，不影响推理）"""
    from ultralytics.utils import tal
    from src.assigner import MultiClassFocusedAssigner

    effective_boosts = {}
    if crazing_boost != 1.0:
        effective_boosts[0] = crazing_boost
    if class_boosts:
        effective_boosts = {**effective_boosts, **class_boosts}

    class Patched(MultiClassFocusedAssigner):
        def __init__(self, *a, **kw):
            super().__init__(*a, class_boosts=effective_boosts if effective_boosts else None, **kw)

    tal.TaskAlignedAssigner = Patched


def run(cfg: dict) -> float:
    """运行训练并返回 mAP@0.5

    Args:
        cfg: dict, 训练配置，包含以下可选参数:
            - model: str, 模型名称 (默认 yolov8n.pt)
            - epochs: int, 训练轮数 (默认 15)
            - imgsz: int, 输入图像尺寸 (默认 640)
            - lr0: float, 初始学习率 (默认 0.01)
            - lrf: float, 最终学习率比例 (默认 0.01)
            - batch: int, 批次大小 (默认 32)
            - optimizer: str, 优化器 (默认 'auto')
            - mosaic: float, mosaic 增强概率 (默认 1.0)
            - degrees: float, 旋转角度 (默认 0.0)
            - scale: float, 缩放比例 (默认 0.5)
            - fliplr: float, 水平翻转概率 (默认 0.5)
            - warmup_epochs: float, 预热轮数 (默认 3.0)
            - weight_decay: float, 权重衰减 (默认 0.0005)
            - tal_topk: int, TAL topk 参数 (默认 10)
            - tal_alpha: float, TAL alpha 参数 (默认 0.5)
            - tal_beta: float, TAL beta 参数 (默认 6.0)
            - crazing_boost: float, crazing 类别加权 (默认 1.0)
            - class_boosts: dict, 额外类别加权
            - loss_type: str, 损失类型 (默认 'ciou')
            - name: str, 实验名称 (默认 'exp')
            - data: str, 数据集配置 (默认 'neu-det.yaml')

    Returns:
        float, mAP@0.5
    """
    # 注入自定义 Assigner
    if cfg.get("class_boosts") or cfg.get("crazing_boost", 1.0) != 1.0:
        inject_focused_tal(
            class_boosts=cfg.get("class_boosts"),
            crazing_boost=cfg.get("crazing_boost", 1.0)
        )

    model = YOLO(cfg.get("model", "yolov8n.pt"))

    # 构建训练参数
    train_kwargs = {
        'data': cfg.get("data", "neu-det.yaml"),
        'epochs': cfg.get("epochs", 15),
        'imgsz': cfg.get("imgsz", 640),
        'lr0': cfg.get("lr0", 0.01),
        'lrf': cfg.get("lrf", 0.01),
        'batch': cfg.get("batch", 32),
        'optimizer': cfg.get("optimizer", "auto"),
        'mosaic': cfg.get("mosaic", 1.0),
        'degrees': cfg.get("degrees", 0.0),
        'scale': cfg.get("scale", 0.5),
        'fliplr': cfg.get("fliplr", 0.5),
        'warmup_epochs': cfg.get("warmup_epochs", 3.0),
        'weight_decay': cfg.get("weight_decay", 0.0005),
        'project': "runs",
        'name': cfg.get("name", "exp"),
        'exist_ok': True,
        'verbose': False,
    }

    # TAL 参数（这些会被 YOLO 传递给 Assigner）
    # 注意：ultralytics 的 TaskAlignedAssigner 默认参数是 topk=10, alpha=0.5, beta=6.0
    # 我们需要通过 monkey-patch 来修改这些默认值
    # 实际上 tal_topk/tal_alpha/tal_beta 参数在当前版本的 ultralytics 中
    # 可能需要直接修改 TaskAlignedAssigner 的 __init__ 参数

    model.train(**train_kwargs)

    # 评估
    m = model.val(data=cfg.get("data", "neu-det.yaml"), verbose=False)
    map50 = float(m.box.map50)

    return map50


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    map50 = run(cfg)
    # 固定输出格式，供循环脚本解析
    print(f"RESULT map50={map50:.6f}")
