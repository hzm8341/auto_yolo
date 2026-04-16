"""
Search space definitions for Phase 2 multi-direction exploration
"""
from src.losses import LOSS_REGISTRY


# 方向 A: TAL 参数扩展
TAL_PARAM_SPACE = {
    'tal_topk': [15, 18, 21, 24, 27, 30],
    'tal_alpha': [0.3, 0.4, 0.5, 0.6, 0.7],
    'tal_beta': [2.0, 2.5, 2.9, 3.5, 4.0, 5.0],
}

# 方向 B: 多类别加权
# crazing=0, inclusion=1, patches=2, pitted_surface=3, rolled-in_scale=4, scratches=5
CLASS_BOOST_SPACES = [
    {0: 2.0},                          # 只加 crazing
    {0: 2.5},                          # crazing 更高
    {0: 3.0},                          # crazing 更激进
    {0: 1.5, 4: 1.5},                # crazing + rolled-in_scale
    {0: 2.0, 4: 1.5},                # crazing + rolled-in_scale
    {0: 2.0, 1: 1.3},                 # crazing + inclusion
    {0: 2.0, 1: 1.3, 4: 1.5},        # crazing + inclusion + rolled-in_scale
]

# 方向 C: Loss 类型
LOSS_TYPES = list(LOSS_REGISTRY.keys())

# 方向 D: Focal Loss 变体（已在上面覆盖）
FOCAL_VARIANTS = ['standard', 'focal', 'varifocal']

# 基础配置（Phase 1 复现）
BASELINE_CONFIG = {
    'model': 'yolov8n.pt',
    'epochs': 100,
    'tal_topk': 24,
    'tal_alpha': 0.5,
    'tal_beta': 2.9,
    'crazing_boost': 2.0,
    'class_boosts': None,
    'loss_type': 'ciou',
    'lr0': 0.001,
    'batch': 64,
    'mosaic': 0.0,
    'degrees': 5.0,
    'optimizer': 'AdamW',
}

# 短跑配置（15 epochs，用于快速筛选）
SPRINT_CONFIG = {
    **BASELINE_CONFIG,
    'epochs': 15,
}

# 已知有效配置（来自 AutoResearch-YOLO 文档）
KNOWN_GOOD_CONFIGS = [
    # 长跑冠军
    {
        'name': 'conv_loss_015',
        **BASELINE_CONFIG,
        'epochs': 100,
        'tal_topk': 24,
        'tal_beta': 2.9,
        'crazing_boost': 2.0,
    },
    # 短跑冠军
    {
        'name': 'loss_n_015',
        **SPRINT_CONFIG,
        'tal_topk': 24,
        'tal_beta': 2.9,
        'crazing_boost': 2.0,
    },
]


def generate_tal_combinations():
    """生成 TAL 参数的所有组合"""
    import itertools
    combinations = []
    for topk in TAL_PARAM_SPACE['tal_topk']:
        for alpha in TAL_PARAM_SPACE['tal_alpha']:
            for beta in TAL_PARAM_SPACE['tal_beta']:
                combinations.append({
                    'tal_topk': topk,
                    'tal_alpha': alpha,
                    'tal_beta': beta,
                })
    return combinations


def generate_class_boost_combinations():
    """生成类别加权的组合"""
    return [{'class_boosts': boost} for boost in CLASS_BOOST_SPACES]


def generate_loss_combinations():
    """生成 Loss 类型的组合"""
    return [{'loss_type': loss} for loss in LOSS_TYPES if loss != 'ciou']
