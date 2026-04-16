"""
Loss functions for YOLO AutoResearch v2

包含 SIoU、EIoU、Varifocal Loss 等改进的损失函数实现。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIoULoss(nn.Module):
    """Shape-IoU Loss

    考虑边框的形状相似性，而不仅仅是几何距离。
    论文: https://arxiv.org/abs/2205.12740
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """计算 SIoU Loss

        Args:
            pred: [N, 4] 预测边框 (x1, y1, x2, y2)
            target: [N, 4] 目标边框 (x1, y1, x2, y2)

        Returns:
            scalar loss
        """
        # 中心点
        pred_xy = (pred[:, :2] + pred[:, 2:]) / 2
        target_xy = (target[:, :2] + target[:, 2:]) / 2

        # 宽高
        pred_wh = pred[:, 2:] - pred[:, :2]
        target_wh = target[:, 2:] - target[:, :2]

        # 距离损失
        d_xy = (pred_xy - target_xy) ** 2
        d_diag = d_xy.sum(dim=1)

        # 形状损失
        v_w = torch.abs(pred_wh[:, 0] - target_wh[:, 0])
        v_h = torch.abs(pred_wh[:, 1] - target_wh[:, 1])
        v = (4 / torch.pi ** 2) * (
            torch.atan(target_wh[:, 0] / (target_wh[:, 1] + 1e-8)) -
            torch.atan(pred_wh[:, 0] / (pred_wh[:, 1] + 1e-8))
        ) ** 2

        alpha = v / ((1 - torch.exp(-v)) + 1e-8)

        siou = 1 - torch.exp(-d_diag) + alpha * v
        return siou.mean()


class EIoULoss(nn.Module):
    """Efficient-IoU Loss

    分别考虑重叠面积、距离、形状三个因素。
    论文: https://arxiv.org/abs/2101.08158
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """计算 EIoU Loss

        Args:
            pred: [N, 4] 预测边框 (x1, y1, x2, y2)
            target: [N, 4] 目标边框 (x1, y1, x2, y2)

        Returns:
            scalar loss
        """
        # 面积
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

        # 距离损失（对角线距离）
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        d_diag = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # 形状损失
        v_w = torch.abs(pred[:, 2] - pred[:, 0] - (target[:, 2] - target[:, 0]))
        v_h = torch.abs(pred[:, 3] - pred[:, 1] - (target[:, 3] - target[:, 1]))
        v = (4 / torch.pi ** 2) * (
            torch.atan((target[:, 2] - target[:, 0]) / (target[:, 3] - target[:, 1] + 1e-8)) -
            torch.atan((pred[:, 2] - pred[:, 0]) / (pred[:, 3] - pred[:, 1] + 1e-8))
        ) ** 2

        alpha = v / ((1 - torch.exp(-v)) + 1e-8)

        # 假设图像尺寸为 640
        eiou = iou - alpha * v - d_diag / (640 ** 2)
        return (1 - eiou).mean()


class FocalLoss(nn.Module):
    """Focal Loss

    用于处理类别不平衡的损失函数。
    论文: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """计算 Focal Loss

        Args:
            pred: [N, 1] 预测分数 (logits)
            target: [N, 1] 目标分数 (0 or 1)

        Returns:
            scalar loss
        """
        pred_prob = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        p_t = torch.where(target == 1, pred_prob, 1 - pred_prob)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha >= 0:
            alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class VarifocalLoss(nn.Module):
    """Varifocal Loss

    用于目标检测的聚焦损失函数，对正负样本使用不同策略。
    论文: https://arxiv.org/abs/2008.13367
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """计算 Varifocal Loss

        Args:
            pred: [N, 1] 预测分数
            target: [N, 1] 目标分数 (IoU-aware labels)

        Returns:
            scalar loss
        """
        pred = pred.clamp(1e-8, 1 - 1e-8)
        loss = self.alpha * target * torch.log(pred) + \
               (1 - self.alpha) * (1 - target) * torch.log(1 - pred)
        loss = -loss

        # 聚焦因子
        pred_prob = torch.where(target >= 0.5, pred, 1 - pred)
        focal_weight = torch.pow(1 - pred_prob, self.gamma)

        return (focal_weight * loss).mean()


# Loss 名称到类的映射
LOSS_REGISTRY = {
    'ciou': None,  # 默认使用 YOLO 内置的 CIoU
    'siou': SIoULoss,
    'eiou': EIoULoss,
    'focal': FocalLoss,
    'varifocal': VarifocalLoss,
}


def get_loss(name):
    """获取指定名称的损失函数

    Args:
        name: str, 损失函数名称

    Returns:
        nn.Module 实例，如果 name 是 'ciou' 则返回 None（使用默认）
    """
    if name in LOSS_REGISTRY:
        cls = LOSS_REGISTRY[name]
        return cls() if cls else None
    raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
