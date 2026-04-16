"""
MultiClassFocus Assigner for YOLO AutoResearch v2

在正样本分配时，给指定类别额外加权，让模型在梯度更新时
更多地学习困难类别的特征。
"""
import torch
from ultralytics.utils.tal import TaskAlignedAssigner


class MultiClassFocusedAssigner(TaskAlignedAssigner):
    """可配置的多类别加权 Assigner

    在正样本分配时，给指定类别额外加权，让模型在梯度更新时
    更多地学习困难类别的特征。

    Args:
        class_boosts: dict, 类别加权字典，格式 {class_id: boost_factor}
                      例如 {0: 2.0, 4: 1.5} 表示对类别 0 加权 2.0x，对类别 4 加权 1.5x
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
    """注入自定义 Assigner（monkey-patch，只影响训练，不影响推理）

    Args:
        class_boosts: dict, 额外的类别加权字典
        crazing_boost: float, crazing(类别0)的加权值，默认为1.0（不加权）
    """
    from ultralytics.utils import tal

    # 合并 crazing_boost 到 class_boosts
    effective_boosts = {}
    if crazing_boost != 1.0:
        effective_boosts[0] = crazing_boost

    if class_boosts:
        # class_boosts 中的值覆盖 crazing_boost（如果有冲突）
        effective_boosts = {**effective_boosts, **class_boosts}

    class Patched(MultiClassFocusedAssigner):
        def __init__(self, *a, **kw):
            super().__init__(*a, class_boosts=effective_boosts if effective_boosts else None, **kw)

    tal.TaskAlignedAssigner = Patched
