"""
Test for MultiClassFocusedAssigner
"""
import torch
import sys
sys.path.insert(0, '/media/hzm/Data/auto_yolo/yolo-autoresearch-v2')

from src.assigner import MultiClassFocusedAssigner, inject_focused_tal


def test_class_boosts():
    """测试多类别加权功能"""
    from ultralytics.utils.tal import TaskAlignedAssigner

    # 创建测试数据
    torch.manual_seed(42)
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

    # 验证：crazing(0) 的 GT 应该有更高的 metric
    gt_is_crazing = gt_labels.squeeze(-1).eq(0)

    print(f"GT labels: {gt_labels.squeeze(-1)}")
    print(f"Crazing GT indices: {gt_is_crazing.nonzero().squeeze(-1).tolist()}")

    # 计算 crazing 的 metric 比例
    if gt_is_crazing.any():
        crazing_idx = gt_is_crazing.nonzero().squeeze(-1)
        base_crazing_metric = metrics_base[crazing_idx].mean()
        boosted_crazing_metric = metrics_boosted[crazing_idx].mean()
        ratio = boosted_crazing_metric / (base_crazing_metric + 1e-8)
        print(f"Base crazing metric: {base_crazing_metric:.4f}")
        print(f"Boosted crazing metric: {boosted_crazing_metric:.4f}")
        print(f"Boost ratio: {ratio:.4f}")
        assert 1.9 <= ratio <= 2.1, f"Expected ratio ~2.0, got {ratio:.4f}"
        print("✓ Test passed: MultiClassFocusAssigner boosts crazing class by 2x")

    # 验证非 crazing 类没有被 boost
    non_crazing_idx = (~gt_is_crazing).nonzero().squeeze(-1)
    if non_crazing_idx.numel() > 0:
        base_non_crazing = metrics_base[non_crazing_idx].mean()
        boosted_non_crazing = metrics_boosted[non_crazing_idx].mean()
        ratio_non = boosted_non_crazing / (base_non_crazing + 1e-8)
        print(f"Non-crazing ratio: {ratio_non:.4f}")
        assert 0.99 <= ratio_non <= 1.01, f"Non-crazing should not be boosted, got ratio {ratio_non:.4f}"
        print("✓ Test passed: Non-crazing classes are not boosted")


def test_inject_focused_tal():
    """测试 monkey-patch 注入功能"""
    from ultralytics.utils import tal

    # 保存原始 Assigner
    OriginalAssigner = tal.TaskAlignedAssigner

    # 注入
    inject_focused_tal(crazing_boost=2.5)

    # 验证被替换了
    assert tal.TaskAlignedAssigner != OriginalAssigner
    print("✓ Test passed: inject_focused_tal replaces TaskAlignedAssigner")

    # 恢复（通过再次注入空的）
    inject_focused_tal(crazing_boost=1.0)


if __name__ == "__main__":
    test_class_boosts()
    test_inject_focused_tal()
    print("\nAll tests passed!")
