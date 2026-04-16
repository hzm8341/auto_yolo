"""
Test for loss functions
"""
import torch
import sys
sys.path.insert(0, '/media/hzm/Data/auto_yolo/yolo-autoresearch-v2')

from src.losses import SIoULoss, EIoULoss, FocalLoss, VarifocalLoss, get_loss


def test_siou():
    pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    loss_fn = SIoULoss()
    loss = loss_fn(pred, target)
    assert loss.item() >= 0, f"SIoU loss should be non-negative, got {loss.item()}"
    print(f"✓ SIoU loss (identical boxes): {loss.item():.6f}")

    # Test with different boxes
    pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target = torch.tensor([[2, 2, 12, 12]], dtype=torch.float32)
    loss = loss_fn(pred, target)
    assert loss.item() >= 0, f"SIoU loss should be non-negative, got {loss.item()}"
    print(f"✓ SIoU loss (offset boxes): {loss.item():.6f}")


def test_eiou():
    pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    loss_fn = EIoULoss()
    loss = loss_fn(pred, target)
    assert loss.item() >= 0, f"EIoU loss should be non-negative, got {loss.item()}"
    print(f"✓ EIoU loss (identical boxes): {loss.item():.6f}")

    # Test with different boxes
    pred = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target = torch.tensor([[5, 5, 15, 15]], dtype=torch.float32)
    loss = loss_fn(pred, target)
    assert loss.item() >= 0, f"EIoU loss should be non-negative, got {loss.item()}"
    print(f"✓ EIoU loss (offset boxes): {loss.item():.6f}")


def test_focal():
    pred = torch.randn(10, 1)
    target = torch.randint(0, 2, (10, 1)).float()
    loss_fn = FocalLoss()
    loss = loss_fn(pred, target)
    assert loss.item() >= 0, f"Focal loss should be non-negative, got {loss.item()}"
    print(f"✓ Focal loss: {loss.item():.6f}")


def test_varifocal():
    pred = torch.randn(10, 1).clamp(1e-6, 1 - 1e-6)
    target = torch.rand(10, 1)
    loss_fn = VarifocalLoss()
    loss = loss_fn(pred, target)
    assert loss.item() >= 0, f"Varifocal loss should be non-negative, got {loss.item()}"
    print(f"✓ Varifocal loss: {loss.item():.6f}")


def test_get_loss():
    assert get_loss('siou') is not None
    assert get_loss('eiou') is not None
    assert get_loss('focal') is not None
    assert get_loss('varifocal') is not None
    assert get_loss('ciou') is None  # ciou returns None (use default)
    print("✓ get_loss works for all loss types")

    try:
        get_loss('unknown')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ get_loss raises ValueError for unknown loss: {e}")


if __name__ == "__main__":
    test_siou()
    test_eiou()
    test_focal()
    test_varifocal()
    test_get_loss()
    print("\nAll loss tests passed!")
