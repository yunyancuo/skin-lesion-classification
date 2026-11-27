"""测试 models.py：ResNet18 构建与前向输出形状（无 torch 环境自动优雅跳过）。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from models import SUPPORTED_MODELS, build_model  # noqa: E402


def test_build_resnet18_scratch_forward_shape() -> None:
    """验证随机初始化 ResNet18 支持 7 类输出且前向张量形状正确。"""
    model = build_model("resnet18_scratch", num_classes=7)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert tuple(logits.shape) == (2, 7)
    assert torch.isfinite(logits).all()


def test_build_model_invalid_args() -> None:
    """非法模型名与非正数类别数必须抛 ValueError。"""
    assert "resnet18_scratch" in SUPPORTED_MODELS
    assert "resnet18_pretrained" in SUPPORTED_MODELS

    with pytest.raises(ValueError, match="未知模型"):
        build_model("unknown_backbone", num_classes=7)
    with pytest.raises(ValueError, match="正整数"):
        build_model("resnet18_scratch", num_classes=0)
