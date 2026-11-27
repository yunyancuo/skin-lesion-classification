"""模型定义：ResNet18 基线（随机初始化）与改进版（ImageNet 预训练微调）。"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

SUPPORTED_MODELS: tuple[str, ...] = ("resnet18_scratch", "resnet18_pretrained")


def build_model(name: str = "resnet18_scratch", num_classes: int = 7) -> nn.Module:
    """构建皮肤病变分类 ResNet18 模型。

    Args:
        name: 模型名称，支持 ``resnet18_scratch`` 与 ``resnet18_pretrained``。
        num_classes: 输出类别数，HAM10000 默认为 ``7``。

    Returns:
        初始化完成的 ``torch.nn.Module`` 实例。

    Raises:
        ValueError: ``num_classes <= 0`` 或 ``name`` 不在支持列表中。
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes 必须为正整数，当前为: {num_classes}")
    if name == "resnet18_scratch":
        return resnet18(weights=None, num_classes=num_classes)
    if name == "resnet18_pretrained":
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    raise ValueError(f"未知模型: {name}（可选: {', '.join(SUPPORTED_MODELS)}）")
