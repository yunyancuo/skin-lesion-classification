"""模型：ResNet18 基线（随机初始化）与改进版（ImageNet 预训练微调）。"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(name="resnet18_scratch", num_classes=7):
    if name == "resnet18_scratch":
        net = resnet18(weights=None, num_classes=num_classes)
        return net
    if name == "resnet18_pretrained":
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    raise ValueError(f"未知模型: {name}")
