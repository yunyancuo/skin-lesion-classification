# 基于皮肤镜图像的皮肤病变分类（HAM10000）

课程实践仓库 · 选题：基于皮肤镜图像的皮肤病变分类

- **数据集**：HAM10000，10015 张 600×450 皮肤镜图像，7 类病变
- **基线模型**：ResNet18（随机初始化）+ 交叉熵 + 基础增强
- **改进模型**：ResNet18（ImageNet 预训练微调）+ 类别加权采样 + 强增强
- **核心难点**：类别严重不平衡（痣 6705 张 vs 皮肤纤维瘤 115 张），用平衡准确率 / 每类召回评估

## 目录结构

```text
skin-lesion-classification/
├── data/
│   └── README.md        # 数据下载与放置说明（数据不入库）
├── src/
│   ├── prepare.py       # 解包 + 按 lesion_id 分层划分（防数据泄漏）
│   ├── dataset.py       # Dataset：中心裁剪/缩放/增强/归一化
│   ├── models.py        # ResNet18 scratch / pretrained
│   ├── train.py         # 训练（支持加权采样，按 val acc 存最优）
│   ├── evaluate.py      # 测试评估 + 混淆矩阵 + 每类报告
│   └── eda.py           # 类别分布图 / 每类样例图
└── outputs/             # 训练/评估产物（gitignore）
```

## 快速开始

数据获取有两条路，任选其一：

```bash
# 路 A（本次实验实际使用）：HF 镜像 parquet → 还原 + 官方清单过滤 + 病灶分组划分
#   先把 marmal88/skin_cancer 的 8 个 parquet 与 HAM10000_metadata.csv 放到 data_src/
python src/parquet_extract.py --src-dir data_src --out-dir data/ham10000

# 路 B：Kaggle 原始 zip（kmader/skin-cancer-mnist-ham10000，约 5.2GB）
python src/prepare.py --zip data/ham10000.zip --out-dir data/ham10000
```

```bash
# EDA 出图
python src/eda.py

# 训练基线与改进
python src/train.py --model resnet18_scratch --augment base --epochs 30
python src/train.py --model resnet18_pretrained --weighted --augment strong --epochs 30

# 测试集评估
python src/evaluate.py --weights outputs/scratch/checkpoints/best.pth
python src/evaluate.py --weights outputs/pretrained_weighted/checkpoints/best.pth

# 训练曲线
python src/plot_curves.py --logs outputs/scratch/train_log.csv outputs/pretrained_weighted/train_log.csv \
    --names "ResNet18 (scratch)" "ResNet18 (pretrained+weighted)" --out outputs/figures/train_curves.png
```

注意：ImageNet 预训练权重需要 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
（download.pytorch.org 国内不通时，可从 GitHub 上转存的老版 `resnet18-5c106cde.pth`
下载后 `torch.load(..., weights_only=False)` 读入再 `torch.save` 重存为新格式）。

## 参考水平（HAM10000 文献常见值）

| 指标 | ResNet 级模型常见区间 |
|---|---|
| Accuracy | 0.83 ~ 0.87 |
| Balanced Accuracy | 0.65 ~ 0.75 |
| Macro AUC | 0.91 ~ 0.94 |
