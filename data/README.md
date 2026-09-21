# 数据集放置说明

HAM10000 数据集（约 2.6GB）不入库。下载后放到 `data/ham10000.zip`：

- Kaggle：<https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>（推荐，含全部原图 + metadata）
- Harvard Dataverse：HAM10000 官方发布页（Tschandl et al., 2018）

然后运行：

```bash
python src/prepare.py --zip data/ham10000.zip --out-dir data/ham10000
```

会生成 `images/*.jpg` 与 `splits.csv`（train/val/test 按 lesion_id 分层划分）。
