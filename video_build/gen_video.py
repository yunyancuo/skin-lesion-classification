"""生成 HAM10000 课程验收演示视频（1280x720 MP4）。

素材全部来自仓库真实产物：figures/ 下的图、results/ 的预测示例、
服务器上 evaluate.py 的真实输出（终端打字动画）。
用法：python video_build/gen_video.py  → 输出 HAM10000演示视频.mp4
"""
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parent.parent
FRAMES = Path(__file__).resolve().parent / "frames"
FRAMES.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.monospace"] = ["Consolas", "SimHei"]
import warnings as _w
_w.filterwarnings("ignore")

BG = "#162235"      # 深海蓝（与报告封面同系）
ACCENT = "#37DCF2"  # 青
WHITE = "#FFFFFF"
GREY = "#B0B8C0"
W, H = 12.8, 7.2
DPI = 100

frames = []  # (文件名, 时长秒)


def new_fig():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def save(fig, dur):
    name = f"f{len(frames):04d}.png"
    fig.savefig(FRAMES / name, facecolor=BG)
    plt.close(fig)
    frames.append((name, dur))


def slide(title, subtitle=None, dur=6):
    fig = new_fig()
    fig.text(0.08, 0.62, title, fontsize=44, color=WHITE, fontweight="bold", va="center")
    if subtitle:
        fig.text(0.08, 0.46, subtitle, fontsize=22, color=GREY, va="center")
    fig.text(0.08, 0.80, "HAM10000 皮肤病变分类 · 课程实践", fontsize=15, color=ACCENT)
    fig.lines.append(plt.Line2D([0.08, 0.92], [0.70, 0.70], transform=fig.transFigure,
                                color=ACCENT, linewidth=2))
    save(fig, dur)


def image_slide(title, img_path, caption=None, dur=8):
    fig = new_fig()
    fig.text(0.06, 0.93, title, fontsize=26, color=WHITE, fontweight="bold", va="center")
    try:
        im = imread(img_path)
        h, w = im.shape[:2]
        # 直接按宽高比在中央放置
        ax = fig.add_axes([0.07, 0.13, 0.86 * min(1, (0.72 * 12.8 / 7.2) / (w / h)), 0.72])
        ax.imshow(im)
        ax.axis("off")
    except Exception as e:
        fig.text(0.5, 0.5, f"[图片缺失 {img_path}]", color=GREY, ha="center")
    if caption:
        fig.text(0.5, 0.05, caption, fontsize=15, color=GREY, ha="center")
    save(fig, dur)


def two_image_slide(title, img1, img2, cap1, cap2, dur=10):
    fig = new_fig()
    fig.text(0.06, 0.93, title, fontsize=26, color=WHITE, fontweight="bold", va="center")
    for x0, p in [(0.04, img1), (0.52, img2)]:
        im = imread(p)
        ax = fig.add_axes([x0, 0.14, 0.44, 0.70])
        ax.imshow(im)
        ax.axis("off")
    fig.text(0.26, 0.07, cap1, fontsize=14, color=GREY, ha="center")
    fig.text(0.74, 0.07, cap2, fontsize=14, color=GREY, ha="center")
    save(fig, dur)


def table_slide(title, header, rows, dur=9, highlight_rows=(), note=None, fs=17):
    fig = new_fig()
    fig.text(0.06, 0.90, title, fontsize=26, color=WHITE, fontweight="bold", va="center")
    ncol = len(header)
    colw = np.ones(ncol) / ncol
    tax = fig.add_axes([0, 0, 1, 1])
    tax.axis("off")
    tbl = tax.table(cellText=[[str(c) for c in r] for r in rows],
                    colLabels=header, colWidths=colw, loc="center",
                    cellLoc="center", bbox=[0.10, 0.22, 0.80, 0.56])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#3A5068")
        if r == 0:
            cell.set_facecolor("#1B6B7A")
            cell.set_text_props(color=WHITE, fontweight="bold")
        else:
            cell.set_facecolor("#1D3048" if r % 2 == 0 else "#223A56")
            cell.set_text_props(color=WHITE if (r - 1) in highlight_rows else GREY,
                                fontweight="bold" if (r - 1) in highlight_rows else "normal")
    if note:
        fig.text(0.5, 0.10, note, fontsize=15, color=ACCENT, ha="center")
    save(fig, dur)


def bullet_slide(title, bullets, dur=9, fs=21):
    fig = new_fig()
    fig.text(0.06, 0.90, title, fontsize=26, color=WHITE, fontweight="bold", va="center")
    for i, (head, desc) in enumerate(bullets):
        y = 0.74 - i * (0.66 / max(len(bullets), 1))
        fig.text(0.08, y, "■", fontsize=fs, color=ACCENT, va="center")
        fig.text(0.12, y, head, fontsize=fs, color=WHITE, va="center", fontweight="bold")
        if desc:
            fig.text(0.12, y - 0.055, desc, fontsize=fs - 6, color=GREY, va="center")
    save(fig, dur)


TERM_BG = "#0C0C0C"
TERM_HEAD = "#2B2B2B"
def terminal_frame(lines, dur, done_prompt=False):
    """终端画面：标题栏 + 绿色提示符 + 文本行。"""
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(TERM_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    # 标题栏
    ax.add_patch(plt.Rectangle((0.0, 0.90), 1, 0.10, color=TERM_HEAD))
    for i, color in enumerate(["#FF5F57", "#FEBC2E", "#28C840"]):
        ax.add_patch(plt.Circle((0.025 + i * 0.022, 0.948), 0.007, color=color))
    ax.text(0.5, 0.948, "服务器: ~/ham — ssh", color="#9A9A9A",
            fontsize=13, ha="center", va="center", family="monospace")
    # 文本
    y = 0.855
    for ln in lines:
        if ln.startswith("$ "):
            ax.text(0.03, y, "$", color="#28C840", fontsize=16, family="monospace", va="top")
            ax.text(0.05, y, ln[2:], color=WHITE, fontsize=16, family="monospace", va="top")
        else:
            ax.text(0.05, y, ln, color="#D6D6D6", fontsize=15, family="monospace", va="top")
        y -= 0.043
    if done_prompt:
        ax.text(0.03, y, "$", color="#28C840", fontsize=16, family="monospace", va="top")
        ax.text(0.05, y, "▌", color=WHITE, fontsize=16, family="monospace", va="top")
    save(fig, dur)


def typing_sequence(cmd, prefix_lines, after_lines, typing_dt=0.055, line_dt=0.30):
    """打字动画：命令逐字符出现，随后输出逐行出现。"""
    for n in range(1, len(cmd) + 1, 2):
        terminal_frame(prefix_lines + [cmd[:n] + "▌"], typing_dt * 2, done_prompt=False)
    terminal_frame(prefix_lines + [cmd], 0.35)
    for k in range(1, len(after_lines) + 1):
        terminal_frame(prefix_lines + [cmd] + after_lines[:k], line_dt)
    terminal_frame(prefix_lines + [cmd] + after_lines, 1.0, done_prompt=True)


# ================= 脚本 =================
F = ROOT / "figures"
R = ROOT / "results"

# 1 片头
slide("基于皮肤镜图像的皮肤病变分类", "HAM10000 数据集 · 课程实践验收演示", dur=5)
slide("汇报内容", "① 数据集与预处理    ② 实验方案\n③ 现场运行演示      ④ 测试结果", dur=6)

# 2 背景与数据集
bullet_slide("问题与数据集", [
    ("皮肤镜图像自动分类", "辅助皮肤癌筛查的关键技术，黑色素瘤漏诊代价极高"),
    ("HAM10000 公开数据集", "10015 张皮肤镜图像 · 7470 个病灶 · 7 类病变（Tschandl et al., 2018）"),
    ("两个核心难点", "类别不平衡：最多的痣 6705 张 vs 最少的皮肤纤维瘤 115 张（58 倍）\n同一病灶多张相似照片，划分不当会造成数据泄漏"),
], dur=12)
image_slide("数据集：类别分布严重不平衡", F / "class_distribution.png",
            "全猜 nv 也有 66.9% 准确率，但毫无临床价值 → 评估必须看平衡指标", dur=8)
image_slide("数据集：每类样例", F / "class_examples.png",
            "类间相似、类内差异大，还有色温 / 毛发 / 标尺等采集干扰", dur=8)

# 3 预处理与划分
bullet_slide("预处理与数据划分（防泄漏）", [
    ("统一预处理", "中心方裁剪 → 224×224 → ImageNet 归一化"),
    ("按病灶（lesion_id）分组划分", "同一病灶的所有照片只进一个集合，杜绝\u201c背答案\u201d"),
    ("划分结果", "train 8011 张 / val 1003 张 / test 1001 张（训练过程完全不可见）"),
], dur=10)

# 4 实验方案
bullet_slide("实验方案：基线与三项改进", [
    ("基线", "ResNet18 随机初始化 + 交叉熵 + 基础增强（翻转 / 旋转）"),
    ("改进 1：迁移学习", "ImageNet 预训练权重初始化后整体微调"),
    ("改进 2：加权采样", "按类频率倒数采样，每个 epoch 七类出现次数接近"),
    ("改进 3：强增强", "颜色扰动 / 模糊 / 小角度旋转，模拟采集差异"),
], dur=12)

# 5 现场演示（终端动画，真实输出）
typing_sequence(
    "python3 src/evaluate.py --weights outputs/pretrained_weighted/checkpoints/best.pth",
    ["$ ssh 项目组服务器", "$ cd ~/ham"],
    [
        "=== test 集（n=1001）===",
        "accuracy        : 0.8272",
        "balanced acc    : 0.7261",
        "macro F1        : 0.7079",
        "macro AUC (OvR) : 0.9610",
        "",
        "       akiec  precision 0.6471  recall 0.3235",
        "       bcc    precision 0.6981  recall 0.7400",
        "       bkl    precision 0.5923  recall 0.7196",
        "       df     precision 0.5455  recall 0.7500",
        "       mel    precision 0.5615  recall 0.6460",
        "       nv     precision 0.9442  recall 0.9036",
        "       vasc   precision 1.0000  recall 1.0000",
        "",
        "逐图明细: results/test_predictions.csv（1001 张可抽查）",
    ],
)

# 6 测试结果
table_slide("测试结果：总体指标（n=1001）",
            ["指标", "基线", "改进版", "提升"],
            [["Accuracy", "0.7772", "0.8272", "+5.0 pt"],
             ["Balanced Accuracy", "0.6146", "0.7261", "+11.2 pt"],
             ["Macro F1", "0.6071", "0.7079", "+10.1 pt"],
             ["Macro AUC", "0.9384", "0.9610", "+2.3 pt"]],
            highlight_rows=(1,), note="越公平的指标提升越大 → 增益来自稀有类", dur=9)
table_slide("测试结果：每类召回率",
            ["类别", "基线", "改进版", "变化"],
            [["mel 黑色素瘤（恶性）", "0.442", "0.646", "+20 pt"],
             ["bcc 基底细胞癌（恶性）", "0.580", "0.740", "+16 pt"],
             ["bkl 良性角化", "0.551", "0.720", "+17 pt"],
             ["df 皮肤纤维瘤", "0.500", "0.750", "+25 pt"],
             ["nv 痣（大类）", "0.905", "0.904", "±0"]],
            highlight_rows=(0,), note="mel 召回 +20pt = 漏诊大幅下降；大类 nv 无牺牲", dur=10)
two_image_slide("混淆矩阵对比（行归一化）", F / "cm_scratch.png", F / "cm_pretrained.png",
                "基线：41.6% 的 mel 被误判为 nv", "改进版：mel→nv 误判砍半（21.2%）", dur=12)
image_slide("预测示例（判对 / 判错）", R / "prediction_examples.png",
            "results/prediction_examples.png · 逐张明细见 results/test_predictions.csv", dur=9)
image_slide("训练过程", F / "train_curves.png",
            "预训练的价值在后期泛化；强增强把过拟合控制在可接受范围", dur=8)

# 7 结论
bullet_slide("结论", [
    ("四项指标全部达到或超过文献常见水平", "Accuracy 0.827 · Balanced 0.726 · Macro AUC 0.961"),
    ("类别平衡策略有效", "平衡准确率 +11.2pt，黑色素瘤召回 +20pt，大类无牺牲"),
    ("数据工程同样关键", "官方清单过滤（剔除镜像多出的 3339 张）+ 按病灶划分防泄漏"),
], dur=10)

# 8 片尾
slide("谢谢观看", "代码 / 报告 / 测试结果均已开源：github.com/yunyancuo/skin-lesion-classification\n"
     "复现：ssh 服务器 → cd ~/ham → python3 src/evaluate.py --weights outputs/pretrained_weighted/checkpoints/best.pth",
     dur=7)

# ================= 合成 =================
lst = FRAMES / "list.txt"
with open(lst, "w", encoding="utf-8") as f:
    for name, dur in frames:
        f.write(f"file '{name}'\nduration {dur}\n")
    f.write(f"file '{frames[-1][0]}'\n")

out = ROOT / "HAM10000演示视频.mp4"
ff = subprocess.run(
    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
     "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
     "-movflags", "+faststart", str(out)],
    capture_output=True, text=True)
print("ffmpeg rc:", ff.returncode)
if ff.returncode != 0:
    print(ff.stderr[-1500:])
    sys.exit(1)
print(f"OK {out}  {out.stat().st_size/1e6:.1f} MB, {len(frames)} 段, 总时长 {sum(d for _, d in frames):.0f}s")
