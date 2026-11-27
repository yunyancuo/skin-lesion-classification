"""统一中文字体设置：优先使用项目自带 SimHei，回退各平台常见中文字体。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

DEFAULT_FONT_PATH = Path(__file__).resolve().parent.parent / "fonts" / "simhei.ttf"
FALLBACK_FONTS = [
    "SimHei",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "PingFang SC",
    "DejaVu Sans",
]


def configure_cn_font(font_path: str | Path | None = None) -> list[str]:
    """配置 Matplotlib 中文字体及负号显示。

    Args:
        font_path: 可选的自定义字体文件路径，默认检查项目根目录下的 ``fonts/simhei.ttf``。

    Returns:
        最终写入 ``matplotlib.rcParams["font.sans-serif"]`` 的字体优先列表。
    """
    target = Path(font_path) if font_path is not None else DEFAULT_FONT_PATH
    if target.is_file():
        font_manager.fontManager.addfont(str(target))
        fonts = list(FALLBACK_FONTS)
    else:
        fonts = [f for f in FALLBACK_FONTS if f != "SimHei"] + ["SimHei"]

    matplotlib.rcParams["font.sans-serif"] = fonts
    matplotlib.rcParams["axes.unicode_minus"] = False
    return fonts


# 保持原有 `import cn_font` 即生效的向后兼容行为
configure_cn_font()
