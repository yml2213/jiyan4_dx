from __future__ import annotations

from typing import Tuple
from PIL import Image
import os


def auto_hw_from_ref(ref: str = 'images/1.jpg', multiple: int = 64, default_hw: Tuple[int, int] = (192, 320)) -> Tuple[int, int, int, int]:
    """
    根据参考图片尺寸，返回适配的输入高宽（均为 multiple 的倍数）以及网格尺寸。

    返回: (h, w, grid_w, grid_h)
      - h, w: 输入张量的高和宽
      - grid_w, grid_h: 宽、高方向上的网格数（= w/multiple, h/multiple）
    """
    h, w = default_hw
    grid_w = w // multiple
    grid_h = h // multiple
    try:
        if os.path.exists(ref):
            with Image.open(ref) as im:
                w0, h0 = im.size  # PIL: (width, height)
                grid_w = max(1, round(w0 / multiple))
                grid_h = max(1, round(h0 / multiple))
                w = grid_w * multiple
                h = grid_h * multiple
    except Exception:
        # 读取失败则使用默认尺寸
        pass
    return h, w, grid_w, grid_h

