"""图像/区域辅助工具。"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def ndarray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """RGB ndarray (H, W, 3) -> QPixmap"""
    if arr is None:
        return QPixmap()
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def crop_region(arr: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    """region=(x, y, w, h) 相对坐标 0~1，返回裁剪后的 ndarray。"""
    if arr is None:
        return arr
    h, w, _ = arr.shape
    x = max(0, min(int(region[0] * w), w - 1))
    y = max(0, min(int(region[1] * h), h - 1))
    rw = max(1, int(region[2] * w))
    rh = max(1, int(region[3] * h))
    rw = min(rw, w - x)
    rh = min(rh, h - y)
    return arr[y : y + rh, x : x + rw, :]


def format_relative_ms(ms: float) -> str:
    val = int(round(ms))
    if val == 0:
        return "0ms"
    if val > 0:
        return f"+{val}ms"
    return f"{val}ms"  # 负号自带


def format_duration(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"
