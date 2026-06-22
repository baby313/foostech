"""视频显示 + 矩形区域选择 widget。"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from .utils import ndarray_to_qpixmap


class VideoCanvas(QWidget):
    """显示当前帧并支持鼠标框选；选区为相对坐标 (x, y, w, h) 0~1。"""

    region_changed = pyqtSignal(object)  # tuple|None
    region_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 360)
        self._pixmap: Optional[QPixmap] = None
        self._frame_size: tuple[int, int] = (0, 0)  # (W, H) of source video frame
        self._region: Optional[tuple[float, float, float, float]] = None
        self._dragging = False
        self._drag_start: Optional[QPoint] = None
        self._drag_end: Optional[QPoint] = None
        self.setStyleSheet("background-color: #0b0d12;")

    # ------------- 数据 ------------
    def set_frame(self, arr: Optional[np.ndarray]):
        if arr is None:
            self._pixmap = None
            self._frame_size = (0, 0)
        else:
            h, w, _ = arr.shape
            self._frame_size = (w, h)
            self._pixmap = ndarray_to_qpixmap(arr)
        self.update()

    def get_region(self) -> Optional[tuple[float, float, float, float]]:
        return self._region

    def clear_region(self):
        self._region = None
        self.region_cleared.emit()
        self.update()

    # ------------- 几何换算 ------------
    def _draw_rect(self) -> QRectF:
        """视频在 widget 中的实际绘制矩形（保持 16:9 比例缩放）。"""
        if not self._pixmap or self._frame_size == (0, 0):
            return QRectF(0, 0, 0, 0)
        ww, wh = self.width(), self.height()
        fw, fh = self._frame_size
        scale = min(ww / fw, wh / fh)
        dw, dh = fw * scale, fh * scale
        dx = (ww - dw) / 2
        dy = (wh - dh) / 2
        return QRectF(dx, dy, dw, dh)

    def _widget_to_relative(self, pos: QPoint) -> Optional[tuple[float, float]]:
        r = self._draw_rect()
        if r.width() <= 0:
            return None
        x = (pos.x() - r.x()) / r.width()
        y = (pos.y() - r.y()) / r.height()
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        return (x, y)

    def _relative_to_widget(self, rel: tuple[float, float, float, float]) -> QRectF:
        r = self._draw_rect()
        rx, ry, rw, rh = rel
        return QRectF(r.x() + rx * r.width(), r.y() + ry * r.height(), rw * r.width(), rh * r.height())

    # ------------- 鼠标事件 ------------
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._drag_end = self._drag_start
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.clear_region()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            self._drag_end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            if self._drag_start and self._drag_end:
                p1 = self._widget_to_relative(self._drag_start)
                p2 = self._widget_to_relative(self._drag_end)
                if p1 and p2:
                    x = min(p1[0], p2[0])
                    y = min(p1[1], p2[1])
                    w = abs(p2[0] - p1[0])
                    h = abs(p2[1] - p1[1])
                    if w >= 0.01 and h >= 0.01:
                        self._region = (x, y, w, h)
                        self.region_changed.emit(self._region)
                    else:
                        # 太小当作清除
                        self.clear_region()
            self._drag_start = None
            self._drag_end = None
            self.update()

    # ------------- 绘制 ------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b0d12"))

        if self._pixmap is None:
            painter.setPen(QColor("#5b6478"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "请打开视频文件")
            return

        rect = self._draw_rect()
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))

        # 绘制确认的选区
        if self._region is not None:
            pen = QPen(QColor("#22d3ee"), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(34, 211, 238, 36))
            painter.drawRect(self._relative_to_widget(self._region))

        # 绘制拖拽中的选框
        if self._dragging and self._drag_start and self._drag_end:
            pen = QPen(QColor("#3b82f6"), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(59, 130, 246, 28))
            painter.drawRect(QRect(self._drag_start, self._drag_end).normalized())
