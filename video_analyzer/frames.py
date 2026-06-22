"""30 帧预览网格 + 单帧放大查看器。"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .utils import format_relative_ms, ndarray_to_qpixmap


class FrameCell(QWidget):
    """单个帧缩略图：图片 + 左下角时间戳 + 是否当前帧高亮。"""

    clicked = pyqtSignal(int)  # rel_index

    def __init__(self, rel_index: int, parent=None):
        super().__init__(parent)
        self.rel_index = rel_index
        self.rel_ms: float = 0.0
        self._pixmap: Optional[QPixmap] = None
        self._is_current = rel_index == 0
        self.setMinimumSize(QSize(150, 90))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_data(self, arr: Optional[np.ndarray], rel_ms: float):
        self.rel_ms = rel_ms
        self._pixmap = ndarray_to_qpixmap(arr) if arr is not None else None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.rel_index)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.rect()
        # 背景
        p.fillRect(rect, QColor("#11141b"))
        # 图片 letterbox
        if self._pixmap and not self._pixmap.isNull():
            pw, ph = self._pixmap.width(), self._pixmap.height()
            ww, wh = rect.width(), rect.height()
            if pw > 0 and ph > 0:
                scale = min(ww / pw, wh / ph)
                dw, dh = int(pw * scale), int(ph * scale)
                dx = (ww - dw) // 2
                dy = (wh - dh) // 2
                p.drawPixmap(dx, dy, dw, dh, self._pixmap)
        else:
            p.setPen(QColor("#3a4255"))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "—")

        # 边框
        if self._is_current:
            pen = QPen(QColor("#22d3ee"), 2)
        else:
            pen = QPen(QColor("#1f2533"), 1)
        p.setPen(pen)
        p.drawRect(rect.adjusted(0, 0, -1, -1))

        # 左下角时间戳 chip
        text = format_relative_ms(self.rel_ms)
        font = QFont("Menlo", 10)
        font.setBold(self._is_current)
        p.setFont(font)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text) + 10
        th = fm.height() + 4
        chip_x, chip_y = 4, rect.height() - th - 4
        p.setBrush(QColor(0, 0, 0, 170))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(chip_x, chip_y, tw, th, 4, 4)
        if self._is_current:
            p.setPen(QColor("#22d3ee"))
        elif self.rel_ms < 0:
            p.setPen(QColor("#9aa3b8"))
        else:
            p.setPen(QColor("#ffffff"))
        p.drawText(chip_x + 5, chip_y + th - 5, text)


class FrameGrid(QWidget):
    """5 行 6 列 = 30 帧。"""

    cell_clicked = pyqtSignal(int)  # rel_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cells: list[FrameCell] = []
        layout = QGridLayout(self)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        # 30 帧：rel_index from -15 to +14
        rel_indices = list(range(-15, 15))
        for i, rel in enumerate(rel_indices):
            cell = FrameCell(rel)
            cell.clicked.connect(self.cell_clicked.emit)
            layout.addWidget(cell, i // 6, i % 6)
            self.cells.append(cell)

    def update_frames(self, items: list[tuple[int, float, Optional[np.ndarray]]]):
        # items: [(rel, rel_ms, arr)]
        rel_to_item = {it[0]: it for it in items}
        for cell in self.cells:
            it = rel_to_item.get(cell.rel_index)
            if it is None:
                cell.set_data(None, cell.rel_index * 0.0)
            else:
                cell.set_data(it[2], it[1])

    def clear(self):
        for cell in self.cells:
            cell.set_data(None, cell.rel_index * 0.0)


class FrameViewer(QDialog):
    """单帧放大查看器。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("帧详情")
        self.setModal(True)
        self.resize(960, 600)
        self.setStyleSheet("background-color: #0b0d12;")

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: #0b0d12;")

        self.info = QLabel(self)
        self.info.setStyleSheet("color: #cbd2e0; font-family: Menlo; padding: 6px;")

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet(
            "QPushButton { background:#1f2533; color:#e4e8f1; padding:6px 16px; border:1px solid #2c3346; }"
            " QPushButton:hover { background:#262d3f; }"
        )

        bottom = QHBoxLayout()
        bottom.addWidget(self.info)
        bottom.addStretch(1)
        bottom.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.label, 1)
        layout.addLayout(bottom)

        self._pixmap: Optional[QPixmap] = None

    def show_frame(self, arr: Optional[np.ndarray], info_text: str):
        self._pixmap = ndarray_to_qpixmap(arr) if arr is not None else None
        self.info.setText(info_text)
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if not self._pixmap:
            self.label.clear()
            return
        sz = self.label.size()
        self.label.setPixmap(
            self._pixmap.scaled(
                sz, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
