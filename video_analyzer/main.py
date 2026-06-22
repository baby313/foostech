"""主窗口：组装播放器、进度条、帧网格。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .canvas import VideoCanvas
from .engine import VideoEngine
from .frames import FrameGrid, FrameViewer
from .utils import crop_region, format_duration, format_relative_ms

DARK_QSS = """
QMainWindow, QWidget { background-color: #0b0d12; color: #e4e8f1; }
QLabel { color: #cbd2e0; }
QToolBar { background: #0f1218; border: 0; padding: 6px; spacing: 8px; }
QPushButton { background:#1f2533; color:#e4e8f1; padding:6px 14px; border:1px solid #2c3346; border-radius:0px; }
QPushButton:hover { background:#262d3f; }
QPushButton:disabled { color:#5b6478; background:#161a23; }
QSlider::groove:horizontal { height: 6px; background:#1f2533; border-radius:0px; }
QSlider::handle:horizontal { width:14px; margin:-5px 0; background:#22d3ee; border-radius:0px; }
QSlider::sub-page:horizontal { background:#3b82f6; }
QStatusBar { background:#0f1218; color:#9aa3b8; }
"""


class FrameJobSignals(QObject):
    finished = pyqtSignal(int, list)  # job_id, items


class FrameJob(QRunnable):
    def __init__(self, engine: VideoEngine, base_idx: int, job_id: int):
        super().__init__()
        self.engine = engine
        self.base_idx = base_idx
        self.job_id = job_id
        self.signals = FrameJobSignals()

    def run(self):
        try:
            items = self.engine.get_frames_around(self.base_idx, before=15, after=14)
        except Exception as e:
            print(f"[FrameJob] error: {e}", file=sys.stderr)
            items = []
        self.signals.finished.emit(self.job_id, items)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频帧分析播放器")
        self.resize(1440, 900)
        self.setStyleSheet(DARK_QSS)

        self.engine: Optional[VideoEngine] = None
        self.current_frame_idx: int = 0
        self.last_main_frame: Optional[np.ndarray] = None
        self.cached_grid: list[tuple[int, float, Optional[np.ndarray]]] = []
        self.region: Optional[tuple[float, float, float, float]] = None

        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(2)
        self._job_seq = 0
        self._latest_job_id = -1

        self._build_ui()
        self._build_menu()

        # 防抖：滑动停止 120ms 才刷新 30 帧
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._refresh_grid_now)

        self.viewer = FrameViewer(self)

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # 元信息栏
        self.meta_label = QLabel("尚未打开视频")
        self.meta_label.setStyleSheet(
            "background:#11141b; padding:8px 12px; font-family:Menlo; color:#cbd2e0;"
            " border:1px solid #1f2533;"
        )
        root.addWidget(self.meta_label)

        # 视频画布
        self.canvas = VideoCanvas()
        self.canvas.region_changed.connect(self._on_region_changed)
        self.canvas.region_cleared.connect(self._on_region_cleared)
        root.addWidget(self.canvas, 4)

        # 进度条 + 时间
        time_row = QHBoxLayout()
        self.time_label = QLabel("00:00.000 / 00:00.000  |  Frame 0/0")
        self.time_label.setStyleSheet("font-family: Menlo; color:#9aa3b8;")
        time_row.addWidget(self.time_label)
        time_row.addStretch(1)

        self.region_label = QLabel("无选区  |  右键画面清除选区")
        self.region_label.setStyleSheet("font-family: Menlo; color:#5b6478;")
        time_row.addWidget(self.region_label)
        root.addLayout(time_row)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)
        root.addWidget(self.slider)

        # 帧网格区
        grid_header = QHBoxLayout()
        self.grid_title = QLabel("前后 30 帧预览  ·  以当前帧为 0ms")
        self.grid_title.setStyleSheet("font-weight:600; color:#e4e8f1; padding:4px 0;")
        grid_header.addWidget(self.grid_title)
        grid_header.addStretch(1)
        hint = QLabel("点击格子放大 · 在主画面拖拽框选区域可显示特写")
        hint.setStyleSheet("color:#5b6478; font-size:11px;")
        grid_header.addWidget(hint)
        root.addLayout(grid_header)

        self.grid = FrameGrid()
        self.grid.cell_clicked.connect(self._on_cell_clicked)
        root.addWidget(self.grid, 3)

        # 状态栏
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

    def _build_menu(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_btn = QPushButton("打开视频")
        open_btn.clicked.connect(self.open_file)
        toolbar.addWidget(open_btn)

        self.clear_region_btn = QPushButton("清除选区")
        self.clear_region_btn.clicked.connect(lambda: self.canvas.clear_region())
        toolbar.addWidget(self.clear_region_btn)

        self.refresh_btn = QPushButton("刷新预览")
        self.refresh_btn.clicked.connect(self._schedule_refresh)
        toolbar.addWidget(self.refresh_btn)

        # 快捷键 ⌘O
        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file)
        self.addAction(open_action)

    # ---------------- 文件 ----------------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            str(Path.home()),
            "视频文件 (*.mp4 *.mov *.mkv *.avi *.webm *.flv *.m4v);;所有文件 (*)",
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path: str):
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        try:
            self.engine = VideoEngine(path)
        except Exception as e:
            self.statusBar().showMessage(f"打开失败：{e}")
            return

        meta = self.engine.metadata
        self.meta_label.setText(
            f"  {Path(path).name}    |    分辨率 "
            f"<span style='color:#22d3ee'>{meta.width}×{meta.height}</span>    |    "
            f"FPS <span style='color:#22d3ee'>{meta.fps_text}</span>"
            f"    |    帧间隔 <span style='color:#22d3ee'>{meta.frame_interval_ms:.2f}ms</span>"
            f"    |    时长 {format_duration(meta.duration)}"
            f"    |    编码 {meta.codec}    |    像素格式 {meta.pix_fmt}"
        )
        # QLabel 默认富文本支持
        self.meta_label.setTextFormat(Qt.TextFormat.RichText)

        max_frames = max(0, meta.total_frames - 1)
        self.slider.setRange(0, max_frames)
        self.slider.setValue(0)
        self.current_frame_idx = 0
        self.region = None
        self.canvas.clear_region()
        self._update_main_frame(0)
        self._schedule_refresh()
        self.statusBar().showMessage(f"已加载：{Path(path).name}")

    # ---------------- 进度条 ----------------
    def _on_slider_moved(self, value: int):
        if not self.engine:
            return
        self.current_frame_idx = value
        self._update_main_frame(value)
        self._schedule_refresh()

    def _on_slider_released(self):
        if not self.engine:
            return
        self._refresh_grid_now()

    def _update_main_frame(self, idx: int):
        if not self.engine:
            return
        meta = self.engine.metadata
        try:
            arr = self.engine.get_frame_at_index(idx)
        except Exception as e:
            self.statusBar().showMessage(f"取帧失败：{e}")
            return
        self.last_main_frame = arr
        self.canvas.set_frame(arr)
        cur_t = idx / meta.fps if meta.fps else 0
        self.time_label.setText(
            f"{format_duration(cur_t)} / {format_duration(meta.duration)}  |  "
            f"Frame {idx}/{meta.total_frames}"
        )

    # ---------------- 30 帧预览 ----------------
    def _schedule_refresh(self):
        if not self.engine:
            return
        self._refresh_timer.start()

    def _refresh_grid_now(self):
        if not self.engine:
            return
        self._job_seq += 1
        self._latest_job_id = self._job_seq
        job = FrameJob(self.engine, self.current_frame_idx, self._job_seq)
        job.signals.finished.connect(self._on_grid_ready)
        self.grid_title.setText("前后 30 帧预览  ·  正在解码…")
        self.thread_pool.start(job)

    def _on_grid_ready(self, job_id: int, items: list):
        if job_id != self._latest_job_id:
            return  # 丢弃过期结果
        self.cached_grid = items
        self._render_grid()
        if self.region:
            self.grid_title.setText("区域特写  ·  以当前帧为 0ms")
        else:
            self.grid_title.setText("前后 30 帧预览  ·  以当前帧为 0ms")

    def _render_grid(self):
        items = self.cached_grid
        if not items:
            self.grid.clear()
            return
        if self.region:
            new_items = []
            for rel, rel_ms, arr in items:
                if arr is None:
                    new_items.append((rel, rel_ms, None))
                else:
                    new_items.append((rel, rel_ms, crop_region(arr, self.region)))
            self.grid.update_frames(new_items)
        else:
            self.grid.update_frames(items)

    # ---------------- 选区 ----------------
    def _on_region_changed(self, region):
        self.region = region
        rx, ry, rw, rh = region
        self.region_label.setText(
            f"选区 x={rx:.2f} y={ry:.2f} w={rw:.2f} h={rh:.2f}  |  右键画面清除"
        )
        self.region_label.setStyleSheet("font-family: Menlo; color:#22d3ee;")
        self._render_grid()
        self.grid_title.setText("区域特写  ·  以当前帧为 0ms")

    def _on_region_cleared(self):
        self.region = None
        self.region_label.setText("无选区  |  右键画面清除选区")
        self.region_label.setStyleSheet("font-family: Menlo; color:#5b6478;")
        self._render_grid()
        self.grid_title.setText("前后 30 帧预览  ·  以当前帧为 0ms")

    # ---------------- 单帧放大 ----------------
    def _on_cell_clicked(self, rel_index: int):
        if not self.engine or not self.cached_grid:
            return
        target = next((it for it in self.cached_grid if it[0] == rel_index), None)
        if target is None:
            return
        rel, rel_ms, arr = target
        if arr is None:
            return
        if self.region:
            arr_show = crop_region(arr, self.region)
            tag = "区域特写"
        else:
            arr_show = arr
            tag = "整帧"
        meta = self.engine.metadata
        abs_idx = self.current_frame_idx + rel
        abs_t = abs_idx / meta.fps
        info = (
            f"{tag}  |  相对 {format_relative_ms(rel_ms)}  |  绝对帧 {abs_idx}  |  "
            f"绝对时间 {format_duration(abs_t)}  |  FPS {meta.fps_text}"
        )
        self.viewer.show_frame(arr_show, info)
        self.viewer.exec()

    # ---------------- 关闭 ----------------
    def closeEvent(self, event):
        if self.engine:
            self.engine.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("视频帧分析播放器")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
