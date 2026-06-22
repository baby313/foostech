"""主窗口：组装播放器、进度条、帧网格。

并发与内存模型（关键设计）
─────────────────────────
• 唯一解码线程 DecoderWorker：独占持有 VideoEngine，PyAV container 永远只在该线程使用，
  彻底消除 GUI 线程与解码线程并发 seek/decode 导致的 SIGABRT。
• 请求槽 latest-wins：GUI 不论以多快频率拖拽，都只往一个槽位写"最新请求"，worker
  消耗完上一个就拿"现在最新的"，中间被覆盖的全部丢弃 -> 永远不会有 backlog。
• 缩略图硬上限：30 帧网格按 thumb_max_h 缩放 + 主画面按 MAIN_CANVAS_MAX_H 缩放，
  内存占用与视频原始分辨率解耦，4K 与 720p 占用几乎一致。
• 弹窗大图按需取：cell 被点击时再 reseek 解原帧，关闭即释放。
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import (
    QMutex,
    QObject,
    QThread,
    QTimer,
    QWaitCondition,
    Qt,
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
from .utils import format_duration, format_relative_ms

THUMB_MAX_H = 180          # 30 帧网格缩略图最大高
MAIN_CANVAS_MAX_H = 720    # 主画面解码最大高（拖拽时主帧硬上限）


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


# ============================================================
# DecoderWorker：唯一解码线程
# ============================================================
class DecoderWorker(QThread):
    """独占解码线程。

    支持两种请求：
      1) preview(idx)   —— 拖拽中需要的低分辨率主帧 + 30 帧缩略图
      2) full_frame(idx)—— 点击格子需要的原始大图（区域裁剪在 GUI 侧做）

    任意时刻最多 1 个待处理请求；后到的覆盖前者。
    """

    main_ready = pyqtSignal(int, object)            # (idx, ndarray|None)
    grid_ready = pyqtSignal(int, object, object)    # (idx, region|None, list[(rel,rel_ms,arr)])
    full_ready = pyqtSignal(int, object)            # (abs_idx, ndarray|None)
    error = pyqtSignal(str)

    def __init__(self, engine: VideoEngine, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.engine = engine
        self._mutex = QMutex()
        self._cv = QWaitCondition()
        self._stop = False

        # latest-wins slots
        self._pending_main: Optional[int] = None
        self._pending_grid: Optional[tuple[int, Optional[tuple[float, float, float, float]]]] = None
        self._pending_full: Optional[int] = None

    # ----- 公共 API（GUI 线程调用）-----
    def request_preview(self, idx: int, region):
        self._mutex.lock()
        self._pending_main = idx
        self._pending_grid = (idx, region)
        self._cv.wakeAll()
        self._mutex.unlock()

    def request_main_only(self, idx: int):
        self._mutex.lock()
        self._pending_main = idx
        self._cv.wakeAll()
        self._mutex.unlock()

    def request_grid_only(self, idx: int, region):
        self._mutex.lock()
        self._pending_grid = (idx, region)
        self._cv.wakeAll()
        self._mutex.unlock()

    def request_full(self, abs_idx: int):
        self._mutex.lock()
        self._pending_full = abs_idx
        self._cv.wakeAll()
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._stop = True
        self._cv.wakeAll()
        self._mutex.unlock()

    # ----- 线程主循环 -----
    def run(self):
        while True:
            self._mutex.lock()
            while (
                not self._stop
                and self._pending_main is None
                and self._pending_grid is None
                and self._pending_full is None
            ):
                self._cv.wait(self._mutex)
            if self._stop:
                self._mutex.unlock()
                return
            # 取最新请求并立刻清空槽，让 GUI 可以继续覆盖下一次
            main_idx = self._pending_main
            grid_req = self._pending_grid
            full_idx = self._pending_full
            self._pending_main = None
            self._pending_grid = None
            self._pending_full = None
            self._mutex.unlock()

            # 优先级：full（用户点击放大，重要）> main（响应拖拽）> grid（重活，最后做）
            if full_idx is not None:
                self._do_full(full_idx)
            if main_idx is not None and self._not_superseded_main():
                self._do_main(main_idx)
            if grid_req is not None and self._not_superseded_grid():
                self._do_grid(*grid_req)

    def _not_superseded_main(self) -> bool:
        self._mutex.lock()
        ok = self._pending_main is None
        self._mutex.unlock()
        return ok

    def _not_superseded_grid(self) -> bool:
        self._mutex.lock()
        ok = self._pending_grid is None
        self._mutex.unlock()
        return ok

    def _do_main(self, idx: int):
        try:
            arr = self.engine.get_frame_at_index(idx, max_h=MAIN_CANVAS_MAX_H)
            # 保险拷贝，断开和 PyAV frame buffer 的引用
            if arr is not None:
                arr = np.ascontiguousarray(arr).copy()
            self.main_ready.emit(idx, arr)
        except Exception as e:
            self.error.emit(f"main decode err: {e}")

    def _do_grid(self, idx: int, region):
        try:
            items = self.engine.get_frames_around(
                idx, before=15, after=14, thumb_max_h=THUMB_MAX_H, region=region
            )
            # 同样断开引用 + 去重唯一对象
            seen = {}
            new_items = []
            for rel, rel_ms, arr in items:
                if arr is None:
                    new_items.append((rel, rel_ms, None))
                    continue
                key = id(arr)
                if key not in seen:
                    seen[key] = np.ascontiguousarray(arr).copy()
                new_items.append((rel, rel_ms, seen[key]))
            self.grid_ready.emit(idx, region, new_items)
        except Exception as e:
            self.error.emit(f"grid decode err: {e}")

    def _do_full(self, abs_idx: int):
        try:
            arr = self.engine.get_frame_full(abs_idx)
            if arr is not None:
                arr = np.ascontiguousarray(arr).copy()
            self.full_ready.emit(abs_idx, arr)
        except Exception as e:
            self.error.emit(f"full decode err: {e}")


# ============================================================
# MainWindow
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频帧分析播放器")
        self.resize(1440, 900)
        self.setStyleSheet(DARK_QSS)

        self.engine: Optional[VideoEngine] = None
        self.worker: Optional[DecoderWorker] = None

        self.current_frame_idx: int = 0
        self.cached_grid: list[tuple[int, float, Optional[np.ndarray]]] = []
        self.region: Optional[tuple[float, float, float, float]] = None

        self._build_ui()
        self._build_menu()

        # 拖拽节流：每 N ms 才向 worker 发一次 main 请求；grid 请求只在停止后再发
        self._main_throttle = QTimer(self)
        self._main_throttle.setSingleShot(True)
        self._main_throttle.setInterval(40)  # 25fps 上限
        self._main_throttle.timeout.connect(self._flush_main_request)
        self._pending_main_idx: Optional[int] = None

        # 网格防抖：拖拽停止 200ms 才解码 30 帧
        self._grid_debounce = QTimer(self)
        self._grid_debounce.setSingleShot(True)
        self._grid_debounce.setInterval(200)
        self._grid_debounce.timeout.connect(self._flush_grid_request)

        self.viewer = FrameViewer(self)

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self.meta_label = QLabel("尚未打开视频")
        self.meta_label.setStyleSheet(
            "background:#11141b; padding:8px 12px; font-family:Menlo; color:#cbd2e0;"
            " border:1px solid #1f2533;"
        )
        root.addWidget(self.meta_label)

        # ----- 中部：左主画面 + 右帧网格 -----
        body = QHBoxLayout()
        body.setSpacing(10)

        # 左：主画面 + 主画面下方的时间/选区状态行
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        self.canvas = VideoCanvas()
        self.canvas.region_changed.connect(self._on_region_changed)
        self.canvas.region_cleared.connect(self._on_region_cleared)
        left_col.addWidget(self.canvas, 1)

        time_row = QHBoxLayout()
        self.time_label = QLabel("00:00.000 / 00:00.000  |  Frame 0/0")
        self.time_label.setStyleSheet("font-family: Menlo; color:#9aa3b8;")
        time_row.addWidget(self.time_label)
        time_row.addStretch(1)
        self.region_label = QLabel("无选区  |  右键画面清除选区")
        self.region_label.setStyleSheet("font-family: Menlo; color:#5b6478;")
        time_row.addWidget(self.region_label)
        left_col.addLayout(time_row)
        body.addLayout(left_col, 5)

        # 右：30 帧预览
        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        grid_header = QHBoxLayout()
        self.grid_title = QLabel("前后 30 帧预览  ·  以当前帧为 0ms")
        self.grid_title.setStyleSheet("font-weight:600; color:#e4e8f1; padding:4px 0;")
        grid_header.addWidget(self.grid_title)
        grid_header.addStretch(1)
        hint = QLabel("点击格子放大")
        hint.setStyleSheet("color:#5b6478; font-size:11px;")
        grid_header.addWidget(hint)
        right_col.addLayout(grid_header)

        self.grid = FrameGrid()
        self.grid.cell_clicked.connect(self._on_cell_clicked)
        right_col.addWidget(self.grid, 1)
        body.addLayout(right_col, 4)

        root.addLayout(body, 1)

        # ----- 底部：进度条贯穿左右 -----
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        root.addWidget(self.slider)

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
        self.refresh_btn.clicked.connect(lambda: self._grid_debounce.start(0))
        toolbar.addWidget(self.refresh_btn)

        for delta, label in ((-30, "−30 帧"), (-10, "−10 帧"), (10, "+10 帧"), (30, "+30 帧")):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, d=delta: self._jump_frames(d))
            toolbar.addWidget(btn)

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
        # 销毁旧 worker + engine
        self._teardown_worker()

        try:
            engine = VideoEngine(path)
        except Exception as e:
            self.statusBar().showMessage(f"打开失败：{e}")
            return
        self.engine = engine

        # 启动单解码线程
        self.worker = DecoderWorker(engine, parent=self)
        self.worker.main_ready.connect(self._on_main_ready, Qt.ConnectionType.QueuedConnection)
        self.worker.grid_ready.connect(self._on_grid_ready, Qt.ConnectionType.QueuedConnection)
        self.worker.full_ready.connect(self._on_full_ready, Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(lambda msg: self.statusBar().showMessage(msg))
        self.worker.start()

        meta = engine.metadata
        self.meta_label.setTextFormat(Qt.TextFormat.RichText)
        self.meta_label.setText(
            f"  {Path(path).name}    |    分辨率 "
            f"<span style='color:#22d3ee'>{meta.width}×{meta.height}</span>    |    "
            f"FPS <span style='color:#22d3ee'>{meta.fps_text}</span>"
            f"    |    帧间隔 <span style='color:#22d3ee'>{meta.frame_interval_ms:.2f}ms</span>"
            f"    |    时长 {format_duration(meta.duration)}"
            f"    |    编码 {meta.codec}    |    像素格式 {meta.pix_fmt}"
        )

        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, meta.total_frames - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.current_frame_idx = 0
        self.region = None
        self.canvas.clear_region()
        self.cached_grid = []
        self.grid.clear()

        # 首帧 + 网格
        self._pending_main_idx = 0
        self._flush_main_request()
        self._grid_debounce.start(0)

        self._update_time_label(0)
        self.statusBar().showMessage(f"已加载：{Path(path).name}")

    def _teardown_worker(self):
        if self.worker is not None:
            try:
                self.worker.stop()
                self.worker.wait(2000)
            except Exception:
                pass
            self.worker = None
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    # ---------------- 进度条 ----------------
    def _on_slider_moved(self, value: int):
        # 鼠标拖动
        self._on_slider_value_changed(value)

    def _on_slider_value_changed(self, value: int):
        if not self.engine:
            return
        self.current_frame_idx = value
        self._update_time_label(value)
        # main: 节流
        self._pending_main_idx = value
        if not self._main_throttle.isActive():
            self._flush_main_request()
        # grid: 防抖
        self._grid_debounce.start()

    def _on_slider_released(self):
        if not self.engine:
            return
        # 立即精修一次主帧 + 立即出 grid
        self._pending_main_idx = self.current_frame_idx
        self._flush_main_request()
        self._grid_debounce.start(0)

    def _jump_frames(self, delta: int):
        if not self.engine:
            return
        meta = self.engine.metadata
        target = max(0, min(meta.total_frames - 1, self.current_frame_idx + delta))
        if target == self.current_frame_idx:
            return
        # 走 slider.setValue 触发已有的节流/防抖管线，逻辑统一
        self.slider.setValue(target)
        # 视为"释放"：立刻出主帧 + 立刻刷网格
        self._pending_main_idx = target
        self._flush_main_request()
        self._grid_debounce.start(0)

    def _flush_main_request(self):
        if self.worker is None or self._pending_main_idx is None:
            return
        idx = self._pending_main_idx
        self._pending_main_idx = None
        self.worker.request_main_only(idx)
        self._main_throttle.start()

    def _flush_grid_request(self):
        if self.worker is None:
            return
        # 切换前清旧缩略图，避免新旧两份共存
        self.cached_grid = []
        self.grid.clear()
        gc.collect()
        self.grid_title.setText(
            "区域特写  ·  正在解码…" if self.region else "前后 30 帧预览  ·  正在解码…"
        )
        self.worker.request_grid_only(self.current_frame_idx, self.region)

    def _update_time_label(self, idx: int):
        if not self.engine:
            return
        meta = self.engine.metadata
        cur_t = idx / meta.fps if meta.fps else 0
        self.time_label.setText(
            f"{format_duration(cur_t)} / {format_duration(meta.duration)}  |  "
            f"Frame {idx}/{meta.total_frames}"
        )

    # ---------------- worker 回调（GUI 线程）----------------
    def _on_main_ready(self, idx: int, arr):
        if idx != self.current_frame_idx:
            return  # 已被新请求覆盖，丢弃
        self.canvas.set_frame(arr)

    def _on_grid_ready(self, idx: int, region, items):
        # 过期：用户已经移到别处或选区已变
        if idx != self.current_frame_idx or region != self.region:
            return
        self.cached_grid = items
        self.grid.update_frames(items)
        self.grid_title.setText(
            "区域特写  ·  以当前帧为 0ms" if region else "前后 30 帧预览  ·  以当前帧为 0ms"
        )

    def _on_full_ready(self, abs_idx: int, arr):
        meta = self.engine.metadata if self.engine else None
        if arr is None or meta is None:
            return
        if self.region:
            from .utils import crop_region

            arr_show = crop_region(arr, self.region)
            tag = "区域特写"
        else:
            arr_show = arr
            tag = "整帧"
        rel = abs_idx - self.current_frame_idx
        rel_ms = rel * meta.frame_interval_ms
        abs_t = abs_idx / meta.fps
        info = (
            f"{tag}  |  相对 {format_relative_ms(rel_ms)}  |  绝对帧 {abs_idx}  |  "
            f"绝对时间 {format_duration(abs_t)}  |  FPS {meta.fps_text}"
        )
        self.viewer.show_frame(arr_show, info)
        self.viewer.exec()
        self.viewer.show_frame(None, "")
        gc.collect()

    # ---------------- 选区 ----------------
    def _on_region_changed(self, region):
        self.region = region
        rx, ry, rw, rh = region
        self.region_label.setText(
            f"选区 x={rx:.2f} y={ry:.2f} w={rw:.2f} h={rh:.2f}  |  右键画面清除"
        )
        self.region_label.setStyleSheet("font-family: Menlo; color:#22d3ee;")
        self._grid_debounce.start(0)

    def _on_region_cleared(self):
        if self.region is None:
            return
        self.region = None
        self.region_label.setText("无选区  |  右键画面清除选区")
        self.region_label.setStyleSheet("font-family: Menlo; color:#5b6478;")
        self._grid_debounce.start(0)

    # ---------------- 单帧放大 ----------------
    def _on_cell_clicked(self, rel_index: int):
        if not self.engine or self.worker is None:
            return
        abs_idx = self.current_frame_idx + rel_index
        self.worker.request_full(abs_idx)

    # ---------------- 关闭 ----------------
    def closeEvent(self, event):
        self._teardown_worker()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("视频帧分析播放器")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
