"""压力测试：模拟"狂拖进度条"场景，验证不崩溃 + RSS 不爆。

用法：
    python -m video_analyzer.stress /path/to/video.mp4 [iterations]

模拟流程：
1. 创建 MainWindow 并加载视频（不显示窗口，QTest 模式）。
2. 在 GUI 主循环中以 5~30ms 间隔随机拖动 slider，连续 N 次。
3. 期间监控 RSS 峰值，结束前打印结果。
4. 故意触发：选区切换、点击格子放大（取大图）、再继续拖。
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication

from video_analyzer.main import MainWindow


def cur_rss_mb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()
    return float(out) / 1024


def run(video_path: str, iterations: int = 200):
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win._load_video(video_path)
    win.show()
    app.processEvents()
    if win.engine is None:
        print("加载视频失败")
        return

    total_frames = win.engine.metadata.total_frames
    print(f"video frames={total_frames}, iterations={iterations}")

    start_rss = cur_rss_mb()
    peak_rss = start_rss
    rng = random.Random(42)
    counter = {"i": 0}

    timer = QTimer()
    rss_log = []

    def step():
        nonlocal peak_rss
        i = counter["i"]
        if i >= iterations:
            timer.stop()
            print(f"\n[done] iterations={iterations}")
            print(f"[done] start RSS={start_rss:.1f} MB  peak RSS={peak_rss:.1f} MB  Δ={peak_rss - start_rss:+.1f} MB")
            # 等 worker 把队列消干净再退出
            QTimer.singleShot(800, lambda: app.quit())
            return
        # 随机拖到任意帧
        target = rng.randint(0, total_frames - 1)
        win.slider.setValue(target)
        # 偶尔切区域
        if i % 25 == 5:
            win.canvas._region = (0.1 + rng.random() * 0.4, 0.1, 0.3, 0.3)
            win._on_region_changed(win.canvas._region)
        if i % 25 == 18:
            win._on_region_cleared()
        # 偶尔点格子（触发 full frame）
        if i % 50 == 30:
            win._on_cell_clicked(rng.choice([-10, -5, 0, 5, 10]))
            # 立即把 viewer 关掉（exec 会阻塞，所以单独触发）
            QTimer.singleShot(50, win.viewer.accept)
        cur = cur_rss_mb()
        peak_rss = max(peak_rss, cur)
        rss_log.append(cur)
        if i % 25 == 0:
            print(f"  iter {i:3d}  target={target:5d}  RSS={cur:.1f} MB")
        counter["i"] += 1

    interval_ms = 25  # 比真人狂拖还快
    timer.timeout.connect(step)
    timer.start(interval_ms)

    app.exec()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m video_analyzer.stress <video> [iterations]")
        sys.exit(1)
    iters = int(sys.argv[2]) if len(sys.argv) >= 3 else 200
    run(sys.argv[1], iters)
