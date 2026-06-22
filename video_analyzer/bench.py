"""验证内存假设：测量 30 帧批量提取的物理内存占用。

用法（在 video_analyzer 目录的父目录运行）：
    .venv/bin/python -m video_analyzer.bench [视频路径]

输出：
- 视频元信息
- 单帧 ndarray 大小
- 30 帧累计字节、RSS 增量
- 主帧 + 30 帧 + QPixmap 模拟之后的 RSS 峰值
"""
from __future__ import annotations

import gc
import os
import resource
import sys
from pathlib import Path

import numpy as np

from video_analyzer.engine import VideoEngine
from video_analyzer.utils import crop_region, ndarray_to_qpixmap


def rss_mb() -> float:
    # macOS: ru_maxrss 单位是字节
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def cur_rss_mb() -> float:
    # 当前 RSS（粗略），通过 /proc 不可用 -> 用 ps
    try:
        import subprocess

        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()
        return float(out) / 1024
    except Exception:
        return -1


def fmt_bytes(b: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} TB"


def main():
    if len(sys.argv) < 2:
        print("用法: python -m video_analyzer.bench <video_path>")
        sys.exit(1)
    path = sys.argv[1]
    if not Path(path).exists():
        print(f"文件不存在: {path}")
        sys.exit(1)

    print(f"[step 0] 启动 RSS = {cur_rss_mb():.1f} MB")

    engine = VideoEngine(path)
    m = engine.metadata
    per_frame_bytes = m.width * m.height * 3
    print(f"[meta] {m.width}x{m.height}  fps={m.fps:.3f}  total_frames={m.total_frames}")
    print(f"[meta] codec={m.codec}  pix_fmt={m.pix_fmt}  duration={m.duration:.2f}s")
    print(f"[meta] 单帧 RGB ndarray 理论大小 = {fmt_bytes(per_frame_bytes)}")
    print(f"[meta] 30 帧理论累计 = {fmt_bytes(per_frame_bytes * 30)}")

    base = m.total_frames // 2
    print(f"\n[step 1] 仅取主帧（max_h=1080）idx={base}")
    rss0 = cur_rss_mb()
    main_frame = engine.get_frame_at_index(base, max_h=1080)
    rss1 = cur_rss_mb()
    print(f"  主帧 shape={main_frame.shape if main_frame is not None else None}")
    print(f"  RSS: {rss0:.1f} -> {rss1:.1f} MB (Δ {rss1 - rss0:+.1f})")

    print(f"\n[step 2] 批量取前后 30 帧 缩略图（thumb_max_h=200）")
    rss2 = cur_rss_mb()
    items = engine.get_frames_around(base, before=15, after=14, thumb_max_h=200)
    rss3 = cur_rss_mb()
    decoded = sum(1 for _, _, a in items if a is not None)
    total_bytes = sum(a.nbytes for _, _, a in items if a is not None)
    print(f"  返回帧数 = {len(items)}, 实际有图 = {decoded}")
    print(f"  实际累计字节 = {fmt_bytes(total_bytes)}")
    print(f"  RSS: {rss2:.1f} -> {rss3:.1f} MB (Δ {rss3 - rss2:+.1f})")

    print(f"\n[step 3] 模拟 GUI 把 30 帧都渲染成 QPixmap")
    try:
        from PyQt6.QtWidgets import QApplication

        if QApplication.instance() is None:
            app = QApplication(sys.argv)
        rss4 = cur_rss_mb()
        pixmaps = [ndarray_to_qpixmap(a) if a is not None else None for _, _, a in items]
        rss5 = cur_rss_mb()
        ok = sum(1 for p in pixmaps if p is not None and not p.isNull())
        print(f"  生成 QPixmap: {ok} 个")
        print(f"  RSS: {rss4:.1f} -> {rss5:.1f} MB (Δ {rss5 - rss4:+.1f})")
    except Exception as e:
        print(f"  QPixmap 阶段失败: {e}")

    print(f"\n[step 4] 模拟连续 5 次拖拽（每次重取 30 帧），观察是否累积")
    rss_before = cur_rss_mb()
    for i in range(5):
        idx = max(15, min(base + i * 5, m.total_frames - 16))
        items_i = engine.get_frames_around(idx, before=15, after=14, thumb_max_h=200)
        cur = cur_rss_mb()
        print(f"  iter {i+1} idx={idx}  decoded={sum(1 for _,_,a in items_i if a is not None)}  RSS={cur:.1f} MB")
        del items_i
        gc.collect()
    rss_after = cur_rss_mb()
    print(f"  5 次拖拽后 RSS Δ = {rss_after - rss_before:+.1f} MB")

    print(f"\n[final] peak ru_maxrss ≈ {rss_mb():.1f} MB")
    engine.close()


if __name__ == "__main__":
    main()
