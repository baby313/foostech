"""视频解码引擎：基于 PyAV (FFmpeg) 解析元信息和按时间精确取帧。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

import av
import numpy as np


@dataclass
class VideoMetadata:
    width: int
    height: int
    duration: float          # 秒
    fps: float               # 帧率
    frame_interval_ms: float # 1000 / fps
    total_frames: int
    codec: str
    pix_fmt: str
    time_base: Fraction
    bitrate: Optional[int] = None

    @property
    def fps_text(self) -> str:
        return f"{self.fps:.3f}".rstrip("0").rstrip(".")


class VideoEngine:
    """封装 PyAV 视频解码，提供按帧索引/时间取帧能力。"""

    def __init__(self, path: str):
        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        # 关键：设置 thread_type 提升 seek 后解码速度
        self.stream.thread_type = "AUTO"

        meta = self._probe_metadata()
        self.metadata = meta

    def _probe_metadata(self) -> VideoMetadata:
        s = self.stream
        # average_rate 通常更稳定，回退 base_rate
        rate = s.average_rate or s.base_rate or s.guessed_rate
        fps = float(rate) if rate else 30.0
        if fps <= 0 or math.isnan(fps):
            fps = 30.0

        time_base = s.time_base or Fraction(1, 1000)
        duration_sec = 0.0
        if s.duration is not None:
            duration_sec = float(s.duration * time_base)
        elif self.container.duration:
            duration_sec = self.container.duration / av.time_base

        total_frames = s.frames or int(round(duration_sec * fps))

        codec_ctx = s.codec_context
        return VideoMetadata(
            width=codec_ctx.width,
            height=codec_ctx.height,
            duration=duration_sec,
            fps=fps,
            frame_interval_ms=1000.0 / fps,
            total_frames=total_frames,
            codec=codec_ctx.name,
            pix_fmt=codec_ctx.pix_fmt or "unknown",
            time_base=time_base,
            bitrate=self.container.bit_rate,
        )

    def close(self):
        try:
            self.container.close()
        except Exception:
            pass

    # ---------- 取帧 ----------
    def get_frame_at_time(self, t_sec: float) -> Optional[np.ndarray]:
        """按秒数取最接近的关键帧之后解码到目标时间点的帧，返回 RGB ndarray。"""
        meta = self.metadata
        t_sec = max(0.0, min(t_sec, max(meta.duration - 1e-3, 0.0)))

        # 转换为 stream 时基的 PTS
        target_pts = int(round(t_sec / float(meta.time_base)))
        # seek 到目标之前的关键帧
        self.container.seek(target_pts, stream=self.stream, any_frame=False, backward=True)

        last_frame = None
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * meta.time_base)
            if frame_time + 1e-6 >= t_sec:
                last_frame = frame
                break
            last_frame = frame

        if last_frame is None:
            return None
        # 转 RGB ndarray
        return last_frame.to_ndarray(format="rgb24")

    def get_frame_at_index(self, idx: int) -> Optional[np.ndarray]:
        t = idx / self.metadata.fps
        return self.get_frame_at_time(t)

    def get_frames_around(self, base_idx: int, before: int = 15, after: int = 14) -> list[tuple[int, float, Optional[np.ndarray]]]:
        """获取以 base_idx 为 0 点，前 before、后 after 帧的批量结果。
        返回 [(rel_index, rel_ms, ndarray|None)]
        """
        meta = self.metadata
        results: list[tuple[int, float, Optional[np.ndarray]]] = []

        start_idx = base_idx - before
        end_idx = base_idx + after  # 包含

        # 计算 seek 起点时间
        t_start = max(0.0, start_idx / meta.fps)
        target_pts = int(round(t_start / float(meta.time_base)))
        self.container.seek(target_pts, stream=self.stream, any_frame=False, backward=True)

        # 期望的时间序列
        wanted = []
        for rel in range(-before, after + 1):
            abs_idx = base_idx + rel
            t = abs_idx / meta.fps
            wanted.append((rel, abs_idx, t, rel * meta.frame_interval_ms))

        wi = 0
        last_frame_arr = None
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                continue
            ftime = float(frame.pts * meta.time_base)
            arr = None
            # 把所有目标时间 <= ftime 的项都填上当前帧
            while wi < len(wanted):
                rel, abs_idx, t, rel_ms = wanted[wi]
                if abs_idx < 0 or t > meta.duration:
                    results.append((rel, rel_ms, None))
                    wi += 1
                    continue
                if ftime + 1e-6 >= t:
                    if arr is None:
                        arr = frame.to_ndarray(format="rgb24")
                    results.append((rel, rel_ms, arr))
                    wi += 1
                else:
                    break
            if wi >= len(wanted):
                break

        # 剩余未填的（视频结束）
        while wi < len(wanted):
            rel, abs_idx, t, rel_ms = wanted[wi]
            results.append((rel, rel_ms, None))
            wi += 1

        return results
