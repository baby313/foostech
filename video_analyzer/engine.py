"""视频解码引擎：基于 PyAV (FFmpeg) 解析元信息和按时间精确取帧。

为减少内存占用，提供以下能力：
- 主帧解码可指定 max_height 让 swscale 直接降采样
- 30 帧批量解码可指定 thumb_max_height 与 region，输出已经是缩略图/特写
- 原始大图仅在用户显式放大时通过 get_frame_full 再次解出
"""
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
    duration: float
    fps: float
    frame_interval_ms: float
    total_frames: int
    codec: str
    pix_fmt: str
    time_base: Fraction
    bitrate: Optional[int] = None

    @property
    def fps_text(self) -> str:
        return f"{self.fps:.3f}".rstrip("0").rstrip(".")


def _scaled_size(src_w: int, src_h: int, max_h: Optional[int]) -> tuple[int, int]:
    if not max_h or src_h <= max_h:
        return src_w, src_h
    scale = max_h / src_h
    new_w = max(2, int(round(src_w * scale)) // 2 * 2)
    return new_w, max_h


class VideoEngine:
    def __init__(self, path: str):
        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.metadata = self._probe_metadata()

    def _probe_metadata(self) -> VideoMetadata:
        s = self.stream
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
        ctx = s.codec_context
        return VideoMetadata(
            width=ctx.width,
            height=ctx.height,
            duration=duration_sec,
            fps=fps,
            frame_interval_ms=1000.0 / fps,
            total_frames=total_frames,
            codec=ctx.name,
            pix_fmt=ctx.pix_fmt or "unknown",
            time_base=time_base,
            bitrate=self.container.bit_rate,
        )

    def close(self):
        try:
            self.container.close()
        except Exception:
            pass

    # ---------- 内部：seek + 解码到目标 ----------
    def _seek_to(self, t_sec: float):
        meta = self.metadata
        target_pts = int(round(t_sec / float(meta.time_base)))
        self.container.seek(target_pts, stream=self.stream, any_frame=False, backward=True)

    def _frame_to_rgb(self, frame, max_h: Optional[int]) -> np.ndarray:
        """用 swscale 把帧转 RGB，并可选下采样。"""
        new_w, new_h = _scaled_size(frame.width, frame.height, max_h)
        if new_w == frame.width and new_h == frame.height:
            return frame.to_ndarray(format="rgb24")
        # reformat 到目标尺寸 + RGB
        reformatted = frame.reformat(width=new_w, height=new_h, format="rgb24")
        return reformatted.to_ndarray()

    # ---------- 取帧 API ----------
    def get_frame_at_time(self, t_sec: float, max_h: Optional[int] = None) -> Optional[np.ndarray]:
        meta = self.metadata
        t_sec = max(0.0, min(t_sec, max(meta.duration - 1e-3, 0.0)))
        self._seek_to(t_sec)

        chosen = None
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                continue
            ftime = float(frame.pts * meta.time_base)
            chosen = frame
            if ftime + 1e-6 >= t_sec:
                break
        if chosen is None:
            return None
        return self._frame_to_rgb(chosen, max_h)

    def get_frame_at_index(self, idx: int, max_h: Optional[int] = None) -> Optional[np.ndarray]:
        return self.get_frame_at_time(idx / self.metadata.fps, max_h=max_h)

    def get_frame_full(self, idx: int) -> Optional[np.ndarray]:
        return self.get_frame_at_index(idx, max_h=None)

    def get_frames_around(
        self,
        base_idx: int,
        before: int = 15,
        after: int = 14,
        thumb_max_h: Optional[int] = 200,
        region: Optional[tuple[float, float, float, float]] = None,
    ) -> list[tuple[int, float, Optional[np.ndarray]]]:
        """批量取前后帧。

        - thumb_max_h: 缩略图最大高度（None 表示不缩放）
        - region: 若给出，先按相对坐标在原始帧空间裁剪，再缩到 thumb_max_h
        """
        meta = self.metadata
        results: list[tuple[int, float, Optional[np.ndarray]]] = []

        start_idx = base_idx - before
        t_start = max(0.0, start_idx / meta.fps)
        self._seek_to(t_start)

        wanted = []
        for rel in range(-before, after + 1):
            abs_idx = base_idx + rel
            t = abs_idx / meta.fps
            wanted.append((rel, abs_idx, t, rel * meta.frame_interval_ms))

        wi = 0
        cached_arr: Optional[np.ndarray] = None
        last_pts: Optional[int] = None
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                continue
            ftime = float(frame.pts * meta.time_base)
            # 当前帧解码出的目标 ndarray 在被多个 wanted 命中时复用，避免重复 reformat
            cached_arr = None
            while wi < len(wanted):
                rel, abs_idx, t, rel_ms = wanted[wi]
                if abs_idx < 0 or t > meta.duration:
                    results.append((rel, rel_ms, None))
                    wi += 1
                    continue
                if ftime + 1e-6 >= t:
                    if cached_arr is None:
                        cached_arr = self._extract_thumb(frame, thumb_max_h, region)
                    results.append((rel, rel_ms, cached_arr))
                    wi += 1
                else:
                    break
            if wi >= len(wanted):
                break

        while wi < len(wanted):
            rel, abs_idx, t, rel_ms = wanted[wi]
            results.append((rel, rel_ms, None))
            wi += 1

        return results

    def _extract_thumb(
        self,
        frame,
        thumb_max_h: Optional[int],
        region: Optional[tuple[float, float, float, float]],
    ) -> np.ndarray:
        if region is None:
            return self._frame_to_rgb(frame, thumb_max_h)
        # 区域：先到原 RGB（不可避免），切片，再 resize 到缩略
        full = frame.to_ndarray(format="rgb24")
        h, w, _ = full.shape
        rx, ry, rw, rh = region
        x = max(0, min(int(rx * w), w - 1))
        y = max(0, min(int(ry * h), h - 1))
        cw = max(1, min(int(rw * w), w - x))
        ch = max(1, min(int(rh * h), h - y))
        sub = full[y : y + ch, x : x + cw, :].copy()
        del full
        if thumb_max_h and sub.shape[0] > thumb_max_h:
            from PIL import Image

            img = Image.fromarray(sub)
            new_h = thumb_max_h
            new_w = max(2, int(sub.shape[1] * new_h / sub.shape[0]))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            sub = np.asarray(img)
        return sub
