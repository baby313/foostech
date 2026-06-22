#!/usr/bin/env bash
# 启动视频帧分析播放器（macOS）
# 用法：./run.sh   或   bash run.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "[setup] 创建虚拟环境 .venv ..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install PyQt6 av Pillow numpy
fi

# 必须从父目录运行，python 才能 import video_analyzer 包
cd "$PARENT_DIR"
exec "$SCRIPT_DIR/.venv/bin/python" -m video_analyzer.main "$@"
