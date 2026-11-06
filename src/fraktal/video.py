"""Video from rendered frames (ADR-0011 phase 1, FRK-78).

``encode_frames`` turns ``frames/frame_NNNN.png`` into an H.264 MP4 and ``encode_labels`` does the
same for ``labels/frame_NNNN.png`` after colouring each class. The label video is a lossy preview:
the PNG label images remain the exact ground truth. Needs ``ffmpeg`` on the PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio

from fraktal import labels3d

CLASS_COLOURS = {  # by class id; chosen to be easy to tell apart
    0: (0, 0, 0),
    1: (90, 90, 90),
    2: (230, 60, 50),
    3: (40, 180, 70),
    4: (60, 120, 240),
    5: (240, 210, 50),
    6: (250, 140, 30),
    7: (255, 255, 255),
}


class VideoError(RuntimeError):
    pass


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.moveaxis(src.read()[:3], 0, -1)


def colourise(label_rgb: np.ndarray) -> np.ndarray:
    """An (H, W, 3) label image to class colours."""
    semantic, _ = labels3d.decode(label_rgb)
    table = np.zeros((256, 3), dtype=np.uint8)
    for cls, colour in CLASS_COLOURS.items():
        table[cls] = colour
    return np.asarray(table[semantic])


def _encode(frames: Iterator[np.ndarray], out: Path, fps: float, size: tuple[int, int]) -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise VideoError("ffmpeg is not installed (apt install ffmpeg)")
    w, h = size
    cmd = [
        exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", f"{fps:g}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if code:
        raise VideoError(f"ffmpeg exited with {code}: {err.strip()[-300:]}")


def _files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("frame_*.png"))
    if not files:
        raise VideoError(f"no frame_*.png in {directory}")
    return files


def encode_frames(frames_dir: Path, out: Path, fps: float) -> Path:
    files = _files(frames_dir)
    h, w = _read(files[0]).shape[:2]
    _encode((_read(f) for f in files), out, fps, (w, h))
    return out


def encode_labels(labels_dir: Path, out: Path, fps: float) -> Path:
    files = _files(labels_dir)
    h, w = _read(files[0]).shape[:2]
    _encode((colourise(_read(f)) for f in files), out, fps, (w, h))
    return out


__all__ = ["CLASS_COLOURS", "VideoError", "colourise", "encode_frames", "encode_labels"]
