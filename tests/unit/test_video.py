import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import rasterio

from fraktal import labels3d, video

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


def _write_png(path: Path, rgb: np.ndarray) -> None:
    h, w = rgb.shape[:2]
    profile = {"driver": "PNG", "count": 3, "dtype": "uint8", "height": h, "width": w}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(rgb, -1, 0))


def test_colourise_maps_each_class_to_its_colour() -> None:
    semantic = np.array([[0, 2], [3, 5]], dtype=np.uint8)
    out = video.colourise(labels3d.encode(semantic, np.zeros((2, 2), dtype=int)))
    assert out.shape == (2, 2, 3)
    assert tuple(out[0, 1]) == video.CLASS_COLOURS[2]
    assert tuple(out[1, 1]) == video.CLASS_COLOURS[5]


def test_no_frames_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(video.VideoError, match="no frame"):
        video.encode_frames(tmp_path, tmp_path / "x.mp4", 12.0)


@needs_ffmpeg
def test_frames_and_labels_become_playable_videos_with_the_right_length(tmp_path: Path) -> None:
    (tmp_path / "frames").mkdir()
    (tmp_path / "labels").mkdir()
    for i in range(12):  # an odd width: the encoder pads to an even size
        grey = np.full((32, 33, 3), i * 10, "uint8")
        _write_png(tmp_path / "frames" / f"frame_{i:04d}.png", grey)
        sem = np.full((32, 33), 2, dtype=np.uint8)
        _write_png(tmp_path / "labels" / f"frame_{i:04d}.png", labels3d.encode(sem, sem * 0))
    a = video.encode_frames(tmp_path / "frames", tmp_path / "a.mp4", 6.0)
    b = video.encode_labels(tmp_path / "labels", tmp_path / "b.mp4", 6.0)
    for mp4 in (a, b):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=nb_frames,r_frame_rate,codec_name",
             "-of", "default=nw=1", str(mp4)],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        for want in ("codec_name=h264", "nb_frames=12", "r_frame_rate=6/1"):
            assert want in probe
