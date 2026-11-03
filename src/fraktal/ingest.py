"""Fetch and cache the SpaceNet 2 Khartoum train data (ADR-0008).

The archive downloads to ``<data>/downloads/*.part`` and resumes after an interruption. It is
renamed only once its size matches the server's, then extracted into a temporary directory that
is renamed into place, and a ``.complete`` marker is written last. A later run that finds the
marker returns ``cached`` without touching the network.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fraktal.log import emit

URL = (
    "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings/tarballs/"
    "SN2_buildings_train_AOI_5_Khartoum.tar.gz"
)
DATASET = "spacenet2-khartoum"
CHUNK = 1 << 20
MARKER = ".complete"
PROGRESS_BYTES = 256 << 20


class Transport(Protocol):
    def size(self, url: str) -> int: ...

    def read(self, url: str, start: int) -> Iterator[bytes]:
        """Yield the body from byte ``start`` onwards."""
        ...


class HttpTransport:
    def size(self, url: str) -> int:
        request = urllib.request.Request(url, method="HEAD")  # noqa: S310 - fixed https URL
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return int(response.headers["Content-Length"])

    def read(self, url: str, start: int) -> Iterator[bytes]:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"})  # noqa: S310
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            if start and response.status != 206:
                raise OSError(f"server ignored the Range request (status {response.status})")
            while chunk := response.read(CHUNK):
                yield chunk


@dataclass(frozen=True)
class IngestResult:
    status: str  # "cached" or "fetched"
    path: Path
    bytes: int
    resumed_from: int = 0


def ingest(
    data_dir: Path,
    *,
    url: str = URL,
    transport: Transport | None = None,
    keep_archive: bool = False,
) -> IngestResult:
    transport = transport or HttpTransport()
    target = data_dir / "spacenet2" / "khartoum"
    marker = target / MARKER
    if marker.is_file():
        info = json.loads(marker.read_text())
        return IngestResult("cached", target, int(info["archive_bytes"]))

    expected = transport.size(url)
    downloads = data_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / url.rsplit("/", 1)[1]
    part = archive.with_name(archive.name + ".part")

    resumed_from = 0
    if archive.is_file() and archive.stat().st_size != expected:
        archive.unlink()  # a truncated or foreign file: start over
    if not archive.is_file():
        if part.is_file() and part.stat().st_size > expected:
            part.unlink()
        resumed_from = part.stat().st_size if part.is_file() else 0
        if resumed_from < expected:
            emit("download_start", bytes=expected, resumed_from=resumed_from)
            with part.open("ab") as out:
                last = resumed_from
                for chunk in transport.read(url, resumed_from):
                    out.write(chunk)
                    got = out.tell()
                    if got - last >= PROGRESS_BYTES:
                        emit("download_progress", bytes=got, of=expected)
                        last = got
        if part.stat().st_size != expected:
            raise OSError(
                f"download incomplete: got {part.stat().st_size} of {expected} bytes; "
                "run ingest again to resume"
            )
        part.rename(archive)

    staging = data_dir / "spacenet2" / "khartoum.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    emit("extract_start", archive=archive.name)
    with tarfile.open(archive) as tar:
        tar.extractall(staging, filter="data")
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    marker.write_text(json.dumps({"dataset": DATASET, "url": url, "archive_bytes": expected}))
    if not keep_archive:
        archive.unlink()
    return IngestResult("fetched", target, expected, resumed_from)
